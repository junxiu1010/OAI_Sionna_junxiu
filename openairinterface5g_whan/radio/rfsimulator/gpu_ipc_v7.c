/*
 * gpu_ipc_v7.c - CUDA IPC Circular Buffer with Futex-based Notification
 *
 * Evolution of V6: per-buffer nbAnt and cir_size for asymmetric MIMO,
 * plus futex wait/wake to replace usleep polling.
 *
 * Read functions block internally (futex_wait + timeout) so callers
 * never need their own retry loop.
 *
 * CUDA loaded at runtime via dlopen — no build-time dependency.
 */

#include "gpu_ipc_v7.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>

/* ── CUDA type definitions (no CUDA headers needed) ── */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaIpcMemLazyEnablePeerAccess 1

typedef struct { char reserved[GPU_IPC_V7_HANDLE_SIZE]; } cudaIpcMemHandle_t;

typedef cudaError_t (*fn_cudaMemcpy)(void *, const void *, size_t, int);
typedef cudaError_t (*fn_cudaMemset)(void *, int, size_t);
typedef cudaError_t (*fn_cudaIpcOpenMemHandle)(void **, cudaIpcMemHandle_t, unsigned int);
typedef cudaError_t (*fn_cudaIpcCloseMemHandle)(void *);
typedef cudaError_t (*fn_cudaSetDevice)(int);
typedef const char* (*fn_cudaGetErrorString)(cudaError_t);
typedef cudaError_t (*fn_cudaDeviceSynchronize)(void);

static fn_cudaMemcpy            p_cudaMemcpy;
static fn_cudaMemset            p_cudaMemset;
static fn_cudaIpcOpenMemHandle  p_cudaIpcOpenMemHandle;
static fn_cudaIpcCloseMemHandle p_cudaIpcCloseMemHandle;
static fn_cudaSetDevice         p_cudaSetDevice;
static fn_cudaGetErrorString    p_cudaGetErrorString;
static fn_cudaDeviceSynchronize p_cudaDeviceSynchronize;

#define LOG_V7(fmt, ...) fprintf(stderr, "[GPU_IPC_V7] " fmt "\n", ##__VA_ARGS__)

/* ── SHM helpers ── */

static inline volatile uint32_t *shm_u32(char *b, int off) {
    return (volatile uint32_t *)(b + off);
}
static inline volatile uint64_t *shm_u64(char *b, int off) {
    return (volatile uint64_t *)(b + off);
}

static inline uint32_t circ_offset(uint64_t ts, int nbAnt, uint32_t cir_size) {
    return (uint32_t)((ts * (uint64_t)nbAnt) % (uint64_t)cir_size);
}

/* ── Futex helpers ── */

static int v7_futex_wait(volatile uint32_t *addr, uint32_t expected, int timeout_ms)
{
    struct timespec ts;
    struct timespec *pts = NULL;
    if (timeout_ms > 0) {
        ts.tv_sec  = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000L;
        pts = &ts;
    }
    return (int)syscall(SYS_futex, addr, FUTEX_WAIT, expected, pts, NULL, 0);
}

static int v7_futex_wake(volatile uint32_t *addr)
{
    return (int)syscall(SYS_futex, addr, FUTEX_WAKE, 1, NULL, NULL, 0);
}

/* ── CUDA runtime loading ── */

static int load_cuda_runtime_v7(gpu_ipc_v7_ctx_t *ctx)
{
    const char *libs[] = { "libcudart.so", "libcudart.so.12", "libcudart.so.11", NULL };
    for (int i = 0; libs[i]; i++) {
        ctx->cuda_lib = dlopen(libs[i], RTLD_LAZY);
        if (ctx->cuda_lib) { LOG_V7("Loaded %s", libs[i]); break; }
    }
    if (!ctx->cuda_lib) { LOG_V7("Failed to load libcudart: %s", dlerror()); return -1; }

#define LOAD(name) do { \
    p_##name = (fn_##name)dlsym(ctx->cuda_lib, #name); \
    if (!p_##name) { LOG_V7("dlsym(%s) failed", #name); dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; return -1; } \
} while(0)
    LOAD(cudaMemcpy);
    LOAD(cudaIpcOpenMemHandle);
    LOAD(cudaIpcCloseMemHandle);
    LOAD(cudaSetDevice);
    LOAD(cudaGetErrorString);
    p_cudaMemset = (fn_cudaMemset)dlsym(ctx->cuda_lib, "cudaMemset");
    p_cudaDeviceSynchronize = (fn_cudaDeviceSynchronize)dlsym(ctx->cuda_lib, "cudaDeviceSynchronize");
#undef LOAD

    cudaError_t err = p_cudaSetDevice(0);
    if (err != cudaSuccess) {
        LOG_V7("cudaSetDevice(0) failed: %s", p_cudaGetErrorString(err));
        dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; return -1;
    }
    return 0;
}

/* ── IPC handle opener ── */

static void *open_handle_at(gpu_ipc_v7_ctx_t *ctx, int offset, const char *name)
{
    cudaIpcMemHandle_t handle;
    memcpy(handle.reserved, ctx->shm_raw + offset, GPU_IPC_V7_HANDLE_SIZE);
    void *ptr = NULL;
    cudaError_t err = p_cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        LOG_V7("cudaIpcOpenMemHandle(%s) failed: %s", name, p_cudaGetErrorString(err));
        return NULL;
    }
    LOG_V7("CLIENT: opened %s (ptr=%p)", name, ptr);
    return ptr;
}

/* ══════════════════════════════════════════════════════════════════
 * Initialization
 * ══════════════════════════════════════════════════════════════════ */

int gpu_ipc_v7_init(gpu_ipc_v7_ctx_t *ctx, gpu_ipc_v7_role_t role)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->role   = role;
    ctx->shm_fd = -1;

    if (load_cuda_runtime_v7(ctx) != 0) return -1;

    const char *ue_idx_env = getenv("RFSIM_GPU_IPC_UE_IDX");
    if (role == GPU_IPC_V7_ROLE_UE && ue_idx_env) {
        int ue_idx = atoi(ue_idx_env);
        snprintf(ctx->shm_path, sizeof(ctx->shm_path),
                 GPU_IPC_V7_SHM_DIR "/gpu_ipc_shm_ue%d", ue_idx);
        LOG_V7("UE role: per-UE SHM path %s (ue_idx=%d)", ctx->shm_path, ue_idx);
    } else {
        snprintf(ctx->shm_path, sizeof(ctx->shm_path), "%s", GPU_IPC_V7_SHM_PATH);
    }

    struct stat st;
    if (stat(GPU_IPC_V7_SHM_DIR, &st) != 0)
        if (mkdir(GPU_IPC_V7_SHM_DIR, 0777) != 0 && errno != EEXIST)
            { LOG_V7("mkdir failed: %s", strerror(errno)); return -1; }

    int wait = 0;
    while (access(ctx->shm_path, F_OK) != 0) {
        usleep(10000);
        if (++wait > 3000) { LOG_V7("timeout waiting for shm %s (30s)", ctx->shm_path); return -1; }
    }

    ctx->shm_fd = open(ctx->shm_path, O_RDWR, 0666);
    if (ctx->shm_fd < 0)
        { LOG_V7("open %s failed: %s", ctx->shm_path, strerror(errno)); return -1; }

    if (ftruncate(ctx->shm_fd, GPU_IPC_V7_SHM_SIZE) != 0)
        { LOG_V7("ftruncate failed: %s", strerror(errno)); close(ctx->shm_fd); return -1; }

    ctx->shm_raw = (char *)mmap(NULL, GPU_IPC_V7_SHM_SIZE,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                ctx->shm_fd, 0);
    if (ctx->shm_raw == MAP_FAILED)
        { LOG_V7("mmap failed: %s", strerror(errno)); close(ctx->shm_fd); return -1; }

    LOG_V7("CLIENT(%s): waiting for server...",
           role == GPU_IPC_V7_ROLE_GNB ? "gNB" : "UE");
    wait = 0;
    while (*shm_u32(ctx->shm_raw, _V7_OFF_MAGIC) != GPU_IPC_V7_MAGIC) {
        usleep(10000);
        if (++wait > 3000) { LOG_V7("timeout waiting for magic (30s)"); return -1; }
    }

    uint32_t ver = *shm_u32(ctx->shm_raw, _V7_OFF_VERSION);
    if (ver != GPU_IPC_V7_VERSION) {
        LOG_V7("version mismatch (got %u, expected %u)", ver, GPU_IPC_V7_VERSION);
        return -1;
    }

    ctx->dl_tx_nbAnt    = *shm_u32(ctx->shm_raw, _V7_OFF_DL_TX_NBANT);
    ctx->dl_tx_cir_size = *shm_u32(ctx->shm_raw, _V7_OFF_DL_TX_CIR_SIZE);
    ctx->dl_rx_nbAnt    = *shm_u32(ctx->shm_raw, _V7_OFF_DL_RX_NBANT);
    ctx->dl_rx_cir_size = *shm_u32(ctx->shm_raw, _V7_OFF_DL_RX_CIR_SIZE);
    ctx->ul_tx_nbAnt    = *shm_u32(ctx->shm_raw, _V7_OFF_UL_TX_NBANT);
    ctx->ul_tx_cir_size = *shm_u32(ctx->shm_raw, _V7_OFF_UL_TX_CIR_SIZE);
    ctx->ul_rx_nbAnt    = *shm_u32(ctx->shm_raw, _V7_OFF_UL_RX_NBANT);
    ctx->ul_rx_cir_size = *shm_u32(ctx->shm_raw, _V7_OFF_UL_RX_CIR_SIZE);

    LOG_V7("CLIENT: per-buffer config: dl_tx(ant=%u,cir=%u) dl_rx(ant=%u,cir=%u) "
           "ul_tx(ant=%u,cir=%u) ul_rx(ant=%u,cir=%u)",
           ctx->dl_tx_nbAnt, ctx->dl_tx_cir_size,
           ctx->dl_rx_nbAnt, ctx->dl_rx_cir_size,
           ctx->ul_tx_nbAnt, ctx->ul_tx_cir_size,
           ctx->ul_rx_nbAnt, ctx->ul_rx_cir_size);

    if (role == GPU_IPC_V7_ROLE_GNB) {
        ctx->gpu_dl_tx = open_handle_at(ctx, _V7_OFF_DL_TX_HANDLE, "dl_tx");
        ctx->gpu_ul_rx = open_handle_at(ctx, _V7_OFF_UL_RX_HANDLE, "ul_rx");
        if (!ctx->gpu_dl_tx || !ctx->gpu_ul_rx) return -1;
    } else {
        ctx->gpu_dl_rx = open_handle_at(ctx, _V7_OFF_DL_RX_HANDLE, "dl_rx");
        ctx->gpu_ul_tx = open_handle_at(ctx, _V7_OFF_UL_TX_HANDLE, "ul_tx");
        if (!ctx->gpu_dl_rx || !ctx->gpu_ul_tx) return -1;
    }

    ctx->initialized = 1;
    LOG_V7("CLIENT(%s): init complete (futex enabled)",
           role == GPU_IPC_V7_ROLE_GNB ? "gNB" : "UE");
    return 0;
}

void gpu_ipc_v7_cleanup(gpu_ipc_v7_ctx_t *ctx)
{
    if (!ctx->initialized) return;
    if (ctx->gpu_dl_tx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_tx);
    if (ctx->gpu_dl_rx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_rx);
    if (ctx->gpu_ul_tx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_tx);
    if (ctx->gpu_ul_rx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_rx);
    if (ctx->shm_raw && ctx->shm_raw != MAP_FAILED)
        munmap(ctx->shm_raw, GPU_IPC_V7_SHM_SIZE);
    if (ctx->shm_fd >= 0) close(ctx->shm_fd);
    if (ctx->cuda_lib) dlclose(ctx->cuda_lib);
    ctx->initialized = 0;
    LOG_V7("CLIENT(%s): cleanup done",
           ctx->role == GPU_IPC_V7_ROLE_GNB ? "gNB" : "UE");
}

/* ══════════════════════════════════════════════════════════════════
 * Internal: circular buffer write (H2D) with wrap + gap zero-fill
 * After completing GPU write: memory barrier → update timestamp →
 *   increment seq counter → futex_wake.
 * ══════════════════════════════════════════════════════════════════ */

static int v7_circ_write(gpu_ipc_v7_ctx_t *ctx, void *gpu_base,
                         const void *samples, int nsamps, int nbAnt,
                         uint64_t timestamp, uint32_t buf_cir_size,
                         int last_ts_off, int last_nsamps_off,
                         int seq_off)
{
    if (!ctx->initialized) return GPU_IPC_V7_ERROR;

    uint32_t cir = buf_cir_size;
    size_t total_samples = (size_t)nsamps * nbAnt;

    /* Gap zero-fill */
    if (ctx->lastWriteEnd > 0 && ctx->lastWriteEnd < timestamp && p_cudaMemset) {
        uint64_t gap_len = timestamp - ctx->lastWriteEnd;
        if (gap_len < (uint64_t)cir) {
            size_t gap_samples = (size_t)gap_len * nbAnt;
            uint32_t gap_off = circ_offset(ctx->lastWriteEnd, nbAnt, cir);
            if (gap_off + gap_samples <= cir) {
                p_cudaMemset((char *)gpu_base + (size_t)gap_off * GPU_IPC_V7_SAMPLE_SIZE,
                             0, gap_samples * GPU_IPC_V7_SAMPLE_SIZE);
            } else {
                size_t tail = cir - gap_off;
                p_cudaMemset((char *)gpu_base + (size_t)gap_off * GPU_IPC_V7_SAMPLE_SIZE,
                             0, tail * GPU_IPC_V7_SAMPLE_SIZE);
                p_cudaMemset(gpu_base, 0, (gap_samples - tail) * GPU_IPC_V7_SAMPLE_SIZE);
            }
        }
    }
    ctx->lastWriteEnd = timestamp + (uint64_t)nsamps;

    /* H2D copy with wrap-around */
    uint32_t off = circ_offset(timestamp, nbAnt, cir);
    size_t data_size = total_samples * GPU_IPC_V7_SAMPLE_SIZE;

    /* Periodic diagnostic: check source data content */
    {
        static uint64_t _write_count = 0;
        _write_count++;
        if (_write_count == 1 || _write_count == 10 || _write_count == 100 ||
            _write_count == 500 || _write_count == 1000 || _write_count == 5000) {
            int16_t *s16 = (int16_t *)samples;
            int total_i16 = (int)(total_samples * 2);
            int check_len = total_i16 < 2048 ? total_i16 : 2048;
            int src_nz = 0;
            int16_t src_max = 0;
            for (int i = 0; i < check_len; i++) {
                if (s16[i] != 0) src_nz++;
                int16_t av = s16[i] < 0 ? -s16[i] : s16[i];
                if (av > src_max) src_max = av;
            }
            LOG_V7("DL WRITE DIAG #%lu: src_nz=%d/%d src_max=%d "
                   "ts=%lu nsamps=%d nbAnt=%d off=%u cir=%u "
                   "first8=[%d,%d,%d,%d,%d,%d,%d,%d]",
                   (unsigned long)_write_count,
                   src_nz, check_len, (int)src_max,
                   (unsigned long)timestamp, nsamps, nbAnt, off, cir,
                   check_len>0?s16[0]:0, check_len>1?s16[1]:0,
                   check_len>2?s16[2]:0, check_len>3?s16[3]:0,
                   check_len>4?s16[4]:0, check_len>5?s16[5]:0,
                   check_len>6?s16[6]:0, check_len>7?s16[7]:0);
        }
    }

    if (off + total_samples <= cir) {
        void *dst = (char *)gpu_base + (size_t)off * GPU_IPC_V7_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(dst, samples, data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V7("write H2D failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
    } else {
        uint32_t tail = cir - off;
        size_t tail_bytes = (size_t)tail * GPU_IPC_V7_SAMPLE_SIZE;

        void *dst_tail = (char *)gpu_base + (size_t)off * GPU_IPC_V7_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(dst_tail, samples, tail_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V7("write H2D tail failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
        err = p_cudaMemcpy(gpu_base, (const char *)samples + tail_bytes,
                           data_size - tail_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V7("write H2D head failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
    }

    /*
     * GPU write complete → update metadata → wake waiter.
     * cudaMemcpy with default stream is synchronous on the host side,
     * so GPU write is already complete at this point.
     */
    if (last_ts_off >= 0) {
        *shm_u64(ctx->shm_raw, last_ts_off) = timestamp;
        if (last_nsamps_off >= 0)
            *shm_u32(ctx->shm_raw, last_nsamps_off) = (uint32_t)nsamps;
    }
    __sync_synchronize();

    if (seq_off >= 0) {
        __sync_fetch_and_add(shm_u32(ctx->shm_raw, seq_off), 1);
        v7_futex_wake(shm_u32(ctx->shm_raw, seq_off));
    }

    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════
 * Internal: circular buffer read (D2H) with futex wait.
 * This function owns all wait/timeout logic. Callers never loop.
 * ══════════════════════════════════════════════════════════════════ */

static int v7_circ_read(gpu_ipc_v7_ctx_t *ctx, void *gpu_base,
                        void *samples, int nsamps, int nbAnt,
                        uint64_t target_ts, uint32_t buf_cir_size,
                        int last_ts_off, int consumer_ts_off,
                        int seq_off, int timeout_ms)
{
    if (!ctx->initialized) return GPU_IPC_V7_ERROR;

    uint64_t need_end = target_ts + (uint64_t)nsamps - 1;

    /* Fast path: data already available */
    uint64_t last = *shm_u64(ctx->shm_raw, last_ts_off);
    if (last >= need_end)
        goto do_read;

    /* Slow path: futex wait with timeout */
    if (seq_off >= 0 && timeout_ms > 0) {
        uint32_t cur_seq = *shm_u32(ctx->shm_raw, seq_off);
        __sync_synchronize();
        last = *shm_u64(ctx->shm_raw, last_ts_off);
        if (last >= need_end)
            goto do_read;

        v7_futex_wait(shm_u32(ctx->shm_raw, seq_off), cur_seq, timeout_ms);

        /* Re-check after wakeup */
        __sync_synchronize();
        last = *shm_u64(ctx->shm_raw, last_ts_off);
        if (last < need_end)
            return GPU_IPC_V7_EMPTY;
    } else {
        return GPU_IPC_V7_EMPTY;
    }

do_read:
    ;
    uint32_t cir = buf_cir_size;
    uint32_t off = circ_offset(target_ts, nbAnt, cir);
    size_t total_samples = (size_t)nsamps * nbAnt;
    size_t data_size = total_samples * GPU_IPC_V7_SAMPLE_SIZE;

    if (off + total_samples <= cir) {
        void *src = (char *)gpu_base + (size_t)off * GPU_IPC_V7_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(samples, src, data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V7("read D2H failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
    } else {
        uint32_t tail = cir - off;
        size_t tail_bytes = (size_t)tail * GPU_IPC_V7_SAMPLE_SIZE;

        void *src_tail = (char *)gpu_base + (size_t)off * GPU_IPC_V7_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(samples, src_tail, tail_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V7("read D2H tail failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
        err = p_cudaMemcpy((char *)samples + tail_bytes, gpu_base,
                           data_size - tail_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V7("read D2H head failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V7_ERROR;
        }
    }

    if (consumer_ts_off >= 0)
        *shm_u64(ctx->shm_raw, consumer_ts_off) = target_ts + (uint64_t)nsamps - 1;

    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — each function uses its buffer's cir_size + futex
 * ══════════════════════════════════════════════════════════════════ */

int gpu_ipc_v7_dl_write(gpu_ipc_v7_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V7_ROLE_GNB)
        return GPU_IPC_V7_ERROR;
    return v7_circ_write(ctx, ctx->gpu_dl_tx, samples, nsamps, nbAnt, timestamp,
                         ctx->dl_tx_cir_size,
                         _V7_OFF_LAST_DL_TX_TS, _V7_OFF_LAST_DL_TX_NSAMPS,
                         _V7_OFF_DL_TX_SEQ);
}

int gpu_ipc_v7_dl_read(gpu_ipc_v7_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t target_ts,
                       int timeout_ms)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V7_ROLE_UE)
        return GPU_IPC_V7_ERROR;
    return v7_circ_read(ctx, ctx->gpu_dl_rx, samples, nsamps, nbAnt, target_ts,
                        ctx->dl_rx_cir_size,
                        _V7_OFF_LAST_DL_RX_TS, _V7_OFF_DL_CONSUMER_TS,
                        _V7_OFF_DL_RX_SEQ, timeout_ms);
}

int gpu_ipc_v7_ul_write(gpu_ipc_v7_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V7_ROLE_UE)
        return GPU_IPC_V7_ERROR;
    return v7_circ_write(ctx, ctx->gpu_ul_tx, samples, nsamps, nbAnt, timestamp,
                         ctx->ul_tx_cir_size,
                         _V7_OFF_LAST_UL_TX_TS, _V7_OFF_LAST_UL_TX_NSAMPS,
                         _V7_OFF_UL_TX_SEQ);
}

int gpu_ipc_v7_ul_read(gpu_ipc_v7_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t target_ts,
                       int timeout_ms)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V7_ROLE_GNB)
        return GPU_IPC_V7_ERROR;
    return v7_circ_read(ctx, ctx->gpu_ul_rx, samples, nsamps, nbAnt, target_ts,
                        ctx->ul_rx_cir_size,
                        _V7_OFF_LAST_UL_RX_TS, _V7_OFF_UL_CONSUMER_TS,
                        _V7_OFF_UL_RX_SEQ, timeout_ms);
}

void gpu_ipc_v7_set_ul_consumer_ts(gpu_ipc_v7_ctx_t *ctx, uint64_t ts)
{
    if (!ctx->initialized) return;
    if (ctx->role != GPU_IPC_V7_ROLE_GNB) return;
    *shm_u64(ctx->shm_raw, _V7_OFF_UL_CONSUMER_TS) = ts;
}
