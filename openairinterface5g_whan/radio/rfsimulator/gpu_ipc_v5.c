/*
 * gpu_ipc_v5.c - CUDA IPC Circular Buffer (G1B, Single-UE)
 *
 * Circular buffer with timestamp indexing over GPU shared memory.
 * Socket-mode 1:1 correspondence: gap zero-fill, wait-with-timeout,
 * initial sync via first received timestamp.
 *
 * CUDA loaded at runtime via dlopen — no build-time dependency.
 */

#include "gpu_ipc_v5.h"

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

/* ── CUDA type definitions (no CUDA headers needed) ── */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaIpcMemLazyEnablePeerAccess 1

typedef struct { char reserved[GPU_IPC_V5_HANDLE_SIZE]; } cudaIpcMemHandle_t;

typedef cudaError_t (*fn_cudaMemcpy)(void *, const void *, size_t, int);
typedef cudaError_t (*fn_cudaMemset)(void *, int, size_t);
typedef cudaError_t (*fn_cudaIpcOpenMemHandle)(void **, cudaIpcMemHandle_t, unsigned int);
typedef cudaError_t (*fn_cudaIpcCloseMemHandle)(void *);
typedef cudaError_t (*fn_cudaSetDevice)(int);
typedef const char* (*fn_cudaGetErrorString)(cudaError_t);

static fn_cudaMemcpy            p_cudaMemcpy;
static fn_cudaMemset            p_cudaMemset;
static fn_cudaIpcOpenMemHandle  p_cudaIpcOpenMemHandle;
static fn_cudaIpcCloseMemHandle p_cudaIpcCloseMemHandle;
static fn_cudaSetDevice         p_cudaSetDevice;
static fn_cudaGetErrorString    p_cudaGetErrorString;

#define LOG_V5(fmt, ...) fprintf(stderr, "[GPU_IPC_V5] " fmt "\n", ##__VA_ARGS__)

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

/* ── CUDA runtime loading ── */

static int load_cuda_runtime_v5(gpu_ipc_v5_ctx_t *ctx)
{
    const char *libs[] = { "libcudart.so", "libcudart.so.12", "libcudart.so.11", NULL };
    for (int i = 0; libs[i]; i++) {
        ctx->cuda_lib = dlopen(libs[i], RTLD_LAZY);
        if (ctx->cuda_lib) { LOG_V5("Loaded %s", libs[i]); break; }
    }
    if (!ctx->cuda_lib) { LOG_V5("Failed to load libcudart: %s", dlerror()); return -1; }

#define LOAD(name) do { \
    p_##name = (fn_##name)dlsym(ctx->cuda_lib, #name); \
    if (!p_##name) { LOG_V5("dlsym(%s) failed", #name); dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; return -1; } \
} while(0)
    LOAD(cudaMemcpy);
    LOAD(cudaIpcOpenMemHandle);
    LOAD(cudaIpcCloseMemHandle);
    LOAD(cudaSetDevice);
    LOAD(cudaGetErrorString);
    p_cudaMemset = (fn_cudaMemset)dlsym(ctx->cuda_lib, "cudaMemset");
#undef LOAD

    cudaError_t err = p_cudaSetDevice(0);
    if (err != cudaSuccess) {
        LOG_V5("cudaSetDevice(0) failed: %s", p_cudaGetErrorString(err));
        dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; return -1;
    }
    return 0;
}

/* ── IPC handle opener ── */

static void *open_handle_at(gpu_ipc_v5_ctx_t *ctx, int offset, const char *name)
{
    cudaIpcMemHandle_t handle;
    memcpy(handle.reserved, ctx->shm_raw + offset, GPU_IPC_V5_HANDLE_SIZE);
    void *ptr = NULL;
    cudaError_t err = p_cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        LOG_V5("cudaIpcOpenMemHandle(%s) failed: %s", name, p_cudaGetErrorString(err));
        return NULL;
    }
    LOG_V5("CLIENT: opened %s (ptr=%p)", name, ptr);
    return ptr;
}

/* ══════════════════════════════════════════════════════════════════
 * Initialization
 * ══════════════════════════════════════════════════════════════════ */

int gpu_ipc_v5_init(gpu_ipc_v5_ctx_t *ctx, gpu_ipc_v5_role_t role)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->role   = role;
    ctx->shm_fd = -1;

    if (load_cuda_runtime_v5(ctx) != 0) return -1;

    struct stat st;
    if (stat(GPU_IPC_V5_SHM_DIR, &st) != 0)
        if (mkdir(GPU_IPC_V5_SHM_DIR, 0777) != 0 && errno != EEXIST)
            { LOG_V5("mkdir failed: %s", strerror(errno)); return -1; }

    int wait = 0;
    while (access(GPU_IPC_V5_SHM_PATH, F_OK) != 0) {
        usleep(10000);
        if (++wait > 3000) { LOG_V5("timeout waiting for shm (30s)"); return -1; }
    }

    ctx->shm_fd = open(GPU_IPC_V5_SHM_PATH, O_RDWR, 0666);
    if (ctx->shm_fd < 0)
        { LOG_V5("open failed: %s", strerror(errno)); return -1; }

    if (ftruncate(ctx->shm_fd, GPU_IPC_V5_SHM_SIZE) != 0)
        { LOG_V5("ftruncate failed: %s", strerror(errno)); close(ctx->shm_fd); return -1; }

    ctx->shm_raw = (char *)mmap(NULL, GPU_IPC_V5_SHM_SIZE,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                ctx->shm_fd, 0);
    if (ctx->shm_raw == MAP_FAILED)
        { LOG_V5("mmap failed: %s", strerror(errno)); close(ctx->shm_fd); return -1; }

    LOG_V5("CLIENT(%s): waiting for server...",
           role == GPU_IPC_V5_ROLE_GNB ? "gNB" : "UE");
    wait = 0;
    while (*shm_u32(ctx->shm_raw, _V5_OFF_MAGIC) != GPU_IPC_V5_MAGIC) {
        usleep(10000);
        if (++wait > 3000) { LOG_V5("timeout waiting for magic (30s)"); return -1; }
    }

    uint32_t ver = *shm_u32(ctx->shm_raw, _V5_OFF_VERSION);
    if (ver != GPU_IPC_V5_VERSION) {
        LOG_V5("version mismatch (got %u, expected %u)", ver, GPU_IPC_V5_VERSION);
        return -1;
    }

    ctx->cir_size = *shm_u32(ctx->shm_raw, _V5_OFF_CIR_SIZE);
    LOG_V5("CLIENT: server ready (cir_size=%u)", ctx->cir_size);

    if (role == GPU_IPC_V5_ROLE_GNB) {
        ctx->gpu_dl_tx = open_handle_at(ctx, _V5_OFF_DL_TX_HANDLE, "dl_tx");
        ctx->gpu_ul_rx = open_handle_at(ctx, _V5_OFF_UL_RX_HANDLE, "ul_rx");
        if (!ctx->gpu_dl_tx || !ctx->gpu_ul_rx) return -1;
    } else {
        ctx->gpu_dl_rx = open_handle_at(ctx, _V5_OFF_DL_RX_HANDLE, "dl_rx");
        ctx->gpu_ul_tx = open_handle_at(ctx, _V5_OFF_UL_TX_HANDLE, "ul_tx");
        if (!ctx->gpu_dl_rx || !ctx->gpu_ul_tx) return -1;
    }

    ctx->initialized = 1;
    LOG_V5("CLIENT(%s): init complete (cir_size=%u, buf_bytes=%zu)",
           role == GPU_IPC_V5_ROLE_GNB ? "gNB" : "UE",
           ctx->cir_size, (size_t)ctx->cir_size * GPU_IPC_V5_SAMPLE_SIZE);
    return 0;
}

void gpu_ipc_v5_cleanup(gpu_ipc_v5_ctx_t *ctx)
{
    if (!ctx->initialized) return;
    if (ctx->gpu_dl_tx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_tx);
    if (ctx->gpu_dl_rx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_rx);
    if (ctx->gpu_ul_tx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_tx);
    if (ctx->gpu_ul_rx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_rx);
    if (ctx->shm_raw && ctx->shm_raw != MAP_FAILED)
        munmap(ctx->shm_raw, GPU_IPC_V5_SHM_SIZE);
    if (ctx->shm_fd >= 0) close(ctx->shm_fd);
    if (ctx->cuda_lib) dlclose(ctx->cuda_lib);
    ctx->initialized = 0;
    LOG_V5("CLIENT(%s): cleanup done",
           ctx->role == GPU_IPC_V5_ROLE_GNB ? "gNB" : "UE");
}

/* ══════════════════════════════════════════════════════════════════
 * Internal: circular buffer write (H2D) with wrap + gap zero-fill
 * ══════════════════════════════════════════════════════════════════ */

static int v5_circ_write(gpu_ipc_v5_ctx_t *ctx, void *gpu_base,
                         const void *samples, int nsamps, int nbAnt,
                         uint64_t timestamp, int last_ts_off,
                         int last_nsamps_off, int last_nbant_off)
{
    if (!ctx->initialized) return GPU_IPC_V5_ERROR;

    uint32_t cir = ctx->cir_size;
    size_t total_samples = (size_t)nsamps * nbAnt;

    /* Gap zero-fill: if there's a gap since last write, zero-fill it */
    if (ctx->lastWriteEnd > 0 && ctx->lastWriteEnd < timestamp && p_cudaMemset) {
        uint64_t gap_len = timestamp - ctx->lastWriteEnd;
        if (gap_len < (uint64_t)cir) {
            size_t gap_samples = (size_t)gap_len * nbAnt;
            uint32_t gap_off = circ_offset(ctx->lastWriteEnd, nbAnt, cir);
            if (gap_off + gap_samples <= cir) {
                p_cudaMemset((char *)gpu_base + (size_t)gap_off * GPU_IPC_V5_SAMPLE_SIZE,
                             0, gap_samples * GPU_IPC_V5_SAMPLE_SIZE);
            } else {
                size_t tail = cir - gap_off;
                p_cudaMemset((char *)gpu_base + (size_t)gap_off * GPU_IPC_V5_SAMPLE_SIZE,
                             0, tail * GPU_IPC_V5_SAMPLE_SIZE);
                p_cudaMemset(gpu_base, 0, (gap_samples - tail) * GPU_IPC_V5_SAMPLE_SIZE);
            }
        }
    }
    ctx->lastWriteEnd = timestamp + (uint64_t)nsamps;

    /* H2D copy with wrap-around */
    uint32_t off = circ_offset(timestamp, nbAnt, cir);
    size_t data_size = total_samples * GPU_IPC_V5_SAMPLE_SIZE;

    if (off + total_samples <= cir) {
        void *dst = (char *)gpu_base + (size_t)off * GPU_IPC_V5_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(dst, samples, data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V5("write H2D failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
    } else {
        uint32_t tail = cir - off;
        size_t tail_bytes = (size_t)tail * GPU_IPC_V5_SAMPLE_SIZE;

        void *dst_tail = (char *)gpu_base + (size_t)off * GPU_IPC_V5_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(dst_tail, samples, tail_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V5("write H2D tail failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
        err = p_cudaMemcpy(gpu_base, (const char *)samples + tail_bytes,
                           data_size - tail_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_V5("write H2D head failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
    }

    /* Update SHM timestamp atomically */
    if (last_ts_off >= 0) {
        *shm_u64(ctx->shm_raw, last_ts_off) = timestamp;
        if (last_nsamps_off >= 0)
            *shm_u32(ctx->shm_raw, last_nsamps_off) = (uint32_t)nsamps;
        if (last_nbant_off >= 0)
            *shm_u32(ctx->shm_raw, last_nbant_off) = (uint32_t)nbAnt;
    }
    __sync_synchronize();

    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════
 * Internal: circular buffer read (D2H) with wrap
 * ══════════════════════════════════════════════════════════════════ */

static int v5_circ_read(gpu_ipc_v5_ctx_t *ctx, void *gpu_base,
                        void *samples, int nsamps, int nbAnt,
                        uint64_t target_ts, int last_ts_off,
                        int consumer_ts_off)
{
    if (!ctx->initialized) return GPU_IPC_V5_ERROR;

    /* Check if the FULL range [target_ts, target_ts+nsamps) has been written */
    uint64_t last = *shm_u64(ctx->shm_raw, last_ts_off);
    uint64_t need_end = target_ts + (uint64_t)nsamps - 1;
    if (last < need_end)
        return GPU_IPC_V5_EMPTY;

    uint32_t cir = ctx->cir_size;
    uint32_t off = circ_offset(target_ts, nbAnt, cir);
    size_t total_samples = (size_t)nsamps * nbAnt;
    size_t data_size = total_samples * GPU_IPC_V5_SAMPLE_SIZE;

    if (off + total_samples <= cir) {
        void *src = (char *)gpu_base + (size_t)off * GPU_IPC_V5_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(samples, src, data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V5("read D2H failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
    } else {
        uint32_t tail = cir - off;
        size_t tail_bytes = (size_t)tail * GPU_IPC_V5_SAMPLE_SIZE;

        void *src_tail = (char *)gpu_base + (size_t)off * GPU_IPC_V5_SAMPLE_SIZE;
        cudaError_t err = p_cudaMemcpy(samples, src_tail, tail_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V5("read D2H tail failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
        err = p_cudaMemcpy((char *)samples + tail_bytes, gpu_base,
                           data_size - tail_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_V5("read D2H head failed: %s", p_cudaGetErrorString(err));
            return GPU_IPC_V5_ERROR;
        }
    }

    if (consumer_ts_off >= 0)
        *shm_u64(ctx->shm_raw, consumer_ts_off) = target_ts + (uint64_t)nsamps - 1;

    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════
 * Public API
 * ══════════════════════════════════════════════════════════════════ */

int gpu_ipc_v5_dl_write(gpu_ipc_v5_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V5_ROLE_GNB)
        return GPU_IPC_V5_ERROR;
    return v5_circ_write(ctx, ctx->gpu_dl_tx, samples, nsamps, nbAnt, timestamp,
                         _V5_OFF_LAST_DL_TX_TS, _V5_OFF_LAST_DL_TX_NSAMPS,
                         _V5_OFF_LAST_DL_TX_NBANT);
}

int gpu_ipc_v5_dl_read(gpu_ipc_v5_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t target_ts)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V5_ROLE_UE)
        return GPU_IPC_V5_ERROR;
    return v5_circ_read(ctx, ctx->gpu_dl_rx, samples, nsamps, nbAnt, target_ts,
                        _V5_OFF_LAST_DL_RX_TS, _V5_OFF_DL_CONSUMER_TS);
}

int gpu_ipc_v5_ul_write(gpu_ipc_v5_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V5_ROLE_UE)
        return GPU_IPC_V5_ERROR;
    return v5_circ_write(ctx, ctx->gpu_ul_tx, samples, nsamps, nbAnt, timestamp,
                         _V5_OFF_LAST_UL_TX_TS, _V5_OFF_LAST_UL_TX_NSAMPS,
                         _V5_OFF_LAST_UL_TX_NBANT);
}

int gpu_ipc_v5_ul_read(gpu_ipc_v5_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t target_ts)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V5_ROLE_GNB)
        return GPU_IPC_V5_ERROR;
    return v5_circ_read(ctx, ctx->gpu_ul_rx, samples, nsamps, nbAnt, target_ts,
                        _V5_OFF_LAST_UL_RX_TS, _V5_OFF_UL_CONSUMER_TS);
}
