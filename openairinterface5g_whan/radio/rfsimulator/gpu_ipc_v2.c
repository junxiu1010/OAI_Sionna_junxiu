/*
 * gpu_ipc_v2.c - Multi-UE CUDA IPC with SPSC ring buffers (G1A)
 *
 * All CUDA functions loaded at runtime via dlopen (no build-time dependency).
 * OAI processes (gNB/UE) are always CLIENTs; Proxy (Python) is the SERVER.
 *
 * Each data path uses a lock-free SPSC ring buffer in GPU memory,
 * with head/tail counters and per-slot metadata in shared memory.
 */

#include "gpu_ipc_v2.h"

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
#include <sched.h>

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaIpcMemLazyEnablePeerAccess 1

typedef struct {
    char reserved[GPU_IPC_V2_HANDLE_SIZE];
} cudaIpcMemHandle_t;

typedef cudaError_t (*fn_cudaMemcpy)(void *dst, const void *src,
                                     size_t count, int kind);
typedef cudaError_t (*fn_cudaIpcOpenMemHandle)(void **devPtr,
                                               cudaIpcMemHandle_t handle,
                                               unsigned int flags);
typedef cudaError_t (*fn_cudaIpcCloseMemHandle)(void *devPtr);
typedef cudaError_t (*fn_cudaSetDevice)(int device);
typedef const char* (*fn_cudaGetErrorString)(cudaError_t error);

static fn_cudaMemcpy            p_cudaMemcpy;
static fn_cudaIpcOpenMemHandle  p_cudaIpcOpenMemHandle;
static fn_cudaIpcCloseMemHandle p_cudaIpcCloseMemHandle;
static fn_cudaSetDevice         p_cudaSetDevice;
static fn_cudaGetErrorString    p_cudaGetErrorString;

#define LOG_V2(fmt, ...) fprintf(stderr, "[GPU_IPC_V2] " fmt "\n", ##__VA_ARGS__)

/* ── SHM pointer helpers ── */

static inline volatile int32_t *shm_i32(char *base, int off) {
    return (volatile int32_t *)(base + off);
}
static inline volatile uint32_t *shm_u32(char *base, int off) {
    return (volatile uint32_t *)(base + off);
}
static inline volatile uint64_t *shm_u64(char *base, int off) {
    return (volatile uint64_t *)(base + off);
}

/* ── Ring helper: slot metadata offset within SHM ── */

static inline int ring_slot_meta_off(int ring_base, int slot_idx) {
    return ring_base + _V2_RING_OFF_SLOTS + slot_idx * _V2_RING_SLOT_META_SIZE;
}

/* ── CUDA runtime loading ── */

static int load_cuda_runtime_v2(gpu_ipc_v2_ctx_t *ctx)
{
    const char *libs[] = { "libcudart.so", "libcudart.so.12", "libcudart.so.11", NULL };

    for (int i = 0; libs[i]; i++) {
        ctx->cuda_lib = dlopen(libs[i], RTLD_LAZY);
        if (ctx->cuda_lib) {
            LOG_V2("Loaded %s", libs[i]);
            break;
        }
    }
    if (!ctx->cuda_lib) {
        LOG_V2("Failed to load libcudart.so: %s", dlerror());
        return -1;
    }

#define LOAD_SYM(name) do { \
    p_##name = (fn_##name)dlsym(ctx->cuda_lib, #name); \
    if (!p_##name) { \
        LOG_V2("dlsym(%s) failed: %s", #name, dlerror()); \
        dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; return -1; \
    } \
} while(0)

    LOAD_SYM(cudaMemcpy);
    LOAD_SYM(cudaIpcOpenMemHandle);
    LOAD_SYM(cudaIpcCloseMemHandle);
    LOAD_SYM(cudaSetDevice);
    LOAD_SYM(cudaGetErrorString);
#undef LOAD_SYM

    cudaError_t err = p_cudaSetDevice(0);
    if (err != cudaSuccess) {
        LOG_V2("cudaSetDevice(0) failed: %s", p_cudaGetErrorString(err));
        dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL;
        return -1;
    }
    return 0;
}

static void *open_handle_at(gpu_ipc_v2_ctx_t *ctx, int offset, const char *name)
{
    cudaIpcMemHandle_t handle;
    memcpy(handle.reserved, ctx->shm_raw + offset, GPU_IPC_V2_HANDLE_SIZE);

    void *ptr = NULL;
    cudaError_t err = p_cudaIpcOpenMemHandle(&ptr, handle,
                                             cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        LOG_V2("cudaIpcOpenMemHandle(%s) failed: %s", name, p_cudaGetErrorString(err));
        return NULL;
    }
    LOG_V2("CLIENT: opened %s (ptr=%p)", name, ptr);
    return ptr;
}

/* ── Initialization ── */

int gpu_ipc_v2_init(gpu_ipc_v2_ctx_t *ctx, gpu_ipc_v2_role_t role, int ue_idx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->role = role;
    ctx->ue_idx = ue_idx;
    ctx->shm_fd = -1;

    if (load_cuda_runtime_v2(ctx) != 0)
        return -1;

    /* Wait for SHM file */
    struct stat st;
    if (stat(GPU_IPC_V2_SHM_DIR, &st) != 0) {
        if (mkdir(GPU_IPC_V2_SHM_DIR, 0777) != 0 && errno != EEXIST) {
            LOG_V2("mkdir failed: %s", strerror(errno));
            return -1;
        }
    }

    int wait_count = 0;
    while (access(GPU_IPC_V2_SHM_PATH, F_OK) != 0) {
        usleep(10000);
        if (++wait_count > 3000) {
            LOG_V2("CLIENT: timeout waiting for shm file (30s)");
            return -1;
        }
    }

    ctx->shm_fd = open(GPU_IPC_V2_SHM_PATH, O_RDWR, 0666);
    if (ctx->shm_fd < 0) {
        LOG_V2("open(%s) failed: %s", GPU_IPC_V2_SHM_PATH, strerror(errno));
        return -1;
    }

    if (ftruncate(ctx->shm_fd, GPU_IPC_V2_SHM_SIZE) != 0) {
        LOG_V2("ftruncate failed: %s", strerror(errno));
        close(ctx->shm_fd);
        return -1;
    }

    ctx->shm_raw = (char *)mmap(NULL, GPU_IPC_V2_SHM_SIZE,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                ctx->shm_fd, 0);
    if (ctx->shm_raw == MAP_FAILED) {
        LOG_V2("mmap failed: %s", strerror(errno));
        close(ctx->shm_fd);
        return -1;
    }

    /* Wait for magic */
    LOG_V2("CLIENT(%s idx=%d): waiting for server...",
           role == GPU_IPC_V2_ROLE_GNB ? "gNB" : "UE", ue_idx);
    wait_count = 0;
    while (*shm_u32(ctx->shm_raw, _V2_OFF_MAGIC) != GPU_IPC_V2_MAGIC) {
        usleep(10000);
        if (++wait_count > 3000) {
            LOG_V2("CLIENT: timeout waiting for magic (30s)");
            return -1;
        }
    }

    uint32_t ver = *shm_u32(ctx->shm_raw, _V2_OFF_VERSION);
    if (ver != GPU_IPC_V2_VERSION) {
        LOG_V2("CLIENT: version mismatch (got %u, expected %u)", ver, GPU_IPC_V2_VERSION);
        return -1;
    }

    ctx->buf_size    = *shm_u32(ctx->shm_raw, _V2_OFF_BUF_SIZE);
    ctx->num_ues     = *shm_u32(ctx->shm_raw, _V2_OFF_NUM_UES);
    ctx->ring_depth  = *shm_u32(ctx->shm_raw, _V2_OFF_RING_DEPTH);

    if (ctx->ring_depth != GPU_IPC_V2_RING_DEPTH) {
        LOG_V2("CLIENT: ring_depth mismatch (shm=%d, compiled=%d)",
               ctx->ring_depth, GPU_IPC_V2_RING_DEPTH);
        return -1;
    }

    LOG_V2("CLIENT: server ready (num_ues=%d, buf_size=%zu, ring_depth=%d)",
           ctx->num_ues, ctx->buf_size, ctx->ring_depth);

    if (role == GPU_IPC_V2_ROLE_GNB) {
        ctx->gpu_dl_tx = open_handle_at(ctx, _V2_OFF_DL_TX_HANDLE, "dl_tx_ring");
        ctx->gpu_ul_rx = open_handle_at(ctx, _V2_OFF_UL_RX_HANDLE, "ul_rx_ring");
        if (!ctx->gpu_dl_tx || !ctx->gpu_ul_rx) return -1;
    } else {
        if (ue_idx < 0 || ue_idx >= (int)ctx->num_ues) {
            LOG_V2("CLIENT: ue_idx=%d out of range (num_ues=%d)", ue_idx, ctx->num_ues);
            return -1;
        }
        char label[32];
        snprintf(label, sizeof(label), "dl_rx_ring[%d]", ue_idx);
        ctx->gpu_dl_rx = open_handle_at(ctx,
            _V2_OFF_DL_RX_HANDLES + ue_idx * GPU_IPC_V2_HANDLE_SIZE, label);
        snprintf(label, sizeof(label), "ul_tx_ring[%d]", ue_idx);
        ctx->gpu_ul_tx = open_handle_at(ctx,
            _V2_OFF_UL_TX_HANDLES + ue_idx * GPU_IPC_V2_HANDLE_SIZE, label);
        if (!ctx->gpu_dl_rx || !ctx->gpu_ul_tx) return -1;

        ctx->dl_leftover = (char *)calloc(1, ctx->buf_size);
        if (!ctx->dl_leftover) {
            LOG_V2("CLIENT: failed to alloc dl_leftover (%zu bytes)", ctx->buf_size);
            return -1;
        }
        ctx->dl_leftover_bytes = 0;
        ctx->dl_leftover_off   = 0;
        ctx->dl_leftover_ts    = 0;
        ctx->dl_slot_full_bytes = 0;
    }
    ctx->ul_ts_synced = 0;

    ctx->initialized = 1;
    LOG_V2("CLIENT(%s idx=%d): initialization complete",
           role == GPU_IPC_V2_ROLE_GNB ? "gNB" : "UE", ue_idx);
    return 0;
}

void gpu_ipc_v2_cleanup(gpu_ipc_v2_ctx_t *ctx)
{
    if (!ctx->initialized) return;

    if (ctx->gpu_dl_tx)  p_cudaIpcCloseMemHandle(ctx->gpu_dl_tx);
    if (ctx->gpu_ul_rx)  p_cudaIpcCloseMemHandle(ctx->gpu_ul_rx);
    if (ctx->gpu_dl_rx)  p_cudaIpcCloseMemHandle(ctx->gpu_dl_rx);
    if (ctx->gpu_ul_tx)  p_cudaIpcCloseMemHandle(ctx->gpu_ul_tx);

    if (ctx->dl_leftover) {
        free(ctx->dl_leftover);
        ctx->dl_leftover = NULL;
    }

    if (ctx->shm_raw && ctx->shm_raw != MAP_FAILED)
        munmap(ctx->shm_raw, GPU_IPC_V2_SHM_SIZE);
    if (ctx->shm_fd >= 0)
        close(ctx->shm_fd);
    if (ctx->cuda_lib)
        dlclose(ctx->cuda_lib);

    ctx->initialized = 0;
    LOG_V2("CLIENT(%s idx=%d): cleanup done",
           ctx->role == GPU_IPC_V2_ROLE_GNB ? "gNB" : "UE", ctx->ue_idx);
}

/* ══════════════════════════════════════════════════════════════════════
 * gNB DL write: enqueue to dl_tx ring (blocking when full)
 * ══════════════════════════════════════════════════════════════════════ */

int gpu_ipc_v2_dl_write(gpu_ipc_v2_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V2_ROLE_GNB) return -1;

    size_t data_size = (size_t)nsamps * nbAnt * 4;
    if (data_size > ctx->buf_size) {
        LOG_V2("dl_write: data_size %zu > slot_size %zu", data_size, ctx->buf_size);
        return -1;
    }

    const int roff = _V2_OFF_DL_TX_RING;
    uint32_t head = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD),
                                    __ATOMIC_RELAXED);

    /* Block until ring has space */
    for (;;) {
        uint32_t tail = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL),
                                        __ATOMIC_ACQUIRE);
        if ((head - tail) < (uint32_t)ctx->ring_depth)
            break;
        sched_yield();
    }

    uint32_t slot = head & GPU_IPC_V2_RING_MASK;
    void *dst = (char *)ctx->gpu_dl_tx + (size_t)slot * ctx->buf_size;

    cudaError_t err = p_cudaMemcpy(dst, samples, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_V2("dl_write H2D failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    int meta = ring_slot_meta_off(roff, slot);
    *shm_u64(ctx->shm_raw, meta + _V2_SLOT_OFF_TS)     = timestamp;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_NSAMPS)  = nsamps;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_NBANT)   = nbAnt;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_DSIZE)   = (int32_t)data_size;

    __atomic_store_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD),
                     head + 1, __ATOMIC_RELEASE);
    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════════
 * UE DL read: dequeue from dl_rx[ue_idx] ring (blocking when empty)
 * ══════════════════════════════════════════════════════════════════════ */

int gpu_ipc_v2_dl_read(gpu_ipc_v2_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t *timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V2_ROLE_UE) return -1;
    static int _dl_read_call = 0;
    int call_nr = _dl_read_call++;

    size_t want = (size_t)nsamps * nbAnt * 4;
    char  *dst  = (char *)samples;
    size_t filled = 0;

    /* Phase 1: drain leftover from a previous partial read */
    if (ctx->dl_leftover_bytes > 0) {
        size_t from_left = (size_t)ctx->dl_leftover_bytes < want
                         ? (size_t)ctx->dl_leftover_bytes : want;
        memcpy(dst, ctx->dl_leftover + ctx->dl_leftover_off, from_left);
        ctx->dl_leftover_off   += (int)from_left;
        ctx->dl_leftover_bytes -= (int)from_left;
        filled += from_left;
        if (timestamp)
            *timestamp = ctx->dl_leftover_ts;
        if (call_nr < 200)
            LOG_V2("dl_read[%d] #%d leftover drain %zu/%zu bytes (off=%d rem=%d)",
                   ctx->ue_idx, call_nr, from_left, want, ctx->dl_leftover_off, ctx->dl_leftover_bytes);
        if (filled >= want)
            return nsamps;
    }

    /* Phase 2: fetch ONE ring slot.
     * Each ring slot maps to one NR slot. We consume exactly one slot
     * per call, with leftover handling for partial reads (syncInFrame).
     * Mixed slots have fewer bytes (only DL symbols); the remainder
     * must be zero-padded, NOT filled from the next ring slot. */
    int roff = _V2_OFF_DL_RX_RINGS + ctx->ue_idx * _V2_RING_CTRL_SIZE;
    uint32_t tail = __atomic_load_n(
        shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL), __ATOMIC_RELAXED);

    int wait_iters = 0;
    uint32_t head;
    for (;;) {
        head = __atomic_load_n(
            shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD), __ATOMIC_ACQUIRE);
        if (head != tail)
            break;
        if (++wait_iters > 1000) {
            if (call_nr < 200)
                LOG_V2("dl_read[%d] #%d TIMEOUT ring empty (filled=%zu want=%zu h=%u t=%u)",
                       ctx->ue_idx, call_nr, filled, want, head, tail);
            if (filled == 0)
                return 0;
            memset(dst + filled, 0, want - filled);
            return nsamps;
        }
        usleep(10);
    }

    uint32_t slot_idx = tail & GPU_IPC_V2_RING_MASK;
    int meta = ring_slot_meta_off(roff, slot_idx);

    uint64_t slot_ts = *shm_u64(ctx->shm_raw, meta + _V2_SLOT_OFF_TS);
    int32_t  avail   = *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_DSIZE);
    if (avail <= 0) avail = 0;

    if (filled == 0 && timestamp)
        *timestamp = slot_ts;

    void *src = (char *)ctx->gpu_dl_rx + (size_t)slot_idx * ctx->buf_size;
    size_t actual_bytes = (size_t)avail < ctx->buf_size ? (size_t)avail : ctx->buf_size;

    cudaError_t err = p_cudaMemcpy(ctx->dl_leftover, src, actual_bytes,
                                   cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_V2("dl_read[%d] D2H failed: %s", ctx->ue_idx, p_cudaGetErrorString(err));
        return -1;
    }

    __atomic_store_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL),
                     tail + 1, __ATOMIC_RELEASE);

    /* Standard slot stream size = nsamps * nbAnt * 4.
     * Set once from the caller's parameters (readFrame passes the true
     * slot size on every call).  Ring entries may be larger due to
     * sf_extension; those extra prefix bytes must be skipped. */
    if (ctx->dl_slot_full_bytes == 0)
        ctx->dl_slot_full_bytes = (size_t)nsamps * nbAnt * 4;

    size_t slot_bytes = ctx->dl_slot_full_bytes;

    /* If the entry has MORE bytes than the standard slot (sf_extension),
     * the extra bytes are a time-domain prefix that overlaps with the
     * tail of the previous slot in the continuous stream.  Skip it. */
    size_t skip = 0;
    if (actual_bytes > slot_bytes) {
        skip = actual_bytes - slot_bytes;
        actual_bytes = slot_bytes;
    }

    /* Zero-pad short entries (mixed slots) in dl_leftover so that the
     * continuous stream has no gaps when reads span ring entries. */
    if (actual_bytes < slot_bytes)
        memset(ctx->dl_leftover + skip + actual_bytes, 0,
               slot_bytes - actual_bytes);

    size_t remaining_want = want - filled;
    size_t take = slot_bytes < remaining_want ? slot_bytes : remaining_want;

    memcpy(dst + filled, ctx->dl_leftover + skip, take);
    filled += take;

    if (take < slot_bytes) {
        ctx->dl_leftover_off   = (int)(skip + take);
        ctx->dl_leftover_bytes = (int)(slot_bytes - take);
        ctx->dl_leftover_ts    = slot_ts;
    } else {
        ctx->dl_leftover_off   = 0;
        ctx->dl_leftover_bytes = 0;
    }

    if (filled < want)
        memset(dst + filled, 0, want - filled);

    if (call_nr < 200)
        LOG_V2("dl_read[%d] #%d nsamps=%d ts=%lu dsize=%d skip=%zu actual=%zu full=%zu take=%zu "
               "leftover_rem=%d filled=%zu want=%zu",
               ctx->ue_idx, call_nr, nsamps, (unsigned long)slot_ts, avail,
               skip, actual_bytes, slot_bytes, take,
               ctx->dl_leftover_bytes, filled, want);

    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════════
 * UE UL write: enqueue to ul_tx[ue_idx] ring (non-blocking skip when full)
 * ══════════════════════════════════════════════════════════════════════ */

int gpu_ipc_v2_ul_write(gpu_ipc_v2_ctx_t *ctx, const void *samples,
                        int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V2_ROLE_UE) return -1;

    size_t data_size = (size_t)nsamps * nbAnt * 4;
    if (data_size > ctx->buf_size) {
        LOG_V2("ul_write[%d]: data_size %zu > slot_size %zu",
               ctx->ue_idx, data_size, ctx->buf_size);
        return -1;
    }

    int roff = _V2_OFF_UL_TX_RINGS + ctx->ue_idx * _V2_RING_CTRL_SIZE;

    uint32_t head = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD),
                                    __ATOMIC_RELAXED);
    uint32_t tail = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL),
                                    __ATOMIC_ACQUIRE);

    if ((head - tail) >= (uint32_t)ctx->ring_depth)
        return nsamps;   /* ring full — non-blocking skip */

    uint32_t slot = head & GPU_IPC_V2_RING_MASK;
    void *dst = (char *)ctx->gpu_ul_tx + (size_t)slot * ctx->buf_size;

    cudaError_t err = p_cudaMemcpy(dst, samples, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_V2("ul_write[%d] H2D failed: %s", ctx->ue_idx, p_cudaGetErrorString(err));
        return -1;
    }

    int meta = ring_slot_meta_off(roff, slot);
    *shm_u64(ctx->shm_raw, meta + _V2_SLOT_OFF_TS)     = timestamp;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_NSAMPS)  = nsamps;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_NBANT)   = nbAnt;
    *shm_i32(ctx->shm_raw, meta + _V2_SLOT_OFF_DSIZE)   = (int32_t)data_size;

    __atomic_store_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD),
                     head + 1, __ATOMIC_RELEASE);
    return nsamps;
}

/* ══════════════════════════════════════════════════════════════════════
 * gNB UL read: dequeue from ul_rx ring (non-blocking, returns 0 if empty)
 * ══════════════════════════════════════════════════════════════════════ */

int gpu_ipc_v2_ul_read(gpu_ipc_v2_ctx_t *ctx, void *samples,
                       int nsamps, int nbAnt, uint64_t *timestamp)
{
    if (!ctx->initialized || ctx->role != GPU_IPC_V2_ROLE_GNB) return -1;

    const int roff = _V2_OFF_UL_RX_RING;

    uint32_t tail = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL),
                                    __ATOMIC_RELAXED);
    uint32_t head = __atomic_load_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_HEAD),
                                    __ATOMIC_ACQUIRE);

    if (head == tail)
        return 0;   /* ring empty */

    uint32_t slot = tail & GPU_IPC_V2_RING_MASK;
    size_t want = (size_t)nsamps * nbAnt * 4;
    void *src = (char *)ctx->gpu_ul_rx + (size_t)slot * ctx->buf_size;

    int meta = ring_slot_meta_off(roff, slot);
    uint64_t slot_ts = *shm_u64(ctx->shm_raw, meta + _V2_SLOT_OFF_TS);

    cudaError_t err = p_cudaMemcpy(samples, src, want, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_V2("ul_read D2H failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    if (timestamp)
        *timestamp = slot_ts;

    __atomic_store_n(shm_u32(ctx->shm_raw, roff + _V2_RING_OFF_TAIL),
                     tail + 1, __ATOMIC_RELEASE);
    return nsamps;
}
