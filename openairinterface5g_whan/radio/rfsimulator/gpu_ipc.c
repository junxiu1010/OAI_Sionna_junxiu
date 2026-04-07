/*
 * gpu_ipc.c - CUDA IPC implementation for rfsimulator (dlopen approach)
 *
 * All CUDA functions are loaded at runtime via dlopen("libcudart.so").
 * No build-time CUDA dependency — OAI builds normally without CUDA toolkit.
 * If libcudart.so is not found at runtime, gpu_ipc_init() returns -1
 * and the caller falls back to socket mode.
 */

#include "gpu_ipc.h"

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

/* ── CUDA type definitions (no CUDA headers needed) ── */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaIpcMemLazyEnablePeerAccess 1

/*
 * cudaIpcMemHandle_t must be a STRUCT, not a char array typedef.
 * In C, array types decay to pointers when passed as function arguments,
 * but the real CUDA API passes this 64-byte struct BY VALUE on the stack.
 * Using a struct preserves correct ABI calling convention.
 */
typedef struct {
    char reserved[GPU_IPC_HANDLE_SIZE];
} cudaIpcMemHandle_t;

/* ── CUDA function pointer types ── */

typedef cudaError_t (*fn_cudaMalloc)(void **devPtr, size_t size);
typedef cudaError_t (*fn_cudaFree)(void *devPtr);
typedef cudaError_t (*fn_cudaMemcpy)(void *dst, const void *src,
                                     size_t count, int kind);
typedef cudaError_t (*fn_cudaIpcGetMemHandle)(cudaIpcMemHandle_t *handle,
                                              void *devPtr);
typedef cudaError_t (*fn_cudaIpcOpenMemHandle)(void **devPtr,
                                               cudaIpcMemHandle_t handle,
                                               unsigned int flags);
typedef cudaError_t (*fn_cudaIpcCloseMemHandle)(void *devPtr);
typedef cudaError_t (*fn_cudaSetDevice)(int device);
typedef const char* (*fn_cudaGetErrorString)(cudaError_t error);

/* ── CUDA function pointers (module-local) ── */

static fn_cudaMalloc           p_cudaMalloc;
static fn_cudaFree             p_cudaFree;
static fn_cudaMemcpy           p_cudaMemcpy;
static fn_cudaIpcGetMemHandle  p_cudaIpcGetMemHandle;
static fn_cudaIpcOpenMemHandle p_cudaIpcOpenMemHandle;
static fn_cudaIpcCloseMemHandle p_cudaIpcCloseMemHandle;
static fn_cudaSetDevice        p_cudaSetDevice;
static fn_cudaGetErrorString   p_cudaGetErrorString;

#define LOG_GPU(fmt, ...) fprintf(stderr, "[GPU_IPC] " fmt "\n", ##__VA_ARGS__)

/* ── Load CUDA runtime via dlopen ── */

static int load_cuda_runtime(gpu_ipc_ctx_t *ctx)
{
    const char *libs[] = {
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11",
        NULL
    };

    for (int i = 0; libs[i]; i++) {
        ctx->cuda_lib = dlopen(libs[i], RTLD_LAZY);
        if (ctx->cuda_lib) {
            LOG_GPU("Loaded %s", libs[i]);
            break;
        }
    }
    if (!ctx->cuda_lib) {
        LOG_GPU("Failed to load libcudart.so: %s", dlerror());
        return -1;
    }

#define LOAD_SYM(name) do { \
    p_##name = (fn_##name)dlsym(ctx->cuda_lib, #name); \
    if (!p_##name) { \
        LOG_GPU("dlsym(%s) failed: %s", #name, dlerror()); \
        dlclose(ctx->cuda_lib); ctx->cuda_lib = NULL; \
        return -1; \
    } \
} while(0)

    LOAD_SYM(cudaMalloc);
    LOAD_SYM(cudaFree);
    LOAD_SYM(cudaMemcpy);
    LOAD_SYM(cudaIpcGetMemHandle);
    LOAD_SYM(cudaIpcOpenMemHandle);
    LOAD_SYM(cudaIpcCloseMemHandle);
    LOAD_SYM(cudaSetDevice);
    LOAD_SYM(cudaGetErrorString);

#undef LOAD_SYM

    cudaError_t err = p_cudaSetDevice(0);
    if (err != cudaSuccess) {
        LOG_GPU("cudaSetDevice(0) failed: %s", p_cudaGetErrorString(err));
        dlclose(ctx->cuda_lib);
        ctx->cuda_lib = NULL;
        return -1;
    }

    return 0;
}

/* ── Shared memory file helpers ── */

static int create_shm_dir(void)
{
    struct stat st;
    if (stat(GPU_IPC_SHM_DIR, &st) == 0)
        return 0;
    if (mkdir(GPU_IPC_SHM_DIR, 0777) != 0 && errno != EEXIST) {
        LOG_GPU("mkdir(%s) failed: %s", GPU_IPC_SHM_DIR, strerror(errno));
        return -1;
    }
    chmod(GPU_IPC_SHM_DIR, 0777);
    return 0;
}

static gpu_ipc_shm_t *map_shm(int fd)
{
    if (ftruncate(fd, GPU_IPC_SHM_SIZE) != 0) {
        LOG_GPU("ftruncate failed: %s", strerror(errno));
        return NULL;
    }
    void *ptr = mmap(NULL, GPU_IPC_SHM_SIZE,
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        LOG_GPU("mmap failed: %s", strerror(errno));
        return NULL;
    }
    return (gpu_ipc_shm_t *)ptr;
}

/* ── CLIENT init: wait for Proxy (SERVER), open IPC handles ── */

static int client_init(gpu_ipc_ctx_t *ctx)
{
    cudaError_t err;
    cudaIpcMemHandle_t handle;

    /* Wait for server to be ready */
    LOG_GPU("CLIENT: waiting for server (magic=0x%08X)...", GPU_IPC_MAGIC);
    int wait_count = 0;
    while (__sync_val_compare_and_swap(&ctx->shm->magic, GPU_IPC_MAGIC, GPU_IPC_MAGIC)
           != GPU_IPC_MAGIC) {
        usleep(10000); /* 10ms */
        if (++wait_count > 3000) { /* 30 seconds timeout */
            LOG_GPU("CLIENT: timeout waiting for server");
            return -1;
        }
    }
    LOG_GPU("CLIENT: server ready (waited %d ms)", wait_count * 10);

    if (ctx->shm->version != GPU_IPC_VERSION) {
        LOG_GPU("CLIENT: version mismatch (got %u, expected %u)",
                ctx->shm->version, GPU_IPC_VERSION);
        return -1;
    }

    ctx->buf_size = ctx->shm->buf_size;

    /* Open IPC handles */
    char *handle_src[] = {
        ctx->shm->dl_tx_handle, ctx->shm->dl_rx_handle,
        ctx->shm->ul_tx_handle, ctx->shm->ul_rx_handle
    };
    void **bufs[] = { &ctx->gpu_dl_tx, &ctx->gpu_dl_rx,
                      &ctx->gpu_ul_tx, &ctx->gpu_ul_rx };
    char *names[] = { "dl_tx", "dl_rx", "ul_tx", "ul_rx" };

    for (int i = 0; i < 4; i++) {
        memcpy(handle.reserved, handle_src[i], GPU_IPC_HANDLE_SIZE);
        err = p_cudaIpcOpenMemHandle(bufs[i], handle,
                                     cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            LOG_GPU("cudaIpcOpenMemHandle(%s) failed: %s",
                    names[i], p_cudaGetErrorString(err));
            return -1;
        }
        LOG_GPU("CLIENT: opened %s (ptr=%p)", names[i], *bufs[i]);
    }

    LOG_GPU("CLIENT: initialization complete");
    return 0;
}

/* ── Public API ── */

int gpu_ipc_init(gpu_ipc_ctx_t *ctx, gpu_ipc_role_t role, size_t buf_size)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->role = role;
    ctx->buf_size = buf_size;
    ctx->shm_fd = -1;

    if (load_cuda_runtime(ctx) != 0)
        return -1;

    if (create_shm_dir() != 0)
        return -1;

    /* All OAI processes are CLIENTs — Proxy (Python) is the SERVER that
     * allocates GPU memory and creates the shm file.  Wait for it. */
    int wait_count = 0;
    while (access(GPU_IPC_SHM_PATH, F_OK) != 0) {
        usleep(10000);
        if (++wait_count > 3000) {
            LOG_GPU("CLIENT: timeout waiting for shm file (30s)");
            return -1;
        }
    }

    ctx->shm_fd = open(GPU_IPC_SHM_PATH, O_RDWR, 0666);
    if (ctx->shm_fd < 0) {
        LOG_GPU("open(%s) failed: %s", GPU_IPC_SHM_PATH, strerror(errno));
        return -1;
    }

    ctx->shm = map_shm(ctx->shm_fd);
    if (!ctx->shm) {
        close(ctx->shm_fd);
        return -1;
    }

    int ret = client_init(ctx);
    if (ret == 0)
        ctx->initialized = 1;
    return ret;
}

void gpu_ipc_cleanup(gpu_ipc_ctx_t *ctx)
{
    if (!ctx->initialized)
        return;

    /* OAI side is always a CLIENT — close IPC handles, don't free */
    if (ctx->gpu_dl_tx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_tx);
    if (ctx->gpu_dl_rx) p_cudaIpcCloseMemHandle(ctx->gpu_dl_rx);
    if (ctx->gpu_ul_tx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_tx);
    if (ctx->gpu_ul_rx) p_cudaIpcCloseMemHandle(ctx->gpu_ul_rx);

    if (ctx->shm)
        munmap(ctx->shm, GPU_IPC_SHM_SIZE);
    if (ctx->shm_fd >= 0)
        close(ctx->shm_fd);
    if (ctx->cuda_lib)
        dlclose(ctx->cuda_lib);

    ctx->initialized = 0;
    LOG_GPU("CLIENT(%s): cleanup done",
            ctx->role == GPU_IPC_ROLE_SERVER ? "gNB" : "UE");
}

/* ── DL path ── */

int gpu_ipc_dl_write(gpu_ipc_ctx_t *ctx, const void *samples,
                     int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized) return -1;

    size_t data_size = (size_t)nsamps * nbAnt * 4; /* sizeof(c16_t) = 4 */
    if (data_size > ctx->buf_size) {
        LOG_GPU("dl_write: data_size %zu > buf_size %zu", data_size, ctx->buf_size);
        return -1;
    }

    /* Block until Proxy consumed previous DL data, then write new data.
     * dl_write_count is incremented so the RU thread can pace itself
     * to L1-TX speed, preventing PRACH fill/free desynchronization. */
    while (__atomic_load_n(&ctx->shm->dl_tx_ready, __ATOMIC_ACQUIRE) != 0)
        sched_yield();

    cudaError_t err = p_cudaMemcpy(ctx->gpu_dl_tx, samples, data_size,
                                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_GPU("dl_write H2D failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    ctx->shm->dl_timestamp = timestamp;
    ctx->shm->dl_nsamps = nsamps;
    ctx->shm->dl_nbAnt = nbAnt;
    ctx->shm->dl_data_size = (int32_t)data_size;

    __atomic_store_n(&ctx->shm->dl_tx_ready, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&ctx->dl_write_count, 1, __ATOMIC_RELEASE);
    return nsamps;
}

int gpu_ipc_dl_read(gpu_ipc_ctx_t *ctx, void *samples,
                    int nsamps, int nbAnt, uint64_t *timestamp)
{
    if (!ctx->initialized) return -1;

    while (__atomic_load_n(&ctx->shm->dl_rx_ready, __ATOMIC_ACQUIRE) == 0)
        sched_yield();

    size_t avail = (size_t)ctx->shm->dl_data_size;
    size_t want  = (size_t)nsamps * nbAnt * 4;
    size_t copy  = avail < want ? avail : want;

    if (timestamp)
        *timestamp = ctx->shm->dl_timestamp;

    if (copy < want)
        memset(samples, 0, want);

    cudaError_t err = p_cudaMemcpy(samples, ctx->gpu_dl_rx, copy,
                                   cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_GPU("dl_read D2H failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    __atomic_store_n(&ctx->shm->dl_rx_ready, 0, __ATOMIC_RELEASE);
    return nsamps;
}

/* ── UL path ── */

int gpu_ipc_ul_write(gpu_ipc_ctx_t *ctx, const void *samples,
                     int nsamps, int nbAnt, uint64_t timestamp)
{
    if (!ctx->initialized) return -1;

    size_t data_size = (size_t)nsamps * nbAnt * 4;
    if (data_size > ctx->buf_size) {
        LOG_GPU("ul_write: data_size %zu > buf_size %zu", data_size, ctx->buf_size);
        return -1;
    }

    if (__atomic_load_n(&ctx->shm->ul_tx_ready, __ATOMIC_ACQUIRE) != 0)
        return nsamps;

    cudaError_t err = p_cudaMemcpy(ctx->gpu_ul_tx, samples, data_size,
                                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_GPU("ul_write H2D failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    ctx->shm->ul_timestamp = timestamp;
    ctx->shm->ul_nsamps = nsamps;
    ctx->shm->ul_nbAnt = nbAnt;
    ctx->shm->ul_data_size = (int32_t)data_size;

    __atomic_store_n(&ctx->shm->ul_tx_ready, 1, __ATOMIC_RELEASE);
    return nsamps;
}

int gpu_ipc_ul_read(gpu_ipc_ctx_t *ctx, void *samples,
                    int nsamps, int nbAnt, uint64_t *timestamp)
{
    if (!ctx->initialized) return -1;

    while (__atomic_load_n(&ctx->shm->ul_rx_ready, __ATOMIC_ACQUIRE) == 0)
        sched_yield();

    size_t avail = (size_t)ctx->shm->ul_data_size;
    size_t want  = (size_t)nsamps * nbAnt * 4;
    size_t copy  = avail < want ? avail : want;

    if (timestamp)
        *timestamp = ctx->shm->ul_timestamp;

    if (copy < want)
        memset(samples, 0, want);

    cudaError_t err = p_cudaMemcpy(samples, ctx->gpu_ul_rx, copy,
                                   cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_GPU("ul_read D2H failed: %s", p_cudaGetErrorString(err));
        return -1;
    }

    __atomic_store_n(&ctx->shm->ul_rx_ready, 0, __ATOMIC_RELEASE);
    return nsamps;
}
