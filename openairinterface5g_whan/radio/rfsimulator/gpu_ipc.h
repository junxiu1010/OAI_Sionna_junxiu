/*
 * gpu_ipc.h - CUDA IPC shared memory interface for rfsimulator
 *
 * Replaces TCP socket communication between OAI gNB/UE and Sionna Proxy
 * with GPU shared memory via CUDA IPC. Both DL and UL paths use GPU buffers.
 *
 * Architecture:
 *   Proxy (SERVER) allocates 4 GPU buffers + creates IPC handles in shared file
 *   gNB  (CLIENT) opens all 4 handles via cudaIpcOpenMemHandle()
 *   UE   (CLIENT) opens all 4 handles via cudaIpcOpenMemHandle()
 *
 *   The Proxy owns GPU memory so the RAN side (OAI / Aerial) can be swapped
 *   without changing the memory allocator.
 *
 * Data flow:
 *   DL: gNB H2D -> dl_tx_buf -> Proxy GPU process -> dl_rx_buf -> UE D2H
 *   UL: UE  H2D -> ul_tx_buf -> Proxy GPU copy    -> ul_rx_buf -> gNB D2H
 *
 * Sync: mmap'd shared file with volatile flags + memory fences
 * CUDA: loaded at runtime via dlopen("libcudart.so") - no build-time dependency
 */

#ifndef GPU_IPC_H
#define GPU_IPC_H

#include <stdint.h>
#include <stddef.h>

#define GPU_IPC_SHM_DIR      "/tmp/oai_gpu_ipc"
#define GPU_IPC_SHM_PATH     "/tmp/oai_gpu_ipc/gpu_ipc_shm"
#define GPU_IPC_MAGIC         0x47505531  /* "GPU1" */
#define GPU_IPC_VERSION       1
#define GPU_IPC_HANDLE_SIZE   64          /* sizeof(cudaIpcMemHandle_t) */
#define GPU_IPC_SHM_SIZE      512
#define GPU_IPC_MAX_DATA_SIZE (61440 * 4) /* 240KB: 1 NR subframe SISO 61.44MHz (covers any per-call write) */

/* Shared memory layout (mmap'd file, 512 bytes)
 *
 * Offset   Size   Field
 * ------   ----   -----
 * 0        64     dl_tx IPC handle
 * 64       64     dl_rx IPC handle
 * 128      64     ul_tx IPC handle
 * 192      64     ul_rx IPC handle
 * 256      4      dl_tx_ready  (volatile int32)
 * 260      4      dl_rx_ready  (volatile int32)
 * 264      4      ul_tx_ready  (volatile int32)
 * 268      4      ul_rx_ready  (volatile int32)
 * 272      8      dl_timestamp (uint64)
 * 280      4      dl_nsamps    (int32)
 * 284      4      dl_nbAnt     (int32)
 * 288      4      dl_data_size (int32)
 * 292      4      reserved
 * 296      8      ul_timestamp (uint64)
 * 304      4      ul_nsamps    (int32)
 * 308      4      ul_nbAnt     (int32)
 * 312      4      ul_data_size (int32)
 * 316      4      reserved
 * 320      4      magic        (uint32, GPU_IPC_MAGIC when ready)
 * 324      4      version      (uint32, GPU_IPC_VERSION)
 * 328      4      buf_size     (uint32, allocated buffer size in bytes)
 * 332-511         reserved
 */
typedef struct __attribute__((packed)) {
    char     dl_tx_handle[GPU_IPC_HANDLE_SIZE]; /* 0   */
    char     dl_rx_handle[GPU_IPC_HANDLE_SIZE]; /* 64  */
    char     ul_tx_handle[GPU_IPC_HANDLE_SIZE]; /* 128 */
    char     ul_rx_handle[GPU_IPC_HANDLE_SIZE]; /* 192 */

    volatile int32_t dl_tx_ready;               /* 256 */
    volatile int32_t dl_rx_ready;               /* 260 */
    volatile int32_t ul_tx_ready;               /* 264 */
    volatile int32_t ul_rx_ready;               /* 268 */

    uint64_t dl_timestamp;                      /* 272 */
    int32_t  dl_nsamps;                         /* 280 */
    int32_t  dl_nbAnt;                          /* 284 */
    int32_t  dl_data_size;                      /* 288 */
    int32_t  _reserved1;                        /* 292 */

    uint64_t ul_timestamp;                      /* 296 */
    int32_t  ul_nsamps;                         /* 304 */
    int32_t  ul_nbAnt;                          /* 308 */
    int32_t  ul_data_size;                      /* 312 */
    int32_t  _reserved2;                        /* 316 */

    uint32_t magic;                             /* 320 */
    uint32_t version;                           /* 324 */
    uint32_t buf_size;                          /* 328 */
    char     _pad[512 - 332];                   /* 332-511 */
} gpu_ipc_shm_t;

typedef enum {
    GPU_IPC_ROLE_SERVER = 0, /* gNB: allocates GPU memory, creates handles */
    GPU_IPC_ROLE_CLIENT = 1  /* UE:  opens existing handles */
} gpu_ipc_role_t;

typedef struct {
    gpu_ipc_shm_t *shm;       /* mmap'd shared memory */
    void *gpu_dl_tx;           /* GPU buffer: gNB DL write */
    void *gpu_dl_rx;           /* GPU buffer: UE DL read */
    void *gpu_ul_tx;           /* GPU buffer: UE UL write */
    void *gpu_ul_rx;           /* GPU buffer: gNB UL read */
    gpu_ipc_role_t role;
    int shm_fd;                /* file descriptor for mmap */
    void *cuda_lib;            /* dlopen handle for libcudart.so */
    size_t buf_size;           /* per-buffer allocation size */
    int initialized;           /* 1 if init succeeded */
    volatile int64_t dl_write_count; /* gNB: incremented after each DL write */
} gpu_ipc_ctx_t;

/*
 * Initialize GPU IPC context.
 * SERVER: cudaMalloc x4, export IPC handles, create shared file
 * CLIENT: wait for shared file, open IPC handles
 * Returns 0 on success, -1 on failure.
 */
int gpu_ipc_init(gpu_ipc_ctx_t *ctx, gpu_ipc_role_t role, size_t buf_size);

/*
 * Cleanup: free GPU memory (SERVER) or close handles (CLIENT), unmap shm.
 */
void gpu_ipc_cleanup(gpu_ipc_ctx_t *ctx);

/*
 * DL write: gNB copies samples to GPU (H2D), sets dl_tx_ready.
 * Blocks until previous dl_tx_ready is consumed (==0).
 */
int gpu_ipc_dl_write(gpu_ipc_ctx_t *ctx, const void *samples,
                     int nsamps, int nbAnt, uint64_t timestamp);

/*
 * DL read: UE waits for dl_rx_ready, copies from GPU (D2H).
 * Returns number of samples read, or -1 on error.
 */
int gpu_ipc_dl_read(gpu_ipc_ctx_t *ctx, void *samples,
                    int nsamps, int nbAnt, uint64_t *timestamp);

/*
 * UL write: UE copies samples to GPU (H2D), sets ul_tx_ready.
 */
int gpu_ipc_ul_write(gpu_ipc_ctx_t *ctx, const void *samples,
                     int nsamps, int nbAnt, uint64_t timestamp);

/*
 * UL read: gNB waits for ul_rx_ready, copies from GPU (D2H).
 */
int gpu_ipc_ul_read(gpu_ipc_ctx_t *ctx, void *samples,
                    int nsamps, int nbAnt, uint64_t *timestamp);

#endif /* GPU_IPC_H */
