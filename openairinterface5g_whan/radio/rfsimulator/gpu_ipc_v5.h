/*
 * gpu_ipc_v5.h - CUDA IPC Circular Buffer (G1B, Single-UE)
 *
 * Socket-mode-compatible circular buffer over GPU shared memory.
 * Timestamp-indexed position: offset = (ts * nbAnt) % cir_size
 * Supports variable nsamps natively (no packetizer needed).
 *
 * Architecture (single UE):
 *   Proxy (SERVER) allocates 4 GPU circular buffers + SHM:
 *     dl_tx : gNB writes  → Proxy reads
 *     dl_rx : Proxy writes → UE reads
 *     ul_tx : UE writes   → Proxy reads
 *     ul_rx : Proxy writes → gNB reads
 *
 *   Each buffer = CirSize * sizeof(c16_t) bytes (contiguous circular samples).
 *
 * Differences from V1 (gpu_ipc.h):
 *   - Circular buffer with timestamp indexing (no ready-flag ping-pong)
 *   - Gap zero-fill on writer side
 *   - Wait-with-timeout on reader side
 *   - Initial sync via first received timestamp
 *
 * CUDA: loaded at runtime via dlopen — no build-time CUDA dependency.
 */

#ifndef GPU_IPC_V5_H
#define GPU_IPC_V5_H

#include <stdint.h>
#include <stddef.h>

#define GPU_IPC_V5_SHM_DIR      "/tmp/oai_gpu_ipc"
#define GPU_IPC_V5_SHM_PATH     "/tmp/oai_gpu_ipc/gpu_ipc_shm"
#define GPU_IPC_V5_MAGIC        0x47505536  /* "GPU6" */
#define GPU_IPC_V5_VERSION      7
#define GPU_IPC_V5_HANDLE_SIZE  64
#define GPU_IPC_V5_SHM_SIZE     4096

/* 10 ms @ 30.72 MHz NR → matches socket mode circularBuf sizing */
#define GPU_IPC_V5_CIR_SIZE     460800
#define GPU_IPC_V5_SAMPLE_SIZE  4   /* sizeof(c16_t) = sizeof(int32_t) */
#define GPU_IPC_V5_BUF_BYTES    ((size_t)GPU_IPC_V5_CIR_SIZE * GPU_IPC_V5_SAMPLE_SIZE)

/* Read return codes */
#define GPU_IPC_V5_OK           1
#define GPU_IPC_V5_EMPTY        0
#define GPU_IPC_V5_ERROR        (-1)

/*
 * SHM Layout (4096 bytes):
 *
 * === IPC Handles (0-255) ===
 *   0    : dl_tx handle (64B)
 *   64   : dl_rx handle (64B)
 *   128  : ul_tx handle (64B)
 *   192  : ul_rx handle (64B)
 *
 * === Global metadata (256-271) ===
 *   256  : magic       (uint32)
 *   260  : version     (uint32)
 *   264  : cir_size    (uint32, circular buffer size in sample slots)
 *   268  : num_ues     (uint32, currently 1)
 *
 * === Producer timestamps (272-319) ===
 *   272  : last_dl_tx_ts     (uint64) — gNB's last DL write timestamp
 *   280  : last_dl_tx_nsamps (uint32)
 *   284  : last_dl_tx_nbAnt  (uint32)
 *   288  : last_dl_rx_ts     (uint64) — Proxy's last DL→UE write ts
 *   296  : last_ul_tx_ts     (uint64) — UE's last UL write timestamp
 *   304  : last_ul_tx_nsamps (uint32)
 *   308  : last_ul_tx_nbAnt  (uint32)
 *   312  : last_ul_rx_ts     (uint64) — Proxy's last UL→gNB write ts
 *
 * === Consumer timestamps (320-335) ===
 *   320  : dl_consumer_ts    (uint64) — UE's read progress
 *   328  : ul_consumer_ts    (uint64) — gNB's read progress
 *
 * 336-4095: reserved
 */

/* ── SHM offset macros ── */
#define _V5_OFF_DL_TX_HANDLE        0
#define _V5_OFF_DL_RX_HANDLE        64
#define _V5_OFF_UL_TX_HANDLE        128
#define _V5_OFF_UL_RX_HANDLE        192

#define _V5_OFF_MAGIC               256
#define _V5_OFF_VERSION             260
#define _V5_OFF_CIR_SIZE            264
#define _V5_OFF_NUM_UES             268

#define _V5_OFF_LAST_DL_TX_TS       272
#define _V5_OFF_LAST_DL_TX_NSAMPS   280
#define _V5_OFF_LAST_DL_TX_NBANT    284
#define _V5_OFF_LAST_DL_RX_TS       288
#define _V5_OFF_LAST_UL_TX_TS       296
#define _V5_OFF_LAST_UL_TX_NSAMPS   304
#define _V5_OFF_LAST_UL_TX_NBANT    308
#define _V5_OFF_LAST_UL_RX_TS       312

#define _V5_OFF_DL_CONSUMER_TS      320
#define _V5_OFF_UL_CONSUMER_TS      328

typedef enum {
    GPU_IPC_V5_ROLE_GNB = 0,
    GPU_IPC_V5_ROLE_UE  = 1
} gpu_ipc_v5_role_t;

typedef struct {
    char    *shm_raw;       /* mmap'd shared memory (raw bytes) */
    void    *gpu_dl_tx;     /* GPU buffer pointers */
    void    *gpu_dl_rx;
    void    *gpu_ul_tx;
    void    *gpu_ul_rx;
    gpu_ipc_v5_role_t role;
    int     shm_fd;
    void    *cuda_lib;      /* dlopen handle for libcudart.so */
    uint32_t cir_size;      /* circular buffer size in sample slots */
    int     initialized;
    uint64_t lastWriteEnd;  /* writer-side: ts+nsamps after last write (gap-fill) */
} gpu_ipc_v5_ctx_t;

/* ── Init / Cleanup ── */
int  gpu_ipc_v5_init(gpu_ipc_v5_ctx_t *ctx, gpu_ipc_v5_role_t role);
void gpu_ipc_v5_cleanup(gpu_ipc_v5_ctx_t *ctx);

/* ── gNB DL write: interleaved samples → H2D circular dl_tx ── */
int  gpu_ipc_v5_dl_write(gpu_ipc_v5_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── UE DL read: D2H from circular dl_rx at target_ts ── */
int  gpu_ipc_v5_dl_read(gpu_ipc_v5_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

/* ── UE UL write: interleaved samples → H2D circular ul_tx ── */
int  gpu_ipc_v5_ul_write(gpu_ipc_v5_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── gNB UL read: D2H from circular ul_rx at target_ts ── */
int  gpu_ipc_v5_ul_read(gpu_ipc_v5_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

#endif /* GPU_IPC_V5_H */
