/*
 * gpu_ipc_v6.h - CUDA IPC Circular Buffer with Per-Buffer Antenna Config
 *
 * Evolution of V5: each of the 4 buffers has its own nbAnt and cir_size,
 * enabling asymmetric MIMO (gNB and UE with different antenna counts).
 *
 * Architecture (single UE):
 *   Proxy (SERVER) allocates 4 GPU circular buffers + SHM:
 *     dl_tx : gNB writes  → Proxy reads   (nbAnt = GNB_ANT)
 *     dl_rx : Proxy writes → UE reads      (nbAnt = UE_ANT)
 *     ul_tx : UE writes   → Proxy reads    (nbAnt = UE_ANT)
 *     ul_rx : Proxy writes → gNB reads     (nbAnt = GNB_ANT)
 *
 *   Each buffer = cir_time * buffer_nbAnt * sizeof(c16_t) bytes.
 *   Offset for timestamp ts: (ts * buffer_nbAnt) % buffer_cir_size
 *
 * Differences from V5:
 *   - Per-buffer nbAnt and cir_size (asymmetric MIMO support)
 *   - 4 buffers can have different sizes
 *   - Proxy handles antenna mapping between mismatched counts
 *
 * Preserved from V5 (B/C sets):
 *   - Circular buffer with timestamp indexing
 *   - Gap zero-fill on writer side
 *   - Wait-with-timeout on reader side
 *   - Initial sync via first received timestamp
 *   - Range availability check (full nsamps range)
 *
 * CUDA: loaded at runtime via dlopen — no build-time CUDA dependency.
 */

#ifndef GPU_IPC_V6_H
#define GPU_IPC_V6_H

#include <stdint.h>
#include <stddef.h>

#define GPU_IPC_V6_SHM_DIR      "/tmp/oai_gpu_ipc"
#define GPU_IPC_V6_SHM_PATH     "/tmp/oai_gpu_ipc/gpu_ipc_shm"
#define GPU_IPC_V6_MAGIC        0x47505537  /* "GPU7" */
#define GPU_IPC_V6_VERSION      1
#define GPU_IPC_V6_HANDLE_SIZE  64
#define GPU_IPC_V6_SHM_SIZE     4096

#define GPU_IPC_V6_CIR_TIME     460800  /* base time slots: 10 ms @ 30.72 MHz */
#define GPU_IPC_V6_SAMPLE_SIZE  4       /* sizeof(c16_t) = sizeof(int32_t) */

#define GPU_IPC_V6_OK           1
#define GPU_IPC_V6_EMPTY        0
#define GPU_IPC_V6_ERROR        (-1)

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
 *   264  : cir_time    (uint32, base time slots — same for all buffers)
 *   268  : num_ues     (uint32, currently 1)
 *
 * === Per-buffer antenna config (272-303) ===
 *   272  : dl_tx_nbAnt    (uint32) — gNB TX antennas
 *   276  : dl_tx_cir_size (uint32) — cir_time * dl_tx_nbAnt
 *   280  : dl_rx_nbAnt    (uint32) — UE RX antennas
 *   284  : dl_rx_cir_size (uint32) — cir_time * dl_rx_nbAnt
 *   288  : ul_tx_nbAnt    (uint32) — UE TX antennas
 *   292  : ul_tx_cir_size (uint32) — cir_time * ul_tx_nbAnt
 *   296  : ul_rx_nbAnt    (uint32) — gNB RX antennas
 *   300  : ul_rx_cir_size (uint32) — cir_time * ul_rx_nbAnt
 *
 * === Producer timestamps (304-351) ===
 *   304  : last_dl_tx_ts     (uint64) — gNB's last DL write timestamp
 *   312  : last_dl_tx_nsamps (uint32)
 *   316  : (pad)             (uint32)
 *   320  : last_dl_rx_ts     (uint64) — Proxy's last DL→UE write ts
 *   328  : last_ul_tx_ts     (uint64) — UE's last UL write timestamp
 *   336  : last_ul_tx_nsamps (uint32)
 *   340  : (pad)             (uint32)
 *   344  : last_ul_rx_ts     (uint64) — Proxy's last UL→gNB write ts
 *
 * === Consumer timestamps (352-367) ===
 *   352  : dl_consumer_ts    (uint64) — UE's read progress
 *   360  : ul_consumer_ts    (uint64) — gNB's read progress
 *
 * 368-4095: reserved
 */

/* ── SHM offset macros ── */
#define _V6_OFF_DL_TX_HANDLE        0
#define _V6_OFF_DL_RX_HANDLE        64
#define _V6_OFF_UL_TX_HANDLE        128
#define _V6_OFF_UL_RX_HANDLE        192

#define _V6_OFF_MAGIC               256
#define _V6_OFF_VERSION             260
#define _V6_OFF_CIR_TIME            264
#define _V6_OFF_NUM_UES             268

#define _V6_OFF_DL_TX_NBANT         272
#define _V6_OFF_DL_TX_CIR_SIZE      276
#define _V6_OFF_DL_RX_NBANT         280
#define _V6_OFF_DL_RX_CIR_SIZE      284
#define _V6_OFF_UL_TX_NBANT         288
#define _V6_OFF_UL_TX_CIR_SIZE      292
#define _V6_OFF_UL_RX_NBANT         296
#define _V6_OFF_UL_RX_CIR_SIZE      300

#define _V6_OFF_LAST_DL_TX_TS       304
#define _V6_OFF_LAST_DL_TX_NSAMPS   312
#define _V6_OFF_LAST_DL_RX_TS       320
#define _V6_OFF_LAST_UL_TX_TS       328
#define _V6_OFF_LAST_UL_TX_NSAMPS   336
#define _V6_OFF_LAST_UL_RX_TS       344

#define _V6_OFF_DL_CONSUMER_TS      352
#define _V6_OFF_UL_CONSUMER_TS      360

typedef enum {
    GPU_IPC_V6_ROLE_GNB = 0,
    GPU_IPC_V6_ROLE_UE  = 1
} gpu_ipc_v6_role_t;

typedef struct {
    char    *shm_raw;
    void    *gpu_dl_tx;
    void    *gpu_dl_rx;
    void    *gpu_ul_tx;
    void    *gpu_ul_rx;
    gpu_ipc_v6_role_t role;
    int     shm_fd;
    void    *cuda_lib;
    /* per-buffer config (read from SHM at init) */
    uint32_t dl_tx_nbAnt;   uint32_t dl_tx_cir_size;
    uint32_t dl_rx_nbAnt;   uint32_t dl_rx_cir_size;
    uint32_t ul_tx_nbAnt;   uint32_t ul_tx_cir_size;
    uint32_t ul_rx_nbAnt;   uint32_t ul_rx_cir_size;
    int     initialized;
    uint64_t lastWriteEnd;
} gpu_ipc_v6_ctx_t;

/* ── Init / Cleanup ── */
int  gpu_ipc_v6_init(gpu_ipc_v6_ctx_t *ctx, gpu_ipc_v6_role_t role, int ue_idx);
void gpu_ipc_v6_cleanup(gpu_ipc_v6_ctx_t *ctx);

/* ── gNB DL write: interleaved samples → H2D circular dl_tx ── */
int  gpu_ipc_v6_dl_write(gpu_ipc_v6_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── UE DL read: D2H from circular dl_rx at target_ts ── */
int  gpu_ipc_v6_dl_read(gpu_ipc_v6_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

/* ── UE UL write: interleaved samples → H2D circular ul_tx ── */
int  gpu_ipc_v6_ul_write(gpu_ipc_v6_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── gNB UL read: D2H from circular ul_rx at target_ts ── */
int  gpu_ipc_v6_ul_read(gpu_ipc_v6_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

#endif /* GPU_IPC_V6_H */
