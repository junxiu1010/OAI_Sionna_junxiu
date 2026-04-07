/*
 * gpu_ipc_v4.h - CUDA IPC Circular Buffers (G1A)
 *
 * Timestamp-indexed circular buffer matching socket-mode semantics.
 * Position: offset = (timestamp * nbAnt) % CirSize (sample index)
 * Supports variable nsamps natively (no packetizer).
 *
 * Architecture:
 *   Proxy (SERVER) allocates (2 + 2*N) GPU circular buffers:
 *     dl_tx    : 1 buf  (gNB writes, Proxy reads)
 *     ul_rx    : 1 buf  (Proxy writes, gNB reads)
 *     dl_rx[k] : N buf  (Proxy writes, UE_k reads)
 *     ul_tx[k] : N buf  (UE_k writes, Proxy reads)
 *
 *   Each buffer = CirSize * sizeof(c16_t) bytes (contiguous samples).
 *   Access: gpu_base + (offset % CirSize) * 4
 *
 *   Per-write metadata ring (2 + 2*N = 18 max):
 *     dl_tx_meta    : gNB pushes {ts,nsamps,nbAnt}, Proxy pops
 *     ul_rx_meta    : Proxy pushes, gNB pops
 *     dl_rx_meta[k] : Proxy pushes, UE_k pops
 *     ul_tx_meta[k] : UE_k pushes, Proxy pops
 *
 * SHM layout: 12288 bytes (12KB)
 */

#ifndef GPU_IPC_V4_H
#define GPU_IPC_V4_H

#include <stdint.h>
#include <stddef.h>

#define GPU_IPC_V4_SHM_DIR          "/tmp/oai_gpu_ipc"
#define GPU_IPC_V4_SHM_PATH         "/tmp/oai_gpu_ipc/gpu_ipc_shm"
#define GPU_IPC_V4_MAGIC            0x47505535  /* "GPU5" */
#define GPU_IPC_V4_VERSION          6           /* bumped: meta ring added */
#define GPU_IPC_V4_HANDLE_SIZE      64
#define GPU_IPC_V4_MAX_UE           8
#define GPU_IPC_V4_SHM_SIZE         12288

/* CirSize = 10ms @ 40MHz NR (matches simulator.c minCirSize) */
#define GPU_IPC_V4_CIR_SIZE         460800
#define GPU_IPC_V4_SAMPLE_SIZE      4   /* sizeof(c16_t) */
#define GPU_IPC_V4_BUF_BYTES        (GPU_IPC_V4_CIR_SIZE * GPU_IPC_V4_SAMPLE_SIZE)

/* Read return codes */
#define GPU_IPC_V4_OK               1
#define GPU_IPC_V4_EMPTY            0
#define GPU_IPC_V4_OVERWRITTEN      (-2)
#define GPU_IPC_V4_ERROR            (-1)

/* ── Meta Ring constants ── */
#define GPU_IPC_V4_META_RING_DEPTH  32
#define GPU_IPC_V4_META_RING_MASK   (GPU_IPC_V4_META_RING_DEPTH - 1)
#define GPU_IPC_V4_META_ENTRY_SIZE  16  /* {ts(8) + nsamps(4) + nbAnt(4)} */
#define GPU_IPC_V4_META_CTRL_SIZE   8   /* {write_idx(4) + read_idx(4)} */
#define GPU_IPC_V4_META_RING_SIZE   (GPU_IPC_V4_META_CTRL_SIZE + GPU_IPC_V4_META_RING_DEPTH * GPU_IPC_V4_META_ENTRY_SIZE)  /* 520 */

/*
 * SHM layout (12288 bytes):
 *
 * === IPC Handle area (0-1151) ===
 *   0      : dl_tx handle (64B)
 *   64     : ul_rx handle (64B)
 *   128    : dl_rx handles[MAX_UE] (64*8 = 512B)
 *   640    : ul_tx handles[MAX_UE] (64*8 = 512B)
 *
 * === Global area (1152-1535) ===
 *   1152   : cir_size      (uint32)
 *   1156   : chan_offset    (uint32)
 *   1160   : num_ues        (uint32)
 *   1164   : magic          (uint32)
 *   1168   : version        (uint32)
 *   1172   : last_dl_tx_ts  (uint64)  [legacy, kept for compat]
 *   1180   : last_dl_tx_nsamps (uint32) [legacy]
 *   1184   : last_dl_tx_nbAnt  (uint32) [legacy]
 *   1188   : last_ul_rx_ts  (uint64)
 *   1196   : last_dl_rx_ts[8] (uint64*8)
 *   1260   : last_ul_tx_ts[8] (uint64*8)
 *   1324   : dl_rx_consumer_ts[8] (uint64*8)
 *   1388   : ul_rx_consumer_ts (uint64)
 *   1396-1535: reserved
 *
 * === Meta Ring area (1536-11255) ===
 *   Per-ring: 520 bytes = ctrl(8B) + 32 entries * 16B
 *     ctrl: write_idx(uint32) + read_idx(uint32)
 *     entry: timestamp(uint64) + nsamps(uint32) + nbAnt(uint32)
 *
 *   1536 + 0*520   : dl_tx_meta         (gNB → Proxy)
 *   1536 + 1*520   : ul_rx_meta         (Proxy → gNB)
 *   1536 + 2*520   : dl_rx_meta[0]      (Proxy → UE_0)
 *   1536 + 3*520   : dl_rx_meta[1]      (Proxy → UE_1)
 *   ...
 *   1536 + 9*520   : dl_rx_meta[7]      (Proxy → UE_7)
 *   1536 + 10*520  : ul_tx_meta[0]      (UE_0 → Proxy)
 *   1536 + 11*520  : ul_tx_meta[1]      (UE_1 → Proxy)
 *   ...
 *   1536 + 17*520  : ul_tx_meta[7]      (UE_7 → Proxy)
 */

/* ── IPC Handle offsets ── */
#define _V4_OFF_DL_TX_HANDLE     0
#define _V4_OFF_UL_RX_HANDLE     64
#define _V4_OFF_DL_RX_HANDLES    128   /* + k*64 */
#define _V4_OFF_UL_TX_HANDLES    640   /* + k*64 */

/* ── Global area offsets ── */
#define _V4_OFF_GLOBAL           1152
#define _V4_OFF_CIR_SIZE         (_V4_OFF_GLOBAL)
#define _V4_OFF_CHAN_OFFSET      (_V4_OFF_GLOBAL + 4)
#define _V4_OFF_NUM_UES          (_V4_OFF_GLOBAL + 8)
#define _V4_OFF_MAGIC            (_V4_OFF_GLOBAL + 12)
#define _V4_OFF_VERSION          (_V4_OFF_GLOBAL + 16)
#define _V4_OFF_LAST_DL_TX_TS    (_V4_OFF_GLOBAL + 20)
#define _V4_OFF_LAST_DL_TX_NSAMPS (_V4_OFF_GLOBAL + 28)
#define _V4_OFF_LAST_DL_TX_NBANT (_V4_OFF_GLOBAL + 32)
#define _V4_OFF_LAST_UL_RX_TS    (_V4_OFF_GLOBAL + 36)
#define _V4_OFF_LAST_DL_RX_TS    (_V4_OFF_GLOBAL + 44)   /* + k*8 */
#define _V4_OFF_LAST_UL_TX_TS    (_V4_OFF_GLOBAL + 108)  /* + k*8 */
#define _V4_OFF_DL_RX_CONSUMER_TS (_V4_OFF_GLOBAL + 172)  /* + k*8 */
#define _V4_OFF_UL_RX_CONSUMER_TS (_V4_OFF_GLOBAL + 236)

/* ── Meta Ring offsets ── */
#define _V4_OFF_META_RINGS       1536
#define _V4_OFF_DL_TX_META       (_V4_OFF_META_RINGS + 0 * GPU_IPC_V4_META_RING_SIZE)
#define _V4_OFF_UL_RX_META       (_V4_OFF_META_RINGS + 1 * GPU_IPC_V4_META_RING_SIZE)
#define _V4_OFF_DL_RX_META(k)    (_V4_OFF_META_RINGS + (2 + (k)) * GPU_IPC_V4_META_RING_SIZE)
#define _V4_OFF_UL_TX_META(k)    (_V4_OFF_META_RINGS + (10 + (k)) * GPU_IPC_V4_META_RING_SIZE)

/* ── Meta Ring entry layout within a ring ── */
#define _V4_META_OFF_WRITE_IDX   0
#define _V4_META_OFF_READ_IDX    4
#define _V4_META_OFF_ENTRIES     8   /* + idx * 16 */
#define _V4_META_ENTRY_OFF_TS    0
#define _V4_META_ENTRY_OFF_NSAMPS 8
#define _V4_META_ENTRY_OFF_NBANT 12

typedef enum {
    GPU_IPC_V4_ROLE_GNB = 0,
    GPU_IPC_V4_ROLE_UE  = 1
} gpu_ipc_v4_role_t;

typedef struct {
    char    *shm_raw;
    void    *gpu_dl_tx;
    void    *gpu_ul_rx;
    void    *gpu_dl_rx;
    void    *gpu_ul_tx;
    gpu_ipc_v4_role_t role;
    int     ue_idx;
    int     num_ues;
    int     shm_fd;
    void    *cuda_lib;
    uint32_t cir_size;
    uint32_t chan_offset;
    int     initialized;
    int     ul_ts_synced;
    uint64_t lastReceivedTS;  /* for gap zero-fill, like socket mode */
} gpu_ipc_v4_ctx_t;

/* ── Initialization / Cleanup ── */

int  gpu_ipc_v4_init(gpu_ipc_v4_ctx_t *ctx, gpu_ipc_v4_role_t role, int ue_idx);
void gpu_ipc_v4_cleanup(gpu_ipc_v4_ctx_t *ctx);

/* ── gNB DL write (auto-pushes dl_tx_meta) ── */
int  gpu_ipc_v4_dl_write(gpu_ipc_v4_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── UE DL read ── */
int  gpu_ipc_v4_dl_read(gpu_ipc_v4_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

/* ── UE UL write (auto-pushes ul_tx_meta[ue_idx]) ── */
int  gpu_ipc_v4_ul_write(gpu_ipc_v4_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* ── gNB UL read ── */
int  gpu_ipc_v4_ul_read(gpu_ipc_v4_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t target_ts);

/* ── Convenience: latest ts for sync ── */
uint64_t gpu_ipc_v4_dl_latest_ts(gpu_ipc_v4_ctx_t *ctx);
uint64_t gpu_ipc_v4_ul_latest_ts(gpu_ipc_v4_ctx_t *ctx);
uint64_t gpu_ipc_v4_dl_consumer_ts(gpu_ipc_v4_ctx_t *ctx);
uint64_t gpu_ipc_v4_ul_consumer_ts(gpu_ipc_v4_ctx_t *ctx);
void gpu_ipc_v4_set_dl_consumer_ts(gpu_ipc_v4_ctx_t *ctx, uint64_t ts);
void gpu_ipc_v4_set_ul_consumer_ts(gpu_ipc_v4_ctx_t *ctx, uint64_t ts);

/* ── Meta Ring API ── */
int  v4_meta_push(char *shm, int ring_off, uint64_t ts, uint32_t nsamps, uint32_t nbAnt);
int  v4_meta_pop(char *shm, int ring_off, uint64_t *ts, uint32_t *nsamps, uint32_t *nbAnt);
int  v4_meta_available(char *shm, int ring_off);

/* ── Meta Ring drain: pop all available entries, apply gap zero-fill to GPU circular buffer ── */
int  v4_meta_drain_and_zerofill(gpu_ipc_v4_ctx_t *ctx, void *gpu_base, int meta_ring_off);

#endif /* GPU_IPC_V4_H */
