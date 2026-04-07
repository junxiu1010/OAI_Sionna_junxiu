/*
 * gpu_ipc_v2.h - Multi-UE CUDA IPC interface with SPSC ring buffers (G1A)
 *
 * Extends G0 gpu_ipc.h to support N UEs with per-UE GPU buffers.
 * All data paths use lock-free SPSC ring buffers (depth=16) to decouple
 * OAI timing from Proxy processing, matching socket-mode buffering behavior.
 *
 * Architecture:
 *   Proxy (SERVER) allocates (2 + 2*N) GPU ring buffers:
 *     dl_tx_ring    : 1 ring   (gNB writes broadcast DL, Proxy reads)
 *     ul_rx_ring    : 1 ring   (Proxy writes summed UL, gNB reads)
 *     dl_rx_ring[k] : N rings  (Proxy writes per-UE DL, UE_k reads)
 *     ul_tx_ring[k] : N rings  (UE_k writes UL, Proxy reads)
 *
 *   Each ring = RING_DEPTH contiguous GPU slots under a single IPC handle.
 *   Slot N accessed at: gpu_base + (idx & RING_MASK) * slot_size
 *
 *   gNB (CLIENT) uses dl_tx_ring + ul_rx_ring
 *   UE  (CLIENT) uses dl_rx_ring[ue_idx] + ul_tx_ring[ue_idx]
 *
 * SHM layout: 16384 bytes (see offset constants below)
 */

#ifndef GPU_IPC_V2_H
#define GPU_IPC_V2_H

#include <stdint.h>
#include <stddef.h>

#define GPU_IPC_V2_SHM_DIR      "/tmp/oai_gpu_ipc"
#define GPU_IPC_V2_SHM_PATH     "/tmp/oai_gpu_ipc/gpu_ipc_shm"
#define GPU_IPC_V2_MAGIC         0x47505533  /* "GPU3" */
#define GPU_IPC_V2_VERSION       3
#define GPU_IPC_V2_HANDLE_SIZE   64
#define GPU_IPC_V2_MAX_UE        8
#define GPU_IPC_V2_SHM_SIZE      16384
#define GPU_IPC_V2_MAX_DATA_SIZE (61440 * 4) /* 240KB per slot */

#define GPU_IPC_V2_RING_DEPTH    16          /* must be power of 2 */
#define GPU_IPC_V2_RING_MASK     (GPU_IPC_V2_RING_DEPTH - 1)

/*
 * SHM layout (16384 bytes):
 *
 * === IPC Handle area (0-1151) ===
 *   0      : dl_tx handle (64B)
 *   64     : ul_rx handle (64B)
 *   128    : dl_rx handles[MAX_UE] (64*8 = 512B)
 *   640    : ul_tx handles[MAX_UE] (64*8 = 512B)
 *
 * === Ring control area (1152-8207) ===
 *   Per-ring control block = 392 bytes:
 *     head(4) + tail(4) + slots[RING_DEPTH] * 24B
 *     Per-slot metadata (24B): ts(8) + nsamps(4) + nbAnt(4) + data_size(4) + pad(4)
 *
 *   1152 : dl_tx_ring     (392B)  — gNB writes, Proxy reads
 *   1544 : ul_rx_ring     (392B)  — Proxy writes, gNB reads
 *   1936 : dl_rx_ring[8] (3136B)  — Proxy writes, UE reads
 *   5072 : ul_tx_ring[8] (3136B)  — UE writes, Proxy reads
 *
 * === Global area (8208-8227) ===
 *   8208 : ring_depth (uint32)
 *   8212 : num_ues    (uint32)
 *   8216 : magic      (uint32)
 *   8220 : version    (uint32)
 *   8224 : buf_size   (uint32)  — per-SLOT size (not total ring allocation)
 */

/* IPC handle offsets (unchanged from V2) */
#define _V2_OFF_DL_TX_HANDLE     0
#define _V2_OFF_UL_RX_HANDLE     64
#define _V2_OFF_DL_RX_HANDLES    128   /* + k*64 */
#define _V2_OFF_UL_TX_HANDLES    640   /* + k*64 */

/* Ring control: per-slot metadata size and per-ring control block size */
#define _V2_RING_SLOT_META_SIZE  24
#define _V2_RING_CTRL_SIZE       (8 + GPU_IPC_V2_RING_DEPTH * _V2_RING_SLOT_META_SIZE)  /* 392 */

/* Ring control base offsets */
#define _V2_OFF_RING_BASE        1152
#define _V2_OFF_DL_TX_RING       (_V2_OFF_RING_BASE)
#define _V2_OFF_UL_RX_RING       (_V2_OFF_DL_TX_RING  + _V2_RING_CTRL_SIZE)
#define _V2_OFF_DL_RX_RINGS      (_V2_OFF_UL_RX_RING  + _V2_RING_CTRL_SIZE)
#define _V2_OFF_UL_TX_RINGS      (_V2_OFF_DL_RX_RINGS + GPU_IPC_V2_MAX_UE * _V2_RING_CTRL_SIZE)

/* Within a ring control block (relative offsets) */
#define _V2_RING_OFF_HEAD        0
#define _V2_RING_OFF_TAIL        4
#define _V2_RING_OFF_SLOTS       8     /* + slot_idx * _V2_RING_SLOT_META_SIZE */
#define _V2_SLOT_OFF_TS          0
#define _V2_SLOT_OFF_NSAMPS      8
#define _V2_SLOT_OFF_NBANT       12
#define _V2_SLOT_OFF_DSIZE       16

/* Global area */
#define _V2_OFF_GLOBAL           (_V2_OFF_UL_TX_RINGS + GPU_IPC_V2_MAX_UE * _V2_RING_CTRL_SIZE)
#define _V2_OFF_RING_DEPTH       (_V2_OFF_GLOBAL)
#define _V2_OFF_NUM_UES          (_V2_OFF_GLOBAL + 4)
#define _V2_OFF_MAGIC            (_V2_OFF_GLOBAL + 8)
#define _V2_OFF_VERSION          (_V2_OFF_GLOBAL + 12)
#define _V2_OFF_BUF_SIZE         (_V2_OFF_GLOBAL + 16)

typedef enum {
    GPU_IPC_V2_ROLE_GNB = 0, /* gNB: uses dl_tx_ring + ul_rx_ring */
    GPU_IPC_V2_ROLE_UE  = 1  /* UE: uses dl_rx_ring[idx] + ul_tx_ring[idx] */
} gpu_ipc_v2_role_t;

typedef struct {
    char    *shm_raw;          /* raw mmap pointer */
    void    *gpu_dl_tx;        /* gNB only: dl_tx ring base (RING_DEPTH * slot_size bytes) */
    void    *gpu_ul_rx;        /* gNB only: ul_rx ring base */
    void    *gpu_dl_rx;        /* UE only: dl_rx[ue_idx] ring base */
    void    *gpu_ul_tx;        /* UE only: ul_tx[ue_idx] ring base */
    gpu_ipc_v2_role_t role;
    int     ue_idx;            /* UE only: which UE index (0..N-1) */
    int     num_ues;           /* total number of UEs from SHM */
    int     shm_fd;
    void    *cuda_lib;
    size_t  buf_size;          /* per-slot size in bytes */
    int     ring_depth;        /* read from SHM, must match compiled RING_DEPTH */
    int     initialized;

    /* Leftover buffer for partial DL reads (UE only).
     * When caller requests fewer samples than a ring slot, remaining
     * samples are buffered here for the next read. Without this,
     * resynchronization (syncInFrame) discards the tail of a slot,
     * permanently misaligning all subsequent OFDM symbols.
     *
     * dl_slot_full_bytes tracks the full-slot stream size (bytes).
     * Mixed/short slots with fewer actual bytes are zero-padded in
     * dl_leftover to this size so the continuous stream has no gaps. */
    char    *dl_leftover;      /* host-side buffer, allocated = buf_size */
    int      dl_leftover_bytes; /* valid bytes remaining in dl_leftover */
    int      dl_leftover_off;  /* read offset into dl_leftover */
    uint64_t dl_leftover_ts;   /* timestamp of the leftover's original slot */
    size_t   dl_slot_full_bytes; /* full slot stream contribution in bytes */

    int      ul_ts_synced;     /* gNB only: set once UL timestamp is synced */
} gpu_ipc_v2_ctx_t;

int  gpu_ipc_v2_init(gpu_ipc_v2_ctx_t *ctx, gpu_ipc_v2_role_t role, int ue_idx);
void gpu_ipc_v2_cleanup(gpu_ipc_v2_ctx_t *ctx);

/* gNB DL write: enqueue to dl_tx ring. Blocks if ring full (sched_yield). */
int  gpu_ipc_v2_dl_write(gpu_ipc_v2_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* UE DL read: dequeue from dl_rx[ue_idx] ring. Blocks if ring empty. */
int  gpu_ipc_v2_dl_read(gpu_ipc_v2_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t *timestamp);

/* UE UL write: enqueue to ul_tx[ue_idx] ring. Non-blocking skip if full. */
int  gpu_ipc_v2_ul_write(gpu_ipc_v2_ctx_t *ctx, const void *samples,
                         int nsamps, int nbAnt, uint64_t timestamp);

/* gNB UL read: dequeue from ul_rx ring. Returns 0 if ring empty (non-blocking). */
int  gpu_ipc_v2_ul_read(gpu_ipc_v2_ctx_t *ctx, void *samples,
                        int nsamps, int nbAnt, uint64_t *timestamp);

#endif /* GPU_IPC_V2_H */
