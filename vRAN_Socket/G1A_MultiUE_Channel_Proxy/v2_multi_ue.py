"""
================================================================================
G1A v1_multi_ue.py - Multi-UE Channel Proxy (Socket + GPU IPC dual-mode)

Based on G0 v12_gpu_ipc.py. Extends from 1 gNB - 1 UE to 1 gNB - N UE (SISO).

[Key changes from G0 v12]
  1. Per-UE independent Sionna channels (position, velocity, LOS)
  2. DL: gNB signal -> per-UE channel -> each UE receives different faded signal
  3. UL: per-UE channel applied -> summed -> gNB receives aggregate
  4. Socket mode: per-UE channel routing with UE index tracking
  5. GPU IPC mode: per-UE GPU buffers (dl_rx[k], ul_tx[k])
  6. TDD reciprocity: DL and UL share the same channel coefficients per UE

[Architecture — Batched Element-wise Processing]
  Sionna output: h_k ∈ ℂ^(N_sym × N_FFT)  delay-domain CIR per UE (SISO scalar/subcarrier)
  TDD reciprocity: h_k^T = h_k (SISO), DL/UL share same channel_buffers[k]

  DL: Y_k[s,f] = IFFT( FFT(X_dl[s,f]) ⊙ FFT(h_k[s,f]) )   element-wise product
      batched as (N_UE × N_sym, N_FFT) → single GPU FFT kernel for all UEs

  UL: Z[s,f] = Σ_k IFFT( FFT(X_k[s,f]) ⊙ FFT(h_k[s,f]) )  element-wise product + sum
      batched as (N_UE × N_sym, N_FFT) → single GPU FFT kernel → sum(axis=0)

[Inherited from G0 v12]
  - CUDA Graph, complex128, DLPack, GPU RingBuffer, Batch FFT
  - WindowProfiler, dual-timer, E2E TDD frame statistics
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
import ctypes
import mmap
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging
import threading
import json
import time
from sionna.phy import PI, SPEED_OF_LIGHT
from datetime import datetime
import os
from channel_coefficients_JIN import ChannelCoefficientsGeneratorJIN, random_binary_mask_tf_complex64

try:
    from sionna.phy.channel.tr38901 import PanelArray, Topology, Rays
    print("[Sionna Init] sionna.phy.channel.tr38901 모듈 로드 성공")
except ModuleNotFoundError as e:
    print(f"[Sionna Init] sionna 모듈 로드 실패: {e}")
    import sys
    sys.exit(1)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[GPU Init] CuPy 로드 성공 - GPU 가속 활성화")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("[GPU Init] CuPy 없음 - CPU 모드로 실행")

try:
    import tensorflow as tf
except ImportError:
    tf = None

gpu_num = 0
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

HDR_FMT_LE = "<I I Q I I"
HDR_LEN = struct.calcsize(HDR_FMT_LE)

def unpack_header(b):
    size, nb, ts, frame, subframe = struct.unpack(HDR_FMT_LE, b)
    return size, nb, ts, frame, subframe

MAX_LOG = 300
LOG_LINES = []
DL_LOG_CNT = 0
LAST_TS = None
LAST_WALLTIME = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(os.path.dirname(_SCRIPT_DIR), "saved_rays_data")

def log(direction, size, nb, ts, frame, subframe, samples, note=""):
    global LOG_LINES, DL_LOG_CNT, LAST_TS, LAST_WALLTIME
    if len(LOG_LINES) >= MAX_LOG:
        return
    msg = f"{direction:<17}| size={size:<7} nbAnt={nb:<2} ts={ts:<12} samples={samples:<7} {note}"
    LOG_LINES.append(msg)
    print(msg)
    if direction.strip() == "gNB → Proxy":
        DL_LOG_CNT += 1
        if DL_LOG_CNT % 10 == 0:
            if DL_LOG_CNT == 10:
                print(f"   --- [DL {DL_LOG_CNT}회] ts={ts}")
                LAST_TS = ts
                LAST_WALLTIME = time.time()
            else:
                now = time.time()
                ts_delta = ts - (LAST_TS if LAST_TS is not None else ts)
                wall_delta = now - (LAST_WALLTIME if LAST_WALLTIME is not None else now)
                print(f"   --- [DL {DL_LOG_CNT}회] ts={ts}, Δts={ts_delta}, Δwall={wall_delta:.6f} sec")
                LAST_TS = ts
                LAST_WALLTIME = now

# OFDM NR numerology=1
carrier_frequency = 3.5
FFT_SIZE = 2048
N_FFT = FFT_SIZE
CP1 = 144
CP2 = 160
N_SYM = 14
SYMBOL_SIZES = [CP1 + FFT_SIZE] * 12 + [CP2 + FFT_SIZE] * 2
scs = 30*1e3
Fs = FFT_SIZE*scs

path_loss_dB = 0
pathLossLinear = 10**(path_loss_dB / 20.0)
snr_dB = None
noise_enabled = False
Speed = 3

def radian_to_degree(radian):
    return radian * (180.0 / PI)

def degree_to_radian(degree):
    return degree * (PI / 180.0)

def set_BS(location=[0,0,0], rotation=[0,0], num_rows_per_panel=1, num_cols_per_panel=1, num_rows=1, num_cols=1,
           polarization="single", polarization_type="V", antenna_pattern="38.901",
           panel_vertical_spacing=2.5, panel_horizontal_spacing=2.5):
    BSexample = {
        "location": location, "rotation": rotation,
        "num_rows_per_panel": num_rows_per_panel, "num_cols_per_panel": num_cols_per_panel,
        "num_rows": num_rows, "num_cols": num_cols,
        "polarization": polarization, "polarization_type": polarization_type,
        "antenna_pattern": antenna_pattern,
        "panel_vertical_spacing": panel_vertical_spacing,
        "panel_horizontal_spacing": panel_horizontal_spacing
    }
    tx_antennas = int(BSexample["num_rows_per_panel"] * BSexample["num_cols_per_panel"] *
                      BSexample["num_rows"] * BSexample["num_cols"])
    return BSexample, tx_antennas

def get_ofdm_symbol_indices(total_samples):
    indices = []
    idx = 0
    for s in SYMBOL_SIZES:
        if idx + s > total_samples:
            break
        indices.append((idx, idx + s))
        idx += s
    return indices


# ============================================================================
# WindowProfiler (from G0 v12)
# ============================================================================

class WindowProfiler:
    """Rolling-window latency statistics (avg / p95 / p99 / max)."""
    def __init__(self, name, metrics, window=500, report_interval=100):
        self.name = name
        self.metrics = metrics
        self.window = int(max(10, window))
        self.report_interval = int(max(1, report_interval))
        self.samples = 0
        self.buffers = {m: deque(maxlen=self.window) for m in self.metrics}

    def add(self, tag="", **kwargs):
        self.samples += 1
        for key in self.metrics:
            if key in kwargs and kwargs[key] is not None:
                self.buffers[key].append(float(kwargs[key]))
        if self.samples % self.report_interval == 0:
            self.print_report(tag=tag)

    def _fmt(self, values):
        if not values:
            return "n/a"
        arr = np.asarray(values, dtype=np.float64)
        return (f"avg={arr.mean():.3f} p95={np.percentile(arr,95):.3f} "
                f"p99={np.percentile(arr,99):.3f} max={arr.max():.3f} ms")

    def print_report(self, tag=""):
        tag_txt = f" {tag}" if tag else ""
        print(f"\n[PROFILE {self.name}#{self.samples}{tag_txt}] window={self.window}")
        for key in self.metrics:
            print(f"  - {key:<14} {self._fmt(self.buffers[key])}")


# ============================================================================
# GPU IPC Interface - Multi-UE Extension
# ============================================================================

GPU_IPC_SHM_PATH = "/tmp/oai_gpu_ipc/gpu_ipc_shm"
GPU_IPC_MAGIC = 0x47505533      # "GPU3" - ring buffer version
GPU_IPC_VERSION = 3
GPU_IPC_HANDLE_SIZE = 64
GPU_IPC_MAX_UE = 8
GPU_IPC_MAX_DATA_SIZE = 61440 * 4  # 240KB per slot
GPU_IPC_SHM_SIZE = 4096
GPU_IPC_RING_DEPTH = 4
GPU_IPC_RING_MASK = GPU_IPC_RING_DEPTH - 1

# SHM layout (4096 bytes):
#
# === IPC Handle area (0-1151, unchanged) ===
#   0      : dl_tx handle (64B)
#   64     : ul_rx handle (64B)
#   128    : dl_rx handles[MAX_UE] (64*8 = 512B)
#   640    : ul_tx handles[MAX_UE] (64*8 = 512B)
#
# === Ring control area (1152-3023) ===
#   Per-ring control block = 104 bytes:
#     head(4) + tail(4) + slots[RING_DEPTH] * {ts(8)+nsamps(4)+nbAnt(4)+data_size(4)+pad(4)}
#
#   1152 : dl_tx_ring    (104B)  - gNB writes, Proxy reads
#   1256 : ul_rx_ring    (104B)  - Proxy writes, gNB reads
#   1360 : dl_rx_ring[8] (832B)  - Proxy writes, UE reads
#   2192 : ul_tx_ring[8] (832B)  - UE writes, Proxy reads
#
# === Global area (3024-3043) ===
#   3024 : ring_depth (uint32)
#   3028 : num_ues    (uint32)
#   3032 : magic      (uint32)
#   3036 : version    (uint32)
#   3040 : buf_size   (uint32)  - per-SLOT size

_OFF_DL_TX_HANDLE    = 0
_OFF_UL_RX_HANDLE    = 64
_OFF_DL_RX_HANDLES   = 128    # + k*64
_OFF_UL_TX_HANDLES   = 640    # + k*64

_RING_SLOT_META_SIZE = 24
_RING_CTRL_SIZE      = 8 + GPU_IPC_RING_DEPTH * _RING_SLOT_META_SIZE  # 104

_OFF_RING_BASE       = 1152
_OFF_DL_TX_RING      = _OFF_RING_BASE
_OFF_UL_RX_RING      = _OFF_DL_TX_RING  + _RING_CTRL_SIZE
_OFF_DL_RX_RINGS     = _OFF_UL_RX_RING  + _RING_CTRL_SIZE
_OFF_UL_TX_RINGS     = _OFF_DL_RX_RINGS + GPU_IPC_MAX_UE * _RING_CTRL_SIZE

_OFF_GLOBAL          = _OFF_UL_TX_RINGS + GPU_IPC_MAX_UE * _RING_CTRL_SIZE
_OFF_RING_DEPTH      = _OFF_GLOBAL
_OFF_NUM_UES         = _OFF_GLOBAL + 4
_OFF_MAGIC           = _OFF_GLOBAL + 8
_OFF_VERSION         = _OFF_GLOBAL + 12
_OFF_BUF_SIZE        = _OFF_GLOBAL + 16


class GPUIpcInterface:
    """
    Multi-UE CUDA IPC with SPSC ring buffers -- SERVER role.

    Each data path is a lock-free ring of GPU_IPC_RING_DEPTH slots.
    Proxy allocates (2 + 2*N) ring buffers, each RING_DEPTH * slot_size bytes.

    Rings:
      dl_tx_ring    : 1 ring  -- gNB enqueues DL, Proxy dequeues
      ul_rx_ring    : 1 ring  -- Proxy enqueues summed UL, gNB dequeues
      dl_rx_ring[k] : N rings -- Proxy enqueues per-UE DL, UE_k dequeues
      ul_tx_ring[k] : N rings -- UE_k enqueues UL, Proxy dequeues
    """

    def __init__(self, shm_path=GPU_IPC_SHM_PATH, num_ues=1):
        self.shm_path = shm_path
        self.num_ues = min(num_ues, GPU_IPC_MAX_UE)
        self.ring_depth = GPU_IPC_RING_DEPTH
        self.ring_mask = GPU_IPC_RING_MASK
        self.shm_fd = None
        self.shm_mm = None
        self.gpu_dl_tx_ptr = 0
        self.gpu_ul_rx_ptr = 0
        self.gpu_dl_rx_ptrs: List[int] = []
        self.gpu_ul_tx_ptrs: List[int] = []
        self._gpu_mem = []
        self.buf_size = GPU_IPC_MAX_DATA_SIZE   # per-slot size
        self.initialized = False

    def init(self):
        shm_dir = os.path.dirname(self.shm_path)
        os.makedirs(shm_dir, mode=0o777, exist_ok=True)

        self.shm_fd = os.open(self.shm_path,
                              os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
        os.ftruncate(self.shm_fd, GPU_IPC_SHM_SIZE)
        self.shm_mm = mmap.mmap(self.shm_fd, GPU_IPC_SHM_SIZE)
        self.shm_mm[:] = b'\x00' * GPU_IPC_SHM_SIZE

        ring_alloc = self.buf_size * self.ring_depth

        def _alloc_ring_and_export(name, handle_offset):
            mem = cp.cuda.alloc(ring_alloc)
            self._gpu_mem.append(mem)
            ptr = mem.ptr
            handle_bytes = cp.cuda.runtime.ipcGetMemHandle(ptr)
            self.shm_mm[handle_offset:handle_offset + GPU_IPC_HANDLE_SIZE] = handle_bytes
            print(f"[GPU IPC] SERVER: ring {name} "
                  f"({self.ring_depth}x{self.buf_size}={ring_alloc} bytes, ptr=0x{ptr:x})")
            return ptr

        self.gpu_dl_tx_ptr = _alloc_ring_and_export("dl_tx", _OFF_DL_TX_HANDLE)
        self.gpu_ul_rx_ptr = _alloc_ring_and_export("ul_rx", _OFF_UL_RX_HANDLE)

        self.gpu_dl_rx_ptrs = []
        for k in range(self.num_ues):
            ptr = _alloc_ring_and_export(
                f"dl_rx[{k}]", _OFF_DL_RX_HANDLES + k * GPU_IPC_HANDLE_SIZE)
            self.gpu_dl_rx_ptrs.append(ptr)

        self.gpu_ul_tx_ptrs = []
        for k in range(self.num_ues):
            ptr = _alloc_ring_and_export(
                f"ul_tx[{k}]", _OFF_UL_TX_HANDLES + k * GPU_IPC_HANDLE_SIZE)
            self.gpu_ul_tx_ptrs.append(ptr)

        struct.pack_into('<I', self.shm_mm, _OFF_RING_DEPTH, self.ring_depth)
        struct.pack_into('<I', self.shm_mm, _OFF_NUM_UES, self.num_ues)
        struct.pack_into('<I', self.shm_mm, _OFF_BUF_SIZE, self.buf_size)
        struct.pack_into('<I', self.shm_mm, _OFF_VERSION, GPU_IPC_VERSION)

        self.shm_mm.flush()
        struct.pack_into('<I', self.shm_mm, _OFF_MAGIC, GPU_IPC_MAGIC)
        self.shm_mm.flush()

        self.initialized = True
        total_rings = 2 + 2 * self.num_ues
        print(f"[GPU IPC] SERVER: ready (magic=0x{GPU_IPC_MAGIC:08X}, "
              f"version={GPU_IPC_VERSION}, num_ues={self.num_ues}, "
              f"ring_depth={self.ring_depth}, slot={self.buf_size}, "
              f"total_gpu_mem={total_rings * ring_alloc / 1024:.0f}KB)")
        return True

    # ── GPU array access ──

    def get_gpu_array(self, ptr, nbytes, dtype=cp.int16):
        n_elements = nbytes // dtype().itemsize
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(n_elements, dtype=dtype, memptr=memptr)

    def _slot_gpu_ptr(self, ring_base_ptr, slot_idx):
        return ring_base_ptr + slot_idx * self.buf_size

    # ── Low-level ring helpers ──

    def _ring_head(self, ring_off):
        return struct.unpack_from('<I', self.shm_mm, ring_off)[0]

    def _ring_tail(self, ring_off):
        return struct.unpack_from('<I', self.shm_mm, ring_off + 4)[0]

    def _set_ring_head(self, ring_off, val):
        struct.pack_into('<I', self.shm_mm, ring_off, val & 0xFFFFFFFF)
        self.shm_mm.flush()

    def _set_ring_tail(self, ring_off, val):
        struct.pack_into('<I', self.shm_mm, ring_off + 4, val & 0xFFFFFFFF)
        self.shm_mm.flush()

    def _ring_count(self, ring_off):
        """Items available in ring (unsigned head - tail)."""
        return (self._ring_head(ring_off) - self._ring_tail(ring_off)) & 0xFFFFFFFF

    def _ring_readable(self, ring_off):
        return self._ring_count(ring_off) > 0

    def _ring_writable(self, ring_off):
        return self._ring_count(ring_off) < self.ring_depth

    def _slot_meta_off(self, ring_off, slot_idx):
        return ring_off + 8 + slot_idx * _RING_SLOT_META_SIZE

    def _read_slot_meta(self, ring_off, slot_idx):
        base = self._slot_meta_off(ring_off, slot_idx)
        ts = struct.unpack_from('<Q', self.shm_mm, base)[0]
        nsamps = struct.unpack_from('<i', self.shm_mm, base + 8)[0]
        nbAnt = struct.unpack_from('<i', self.shm_mm, base + 12)[0]
        data_size = struct.unpack_from('<i', self.shm_mm, base + 16)[0]
        return ts, nsamps, nbAnt, data_size

    def _write_slot_meta(self, ring_off, slot_idx, ts, nsamps, nbAnt, data_size):
        base = self._slot_meta_off(ring_off, slot_idx)
        struct.pack_into('<Q', self.shm_mm, base, ts)
        struct.pack_into('<i', self.shm_mm, base + 8, nsamps)
        struct.pack_into('<i', self.shm_mm, base + 12, nbAnt)
        struct.pack_into('<i', self.shm_mm, base + 16, data_size)
        self.shm_mm.flush()

    # ── DL TX ring (gNB writes, Proxy reads) ──

    def dl_tx_available(self):
        return self._ring_readable(_OFF_DL_TX_RING)

    def dequeue_dl_tx(self):
        """Dequeue from dl_tx ring. Returns (gpu_ptr, ts, nsamps, nbAnt, data_size)."""
        roff = _OFF_DL_TX_RING
        tail = self._ring_tail(roff)
        slot = tail & self.ring_mask
        ts, nsamps, nbAnt, data_size = self._read_slot_meta(roff, slot)
        gpu_ptr = self._slot_gpu_ptr(self.gpu_dl_tx_ptr, slot)
        return gpu_ptr, ts, nsamps, nbAnt, data_size

    def advance_dl_tx(self):
        """Advance dl_tx tail (Proxy consumed one item)."""
        roff = _OFF_DL_TX_RING
        tail = self._ring_tail(roff)
        self._set_ring_tail(roff, tail + 1)

    # ── DL RX rings (Proxy writes, UE reads) ──

    def enqueue_dl_rx(self, ue_idx, ts, nsamps, nbAnt, data_size):
        """Get write slot in dl_rx[ue_idx] ring. Blocks if full. Returns gpu_ptr."""
        roff = _OFF_DL_RX_RINGS + ue_idx * _RING_CTRL_SIZE
        _wait = 0
        while not self._ring_writable(roff):
            time.sleep(0.00001)
            _wait += 1
            if _wait % 100000 == 0:
                h, t = self._ring_head(roff), self._ring_tail(roff)
                print(f"[WARN] enqueue_dl_rx[{ue_idx}] blocked {_wait} iters (h={h} t={t})")
        head = self._ring_head(roff)
        slot = head & self.ring_mask
        self._write_slot_meta(roff, slot, ts, nsamps, nbAnt, data_size)
        return self._slot_gpu_ptr(self.gpu_dl_rx_ptrs[ue_idx], slot)

    def advance_dl_rx(self, ue_idx):
        """Advance dl_rx[ue_idx] head (Proxy wrote one item, visible to UE)."""
        roff = _OFF_DL_RX_RINGS + ue_idx * _RING_CTRL_SIZE
        head = self._ring_head(roff)
        self._set_ring_head(roff, head + 1)

    # ── UL TX rings (UE writes, Proxy reads) ──

    def ul_tx_available(self, ue_idx):
        roff = _OFF_UL_TX_RINGS + ue_idx * _RING_CTRL_SIZE
        return self._ring_readable(roff)

    def dequeue_ul_tx(self, ue_idx):
        """Dequeue from ul_tx[ue_idx] ring. Returns (gpu_ptr, ts, nsamps, nbAnt, data_size)."""
        roff = _OFF_UL_TX_RINGS + ue_idx * _RING_CTRL_SIZE
        tail = self._ring_tail(roff)
        slot = tail & self.ring_mask
        ts, nsamps, nbAnt, data_size = self._read_slot_meta(roff, slot)
        gpu_ptr = self._slot_gpu_ptr(self.gpu_ul_tx_ptrs[ue_idx], slot)
        return gpu_ptr, ts, nsamps, nbAnt, data_size

    def advance_ul_tx(self, ue_idx):
        """Advance ul_tx[ue_idx] tail (Proxy consumed one item)."""
        roff = _OFF_UL_TX_RINGS + ue_idx * _RING_CTRL_SIZE
        tail = self._ring_tail(roff)
        self._set_ring_tail(roff, tail + 1)

    # ── UL RX ring (Proxy writes, gNB reads) ──

    def enqueue_ul_rx(self, ts, nsamps, nbAnt, data_size):
        """Get write slot in ul_rx ring. Non-blocking: returns None if full."""
        roff = _OFF_UL_RX_RING
        if not self._ring_writable(roff):
            return None
        head = self._ring_head(roff)
        slot = head & self.ring_mask
        self._write_slot_meta(roff, slot, ts, nsamps, nbAnt, data_size)
        return self._slot_gpu_ptr(self.gpu_ul_rx_ptr, slot)

    def advance_ul_rx(self):
        """Advance ul_rx head (Proxy wrote one item, visible to gNB)."""
        roff = _OFF_UL_RX_RING
        head = self._ring_head(roff)
        self._set_ring_head(roff, head + 1)

    def cleanup(self):
        if not self.initialized:
            return
        self._gpu_mem.clear()
        self.gpu_dl_tx_ptr = 0
        self.gpu_ul_rx_ptr = 0
        self.gpu_dl_rx_ptrs.clear()
        self.gpu_ul_tx_ptrs.clear()
        try:
            if self.shm_mm:
                self.shm_mm.close()
        except BufferError:
            pass
        if self.shm_fd is not None:
            os.close(self.shm_fd)
        try:
            os.unlink(self.shm_path)
        except OSError:
            pass
        self.initialized = False
        print("[GPU IPC] Cleanup done")


# ============================================================================
# GPU Slot Pipeline (from G0 v12, unchanged)
# ============================================================================

class GPUSlotPipeline:
    WARMUP_SLOTS = 3

    def __init__(self, fft_size=2048, enable_gpu=True, use_pinned_memory=True,
                 use_cuda_graph=True, profile_interval=100, profile_window=500,
                 dual_timer_compare=True):
        self.fft_size = fft_size
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.use_pinned_memory = use_pinned_memory
        self.slot_counter = 0
        self.profile_interval = max(1, int(profile_interval))
        self.profile_window = max(10, int(profile_window))
        self.dual_timer_compare = bool(dual_timer_compare)

        _sock_metrics = ["H2D", "CH_COPY", "NOISE_PREP", "GPU_COMPUTE", "D2H", "TOTAL"]
        self.profile_gpu = WindowProfiler(
            "GPU_SLOT", _sock_metrics,
            window=self.profile_window, report_interval=self.profile_interval)
        self.profile_gpu_evt = WindowProfiler(
            "GPU_SLOT_EVT", _sock_metrics,
            window=self.profile_window, report_interval=self.profile_interval)
        self.profile_gpu_diff = WindowProfiler(
            "GPU_SLOT_CPU-EVT", _sock_metrics,
            window=self.profile_window, report_interval=self.profile_interval)

        _ipc_metrics = ["GPU_COPY_IN", "CH_COPY", "NOISE_PREP", "GPU_COMPUTE", "GPU_COPY_OUT", "TOTAL"]
        self.profile_ipc = WindowProfiler(
            "IPC_SLOT", _ipc_metrics,
            window=self.profile_window, report_interval=self.profile_interval)
        self.profile_ipc_evt = WindowProfiler(
            "IPC_SLOT_EVT", _ipc_metrics,
            window=self.profile_window, report_interval=self.profile_interval)
        self.profile_ipc_diff = WindowProfiler(
            "IPC_SLOT_CPU-EVT", _ipc_metrics,
            window=self.profile_window, report_interval=self.profile_interval)

        self.n_sym = N_SYM
        self.total_cpx = sum(SYMBOL_SIZES)
        self.total_int16 = self.total_cpx * 2

        if not self.enable_gpu:
            print("[GPU Pipeline] GPU disabled - CPU numpy mode")
            return

        print(f"[GPU Pipeline G1A] Initializing (complex128, CUDA Graph)...")

        self.stream = cp.cuda.Stream(non_blocking=True)

        sym_bounds = get_ofdm_symbol_indices(self.total_cpx)

        ext_idx = cp.zeros((self.n_sym, fft_size), dtype=cp.int64)
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            ext_idx[i] = cp.arange(s + cp_l, e)
        self.gpu_ext_idx = ext_idx

        data_dst = []
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            data_dst.append(cp.arange(s + cp_l, e, dtype=cp.int64))
        self.gpu_data_dst = cp.concatenate(data_dst)

        cp_dst_list, cp_src_list = [], []
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            cp_dst_list.append(cp.arange(s, s + cp_l, dtype=cp.int64))
            cp_src_list.append(cp.arange(i * fft_size + fft_size - cp_l,
                                         i * fft_size + fft_size, dtype=cp.int64))
        self.gpu_cp_dst = cp.concatenate(cp_dst_list)
        self.gpu_cp_src = cp.concatenate(cp_src_list)

        if self.use_pinned_memory:
            self.pinned_iq_in_buf = cp.cuda.alloc_pinned_memory(self.total_int16 * 2)
            self.pinned_iq_out_buf = cp.cuda.alloc_pinned_memory(self.total_int16 * 2)
            self.pinned_iq_in = np.frombuffer(self.pinned_iq_in_buf, dtype=np.int16,
                                              count=self.total_int16)
            self.pinned_iq_out = np.frombuffer(self.pinned_iq_out_buf, dtype=np.int16,
                                               count=self.total_int16)

        self.gpu_iq_in = cp.zeros(self.total_int16, dtype=cp.int16)
        self.gpu_iq_out = cp.zeros(self.total_int16, dtype=cp.int16)
        self.gpu_x = cp.zeros((self.n_sym, fft_size), dtype=cp.complex128)
        self.gpu_H = cp.zeros((self.n_sym, fft_size), dtype=cp.complex128)
        self.gpu_out = cp.zeros(self.total_cpx, dtype=cp.complex128)
        self.gpu_noise_r = cp.zeros(self.total_cpx, dtype=cp.float64)
        self.gpu_noise_i = cp.zeros(self.total_cpx, dtype=cp.float64)

        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph = None
        self.graph_captured = False
        self.warmup_count = 0
        self._graph_pl_linear = None
        self._graph_noise_on = None
        self._graph_snr_db = None

        print(f"[GPU Pipeline G1A] Initialization complete")

    def _regenerate_noise(self):
        self.gpu_noise_r[:] = cp.random.randn(self.total_cpx).astype(cp.float64)
        self.gpu_noise_i[:] = cp.random.randn(self.total_cpx).astype(cp.float64)

    def _gpu_compute_core(self, pl_linear, snr_db, noise_on):
        self._tmp_f64 = self.gpu_iq_in.astype(cp.float64)
        self._tmp_cpx = self._tmp_f64[::2] + 1j * self._tmp_f64[1::2]
        self.gpu_x[:] = self._tmp_cpx[self.gpu_ext_idx]

        self._tmp_Xf = cp.fft.fft(self.gpu_x, axis=1)
        self._tmp_Hf = cp.fft.fft(self.gpu_H, axis=1)
        self._tmp_XfHf = self._tmp_Xf * self._tmp_Hf
        self._tmp_y = cp.fft.ifft(self._tmp_XfHf, axis=1)

        self.gpu_out[:] = 0
        self._tmp_y_flat = self._tmp_y.ravel()
        self.gpu_out[self.gpu_data_dst] = self._tmp_y_flat
        self.gpu_out[self.gpu_cp_dst] = self._tmp_y_flat[self.gpu_cp_src]

        if pl_linear != 1.0:
            self.gpu_out *= cp.float64(pl_linear)

        if noise_on and snr_db is not None:
            self._tmp_abs_sq = cp.abs(self.gpu_out) ** 2
            self._tmp_sig_pwr = cp.mean(self._tmp_abs_sq)
            snr_linear = cp.float64(10.0 ** (snr_db / 10.0))
            self._tmp_n_pwr = self._tmp_sig_pwr / snr_linear
            self._tmp_n_std = cp.sqrt(self._tmp_n_pwr / cp.float64(2.0))
            self._tmp_noise = self._tmp_n_std * (
                self.gpu_noise_r + 1j * self.gpu_noise_i
            )
            self.gpu_out += self._tmp_noise

        self._tmp_clip_r = cp.clip(cp.around(self.gpu_out.real), -32768, 32767)
        self._tmp_clip_i = cp.clip(cp.around(self.gpu_out.imag), -32768, 32767)
        self.gpu_iq_out[::2] = self._tmp_clip_r.astype(cp.int16)
        self.gpu_iq_out[1::2] = self._tmp_clip_i.astype(cp.int16)

    def _try_capture_graph(self, pl_linear, snr_db, noise_on):
        try:
            self.stream.begin_capture()
            self._gpu_compute_core(pl_linear, snr_db, noise_on)
            self.cuda_graph = self.stream.end_capture()
            self.graph_captured = True
            self._graph_pl_linear = pl_linear
            self._graph_noise_on = noise_on
            self._graph_snr_db = snr_db
            print(f"[CUDA Graph] Capture success "
                  f"(PL={pl_linear}, noise={'ON' if noise_on else 'OFF'}"
                  f"{f', SNR={snr_db}dB' if noise_on and snr_db is not None else ''})")
        except Exception as e:
            try:
                self.stream.end_capture()
            except:
                pass
            try:
                cp.cuda.Device(0).synchronize()
            except:
                pass
            print(f"[CUDA Graph] Capture failed - fallback: {e}")
            self.graph_captured = False
            self.use_cuda_graph = False

    def _need_recapture(self, pl_linear, snr_db, noise_on):
        if not self.graph_captured:
            return False
        return (self._graph_pl_linear != pl_linear or
                self._graph_noise_on != noise_on or
                self._graph_snr_db != snr_db)

    def _run_gpu_kernel(self, pl_linear, snr_db, noise_on):
        """Run GPU compute: CUDA Graph launch or direct compute."""
        if noise_on and snr_db is not None:
            self._regenerate_noise()

        if self._need_recapture(pl_linear, snr_db, noise_on):
            self.graph_captured = False
            self.warmup_count = self.WARMUP_SLOTS

        if self.graph_captured:
            self.cuda_graph.launch(self.stream)
        elif self.use_cuda_graph and self.warmup_count >= self.WARMUP_SLOTS:
            self._try_capture_graph(pl_linear, snr_db, noise_on)
            if self.graph_captured:
                self.cuda_graph.launch(self.stream)
            else:
                self._gpu_compute_core(pl_linear, snr_db, noise_on)
        else:
            self._gpu_compute_core(pl_linear, snr_db, noise_on)
            self.warmup_count += 1

    def process_slot(self, iq_bytes, channels_gpu, pl_linear, snr_db, noise_on):
        """Socket mode: raw bytes in -> int16 bytes out"""
        n_iq = len(iq_bytes) // 2
        n_cpx = n_iq // 2

        if not self.enable_gpu or n_cpx != self.total_cpx:
            return self._cpu_fallback(iq_bytes, channels_gpu, pl_linear, snr_db, noise_on)

        with self.stream:
            if self.use_pinned_memory:
                ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
                self.gpu_iq_in.set(self.pinned_iq_in, stream=self.stream)
            else:
                iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
                self.gpu_iq_in[:] = cp.asarray(iq_int16)

            n_ch = min(channels_gpu.shape[0], self.n_sym)
            n_w = min(channels_gpu.shape[1], self.fft_size)
            self.gpu_H[:] = 0
            if channels_gpu.dtype != cp.complex128:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w].astype(cp.complex128)
            else:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w]

            self._run_gpu_kernel(pl_linear, snr_db, noise_on)

            if self.use_pinned_memory:
                self.gpu_iq_out.get(out=self.pinned_iq_out, stream=self.stream)
                self.stream.synchronize()
                result = self.pinned_iq_out.tobytes()
            else:
                out_host = self.gpu_iq_out.get(stream=self.stream)
                self.stream.synchronize()
                result = out_host.tobytes()

        self.slot_counter += 1
        return result

    def process_slot_ipc(self, gpu_iq_in_arr, channels_gpu, pl_linear, snr_db, noise_on, gpu_iq_out_arr):
        """GPU IPC mode: GPU array in -> GPU process -> GPU array out"""
        if not self.enable_gpu:
            return

        with self.stream:
            self.gpu_iq_in[:] = gpu_iq_in_arr[:self.total_int16]

            n_ch = min(channels_gpu.shape[0], self.n_sym)
            n_w = min(channels_gpu.shape[1], self.fft_size)
            self.gpu_H[:] = 0
            if channels_gpu.dtype != cp.complex128:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w].astype(cp.complex128)
            else:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w]

            self._run_gpu_kernel(pl_linear, snr_db, noise_on)

            gpu_iq_out_arr[:self.total_int16] = self.gpu_iq_out[:]
            self.stream.synchronize()

        self.slot_counter += 1

    def get_gpu_out_complex(self):
        """Return the complex128 output before int16 clipping (for UL summation)."""
        return self.gpu_out.copy()

    def process_slot_ipc_complex_out(self, gpu_iq_in_arr, channels_gpu, pl_linear, snr_db, noise_on):
        """GPU IPC mode: returns complex128 output for UL accumulation (no int16 clip)."""
        if not self.enable_gpu:
            return None

        with self.stream:
            self.gpu_iq_in[:] = gpu_iq_in_arr[:self.total_int16]

            n_ch = min(channels_gpu.shape[0], self.n_sym)
            n_w = min(channels_gpu.shape[1], self.fft_size)
            self.gpu_H[:] = 0
            if channels_gpu.dtype != cp.complex128:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w].astype(cp.complex128)
            else:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w]

            self._run_gpu_kernel(pl_linear, snr_db, noise_on)
            self.stream.synchronize()

        self.slot_counter += 1
        return self.gpu_out.copy()

    def complex_to_int16_gpu(self, cpx_signal, gpu_iq_out_arr):
        """Convert complex128 GPU array to int16 and write to IPC buffer."""
        with self.stream:
            clip_r = cp.clip(cp.around(cpx_signal.real), -32768, 32767)
            clip_i = cp.clip(cp.around(cpx_signal.imag), -32768, 32767)
            gpu_iq_out_arr[::2] = clip_r.astype(cp.int16)
            gpu_iq_out_arr[1::2] = clip_i.astype(cp.int16)
            self.stream.synchronize()

    def complex_to_int16_bytes(self, cpx_signal):
        """Convert complex128 GPU array to int16 bytes (for socket mode UL sum)."""
        with self.stream:
            clip_r = cp.clip(cp.around(cpx_signal.real), -32768, 32767)
            clip_i = cp.clip(cp.around(cpx_signal.imag), -32768, 32767)
            out = cp.zeros(len(cpx_signal) * 2, dtype=cp.int16)
            out[::2] = clip_r.astype(cp.int16)
            out[1::2] = clip_i.astype(cp.int16)
            self.stream.synchronize()
        if self.use_pinned_memory:
            out.get(out=self.pinned_iq_out, stream=self.stream)
            self.stream.synchronize()
            return self.pinned_iq_out.tobytes()
        return out.get().tobytes()

    # ── Batch processing for Multi-UE (fully vectorized, no per-UE loops) ──

    def _build_H_batch(self, channels_list, n_ue):
        """Stack per-UE channel arrays into (N_UE, N_SYM, fft_size) tensor."""
        H_batch = cp.zeros((n_ue, self.n_sym, self.fft_size), dtype=cp.complex128)
        for k, ch in enumerate(channels_list):
            n_ch = min(ch.shape[0], self.n_sym)
            n_w = min(ch.shape[1], self.fft_size)
            if ch.dtype != cp.complex128:
                H_batch[k, :n_ch, :n_w] = ch[:n_ch, :n_w].astype(cp.complex128)
            else:
                H_batch[k, :n_ch, :n_w] = ch[:n_ch, :n_w]
        return H_batch

    def _batch_fft_convolve(self, x_batch, H_batch, n_ue):
        """Batched FFT convolution: (N_UE, N_SYM, fft) -> (N_UE, N_SYM, fft)."""
        x_flat = x_batch.reshape(n_ue * self.n_sym, self.fft_size)
        h_flat = H_batch.reshape(n_ue * self.n_sym, self.fft_size)
        Xf = cp.fft.fft(x_flat, axis=1)
        Hf = cp.fft.fft(h_flat, axis=1)
        return cp.fft.ifft(Xf * Hf, axis=1).reshape(n_ue, self.n_sym, self.fft_size)

    def _batch_ofdm_reconstruct(self, y_batch, n_ue, pl_linear, snr_db, noise_on):
        """Vectorized OFDM reconstruction: (N_UE, N_SYM, fft) -> (N_UE, total_cpx).

        CP insertion, path loss, and AWGN all applied as (N_UE, ...) tensor ops.
        """
        y_all = y_batch.reshape(n_ue, self.n_sym * self.fft_size)
        out_batch = cp.zeros((n_ue, self.total_cpx), dtype=cp.complex128)
        out_batch[:, self.gpu_data_dst] = y_all
        out_batch[:, self.gpu_cp_dst] = y_all[:, self.gpu_cp_src]

        if pl_linear != 1.0:
            out_batch *= cp.float64(pl_linear)

        if noise_on and snr_db is not None:
            sig_pwr = cp.mean(cp.abs(out_batch) ** 2, axis=1, keepdims=True)
            snr_lin = cp.float64(10.0 ** (snr_db / 10.0))
            n_std = cp.sqrt(sig_pwr / snr_lin / cp.float64(2.0))
            noise = n_std * (cp.random.randn(n_ue, self.total_cpx).astype(cp.float64)
                             + 1j * cp.random.randn(n_ue, self.total_cpx).astype(cp.float64))
            out_batch += noise

        return out_batch

    def _batch_complex_to_int16(self, out_batch, n_ue):
        """Vectorized complex128 -> int16 conversion: (N_UE, total_cpx) -> (N_UE, total_int16)."""
        clip_r = cp.clip(cp.around(out_batch.real), -32768, 32767)
        clip_i = cp.clip(cp.around(out_batch.imag), -32768, 32767)
        iq_batch = cp.zeros((n_ue, self.total_int16), dtype=cp.int16)
        iq_batch[:, ::2] = clip_r.astype(cp.int16)
        iq_batch[:, 1::2] = clip_i.astype(cp.int16)
        return iq_batch

    def process_slot_batch_dl(self, iq_bytes, channels_list, pl_linear, snr_db, noise_on):
        """Batch DL: same input signal x N channels -> N output byte arrays.

        Fully vectorized: single batched FFT, batched OFDM reconstruct,
        batched int16 conversion, single D2H transfer.
        """
        n_ue = len(channels_list)
        if n_ue == 0:
            return []
        if not self.enable_gpu or n_ue == 1:
            return [self.process_slot(iq_bytes, channels_list[0], pl_linear, snr_db, noise_on)]

        n_iq = len(iq_bytes) // 2
        n_cpx = n_iq // 2
        if n_cpx != self.total_cpx:
            return [self.process_slot(iq_bytes, ch, pl_linear, snr_db, noise_on)
                    for ch in channels_list]

        with self.stream:
            if self.use_pinned_memory:
                ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
                self.gpu_iq_in.set(self.pinned_iq_in, stream=self.stream)
            else:
                iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
                self.gpu_iq_in[:] = cp.asarray(iq_int16)

            tmp_f64 = self.gpu_iq_in.astype(cp.float64)
            tmp_cpx = tmp_f64[::2] + 1j * tmp_f64[1::2]
            x_syms = tmp_cpx[self.gpu_ext_idx]  # (N_SYM, fft_size)

            x_batch = cp.broadcast_to(x_syms[None, :, :], (n_ue, self.n_sym, self.fft_size)).copy()
            H_batch = self._build_H_batch(channels_list, n_ue)

            y_batch = self._batch_fft_convolve(x_batch, H_batch, n_ue)
            out_batch = self._batch_ofdm_reconstruct(y_batch, n_ue, pl_linear, snr_db, noise_on)
            iq_batch = self._batch_complex_to_int16(out_batch, n_ue)

            # Single D2H transfer for all UEs at once
            self.stream.synchronize()
            iq_host = iq_batch.get()

        self.slot_counter += 1
        return [iq_host[k].tobytes() for k in range(n_ue)]

    def process_slot_batch_dl_ipc(self, gpu_iq_in_arr, channels_list, pl_linear,
                                  snr_db, noise_on, gpu_iq_out_arrs):
        """Batch DL IPC: same GPU input x N channels -> N GPU outputs.

        Fully vectorized: batched FFT + reconstruct + int16 -> scatter to per-UE buffers.
        """
        n_ue = len(channels_list)
        if not self.enable_gpu or n_ue == 0:
            return

        with self.stream:
            self.gpu_iq_in[:] = gpu_iq_in_arr[:self.total_int16]

            tmp_f64 = self.gpu_iq_in.astype(cp.float64)
            tmp_cpx = tmp_f64[::2] + 1j * tmp_f64[1::2]
            x_syms = tmp_cpx[self.gpu_ext_idx]

            x_batch = cp.broadcast_to(x_syms[None, :, :], (n_ue, self.n_sym, self.fft_size)).copy()
            H_batch = self._build_H_batch(channels_list, n_ue)

            y_batch = self._batch_fft_convolve(x_batch, H_batch, n_ue)
            out_batch = self._batch_ofdm_reconstruct(y_batch, n_ue, pl_linear, snr_db, noise_on)
            iq_batch = self._batch_complex_to_int16(out_batch, n_ue)

            for k in range(n_ue):
                gpu_iq_out_arrs[k][:self.total_int16] = iq_batch[k]

            self.stream.synchronize()
        self.slot_counter += 1

    def process_slot_batch_ul_ipc(self, gpu_iq_in_arrs, channels_list, pl_linear,
                                  snr_db, noise_on, gpu_iq_out_arr):
        """Batch UL IPC: N GPU inputs x N channels -> summed GPU output.

        Fully vectorized: batched parse, batched FFT, batched reconstruct,
        sum(axis=0), single int16 conversion.
        """
        n_ue = len(gpu_iq_in_arrs)
        if not self.enable_gpu or n_ue == 0:
            return

        with self.stream:
            iq_batch = cp.stack([arr[:self.total_int16] for arr in gpu_iq_in_arrs])
            f64_batch = iq_batch.astype(cp.float64)
            cpx_batch = f64_batch[:, ::2] + 1j * f64_batch[:, 1::2]  # (N_UE, total_cpx)

            x_batch = cpx_batch[:, self.gpu_ext_idx]  # (N_UE, N_SYM, fft_size)
            H_batch = self._build_H_batch(channels_list, n_ue)

            y_batch = self._batch_fft_convolve(x_batch, H_batch, n_ue)
            out_batch = self._batch_ofdm_reconstruct(y_batch, n_ue, pl_linear, snr_db, noise_on)

            # Vectorized sum across all UEs -> single (total_cpx,) signal
            accum = out_batch.sum(axis=0)

            clip_r = cp.clip(cp.around(accum.real), -32768, 32767)
            clip_i = cp.clip(cp.around(accum.imag), -32768, 32767)
            gpu_iq_out_arr[::2] = clip_r.astype(cp.int16)
            gpu_iq_out_arr[1::2] = clip_i.astype(cp.int16)
            self.stream.synchronize()

        self.slot_counter += 1

    def _cpu_fallback(self, iq_bytes, channels_gpu, pl_linear, snr_db, noise_on):
        iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
        x_cpx = iq_int16[::2].astype(np.float64) + 1j * iq_int16[1::2].astype(np.float64)
        n_cpx = len(x_cpx)
        sym_idx = get_ofdm_symbol_indices(n_cpx)

        if GPU_AVAILABLE and hasattr(channels_gpu, 'get'):
            ch_np = cp.asnumpy(channels_gpu)
        else:
            ch_np = np.asarray(channels_gpu)

        out = np.zeros(n_cpx, dtype=np.complex128)
        for i, (s, e) in enumerate(sym_idx):
            cp_l = CP1 if i < 12 else CP2
            sym = x_cpx[s + cp_l : e]
            h = ch_np[i] if i < ch_np.shape[0] else np.ones(self.fft_size, dtype=np.complex64)
            Xf = np.fft.fft(sym)
            Hf = np.fft.fft(h, self.fft_size)
            y = np.fft.ifft(Xf * Hf)
            out[s + cp_l : e] = y[:self.fft_size]
            out[s : s + cp_l] = y[self.fft_size - cp_l : self.fft_size]

        out *= pl_linear
        if noise_on and snr_db is not None:
            sp = np.mean(np.abs(out) ** 2)
            if sp > 0:
                ns = np.sqrt(sp / (10.0 ** (snr_db / 10.0)) / 2.0)
                out += ns * (np.random.randn(n_cpx) + 1j * np.random.randn(n_cpx))

        y16 = np.empty(n_cpx * 2, dtype='<i2')
        y16[::2] = np.clip(np.round(out.real), -32768, 32767).astype('<i2')
        y16[1::2] = np.clip(np.round(out.imag), -32768, 32767).astype('<i2')
        return y16.tobytes()


# ============================================================================
# Socket-mode Endpoint (from G0 v12)
# ============================================================================

@dataclass
class Endpoint:
    sock:  socket.socket
    role:  str
    ue_idx: int = -1  # G1A: assigned UE index (-1 = unassigned / gNB)
    rx:    bytearray = field(default_factory=bytearray)
    stage: str = "hdr"
    pay_len: int = 0
    hdr_raw: bytes = b""
    hdr_vals: Tuple[int,int,int,int,int]|None = None
    closed: bool = False

    def fileno(self):
        return self.sock.fileno()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.close()
        finally:
            pass

    def read_blocks(self):
        blocks = []
        try:
            chunk = self.sock.recv(65536)
        except BlockingIOError:
            return blocks
        except OSError:
            self.close()
            return blocks
        if not chunk:
            self.close()
            return blocks
        self.rx.extend(chunk)

        while True:
            if self.stage == "hdr":
                if len(self.rx) < HDR_LEN:
                    break
                self.hdr_raw = bytes(self.rx[:HDR_LEN])
                del self.rx[:HDR_LEN]
                size, nb, ts, frame, subframe = unpack_header(self.hdr_raw)
                self.hdr_vals = (size, nb, ts, frame, subframe)
                self.pay_len = size * nb * 4
                self.stage = "pay"

            if self.stage == "pay":
                if len(self.rx) < self.pay_len:
                    break
                payload = bytes(self.rx[:self.pay_len])
                del self.rx[:self.pay_len]
                blocks.append((self.hdr_raw, self.hdr_vals, payload))
                self.stage = "hdr"
                self.hdr_raw = b""
                self.hdr_vals = None
                self.pay_len = 0
        return blocks

    def send(self, h, p):
        try:
            self.sock.sendall(h+p)
        except OSError:
            self.close()


# ============================================================================
# RingBuffer & ChannelProducer (from G0 v12)
# ============================================================================

class RingBuffer:
    def __init__(self, shape, dtype=cp.complex64, maxlen=1024):
        if GPU_AVAILABLE:
            self.buffer = cp.zeros((maxlen,) + shape, dtype=dtype)
            self.is_gpu = True
        else:
            self.buffer = np.zeros((maxlen,) + shape, dtype=np.complex64)
            self.is_gpu = False
        self.maxlen = maxlen
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put_batch(self, data_batch):
        n = data_batch.shape[0]
        with self.not_full:
            for i in range(n):
                while self.count == self.maxlen:
                    self.not_full.wait()
                self.buffer[self.write_idx] = data_batch[i]
                self.write_idx = (self.write_idx + 1) % self.maxlen
                self.count += 1
            self.not_empty.notify_all()

    def get_batch(self, n):
        with self.not_empty:
            while self.count < n:
                self.not_empty.wait()
            end = self.read_idx + n
            if end <= self.maxlen:
                batch = self.buffer[self.read_idx:end].copy()
            else:
                lib = cp if self.is_gpu else np
                batch = lib.concatenate([
                    self.buffer[self.read_idx:],
                    self.buffer[:end - self.maxlen]
                ])
            self.read_idx = end % self.maxlen
            self.count -= n
            self.not_full.notify_all()
        return batch

    @property
    def available(self):
        with self.lock:
            return self.count


class ChannelProducer(threading.Thread):
    """Produces channel coefficients for a single UE (1 BS - 1 UE SISO)."""

    def __init__(self, buffer, channel_generator, topology, params,
                 h_field_array_power, aoa_delay, zoa_delay,
                 buffer_symbol_size=32, ue_label=""):
        super().__init__()
        self.buffer = buffer
        self.channel_generator = channel_generator
        self.topology = topology
        self.params = params
        self.h_field_array_power = h_field_array_power
        self.aoa_delay = aoa_delay
        self.zoa_delay = zoa_delay
        self.buffer_symbol_size = buffer_symbol_size
        self.stop_event = threading.Event()
        self.daemon = True
        self.symbol_counter = 0
        self.ue_label = ue_label

    def run(self):
        if GPU_AVAILABLE:
            cp.cuda.Device(0).use()
        while not self.stop_event.is_set():
            sample_times = tf.cast(
                tf.range(self.buffer_symbol_size), self.channel_generator.rdtype
            ) / tf.constant(self.params['scs'], self.channel_generator.rdtype)
            ActiveUE_component = random_binary_mask_tf_complex64(self.params['N_UE'], k=self.params['N_UE_active'])
            ActiveUE = tf.constant(ActiveUE_component, dtype=tf.complex64)
            ServingBS_component = random_binary_mask_tf_complex64(self.params['N_BS'], k=self.params['N_BS_serving'])
            ServingBS = tf.constant(ServingBS_component, dtype=tf.complex64)
            h_delay, _, _, _ = self.channel_generator._H_TTI_sequential_fft_o_ELW2_noProfile(
                self.topology, ActiveUE, ServingBS, sample_times,
                self.h_field_array_power, self.aoa_delay, self.zoa_delay
            )
            h_delay = tf.squeeze(h_delay)

            h_c128 = tf.cast(h_delay, tf.complex128)
            energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=-1, keepdims=True)
            h_norm = h_delay / tf.cast(tf.sqrt(energy), h_delay.dtype)
            h_c64 = tf.cast(h_norm, tf.complex64)

            try:
                h_cp_batch = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c64)).copy()
            except Exception:
                h_cp_batch = cp.asarray(h_c64.numpy())

            self.buffer.put_batch(h_cp_batch)
            self.symbol_counter += self.buffer_symbol_size


# ============================================================================
# Proxy - Multi-UE (Socket + GPU IPC dual-mode)
# ============================================================================

class Proxy:
    def __init__(self, mode="socket", num_ues=2,
                 ue_port=6014, gnb_host="127.0.0.1", gnb_port=6013,
                 log_level="info", ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np",
                 custom_channel=False, buffer_len=4096, buffer_symbol_size=42,
                 enable_gpu=True, use_pinned_memory=True, use_cuda_graph=True,
                 ipc_shm_path=GPU_IPC_SHM_PATH,
                 profile_interval=100, profile_window=500, dual_timer_compare=True,
                 batch_mode=True):
        self.mode = mode
        self.num_ues = min(num_ues, GPU_IPC_MAX_UE)
        self.batch_mode = batch_mode
        self.prev_ts = None
        self.global_symbol_count = 0
        self.slot_sample_accum = 0
        self.ch_en = ch_en
        self.ch_L = ch_L
        self.ch_dd = ch_dd
        self.log_plot = log_plot
        self.conv_mode = conv_mode
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.fft_lib = fft_lib
        self.custom_channel = custom_channel
        self.buffer_len = buffer_len
        self.buffer_symbol_size = buffer_symbol_size
        self.enable_gpu = enable_gpu
        self.use_pinned_memory = use_pinned_memory
        self.use_cuda_graph = use_cuda_graph
        self.ipc_shm_path = ipc_shm_path
        self.profile_interval = max(1, int(profile_interval))
        self.profile_window = max(10, int(profile_window))
        self.dual_timer_compare = bool(dual_timer_compare)

        self.ipc = None

        self.profile_ofdm = WindowProfiler(
            "OFDM_SLOT",
            ["CH_GET", "CH_PAD", "GPU_PROC", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval)
        self.profile_proxy = WindowProfiler(
            "PROXY_E2E",
            ["PROC", "SEND", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval)

        self._e2e_slot_count = 0
        self._e2e_frame_slots = 10
        self._e2e_last_wall = None
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0
        self._e2e_dl_per_ue_ms = [0.0] * self.num_ues
        self._e2e_ul_per_ue_ms = [0.0] * self.num_ues

        # Socket mode setup
        if mode == "socket":
            self.sel = selectors.DefaultSelector()
            self.lis = socket.socket()
            self.lis.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.lis.bind(("0.0.0.0", ue_port))
            self.lis.listen()
            self.lis.setblocking(False)
            self.sel.register(self.lis, selectors.EVENT_READ, data="UE_LIS")
            print(f"[INFO] Socket mode: UE listen 0.0.0.0:{ue_port}")
            self.gnb_host, self.gnb_port = gnb_host, gnb_port
            self.gnb_ep: Optional[Endpoint] = None
            self.ues: Dict[int, Endpoint] = {}
            self.ue_index_map: Dict[int, int] = {}  # fd -> ue_idx
            self._next_ue_idx = 0
            self.gnb_hshake: Optional[Tuple[bytes, bytes]] = None
            # UL accumulation buffers: ts -> {ue_idx_set, accumulated complex signal}
            self._ul_accum: Dict[int, dict] = {}
            self._ul_accum_hdr: Dict[int, bytes] = {}
        else:
            print(f"[INFO] GPU IPC mode: shm_path={ipc_shm_path}, num_ues={self.num_ues}")

        self._init_channels(buffer_len, buffer_symbol_size)

    def _init_channels(self, buffer_len, buffer_symbol_size):
        """Initialize per-UE Sionna channel models and GPU pipeline."""
        self.channel_buffers: List[RingBuffer] = []
        self.producers: List[ChannelProducer] = []

        if not self.custom_channel:
            self.gpu_slot_pipeline = GPUSlotPipeline(
                FFT_SIZE, enable_gpu=self.enable_gpu,
                use_pinned_memory=self.use_pinned_memory,
                use_cuda_graph=self.use_cuda_graph,
                profile_interval=self.profile_interval,
                profile_window=self.profile_window,
                dual_timer_compare=self.dual_timer_compare)
            return

        phi_r_rays = tf.convert_to_tensor(np.load(directory + "/phi_r_rays_for_ChannelBlock.npy"))
        phi_t_rays = tf.convert_to_tensor(np.load(directory + "/phi_t_rays_for_ChannelBlock.npy"))
        theta_r_rays = tf.convert_to_tensor(np.load(directory + "/theta_r_rays_for_ChannelBlock.npy"))
        theta_t_rays = tf.convert_to_tensor(np.load(directory + "/theta_t_rays_for_ChannelBlock.npy"))
        power_rays = tf.convert_to_tensor(np.load(directory + "/power_rays_for_ChannelBlock.npy"))
        tau_rays = tf.convert_to_tensor(np.load(directory + "/tau_rays_for_ChannelBlock.npy"))

        batch_size = 1
        N_BS = 1
        N_UE_sionna = 1  # Each channel is 1 BS - 1 UE; we create N independent ones
        BSexample, _ = set_BS()

        ArrayRX = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1, num_rows=1, num_cols=1,
            polarization='single', polarization_type='V', antenna_pattern='omni',
            carrier_frequency=carrier_frequency
        )
        ArrayTX = PanelArray(
            num_rows_per_panel=BSexample["num_rows_per_panel"],
            num_cols_per_panel=BSexample["num_cols_per_panel"],
            num_rows=BSexample["num_rows"], num_cols=BSexample["num_cols"],
            polarization=BSexample["polarization"],
            polarization_type=BSexample["polarization_type"],
            antenna_pattern=BSexample["antenna_pattern"],
            carrier_frequency=carrier_frequency,
            panel_vertical_spacing=BSexample["panel_vertical_spacing"],
            panel_horizontal_spacing=BSexample["panel_horizontal_spacing"]
        )

        mean_xpr_list = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
        stddev_xpr_list = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}
        mean_xpr = mean_xpr_list["UMa-NLOS"]
        stddev_xpr = stddev_xpr_list["UMa-NLOS"]

        per_ue_buffer_len = max(1000, buffer_len // self.num_ues)
        per_ue_sym_size = max(32, buffer_symbol_size // self.num_ues)

        print(f"[G1A] Initializing {self.num_ues} independent UE channels...")
        print(f"[G1A] Per-UE buffer: len={per_ue_buffer_len}, sym_batch={per_ue_sym_size}")

        Channel_Generator = ChannelCoefficientsGeneratorJIN(carrier_frequency, scs, ArrayTX, ArrayRX, False)

        for ue_k in range(self.num_ues):
            xpr_pdp = 10**(tf.random.normal(
                shape=[batch_size, N_BS, N_UE_sionna, 1, phi_r_rays.shape[-1]],
                mean=mean_xpr, stddev=stddev_xpr
            )/10)

            PDP = Rays(
                delays=tau_rays, powers=power_rays, aoa=phi_r_rays, aod=phi_t_rays,
                zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp
            )

            speed_k = Speed + ue_k * 0.5  # slightly different speed per UE
            velocities = tf.abs(tf.random.normal(
                shape=[batch_size, N_UE_sionna, 3], mean=speed_k, stddev=0.3, dtype=tf.float32))
            moving_end = "rx"
            los_aoa = tf.zeros([batch_size, N_BS, N_UE_sionna])
            los_aod = tf.zeros([batch_size, N_BS, N_UE_sionna])
            los_zoa = tf.zeros([batch_size, N_BS, N_UE_sionna])
            los_zod = tf.zeros([batch_size, N_BS, N_UE_sionna])
            los = tf.random.uniform(
                shape=[batch_size, N_BS, N_UE_sionna], minval=0, maxval=2, dtype=tf.int32) > 0
            dist_k = 1.0 + ue_k * 0.3  # different distance per UE
            distance_3d = tf.constant([[[dist_k]]], dtype=tf.float32)
            tx_orientations = tf.random.normal(
                shape=[batch_size, N_BS, 3], mean=0, stddev=PI/5, dtype=tf.float32)
            rx_orientations = tf.random.normal(
                shape=[batch_size, N_UE_sionna, 3], mean=0, stddev=PI/5, dtype=tf.float32)

            topology = Topology(
                velocities, moving_end, los_aoa, los_aod, los_zoa, los_zod,
                los, distance_3d, tx_orientations, rx_orientations
            )

            h_field_array_power, aoa_delay, zoa_delay = Channel_Generator._H_PDP_FIX(
                topology, PDP, N_FFT, scs)
            h_field_array_power_t = tf.transpose(h_field_array_power, [0, 3, 5, 6, 1, 2, 7, 4])
            aoa_delay_t = tf.transpose(aoa_delay, [0, 3, 1, 2, 4])
            zoa_delay_t = tf.transpose(zoa_delay, [0, 3, 1, 2, 4])

            buf = RingBuffer(
                shape=(FFT_SIZE,),
                dtype=cp.complex64 if GPU_AVAILABLE else np.complex64,
                maxlen=per_ue_buffer_len
            )
            self.channel_buffers.append(buf)

            params = dict(Fs=Fs, scs=scs, N_UE=N_UE_sionna, N_BS=N_BS,
                          N_UE_active=1, N_BS_serving=1)

            producer = ChannelProducer(
                buf, Channel_Generator, topology, params,
                h_field_array_power_t, aoa_delay_t, zoa_delay_t,
                buffer_symbol_size=per_ue_sym_size,
                ue_label=f"UE{ue_k}"
            )
            producer.start()
            self.producers.append(producer)
            print(f"[G1A] UE{ue_k} channel started (dist={dist_k:.1f}, speed={speed_k:.1f})")

        self.gpu_slot_pipeline = GPUSlotPipeline(
            FFT_SIZE, enable_gpu=self.enable_gpu,
            use_pinned_memory=self.use_pinned_memory,
            use_cuda_graph=self.use_cuda_graph,
            profile_interval=self.profile_interval,
            profile_window=self.profile_window,
            dual_timer_compare=self.dual_timer_compare)
        print(f"[G1A] GPU Slot Pipeline initialized ({self.num_ues} UE channels)")

    def _get_channels_for_ue(self, ue_idx, n_sym):
        """Get channel coefficients from UE's buffer, with padding if needed."""
        if ue_idx >= len(self.channel_buffers):
            if GPU_AVAILABLE:
                return cp.ones((N_SYM, FFT_SIZE), dtype=cp.complex64)
            return np.ones((N_SYM, FFT_SIZE), dtype=np.complex64)

        channels = self.channel_buffers[ue_idx].get_batch(n_sym)
        if n_sym < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            pad = lib.ones((N_SYM - n_sym, FFT_SIZE), dtype=channels.dtype)
            channels = lib.concatenate([channels, pad])
        return channels

    # ── Socket mode methods ──

    def connect_gnb(self):
        try:
            s = socket.create_connection((self.gnb_host, self.gnb_port), timeout=5)
            s.setblocking(False)
            self.gnb_ep = Endpoint(s, "gNB")
            self.sel.register(s, selectors.EVENT_READ, data=self.gnb_ep)
            print(f"[INFO] gNB connected {self.gnb_host}:{self.gnb_port}")
        except OSError as e:
            print(f"[WARN] gNB connect fail: {e}")

    def _reconnect_gnb_if_needed(self):
        if self.gnb_ep and not self.gnb_ep.closed:
            return
        if self.gnb_ep:
            try:
                self.sel.unregister(self.gnb_ep.sock)
            except:
                pass
            self.gnb_ep = None
        self.connect_gnb()

    def _accept_ue(self):
        try:
            c, addr = self.lis.accept()
            c.setblocking(False)
        except OSError:
            return
        ue_idx = self._next_ue_idx
        ue = Endpoint(c, "UE", ue_idx=ue_idx)
        fd = ue.fileno()
        if fd in self.ues:
            old = self.ues[fd]
            try: self.sel.unregister(old.sock)
            except: pass
            try: old.sock.close()
            except: pass
        self.ues[fd] = ue
        self.ue_index_map[fd] = ue_idx
        self._next_ue_idx += 1
        self.sel.register(c, selectors.EVENT_READ, data=ue)
        ch_label = f"ch={ue_idx}" if ue_idx < len(self.channel_buffers) else "ch=NONE(>num_ues)"
        print(f"[G1A] UE#{ue_idx} joined {addr} ({ch_label})")
        if self.gnb_hshake:
            h, p = self.gnb_hshake
            ue.send(h, p)

    def _handle_ep(self, ep: Endpoint):
        for hdr_raw, hdr_vals, payload in ep.read_blocks():
            t_blk0 = time.perf_counter()
            size, nb, ts, frame, subframe = hdr_vals
            sample_cnt = size * nb

            if ep.role == "gNB":
                self._handle_gnb_dl(ep, hdr_raw, hdr_vals, payload, t_blk0)
            else:
                self._handle_ue_ul(ep, hdr_raw, hdr_vals, payload, t_blk0)

    def _handle_gnb_dl(self, ep, hdr_raw, hdr_vals, payload, t_blk0):
        """DL: gNB signal -> per-UE channel -> send to each UE.
        Uses batch DL when multiple UEs connected and batch_mode enabled."""
        size, nb, ts, frame, subframe = hdr_vals
        sample_cnt = size * nb

        log("gNB -> Proxy", size, nb, ts, frame, subframe, sample_cnt,
            "(handshake)" if size == 1 else f" ({self.num_ues} UEs)")

        if size == 1:
            self.gnb_hshake = (hdr_raw, payload)
            for u in list(self.ues.values()):
                if not u.closed:
                    u.send(hdr_raw, payload)
            return

        is_channel_slot = (size > 1 and self.ch_en and self.custom_channel)

        n_int16 = len(payload) // 2
        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        n_sym = len(sym_idx)

        active_ues = [(u.ue_idx, u) for u in self.ues.values() if not u.closed]

        # Batch DL path: process all UEs in a single batched FFT
        if (is_channel_slot and self.batch_mode and len(active_ues) > 1
                and all(idx < len(self.channel_buffers) for idx, _ in active_ues)):
            channels_list = [self._get_channels_for_ue(idx, n_sym) for idx, _ in active_ues]
            results = self.gpu_slot_pipeline.process_slot_batch_dl(
                payload, channels_list, pathLossLinear, snr_dB, noise_enabled)
            send_ms = 0.0
            for (_, u), processed in zip(active_ues, results):
                t_send0 = time.perf_counter()
                u.send(hdr_raw, processed)
                send_ms += 1000 * (time.perf_counter() - t_send0)
        else:
            # Sequential per-UE path (fallback)
            send_ms = 0.0
            for ue_idx, u in active_ues:
                if is_channel_slot and ue_idx < len(self.channel_buffers):
                    channels = self._get_channels_for_ue(ue_idx, n_sym)
                    processed = self.gpu_slot_pipeline.process_slot(
                        payload, channels, pathLossLinear, snr_dB, noise_enabled)
                else:
                    processed = payload
                t_send0 = time.perf_counter()
                u.send(hdr_raw, processed)
                send_ms += 1000 * (time.perf_counter() - t_send0)

        if is_channel_slot:
            t_end = time.perf_counter()
            total_ms = 1000 * (t_end - t_blk0)
            proc_ms = total_ms - send_ms
            self.profile_proxy.add(
                tag=f"dir=gNB,ues={len(active_ues)},batch={self.batch_mode}",
                PROC=proc_ms, SEND=send_ms, TOTAL=total_ms)
            self._e2e_slot_count += 1
            self._e2e_proxy_dl_accum_ms += total_ms
            self._e2e_dl_in_frame += 1
            self._check_e2e_frame("Socket+OAI")

    def _handle_ue_ul(self, ep, hdr_raw, hdr_vals, payload, t_blk0):
        """UL: Apply per-UE channel, accumulate, send sum to gNB when all UEs done."""
        size, nb, ts, frame, subframe = hdr_vals
        sample_cnt = size * nb
        ue_idx = ep.ue_idx

        log("UE -> Proxy", size, nb, ts, frame, subframe, sample_cnt,
            f" UE#{ue_idx}")

        if size <= 1 or not self.ch_en or not self.custom_channel:
            if self.gnb_ep and not self.gnb_ep.closed:
                self.gnb_ep.send(hdr_raw, payload)
            return

        n_int16 = len(payload) // 2
        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        n_sym = len(sym_idx)

        if ue_idx < len(self.channel_buffers):
            channels = self._get_channels_for_ue(ue_idx, n_sym)
            # Process through channel and get complex output for accumulation
            processed_bytes = self.gpu_slot_pipeline.process_slot(
                payload, channels, pathLossLinear, snr_dB, noise_enabled)
            # Decode back to complex for accumulation
            proc_int16 = np.frombuffer(processed_bytes, dtype='<i2')
            proc_cpx = proc_int16[::2].astype(np.float64) + 1j * proc_int16[1::2].astype(np.float64)
        else:
            raw_int16 = np.frombuffer(payload, dtype='<i2')
            proc_cpx = raw_int16[::2].astype(np.float64) + 1j * raw_int16[1::2].astype(np.float64)

        active_ue_count = sum(1 for u in self.ues.values() if not u.closed)

        if ts not in self._ul_accum:
            self._ul_accum[ts] = {
                'signal': np.zeros(n_cpx, dtype=np.complex128),
                'ue_set': set(),
                'create_time': time.perf_counter()
            }
            self._ul_accum_hdr[ts] = hdr_raw

        self._ul_accum[ts]['signal'] += proc_cpx
        self._ul_accum[ts]['ue_set'].add(ue_idx)

        all_submitted = (len(self._ul_accum[ts]['ue_set']) >= active_ue_count)

        if all_submitted:
            summed = self._ul_accum[ts]['signal']
            y16 = np.empty(n_cpx * 2, dtype='<i2')
            y16[::2] = np.clip(np.round(summed.real), -32768, 32767).astype('<i2')
            y16[1::2] = np.clip(np.round(summed.imag), -32768, 32767).astype('<i2')

            if self.gnb_ep and not self.gnb_ep.closed:
                self.gnb_ep.send(self._ul_accum_hdr[ts], y16.tobytes())

            del self._ul_accum[ts]
            del self._ul_accum_hdr[ts]

            t_end = time.perf_counter()
            total_ms = 1000 * (t_end - t_blk0)
            self._e2e_slot_count += 1
            self._e2e_proxy_ul_accum_ms += total_ms
            self._e2e_ul_in_frame += 1
            self._check_e2e_frame("Socket+OAI")

        # Garbage-collect stale UL accumulation entries (> 100ms old)
        now = time.perf_counter()
        stale = [k for k, v in self._ul_accum.items()
                 if now - v['create_time'] > 0.1]
        for k in stale:
            del self._ul_accum[k]
            if k in self._ul_accum_hdr:
                del self._ul_accum_hdr[k]

    def _check_e2e_frame(self, overhead_label="Socket+OAI"):
        if self._e2e_slot_count == 0:
            return
        if self._e2e_slot_count % self._e2e_frame_slots != 0:
            return
        now = time.perf_counter()
        if self._e2e_last_wall is not None:
            wall_ms = 1000 * (now - self._e2e_last_wall)
            dl_acc = self._e2e_proxy_dl_accum_ms
            ul_acc = self._e2e_proxy_ul_accum_ms
            proxy_ms = dl_acc + ul_acc
            overhead_ms = wall_ms - proxy_ms
            n = self._e2e_frame_slots
            nd = self._e2e_dl_in_frame
            nu = self._e2e_ul_in_frame
            print(f"\n[E2E frame#{self._e2e_slot_count} "
                  f"({nd}D+{nu}U, {self.num_ues}UE)] "
                  f"wall={wall_ms:.2f}ms  "
                  f"Proxy(DL={dl_acc:.1f}+UL={ul_acc:.1f})"
                  f"={proxy_ms:.2f}ms  "
                  f"{overhead_label}={overhead_ms:.2f}ms  "
                  f"| per slot({n}): "
                  f"wall={wall_ms/n:.2f}  "
                  f"proxy={proxy_ms/n:.2f}  "
                  f"{overhead_label.lower()}={overhead_ms/n:.2f} ms")
            if self.num_ues > 1:
                dl_parts = " ".join(f"UE{k}={self._e2e_dl_per_ue_ms[k]:.1f}"
                                    for k in range(self.num_ues))
                ul_parts = " ".join(f"UE{k}={self._e2e_ul_per_ue_ms[k]:.1f}"
                                    for k in range(self.num_ues))
                print(f"  DL per-UE: [{dl_parts} ms]")
                print(f"  UL per-UE: [{ul_parts} ms]")
        else:
            print(f"\n[E2E frame#{self._e2e_slot_count}] "
                  f"baseline set (comparison starts next frame)")
        self._e2e_last_wall = now
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0
        for k in range(self.num_ues):
            self._e2e_dl_per_ue_ms[k] = 0.0
            self._e2e_ul_per_ue_ms[k] = 0.0

    # ── GPU IPC mode methods ──

    def _warmup_pipeline(self):
        if not self.gpu_slot_pipeline or not self.gpu_slot_pipeline.enable_gpu:
            return
        if not self.channel_buffers:
            return
        print("[G1A IPC] Pre-warming pipeline (XLA compile + CUDA Graph)...")
        t0 = time.time()
        n = self.gpu_slot_pipeline.total_int16
        dummy_in = cp.zeros(n, dtype=cp.int16)
        dummy_out = cp.zeros(n, dtype=cp.int16)
        channels = self.channel_buffers[0].get_batch(N_SYM)
        passes = GPUSlotPipeline.WARMUP_SLOTS + 1
        for i in range(passes):
            self.gpu_slot_pipeline.process_slot_ipc(
                dummy_in, channels, pathLossLinear, snr_dB, noise_enabled, dummy_out
            )
            print(f"  warmup {i+1}/{passes} done ({time.time()-t0:.1f}s)")
        print(f"[G1A IPC] Pipeline ready ({time.time()-t0:.1f}s)")

    def _ipc_process_dl(self):
        """DL: Dequeue from dl_tx ring, apply per-UE channel, enqueue to dl_rx[k] rings."""
        t0 = time.perf_counter()

        dl_tx_ptr, ts, nsamps, nbAnt, data_size = self.ipc.dequeue_dl_tx()
        n_int16 = data_size // 2

        gpu_in = self.ipc.get_gpu_array(dl_tx_ptr, data_size, cp.int16)
        expected_int16 = self.gpu_slot_pipeline.total_int16
        is_full_slot = (n_int16 == expected_int16)
        is_channel = (is_full_slot and self.ch_en and self.custom_channel)

        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx) if is_channel else []
        n_sym = len(sym_idx) if is_channel else 0

        # Batch path: all UEs processed in single batched FFT
        if is_channel and self.batch_mode and self.num_ues > 1:
            channels_list = []
            gpu_outs = []
            for ue_k in range(self.num_ues):
                if ue_k < len(self.channel_buffers):
                    channels_list.append(self._get_channels_for_ue(ue_k, n_sym))
                    rx_ptr = self.ipc.enqueue_dl_rx(ue_k, ts, nsamps, nbAnt, data_size)
                    gpu_outs.append(self.ipc.get_gpu_array(rx_ptr, data_size, cp.int16))
                else:
                    break

            if len(channels_list) == self.num_ues:
                self.gpu_slot_pipeline.process_slot_batch_dl_ipc(
                    gpu_in, channels_list, pathLossLinear, snr_dB, noise_enabled, gpu_outs)
                for ue_k in range(self.num_ues):
                    self.ipc.advance_dl_rx(ue_k)
                self.ipc.advance_dl_tx()
                total_ms = 1000 * (time.perf_counter() - t0)
                per_ue_ms = total_ms / self.num_ues
                for ue_k in range(self.num_ues):
                    self._e2e_dl_per_ue_ms[ue_k] += per_ue_ms
                return total_ms

        # Sequential fallback
        for ue_k in range(self.num_ues):
            t_ue = time.perf_counter()
            rx_ptr = self.ipc.enqueue_dl_rx(ue_k, ts, nsamps, nbAnt, data_size)
            gpu_out = self.ipc.get_gpu_array(rx_ptr, data_size, cp.int16)

            if is_channel and ue_k < len(self.channel_buffers):
                channels = self._get_channels_for_ue(ue_k, n_sym)
                self.gpu_slot_pipeline.process_slot_ipc(
                    gpu_in, channels, pathLossLinear, snr_dB, noise_enabled, gpu_out)
            else:
                gpu_out[:n_int16] = gpu_in[:n_int16]
                cp.cuda.Stream.null.synchronize()

            self.ipc.advance_dl_rx(ue_k)
            self._e2e_dl_per_ue_ms[ue_k] += 1000 * (time.perf_counter() - t_ue)

        self.ipc.advance_dl_tx()
        return 1000 * (time.perf_counter() - t0)

    def _ipc_process_ul(self):
        """UL: Dequeue from ul_tx[k] rings, apply channel, accumulate, enqueue to ul_rx ring."""
        t0 = time.perf_counter()
        total_int16 = self.gpu_slot_pipeline.total_int16
        total_cpx = self.gpu_slot_pipeline.total_cpx

        gpu_accum = cp.zeros(total_cpx, dtype=cp.complex128)
        processed_count = 0

        for ue_k in range(self.num_ues):
            if not self.ipc.ul_tx_available(ue_k):
                continue

            t_ue = time.perf_counter()
            ul_tx_ptr, ts, nsamps, nbAnt, data_size = self.ipc.dequeue_ul_tx(ue_k)
            n_int16 = data_size // 2

            gpu_in = self.ipc.get_gpu_array(ul_tx_ptr, data_size, cp.int16)
            expected_int16 = self.gpu_slot_pipeline.total_int16
            is_full_slot = (n_int16 == expected_int16)

            if is_full_slot and self.ch_en and self.custom_channel and ue_k < len(self.channel_buffers):
                n_cpx = n_int16 // 2
                sym_idx = get_ofdm_symbol_indices(n_cpx)
                n_sym = len(sym_idx)
                channels = self._get_channels_for_ue(ue_k, n_sym)
                cpx_out = self.gpu_slot_pipeline.process_slot_ipc_complex_out(
                    gpu_in, channels, pathLossLinear, snr_dB, noise_enabled)
                if cpx_out is not None:
                    gpu_accum += cpx_out
            else:
                f64 = gpu_in[:n_int16].astype(cp.float64)
                cpx = f64[::2] + 1j * f64[1::2]
                gpu_accum[:len(cpx)] += cpx

            self.ipc.advance_ul_tx(ue_k)
            self._e2e_ul_per_ue_ms[ue_k] += 1000 * (time.perf_counter() - t_ue)
            processed_count += 1

        if processed_count > 0:
            ul_rx_ptr = self.ipc.enqueue_ul_rx(0, total_int16 // 2, 1, total_int16 * 2)
            if ul_rx_ptr is not None:
                gpu_ul_rx = self.ipc.get_gpu_array(ul_rx_ptr, total_int16 * 2, cp.int16)
                self.gpu_slot_pipeline.complex_to_int16_gpu(gpu_accum, gpu_ul_rx)
                self.ipc.advance_ul_rx()

        return 1000 * (time.perf_counter() - t0)

    def run_ipc(self):
        self.ipc = GPUIpcInterface(self.ipc_shm_path, num_ues=self.num_ues)
        if not self.ipc.init():
            print("[ERROR] GPU IPC initialization failed")
            return

        self._warmup_pipeline()
        print(f"[G1A IPC] Entering main loop ({self.num_ues} UEs, "
              f"ring_depth={self.ipc.ring_depth})...")
        dl_count = 0
        ul_count = 0
        idle_count = 0
        t_start = time.time()
        t_last_status = time.time()

        try:
            while True:
                processed = False

                if self.ipc.dl_tx_available():
                    dl_ms = self._ipc_process_dl()
                    dl_count += 1
                    processed = True
                    idle_count = 0

                    self._e2e_proxy_dl_accum_ms += dl_ms
                    self._e2e_dl_in_frame += 1
                    self._e2e_slot_count += 1
                    self._check_e2e_frame("IPC+OAI")

                    if dl_count % 100 == 0:
                        elapsed = time.time() - t_start
                        rate = dl_count / elapsed if elapsed > 0 else 0
                        print(f"[G1A IPC] DL: {dl_count}, UL: {ul_count}, "
                              f"rate: {rate:.1f} DL/sec ({self.num_ues} UEs)")

                any_ul = any(self.ipc.ul_tx_available(k) for k in range(self.num_ues))
                if any_ul:
                    ul_ms = self._ipc_process_ul()
                    ul_count += 1
                    processed = True
                    idle_count = 0

                    self._e2e_proxy_ul_accum_ms += ul_ms
                    self._e2e_ul_in_frame += 1
                    self._e2e_slot_count += 1
                    self._check_e2e_frame("IPC+OAI")

                if not processed:
                    idle_count += 1
                    time.sleep(0.0001)

                now = time.time()
                if now - t_last_status > 3.0:
                    dl_h, dl_t = self.ipc._ring_head(_OFF_DL_TX_RING), self.ipc._ring_tail(_OFF_DL_TX_RING)
                    ul_h, ul_t = self.ipc._ring_head(_OFF_UL_RX_RING), self.ipc._ring_tail(_OFF_UL_RX_RING)
                    dr = [f"dl_rx[{k}] h={self.ipc._ring_head(_OFF_DL_RX_RINGS + k * _RING_CTRL_SIZE)}"
                          f"/t={self.ipc._ring_tail(_OFF_DL_RX_RINGS + k * _RING_CTRL_SIZE)}"
                          for k in range(self.num_ues)]
                    ut = [f"ul_tx[{k}] h={self.ipc._ring_head(_OFF_UL_TX_RINGS + k * _RING_CTRL_SIZE)}"
                          f"/t={self.ipc._ring_tail(_OFF_UL_TX_RINGS + k * _RING_CTRL_SIZE)}"
                          for k in range(self.num_ues)]
                    print(f"[RING DBG] DL={dl_count} UL={ul_count} idle={idle_count} | "
                          f"dl_tx h={dl_h}/t={dl_t} ul_rx h={ul_h}/t={ul_t} | "
                          f"{' '.join(dr)} | {' '.join(ut)}")
                    t_last_status = now

        except KeyboardInterrupt:
            print(f"\n[G1A IPC] Terminated (DL: {dl_count}, UL: {ul_count})")
        finally:
            for p in self.producers:
                p.stop_event.set()
            self.ipc.cleanup()

    def run_socket(self):
        self.connect_gnb()
        try:
            while True:
                for key, _ in self.sel.select(0.5):
                    if key.data == "UE_LIS":
                        self._accept_ue()
                    else:
                        self._handle_ep(key.data)
                self._reconnect_gnb_if_needed()
        except KeyboardInterrupt:
            print("[INFO] terminated by Ctrl-C")
        finally:
            for p in self.producers:
                p.stop_event.set()

    def run(self):
        if self.mode == "gpu-ipc":
            self.run_ipc()
        else:
            self.run_socket()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="G1A Multi-UE Sionna Channel Proxy (Socket + GPU IPC)")

    ap.add_argument("--mode", choices=["socket", "gpu-ipc"], default="socket",
                    help="Communication mode")
    ap.add_argument("--num-ues", type=int, default=2,
                    help=f"Number of UEs (max {GPU_IPC_MAX_UE}, default: 2)")
    ap.add_argument("--ipc-shm-path", default=GPU_IPC_SHM_PATH)

    ap.add_argument("--ue-port", type=int, default=6014)
    ap.add_argument("--gnb-host", default="127.0.0.1")
    ap.add_argument("--gnb-port", type=int, default=6013)
    ap.add_argument("--log", choices=["error", "warn", "info", "debug"], default="info")

    ap.add_argument("--ch-en", dest='ch_en', action="store_true")
    ap.add_argument("--no-ch-en", dest='ch_en', action="store_false")
    ap.set_defaults(ch_en=True)
    ap.add_argument("--ch-dd", type=int, default=0)
    ap.add_argument("--ch-L", type=int, default=32)
    ap.add_argument("--log-plot", action="store_true", default=False)
    ap.add_argument("--conv-mode", type=str, default="fft", choices=["fft", "oa", "os"])
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--fft-lib", type=str, default="np", choices=["np", "tf"])

    ap.add_argument("--custom-channel", dest='custom_channel', action="store_true")
    ap.add_argument("--no-custom-channel", dest='custom_channel', action="store_false")
    ap.set_defaults(custom_channel=True)
    ap.add_argument("--buffer-len", type=int, default=42000)
    ap.add_argument("--buffer-symbol-size", type=int, default=4200)

    ap.add_argument("--enable-gpu", dest='enable_gpu', action="store_true")
    ap.add_argument("--disable-gpu", dest='enable_gpu', action="store_false")
    ap.set_defaults(enable_gpu=True)
    ap.add_argument("--use-pinned-memory", dest='use_pinned_memory', action="store_true")
    ap.add_argument("--no-pinned-memory", dest='use_pinned_memory', action="store_false")
    ap.set_defaults(use_pinned_memory=True)
    ap.add_argument("--use-cuda-graph", dest='use_cuda_graph', action="store_true")
    ap.add_argument("--no-cuda-graph", dest='use_cuda_graph', action="store_false")
    ap.set_defaults(use_cuda_graph=True)

    ap.add_argument("--path-loss-dB", type=float, default=0.0)
    ap.add_argument("--snr-dB", type=float, default=None)

    ap.add_argument("--profile-interval", type=int, default=100)
    ap.add_argument("--profile-window", type=int, default=500)
    ap.add_argument("--dual-timer-compare", dest='dual_timer_compare', action="store_true")
    ap.add_argument("--no-dual-timer-compare", dest='dual_timer_compare', action="store_false")
    ap.set_defaults(dual_timer_compare=True)

    ap.add_argument("--batch-mode", dest='batch_mode', action="store_true",
                    help="Enable batched FFT processing for multi-UE (default)")
    ap.add_argument("--no-batch-mode", dest='batch_mode', action="store_false",
                    help="Disable batched FFT (sequential per-UE processing)")
    ap.set_defaults(batch_mode=True)

    args = ap.parse_args()

    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled
    path_loss_dB = args.path_loss_dB
    pathLossLinear = 10**(path_loss_dB / 20.0)
    snr_dB = args.snr_dB
    noise_enabled = (snr_dB is not None)

    num_ues = min(args.num_ues, GPU_IPC_MAX_UE)

    print("=" * 80)
    print("G1A: Multi-UE Sionna Channel Proxy (Socket + GPU IPC)")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Number of UEs: {num_ues}")
    if args.mode == "gpu-ipc":
        print(f"IPC SHM Path: {args.ipc_shm_path}")
    else:
        print(f"UE Port: {args.ue_port}, gNB: {args.gnb_host}:{args.gnb_port}")
    print(f"GPU: {'Enabled' if args.enable_gpu and GPU_AVAILABLE else 'Disabled'}")
    print(f"CUDA Graph: {'Enabled' if args.use_cuda_graph else 'Disabled'}")
    print(f"Precision: complex128")
    print(f"Custom Channel: {'Enabled' if args.custom_channel else 'Disabled'}")
    print(f"Path Loss: {path_loss_dB} dB (linear={pathLossLinear:.6f})")
    if noise_enabled:
        print(f"AWGN Noise: Enabled (SNR={snr_dB} dB)")
    else:
        print(f"AWGN Noise: Disabled")
    print("=" * 80)

    print("\n[G1A Features]")
    print(f"  + Multi-UE: {num_ues} independent Sionna channels")
    print(f"  + DL: per-UE channel applied (gNB -> N different faded signals)")
    print(f"  + UL: per-UE channel + summation (N UE signals -> gNB)")
    print(f"  + TDD reciprocity: DL/UL share channel coefficients per UE")
    if args.enable_gpu and GPU_AVAILABLE:
        print("  + Full GPU Pipeline (int16 -> GPU -> int16)")
        print("  + TF->CuPy DLPack (GPU-to-GPU channel transfer)")
        print("  + GPU RingBuffer (per-UE)")
        print("  + CuPy Batch FFT (14 symbols)")
        print("  + complex128 precision (PSS/SSS stability)")
        if args.use_cuda_graph:
            print(f"  + CUDA Graph (warmup {GPUSlotPipeline.WARMUP_SLOTS} slots)")
        if args.mode == "gpu-ipc":
            print(f"  + CUDA IPC multi-UE ({2 + 2*num_ues} GPU buffers)")
        if args.batch_mode:
            print(f"  + Batch FFT: {num_ues} UEs x {N_SYM} symbols = {num_ues * N_SYM} batch dim")
    print(f"Batch Mode: {'Enabled' if args.batch_mode else 'Disabled'}")
    print("=" * 80)
    print()

    Proxy(
        mode=args.mode,
        num_ues=num_ues,
        ue_port=args.ue_port, gnb_host=args.gnb_host, gnb_port=args.gnb_port,
        log_level=args.log, ch_en=args.ch_en, ch_dd=args.ch_dd, ch_L=args.ch_L,
        log_plot=args.log_plot, conv_mode=args.conv_mode, block_size=args.block_size,
        num_blocks=args.num_blocks, fft_lib=args.fft_lib,
        custom_channel=args.custom_channel,
        buffer_len=args.buffer_len, buffer_symbol_size=args.buffer_symbol_size,
        enable_gpu=args.enable_gpu, use_pinned_memory=args.use_pinned_memory,
        use_cuda_graph=args.use_cuda_graph,
        ipc_shm_path=args.ipc_shm_path,
        profile_interval=args.profile_interval,
        profile_window=args.profile_window,
        dual_timer_compare=args.dual_timer_compare,
        batch_mode=args.batch_mode,
    ).run()


if __name__ == "__main__":
    main()
