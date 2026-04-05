"""
================================================================================
v2.py - G1C Multi-UE MIMO Channel Proxy (v1 + multiprocessing ChannelProducer)

[v1 → v2 변경사항]
  1. ChannelProducer: threading.Thread → multiprocessing.Process
     - 프로세스별 독립 TF GPU 컨텍스트 → 2UE+ 환경에서 TF 경합 크래시 해소
  2. RingBuffer → IPCRingBuffer (크로스-프로세스):
     - IPCRingBufferSync: mp.Value + mp.Condition (부모-자식 공유)
     - IPCRingBufferProducer: child 프로세스에서 GPU 버퍼 할당 + CUDA IPC handle 전달
     - IPCRingBufferConsumer: 부모 프로세스에서 ipcOpenMemHandle로 GPU 버퍼 접근
  3. spawn start method: fork 시 CUDA 컨텍스트 상속 문제 방지
  4. TF 메모리 성장: 프로세스별 set_memory_growth(True) → OOM 방지
  5. 크로스-프로세스 GPU 동기화: put_batch 후 device sync 1회

[변경 없음 — v1과 동일]
  - IPC V7 + futex, UL fused RawKernel, CUDA Graph
  - NoiseProducer (threading.Thread — CuPy only, TF 미사용)
  - Proxy 메인 루프, GPUSlotPipeline, GPUIpcV7Interface

[아키텍처]
  ChannelProducerProcess[k] (독립 프로세스: TF+CuPy)
    → IPCRingBufferProducer.put_batch() → CUDA IPC → IPCRingBufferConsumer
  Proxy Main Process (CuPy + CUDA Graph):
    → IPCRingBufferConsumer.get_batch_view() → GPUSlotPipeline
    → gNB/UE IPC V7 polling loop
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
import ctypes
import ctypes.util
import mmap
import multiprocessing as _mp
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import logging
import threading
import json
import time

_mp_ctx = _mp.get_context('spawn')
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

SIONNA_API_IP = "127.0.0.1"
SIONNA_API_PORT = 7000

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
carrier_frequency = 3.5e9
FFT_SIZE = 2048
N_FFT = FFT_SIZE
CP0 = 176
CP1 = 144
N_SYM = 14
SYMBOL_SIZES = [CP0 + FFT_SIZE] + [CP1 + FFT_SIZE] * 13
scs = 30*1e3
Fs = FFT_SIZE*scs

path_loss_dB = 0
pathLossLinear = 10**(path_loss_dB / 20.0)
snr_dB = None
noise_dBFS = None
noise_mode = "none"
noise_enabled = False
noise_std_abs = None
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
# WindowProfiler (ported from v11)
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
# GPU IPC Interface
# ============================================================================

GPU_IPC_V6_SHM_PATH = "/tmp/oai_gpu_ipc/gpu_ipc_shm"
GPU_IPC_V6_MAGIC = 0x47505537
GPU_IPC_V6_VERSION = 1
GPU_IPC_V6_HANDLE_SIZE = 64
GPU_IPC_V6_SHM_SIZE = 4096
GPU_IPC_V6_CIR_TIME = 460800
GPU_IPC_V6_SAMPLE_SIZE = 4

GPU_IPC_V7_SHM_PATH = "/tmp/oai_gpu_ipc/gpu_ipc_shm"
GPU_IPC_V7_MAGIC = 0x47505538
GPU_IPC_V7_VERSION = 1
GPU_IPC_V7_HANDLE_SIZE = 64
GPU_IPC_V7_SHM_SIZE = 4096
GPU_IPC_V7_CIR_TIME = 460800
GPU_IPC_V7_SAMPLE_SIZE = 4
GPU_IPC_V7_OFF_DL_TX_SEQ = 368
GPU_IPC_V7_OFF_DL_RX_SEQ = 372
GPU_IPC_V7_OFF_UL_TX_SEQ = 376
GPU_IPC_V7_OFF_UL_RX_SEQ = 380

import sys as _sys
SYS_futex = 202 if _sys.maxsize > 2**32 else 240
FUTEX_WAKE = 1
_libc = ctypes.CDLL("libc.so.6", use_errno=True)

GPU_IPC_SHM_PATH = GPU_IPC_V7_SHM_PATH

GNB_ANT = int(os.environ.get('GPU_IPC_V5_GNB_ANT', '1'))
UE_ANT = int(os.environ.get('GPU_IPC_V5_UE_ANT', '1'))
GNB_NX = int(os.environ.get('GPU_IPC_V5_GNB_NX', '1'))
GNB_NY = int(os.environ.get('GPU_IPC_V5_GNB_NY', '1'))
UE_NX = int(os.environ.get('GPU_IPC_V5_UE_NX', '1'))
UE_NY = int(os.environ.get('GPU_IPC_V5_UE_NY', '1'))


class GPUIpcV6Interface:
    """
    GPU IPC V6 Per-Buffer Antenna interface — SERVER role (Proxy).

    Each of the 4 buffers has its own nbAnt and cir_size, enabling
    asymmetric MIMO (gNB and UE with different antenna counts).

    Buffer mapping:
      dl_tx: gNB writes (nbAnt=GNB_ANT) → Proxy reads
      dl_rx: Proxy writes → UE reads (nbAnt=UE_ANT)
      ul_tx: UE writes (nbAnt=UE_ANT) → Proxy reads
      ul_rx: Proxy writes → gNB reads (nbAnt=GNB_ANT)
    """

    def __init__(self, gnb_ant=1, ue_ant=1, cir_time=GPU_IPC_V6_CIR_TIME,
                 shm_path=GPU_IPC_V6_SHM_PATH):
        self.shm_path = shm_path
        self.shm_fd = None
        self.shm_mm = None
        self.gpu_dl_tx_ptr = 0
        self.gpu_dl_rx_ptr = 0
        self.gpu_ul_tx_ptr = 0
        self.gpu_ul_rx_ptr = 0
        self._gpu_mem = []
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.cir_time = cir_time
        self.dl_tx_nbAnt = gnb_ant
        self.dl_tx_cir_size = cir_time * gnb_ant
        self.dl_rx_nbAnt = ue_ant
        self.dl_rx_cir_size = cir_time * ue_ant
        self.ul_tx_nbAnt = ue_ant
        self.ul_tx_cir_size = cir_time * ue_ant
        self.ul_rx_nbAnt = gnb_ant
        self.ul_rx_cir_size = cir_time * gnb_ant
        self.initialized = False

    def init(self):
        """Allocate 4 GPU circular buffers with per-buffer sizes."""
        shm_dir = os.path.dirname(self.shm_path)
        os.makedirs(shm_dir, mode=0o777, exist_ok=True)

        self.shm_fd = os.open(self.shm_path,
                              os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
        os.ftruncate(self.shm_fd, GPU_IPC_V6_SHM_SIZE)
        self.shm_mm = mmap.mmap(self.shm_fd, GPU_IPC_V6_SHM_SIZE)
        self.shm_mm[:] = b'\x00' * GPU_IPC_V6_SHM_SIZE

        buf_configs = [
            ('dl_tx', 0,   self.dl_tx_cir_size),
            ('dl_rx', 64,  self.dl_rx_cir_size),
            ('ul_tx', 128, self.ul_tx_cir_size),
            ('ul_rx', 192, self.ul_rx_cir_size),
        ]
        ptrs = []
        for name, handle_off, cir_sz in buf_configs:
            buf_bytes = cir_sz * GPU_IPC_V6_SAMPLE_SIZE
            mem = cp.cuda.alloc(buf_bytes)
            self._gpu_mem.append(mem)
            ptr = mem.ptr
            ptrs.append(ptr)
            cp.cuda.runtime.memset(ptr, 0, buf_bytes)
            handle_bytes = cp.cuda.runtime.ipcGetMemHandle(ptr)
            self.shm_mm[handle_off:handle_off + GPU_IPC_V6_HANDLE_SIZE] = handle_bytes
            print(f"[GPU IPC V6] SERVER: allocated {name} "
                  f"({buf_bytes} bytes, cir_size={cir_sz}, ptr=0x{ptr:x})")

        self.gpu_dl_tx_ptr = ptrs[0]
        self.gpu_dl_rx_ptr = ptrs[1]
        self.gpu_ul_tx_ptr = ptrs[2]
        self.gpu_ul_rx_ptr = ptrs[3]

        struct.pack_into('<I', self.shm_mm, 264, self.cir_time)
        struct.pack_into('<I', self.shm_mm, 268, 1)
        struct.pack_into('<I', self.shm_mm, 272, self.dl_tx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 276, self.dl_tx_cir_size)
        struct.pack_into('<I', self.shm_mm, 280, self.dl_rx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 284, self.dl_rx_cir_size)
        struct.pack_into('<I', self.shm_mm, 288, self.ul_tx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 292, self.ul_tx_cir_size)
        struct.pack_into('<I', self.shm_mm, 296, self.ul_rx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 300, self.ul_rx_cir_size)
        struct.pack_into('<I', self.shm_mm, 260, GPU_IPC_V6_VERSION)
        self.shm_mm.flush()
        struct.pack_into('<I', self.shm_mm, 256, GPU_IPC_V6_MAGIC)
        self.shm_mm.flush()

        self.initialized = True
        print(f"[GPU IPC V6] SERVER: ready (magic=0x{GPU_IPC_V6_MAGIC:08X}, "
              f"version={GPU_IPC_V6_VERSION}, gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant}, "
              f"cir_time={self.cir_time})")
        return True

    def circ_offset(self, ts, nbAnt, cir_size):
        return int((ts * nbAnt) % cir_size)

    def get_gpu_array_at(self, base_ptr, ts, nsamps, nbAnt, cir_size, dtype=cp.int16):
        off = self.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        elem_size = dtype().itemsize
        if off + total <= cir_size:
            byte_off = off * GPU_IPC_V6_SAMPLE_SIZE
            n_elem = (total * GPU_IPC_V6_SAMPLE_SIZE) // elem_size
            mem = cp.cuda.UnownedMemory(base_ptr + byte_off,
                                        total * GPU_IPC_V6_SAMPLE_SIZE, owner=None)
            return cp.ndarray(n_elem, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0)), False
        else:
            return None, True

    def gpu_circ_copy(self, dst_ptr, src_ptr, ts, nsamps, nbAnt, cir_size):
        off = self.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sample_sz = GPU_IPC_V6_SAMPLE_SIZE
        if off + total <= cir_size:
            n_int16 = (total * sample_sz) // 2
            dst = self._make_gpu_array(dst_ptr, off * sample_sz, n_int16)
            src = self._make_gpu_array(src_ptr, off * sample_sz, n_int16)
            dst[:] = src[:]
        else:
            tail = cir_size - off
            head = total - tail
            tail_n = (tail * sample_sz) // 2
            head_n = (head * sample_sz) // 2
            dst_t = self._make_gpu_array(dst_ptr, off * sample_sz, tail_n)
            src_t = self._make_gpu_array(src_ptr, off * sample_sz, tail_n)
            dst_t[:] = src_t[:]
            dst_h = self._make_gpu_array(dst_ptr, 0, head_n)
            src_h = self._make_gpu_array(src_ptr, 0, head_n)
            dst_h[:] = src_h[:]
        cp.cuda.Stream.null.synchronize()

    def bypass_copy(self, dst_ptr, src_ptr, ts, nsamps,
                    src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size):
        """Bypass copy with antenna mapping. Symmetric=direct copy, asymmetric=truncate/pad."""
        if src_nbAnt == dst_nbAnt:
            self.gpu_circ_copy(dst_ptr, src_ptr, ts, nsamps, src_nbAnt, src_cir_size)
        else:
            src_off = self.circ_offset(ts, src_nbAnt, src_cir_size)
            src_total = nsamps * src_nbAnt
            sample_sz = GPU_IPC_V6_SAMPLE_SIZE
            min_ant = min(src_nbAnt, dst_nbAnt)
            if src_off + src_total <= src_cir_size:
                src_arr = self._make_gpu_array(src_ptr, src_off * sample_sz,
                                               (src_total * sample_sz) // 2)
                src_2d = src_arr.view(cp.int16).reshape(nsamps, src_nbAnt * 2)
            else:
                tail = src_cir_size - src_off
                head = src_total - tail
                tail_arr = self._make_gpu_array(src_ptr, src_off * sample_sz, (tail * sample_sz) // 2)
                head_arr = self._make_gpu_array(src_ptr, 0, (head * sample_sz) // 2)
                src_arr = cp.concatenate([tail_arr, head_arr])
                src_2d = src_arr.view(cp.int16).reshape(nsamps, src_nbAnt * 2)

            dst_2d = cp.zeros((nsamps, dst_nbAnt * 2), dtype=cp.int16)
            dst_2d[:, :min_ant * 2] = src_2d[:, :min_ant * 2]
            dst_flat = dst_2d.ravel()

            dst_off = self.circ_offset(ts, dst_nbAnt, dst_cir_size)
            dst_total = nsamps * dst_nbAnt
            if dst_off + dst_total <= dst_cir_size:
                dst_arr = self._make_gpu_array(dst_ptr, dst_off * sample_sz,
                                               (dst_total * sample_sz) // 2)
                dst_arr[:] = dst_flat
            else:
                tail_d = dst_cir_size - dst_off
                head_d = dst_total - tail_d
                dst_t = self._make_gpu_array(dst_ptr, dst_off * sample_sz, (tail_d * sample_sz) // 2)
                dst_t[:] = dst_flat[:(tail_d * sample_sz) // 2]
                dst_h = self._make_gpu_array(dst_ptr, 0, (head_d * sample_sz) // 2)
                dst_h[:] = dst_flat[(tail_d * sample_sz) // 2:]
            cp.cuda.Stream.null.synchronize()

    def _make_gpu_array(self, base_ptr, byte_offset, n_int16):
        mem = cp.cuda.UnownedMemory(base_ptr + byte_offset,
                                    n_int16 * 2, owner=None)
        return cp.ndarray(n_int16, dtype=cp.int16,
                          memptr=cp.cuda.MemoryPointer(mem, 0))

    def read_shm_field(self, offset, fmt):
        return struct.unpack_from(fmt, self.shm_mm, offset)[0]

    def write_shm_field(self, offset, fmt, value):
        struct.pack_into(fmt, self.shm_mm, offset, value)
        self.shm_mm.flush()

    def get_last_dl_tx_ts(self):
        return self.read_shm_field(304, '<Q')

    def get_last_dl_tx_nsamps(self):
        return self.read_shm_field(312, '<I')

    def set_last_dl_rx_ts(self, ts):
        self.write_shm_field(320, '<Q', ts)

    def get_last_ul_tx_ts(self):
        return self.read_shm_field(328, '<Q')

    def get_last_ul_tx_nsamps(self):
        return self.read_shm_field(336, '<I')

    def set_last_ul_rx_ts(self, ts):
        self.write_shm_field(344, '<Q', ts)

    def cleanup(self):
        if not self.initialized:
            return
        self._gpu_mem.clear()
        self.gpu_dl_tx_ptr = 0
        self.gpu_dl_rx_ptr = 0
        self.gpu_ul_tx_ptr = 0
        self.gpu_ul_rx_ptr = 0
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
        print("[GPU IPC V6] Cleanup done")


class GPUIpcV7Interface(GPUIpcV6Interface):
    """GPU IPC V7 — V6 + futex sequence counters for wake notification.
    Proxy only calls futex_wake (never futex_wait)."""

    def __init__(self, gnb_ant=1, ue_ant=1, cir_time=GPU_IPC_V7_CIR_TIME,
                 shm_path=GPU_IPC_V7_SHM_PATH):
        super().__init__(gnb_ant=gnb_ant, ue_ant=ue_ant,
                         cir_time=cir_time, shm_path=shm_path)

    def init(self):
        shm_dir = os.path.dirname(self.shm_path)
        os.makedirs(shm_dir, mode=0o777, exist_ok=True)

        self.shm_fd = os.open(self.shm_path,
                              os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
        os.ftruncate(self.shm_fd, GPU_IPC_V7_SHM_SIZE)
        self.shm_mm = mmap.mmap(self.shm_fd, GPU_IPC_V7_SHM_SIZE)
        self.shm_mm[:] = b'\x00' * GPU_IPC_V7_SHM_SIZE

        buf_configs = [
            ('dl_tx', 0,   self.dl_tx_cir_size),
            ('dl_rx', 64,  self.dl_rx_cir_size),
            ('ul_tx', 128, self.ul_tx_cir_size),
            ('ul_rx', 192, self.ul_rx_cir_size),
        ]
        ptrs = []
        for name, handle_off, cir_sz in buf_configs:
            buf_bytes = cir_sz * GPU_IPC_V7_SAMPLE_SIZE
            mem = cp.cuda.alloc(buf_bytes)
            self._gpu_mem.append(mem)
            ptr = mem.ptr
            ptrs.append(ptr)
            cp.cuda.runtime.memset(ptr, 0, buf_bytes)
            handle_bytes = cp.cuda.runtime.ipcGetMemHandle(ptr)
            self.shm_mm[handle_off:handle_off + GPU_IPC_V7_HANDLE_SIZE] = handle_bytes
            print(f"[GPU IPC V7] SERVER: allocated {name} "
                  f"({buf_bytes} bytes, cir_size={cir_sz}, ptr=0x{ptr:x})")

        self.gpu_dl_tx_ptr = ptrs[0]
        self.gpu_dl_rx_ptr = ptrs[1]
        self.gpu_ul_tx_ptr = ptrs[2]
        self.gpu_ul_rx_ptr = ptrs[3]

        struct.pack_into('<I', self.shm_mm, 264, self.cir_time)
        struct.pack_into('<I', self.shm_mm, 268, 1)
        struct.pack_into('<I', self.shm_mm, 272, self.dl_tx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 276, self.dl_tx_cir_size)
        struct.pack_into('<I', self.shm_mm, 280, self.dl_rx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 284, self.dl_rx_cir_size)
        struct.pack_into('<I', self.shm_mm, 288, self.ul_tx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 292, self.ul_tx_cir_size)
        struct.pack_into('<I', self.shm_mm, 296, self.ul_rx_nbAnt)
        struct.pack_into('<I', self.shm_mm, 300, self.ul_rx_cir_size)
        # Zero out seq counters
        for off in (GPU_IPC_V7_OFF_DL_TX_SEQ, GPU_IPC_V7_OFF_DL_RX_SEQ,
                    GPU_IPC_V7_OFF_UL_TX_SEQ, GPU_IPC_V7_OFF_UL_RX_SEQ):
            struct.pack_into('<I', self.shm_mm, off, 0)
        struct.pack_into('<I', self.shm_mm, 260, GPU_IPC_V7_VERSION)
        self.shm_mm.flush()
        struct.pack_into('<I', self.shm_mm, 256, GPU_IPC_V7_MAGIC)
        self.shm_mm.flush()

        self.initialized = True
        print(f"[GPU IPC V7] SERVER: ready (magic=0x{GPU_IPC_V7_MAGIC:08X}, "
              f"version={GPU_IPC_V7_VERSION}, gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant}, "
              f"cir_time={self.cir_time}, futex=enabled)")
        return True

    def _futex_wake(self, seq_offset):
        """Increment seq counter in SHM and issue futex WAKE."""
        cur = struct.unpack_from('<I', self.shm_mm, seq_offset)[0]
        struct.pack_into('<I', self.shm_mm, seq_offset, cur + 1)
        self.shm_mm.flush()
        addr = ctypes.c_void_p(ctypes.addressof(
            ctypes.c_char.from_buffer(self.shm_mm, seq_offset)))
        _libc.syscall(ctypes.c_long(SYS_futex),
                      addr, ctypes.c_int(FUTEX_WAKE),
                      ctypes.c_int(1),
                      ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0))

    def set_last_dl_rx_ts(self, ts):
        """GPU write must be complete before calling this."""
        self.write_shm_field(320, '<Q', ts)
        self._futex_wake(GPU_IPC_V7_OFF_DL_RX_SEQ)

    def set_last_ul_rx_ts(self, ts):
        """GPU write must be complete before calling this."""
        self.write_shm_field(344, '<Q', ts)
        self._futex_wake(GPU_IPC_V7_OFF_UL_RX_SEQ)


# ============================================================================
# Fused clip+cast RawKernel for UL superposition output
# ============================================================================
if GPU_AVAILABLE:
    _fused_clip_cast_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void fused_clip_cast(
        const double* accum_f64,
        short* out,
        int n_elem)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elem) return;
        double r = round(accum_f64[2 * idx]);
        double i = round(accum_f64[2 * idx + 1]);
        r = fmin(fmax(r, -32768.0), 32767.0);
        i = fmin(fmax(i, -32768.0), 32767.0);
        out[2 * idx]     = (short)r;
        out[2 * idx + 1] = (short)i;
    }
    ''', 'fused_clip_cast')


# ============================================================================
# GPU Slot Pipeline (from v10, with IPC extensions)
# ============================================================================

class GPUSlotPipeline:
    """
    GPU Full Pipeline: v10 CUDA Graph + v12 GPU IPC mode

    Socket mode: int16 bytes in → GPU process → int16 bytes out (v10 behavior)
    IPC mode:    GPU int16 in → GPU process → GPU int16 out (zero-copy)
    """
    WARMUP_SLOTS = 3

    def __init__(self, fft_size=2048, enable_gpu=True, use_pinned_memory=True,
                 use_cuda_graph=True, profile_interval=100, profile_window=500,
                 dual_timer_compare=True, n_tx_in=1, n_rx_out=1,
                 noise_buffer=None):
        self.fft_size = fft_size
        self.n_tx = n_tx_in
        self.n_rx = n_rx_out
        self.noise_buffer = noise_buffer
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
        self.total_int16_in = self.total_cpx * n_tx_in * 2
        self.total_int16_out = self.total_cpx * n_rx_out * 2
        self.total_int16 = self.total_int16_in

        if not self.enable_gpu:
            print("[GPU Pipeline] GPU disabled - CPU numpy mode")
            return

        print(f"[GPU Pipeline v6] MIMO CUDA Graph Pipeline + Dual Noise initializing (n_tx={n_tx_in}, n_rx={n_rx_out})...")
        print(f"[GPU Pipeline v6] Precision: complex128 (float64, PSS stability)")
        print(f"[GPU Pipeline v6] Profiling: interval={self.profile_interval}, "
              f"window={self.profile_window}, dual_timer={'ON' if self.dual_timer_compare else 'OFF'}")

        self.stream = cp.cuda.Stream(non_blocking=True)

        self.sym_bounds = get_ofdm_symbol_indices(self.total_cpx)

        if self.use_pinned_memory:
            self.pinned_iq_in_buf = cp.cuda.alloc_pinned_memory(self.total_int16_in * 2)
            self.pinned_iq_out_buf = cp.cuda.alloc_pinned_memory(self.total_int16_out * 2)
            self.pinned_iq_in = np.frombuffer(self.pinned_iq_in_buf, dtype=np.int16,
                                              count=self.total_int16_in)
            self.pinned_iq_out = np.frombuffer(self.pinned_iq_out_buf, dtype=np.int16,
                                               count=self.total_int16_out)

        self.gpu_iq_in = cp.zeros(self.total_int16_in, dtype=cp.int16)
        self.gpu_iq_out = cp.zeros(self.total_int16_out, dtype=cp.int16)
        self.gpu_x = cp.zeros((self.n_sym, n_tx_in, fft_size), dtype=cp.complex128)
        self.gpu_H = cp.zeros((self.n_sym, n_rx_out, n_tx_in, fft_size), dtype=cp.complex128)
        self.gpu_out = cp.zeros(self.total_cpx * n_rx_out, dtype=cp.complex128)
        self.gpu_noise_r = cp.zeros(self.total_cpx * n_rx_out, dtype=cp.float64)
        self.gpu_noise_i = cp.zeros(self.total_cpx * n_rx_out, dtype=cp.float64)

        self._buf_HX = cp.zeros((self.n_sym, n_rx_out, n_tx_in, fft_size), dtype=cp.complex128)
        self._buf_Yf = cp.zeros((self.n_sym, n_rx_out, fft_size), dtype=cp.complex128)
        self._buf_out_2d = cp.zeros((self.total_cpx, n_rx_out), dtype=cp.complex128)
        self._buf_iq_out_3d = cp.zeros((self.total_cpx, n_rx_out, 2), dtype=cp.float64)

        ext_idx = cp.zeros((N_SYM, fft_size), dtype=cp.int64)
        data_dst = []
        cp_dst_list = []
        cp_src_list = []
        for i, (s, e) in enumerate(self.sym_bounds):
            cp_l = CP0 if i == 0 else CP1
            ext_idx[i] = cp.arange(s + cp_l, e)
            data_dst.append(cp.arange(s + cp_l, e, dtype=cp.int64))
            cp_dst_list.append(cp.arange(s, s + cp_l, dtype=cp.int64))
            cp_src_list.append(cp.arange(
                i * fft_size + fft_size - cp_l,
                i * fft_size + fft_size, dtype=cp.int64))
        self.gpu_ext_idx = ext_idx
        self.gpu_data_dst = cp.concatenate(data_dst)
        self.gpu_cp_dst = cp.concatenate(cp_dst_list)
        self.gpu_cp_src = cp.concatenate(cp_src_list)

        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph = None
        self.graph_captured = False
        self.warmup_count = 0
        self._graph_pl_linear = None
        self._graph_noise_on = None
        self._graph_snr_db = None
        self._graph_noise_std_abs = None

        print(f"[GPU Pipeline v6] Initialization complete (n_tx={self.n_tx}, n_rx={self.n_rx})")

    def _regenerate_noise(self):
        if self.noise_buffer is not None:
            noise_view, n_held = self.noise_buffer.get_batch_view(1)
            self.gpu_noise_r[:] = noise_view[0, 0]
            self.gpu_noise_i[:] = noise_view[0, 1]
            self.noise_buffer.release_batch(n_held)
        else:
            n = self.total_cpx * self.n_rx
            self.gpu_noise_r[:] = cp.random.randn(n).astype(cp.float64)
            self.gpu_noise_i[:] = cp.random.randn(n).astype(cp.float64)

    def _gpu_compute_core(self, pl_linear, snr_db, noise_on, noise_std_abs=None):
        """MIMO channel application: de-interleave → broadcast*sum → re-interleave.
        All ops are CUDA Graph capturable (no cuBLAS, no dynamic alloc, no Python loops).
        gpu_H is already in frequency domain (FFT pre-computed by ChannelProducer).
        noise_std_abs: pre-computed absolute noise std (cp.float64) for dBFS mode, None for relative SNR."""
        n_tx = self.n_tx
        n_rx = self.n_rx

        # 1. De-interleave: flat int16 (sample-major, ant-minor) → complex per antenna
        self._tmp_f64 = self.gpu_iq_in.astype(cp.float64)
        self._tmp_iq_3d = self._tmp_f64.reshape(self.total_cpx, n_tx, 2)
        self._tmp_cpx_2d = self._tmp_iq_3d[:, :, 0] + 1j * self._tmp_iq_3d[:, :, 1]

        # 2. OFDM symbol extraction — GPU index array (no Python for-loop)
        self.gpu_x[:] = self._tmp_cpx_2d[self.gpu_ext_idx].transpose(0, 2, 1)

        # 3. FFT(signal) + broadcast multiply with Hf + sum (Hf pre-computed, no cuBLAS)
        self._tmp_Xf = cp.fft.fft(self.gpu_x, axis=-1)
        cp.multiply(self.gpu_H, self._tmp_Xf[:, cp.newaxis, :, :], out=self._buf_HX)
        cp.sum(self._buf_HX, axis=2, out=self._buf_Yf)
        self._tmp_y = cp.fft.ifft(self._buf_Yf, axis=-1)

        # 4. Reconstruct OFDM — GPU index scatter (no Python for-loop)
        self._buf_out_2d[:] = 0
        self._tmp_y_t = self._tmp_y.transpose(0, 2, 1)
        self._tmp_y_flat = self._tmp_y_t.reshape(-1, n_rx)
        self._buf_out_2d[self.gpu_data_dst] = self._tmp_y_flat
        self._buf_out_2d[self.gpu_cp_dst] = self._tmp_y_flat[self.gpu_cp_src]

        # 5. Path Loss + AWGN
        self.gpu_out[:] = self._buf_out_2d.ravel()
        if pl_linear != 1.0:
            self.gpu_out *= cp.float64(pl_linear)

        if noise_on:
            if noise_std_abs is not None:
                self._tmp_noise = noise_std_abs * (
                    self.gpu_noise_r + 1j * self.gpu_noise_i
                )
            else:
                self._tmp_abs_sq = cp.abs(self.gpu_out) ** 2
                self._tmp_sig_pwr = cp.mean(self._tmp_abs_sq)
                snr_linear = cp.float64(10.0 ** (snr_db / 10.0))
                self._tmp_n_pwr = self._tmp_sig_pwr / snr_linear
                self._tmp_n_std = cp.sqrt(self._tmp_n_pwr / cp.float64(2.0))
                self._tmp_noise = self._tmp_n_std * (
                    self.gpu_noise_r + 1j * self.gpu_noise_i
                )
            self.gpu_out += self._tmp_noise

        # 6. Re-interleave: (total_cpx, n_rx) → flat int16 (pre-allocated buffer)
        self._tmp_out_2d_final = self.gpu_out.reshape(self.total_cpx, n_rx)
        self._buf_iq_out_3d[:, :, 0] = cp.clip(cp.around(self._tmp_out_2d_final.real), -32768, 32767)
        self._buf_iq_out_3d[:, :, 1] = cp.clip(cp.around(self._tmp_out_2d_final.imag), -32768, 32767)
        self.gpu_iq_out[:] = self._buf_iq_out_3d.ravel().astype(cp.int16)

    def _try_capture_graph(self, pl_linear, snr_db, noise_on, noise_std_abs=None):
        try:
            self.stream.begin_capture()
            self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)
            self.cuda_graph = self.stream.end_capture()

            self.graph_captured = True
            self._graph_pl_linear = pl_linear
            self._graph_noise_on = noise_on
            self._graph_snr_db = snr_db
            self._graph_noise_std_abs = noise_std_abs

            _noise_info = "OFF"
            if noise_on:
                _noise_info = f"abs={noise_std_abs}" if noise_std_abs is not None else f"SNR={snr_db}dB"
            print(f"[CUDA Graph] Capture success (PL={pl_linear}, noise={_noise_info})")

        except Exception as e:
            try:
                self.stream.end_capture()
            except:
                pass
            try:
                cp.cuda.Device(0).synchronize()
            except:
                pass
            print(f"[CUDA Graph] Capture failed - fallback to normal: {e}")
            self.graph_captured = False
            self.use_cuda_graph = False

    def _need_recapture(self, pl_linear, snr_db, noise_on, noise_std_abs=None):
        if not self.graph_captured:
            return False
        return (self._graph_pl_linear != pl_linear or
                self._graph_noise_on != noise_on or
                self._graph_snr_db != snr_db or
                self._graph_noise_std_abs != noise_std_abs)

    def process_slot(self, iq_bytes, channels_gpu, pl_linear, snr_db, noise_on, noise_std_abs=None):
        """Socket mode: raw bytes in -> int16 bytes out (v10 compatible)"""
        n_iq = len(iq_bytes) // 2
        n_cpx = n_iq // 2

        if not self.enable_gpu or n_cpx != self.total_cpx:
            return self._cpu_fallback(iq_bytes, channels_gpu, pl_linear, snr_db, noise_on)

        do_profile = (self.slot_counter > 0
                      and self.slot_counter % self.profile_interval == 0)
        do_dual = do_profile and self.dual_timer_compare
        if do_profile:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        if do_dual:
            e_h2d_s, e_h2d_e = cp.cuda.Event(), cp.cuda.Event()
            e_ch_s, e_ch_e = cp.cuda.Event(), cp.cuda.Event()
            e_noise_s, e_noise_e = cp.cuda.Event(), cp.cuda.Event()
            e_gpu_s, e_gpu_e = cp.cuda.Event(), cp.cuda.Event()
            e_d2h_s, e_d2h_e = cp.cuda.Event(), cp.cuda.Event()

        with self.stream:
            if do_dual:
                e_h2d_s.record(self.stream)
            if self.use_pinned_memory:
                ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
                self.gpu_iq_in.set(self.pinned_iq_in, stream=self.stream)
            else:
                iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
                self.gpu_iq_in[:] = cp.asarray(iq_int16)
            if do_dual:
                e_h2d_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t1 = time.perf_counter()

            if do_dual:
                e_ch_s.record(self.stream)
            n_ch = min(channels_gpu.shape[0], self.n_sym)
            if n_ch < self.n_sym:
                self.gpu_H[n_ch:] = 0
            if channels_gpu.ndim == 4:
                n_w = min(channels_gpu.shape[3], self.fft_size)
                ch_slice = channels_gpu[:n_ch, :self.n_rx, :self.n_tx, :n_w]
            elif channels_gpu.ndim == 2:
                n_w = min(channels_gpu.shape[1], self.fft_size)
                ch_slice = channels_gpu[:n_ch, :n_w].reshape(n_ch, 1, 1, n_w)
            else:
                n_w = self.fft_size
                ch_slice = channels_gpu[:n_ch].reshape(n_ch, 1, 1, -1)[:, :, :, :n_w]
            self.gpu_H[:n_ch, :ch_slice.shape[1], :ch_slice.shape[2], :n_w] = ch_slice
            if do_dual:
                e_ch_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t2 = time.perf_counter()

            if do_dual:
                e_noise_s.record(self.stream)
            if noise_on:
                self._regenerate_noise()
            if do_dual:
                e_noise_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t_noise1 = time.perf_counter()

            if self._need_recapture(pl_linear, snr_db, noise_on, noise_std_abs):
                self.graph_captured = False
                self.warmup_count = self.WARMUP_SLOTS

            if do_dual:
                e_gpu_s.record(self.stream)
            if self.graph_captured:
                self.cuda_graph.launch(self.stream)
            elif self.use_cuda_graph and self.warmup_count >= self.WARMUP_SLOTS:
                self._try_capture_graph(pl_linear, snr_db, noise_on, noise_std_abs)
                if self.graph_captured:
                    self.cuda_graph.launch(self.stream)
                else:
                    self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)
            else:
                self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)
                self.warmup_count += 1
            if do_dual:
                e_gpu_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t3 = time.perf_counter()

            if do_dual:
                e_d2h_s.record(self.stream)
            if self.use_pinned_memory:
                self.gpu_iq_out.get(out=self.pinned_iq_out, stream=self.stream)
                if do_dual:
                    e_d2h_e.record(self.stream)
                self.stream.synchronize()
                result = self.pinned_iq_out.tobytes()
            else:
                out_host = self.gpu_iq_out.get(stream=self.stream)
                if do_dual:
                    e_d2h_e.record(self.stream)
                self.stream.synchronize()
                result = out_host.tobytes()

            if do_profile:
                t4 = time.perf_counter()

        self.slot_counter += 1
        if do_profile:
            mode = "GRAPH" if self.graph_captured else "NORMAL"
            cpu_h2d = 1000 * (t1 - t0)
            cpu_ch = 1000 * (t2 - t1)
            cpu_noise = 1000 * (t_noise1 - t2)
            cpu_gpu = 1000 * (t3 - t_noise1)
            cpu_d2h = 1000 * (t4 - t3)
            cpu_total = 1000 * (t4 - t0)
            self.profile_gpu.add(
                tag=f"mode={mode}",
                H2D=cpu_h2d, CH_COPY=cpu_ch, NOISE_PREP=cpu_noise,
                GPU_COMPUTE=cpu_gpu, D2H=cpu_d2h, TOTAL=cpu_total)
            if do_dual:
                evt_h2d = cp.cuda.get_elapsed_time(e_h2d_s, e_h2d_e)
                evt_ch = cp.cuda.get_elapsed_time(e_ch_s, e_ch_e)
                evt_noise = cp.cuda.get_elapsed_time(e_noise_s, e_noise_e)
                evt_gpu = cp.cuda.get_elapsed_time(e_gpu_s, e_gpu_e)
                evt_d2h = cp.cuda.get_elapsed_time(e_d2h_s, e_d2h_e)
                evt_total = evt_h2d + evt_ch + evt_noise + evt_gpu + evt_d2h
                self.profile_gpu_evt.add(
                    tag=f"mode={mode}",
                    H2D=evt_h2d, CH_COPY=evt_ch, NOISE_PREP=evt_noise,
                    GPU_COMPUTE=evt_gpu, D2H=evt_d2h, TOTAL=evt_total)
                self.profile_gpu_diff.add(
                    tag=f"mode={mode}",
                    H2D=cpu_h2d - evt_h2d, CH_COPY=cpu_ch - evt_ch,
                    NOISE_PREP=cpu_noise - evt_noise,
                    GPU_COMPUTE=cpu_gpu - evt_gpu,
                    D2H=cpu_d2h - evt_d2h, TOTAL=cpu_total - evt_total)
        return result

    def process_slot_ipc(self, gpu_iq_in_arr, channels_gpu, pl_linear, snr_db, noise_on, gpu_iq_out_arr, noise_std_abs=None):
        """
        GPU IPC mode: GPU array in -> GPU process -> GPU array out
        MIMO: channels_gpu shape = (N_SYM, N_r, N_t, FFT) or (N_SYM, FFT) for SISO compat.
        """
        if not self.enable_gpu:
            return

        do_profile = (self.slot_counter > 0
                      and self.slot_counter % self.profile_interval == 0)
        do_dual = do_profile and self.dual_timer_compare
        if do_profile:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        if do_dual:
            e_cin_s, e_cin_e = cp.cuda.Event(), cp.cuda.Event()
            e_ch_s, e_ch_e = cp.cuda.Event(), cp.cuda.Event()
            e_noise_s, e_noise_e = cp.cuda.Event(), cp.cuda.Event()
            e_gpu_s, e_gpu_e = cp.cuda.Event(), cp.cuda.Event()
            e_cout_s, e_cout_e = cp.cuda.Event(), cp.cuda.Event()

        with self.stream:
            if do_dual:
                e_cin_s.record(self.stream)
            self.gpu_iq_in[:] = gpu_iq_in_arr[:self.total_int16_in]
            if do_dual:
                e_cin_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t1 = time.perf_counter()

            if do_dual:
                e_ch_s.record(self.stream)
            n_ch = min(channels_gpu.shape[0], self.n_sym)
            if n_ch < self.n_sym:
                self.gpu_H[n_ch:] = 0
            if channels_gpu.ndim == 4:
                n_w = min(channels_gpu.shape[3], self.fft_size)
                ch_slice = channels_gpu[:n_ch, :self.n_rx, :self.n_tx, :n_w]
            elif channels_gpu.ndim == 2:
                n_w = min(channels_gpu.shape[1], self.fft_size)
                ch_slice = channels_gpu[:n_ch, :n_w].reshape(n_ch, 1, 1, n_w)
            else:
                n_w = self.fft_size
                ch_slice = channels_gpu[:n_ch].reshape(n_ch, 1, 1, -1)[:, :, :, :n_w]
            self.gpu_H[:n_ch, :ch_slice.shape[1], :ch_slice.shape[2], :n_w] = ch_slice
            if do_dual:
                e_ch_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t2 = time.perf_counter()

            if do_dual:
                e_noise_s.record(self.stream)
            if noise_on:
                self._regenerate_noise()
            if do_dual:
                e_noise_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t_noise1 = time.perf_counter()

            if self._need_recapture(pl_linear, snr_db, noise_on, noise_std_abs):
                self.graph_captured = False
                self.warmup_count = self.WARMUP_SLOTS

            if do_dual:
                e_gpu_s.record(self.stream)
            if self.graph_captured:
                self.cuda_graph.launch(self.stream)
            elif self.use_cuda_graph and self.warmup_count >= self.WARMUP_SLOTS:
                self._try_capture_graph(pl_linear, snr_db, noise_on, noise_std_abs)
                if self.graph_captured:
                    self.cuda_graph.launch(self.stream)
                else:
                    self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)
            else:
                self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)
                self.warmup_count += 1
            if do_dual:
                e_gpu_e.record(self.stream)

            if do_profile:
                self.stream.synchronize(); t3 = time.perf_counter()

            if do_dual:
                e_cout_s.record(self.stream)
            gpu_iq_out_arr[:self.total_int16_out] = self.gpu_iq_out[:]
            if do_dual:
                e_cout_e.record(self.stream)
            self.stream.synchronize()

            if do_profile:
                t4 = time.perf_counter()

        self.slot_counter += 1
        if do_profile:
            mode = "GRAPH" if self.graph_captured else "NORMAL"
            cpu_cin = 1000 * (t1 - t0)
            cpu_ch = 1000 * (t2 - t1)
            cpu_noise = 1000 * (t_noise1 - t2)
            cpu_gpu = 1000 * (t3 - t_noise1)
            cpu_cout = 1000 * (t4 - t3)
            cpu_total = 1000 * (t4 - t0)
            self.profile_ipc.add(
                tag=f"mode={mode}",
                GPU_COPY_IN=cpu_cin, CH_COPY=cpu_ch, NOISE_PREP=cpu_noise,
                GPU_COMPUTE=cpu_gpu, GPU_COPY_OUT=cpu_cout, TOTAL=cpu_total)
            if do_dual:
                evt_cin = cp.cuda.get_elapsed_time(e_cin_s, e_cin_e)
                evt_ch = cp.cuda.get_elapsed_time(e_ch_s, e_ch_e)
                evt_noise = cp.cuda.get_elapsed_time(e_noise_s, e_noise_e)
                evt_gpu = cp.cuda.get_elapsed_time(e_gpu_s, e_gpu_e)
                evt_cout = cp.cuda.get_elapsed_time(e_cout_s, e_cout_e)
                evt_total = evt_cin + evt_ch + evt_noise + evt_gpu + evt_cout
                self.profile_ipc_evt.add(
                    tag=f"mode={mode}",
                    GPU_COPY_IN=evt_cin, CH_COPY=evt_ch, NOISE_PREP=evt_noise,
                    GPU_COMPUTE=evt_gpu, GPU_COPY_OUT=evt_cout, TOTAL=evt_total)
                self.profile_ipc_diff.add(
                    tag=f"mode={mode}",
                    GPU_COPY_IN=cpu_cin - evt_cin, CH_COPY=cpu_ch - evt_ch,
                    NOISE_PREP=cpu_noise - evt_noise,
                    GPU_COMPUTE=cpu_gpu - evt_gpu,
                    GPU_COPY_OUT=cpu_cout - evt_cout,
                    TOTAL=cpu_total - evt_total)

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
            cp_l = CP0 if i == 0 else CP1
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
# Socket-mode support classes (v10 backward compatibility)
# ============================================================================

@dataclass
class Endpoint:
    sock:  socket.socket
    role:  str
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
# RingBuffer (threading, for NoiseProducer — unchanged from v1)
# ============================================================================

class RingBuffer:
    def __init__(self, shape, dtype=cp.complex128, maxlen=1024):
        if GPU_AVAILABLE:
            self.buffer = cp.zeros((maxlen,) + shape, dtype=dtype)
            self.is_gpu = True
        else:
            _dtype_map = {
                cp.complex128: np.complex128, np.complex128: np.complex128,
                cp.complex64: np.complex64, np.complex64: np.complex64,
                cp.float64: np.float64, np.float64: np.float64,
                cp.float32: np.float32, np.float32: np.float32,
            }
            np_dtype = _dtype_map.get(dtype, np.complex128)
            self.buffer = np.zeros((maxlen,) + shape, dtype=np_dtype)
            self.is_gpu = False
        self.maxlen = maxlen
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, data):
        with self.not_full:
            while self.count == self.maxlen:
                self.not_full.wait()
            self.buffer[self.write_idx] = data
            self.write_idx = (self.write_idx + 1) % self.maxlen
            self.count += 1
            self.not_empty.notify()

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

    def get(self):
        with self.not_empty:
            while self.count == 0:
                self.not_empty.wait()
            data = self.buffer[self.read_idx].copy()
            self.read_idx = (self.read_idx + 1) % self.maxlen
            self.count -= 1
            self.not_full.notify()
        return data

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

    def get_batch_view(self, n):
        """Return a view (no copy) of n entries. count is NOT decremented,
        so the Producer cannot overwrite this region until release_batch()."""
        with self.not_empty:
            while self.count < n:
                self.not_empty.wait()
            end = self.read_idx + n
            if end <= self.maxlen:
                batch = self.buffer[self.read_idx:end]
            else:
                lib = cp if self.is_gpu else np
                batch = lib.concatenate([
                    self.buffer[self.read_idx:],
                    self.buffer[:end - self.maxlen]
                ])
            self.read_idx = end % self.maxlen
        return batch, n

    def release_batch(self, n):
        """Call after finishing with the view from get_batch_view().
        Decrements count and notifies the Producer."""
        with self.not_full:
            self.count -= n
            self.not_full.notify_all()


# ============================================================================
# IPCRingBuffer — Cross-process GPU ring buffer via CUDA IPC
# ============================================================================

class IPCRingBufferSync:
    """Shared synchronization primitives for cross-process ring buffer.
    Created in the main process; passed to child via mp.Process args."""

    def __init__(self, maxlen, ctx=None):
        c = ctx or _mp_ctx
        self.write_idx = c.Value('i', 0)
        self.read_idx = c.Value('i', 0)
        self.count = c.Value('i', 0)
        self.maxlen = maxlen
        self.lock = c.Lock()
        self.not_empty = c.Condition(self.lock)
        self.not_full = c.Condition(self.lock)


class IPCRingBufferProducer:
    """Producer side — lives inside ChannelProducerProcess.
    Allocates GPU buffer and exposes CUDA IPC handle."""

    def __init__(self, shape, dtype, sync: IPCRingBufferSync):
        import cupy as _cp
        self.buffer = _cp.zeros((sync.maxlen,) + shape, dtype=dtype)
        self.ipc_handle = _cp.cuda.runtime.ipcGetMemHandle(self.buffer.data.ptr)
        self.sync = sync
        self.nbytes = self.buffer.nbytes

    def put_batch(self, data_batch):
        import cupy as _cp
        n = data_batch.shape[0]
        s = self.sync
        with s.not_full:
            for i in range(n):
                while s.count.value == s.maxlen:
                    s.not_full.wait()
                self.buffer[s.write_idx.value] = data_batch[i]
                s.write_idx.value = (s.write_idx.value + 1) % s.maxlen
                s.count.value += 1
            _cp.cuda.Device(0).synchronize()
            s.not_empty.notify_all()


class IPCRingBufferConsumer:
    """Consumer side — lives in the main (Proxy) process.
    Opens CUDA IPC handle to access Producer's GPU buffer."""

    def __init__(self, ipc_handle, shape, dtype, sync: IPCRingBufferSync):
        ptr = cp.cuda.runtime.ipcOpenMemHandle(ipc_handle)
        full_shape = (sync.maxlen,) + shape
        nbytes = int(np.prod(full_shape)) * cp.dtype(dtype).itemsize
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=None)
        self.buffer = cp.ndarray(
            full_shape, dtype=dtype,
            memptr=cp.cuda.MemoryPointer(mem, 0))
        self.sync = sync
        self._ipc_ptr = ptr

    def get_batch_view(self, n, timeout=30.0):
        """Return a view of n entries. Count is NOT decremented until release_batch().
        Returns (batch, n_held) or raises TimeoutError."""
        s = self.sync
        with s.not_empty:
            deadline = time.monotonic() + timeout
            while s.count.value < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"IPCRingBufferConsumer.get_batch_view timed out "
                        f"(need={n}, have={s.count.value})")
                s.not_empty.wait(timeout=remaining)
            end = s.read_idx.value + n
            if end <= s.maxlen:
                batch = self.buffer[s.read_idx.value:end]
            else:
                batch = cp.concatenate([
                    self.buffer[s.read_idx.value:],
                    self.buffer[:end - s.maxlen]])
            s.read_idx.value = end % s.maxlen
        return batch, n

    def release_batch(self, n):
        s = self.sync
        with s.not_full:
            s.count.value -= n
            s.not_full.notify_all()

    def cleanup(self):
        try:
            cp.cuda.runtime.ipcCloseMemHandle(self._ipc_ptr)
        except Exception:
            pass


# ============================================================================
# ChannelProducerProcess — independent TF context per UE
# ============================================================================

class ChannelProducerProcess(_mp_ctx.Process):
    """Each instance runs in a separate process with its own TF/CUDA context.
    Initializes Sionna, allocates GPU ring buffer, and generates channel coefficients."""

    def __init__(self, ue_idx, config, sync, handle_queue, stop_event):
        super().__init__(daemon=True)
        self.ue_idx = ue_idx
        self.config = config
        self.sync = sync
        self.handle_queue = handle_queue
        self.stop_event = stop_event

    def run(self):
        import os as _os
        _os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.get('gpu_num', 0))
        _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

        import tensorflow as _tf
        for _gpu in _tf.config.list_physical_devices('GPU'):
            _tf.config.experimental.set_memory_growth(_gpu, True)

        import cupy as _cp
        _cp.cuda.Device(0).use()

        from sionna.phy import PI, SPEED_OF_LIGHT
        from sionna.phy.channel.tr38901 import PanelArray, Topology, Rays
        from channel_coefficients_JIN import (
            ChannelCoefficientsGeneratorJIN,
            random_binary_mask_tf_complex64,
        )
        import numpy as _np

        cfg = self.config
        ue_idx = self.ue_idx

        npy_dir = cfg['npy_directory']
        phi_r_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/phi_r_rays_for_ChannelBlock.npy"))
        phi_t_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/phi_t_rays_for_ChannelBlock.npy"))
        theta_r_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/theta_r_rays_for_ChannelBlock.npy"))
        theta_t_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/theta_t_rays_for_ChannelBlock.npy"))
        power_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/power_rays_for_ChannelBlock.npy"))
        tau_rays = _tf.convert_to_tensor(_np.load(npy_dir + "/tau_rays_for_ChannelBlock.npy"))

        batch_size = 1
        N_UE = cfg['N_UE']
        N_BS = cfg['N_BS']
        N_FFT_local = cfg['N_FFT']
        scs_local = cfg['scs']
        Fs_local = cfg['Fs']
        carrier_freq = cfg['carrier_frequency']
        buffer_symbol_size = cfg['buffer_symbol_size']
        gnb_nx, gnb_ny = cfg['gnb_nx'], cfg['gnb_ny']
        ue_nx, ue_ny = cfg['ue_nx'], cfg['ue_ny']
        Speed_local = cfg['Speed']
        mean_xpr = cfg['mean_xpr']
        stddev_xpr = cfg['stddev_xpr']

        BSexample = {
            "num_rows_per_panel": gnb_ny, "num_cols_per_panel": gnb_nx,
            "num_rows": 1, "num_cols": 1,
            "polarization": "single", "polarization_type": "V",
        }
        ArrayTX = PanelArray(
            num_rows_per_panel=BSexample["num_rows_per_panel"],
            num_cols_per_panel=BSexample["num_cols_per_panel"],
            num_rows=BSexample["num_rows"], num_cols=BSexample["num_cols"],
            polarization=BSexample["polarization"],
            polarization_type=BSexample["polarization_type"],
            antenna_pattern='omni',
            carrier_frequency=carrier_freq)
        ArrayRX = PanelArray(
            num_rows_per_panel=ue_ny, num_cols_per_panel=ue_nx,
            num_rows=1, num_cols=1,
            polarization='single', polarization_type='V', antenna_pattern='omni',
            carrier_frequency=carrier_freq)

        print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] Sionna init start")

        xpr_pdp = 10**(_tf.random.normal(
            shape=[batch_size, N_BS, N_UE, 1, phi_r_rays.shape[-1]],
            mean=mean_xpr, stddev=stddev_xpr
        )/10)
        PDP = Rays(
            delays=tau_rays, powers=power_rays, aoa=phi_r_rays, aod=phi_t_rays,
            zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp)

        velocities = _tf.abs(_tf.random.normal(
            shape=[batch_size, N_UE, 3], mean=Speed_local, stddev=0.1, dtype=_tf.float32))
        los_aoa = _tf.zeros([batch_size, N_BS, N_UE])
        los_aod = _tf.zeros([batch_size, N_BS, N_UE])
        los_zoa = _tf.zeros([batch_size, N_BS, N_UE])
        los_zod = _tf.zeros([batch_size, N_BS, N_UE])
        los = _tf.random.uniform(
            shape=[batch_size, N_BS, N_UE], minval=0, maxval=2, dtype=_tf.int32) > 0
        distance_3d = _tf.ones([1, N_BS, N_UE])
        tx_orientations = _tf.random.normal(
            shape=[batch_size, N_BS, 3], mean=0, stddev=PI/5, dtype=_tf.float32)
        rx_orientations = _tf.random.normal(
            shape=[batch_size, N_UE, 3], mean=0, stddev=PI/5, dtype=_tf.float32)

        topology_k = Topology(
            velocities, "rx", los_aoa, los_aod, los_zoa, los_zod,
            los, distance_3d, tx_orientations, rx_orientations)

        gen_k = ChannelCoefficientsGeneratorJIN(
            carrier_freq, scs_local, ArrayTX, ArrayRX, False)
        h_field_k, aoa_k, zoa_k = gen_k._H_PDP_FIX(topology_k, PDP, N_FFT_local, scs_local)
        h_field_k = _tf.transpose(h_field_k, [0, 3, 5, 6, 1, 2, 7, 4])
        aoa_k = _tf.transpose(aoa_k, [0, 3, 1, 2, 4])
        zoa_k = _tf.transpose(zoa_k, [0, 3, 1, 2, 4])

        print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] Sionna init done, allocating GPU ring buffer")

        shape = cfg['shape']
        rb = IPCRingBufferProducer(shape, _cp.complex128, self.sync)
        self.handle_queue.put(rb.ipc_handle)
        print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] IPC handle sent, entering generation loop")

        params = dict(Fs=Fs_local, scs=scs_local,
                      N_UE=N_UE, N_BS=N_BS,
                      N_UE_active=cfg['num_rx'], N_BS_serving=cfg['num_tx'])
        symbol_counter = 0

        while not self.stop_event.is_set():
            try:
                sample_times = _tf.cast(
                    _tf.range(buffer_symbol_size), gen_k.rdtype
                ) / _tf.constant(params['scs'], gen_k.rdtype)
                ActiveUE = _tf.constant(
                    random_binary_mask_tf_complex64(params['N_UE'], k=params['N_UE_active']),
                    dtype=_tf.complex64)
                ServingBS = _tf.constant(
                    random_binary_mask_tf_complex64(params['N_BS'], k=params['N_BS_serving']),
                    dtype=_tf.complex64)
                h_delay, _, _, _ = gen_k._H_TTI_sequential_fft_o_ELW2_noProfile(
                    topology_k, ActiveUE, ServingBS, sample_times,
                    h_field_k, aoa_k, zoa_k)
                h_delay = h_delay[0, :, :, 0, 0, :, :]
                h_delay = _tf.transpose(h_delay, [2, 0, 1, 3])

                h_c128 = _tf.cast(h_delay, _tf.complex128)
                energy = _tf.reduce_sum(_tf.abs(h_c128) ** 2, axis=2, keepdims=True)
                h_norm = h_delay / _tf.cast(_tf.sqrt(energy + 1e-30), h_delay.dtype)
                h_c128_norm = _tf.cast(h_norm, _tf.complex128)

                try:
                    h_cp_batch = _cp.from_dlpack(
                        _tf.experimental.dlpack.to_dlpack(h_c128_norm)).copy()
                except Exception:
                    h_cp_batch = _cp.asarray(h_c128_norm.numpy())

                h_cp_batch = _cp.fft.fft(h_cp_batch, axis=-1)
                rb.put_batch(h_cp_batch)
                symbol_counter += buffer_symbol_size

            except _tf.errors.ResourceExhaustedError as e:
                print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] GPU OOM — stopping: {e}")
                break
            except Exception as e:
                print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] ERROR in generation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        print(f"[v2 ChannelProducerProcess UE[{ue_idx}]] Stopped (symbols={symbol_counter})")


class NoiseProducer(threading.Thread):
    """Pre-generate AWGN noise vectors in a background thread.
    Each entry: (2, noise_len) float64 — [0]=real, [1]=imag, standard normal."""
    BATCH_SIZE = 64

    def __init__(self, buffer, noise_len):
        super().__init__()
        self.buffer = buffer
        self.noise_len = noise_len
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        if GPU_AVAILABLE:
            cp.cuda.Device(0).use()
        while not self.stop_event.is_set():
            batch = cp.random.randn(self.BATCH_SIZE, 2, self.noise_len).astype(cp.float64)
            self.buffer.put_batch(batch)


# ============================================================================
# Proxy (dual-mode: socket / gpu-ipc)
# ============================================================================

class Proxy:
    def __init__(self, mode="socket", ue_port=6018, gnb_host="127.0.0.1", gnb_port=6017,
                 log_level="info", ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np",
                 custom_channel=False, buffer_len=1024, buffer_symbol_size=42,
                 enable_gpu=True, use_pinned_memory=True, use_cuda_graph=True,
                 ipc_shm_path=GPU_IPC_SHM_PATH,
                 profile_interval=100, profile_window=500, dual_timer_compare=True,
                 gnb_ant=1, ue_ant=1,
                 gnb_nx=1, gnb_ny=1, ue_nx=1, ue_ny=1,
                 num_ues=1):
        self.mode = mode
        self.num_ues = num_ues
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.gnb_nx = gnb_nx
        self.gnb_ny = gnb_ny
        self.ue_nx = ue_nx
        self.ue_ny = ue_ny
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

        # GPU IPC interface (only used in gpu-ipc mode)
        self.ipc = None

        # Proxy-level profilers (OFDM wrapper + E2E)
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

        # E2E TDD frame statistics
        self._e2e_slot_count = 0
        self._e2e_frame_slots = 10
        self._e2e_next_frame_boundary = 10
        self._e2e_last_wall = None
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0
        self._e2e_dl_per_ue = [0] * num_ues
        self._e2e_ul_per_ue = [0] * num_ues

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
            self.gnb_hshake: Optional[Tuple[bytes, bytes]] = None
        else:
            print(f"[INFO] GPU IPC mode: shm_path={ipc_shm_path}")

        self._init_channel(buffer_len, buffer_symbol_size)

    def _init_channel(self, buffer_len, buffer_symbol_size):
        """Initialize per-UE channel producer processes and GPU pipelines (Multi-UE MIMO v2).
        Channel generation runs in independent processes (multiprocessing.Process).
        Pipelines and noise producers remain in the main process."""
        pipeline_common = dict(
            enable_gpu=self.enable_gpu,
            use_pinned_memory=self.use_pinned_memory,
            use_cuda_graph=self.use_cuda_graph,
            profile_interval=self.profile_interval,
            profile_window=self.profile_window,
            dual_timer_compare=self.dual_timer_compare)

        N = self.num_ues
        self.noise_producers = []
        self._noise_buffers_dl_list = []
        self._noise_buffers_ul_list = []
        self.pipelines_dl = []
        self.pipelines_ul = []
        self.channel_buffers = []
        self.channel_producers = []
        self._channel_stop_events = []

        total_cpx = sum(SYMBOL_SIZES)
        noise_len_dl = total_cpx * self.ue_ant
        noise_len_ul = total_cpx * self.gnb_ant

        for k in range(N):
            nb_dl = None
            nb_ul = None
            if noise_enabled and GPU_AVAILABLE:
                nb_dl = RingBuffer(shape=(2, noise_len_dl), dtype=cp.float64, maxlen=256)
                np_dl = NoiseProducer(nb_dl, noise_len_dl)
                self.noise_producers.append(np_dl)
                nb_ul = RingBuffer(shape=(2, noise_len_ul), dtype=cp.float64, maxlen=256)
                np_ul = NoiseProducer(nb_ul, noise_len_ul)
                self.noise_producers.append(np_ul)
            self._noise_buffers_dl_list.append(nb_dl)
            self._noise_buffers_ul_list.append(nb_ul)

            pdl = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=self.gnb_ant, n_rx_out=self.ue_ant,
                noise_buffer=nb_dl, **pipeline_common)
            pul = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=self.ue_ant, n_rx_out=self.gnb_ant,
                noise_buffer=nb_ul, **pipeline_common)
            self.pipelines_dl.append(pdl)
            self.pipelines_ul.append(pul)
            print(f"[v2] UE[{k}] pipelines created (DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx)")

        self.pipeline_dl = self.pipelines_dl[0]
        self.pipeline_ul = self.pipelines_ul[0]
        self.gpu_slot_pipeline = self.pipeline_dl
        self._noise_buffers_dl = self._noise_buffers_dl_list[0]
        self._noise_buffers_ul = self._noise_buffers_ul_list[0]

        if not self.custom_channel:
            print(f"[v2] Bypass mode — {N} UE(s), no channel")
            return

        self.N_UE = 1
        self.N_BS = 1
        self.num_rx = 1
        self.num_tx = 1

        mean_xpr_list = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
        stddev_xpr_list = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}

        print(f"[v2] Starting {N} ChannelProducerProcess(es) via spawn...")

        for k in range(N):
            sync_k = IPCRingBufferSync(maxlen=buffer_len, ctx=_mp_ctx)
            handle_q_k = _mp_ctx.Queue()
            stop_k = _mp_ctx.Event()

            config_k = {
                'gnb_nx': self.gnb_nx, 'gnb_ny': self.gnb_ny,
                'ue_nx': self.ue_nx, 'ue_ny': self.ue_ny,
                'carrier_frequency': carrier_frequency,
                'scs': scs, 'N_FFT': N_FFT, 'Fs': Fs,
                'buffer_symbol_size': buffer_symbol_size,
                'buffer_len': buffer_len,
                'npy_directory': directory,
                'shape': (self.ue_ant, self.gnb_ant, FFT_SIZE),
                'N_UE': self.N_UE, 'N_BS': self.N_BS,
                'num_rx': self.num_rx, 'num_tx': self.num_tx,
                'Speed': Speed,
                'mean_xpr': mean_xpr_list["UMa-NLOS"],
                'stddev_xpr': stddev_xpr_list["UMa-NLOS"],
                'gpu_num': gpu_num,
            }

            proc_k = ChannelProducerProcess(k, config_k, sync_k, handle_q_k, stop_k)
            proc_k.start()
            print(f"[v2] UE[{k}] ChannelProducerProcess started (pid={proc_k.pid})")

            try:
                ipc_handle = handle_q_k.get(timeout=120)
            except Exception as e:
                print(f"[v2] ERROR: UE[{k}] IPC handle not received within 120s: {e}")
                if proc_k.is_alive():
                    proc_k.terminate()
                raise RuntimeError(f"ChannelProducerProcess[{k}] failed to initialize") from e

            consumer_k = IPCRingBufferConsumer(
                ipc_handle,
                config_k['shape'],
                cp.complex128,
                sync_k)

            self.channel_buffers.append(consumer_k)
            self.channel_producers.append(proc_k)
            self._channel_stop_events.append(stop_k)
            print(f"[v2] UE[{k}] IPCRingBufferConsumer connected (CUDA IPC)")

        self.channel_buffer = self.channel_buffers[0]
        self.producer = self.channel_producers[0]

        print(f"[v2] Multi-UE Channel Proxy initialized: {N} UE(s), "
              f"DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx")

    # ── Socket mode methods (v10 compatible) ──

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
        ue = Endpoint(c, "UE")
        fd = ue.fileno()
        if fd in self.ues:
            old = self.ues[fd]
            try: self.sel.unregister(old.sock)
            except: pass
            try: old.sock.close()
            except: pass
        self.ues[fd] = ue
        self.sel.register(c, selectors.EVENT_READ, data=ue)
        print(f"[INFO] UE joined {addr}")
        if self.gnb_hshake:
            h, p = self.gnb_hshake
            ue.send(h, p)

    def _handle_ep(self, ep: Endpoint):
        for hdr_raw, hdr_vals, payload in ep.read_blocks():
            t_blk0 = time.perf_counter()
            size, nb, ts, frame, subframe = hdr_vals
            sample_cnt = size * nb

            if size > 1 and self.ch_en and self.custom_channel:
                processed = self._process_ofdm_slot(payload, ts)
                ch_note = " (GPU slot pipeline v12)"
            else:
                processed = payload
                ch_note = ""

            t_proc1 = time.perf_counter()
            send_ms = 0.0

            if ep.role == "gNB":
                log("gNB -> Proxy", size, nb, ts, frame, subframe, sample_cnt,
                    "(handshake)" if size == 1 else ch_note)
                if size == 1:
                    self.gnb_hshake = (hdr_raw, payload)
                for u in list(self.ues.values()):
                    if u.closed:
                        continue
                    t_send0 = time.perf_counter()
                    u.send(hdr_raw, processed)
                    send_ms += 1000 * (time.perf_counter() - t_send0)
            else:
                log("UE -> Proxy", size, nb, ts, frame, subframe, sample_cnt, ch_note)
                if self.gnb_ep and not self.gnb_ep.closed:
                    t_send0 = time.perf_counter()
                    self.gnb_ep.send(hdr_raw, processed)
                    send_ms += 1000 * (time.perf_counter() - t_send0)

            if size > 1 and self.ch_en and self.custom_channel:
                total_ms = 1000 * (time.perf_counter() - t_blk0)
                proc_ms = 1000 * (t_proc1 - t_blk0)
                self.profile_proxy.add(
                    tag=f"dir={ep.role}",
                    PROC=proc_ms, SEND=send_ms, TOTAL=total_ms)

                self._e2e_slot_count += 1
                if ep.role == "gNB":
                    self._e2e_proxy_dl_accum_ms += total_ms
                    self._e2e_dl_in_frame += 1
                    self._e2e_dl_per_ue[0] += 1
                else:
                    self._e2e_proxy_ul_accum_ms += total_ms
                    self._e2e_ul_in_frame += 1
                    self._e2e_ul_per_ue[0] += 1
                self._check_e2e_frame("Socket+OAI")

    def _process_ofdm_slot(self, iq_bytes, ts):
        """Process one OFDM slot through GPU pipeline."""
        t_start = time.perf_counter()

        n_int16 = len(iq_bytes) // 2
        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        n_sym = len(sym_idx)

        t_ch0 = time.perf_counter()
        try:
            channels, n_held = self.channel_buffer.get_batch_view(n_sym)
        except TimeoutError:
            return iq_bytes
        t_ch1 = time.perf_counter()

        t_pad0 = t_ch1
        if n_sym < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            pad = lib.ones((N_SYM - n_sym, FFT_SIZE), dtype=channels.dtype)
            channels = lib.concatenate([channels, pad])
        t_pad1 = time.perf_counter()

        t_gpu0 = t_pad1
        result = self.gpu_slot_pipeline.process_slot(
            iq_bytes, channels, pathLossLinear, snr_dB, noise_enabled, noise_std_abs
        )
        self.channel_buffer.release_batch(n_held)
        t_end = time.perf_counter()

        sc = self.gpu_slot_pipeline.slot_counter
        if sc > 0 and sc % self.profile_interval == 0:
            mode = "GRAPH" if self.gpu_slot_pipeline.graph_captured else "NORMAL"
            self.profile_ofdm.add(
                tag=f"mode={mode}",
                CH_GET=1000 * (t_ch1 - t_ch0),
                CH_PAD=1000 * (t_pad1 - t_pad0),
                GPU_PROC=1000 * (t_end - t_gpu0),
                TOTAL=1000 * (t_end - t_start))

        return result

    # ── GPU IPC mode methods ──

    def _ipc_apply_channel(self, src_ptr, dst_ptr, ts, nsamps,
                           src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size,
                           direction):
        """Apply Sionna MIMO channel to one full slot at ts.
        direction: 'DL' uses H, 'UL' uses H^T.
        Falls back to bypass_copy on circular buffer wrap."""
        arr_in, wraps = self.ipc.get_gpu_array_at(
            src_ptr, ts, nsamps, src_nbAnt, src_cir_size, cp.int16)
        if wraps:
            self.ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                 src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            return

        arr_out, _ = self.ipc.get_gpu_array_at(
            dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, cp.int16)

        try:
            channels, n_held = self.channel_buffer.get_batch_view(N_SYM)
        except TimeoutError:
            self.ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                 src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            return

        if channels.shape[0] < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            n_r, n_t = channels.shape[1], channels.shape[2]
            n_pad = N_SYM - channels.shape[0]
            pad = lib.zeros((n_pad, n_r, n_t, FFT_SIZE), dtype=channels.dtype)
            for k in range(min(n_r, n_t)):
                pad[:, k, k, :] = 1.0
            channels = lib.concatenate([channels, pad])

        if direction == "UL":
            channels = channels.transpose(0, 2, 1, 3)

        pipeline = self.pipeline_dl if direction == "DL" else self.pipeline_ul
        pipeline.process_slot_ipc(
            arr_in, channels, pathLossLinear, snr_dB, noise_enabled, arr_out, noise_std_abs
        )
        self.channel_buffer.release_batch(n_held)

    def _ipc_process_range(self, src_ptr, dst_ptr, set_ts_fn,
                           start_ts, delta,
                           src_nbAnt, src_cir_size,
                           dst_nbAnt, dst_cir_size, direction):
        """Process a range [start_ts, start_ts+delta) from src to dst circular buffer.

        V6: supports per-buffer nbAnt/cir_size for asymmetric MIMO bypass.
        Full slot-aligned chunks get channel processing (if enabled).
        Remaining partial data gets bypass copy with antenna mapping.
        Returns (processing_time_ms, num_slots_processed).
        """
        t0 = time.perf_counter()
        slot_samples = self.gpu_slot_pipeline.total_cpx  # 30720
        apply_ch = self.ch_en and self.custom_channel
        pos = int(start_ts)
        remaining = int(delta)
        slots = 0

        while remaining >= slot_samples and apply_ch:
            self._ipc_apply_channel(src_ptr, dst_ptr, pos, slot_samples,
                                    src_nbAnt, src_cir_size,
                                    dst_nbAnt, dst_cir_size, direction)
            pos += slot_samples
            remaining -= slot_samples
            slots += 1

        if remaining > 0:
            self.ipc.bypass_copy(dst_ptr, src_ptr, pos, remaining,
                                 src_nbAnt, src_cir_size,
                                 dst_nbAnt, dst_cir_size)
            pos += remaining

        set_ts_fn(int(start_ts + delta - 1))
        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    # ── Multi-UE IPC methods (G1C) ──

    def _ipc_apply_channel_for_ue(self, src_ptr, dst_ptr, ts, nsamps,
                                   src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size,
                                   src_ipc, dst_ipc, direction, ue_idx):
        """Apply channel for a specific UE. Uses per-UE channel buffer and pipeline.
        Falls back to bypass on timeout (ChannelProducerProcess may be slow/dead)."""
        arr_in, wraps = src_ipc.get_gpu_array_at(
            src_ptr, ts, nsamps, src_nbAnt, src_cir_size, cp.int16)
        if wraps:
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            return

        arr_out, wraps_out = dst_ipc.get_gpu_array_at(
            dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, cp.int16)
        if wraps_out:
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            return

        try:
            channels, n_held = self.channel_buffers[ue_idx].get_batch_view(N_SYM)
        except TimeoutError:
            print(f"[v2 WARN] UE[{ue_idx}] channel buffer timeout — bypass copy")
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            return

        if channels.shape[0] < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            n_r, n_t = channels.shape[1], channels.shape[2]
            n_pad = N_SYM - channels.shape[0]
            pad = lib.zeros((n_pad, n_r, n_t, FFT_SIZE), dtype=channels.dtype)
            for j in range(min(n_r, n_t)):
                pad[:, j, j, :] = 1.0
            channels = lib.concatenate([channels, pad])

        if direction == "UL":
            channels = channels.transpose(0, 2, 1, 3)

        pipeline = self.pipelines_dl[ue_idx] if direction == "DL" else self.pipelines_ul[ue_idx]
        pipeline.process_slot_ipc(
            arr_in, channels, pathLossLinear, snr_dB, noise_enabled, arr_out, noise_std_abs
        )
        self.channel_buffers[ue_idx].release_batch(n_held)

    def _ipc_dl_broadcast(self, start_ts, delta):
        """DL Broadcast: gNB dl_tx → per-UE channel → UE[k] dl_rx."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_dl[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        for k in range(self.num_ues):
            pos = int(start_ts)
            remaining = int(delta)

            while remaining >= slot_samples and apply_ch:
                self._ipc_apply_channel_for_ue(
                    self.ipc_gnb.gpu_dl_tx_ptr, self.ipc_ues[k].gpu_dl_rx_ptr,
                    pos, slot_samples,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size,
                    self.ipc_gnb, self.ipc_ues[k], "DL", k)
                pos += slot_samples
                remaining -= slot_samples
                if k == 0:
                    slots += 1

            if remaining > 0:
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    pos, remaining,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)

            self.ipc_ues[k].set_last_dl_rx_ts(int(start_ts + delta - 1))

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _ipc_ul_combine(self, start_ts, delta):
        """UL Combine: UE[k] ul_tx → per-UE channel → superposition → gNB ul_rx.
        Bypass mode: sequential copy (last UE wins, no superposition)."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_ul[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        if not apply_ch:
            for k in range(self.num_ues):
                pos = int(start_ts)
                remaining = int(delta)
                while remaining > 0:
                    n = min(remaining, slot_samples)
                    self.ipc_ues[k].bypass_copy(
                        self.ipc_gnb.gpu_ul_rx_ptr, self.ipc_ues[k].gpu_ul_tx_ptr,
                        pos, n,
                        self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size,
                        self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size)
                    pos += n
                    remaining -= n
                    if k == 0:
                        slots += 1
            self.ipc_gnb.set_last_ul_rx_ts(int(start_ts + delta - 1))
        else:
            pos = int(start_ts)
            remaining = int(delta)

            while remaining >= slot_samples:
                self._ipc_ul_superposition_slot(pos, slot_samples)
                pos += slot_samples
                remaining -= slot_samples
                slots += 1

            if remaining > 0:
                for k in range(self.num_ues):
                    self.ipc_ues[k].bypass_copy(
                        self.ipc_gnb.gpu_ul_rx_ptr, self.ipc_ues[k].gpu_ul_tx_ptr,
                        pos, remaining,
                        self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size,
                        self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size)

            self.ipc_gnb.set_last_ul_rx_ts(int(start_ts + delta - 1))

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _ipc_ul_superposition_slot(self, ts, nsamps):
        """Apply per-UE UL channels and sum into gNB ul_rx for one slot."""
        self._ul_accum[:] = 0

        for k in range(self.num_ues):
            arr_in, wraps = self.ipc_ues[k].get_gpu_array_at(
                self.ipc_ues[k].gpu_ul_tx_ptr, ts, nsamps,
                self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size, cp.int16)
            if wraps:
                continue

            try:
                channels, n_held = self.channel_buffers[k].get_batch_view(N_SYM)
            except TimeoutError:
                print(f"[v2 WARN] UE[{k}] UL channel buffer timeout — skip this UE")
                continue
            if channels.shape[0] < N_SYM:
                lib = cp if GPU_AVAILABLE else np
                n_r, n_t = channels.shape[1], channels.shape[2]
                n_pad = N_SYM - channels.shape[0]
                pad = lib.zeros((n_pad, n_r, n_t, FFT_SIZE), dtype=channels.dtype)
                for j in range(min(n_r, n_t)):
                    pad[:, j, j, :] = 1.0
                channels = lib.concatenate([channels, pad])

            channels_ul = channels.transpose(0, 2, 1, 3)

            self.pipelines_ul[k].process_slot_ipc(
                arr_in, channels_ul, pathLossLinear, snr_dB, noise_enabled,
                self._ul_dummy_out, noise_std_abs)
            self.channel_buffers[k].release_batch(n_held)

            self._ul_accum += self.pipelines_ul[k].gpu_out

        n_rx = self.gnb_ant
        total = self.pipelines_ul[0].total_cpx
        n_elem = total * n_rx
        accum_f64 = self._ul_accum.view(cp.float64)
        if not hasattr(self, '_ul_fused_out') or self._ul_fused_out.shape[0] != n_elem * 2:
            self._ul_fused_out = cp.zeros(n_elem * 2, dtype=cp.int16)
        threads = 256
        blocks = (n_elem + threads - 1) // threads
        _fused_clip_cast_kernel((blocks,), (threads,),
                                (accum_f64, self._ul_fused_out, n_elem))

        arr_out, wraps_out = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size, cp.int16)
        if not wraps_out:
            arr_out[:] = self._ul_fused_out
            cp.cuda.Stream.null.synchronize()

    def _check_e2e_frame(self, overhead_label="Socket+OAI"):
        """Print E2E TDD frame stats when a frame boundary is crossed."""
        if self._e2e_slot_count < self._e2e_next_frame_boundary:
            return
        now = time.perf_counter()
        if self._e2e_last_wall is not None:
            wall_ms = 1000 * (now - self._e2e_last_wall)
            dl_acc = self._e2e_proxy_dl_accum_ms
            ul_acc = self._e2e_proxy_ul_accum_ms
            proxy_ms = dl_acc + ul_acc
            overhead_ms = wall_ms - proxy_ms
            nd = self._e2e_dl_in_frame
            nu = self._e2e_ul_in_frame
            ns = nd + nu
            if ns <= 0:
                ns = 1
            N = self.num_ues

            ue_parts = []
            for k in range(N):
                dk = self._e2e_dl_per_ue[k]
                uk = self._e2e_ul_per_ue[k]
                ue_parts.append(f"UE{k}({dk}D+{uk}U)")
            ue_str = " ".join(ue_parts)

            print(f"\n[E2E frame#{self._e2e_next_frame_boundary} "
                  f"{N}UE ({nd}D+{nu}U) {ue_str}] "
                  f"wall={wall_ms:.2f}ms  "
                  f"Proxy(DL={dl_acc:.1f}+UL={ul_acc:.1f})"
                  f"={proxy_ms:.2f}ms  "
                  f"{overhead_label}={overhead_ms:.2f}ms  "
                  f"| per slot({nd}+{nu}={ns}): "
                  f"wall={wall_ms/ns:.2f}  "
                  f"proxy={proxy_ms/ns:.2f}  "
                  f"{overhead_label.lower()}={overhead_ms/ns:.2f} ms")
        else:
            print(f"\n[E2E frame#{self._e2e_next_frame_boundary} {self.num_ues}UE] "
                  f"baseline set (comparison starts next frame)")
        self._e2e_next_frame_boundary += self._e2e_frame_slots
        self._e2e_last_wall = now
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0
        for k in range(self.num_ues):
            self._e2e_dl_per_ue[k] = 0
            self._e2e_ul_per_ue[k] = 0

    def _warmup_pipeline(self):
        """Pre-warm TensorFlow XLA + CUDA Graph with dummy data for all UE pipelines."""
        if not self.pipelines_dl or not self.pipelines_dl[0].enable_gpu:
            return

        for np_thread in self.noise_producers:
            np_thread.start()
            print(f"[v2] NoiseProducer started (noise_len={np_thread.noise_len}, batch={np_thread.BATCH_SIZE})")

        if not self.custom_channel:
            print("[v2] Bypass mode — channel warmup skipped")
            return
        print(f"[v2] Pre-warming {self.num_ues} UE(s) DL/UL pipelines...")
        t0 = time.time()
        passes = GPUSlotPipeline.WARMUP_SLOTS + 1
        for k in range(self.num_ues):
            if self.channel_producers and hasattr(self.channel_producers[k], 'is_alive'):
                if not self.channel_producers[k].is_alive():
                    print(f"[v2 WARN] ChannelProducerProcess[{k}] died during warmup")
            for label, pipeline in [("DL", self.pipelines_dl[k]), ("UL", self.pipelines_ul[k])]:
                dummy_in = cp.zeros(pipeline.total_int16_in, dtype=cp.int16)
                dummy_out = cp.zeros(pipeline.total_int16_out, dtype=cp.int16)
                n_r, n_t = pipeline.n_rx, pipeline.n_tx
                dummy_ch = cp.zeros((N_SYM, n_r, n_t, FFT_SIZE),
                                    dtype=cp.complex128 if GPU_AVAILABLE else np.complex128)
                for j in range(min(n_r, n_t)):
                    dummy_ch[:, j, j, :] = 1.0
                for i in range(passes):
                    pipeline.process_slot_ipc(
                        dummy_in, dummy_ch, pathLossLinear, snr_dB, noise_enabled, dummy_out, noise_std_abs
                    )
                print(f"  UE[{k}] {label} warmup done ({time.time()-t0:.1f}s)")
        print(f"[v2] All {self.num_ues} UE(s) pipelines ready ({time.time()-t0:.1f}s)")

    def run_ipc(self):
        """Main loop for GPU IPC V6 Multi-UE — DL broadcast + UL combine.

        Architecture:
          ipc_gnb: shared gNB SHM (dl_tx write by gNB, ul_rx read by gNB)
          ipc_ues[k]: per-UE SHM (dl_rx read by UE k, ul_tx write by UE k)

        DL: gNB dl_tx → per-UE channel → UE[k] dl_rx (broadcast)
        UL bypass: UE[k] ul_tx → sequential copy → gNB ul_rx (last wins)
        UL channel: UE[k] ul_tx → per-UE channel → sum → gNB ul_rx (superposition)
        """
        N = self.num_ues

        self.ipc_gnb = GPUIpcV7Interface(
            gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
            shm_path=self.ipc_shm_path)
        if not self.ipc_gnb.init():
            print("[ERROR] GPU IPC V7 gNB initialization failed")
            return

        self.ipc_ues = []
        for k in range(N):
            shm_path_ue = f"/tmp/oai_gpu_ipc/gpu_ipc_shm_ue{k}"
            ipc_ue = GPUIpcV7Interface(
                gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
                shm_path=shm_path_ue)
            if not ipc_ue.init():
                print(f"[ERROR] GPU IPC V7 UE[{k}] initialization failed")
                return
            self.ipc_ues.append(ipc_ue)
            print(f"[G1C] UE[{k}] IPC V7 ready: {shm_path_ue}")

        self.ipc = self.ipc_gnb

        total_cpx = sum(SYMBOL_SIZES)
        if GPU_AVAILABLE:
            self._ul_accum = cp.zeros(total_cpx * self.gnb_ant, dtype=cp.complex128)
            self._ul_clip_3d = cp.zeros((total_cpx, self.gnb_ant, 2), dtype=cp.float64)
            self._ul_dummy_out = cp.zeros(
                self.pipelines_ul[0].total_int16_out, dtype=cp.int16)

        self._warmup_pipeline()
        print(f"[v2] Entering main loop ({N} UE(s), gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant})...")
        dl_count = 0
        ul_count = 0
        t_start = time.time()
        proxy_dl_head = 0
        apply_ch = self.ch_en and self.custom_channel

        if apply_ch:
            proxy_ul_head_combined = 0
        else:
            proxy_ul_heads = [0] * N

        try:
            while True:
                processed = False

                # --- DL Broadcast ---
                cur_dl_ts = self.ipc_gnb.get_last_dl_tx_ts()
                dl_nsamps = self.ipc_gnb.get_last_dl_tx_nsamps()
                if cur_dl_ts > 0 and dl_nsamps > 0:
                    gnb_dl_head = cur_dl_ts + dl_nsamps
                    if gnb_dl_head > proxy_dl_head:
                        if proxy_dl_head == 0:
                            proxy_dl_head = max(0, gnb_dl_head - self.ipc_gnb.cir_time)
                        delta = int(gnb_dl_head - proxy_dl_head)
                        dl_ms, n_slots = self._ipc_dl_broadcast(proxy_dl_head, delta)
                        proxy_dl_head = gnb_dl_head
                        dl_count += n_slots
                        processed = True

                        self._e2e_proxy_dl_accum_ms += dl_ms
                        self._e2e_dl_in_frame += n_slots
                        for _k in range(N):
                            self._e2e_dl_per_ue[_k] += n_slots
                        self._e2e_slot_count += n_slots
                        self._check_e2e_frame("IPC_G1C+OAI")

                # --- UL Processing ---
                if apply_ch:
                    min_ue_head = None
                    for k in range(N):
                        cur_ul = self.ipc_ues[k].get_last_ul_tx_ts()
                        ul_ns = self.ipc_ues[k].get_last_ul_tx_nsamps()
                        if cur_ul > 0 and ul_ns > 0:
                            head_k = cur_ul + ul_ns
                            if min_ue_head is None or head_k < min_ue_head:
                                min_ue_head = head_k

                    if min_ue_head is not None and min_ue_head > proxy_ul_head_combined:
                        if proxy_ul_head_combined == 0:
                            proxy_ul_head_combined = max(0, min_ue_head - self.ipc_gnb.cir_time)
                        delta = int(min_ue_head - proxy_ul_head_combined)
                        ul_ms, n_slots = self._ipc_ul_combine(
                            proxy_ul_head_combined, delta)
                        proxy_ul_head_combined = min_ue_head
                        ul_count += n_slots
                        processed = True

                        self._e2e_proxy_ul_accum_ms += ul_ms
                        self._e2e_ul_in_frame += n_slots
                        for _k in range(N):
                            self._e2e_ul_per_ue[_k] += n_slots
                        self._e2e_slot_count += n_slots
                        self._check_e2e_frame("IPC_G1C+OAI")
                else:
                    for k in range(N):
                        cur_ul = self.ipc_ues[k].get_last_ul_tx_ts()
                        ul_ns = self.ipc_ues[k].get_last_ul_tx_nsamps()
                        if cur_ul > 0 and ul_ns > 0:
                            ue_head = cur_ul + ul_ns
                            if ue_head > proxy_ul_heads[k]:
                                if proxy_ul_heads[k] == 0:
                                    proxy_ul_heads[k] = max(0, ue_head - self.ipc_ues[k].cir_time)
                                delta = int(ue_head - proxy_ul_heads[k])
                                self.ipc_ues[k].bypass_copy(
                                    self.ipc_gnb.gpu_ul_rx_ptr,
                                    self.ipc_ues[k].gpu_ul_tx_ptr,
                                    proxy_ul_heads[k], delta,
                                    self.ipc_ues[k].ul_tx_nbAnt,
                                    self.ipc_ues[k].ul_tx_cir_size,
                                    self.ipc_gnb.ul_rx_nbAnt,
                                    self.ipc_gnb.ul_rx_cir_size)
                                self.ipc_gnb.set_last_ul_rx_ts(int(ue_head - 1))
                                proxy_ul_heads[k] = ue_head
                                n_bypass_slots = delta // self.pipelines_ul[0].total_cpx
                                ul_count += n_bypass_slots
                                self._e2e_proxy_ul_accum_ms += 0.0
                                self._e2e_ul_in_frame += n_bypass_slots
                                self._e2e_ul_per_ue[k] += n_bypass_slots
                                self._e2e_slot_count += n_bypass_slots
                                self._check_e2e_frame("IPC_G1C+OAI")
                                processed = True

                if not processed:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            print(f"\n[v2] Terminated by Ctrl-C (DL: {dl_count}, UL: {ul_count}, UEs: {N})")
        finally:
            self._cleanup_channel_producers()
            self.ipc_gnb.cleanup()
            for ipc in self.ipc_ues:
                ipc.cleanup()

    def _cleanup_channel_producers(self):
        """Gracefully stop all ChannelProducerProcess instances and release CUDA IPC resources."""
        for i, evt in enumerate(getattr(self, '_channel_stop_events', [])):
            evt.set()

        for i, proc in enumerate(getattr(self, 'channel_producers', [])):
            if hasattr(proc, 'join'):
                proc.join(timeout=3)
                if proc.is_alive():
                    print(f"[v2] ChannelProducerProcess[{i}] did not exit gracefully, killing")
                    proc.kill()
                    proc.join(timeout=2)

        for i, buf in enumerate(getattr(self, 'channel_buffers', [])):
            if hasattr(buf, 'cleanup'):
                try:
                    buf.cleanup()
                except Exception:
                    pass

        print("[v2] All ChannelProducerProcesses cleaned up")

    def run_socket(self):
        """Main loop for socket mode (v10 compatible)."""
        for np_thread in self.noise_producers:
            if not np_thread.is_alive():
                np_thread.start()
                print(f"[v8] NoiseProducer started (noise_len={np_thread.noise_len}, batch={np_thread.BATCH_SIZE})")
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
        description="G1C v2 Multi-UE MIMO Channel Proxy (multiprocessing ChannelProducer)")

    ap.add_argument("--mode", choices=["socket", "gpu-ipc"], default="socket",
                    help="Communication mode: socket (v10 compat) or gpu-ipc (CUDA IPC)")
    ap.add_argument("--ipc-shm-path", default=GPU_IPC_SHM_PATH,
                    help=f"GPU IPC shared memory file path (default: {GPU_IPC_SHM_PATH})")

    ap.add_argument("--ue-port", type=int, default=6018)
    ap.add_argument("--gnb-host", default="127.0.0.1")
    ap.add_argument("--gnb-port", type=int, default=6017)
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
    ap.add_argument("--buffer-len", type=int, default=10500,
                    help="Channel IPC ring buffer depth (v2 default 10500 = 42000/4)")
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
    ap.add_argument("--snr-dB", type=float, default=None,
                    help="Relative SNR in dB (noise scales with signal power after PL)")
    ap.add_argument("--noise-dBFS", type=float, default=None,
                    help="Absolute noise floor in dBFS (0=full scale 32767). Mutually exclusive with --snr-dB")

    ap.add_argument("--profile-interval", type=int, default=100,
                    help="Profiling report interval in slots (default: 100)")
    ap.add_argument("--profile-window", type=int, default=500,
                    help="Rolling window size for avg/p95/p99/max stats (default: 500)")
    ap.add_argument("--dual-timer-compare", dest='dual_timer_compare', action="store_true",
                    help="Profile with both CPU+sync and CUDA Event timers")
    ap.add_argument("--no-dual-timer-compare", dest='dual_timer_compare', action="store_false",
                    help="Disable CUDA Event comparison (CPU+sync only)")
    ap.set_defaults(dual_timer_compare=True)

    ap.add_argument("--gnb-ant", type=int, default=GNB_ANT,
                    help=f"gNB total antenna count (default: {GNB_ANT} from env)")
    ap.add_argument("--ue-ant", type=int, default=UE_ANT,
                    help=f"UE total antenna count (default: {UE_ANT} from env)")
    ap.add_argument("--gnb-nx", type=int, default=GNB_NX,
                    help=f"gNB antenna cols per panel (default: {GNB_NX} from env)")
    ap.add_argument("--gnb-ny", type=int, default=GNB_NY,
                    help=f"gNB antenna rows per panel (default: {GNB_NY} from env)")
    ap.add_argument("--ue-nx", type=int, default=UE_NX,
                    help=f"UE antenna cols per panel (default: {UE_NX} from env)")
    ap.add_argument("--ue-ny", type=int, default=UE_NY,
                    help=f"UE antenna rows per panel (default: {UE_NY} from env)")
    ap.add_argument("--num-ues", type=int, default=1,
                    help="Number of UEs to support (default: 1)")

    args = ap.parse_args()

    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled, noise_mode, noise_dBFS, noise_std_abs
    path_loss_dB = args.path_loss_dB
    pathLossLinear = 10**(path_loss_dB / 20.0)
    snr_dB = args.snr_dB
    noise_dBFS = args.noise_dBFS

    if snr_dB is not None and noise_dBFS is not None:
        print("[ERROR] --snr-dB and --noise-dBFS are mutually exclusive")
        sys.exit(1)

    noise_mode = "none"
    if snr_dB is not None:
        noise_mode = "relative"
    elif noise_dBFS is not None:
        noise_mode = "absolute"
    noise_enabled = (noise_mode != "none")

    if noise_mode == "absolute":
        import math as _math
        _noise_rms = 32767.0 * (10.0 ** (noise_dBFS / 20.0))
        noise_std_abs = cp.float64(_noise_rms / _math.sqrt(2.0)) if GPU_AVAILABLE else None
    else:
        noise_std_abs = None

    print("=" * 80)
    print("G1C v2 Multi-UE MIMO Channel Proxy (multiprocessing ChannelProducer)")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"UEs: {args.num_ues}")
    if args.mode == "gpu-ipc":
        print(f"IPC SHM Path (gNB): {args.ipc_shm_path}")
        for k in range(args.num_ues):
            print(f"IPC SHM Path (UE{k}): /tmp/oai_gpu_ipc/gpu_ipc_shm_ue{k}")
        print(f"  >> Socket ports NOT used (direct GPU shared memory)")
    else:
        print(f"UE Port: {args.ue_port}, gNB: {args.gnb_host}:{args.gnb_port}")
    print(f"GPU Acceleration: {'Enabled' if args.enable_gpu and GPU_AVAILABLE else 'Disabled'}")
    print(f"CUDA Graph: {'Enabled' if args.use_cuda_graph else 'Disabled'}")
    print(f"Pinned Memory: {'Enabled' if args.use_pinned_memory else 'Disabled'}")
    print(f"Precision: complex128 (float64, PSS stability)")
    print(f"Profiling: interval={args.profile_interval} slots, "
          f"window={args.profile_window} samples, "
          f"dual_timer={'ON' if args.dual_timer_compare else 'OFF'}")
    print(f"Custom Channel: {'Enabled' if args.custom_channel else 'Disabled'}")
    print(f"Path Loss: {path_loss_dB} dB (linear={pathLossLinear:.6f})")
    if noise_mode == "relative":
        print(f"AWGN Noise: Relative SNR mode (SNR={snr_dB} dB)")
    elif noise_mode == "absolute":
        _noise_rms = 32767.0 * (10.0 ** (noise_dBFS / 20.0))
        print(f"AWGN Noise: Absolute mode (floor={noise_dBFS} dBFS, rms={_noise_rms:.1f})")
    else:
        print(f"AWGN Noise: Disabled")
    print("=" * 80)

    print("\n[v2 Architecture]")
    if args.enable_gpu and GPU_AVAILABLE:
        print(f"  + Multi-UE: {args.num_ues} UE(s), per-UE IPC/pipeline/channel/noise")
        print("  + ChannelProducer: multiprocessing.Process (spawn, independent TF context)")
        print("  + RingBuffer: CUDA IPC cross-process GPU ring buffer")
        print("  + DL Broadcast: gNB dl_tx → per-UE channel → UE[k] dl_rx")
        if args.custom_channel:
            print("  + UL Superposition: UE[k] ul_tx → per-UE channel → sum → gNB ul_rx")
        else:
            print("  + UL Bypass: UE[k] ul_tx → sequential copy → gNB ul_rx")
        print(f"  + GPU IPC V7 futex per-buffer antenna (gNB={args.gnb_ant}, UE={args.ue_ant})")
        print(f"  + gNB array: {args.gnb_ny}x{args.gnb_nx}, UE array: {args.ue_ny}x{args.ue_nx}")
        if args.use_cuda_graph:
            print(f"  + CUDA Graph (warmup {GPUSlotPipeline.WARMUP_SLOTS} slots, per-UE)")
        print("  + G1B v8 optimizations: CH_COPY view+release, NoiseProducer, pre-computed FFT")
        print(f"  + Noise model: {noise_mode} " +
              (f"(SNR={snr_dB}dB)" if noise_mode == "relative" else
               f"(floor={noise_dBFS}dBFS)" if noise_mode == "absolute" else "(off)"))
        print(f"  + WindowProfiler (interval={args.profile_interval}, window={args.profile_window})")
    print("=" * 80)
    print()

    proxy = Proxy(
        mode=args.mode,
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
        gnb_ant=args.gnb_ant,
        ue_ant=args.ue_ant,
        gnb_nx=args.gnb_nx,
        gnb_ny=args.gnb_ny,
        ue_nx=args.ue_nx,
        ue_ny=args.ue_ny,
        num_ues=args.num_ues,
    )

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    proxy.run()


if __name__ == "__main__":
    main()
