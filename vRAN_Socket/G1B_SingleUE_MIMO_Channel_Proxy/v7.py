"""
================================================================================
v7.py - GPU IPC V6 + MIMO CUDA Graph Pipeline + CH_COPY Optimized (G1B)

[v6 → v7 변경사항]
  1. RingBuffer dtype c128 통일: astype(c128) 변환 제거
  2. RingBuffer get_batch_view/release_batch: .copy() 제거, count 보호로 안전한 view
  3. ChannelProducer: c128 직접 출력 + FFT 사전 수행 (Pipeline에서 fft(H) 제거)
  4. 조건부 제로화: n_ch == N_SYM이면 gpu_H 제로화 생략

[아키텍처]
  gNB(H2D@ts) → [gpu_dl_tx circ] → Proxy(DL: Hf*Xf) → [gpu_dl_rx circ] → UE(D2H@ts)
  UE(H2D@ts) → [gpu_ul_tx circ] → Proxy(UL: Hf^T*Xf) → [gpu_ul_rx circ] → gNB(D2H@ts)

[핵심 특징]
- GPU IPC V6: per-buffer antenna config (비대칭 MIMO)
- DL/UL 대칭 채널 처리 (-b 플래그로 동시 패스스루)
- 채널 생성: TensorFlow/Sionna (GPU), MIMO Hf(N_r, N_t, FFT) — 주파수 도메인
- 신호 처리: CuPy broadcast*sum (CUDA Graph capturable)
- complex128 정밀도 (PSS/SSS 검출 안정성)
- dual noise model: relative SNR (--snr-dB) / absolute dBFS (--noise-dBFS)
- CH_COPY 최적화: view + release, dtype 통일, 조건부 제로화
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
import ctypes
import mmap
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
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

GPU_IPC_SHM_PATH = GPU_IPC_V6_SHM_PATH

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
                 dual_timer_compare=True, n_tx_in=1, n_rx_out=1):
        self.fft_size = fft_size
        self.n_tx = n_tx_in
        self.n_rx = n_rx_out
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
            cp_l = CP1 if i < 12 else CP2
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
# RingBuffer & ChannelProducer (same as v10)
# ============================================================================

class RingBuffer:
    def __init__(self, shape, dtype=cp.complex128, maxlen=1024):
        if GPU_AVAILABLE:
            self.buffer = cp.zeros((maxlen,) + shape, dtype=dtype)
            self.is_gpu = True
        else:
            np_dtype = np.complex128 if dtype in (cp.complex128, np.complex128) else np.complex64
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


class ChannelProducer(threading.Thread):
    def __init__(self, buffer, channel_generator, topology, params, h_field_array_power, aoa_delay, zoa_delay, buffer_symbol_size=32):
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
            # h_delay: [B=1, N_r, N_t, N_BS=1, N_UE=1, N_sym, N_FFT]
            # Remove B, N_BS, N_UE axes explicitly (preserve N_r, N_t)
            h_delay = h_delay[0, :, :, 0, 0, :, :]   # → (N_r, N_t, N_sym, N_FFT)
            h_delay = tf.transpose(h_delay, [2, 0, 1, 3])  # → (N_sym, N_r, N_t, N_FFT)

            h_c128 = tf.cast(h_delay, tf.complex128)
            # ORIGINAL: energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=-1, keepdims=True)
            energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=2, keepdims=True)
            h_norm = h_delay / tf.cast(tf.sqrt(energy + 1e-30), h_delay.dtype)
            h_c128_norm = tf.cast(h_norm, tf.complex128)

            try:
                h_cp_batch = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c128_norm)).copy()
            except Exception:
                h_cp_batch = cp.asarray(h_c128_norm.numpy())

            h_cp_batch = cp.fft.fft(h_cp_batch, axis=-1)

            # h_cp_batch: (N_sym, N_r, N_t, N_FFT) freq-domain c128
            self.buffer.put_batch(h_cp_batch)
            self.symbol_counter += self.buffer_symbol_size


# ============================================================================
# Proxy (dual-mode: socket / gpu-ipc)
# ============================================================================

class Proxy:
    def __init__(self, mode="socket", ue_port=6018, gnb_host="127.0.0.1", gnb_port=6017,
                 log_level="info", ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np",
                 custom_channel=False, buffer_len=4096, buffer_symbol_size=42,
                 enable_gpu=True, use_pinned_memory=True, use_cuda_graph=True,
                 ipc_shm_path=GPU_IPC_SHM_PATH,
                 profile_interval=100, profile_window=500, dual_timer_compare=True,
                 gnb_ant=1, ue_ant=1,
                 gnb_nx=1, gnb_ny=1, ue_nx=1, ue_ny=1):
        self.mode = mode
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
        self._e2e_last_wall = None
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0

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
        """Initialize Sionna channel model and GPU pipeline (MIMO-aware)."""
        pipeline_common = dict(
            enable_gpu=self.enable_gpu,
            use_pinned_memory=self.use_pinned_memory,
            use_cuda_graph=self.use_cuda_graph,
            profile_interval=self.profile_interval,
            profile_window=self.profile_window,
            dual_timer_compare=self.dual_timer_compare)

        if not self.custom_channel:
            self.pipeline_dl = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=self.gnb_ant, n_rx_out=self.ue_ant, **pipeline_common)
            self.pipeline_ul = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=self.ue_ant, n_rx_out=self.gnb_ant, **pipeline_common)
            self.gpu_slot_pipeline = self.pipeline_dl
            return

        phi_r_rays = tf.convert_to_tensor(np.load(directory + "/phi_r_rays_for_ChannelBlock.npy"))
        phi_t_rays = tf.convert_to_tensor(np.load(directory + "/phi_t_rays_for_ChannelBlock.npy"))
        theta_r_rays = tf.convert_to_tensor(np.load(directory + "/theta_r_rays_for_ChannelBlock.npy"))
        theta_t_rays = tf.convert_to_tensor(np.load(directory + "/theta_t_rays_for_ChannelBlock.npy"))
        power_rays = tf.convert_to_tensor(np.load(directory + "/power_rays_for_ChannelBlock.npy"))
        tau_rays = tf.convert_to_tensor(np.load(directory + "/tau_rays_for_ChannelBlock.npy"))

        batch_size = 1
        self.N_UE = 1
        self.N_BS = 1
        self.num_rx = 1
        self.num_tx = 1

        # gNB TX array via set_BS() with Nx/Ny from launch_all.sh
        BSexample, _ = set_BS(
            num_rows_per_panel=self.gnb_ny, num_cols_per_panel=self.gnb_nx)
        ArrayTX = PanelArray(
            num_rows_per_panel=BSexample["num_rows_per_panel"],
            num_cols_per_panel=BSexample["num_cols_per_panel"],
            num_rows=BSexample["num_rows"], num_cols=BSexample["num_cols"],
            polarization=BSexample["polarization"],
            polarization_type=BSexample["polarization_type"],
            antenna_pattern='omni',
            carrier_frequency=carrier_frequency)
        # UE RX array
        ArrayRX = PanelArray(
            num_rows_per_panel=self.ue_ny, num_cols_per_panel=self.ue_nx,
            num_rows=1, num_cols=1,
            polarization='single', polarization_type='V', antenna_pattern='omni',
            carrier_frequency=carrier_frequency)

        print(f"[MIMO] gNB TX array: {self.gnb_ny}x{self.gnb_nx} = {self.gnb_ant} ant")
        print(f"[MIMO] UE  RX array: {self.ue_ny}x{self.ue_nx} = {self.ue_ant} ant")

        mean_xpr_list = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
        stddev_xpr_list = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}
        mean_xpr = mean_xpr_list["UMa-NLOS"]
        stddev_xpr = stddev_xpr_list["UMa-NLOS"]
        xpr_pdp = 10**(tf.random.normal(
            shape=[batch_size, self.N_BS, self.N_UE, 1, phi_r_rays.shape[-1]],
            mean=mean_xpr, stddev=stddev_xpr
        )/10)

        PDP = Rays(
            delays=tau_rays, powers=power_rays, aoa=phi_r_rays, aod=phi_t_rays,
            zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp
        )

        velocities = tf.abs(tf.random.normal(shape=[batch_size, self.N_UE, 3], mean=Speed, stddev=0.1, dtype=tf.float32))
        moving_end = "rx"
        los_aoa = tf.zeros([batch_size, self.N_BS, self.N_UE])
        los_aod = tf.zeros([batch_size, self.N_BS, self.N_UE])
        los_zoa = tf.zeros([batch_size, self.N_BS, self.N_UE])
        los_zod = tf.zeros([batch_size, self.N_BS, self.N_UE])
        los = tf.random.uniform(shape=[batch_size, self.N_BS, self.N_UE], minval=0, maxval=2, dtype=tf.int32) > 0
        distance_3d = tf.ones([1, self.N_BS, self.N_UE])
        tx_orientations = tf.random.normal(shape=[batch_size, self.N_BS, 3], mean=0, stddev=PI/5, dtype=tf.float32)
        rx_orientations = tf.random.normal(shape=[batch_size, self.N_UE, 3], mean=0, stddev=PI/5, dtype=tf.float32)

        self.topology = Topology(
            velocities, moving_end, los_aoa, los_aod, los_zoa, los_zod,
            los, distance_3d, tx_orientations, rx_orientations
        )

        self.Channel_Generator = ChannelCoefficientsGeneratorJIN(carrier_frequency, scs, ArrayTX, ArrayRX, False)
        h_field_array_power, aoa_delay, zoa_delay = self.Channel_Generator._H_PDP_FIX(self.topology, PDP, N_FFT, scs)
        self.h_field_array_power = tf.transpose(h_field_array_power, [0, 3, 5, 6, 1, 2, 7, 4])
        self.aoa_delay = tf.transpose(aoa_delay, [0, 3, 1, 2, 4])
        self.zoa_delay = tf.transpose(zoa_delay, [0, 3, 1, 2, 4])

        # RingBuffer: each entry = (ue_ant, gnb_ant, FFT_SIZE) = MIMO Hf (freq-domain, c128)
        self.channel_buffer = RingBuffer(
            shape=(self.ue_ant, self.gnb_ant, FFT_SIZE),
            dtype=cp.complex128 if GPU_AVAILABLE else np.complex128,
            maxlen=buffer_len
        )
        params = dict(Fs=Fs, scs=scs, N_UE=self.N_UE, N_BS=self.N_BS,
                      N_UE_active=self.num_rx, N_BS_serving=self.num_tx)

        self.producer = ChannelProducer(
            self.channel_buffer, self.Channel_Generator, self.topology, params,
            self.h_field_array_power, self.aoa_delay, self.zoa_delay,
            buffer_symbol_size=buffer_symbol_size
        )
        self.producer.start()

        # DL pipeline: gNB TX → UE RX  (H: ue_ant x gnb_ant)
        self.pipeline_dl = GPUSlotPipeline(
            FFT_SIZE, n_tx_in=self.gnb_ant, n_rx_out=self.ue_ant, **pipeline_common)
        # UL pipeline: UE TX → gNB RX  (H^T: gnb_ant x ue_ant)
        self.pipeline_ul = GPUSlotPipeline(
            FFT_SIZE, n_tx_in=self.ue_ant, n_rx_out=self.gnb_ant, **pipeline_common)
        self.gpu_slot_pipeline = self.pipeline_dl
        print(f"[INFO] GPU Slot Pipeline v5 (MIMO CUDA Graph) initialized "
              f"(GPU={'enabled' if self.enable_gpu and GPU_AVAILABLE else 'CPU mode'}, "
              f"CUDA Graph={'enabled' if self.use_cuda_graph else 'disabled'}, "
              f"DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx)")

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
                else:
                    self._e2e_proxy_ul_accum_ms += total_ms
                    self._e2e_ul_in_frame += 1
                self._check_e2e_frame("Socket+OAI")

    def _process_ofdm_slot(self, iq_bytes, ts):
        """Process one OFDM slot through GPU pipeline."""
        t_start = time.perf_counter()

        n_int16 = len(iq_bytes) // 2
        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        n_sym = len(sym_idx)

        t_ch0 = time.perf_counter()
        channels, n_held = self.channel_buffer.get_batch_view(n_sym)
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

        channels, n_held = self.channel_buffer.get_batch_view(N_SYM)
        # channels: (N_SYM, ue_ant, gnb_ant, FFT_SIZE) freq-domain c128 view

        if channels.shape[0] < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            n_r, n_t = channels.shape[1], channels.shape[2]
            n_pad = N_SYM - channels.shape[0]
            pad = lib.zeros((n_pad, n_r, n_t, FFT_SIZE), dtype=channels.dtype)
            for k in range(min(n_r, n_t)):
                pad[:, k, k, :] = 1.0  # freq-domain identity: flat passthrough
            channels = lib.concatenate([channels, pad])

        if direction == "UL":
            # H^T: (N_SYM, gnb_ant, ue_ant, FFT) for UL
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

    def _check_e2e_frame(self, overhead_label="Socket+OAI"):
        """Print E2E TDD frame stats when a frame boundary is crossed."""
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
                  f"({nd}D+{nu}U)] "
                  f"wall={wall_ms:.2f}ms  "
                  f"Proxy(DL={dl_acc:.1f}+UL={ul_acc:.1f})"
                  f"={proxy_ms:.2f}ms  "
                  f"{overhead_label}={overhead_ms:.2f}ms  "
                  f"| per slot({n}): "
                  f"wall={wall_ms/n:.2f}  "
                  f"proxy={proxy_ms/n:.2f}  "
                  f"{overhead_label.lower()}={overhead_ms/n:.2f} ms")
        else:
            print(f"\n[E2E frame#{self._e2e_slot_count}] "
                  f"baseline set (comparison starts next frame)")
        self._e2e_last_wall = now
        self._e2e_proxy_dl_accum_ms = 0.0
        self._e2e_proxy_ul_accum_ms = 0.0
        self._e2e_dl_in_frame = 0
        self._e2e_ul_in_frame = 0

    def _warmup_pipeline(self):
        """Pre-warm TensorFlow XLA + CUDA Graph with dummy data for DL and UL pipelines."""
        if not self.pipeline_dl or not self.pipeline_dl.enable_gpu:
            return
        if not self.custom_channel:
            print("[GPU IPC V7] Bypass mode — channel warmup skipped")
            return
        print("[GPU IPC V7] Pre-warming DL/UL pipelines (XLA compile + CUDA Graph)...")
        t0 = time.time()
        passes = GPUSlotPipeline.WARMUP_SLOTS + 1
        for label, pipeline in [("DL", self.pipeline_dl), ("UL", self.pipeline_ul)]:
            dummy_in = cp.zeros(pipeline.total_int16_in, dtype=cp.int16)
            dummy_out = cp.zeros(pipeline.total_int16_out, dtype=cp.int16)
            n_r, n_t = pipeline.n_rx, pipeline.n_tx
            dummy_ch = cp.zeros((N_SYM, n_r, n_t, FFT_SIZE),
                                dtype=cp.complex128 if GPU_AVAILABLE else np.complex128)
            for k in range(min(n_r, n_t)):
                dummy_ch[:, k, k, :] = 1.0  # freq-domain identity: flat passthrough
            for i in range(passes):
                pipeline.process_slot_ipc(
                    dummy_in, dummy_ch, pathLossLinear, snr_dB, noise_enabled, dummy_out, noise_std_abs
                )
                print(f"  {label} warmup {i+1}/{passes} done ({time.time()-t0:.1f}s)")
        print(f"[GPU IPC V7] DL/UL pipelines ready ({time.time()-t0:.1f}s)")

    def run_ipc(self):
        """Main loop for GPU IPC V6 — range-based copy on circular buffers.

        Tracks write-head positions (proxy_dl_head / proxy_ul_head) and copies
        all new data from src to dst on each iteration, avoiding missed writes.
        Per-buffer nbAnt/cir_size enables asymmetric MIMO bypass.
        """
        self.ipc = GPUIpcV6Interface(
            gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
            shm_path=self.ipc_shm_path)
        if not self.ipc.init():
            print("[ERROR] GPU IPC V6 initialization failed")
            return

        self._warmup_pipeline()
        print(f"[GPU IPC V6] Entering main loop (gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant})...")
        dl_count = 0
        ul_count = 0
        t_start = time.time()
        proxy_dl_head = 0
        proxy_ul_head = 0

        try:
            while True:
                processed = False

                cur_dl_ts = self.ipc.get_last_dl_tx_ts()
                dl_nsamps = self.ipc.get_last_dl_tx_nsamps()
                if cur_dl_ts > 0 and dl_nsamps > 0:
                    gnb_dl_head = cur_dl_ts + dl_nsamps
                    if gnb_dl_head > proxy_dl_head:
                        if proxy_dl_head == 0:
                            proxy_dl_head = max(0, gnb_dl_head - self.ipc.cir_time)
                        delta = int(gnb_dl_head - proxy_dl_head)
                        dl_ms, n_slots = self._ipc_process_range(
                            self.ipc.gpu_dl_tx_ptr, self.ipc.gpu_dl_rx_ptr,
                            self.ipc.set_last_dl_rx_ts,
                            proxy_dl_head, delta,
                            self.ipc.dl_tx_nbAnt, self.ipc.dl_tx_cir_size,
                            self.ipc.dl_rx_nbAnt, self.ipc.dl_rx_cir_size, "DL")
                        proxy_dl_head = gnb_dl_head
                        dl_count += n_slots
                        processed = True

                        self._e2e_proxy_dl_accum_ms += dl_ms
                        self._e2e_dl_in_frame += n_slots
                        self._e2e_slot_count += n_slots
                        self._check_e2e_frame("IPC_V6+OAI")

                        if dl_count % 100 == 0:
                            elapsed = time.time() - t_start
                            rate = dl_count / elapsed if elapsed > 0 else 0
                            print(f"[GPU IPC V6] DL: {dl_count}, UL: {ul_count}, "
                                  f"rate: {rate:.1f} DL/s, head={proxy_dl_head}")

                cur_ul_ts = self.ipc.get_last_ul_tx_ts()
                ul_nsamps = self.ipc.get_last_ul_tx_nsamps()
                if cur_ul_ts > 0 and ul_nsamps > 0:
                    ue_ul_head = cur_ul_ts + ul_nsamps
                    if ue_ul_head > proxy_ul_head:
                        if proxy_ul_head == 0:
                            proxy_ul_head = max(0, ue_ul_head - self.ipc.cir_time)
                        delta = int(ue_ul_head - proxy_ul_head)
                        ul_ms, n_slots = self._ipc_process_range(
                            self.ipc.gpu_ul_tx_ptr, self.ipc.gpu_ul_rx_ptr,
                            self.ipc.set_last_ul_rx_ts,
                            proxy_ul_head, delta,
                            self.ipc.ul_tx_nbAnt, self.ipc.ul_tx_cir_size,
                            self.ipc.ul_rx_nbAnt, self.ipc.ul_rx_cir_size, "UL")
                        proxy_ul_head = ue_ul_head
                        ul_count += n_slots
                        processed = True

                        self._e2e_proxy_ul_accum_ms += ul_ms
                        self._e2e_ul_in_frame += n_slots
                        self._e2e_slot_count += n_slots
                        self._check_e2e_frame("IPC_V6+OAI")

                if not processed:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            print(f"\n[GPU IPC V6] Terminated by Ctrl-C (DL: {dl_count}, UL: {ul_count})")
        finally:
            self.ipc.cleanup()

    def run_socket(self):
        """Main loop for socket mode (v10 compatible)."""
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
        description="vRAN Sionna Channel Simulator - v6 (GPU IPC V6 + MIMO CUDA Graph + Dual Noise)")

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
    print("vRAN Sionna Channel Simulator - v6: GPU IPC V6 + MIMO CUDA Graph + Dual Noise")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    if args.mode == "gpu-ipc":
        print(f"IPC SHM Path: {args.ipc_shm_path}")
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

    print("\n[Optimization Status]")
    if args.enable_gpu and GPU_AVAILABLE:
        print("  + Full GPU Pipeline (int16 -> GPU -> int16)")
        print("  + TF->CuPy DLPack (GPU-to-GPU channel transfer)")
        print("  + GPU RingBuffer (c128, view+release)")
        print("  + CuPy Batch FFT (14 symbols, signal only — channel FFT pre-computed)")
        print("  + complex128 precision (PSS/SSS stability)")
        if args.use_cuda_graph:
            print(f"  + CUDA Graph (warmup {GPUSlotPipeline.WARMUP_SLOTS} slots)")
        if args.mode == "gpu-ipc":
            print(f"  + [v7] GPU IPC V6 per-buffer antenna (gNB={args.gnb_ant}, UE={args.ue_ant})")
            print(f"  + [v7] gNB array: {args.gnb_ny}x{args.gnb_nx}, UE array: {args.ue_ny}x{args.ue_nx}")
            print("  + [v7] DL: broadcast*sum channel (CUDA Graph capturable)")
            print("  + [v7] UL: H^T broadcast*sum (DL/UL symmetric)")
            print("  + [v7] OFDM symbol: GPU index array (no Python for-loop)")
            print("  + [v7] Pre-allocated intermediate buffers (out= params)")
            print("  + [v7] CH_COPY: c128 unified, view+release (no .copy()), conditional zero-fill")
            print("  + [v7] Channel FFT pre-computed in ChannelProducer (fft(H) removed from Graph)")
            print(f"  + [v7] Noise model: {noise_mode} " +
                  (f"(SNR={snr_dB}dB)" if noise_mode == "relative" else
                   f"(floor={noise_dBFS}dBFS)" if noise_mode == "absolute" else "(off)"))
            if args.gnb_ant != args.ue_ant:
                print(f"  + [v6] Asymmetric MIMO: bypass=truncate/zero-pad, channel=broadcast*sum")
        if args.use_pinned_memory and args.mode == "socket":
            print("  + Pinned Memory int16 I/O (socket mode only)")
        print(f"  + WindowProfiler (rolling avg/p95/p99/max, "
              f"interval={args.profile_interval}, window={args.profile_window})")
        if args.dual_timer_compare:
            print("  + Dual timer: CPU+sync vs CUDA Event comparison")
        print("  + E2E TDD frame statistics (wall vs proxy vs IPC+OAI)")
    print("=" * 80)
    print()

    Proxy(
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
    ).run()


if __name__ == "__main__":
    main()
