"""
================================================================================
v0.py - G1C Multi-UE MIMO Channel Proxy (based on G1B v8)

[G1B v8 → G1C v0 변경사항]
  1. Multi-UE 지원: N개 UE를 동시에 처리 (--num-ues N)
  2. IPC V6 다중 인스턴스: gNB 1개 + UE N개 (per-UE SHM: gpu_ipc_shm_ue{k})
  3. DL Broadcast: gNB dl_tx → per-UE 채널 적용 → UE[k] dl_rx
  4. UL Superposition: UE[k] ul_tx → (선택적 채널 적용) → 합산 → gNB ul_rx
  5. Per-UE 독립 리소스: pipeline, channel buffer, noise producer
  6. GPU Batch Parallel: N개 UE 채널을 단일 배치 GPU 연산으로 동시 처리

[아키텍처]
  gNB(H2D@ts) → [gpu_dl_tx circ] → Proxy(DL broadcast) → [gpu_dl_rx[k] circ] → UE[k](D2H@ts)
  UE[k](H2D@ts) → [gpu_ul_tx[k] circ] → Proxy(UL combine) → [gpu_ul_rx circ] → gNB(D2H@ts)

[핵심 특징]
- GPU IPC V6 Multi-instance: per-UE SHM 경로 분리 (RFSIM_GPU_IPC_UE_IDX)
- Per-UE 독립 채널/노이즈/파이프라인 (DL/UL 대칭)
- UL Superposition: complex128 합산 후 단일 clip (정밀도 보존)
- G1B v8 최적화 계승: CUDA Graph, CH_COPY view+release, NoiseProducer
- Backward compatible: num_ues=1 시 G1B v8과 동일 동작
- GPU Batch: static 채널 모드 + 2+UE 시 자동 활성화
    DL: FFT 1회 → N채널 동시 곱셈 → N출력 (순차 대비 ~Nx 가속)
    UL: N입력 스택 → 배치 FFT/곱셈 → 중첩 합산 (순차 대비 ~Nx 가속)
================================================================================
"""
import argparse, math, selectors, socket, struct, sys, numpy as np
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
carrier_frequency = 3.5e9
FFT_SIZE = 2048
N_FFT = FFT_SIZE
CP1 = 144
CP0 = 176
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
            ptr = cp.cuda.runtime.malloc(buf_bytes)
            self._gpu_mem.append(('raw', ptr))
            ptrs.append(ptr)
            cp.cuda.runtime.memset(ptr, 0, buf_bytes)
            handle_bytes = cp.cuda.runtime.ipcGetMemHandle(ptr)
            self.shm_mm[handle_off:handle_off + GPU_IPC_V6_HANDLE_SIZE] = handle_bytes
            print(f"[GPU IPC V6] SERVER: allocated {name} "
                  f"({buf_bytes} bytes, cir_size={cir_sz}, ptr=0x{ptr:x}) "
                  f"handle_full={handle_bytes.hex()}")

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
        for item in self._gpu_mem:
            try:
                if isinstance(item, tuple) and item[0] == 'raw':
                    cp.cuda.runtime.free(item[1])
            except Exception:
                pass
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
            cp_l = (e - s) - fft_size
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
            except Exception:
                pass
            try:
                cp.cuda.Device(0).synchronize()
            except Exception:
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

    def process_partial_slot_ipc(self, gpu_iq_in_arr, partial_cpx,
                                 channels_gpu, pl_linear, snr_db, noise_on,
                                 gpu_iq_out_arr, noise_std_abs=None):
        """Process partial slot (< total_cpx samples) via zero-padding to full slot."""
        if not self.enable_gpu:
            return
        partial_in = partial_cpx * self.n_tx * 2
        partial_out = partial_cpx * self.n_rx * 2

        with self.stream:
            self.gpu_iq_in[:] = 0
            self.gpu_iq_in[:partial_in] = gpu_iq_in_arr[:partial_in]

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

            if noise_on:
                self._regenerate_noise()

            self._gpu_compute_core(pl_linear, snr_db, noise_on, noise_std_abs)

            gpu_iq_out_arr[:partial_out] = self.gpu_iq_out[:partial_out]
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
    hdr_vals: Optional[Tuple[int,int,int,int,int]] = None
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


_channel_gen_lock = threading.Lock()


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
        self.ready_event = threading.Event()
        self.daemon = True
        self.symbol_counter = 0

    def _generate_one_batch(self):
        with _channel_gen_lock:
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
            h_delay = h_delay[0, :, :, 0, 0, :, :]
            h_delay = tf.transpose(h_delay, [2, 0, 1, 3])

            h_c128 = tf.cast(h_delay, tf.complex128)
            energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=2, keepdims=True)
            h_norm = h_delay / tf.cast(tf.sqrt(energy + 1e-30), h_delay.dtype)
            h_c128_norm = tf.cast(h_norm, tf.complex128)

            try:
                h_cp_batch = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c128_norm)).copy()
            except Exception:
                h_cp_batch = cp.asarray(h_c128_norm.numpy())

            h_cp_batch = cp.fft.fft(h_cp_batch, axis=-1)

            freq_energy = cp.sum(cp.abs(h_cp_batch) ** 2, axis=2, keepdims=True)
            h_cp_batch = h_cp_batch / cp.sqrt(freq_energy + 1e-30)

        self.buffer.put_batch(h_cp_batch)
        self.symbol_counter += self.buffer_symbol_size

    def run(self):
        if GPU_AVAILABLE:
            cp.cuda.Device(0).use()
        self._generate_one_batch()
        self.ready_event.set()
        while not self.stop_event.is_set():
            self._generate_one_batch()


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
                 custom_channel=False, buffer_len=4096, buffer_symbol_size=42,
                 enable_gpu=True, use_pinned_memory=True, use_cuda_graph=True,
                 ipc_shm_path=GPU_IPC_SHM_PATH,
                 profile_interval=100, profile_window=500, dual_timer_compare=True,
                 gnb_ant=1, ue_ant=1,
                 gnb_nx=1, gnb_ny=1, ue_nx=1, ue_ny=1,
                 num_ues=1, channel_mode="dynamic", polarization="single",
                 sector_half_deg=60.0, jitter_std_deg=10.0):
        self.mode = mode
        self.num_ues = num_ues
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.gnb_nx = gnb_nx
        self.gnb_ny = gnb_ny
        self.ue_nx = ue_nx
        self.ue_ny = ue_ny
        self.polarization = polarization
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
        self.channel_mode = channel_mode
        self.sector_half_rad = float(sector_half_deg) * PI / 180.0
        self.jitter_std_rad = float(jitter_std_deg) * PI / 180.0
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

        # MU-MIMO Analyzer (observer mode)
        self.analyzer = None
        self._analyzer_slot_count = 0

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
        """Initialize per-UE Sionna channel model and GPU pipelines (Multi-UE MIMO)."""
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
            print(f"[G1C] UE[{k}] pipelines created (DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx)")

        self.pipeline_dl = self.pipelines_dl[0]
        self.pipeline_ul = self.pipelines_ul[0]
        self.gpu_slot_pipeline = self.pipeline_dl
        self._noise_buffers_dl = self._noise_buffers_dl_list[0]
        self._noise_buffers_ul = self._noise_buffers_ul_list[0]

        if not self.custom_channel:
            print(f"[G1C] Bypass mode — {N} UE(s), no channel (UL superposition enabled)")
            self.static_channels = []
            self.static_channels_ul = []
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

        pol = self.polarization
        pol_type = "cross" if pol == "dual" else "V"

        BSexample, _ = set_BS(
            num_rows_per_panel=self.gnb_ny, num_cols_per_panel=self.gnb_nx,
            polarization=pol, polarization_type=pol_type)
        ArrayTX = PanelArray(
            num_rows_per_panel=BSexample["num_rows_per_panel"],
            num_cols_per_panel=BSexample["num_cols_per_panel"],
            num_rows=BSexample["num_rows"], num_cols=BSexample["num_cols"],
            polarization=BSexample["polarization"],
            polarization_type=BSexample["polarization_type"],
            antenna_pattern='omni',
            carrier_frequency=carrier_frequency)
        ArrayRX = PanelArray(
            num_rows_per_panel=self.ue_ny, num_cols_per_panel=self.ue_nx,
            num_rows=1, num_cols=1,
            polarization=pol, polarization_type=pol_type,
            antenna_pattern='omni',
            carrier_frequency=carrier_frequency)

        pol_mult = 2 if pol == "dual" else 1
        sionna_gnb = self.gnb_ny * self.gnb_nx * pol_mult
        sionna_ue = self.ue_ny * self.ue_nx * pol_mult
        if sionna_gnb != self.gnb_ant or sionna_ue != self.ue_ant:
            print(f"[MIMO] WARNING: Sionna antenna count mismatch! "
                  f"gnb: IPC={self.gnb_ant} vs Sionna={sionna_gnb}, "
                  f"ue: IPC={self.ue_ant} vs Sionna={sionna_ue}")
            print(f"[MIMO] Hint: with polarization='{pol}', use "
                  f"--gnb-ant {sionna_gnb} --ue-ant {sionna_ue}")

        print(f"[MIMO] gNB TX array: {self.gnb_ny}x{self.gnb_nx} pol={pol} "
              f"→ {sionna_gnb} ant (IPC: {self.gnb_ant})")
        print(f"[MIMO] UE  RX array: {self.ue_ny}x{self.ue_nx} pol={pol} "
              f"→ {sionna_ue} ant (IPC: {self.ue_ant})")

        mean_xpr_list = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
        stddev_xpr_list = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}
        mean_xpr = mean_xpr_list["UMa-NLOS"]
        stddev_xpr = stddev_xpr_list["UMa-NLOS"]

        if self.channel_mode == "static":
            self._init_static_channels(
                N, batch_size, phi_r_rays, phi_t_rays, theta_r_rays, theta_t_rays,
                power_rays, tau_rays, mean_xpr, stddev_xpr, ArrayTX, ArrayRX)
        else:
            self._init_dynamic_channels(
                N, batch_size, buffer_len, buffer_symbol_size,
                phi_r_rays, phi_t_rays, theta_r_rays, theta_t_rays,
                power_rays, tau_rays, mean_xpr, stddev_xpr, ArrayTX, ArrayRX)

    def _build_topology_for_ue(self, batch_size, k):
        """Build a random Sionna Topology for UE k with per-UE angular diversity.

        Each UE is placed at a distinct azimuth angle (AoD) within a 120-degree
        sector (±60°), ensuring different precoding directions and thus diverse
        PMI reports across UEs — critical for SUS MU-MIMO user selection.
        """
        velocities = tf.abs(tf.random.normal(
            shape=[batch_size, self.N_UE, 3], mean=Speed, stddev=0.1, dtype=tf.float32))

        N = max(self.num_ues, 1)
        sector_half = self.sector_half_rad
        base_aod = -sector_half + (2.0 * sector_half) * k / max(N - 1, 1)
        base_aoa = base_aod + PI

        jitter_std = self.jitter_std_rad
        los_aod = tf.fill([batch_size, self.N_BS, self.N_UE], tf.cast(base_aod, tf.float32)) \
                  + tf.random.normal([batch_size, self.N_BS, self.N_UE], stddev=jitter_std)
        los_aoa = tf.fill([batch_size, self.N_BS, self.N_UE], tf.cast(base_aoa, tf.float32)) \
                  + tf.random.normal([batch_size, self.N_BS, self.N_UE], stddev=jitter_std)
        los_zod = tf.fill([batch_size, self.N_BS, self.N_UE], tf.cast(PI / 2.0, tf.float32)) \
                  + tf.random.normal([batch_size, self.N_BS, self.N_UE], stddev=jitter_std)
        los_zoa = tf.fill([batch_size, self.N_BS, self.N_UE], tf.cast(PI / 2.0, tf.float32)) \
                  + tf.random.normal([batch_size, self.N_BS, self.N_UE], stddev=jitter_std)

        los = tf.random.uniform(
            shape=[batch_size, self.N_BS, self.N_UE], minval=0, maxval=2, dtype=tf.int32) > 0
        base_dist = 50.0 + 100.0 * k / max(N - 1, 1)
        distance_3d = tf.fill([1, self.N_BS, self.N_UE], tf.cast(base_dist, tf.float32))
        tx_orientations = tf.random.normal(
            shape=[batch_size, self.N_BS, 3], mean=0, stddev=PI/5, dtype=tf.float32)
        rx_orientations = tf.random.normal(
            shape=[batch_size, self.N_UE, 3], mean=0, stddev=PI/5, dtype=tf.float32)
        return Topology(
            velocities, "rx", los_aoa, los_aod, los_zoa, los_zod,
            los, distance_3d, tx_orientations, rx_orientations)

    def _init_static_channels(self, N, batch_size,
                               phi_r_rays, phi_t_rays, theta_r_rays, theta_t_rays,
                               power_rays, tau_rays, mean_xpr, stddev_xpr,
                               ArrayTX, ArrayRX):
        """Generate one-shot per-UE channel snapshots (static mode).

        Each UE gets a unique but time-invariant frequency-domain channel matrix
        expanded to (N_SYM, ue_ant, gnb_ant, FFT_SIZE) for direct pipeline use.
        GPU memory: N * N_SYM * ue_ant * gnb_ant * FFT_SIZE * 16 bytes.
        """
        self.static_channels = []
        lib = cp if GPU_AVAILABLE else np

        print(f"[G1C-STATIC] Generating {N} static channel snapshots..."
              f" sector_half={self.sector_half_rad*180/PI:.1f}°, jitter_std={self.jitter_std_rad*180/PI:.1f}°")
        t0 = time.time()

        SSB_START_SC = 516
        SSB_BW_SC = 240
        PBCH_DMRS_THRESHOLD = 0.03
        SSB_ENERGY_THRESHOLD = 0.05
        MAX_CHANNEL_RETRIES = 10

        for k in range(N):
            sh = self.sector_half_rad
            ue_az_offset = tf.constant(
                -sh + (2.0 * sh) * k / max(N - 1, 1),
                dtype=phi_r_rays.dtype)
            phi_r_k = phi_r_rays + ue_az_offset
            phi_t_k = phi_t_rays + ue_az_offset

            for attempt in range(MAX_CHANNEL_RETRIES):
                xpr_pdp = 10**(tf.random.normal(
                    shape=[batch_size, self.N_BS, self.N_UE, 1, phi_r_rays.shape[-1]],
                    mean=mean_xpr, stddev=stddev_xpr) / 10)
                PDP = Rays(
                    delays=tau_rays, powers=power_rays, aoa=phi_r_k, aod=phi_t_k,
                    zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp)

                topology_k = self._build_topology_for_ue(batch_size, k)
                gen_k = ChannelCoefficientsGeneratorJIN(
                    carrier_frequency, scs, ArrayTX, ArrayRX, False)
                h_field_k, aoa_k, zoa_k = gen_k._H_PDP_FIX(topology_k, PDP, N_FFT, scs)
                h_field_k = tf.transpose(h_field_k, [0, 3, 5, 6, 1, 2, 7, 4])
                aoa_k = tf.transpose(aoa_k, [0, 3, 1, 2, 4])
                zoa_k = tf.transpose(zoa_k, [0, 3, 1, 2, 4])

                sample_times = tf.constant([0.0], dtype=gen_k.rdtype)
                ActiveUE = tf.constant(
                    random_binary_mask_tf_complex64(self.N_UE, k=self.num_rx), dtype=tf.complex64)
                ServingBS = tf.constant(
                    random_binary_mask_tf_complex64(self.N_BS, k=self.num_tx), dtype=tf.complex64)
                h_delay, _, _, _ = gen_k._H_TTI_sequential_fft_o_ELW2_noProfile(
                    topology_k, ActiveUE, ServingBS, sample_times,
                    h_field_k, aoa_k, zoa_k)
                h_delay = h_delay[0, :, :, 0, 0, :, :]
                h_delay = tf.transpose(h_delay, [2, 0, 1, 3])

                h_c128 = tf.cast(h_delay, tf.complex128)
                energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=2, keepdims=True)
                h_norm = h_delay / tf.cast(tf.sqrt(energy + 1e-30), h_delay.dtype)
                h_c128_norm = tf.cast(h_norm, tf.complex128)

                try:
                    h_snap = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c128_norm)).copy()
                except Exception:
                    h_snap = cp.asarray(h_c128_norm.numpy())

                h_snap = cp.fft.fft(h_snap, axis=-1)

                wideband_energy = float(cp.mean(cp.abs(h_snap) ** 2))
                if wideband_energy > 0:
                    h_snap = h_snap / cp.sqrt(cp.float64(wideband_energy))

                ssb_band = h_snap[:, :, :, SSB_START_SC:SSB_START_SC + SSB_BW_SC]
                ssb_energy = float(cp.mean(cp.abs(ssb_band)**2))
                full_energy = float(cp.mean(cp.abs(h_snap)**2))

                ssb_min_sc_energy = float(cp.min(cp.mean(cp.abs(ssb_band)**2, axis=(0, 1, 2))))
                ssb_max_sc_energy = float(cp.max(cp.mean(cp.abs(ssb_band)**2, axis=(0, 1, 2))))
                sc_variation = ssb_max_sc_energy / max(ssb_min_sc_energy, 1e-30)

                del gen_k, h_field_k, aoa_k, zoa_k, topology_k, h_delay, h_c128
                if GPU_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()

                ssb_ok = full_energy > 0 and ssb_energy >= full_energy * SSB_ENERGY_THRESHOLD
                pbch_ok = ssb_min_sc_energy >= full_energy * PBCH_DMRS_THRESHOLD
                if ssb_ok and pbch_ok:
                    break
                reason = "SSB energy" if not ssb_ok else "PBCH SC null"
                print(f"[G1C-STATIC] UE[{k}] {reason} (ssb={ssb_energy:.2e}, full={full_energy:.2e}, "
                      f"sc_var={sc_variation:.1f}x), retry {attempt+1}/{MAX_CHANNEL_RETRIES}")
                del h_snap, h_norm, h_c128_norm
                if GPU_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
            else:
                print(f"[G1C-STATIC] WARNING: UE[{k}] SSB quality still low after {MAX_CHANNEL_RETRIES} retries, using last attempt")

            h_expanded = lib.broadcast_to(h_snap, (N_SYM,) + h_snap.shape[1:]).copy()
            self.static_channels.append(h_expanded)

            if (k + 1) % 10 == 0 or k == N - 1:
                elapsed = time.time() - t0
                print(f"[G1C-STATIC] UE[{k}] done ({elapsed:.1f}s, "
                      f"{(k+1)*h_expanded.nbytes/1024/1024:.0f} MiB total)")

            del h_snap, h_norm, h_c128_norm
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

        self.static_channels_ul = []
        for ch_dl in self.static_channels:
            ch_ul = ch_dl.transpose(0, 2, 1, 3).copy()
            self.static_channels_ul.append(ch_ul)

        mem_mib = sum(ch.nbytes for ch in self.static_channels) / 1024 / 1024
        mem_ul_mib = sum(ch.nbytes for ch in self.static_channels_ul) / 1024 / 1024
        print(f"[G1C-STATIC] All {N} channels ready ({time.time()-t0:.1f}s, {mem_mib:.0f}+{mem_ul_mib:.0f} MiB GPU)")
        print(f"[G1C-STATIC] DL/UL channels: delay-domain + wideband power normalized")
        print(f"[G1C-STATIC] UL superposition: direct sum + adaptive anti-clipping (threshold=30000)")
        print(f"[G1C] Multi-UE Channel Proxy initialized (STATIC): {N} UE(s), "
              f"DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx")

    def _init_dynamic_channels(self, N, batch_size, buffer_len, buffer_symbol_size,
                                phi_r_rays, phi_t_rays, theta_r_rays, theta_t_rays,
                                power_rays, tau_rays, mean_xpr, stddev_xpr,
                                ArrayTX, ArrayRX):
        """Original dynamic channel mode: per-UE ChannelProducer + RingBuffer."""
        for k in range(N):
            print(f"[G1C] Initializing Sionna channel for UE[{k}]...")
            ue_az_offset = tf.constant(
                -PI / 3.0 + (2.0 * PI / 3.0) * k / max(N - 1, 1),
                dtype=phi_r_rays.dtype)
            phi_r_k = phi_r_rays + ue_az_offset
            phi_t_k = phi_t_rays + ue_az_offset

            xpr_pdp = 10**(tf.random.normal(
                shape=[batch_size, self.N_BS, self.N_UE, 1, phi_r_rays.shape[-1]],
                mean=mean_xpr, stddev=stddev_xpr
            )/10)

            PDP = Rays(
                delays=tau_rays, powers=power_rays, aoa=phi_r_k, aod=phi_t_k,
                zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp
            )

            topology_k = self._build_topology_for_ue(batch_size, k)

            gen_k = ChannelCoefficientsGeneratorJIN(
                carrier_frequency, scs, ArrayTX, ArrayRX, False)
            h_field_k, aoa_k, zoa_k = gen_k._H_PDP_FIX(topology_k, PDP, N_FFT, scs)
            h_field_k = tf.transpose(h_field_k, [0, 3, 5, 6, 1, 2, 7, 4])
            aoa_k = tf.transpose(aoa_k, [0, 3, 1, 2, 4])
            zoa_k = tf.transpose(zoa_k, [0, 3, 1, 2, 4])

            ch_buf_k = RingBuffer(
                shape=(self.ue_ant, self.gnb_ant, FFT_SIZE),
                dtype=cp.complex128 if GPU_AVAILABLE else np.complex128,
                maxlen=buffer_len
            )
            params = dict(Fs=Fs, scs=scs, N_UE=self.N_UE, N_BS=self.N_BS,
                          N_UE_active=self.num_rx, N_BS_serving=self.num_tx)

            prod_k = ChannelProducer(
                ch_buf_k, gen_k, topology_k, params,
                h_field_k, aoa_k, zoa_k,
                buffer_symbol_size=buffer_symbol_size
            )
            self.channel_buffers.append(ch_buf_k)
            self.channel_producers.append(prod_k)
            print(f"[G1C] UE[{k}] channel initialized")

        for k, prod_k in enumerate(self.channel_producers):
            prod_k.start()
            print(f"[G1C] UE[{k}] channel producer started")

        self.channel_buffer = self.channel_buffers[0]
        self.producer = self.channel_producers[0]

        print(f"[G1C] Multi-UE Channel Proxy initialized (DYNAMIC): {N} UE(s), "
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
            except Exception:
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
            except Exception: pass
            try: old.sock.close()
            except Exception: pass
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

        channels, n_held = self._get_channels_for_ue(0, direction)

        pipeline = self.pipeline_dl if direction == "DL" else self.pipeline_ul
        pipeline.process_slot_ipc(
            arr_in, channels, pathLossLinear, snr_dB, noise_enabled, arr_out, noise_std_abs
        )
        self._release_channels_for_ue(0, n_held)

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

    def _get_channels_for_ue(self, ue_idx, direction):
        """Get channel matrix for a UE. Returns (channels, n_held_or_None).
        For static mode, n_held is None (no release needed).
        For dynamic mode, n_held must be passed to release_batch()."""
        if self.channel_mode == "static" and self.static_channels:
            if direction == "UL":
                return self.static_channels_ul[ue_idx], None
            return self.static_channels[ue_idx], None

        channels, n_held = self.channel_buffers[ue_idx].get_batch_view(N_SYM)
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
        return channels, n_held

    def _release_channels_for_ue(self, ue_idx, n_held):
        """Release channel batch view (dynamic mode only)."""
        if n_held is not None:
            self.channel_buffers[ue_idx].release_batch(n_held)

    _diag_dl_ch = 0
    _diag_dl_ch_partial = 0
    _diag_dl_ch_wrap = 0
    _diag_dl_align_bypass = 0
    _diag_dl_align_bypass_samps = 0
    _diag_dl_trail_bypass = 0
    _diag_dl_trail_bypass_samps = 0
    _diag_dl_bypass_wrap_in = 0
    _diag_dl_bypass_wrap_out = 0
    _diag_dl_bypass_zero = 0
    _diag_ul_bypass_zero = 0
    _diag_dl_bypass_partial = 0
    _diag_ul_ch = 0
    _diag_ul_bypass = 0
    _diag_last_print = 0

    def _read_circ_wrap(self, ipc_obj, base_ptr, ts, nsamps, nbAnt, cir_size):
        """Read from circular buffer handling wrap-around by concatenation."""
        off = ipc_obj.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sample_sz = GPU_IPC_V6_SAMPLE_SIZE
        if off + total <= cir_size:
            return ipc_obj._make_gpu_array(base_ptr, off * sample_sz,
                                           (total * sample_sz) // 2)
        tail = cir_size - off
        head = total - tail
        tail_arr = ipc_obj._make_gpu_array(base_ptr, off * sample_sz,
                                           (tail * sample_sz) // 2)
        head_arr = ipc_obj._make_gpu_array(base_ptr, 0,
                                           (head * sample_sz) // 2)
        return cp.concatenate([tail_arr, head_arr])

    def _write_circ_wrap(self, ipc_obj, base_ptr, ts, nsamps, nbAnt, cir_size, data):
        """Write to circular buffer handling wrap-around."""
        off = ipc_obj.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sample_sz = GPU_IPC_V6_SAMPLE_SIZE
        n_int16 = (total * sample_sz) // 2
        if off + total <= cir_size:
            dst = ipc_obj._make_gpu_array(base_ptr, off * sample_sz, n_int16)
            dst[:] = data[:n_int16]
        else:
            tail = cir_size - off
            tail_n = (tail * sample_sz) // 2
            head_n = n_int16 - tail_n
            tail_dst = ipc_obj._make_gpu_array(base_ptr, off * sample_sz, tail_n)
            head_dst = ipc_obj._make_gpu_array(base_ptr, 0, head_n)
            tail_dst[:] = data[:tail_n]
            head_dst[:] = data[tail_n:tail_n + head_n]

    def _ipc_apply_channel_for_ue(self, src_ptr, dst_ptr, ts, nsamps,
                                   src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size,
                                   src_ipc, dst_ipc, direction, ue_idx):
        """Apply channel for a specific UE. Wrapping falls back to bypass."""
        pipeline = self.pipelines_dl[ue_idx] if direction == "DL" else self.pipelines_ul[ue_idx]

        arr_in, wraps_in = src_ipc.get_gpu_array_at(
            src_ptr, ts, nsamps, src_nbAnt, src_cir_size, cp.int16)

        if wraps_in:
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            if direction == "DL":
                self.__class__._diag_dl_bypass_wrap_in += 1
            return

        input_max = int(cp.max(cp.abs(arr_in)))
        if input_max == 0:
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            if direction == "DL":
                self.__class__._diag_dl_bypass_zero += 1
            else:
                self.__class__._diag_ul_bypass_zero += 1
            return

        arr_out_direct, wraps_out = dst_ipc.get_gpu_array_at(
            dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, cp.int16)

        if wraps_out:
            src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
            if direction == "DL":
                self.__class__._diag_dl_bypass_wrap_out += 1
            return

        channels, n_held = self._get_channels_for_ue(ue_idx, direction)

        pipeline.process_slot_ipc(
            arr_in, channels, pathLossLinear, snr_dB,
            noise_enabled, arr_out_direct, noise_std_abs)

        self._release_channels_for_ue(ue_idx, n_held)

        if direction == "DL":
            self.__class__._diag_dl_ch += 1
        else:
            self.__class__._diag_ul_ch += 1

        if direction == "DL":
            total_dl = self._diag_dl_ch
            if total_dl <= 5 or (total_dl <= 100 and total_dl % 20 == 0) or total_dl % 500 == 0:
                out_max = float(cp.max(cp.abs(arr_out_direct)))
                print(f"[CH-DIAG] DL#{total_dl} ts={ts} in_max={input_max} "
                      f"out_max={out_max:.0f} ratio={out_max/(input_max+1e-9):.2f} "
                      f"nsamps={nsamps}",
                      flush=True)

    def _ipc_dl_broadcast(self, start_ts, delta):
        """DL Broadcast: gNB dl_tx → per-UE channel → UE[k] dl_rx.

        Full slots processed through channel (including wrap handling).
        Trailing partial → bypass (preserves raw SSB for PSS detection).
        """
        t0 = time.perf_counter()
        slot_samples = self.pipelines_dl[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        for k in range(self.num_ues):
            pos = int(start_ts)
            remaining = int(delta)

            misalign = pos % slot_samples
            if misalign > 0 and remaining > 0:
                head_partial = min(slot_samples - misalign, remaining)
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    pos, head_partial,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
                if k == 0:
                    self.__class__._diag_dl_trail_bypass += 1
                pos += head_partial
                remaining -= head_partial

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
                if k == 0:
                    self.__class__._diag_dl_trail_bypass += 1
                    self.__class__._diag_dl_trail_bypass_samps += remaining

            self.ipc_ues[k].set_last_dl_rx_ts(int(start_ts + delta - 1))

        now_s = time.time()
        if now_s - self.__class__._diag_last_print >= 10.0:
            self.__class__._diag_last_print = now_s
            ch = self._diag_dl_ch
            bwi = self._diag_dl_bypass_wrap_in
            bwo = self._diag_dl_bypass_wrap_out
            bz = self._diag_dl_bypass_zero
            tb = self._diag_dl_trail_bypass
            total = ch + bwi + bwo + bz + tb
            bp = bwi + bwo + bz + tb
            bp_pct = 100.0 * bp / max(total, 1)
            ch_pct = 100.0 * ch / max(total, 1)
            print(f"[CH-STATS] DL: ch={ch}({ch_pct:.0f}%) bypass(zero={bz} "
                  f"wrap_in={bwi} wrap_out={bwo} trail={tb}) "
                  f"total={total} bypass_pct={bp_pct:.1f}% | "
                  f"UL: ch={self._diag_ul_ch} zero_skip={self._diag_ul_bypass_zero}",
                  flush=True)

        # MU-MIMO Analyzer hook
        if self.analyzer and apply_ch:
            self._analyzer_slot_count += max(slots, 1)
            if self._analyzer_slot_count % self.analyzer.sample_interval == 0:
                try:
                    frame_idx = int(start_ts // (slot_samples * 20))
                    slot_idx = int((start_ts // slot_samples) % 20)
                    channels = {}
                    for k in range(self.num_ues):
                        ch, _ = self._get_channels_for_ue(k, "DL")
                        if GPU_AVAILABLE:
                            channels[k] = cp.asnumpy(ch)
                        else:
                            channels[k] = np.asarray(ch)
                    self.analyzer.on_dl_slot(
                        slot_idx, frame_idx, channels,
                        noise_power=noise_std_abs**2 if noise_std_abs else 1e-10,
                        path_loss_linear=pathLossLinear)
                except Exception as e:
                    print(f"[MuMimoAnalyzer] hook error: {e}")

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _ipc_ul_combine(self, start_ts, delta, active_ues=None):
        """UL Combine: UE[k] ul_tx → per-UE channel → sum → gNB ul_rx.

        Full slots through channel superposition, trailing partial via bypass.
        active_ues: set of UE indices to include (None = all active UEs).
        """
        t0 = time.perf_counter()
        slot_samples = self.pipelines_ul[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        if not apply_ch:
            pos = int(start_ts)
            remaining = int(delta)
            while remaining > 0:
                n = min(remaining, slot_samples)
                self._ipc_ul_bypass_superposition(pos, n, active_ues=active_ues)
                pos += n
                remaining -= n
                slots += 1
            self.ipc_gnb.set_last_ul_rx_ts(int(start_ts + delta - 1))
        else:
            pos = int(start_ts)
            remaining = int(delta)

            misalign = pos % slot_samples
            if misalign > 0 and remaining > 0:
                head_partial = min(slot_samples - misalign, remaining)
                self._ipc_ul_bypass_superposition(pos, head_partial, active_ues=active_ues)
                pos += head_partial
                remaining -= head_partial

            while remaining >= slot_samples:
                self._ipc_ul_superposition_slot(pos, slot_samples, active_ues=active_ues)
                pos += slot_samples
                remaining -= slot_samples
                slots += 1

            if remaining > 0:
                self._ipc_ul_bypass_superposition(pos, remaining, active_ues=active_ues)

            self.ipc_gnb.set_last_ul_rx_ts(int(start_ts + delta - 1))

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _ipc_ul_bypass_superposition(self, ts, nsamps, active_ues=None):
        """Bypass UL superposition: sum raw IQ from all active UEs without channel."""
        n_rx = self.gnb_ant
        n_int16 = nsamps * n_rx * 2
        self._ul_bypass_accum[:n_int16] = 0
        diag = self._ul_diag_slot < 20

        n_added = 0
        for k in range(self.num_ues):
            if active_ues is not None and k not in active_ues:
                continue
            if not self._ue_ul_active[k]:
                continue
            ue_ant = self.ipc_ues[k].ul_tx_nbAnt
            arr_in, wraps = self.ipc_ues[k].get_gpu_array_at(
                self.ipc_ues[k].gpu_ul_tx_ptr, ts, nsamps,
                ue_ant, self.ipc_ues[k].ul_tx_cir_size, cp.int16)
            if wraps:
                if diag:
                    print(f"[UL-DIAG-BP] ts={ts} UE[{k}] WRAP (bypass)")
                continue
            f32 = arr_in.astype(cp.float32)
            n_added += 1
            if ue_ant == n_rx:
                self._ul_bypass_accum[:n_int16] += f32
            else:
                min_ant = min(ue_ant, n_rx)
                src_2d = f32.reshape(nsamps, ue_ant * 2)
                mapped = cp.zeros((nsamps, n_rx * 2), dtype=cp.float32)
                mapped[:, :min_ant * 2] = src_2d[:, :min_ant * 2]
                self._ul_bypass_accum[:n_int16] += mapped.ravel()

        buf = self._ul_bypass_accum[:n_int16]

        if n_added > 1:
            peak = float(cp.max(cp.abs(buf)))
            if peak > 30000.0:
                buf *= cp.float32(30000.0 / peak)
                self.__class__._ul_anticlip_count += 1

        result = cp.clip(
            cp.around(buf), -32768, 32767
        ).astype(cp.int16)

        arr_out, wraps_out = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size, cp.int16)
        if not wraps_out:
            arr_out[:] = result
            cp.cuda.Stream.null.synchronize()

    _ul_diag_slot = 0
    _ul_nonzero_count = 0
    _ul_anticlip_count = 0

    def _ipc_ul_superposition_slot(self, ts, nsamps, active_ues=None):
        """Apply per-UE UL channels and sum into gNB ul_rx for one slot."""
        self._ul_accum[:] = 0
        n_active_ch = 0
        self.__class__._ul_diag_slot += 1
        periodic = (self._ul_diag_slot % 2000 == 0)

        if periodic:
            for k in range(self.num_ues):
                if active_ues is not None and k not in active_ues:
                    continue
                if not self._ue_ul_active[k]:
                    continue
                cir = self.ipc_ues[k].ul_tx_cir_size
                mem = cp.cuda.UnownedMemory(
                    self.ipc_ues[k].gpu_ul_tx_ptr, cir * 4, owner=None)
                full_buf = cp.ndarray(cir * 2, dtype=cp.int16,
                                      memptr=cp.cuda.MemoryPointer(mem, 0))
                buf_max = float(cp.max(cp.abs(full_buf)))
                buf_nnz = int(cp.count_nonzero(full_buf))
                print(f"[UL-DIAG] FULL-SCAN slot#{self._ul_diag_slot} UE[{k}] "
                      f"buf_max={buf_max:.0f} nonzero={buf_nnz}/{cir*2} "
                      f"ptr=0x{self.ipc_ues[k].gpu_ul_tx_ptr:x}")

        for k in range(self.num_ues):
            if active_ues is not None and k not in active_ues:
                continue
            if not self._ue_ul_active[k]:
                continue
            arr_in, wraps = self.ipc_ues[k].get_gpu_array_at(
                self.ipc_ues[k].gpu_ul_tx_ptr, ts, nsamps,
                self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size, cp.int16)
            if wraps:
                continue

            in_max = float(cp.max(cp.abs(arr_in)))
            if in_max > 0:
                self.__class__._ul_nonzero_count += 1
                if self._ul_nonzero_count <= 20:
                    print(f"[UL-DIAG] NONZERO! ts={ts} slot#{self._ul_diag_slot} UE[{k}] input_max={in_max:.0f}")
            if periodic:
                print(f"[UL-DIAG] periodic slot#{self._ul_diag_slot} ts={ts} UE[{k}] input_max={in_max:.0f} nonzero_total={self._ul_nonzero_count}")

            channels_ul, n_held = self._get_channels_for_ue(k, "UL")

            self.pipelines_ul[k].process_slot_ipc(
                arr_in, channels_ul, pathLossLinear, snr_dB, noise_enabled,
                self._ul_dummy_out, noise_std_abs)
            self._release_channels_for_ue(k, n_held)

            self._ul_accum += self.pipelines_ul[k].gpu_out
            n_active_ch += 1

        n_rx = self.gnb_ant
        total = self.pipelines_ul[0].total_cpx
        out_2d = self._ul_accum.reshape(total, n_rx)

        if n_active_ch > 1:
            peak_re = float(cp.max(cp.abs(out_2d.real)))
            peak_im = float(cp.max(cp.abs(out_2d.imag)))
            peak = max(peak_re, peak_im)
            if peak > 30000.0:
                out_2d *= 30000.0 / peak
                self.__class__._ul_anticlip_count += 1

        self._ul_clip_3d[:, :, 0] = cp.clip(cp.around(out_2d.real), -32768, 32767)
        self._ul_clip_3d[:, :, 1] = cp.clip(cp.around(out_2d.imag), -32768, 32767)
        result_int16 = self._ul_clip_3d.ravel().astype(cp.int16)

        arr_out, wraps_out = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size, cp.int16)
        if wraps_out:
            self._write_circ_wrap(self.ipc_gnb, self.ipc_gnb.gpu_ul_rx_ptr,
                                  ts, nsamps, self.ipc_gnb.ul_rx_nbAnt,
                                  self.ipc_gnb.ul_rx_cir_size, result_int16)
        else:
            arr_out[:] = result_int16
        cp.cuda.Stream.null.synchronize()

    def _ipc_ul_superposition_partial(self, ts, nsamps, active_ues=None):
        """Apply per-UE UL channels and sum into gNB ul_rx for partial slot."""
        n_rx = self.gnb_ant
        partial_cpx = nsamps
        partial_accum = cp.zeros(partial_cpx * n_rx, dtype=cp.complex128)
        n_active_ch = 0

        for k in range(self.num_ues):
            if active_ues is not None and k not in active_ues:
                continue
            if not self._ue_ul_active[k]:
                continue

            arr_in = self._read_circ_wrap(
                self.ipc_ues[k], self.ipc_ues[k].gpu_ul_tx_ptr,
                ts, nsamps, self.ipc_ues[k].ul_tx_nbAnt,
                self.ipc_ues[k].ul_tx_cir_size)

            channels_ul, n_held = self._get_channels_for_ue(k, "UL")
            tmp_out = cp.zeros(nsamps * n_rx * 2, dtype=cp.int16)
            self.pipelines_ul[k].process_partial_slot_ipc(
                arr_in, nsamps, channels_ul, pathLossLinear, snr_dB,
                noise_enabled, tmp_out, noise_std_abs)
            self._release_channels_for_ue(k, n_held)

            partial_accum += self.pipelines_ul[k].gpu_out[:partial_cpx * n_rx]
            n_active_ch += 1

        out_2d = partial_accum.reshape(partial_cpx, n_rx)

        if n_active_ch > 1:
            peak_re = float(cp.max(cp.abs(out_2d.real)))
            peak_im = float(cp.max(cp.abs(out_2d.imag)))
            peak = max(peak_re, peak_im)
            if peak > 30000.0:
                out_2d *= 30000.0 / peak
                self.__class__._ul_anticlip_count += 1

        clip_3d = cp.zeros((partial_cpx, n_rx, 2), dtype=cp.float64)
        clip_3d[:, :, 0] = cp.clip(cp.around(out_2d.real), -32768, 32767)
        clip_3d[:, :, 1] = cp.clip(cp.around(out_2d.imag), -32768, 32767)
        result_int16 = clip_3d.ravel().astype(cp.int16)

        arr_out, wraps_out = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size, cp.int16)
        if wraps_out:
            self._write_circ_wrap(self.ipc_gnb, self.ipc_gnb.gpu_ul_rx_ptr,
                                  ts, nsamps, self.ipc_gnb.ul_rx_nbAnt,
                                  self.ipc_gnb.ul_rx_cir_size, result_int16)
        else:
            n_int16 = nsamps * n_rx * 2
            arr_out[:n_int16] = result_int16[:n_int16]
        cp.cuda.Stream.null.synchronize()

    # ── Batch GPU Processing (all UEs in parallel) ──

    def _init_batch_buffers(self):
        """Pre-stack static channels and set up shared OFDM index arrays for batch mode."""
        if not GPU_AVAILABLE:
            self._batch_enabled = False
            return

        N = self.num_ues
        if N <= 1:
            self._batch_enabled = False
            return

        if self.channel_mode != "static" or not self.static_channels:
            self._batch_enabled = False
            print("[BATCH] Batch mode requires static channel mode — falling back to sequential")
            return

        t0 = time.time()

        self._batch_H_dl = cp.stack(self.static_channels, axis=0)
        self._batch_H_ul = cp.stack(self.static_channels_ul, axis=0)

        self._batch_stream = cp.cuda.Stream(non_blocking=True)

        p0 = self.pipelines_dl[0]
        self._b_ext_idx = p0.gpu_ext_idx
        self._b_data_dst = p0.gpu_data_dst
        self._b_cp_dst = p0.gpu_cp_dst
        self._b_cp_src = p0.gpu_cp_src

        self._batch_enabled = True
        mem_mib = self._batch_H_dl.nbytes / 1024 / 1024
        print(f"[BATCH] GPU batch processing enabled: {N} UEs, "
              f"H_dl={list(self._batch_H_dl.shape)}, {mem_mib:.0f} MiB, {time.time()-t0:.1f}s")

    def _batch_dl_slot(self, ts, nsamps):
        """Process one DL slot for ALL UEs in a single batched GPU pass.

        gNB input is de-interleaved and FFT'd once, then multiplied with
        all N channel matrices simultaneously.
        """
        N = self.num_ues
        n_tx = self.gnb_ant
        n_rx = self.ue_ant
        total_cpx = nsamps

        arr_gnb, wraps = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_dl_tx_ptr, ts, nsamps,
            self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size, cp.int16)
        if wraps:
            for k in range(N):
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    ts, nsamps,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
            return

        input_max = int(cp.max(cp.abs(arr_gnb)))
        if input_max == 0:
            for k in range(N):
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    ts, nsamps,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
            self.__class__._diag_dl_bypass_zero += 1
            return

        ue_outs = []
        for k in range(N):
            arr_out, wr = self.ipc_ues[k].get_gpu_array_at(
                self.ipc_ues[k].gpu_dl_rx_ptr, ts, nsamps,
                self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size, cp.int16)
            if wr:
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    ts, nsamps,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
                ue_outs.append(None)
            else:
                ue_outs.append(arr_out)

        int16_in = nsamps * n_tx * 2
        int16_out = nsamps * n_rx * 2

        with self._batch_stream:
            tmp_f64 = arr_gnb[:int16_in].astype(cp.float64)
            tmp_iq_3d = tmp_f64.reshape(total_cpx, n_tx, 2)
            cpx_2d = tmp_iq_3d[:, :, 0] + 1j * tmp_iq_3d[:, :, 1]

            gpu_x = cpx_2d[self._b_ext_idx].transpose(0, 2, 1)
            Xf = cp.fft.fft(gpu_x, axis=-1)

            Yf = cp.zeros((N, N_SYM, n_rx, FFT_SIZE), dtype=cp.complex128)
            for t in range(n_tx):
                Yf += self._batch_H_dl[:, :, :, t, :] * Xf[cp.newaxis, :, cp.newaxis, t, :]

            y = cp.fft.ifft(Yf, axis=-1)

            y_t = y.transpose(0, 1, 3, 2)
            y_flat = y_t.reshape(N, -1, n_rx)

            buf_out = cp.zeros((N, total_cpx, n_rx), dtype=cp.complex128)
            buf_out[:, self._b_data_dst, :] = y_flat
            buf_out[:, self._b_cp_dst, :] = y_flat[:, self._b_cp_src, :]

            out_flat = buf_out.reshape(N, total_cpx * n_rx)
            if pathLossLinear != 1.0:
                out_flat *= cp.float64(pathLossLinear)

            if noise_enabled:
                n_elem = total_cpx * n_rx
                if noise_std_abs is not None:
                    out_flat += noise_std_abs * (
                        cp.random.randn(N, n_elem).astype(cp.float64)
                        + 1j * cp.random.randn(N, n_elem).astype(cp.float64))
                elif snr_dB is not None:
                    sig_pwr = cp.mean(cp.abs(out_flat) ** 2, axis=1, keepdims=True)
                    snr_lin = cp.float64(10.0 ** (snr_dB / 10.0))
                    n_std = cp.sqrt(sig_pwr / snr_lin / cp.float64(2.0))
                    out_flat += n_std * (
                        cp.random.randn(N, n_elem).astype(cp.float64)
                        + 1j * cp.random.randn(N, n_elem).astype(cp.float64))

            out_2d = out_flat.reshape(N, total_cpx, n_rx)
            iq_3d = cp.empty((N, total_cpx, n_rx, 2), dtype=cp.float64)
            iq_3d[:, :, :, 0] = cp.clip(cp.around(out_2d.real), -32768, 32767)
            iq_3d[:, :, :, 1] = cp.clip(cp.around(out_2d.imag), -32768, 32767)
            iq_int16 = iq_3d.reshape(N, -1).astype(cp.int16)

            for k in range(N):
                if ue_outs[k] is not None:
                    ue_outs[k][:int16_out] = iq_int16[k]

            self._batch_stream.synchronize()

    def _ipc_dl_broadcast_batch(self, start_ts, delta):
        """Batched DL Broadcast: full slots via batch channel, trailing via bypass."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_dl[0].total_cpx
        slots = 0
        pos = int(start_ts)
        remaining = int(delta)

        misalign = pos % slot_samples
        if misalign > 0 and remaining > 0:
            head_partial = min(slot_samples - misalign, remaining)
            for k in range(self.num_ues):
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    pos, head_partial,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
            self.__class__._diag_dl_trail_bypass += 1
            pos += head_partial
            remaining -= head_partial

        while remaining >= slot_samples:
            self._batch_dl_slot(pos, slot_samples)
            pos += slot_samples
            remaining -= slot_samples
            slots += 1

        if remaining > 0:
            for k in range(self.num_ues):
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    pos, remaining,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
            self.__class__._diag_dl_trail_bypass += 1
            self.__class__._diag_dl_trail_bypass_samps += remaining

        for k in range(self.num_ues):
            self.ipc_ues[k].set_last_dl_rx_ts(int(start_ts + delta - 1))

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _batch_ul_slot(self, ts, nsamps, active_ues=None):
        """Process one UL slot for active UEs in a single batched GPU pass.

        Active UE inputs are stacked, batch-processed through their
        respective channels, and the results are superimposed.
        """
        N = self.num_ues
        n_tx_ue = self.ue_ant
        n_rx_gnb = self.gnb_ant
        total_cpx = nsamps

        active_indices = []
        ue_arrs = []
        for k in range(N):
            if active_ues is not None and k not in active_ues:
                continue
            if not self._ue_ul_active[k]:
                continue
            arr_in, wraps = self.ipc_ues[k].get_gpu_array_at(
                self.ipc_ues[k].gpu_ul_tx_ptr, ts, nsamps,
                self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size, cp.int16)
            if wraps:
                continue
            active_indices.append(k)
            ue_arrs.append(arr_in)

        n_active = len(active_indices)

        arr_gnb_out, wraps_out = self.ipc_gnb.get_gpu_array_at(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size, cp.int16)

        if n_active == 0 or wraps_out:
            if not wraps_out:
                arr_gnb_out[:] = 0
                cp.cuda.Stream.null.synchronize()
            return

        int16_in = nsamps * n_tx_ue * 2

        with self._batch_stream:
            batch_iq = cp.stack([a[:int16_in] for a in ue_arrs])

            batch_H_ul = self._batch_H_ul[active_indices]

            tmp_f64 = batch_iq.astype(cp.float64)
            tmp_iq_3d = tmp_f64.reshape(n_active, total_cpx, n_tx_ue, 2)
            cpx_2d = tmp_iq_3d[:, :, :, 0] + 1j * tmp_iq_3d[:, :, :, 1]

            gpu_x = cpx_2d[:, self._b_ext_idx, :].transpose(0, 1, 3, 2)
            Xf = cp.fft.fft(gpu_x, axis=-1)

            Yf = cp.zeros((n_active, N_SYM, n_rx_gnb, FFT_SIZE), dtype=cp.complex128)
            for t in range(n_tx_ue):
                Yf += batch_H_ul[:, :, :, t, :] * Xf[:, :, cp.newaxis, t, :]

            y = cp.fft.ifft(Yf, axis=-1)

            y_t = y.transpose(0, 1, 3, 2)
            y_flat = y_t.reshape(n_active, -1, n_rx_gnb)

            buf_out = cp.zeros((n_active, total_cpx, n_rx_gnb), dtype=cp.complex128)
            buf_out[:, self._b_data_dst, :] = y_flat
            buf_out[:, self._b_cp_dst, :] = y_flat[:, self._b_cp_src, :]

            out_flat = buf_out.reshape(n_active, total_cpx * n_rx_gnb)
            if pathLossLinear != 1.0:
                out_flat *= cp.float64(pathLossLinear)

            if noise_enabled:
                n_elem = total_cpx * n_rx_gnb
                if noise_std_abs is not None:
                    out_flat += noise_std_abs * (
                        cp.random.randn(n_active, n_elem).astype(cp.float64)
                        + 1j * cp.random.randn(n_active, n_elem).astype(cp.float64))
                elif snr_dB is not None:
                    sig_pwr = cp.mean(cp.abs(out_flat) ** 2, axis=1, keepdims=True)
                    snr_lin = cp.float64(10.0 ** (snr_dB / 10.0))
                    n_std = cp.sqrt(sig_pwr / snr_lin / cp.float64(2.0))
                    out_flat += n_std * (
                        cp.random.randn(n_active, n_elem).astype(cp.float64)
                        + 1j * cp.random.randn(n_active, n_elem).astype(cp.float64))

            combined = cp.sum(out_flat, axis=0)

            out_2d = combined.reshape(total_cpx, n_rx_gnb)

            if n_active > 1:
                peak_re = float(cp.max(cp.abs(out_2d.real)))
                peak_im = float(cp.max(cp.abs(out_2d.imag)))
                peak = max(peak_re, peak_im)
                if peak > 30000.0:
                    out_2d *= 30000.0 / peak
                    self.__class__._ul_anticlip_count += 1

            iq_3d = cp.empty((total_cpx, n_rx_gnb, 2), dtype=cp.float64)
            iq_3d[:, :, 0] = cp.clip(cp.around(out_2d.real), -32768, 32767)
            iq_3d[:, :, 1] = cp.clip(cp.around(out_2d.imag), -32768, 32767)
            result_int16 = iq_3d.ravel().astype(cp.int16)

            arr_gnb_out[:] = result_int16
            self._batch_stream.synchronize()

    def _ipc_ul_combine_batch(self, start_ts, delta, active_ues=None):
        """Batched UL Combine with slot-boundary alignment."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_ul[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        if not apply_ch:
            pos = int(start_ts)
            remaining = int(delta)
            while remaining > 0:
                n = min(remaining, slot_samples)
                self._ipc_ul_bypass_superposition(pos, n, active_ues=active_ues)
                pos += n
                remaining -= n
                slots += 1
        else:
            pos = int(start_ts)
            remaining = int(delta)

            misalign = pos % slot_samples
            if misalign > 0 and remaining > 0:
                head_partial = min(slot_samples - misalign, remaining)
                self._ipc_ul_bypass_superposition(pos, head_partial, active_ues=active_ues)
                pos += head_partial
                remaining -= head_partial

            while remaining >= slot_samples:
                self._batch_ul_slot(pos, slot_samples, active_ues=active_ues)
                pos += slot_samples
                remaining -= slot_samples
                slots += 1
            if remaining > 0:
                self._ipc_ul_bypass_superposition(pos, remaining, active_ues=active_ues)
        self.ipc_gnb.set_last_ul_rx_ts(int(start_ts + delta - 1))
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
        """Pre-warm TensorFlow XLA + CUDA Graph with dummy data for all UE pipelines."""
        if not self.pipelines_dl or not self.pipelines_dl[0].enable_gpu:
            return

        for np_thread in self.noise_producers:
            np_thread.start()
            print(f"[G1C] NoiseProducer started (noise_len={np_thread.noise_len}, batch={np_thread.BATCH_SIZE})")

        if not self.custom_channel:
            print("[G1C] Bypass mode — channel warmup skipped")
            return

        mode_tag = "STATIC" if self.channel_mode == "static" else "DYNAMIC"
        print(f"[G1C] Pre-warming {self.num_ues} UE(s) DL/UL pipelines ({mode_tag})...")
        t0 = time.time()
        passes = GPUSlotPipeline.WARMUP_SLOTS + 1
        for k in range(self.num_ues):
            if self.channel_mode == "static" and self.static_channels:
                ch_dl = self.static_channels[k]
                ch_ul = self.static_channels_ul[k]
            else:
                n_r, n_t = self.ue_ant, self.gnb_ant
                ch_dl = cp.zeros((N_SYM, n_r, n_t, FFT_SIZE),
                                 dtype=cp.complex128 if GPU_AVAILABLE else np.complex128)
                for j in range(min(n_r, n_t)):
                    ch_dl[:, j, j, :] = 1.0
                ch_ul = ch_dl.transpose(0, 2, 1, 3).copy()

            for label, pipeline, ch in [("DL", self.pipelines_dl[k], ch_dl),
                                        ("UL", self.pipelines_ul[k], ch_ul)]:
                dummy_in = cp.zeros(pipeline.total_int16_in, dtype=cp.int16)
                dummy_out = cp.zeros(pipeline.total_int16_out, dtype=cp.int16)
                for i in range(passes):
                    pipeline.process_slot_ipc(
                        dummy_in, ch, pathLossLinear, snr_dB, noise_enabled, dummy_out, noise_std_abs
                    )
                print(f"  UE[{k}] {label} warmup done ({time.time()-t0:.1f}s)")
        print(f"[G1C] All {self.num_ues} UE(s) pipelines ready ({mode_tag}, {time.time()-t0:.1f}s)")

    def run_ipc(self):
        """Main loop for GPU IPC V6 Multi-UE — DL broadcast + UL combine.

        Architecture:
          ipc_gnb: shared gNB SHM (dl_tx write by gNB, ul_rx read by gNB)
          ipc_ues[k]: per-UE SHM (dl_rx read by UE k, ul_tx write by UE k)

        DL: gNB dl_tx → per-UE channel → UE[k] dl_rx (broadcast)
        UL: UE[k] ul_tx → (optional channel) → sum → gNB ul_rx (superposition)
        """
        N = self.num_ues

        self.ipc_gnb = GPUIpcV6Interface(
            gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
            shm_path=self.ipc_shm_path)
        if not self.ipc_gnb.init():
            print("[ERROR] GPU IPC V6 gNB initialization failed")
            return

        self.ipc_ues = []
        for k in range(N):
            shm_path_ue = f"/tmp/oai_gpu_ipc/gpu_ipc_shm_ue{k}"
            ipc_ue = GPUIpcV6Interface(
                gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
                shm_path=shm_path_ue)
            if not ipc_ue.init():
                print(f"[ERROR] GPU IPC V6 UE[{k}] initialization failed")
                return
            self.ipc_ues.append(ipc_ue)
            print(f"[G1C] UE[{k}] IPC ready: {shm_path_ue}")

        self.ipc = self.ipc_gnb

        total_cpx = sum(SYMBOL_SIZES)
        if GPU_AVAILABLE:
            self._ul_accum = cp.zeros(total_cpx * self.gnb_ant, dtype=cp.complex128)
            self._ul_clip_3d = cp.zeros((total_cpx, self.gnb_ant, 2), dtype=cp.float64)
            self._ul_dummy_out = cp.zeros(
                self.pipelines_ul[0].total_int16_out, dtype=cp.int16)
            self._ul_bypass_accum = cp.zeros(
                total_cpx * self.gnb_ant * 2, dtype=cp.float32)

        self._warmup_pipeline()

        self._batch_enabled = False
        if GPU_AVAILABLE and N > 1 and self.ch_en and self.custom_channel:
            self._init_batch_buffers()
        _mode_tag = "BATCH" if getattr(self, '_batch_enabled', False) else "SEQ"
        print(f"[G1C] Entering main loop ({N} UE(s), gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant}, {_mode_tag})...")
        dl_count = 0
        ul_count = 0
        t_start = time.time()
        proxy_dl_head = 0
        apply_ch = self.ch_en and self.custom_channel
        _slot_samples = self.pipelines_dl[0].total_cpx if self.pipelines_dl else 30720

        proxy_ul_head_combined = 0
        self._ue_ul_active = [False] * N
        self._stalled_ue_logged = set()
        self._ul_stall_count = 0
        _last_ipc_scan_time = time.time()

        try:
            while True:
                processed = False

                now = time.time()
                if now - _last_ipc_scan_time >= 10.0:
                    _last_ipc_scan_time = now
                    elapsed = now - t_start
                    import numpy as np
                    gnb_ptr = self.ipc_gnb.gpu_dl_tx_ptr
                    gnb_cir = self.ipc_gnb.dl_tx_cir_size
                    gnb_mem = cp.cuda.UnownedMemory(gnb_ptr, gnb_cir * 4, owner=None)
                    gnb_buf = cp.ndarray(gnb_cir * 2, dtype=cp.int16,
                                         memptr=cp.cuda.MemoryPointer(gnb_mem, 0))
                    gnb_max_gpu = float(cp.max(cp.abs(gnb_buf)))
                    gnb_cpu = gnb_buf[:2048].get()
                    gnb_max_cpu = int(np.max(np.abs(gnb_cpu)))
                    gnb_nnz_cpu = int(np.count_nonzero(gnb_cpu))
                    print(f"[IPC-SCAN] t={elapsed:.0f}s gNB dl_tx: "
                          f"gpu_max={gnb_max_gpu:.0f} cpu_max={gnb_max_cpu} "
                          f"cpu_nnz={gnb_nnz_cpu}/2048 "
                          f"first8={gnb_cpu[:8].tolist()} "
                          f"ptr=0x{gnb_ptr:x}", flush=True)
                    for k in range(N):
                        ue_ptr = self.ipc_ues[k].gpu_ul_tx_ptr
                        cir = self.ipc_ues[k].ul_tx_cir_size
                        mem = cp.cuda.UnownedMemory(ue_ptr, cir * 4, owner=None)
                        full_buf = cp.ndarray(cir * 2, dtype=cp.int16,
                                              memptr=cp.cuda.MemoryPointer(mem, 0))
                        ue_max_gpu = float(cp.max(cp.abs(full_buf)))
                        ue_cpu = full_buf[:2048].get()
                        ue_max_cpu = int(np.max(np.abs(ue_cpu)))
                        ue_nnz_cpu = int(np.count_nonzero(ue_cpu))
                        ue_ts = self.ipc_ues[k].get_last_ul_tx_ts()
                        print(f"[IPC-SCAN] t={elapsed:.0f}s UE[{k}] ul_tx: "
                              f"gpu_max={ue_max_gpu:.0f} cpu_max={ue_max_cpu} "
                              f"cpu_nnz={ue_nnz_cpu}/2048 "
                              f"first8={ue_cpu[:8].tolist()} "
                              f"ue_head={ue_ts} proxy_ul={proxy_ul_head_combined} "
                              f"ptr=0x{ue_ptr:x}", flush=True)

                # --- DL Broadcast ---
                cur_dl_ts = self.ipc_gnb.get_last_dl_tx_ts()
                dl_nsamps = self.ipc_gnb.get_last_dl_tx_nsamps()
                if cur_dl_ts > 0 and dl_nsamps > 0:
                    gnb_dl_head = cur_dl_ts + dl_nsamps
                    if gnb_dl_head > proxy_dl_head:
                        if proxy_dl_head == 0:
                            proxy_dl_head = max(0, gnb_dl_head - self.ipc_gnb.cir_time)
                        delta = int(gnb_dl_head - proxy_dl_head)
                        if self._batch_enabled:
                            dl_ms, n_slots = self._ipc_dl_broadcast_batch(proxy_dl_head, delta)
                        else:
                            dl_ms, n_slots = self._ipc_dl_broadcast(proxy_dl_head, delta)
                        proxy_dl_head = gnb_dl_head
                        dl_count += n_slots
                        processed = True

                        self._e2e_proxy_dl_accum_ms += dl_ms
                        self._e2e_dl_in_frame += n_slots
                        self._e2e_slot_count += n_slots
                        self._check_e2e_frame("IPC_G1C+OAI")

                        if dl_count % 100 == 0:
                            elapsed = time.time() - t_start
                            rate = dl_count / elapsed if elapsed > 0 else 0
                            n_stalled = len(self._stalled_ue_logged)
                            print(f"[G1C] DL: {dl_count}, UL: {ul_count}, "
                                  f"rate: {rate:.1f} DL/s, {N}UE, head={proxy_dl_head}"
                                  f", anticlip={self._ul_anticlip_count}"
                                  f", stalled={n_stalled}/{N}")

                # --- UL Processing (active-set stall detection + superposition) ---
                ue_heads = {}
                for k in range(N):
                    cur_ul = self.ipc_ues[k].get_last_ul_tx_ts()
                    ul_ns = self.ipc_ues[k].get_last_ul_tx_nsamps()
                    if cur_ul > 0 and ul_ns > 0:
                        if not self._ue_ul_active[k]:
                            self._ue_ul_active[k] = True
                            print(f"[G1C] UE[{k}] UL active (ts={cur_ul})")
                        ue_heads[k] = cur_ul + ul_ns

                if ue_heads:
                    max_head = max(ue_heads.values())
                    cir = self.ipc_gnb.cir_time
                    active_set = {k for k, h in ue_heads.items()
                                  if max_head - h < cir}
                    stalled = set(ue_heads.keys()) - active_set

                    for sk in stalled:
                        if sk not in self._stalled_ue_logged:
                            self._stalled_ue_logged.add(sk)
                            self._ul_stall_count += 1
                            print(f"[G1C-STALL] UE[{sk}] stall detected — "
                                  f"head={ue_heads[sk]}, max_head={max_head}, "
                                  f"gap={max_head - ue_heads[sk]}, cir_time={cir}")

                    recovered = self._stalled_ue_logged & active_set
                    for rk in recovered:
                        self._stalled_ue_logged.discard(rk)
                        print(f"[G1C-STALL] UE[{rk}] recovered")

                    if active_set:
                        min_active_head = min(ue_heads[k] for k in active_set)

                        if min_active_head > proxy_ul_head_combined:
                            if proxy_ul_head_combined == 0:
                                proxy_ul_head_combined = max(0, min_active_head - cir)
                                print(f"[UL-DIAG] UL start: proxy_ul_head={proxy_ul_head_combined} "
                                      f"min_active_head={min_active_head} active={len(active_set)}/{N} "
                                      f"proxy_dl_head={proxy_dl_head} cir_time={cir}")
                                for k in active_set:
                                    ue_cir = self.ipc_ues[k].ul_tx_cir_size
                                    total_i16 = ue_cir * 2
                                    mem = cp.cuda.UnownedMemory(
                                        self.ipc_ues[k].gpu_ul_tx_ptr,
                                        ue_cir * 4, owner=None)
                                    full_buf = cp.ndarray(total_i16, dtype=cp.int16,
                                        memptr=cp.cuda.MemoryPointer(mem, 0))
                                    buf_max = float(cp.max(cp.abs(full_buf)))
                                    buf_nnz = int(cp.count_nonzero(full_buf))
                                    print(f"[UL-DIAG] UE[{k}] FULL ul_tx buffer scan: "
                                          f"max={buf_max:.0f} nonzero={buf_nnz}/{total_i16} "
                                          f"ptr=0x{self.ipc_ues[k].gpu_ul_tx_ptr:x}", flush=True)
                            delta = int(min_active_head - proxy_ul_head_combined)
                            if self._batch_enabled:
                                ul_ms, n_slots = self._ipc_ul_combine_batch(
                                    proxy_ul_head_combined, delta,
                                    active_ues=active_set)
                            else:
                                ul_ms, n_slots = self._ipc_ul_combine(
                                    proxy_ul_head_combined, delta,
                                    active_ues=active_set)
                            proxy_ul_head_combined = min_active_head
                            ul_count += n_slots
                            processed = True

                            self._e2e_proxy_ul_accum_ms += ul_ms
                            self._e2e_ul_in_frame += n_slots
                            self._e2e_slot_count += n_slots
                            self._check_e2e_frame("IPC_G1C+OAI")

                if not processed:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            print(f"\n[G1C] Terminated by Ctrl-C (DL: {dl_count}, UL: {ul_count}, UEs: {N})")
        finally:
            self.ipc_gnb.cleanup()
            for ipc in self.ipc_ues:
                ipc.cleanup()

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
# Core Emulator helpers
# ============================================================================

def _fetch_core_proxy_params(addr: str) -> dict:
    """Fetch proxy params from Core Emulator via TCP JSON."""
    host, port = addr.split(":")
    port = int(port)
    req = json.dumps({"cmd": "GET_PROXY_PARAMS"}) + "\n"
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(req.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8").strip()
    resp = json.loads(raw)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Core Emulator error: {resp.get('msg')}")
    return resp["params"]


def _core_emulator_listener(addr: str, proxy_ref):
    """Background thread: listen for PROXY_UPDATE events from Core Emulator."""
    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled, noise_mode, noise_dBFS, noise_std_abs
    host, port = addr.split(":")
    port = int(port)

    while True:
        try:
            with socket.create_connection((host, port), timeout=30) as sock:
                req = json.dumps({"cmd": "SUBSCRIBE_PROXY"}) + "\n"
                sock.sendall(req.encode("utf-8"))
                buf = b""
                while True:
                    data = sock.recv(65536)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if msg.get("event") == "PROXY_UPDATE":
                            params = msg.get("params", {})
                            print(f"[Core Emulator] PROXY_UPDATE 수신: {params}")
                            _apply_proxy_hotswap(params, proxy_ref)
        except Exception as e:
            print(f"[Core Emulator Listener] 연결 끊김, 5초 후 재접속: {e}")
            time.sleep(5)


def _apply_proxy_hotswap(params: dict, proxy_ref):
    """Apply hot-swappable proxy parameters at runtime."""
    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled, noise_mode, noise_dBFS, noise_std_abs

    if "path_loss_dB" in params and params["path_loss_dB"] is not None:
        path_loss_dB = float(params["path_loss_dB"])
        pathLossLinear = 10 ** (path_loss_dB / 20.0)
        print(f"[Hotswap] path_loss_dB={path_loss_dB}, linear={pathLossLinear:.6f}")

    if "snr_dB" in params:
        new_snr = params["snr_dB"]
        if new_snr is not None:
            snr_dB = float(new_snr)
            noise_mode = "relative"
            noise_enabled = True
            noise_dBFS = None
            noise_std_abs = None
            print(f"[Hotswap] snr_dB={snr_dB}, mode=relative")
        else:
            snr_dB = None
            if noise_dBFS is None:
                noise_mode = "none"
                noise_enabled = False
            print(f"[Hotswap] snr_dB=None, mode={noise_mode}")

    if "noise_dBFS" in params:
        new_nf = params["noise_dBFS"]
        if new_nf is not None:
            noise_dBFS = float(new_nf)
            noise_mode = "absolute"
            noise_enabled = True
            snr_dB = None
            import math as _math
            _noise_rms = 32767.0 * (10.0 ** (noise_dBFS / 20.0))
            noise_std_abs = cp.float64(_noise_rms / _math.sqrt(2.0)) if GPU_AVAILABLE else None
            print(f"[Hotswap] noise_dBFS={noise_dBFS}, mode=absolute")
        else:
            noise_dBFS = None
            noise_std_abs = None
            if snr_dB is None:
                noise_mode = "none"
                noise_enabled = False
            print(f"[Hotswap] noise_dBFS=None, mode={noise_mode}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="G1C Multi-UE MIMO Channel Proxy (GPU IPC V6 + CUDA Graph)")

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
    ap.add_argument("--num-ues", type=int, default=1,
                    help="Number of UEs to support (default: 1)")
    ap.add_argument("--channel-mode", choices=["dynamic", "static"], default="dynamic",
                    help="Channel generation mode: dynamic (per-symbol Sionna, default) or "
                         "static (one-shot per-UE snapshot, scales to 64+ UEs)")
    ap.add_argument("--polarization", choices=["single", "dual"], default="single",
                    help="Antenna polarization: single (V-only, XP=1) or "
                         "dual (cross-pol ±45°, XP=2). "
                         "dual doubles Sionna antenna count (NxNy→NxNy×2)")
    ap.add_argument("--sector-half-deg", type=float, default=60.0, dest="sector_half_deg",
                    help="Half-sector span in degrees for UE angular placement (default: 60)")
    ap.add_argument("--jitter-std-deg", type=float, default=10.0, dest="jitter_std_deg",
                    help="AoD/AoA jitter std dev in degrees per UE (default: 10)")
    ap.add_argument("--core-emulator", type=str, default=None,
                    help="Core Emulator address HOST:PORT (e.g. localhost:7100). "
                         "When set, initial config is fetched from Core Emulator "
                         "and a background listener receives runtime updates.")
    ap.add_argument("--analyzer", action="store_true", default=False,
                    help="Enable MU-MIMO precoding mismatch analyzer (observer mode)")
    ap.add_argument("--analyzer-interval", type=int, default=20,
                    help="Analyzer sample interval in DL slots (default: 20)")
    ap.add_argument("--analyzer-log-dir", type=str, default=None,
                    help="Directory for analyzer CSV output (default: auto from log dir)")

    args = ap.parse_args()

    # ── Core Emulator 초기 설정 수신 ──
    if args.core_emulator:
        _ce_addr = args.core_emulator
        print(f"[Core Emulator] 연결 중: {_ce_addr}")
        try:
            _ce_params = _fetch_core_proxy_params(_ce_addr)
            print(f"[Core Emulator] 설정 수신 완료: {json.dumps(_ce_params, default=str)}")
            if "num_ues" in _ce_params and _ce_params["num_ues"]:
                args.num_ues = _ce_params["num_ues"]
            if "channel_mode" in _ce_params and _ce_params["channel_mode"]:
                args.channel_mode = _ce_params["channel_mode"]
            if "gnb_nx" in _ce_params:
                args.gnb_nx = _ce_params["gnb_nx"]
            if "gnb_ny" in _ce_params:
                args.gnb_ny = _ce_params["gnb_ny"]
            if "ue_nx" in _ce_params:
                args.ue_nx = _ce_params["ue_nx"]
            if "ue_ny" in _ce_params:
                args.ue_ny = _ce_params["ue_ny"]
            if "polarization" in _ce_params and _ce_params["polarization"]:
                args.polarization = _ce_params["polarization"]
            if "gnb_ant" in _ce_params:
                args.gnb_ant = _ce_params["gnb_ant"]
            if "ue_ant" in _ce_params:
                args.ue_ant = _ce_params["ue_ant"]
            if "path_loss_dB" in _ce_params and _ce_params["path_loss_dB"] is not None:
                args.path_loss_dB = float(_ce_params["path_loss_dB"])
            if "snr_dB" in _ce_params and _ce_params["snr_dB"] is not None:
                args.snr_dB = float(_ce_params["snr_dB"])
            if "noise_dBFS" in _ce_params and _ce_params["noise_dBFS"] is not None:
                args.noise_dBFS = float(_ce_params["noise_dBFS"])
            if "sector_half_deg" in _ce_params and _ce_params["sector_half_deg"] is not None:
                args.sector_half_deg = float(_ce_params["sector_half_deg"])
            if "jitter_std_deg" in _ce_params and _ce_params["jitter_std_deg"] is not None:
                args.jitter_std_deg = float(_ce_params["jitter_std_deg"])
        except Exception as e:
            print(f"[Core Emulator] WARNING: 연결 실패, CLI 기본값 사용: {e}")

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
    print("G1C Multi-UE MIMO Channel Proxy (GPU IPC V6 + CUDA Graph)")
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
    print(f"Custom Channel: {'Enabled' if args.custom_channel else 'Disabled'}"
          f"{' (' + args.channel_mode.upper() + ')' if args.custom_channel else ''}")
    _pol_label = "dual (cross ±45°, XP=2)" if args.polarization == "dual" else "single (V, XP=1)"
    print(f"Polarization: {_pol_label}")
    print(f"Path Loss: {path_loss_dB} dB (linear={pathLossLinear:.6f})")
    if noise_mode == "relative":
        print(f"AWGN Noise: Relative SNR mode (SNR={snr_dB} dB)")
    elif noise_mode == "absolute":
        _noise_rms = 32767.0 * (10.0 ** (noise_dBFS / 20.0))
        print(f"AWGN Noise: Absolute mode (floor={noise_dBFS} dBFS, rms={_noise_rms:.1f})")
    else:
        print(f"AWGN Noise: Disabled")
    print("=" * 80)

    print("\n[G1C Architecture]")
    if args.enable_gpu and GPU_AVAILABLE:
        print(f"  + Multi-UE: {args.num_ues} UE(s), per-UE IPC/pipeline/channel/noise")
        print("  + DL Broadcast: gNB dl_tx → per-UE channel → UE[k] dl_rx")
        if args.custom_channel:
            print("  + UL Superposition: UE[k] ul_tx → per-UE channel → sum → gNB ul_rx")
        else:
            print("  + UL Superposition (bypass): UE[k] ul_tx → sum → gNB ul_rx")
        print(f"  + GPU IPC V6 per-buffer antenna (gNB={args.gnb_ant}, UE={args.ue_ant})")
        _pol_tag = f", pol={args.polarization}" if args.polarization == "dual" else ""
        print(f"  + gNB array: {args.gnb_ny}x{args.gnb_nx}{_pol_tag}, "
              f"UE array: {args.ue_ny}x{args.ue_nx}{_pol_tag}")
        _batch_eligible = (args.num_ues > 1 and args.custom_channel
                            and args.channel_mode == "static")
        if _batch_eligible:
            print(f"  + GPU Batch Parallel: {args.num_ues} UEs batched (DL: 1xFFT+NxH, UL: NxFFT+sum)")
        else:
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
        channel_mode=args.channel_mode,
        polarization=args.polarization,
        sector_half_deg=getattr(args, 'sector_half_deg', 60.0),
        jitter_std_deg=getattr(args, 'jitter_std_deg', 10.0),
    )

    if args.core_emulator:
        _listener_t = threading.Thread(
            target=_core_emulator_listener,
            args=(args.core_emulator, proxy),
            daemon=True,
        )
        _listener_t.start()
        print(f"[Core Emulator] 런타임 리스너 스레드 시작 ({args.core_emulator})")

    if args.analyzer and args.num_ues >= 2:
        from mu_mimo_analyzer import MuMimoAnalyzer
        analyzer_dir = args.analyzer_log_dir or "."
        proxy.analyzer = MuMimoAnalyzer(
            num_ues=args.num_ues,
            gnb_ant=args.gnb_ant,
            ue_ant=args.ue_ant,
            fft_size=FFT_SIZE,
            log_dir=analyzer_dir,
            sample_interval=args.analyzer_interval,
        )
        print(f"[MuMimoAnalyzer] Enabled: {args.num_ues} UEs, "
              f"interval={args.analyzer_interval}, dir={analyzer_dir}")

    try:
        proxy.run()
    finally:
        if proxy.analyzer:
            proxy.analyzer.close()


if __name__ == "__main__":
    main()
