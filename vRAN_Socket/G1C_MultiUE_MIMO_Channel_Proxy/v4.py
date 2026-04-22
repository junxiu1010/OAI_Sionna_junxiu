"""
================================================================================
v4.py - G1C Multi-UE MIMO Channel Proxy (v3 + Unified ChannelProducer)

[v3 → v4 변경사항]
  1. 통합 채널 생산 (Unified ChannelProducer):
     - v3: UE별 독립 ChannelProducerProcess (N_UE=1 × N개 프로세스)
     - v4: 단일 UnifiedChannelProducerProcess (N_UE=실제UE수, 1개 프로세스)
     - TF 프로세스 1개로 통합 → VRAM 절약, GPU 커널 효율 향상
  2. P1B ray 데이터 스택:
     - load_p1b_stacked(): N개 RX ray를 N_UE 차원으로 concat
     - Sionna의 N_UE 배치 차원을 활용한 통합 채널 생성
  3. Non-blocking 링버퍼 분배:
     - try_put_batch(): 버퍼 full 시 drop (blocking 방지)
     - 뻑난 UE의 full 버퍼가 다른 UE 생산을 멈추지 않음

[v3에서 유지]
  - Active-set 기반 UE stall 감지
  - IPCRingBuffer (CUDA IPC cross-process)
  - IPC V7 + futex, UL fused RawKernel, CUDA Graph
  - P1B RX 인덱스 검증, 랜덤 선택

[아키텍처]
  P1B npz → load_p1b_stacked() → 스택된 ray_data (N_UE 차원)
  UnifiedChannelProducerProcess (단일 프로세스: TF+CuPy, N_UE 통합)
    → UE별 슬라이싱 → try_put_batch() → N개 IPCRingBuffer
  Proxy Main Process (CuPy + CUDA Graph):
    → active_set 기반 UL superposition
    → gNB/UE IPC V7 polling loop
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
import bisect
import ctypes
import ctypes.util
import mmap
import multiprocessing as _mp
import signal
import random as _random
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
os.environ.pop('TF_GPU_ALLOCATOR', None)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

HDR_FMT_LE = "<I I Q I I"
HDR_LEN = struct.calcsize(HDR_FMT_LE)

def unpack_header(b):
    size, nb, ts, frame, subframe = struct.unpack(HDR_FMT_LE, b)
    return size, nb, ts, frame, subframe

# ── CsiNet Sidecar Hook (Phase 4 Integration) ──
_CSINET_HOOK = None

def get_csinet_hook():
    """Lazy-init the CsiNet channel hook if CSINET_ENABLED env var is set."""
    global _CSINET_HOOK
    if _CSINET_HOOK is not None:
        return _CSINET_HOOK
    if os.environ.get("CSINET_ENABLED", "0") == "1":
        try:
            import sys as _sys
            _sys.path.insert(0, os.environ.get("CSINET_PATH",
                "/workspace/graduation/csinet"))
            from integration.channel_hook import ChannelHook
            from integration.csinet_engine import CsiNetInferenceEngine
            from integration.csi_injection import CSIInjector

            hook = ChannelHook(enabled=True,
                               csi_rs_period=int(os.environ.get("CSINET_PERIOD", "20")))
            mode = os.environ.get("CSINET_MODE", "baseline")
            gamma = float(os.environ.get("CSINET_GAMMA", "0.25"))
            scenario = os.environ.get("CSINET_SCENARIO", "UMi_NLOS")
            ckpt_dir = os.environ.get("CSINET_CHECKPOINT_DIR", "/workspace/csinet_checkpoints")

            diff_enabled = os.environ.get("CSINET_DIFF_ENABLED", "0") == "1"
            diff_threshold = float(os.environ.get("CSINET_DIFF_THRESHOLD", "0.01"))
            diff_max_stale = int(os.environ.get("CSINET_DIFF_MAX_STALE", "100"))

            engine = CsiNetInferenceEngine(
                mode=mode, compression_ratio=gamma,
                checkpoint_dir=ckpt_dir, scenario=scenario,
                diff_enabled=diff_enabled,
                diff_threshold=diff_threshold,
                diff_max_stale=diff_max_stale)
            injector = CSIInjector()

            def on_channel_captured(cell_idx, ue_idx, H_freq):
                R_H, pdp = hook.get_statistics(cell_idx, ue_idx, n_samples=50)
                if engine.diff_conditioner is not None:
                    H_hat, codeword, diff_info = engine.encode_decode_differential(
                        H_freq, R_H, pdp, cell_idx, ue_idx)
                else:
                    H_hat, codeword = engine.encode_decode(H_freq, R_H, pdp)
                injector.process_channel(cell_idx, ue_idx, H_hat)

            hook.register_callback(on_channel_captured)
            hook._engine = engine
            hook._injector = injector
            _CSINET_HOOK = hook
            diff_str = (f", differential=th{diff_threshold}/stale{diff_max_stale}"
                        if diff_enabled else "")
            print("[CsiNet] Sidecar hook initialized: "
                  f"mode={mode}, gamma={gamma}, scenario={scenario}, "
                  f"ckpt_dir={ckpt_dir}{diff_str}")
        except Exception as e:
            print(f"[CsiNet] Hook init failed: {e}")
            import traceback; traceback.print_exc()
            _CSINET_HOOK = None
    return _CSINET_HOOK

MAX_LOG = 300
LOG_LINES = []
DL_LOG_CNT = 0
LAST_TS = None
LAST_WALLTIME = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(os.path.dirname(_SCRIPT_DIR), "saved_rays_data")


def validate_rx_indices(npz_path, requested_indices):
    """P1B npz의 유효 RX 인덱스와 대조.
    무효 인덱스마다 가장 가까운 유효 인덱스를 안내하고 sys.exit(1)."""
    data = np.load(npz_path, allow_pickle=True)
    valid_set = set(data['rx_indices'].tolist())
    sorted_valid = sorted(valid_set)
    data.close()

    errors = []
    for rx in requested_indices:
        if rx not in valid_set:
            pos = bisect.bisect_left(sorted_valid, rx)
            candidates = []
            if pos > 0:
                candidates.append(sorted_valid[pos - 1])
            if pos < len(sorted_valid):
                candidates.append(sorted_valid[pos])
            nearest = min(candidates, key=lambda x: abs(x - rx))
            errors.append(
                f"  RX{rx}는 존재하지 않습니다. "
                f"가장 가까운 유효 인덱스: RX{nearest}")

    if errors:
        print(f"[ERROR] --ue-rx-indices 검증 실패 ({len(errors)}건):")
        for e in errors:
            print(e)
        print(f"  유효 범위: RX{sorted_valid[0]}~RX{sorted_valid[-1]} "
              f"(총 {len(sorted_valid)}개)")
        import sys; sys.exit(1)


def pick_random_rx_indices(npz_path, num_ues):
    """유효 RX 중 num_ues개 무작위 선택 (중복 없음)."""
    data = np.load(npz_path, allow_pickle=True)
    all_indices = data['rx_indices'].tolist()
    data.close()
    selected = _random.sample(all_indices, num_ues)
    parts = ", ".join(f"UE{i}=RX{rx}" for i, rx in enumerate(selected))
    print(f"[P1B] 랜덤 RX 선택: {parts}")
    return selected


def load_p1b_per_ue(npz_path, rx_index):
    """P1B npz에서 특정 RX의 ray 데이터를 추출, degree→radian 변환.
    Returns: dict with 6 numpy arrays, each shape (1,1,1,1,400)"""
    data = np.load(npz_path, allow_pickle=True)
    rx_list = data['rx_indices'].tolist()
    pos = rx_list.index(rx_index)
    rad = np.pi / 180.0
    result = {
        'tau':     data['tau'][pos],
        'power':   data['power'][pos],
        'phi_r':   data['phi_r_deg'][pos] * rad,
        'phi_t':   data['phi_t_deg'][pos] * rad,
        'theta_r': data['theta_r_deg'][pos] * rad,
        'theta_t': data['theta_t_deg'][pos] * rad,
    }
    data.close()
    return result


def load_p1b_stacked(npz_path, rx_indices):
    """N개 RX의 ray 데이터를 N_UE 차원으로 스택.
    Returns: dict, each value shape (1, 1, N_UE, 1, 400)"""
    per_ue = [load_p1b_per_ue(npz_path, rx) for rx in rx_indices]
    return {
        key: np.concatenate([d[key] for d in per_ue], axis=2)
        for key in per_ue[0].keys()
    }


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
ue_speeds = None  # per-UE speeds list [m/s]; None = use global Speed for all

DL_USE_IDENTITY_CHANNEL = False
PURE_DL_BYPASS = True   # DIAG: skip OFDM pipeline for DL, raw copy gNB TX → UE RX
UL_GAIN_LINEAR = 16.0   # UL signal amplification factor (1.0 = no gain)

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
GPU_IPC_V7_CIR_TIME = 4608000  # 10× increase: 150 slots (75 ms) to prevent buffer overwrite
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

    def circ_read(self, base_ptr, ts, nsamps, nbAnt, cir_size, dtype=cp.int16):
        """Read from GPU circular buffer with wrap-around handling.
        Always returns a valid contiguous CuPy array (copy when wrapping)."""
        off = self.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sz = GPU_IPC_V6_SAMPLE_SIZE
        elem_sz = dtype().itemsize
        if off + total <= cir_size:
            n_elem = (total * sz) // elem_sz
            mem = cp.cuda.UnownedMemory(base_ptr + off * sz,
                                        total * sz, owner=None)
            return cp.ndarray(n_elem, dtype=dtype,
                              memptr=cp.cuda.MemoryPointer(mem, 0))
        tail = cir_size - off
        head = total - tail
        tail_n = (tail * sz) // elem_sz
        head_n = (head * sz) // elem_sz
        arr_tail = cp.ndarray(tail_n, dtype=dtype,
                              memptr=cp.cuda.MemoryPointer(
                                  cp.cuda.UnownedMemory(base_ptr + off * sz,
                                                        tail * sz, owner=None), 0))
        arr_head = cp.ndarray(head_n, dtype=dtype,
                              memptr=cp.cuda.MemoryPointer(
                                  cp.cuda.UnownedMemory(base_ptr,
                                                        head * sz, owner=None), 0))
        return cp.concatenate([arr_tail, arr_head])

    def circ_write(self, base_ptr, ts, nsamps, nbAnt, cir_size, data):
        """Write data to GPU circular buffer with wrap-around handling."""
        off = self.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sz = GPU_IPC_V6_SAMPLE_SIZE
        elem_sz = data.dtype.itemsize
        if off + total <= cir_size:
            n_elem = (total * sz) // elem_sz
            mem = cp.cuda.UnownedMemory(base_ptr + off * sz,
                                        total * sz, owner=None)
            dst = cp.ndarray(n_elem, dtype=data.dtype,
                             memptr=cp.cuda.MemoryPointer(mem, 0))
            dst[:] = data
        else:
            tail = cir_size - off
            head = total - tail
            tail_n = (tail * sz) // elem_sz
            head_n = (head * sz) // elem_sz
            dst_t = cp.ndarray(tail_n, dtype=data.dtype,
                               memptr=cp.cuda.MemoryPointer(
                                   cp.cuda.UnownedMemory(base_ptr + off * sz,
                                                         tail * sz, owner=None), 0))
            dst_t[:] = data[:tail_n]
            dst_h = cp.ndarray(head_n, dtype=data.dtype,
                               memptr=cp.cuda.MemoryPointer(
                                   cp.cuda.UnownedMemory(base_ptr,
                                                         head * sz, owner=None), 0))
            dst_h[:] = data[tail_n:]
        cp.cuda.Stream.null.synchronize()

    def circ_zero(self, base_ptr, ts, nsamps, nbAnt, cir_size):
        """Zero-fill GPU circular buffer with wrap-around handling."""
        off = self.circ_offset(ts, nbAnt, cir_size)
        total = nsamps * nbAnt
        sz = GPU_IPC_V6_SAMPLE_SIZE
        if off + total <= cir_size:
            n16 = (total * sz) // 2
            mem = cp.cuda.UnownedMemory(base_ptr + off * sz,
                                        total * sz, owner=None)
            cp.ndarray(n16, dtype=cp.int16,
                       memptr=cp.cuda.MemoryPointer(mem, 0))[:] = 0
        else:
            tail = cir_size - off
            head = total - tail
            cp.ndarray((tail * sz) // 2, dtype=cp.int16,
                       memptr=cp.cuda.MemoryPointer(
                           cp.cuda.UnownedMemory(base_ptr + off * sz,
                                                 tail * sz, owner=None), 0))[:] = 0
            cp.ndarray((head * sz) // 2, dtype=cp.int16,
                       memptr=cp.cuda.MemoryPointer(
                           cp.cuda.UnownedMemory(base_ptr,
                                                 head * sz, owner=None), 0))[:] = 0
        cp.cuda.Stream.null.synchronize()

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

    def get_ul_consumer_ts(self):
        return self.read_shm_field(360, '<Q')

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
        # Verify SHM content is correct by reading back
        _rb_magic = struct.unpack_from('<I', self.shm_mm, 256)[0]
        _rb_ver   = struct.unpack_from('<I', self.shm_mm, 260)[0]
        _rb = {}
        for _name, _off_ant, _off_cir, _exp_ant, _exp_cir in [
            ('dl_tx', 272, 276, self.dl_tx_nbAnt, self.dl_tx_cir_size),
            ('dl_rx', 280, 284, self.dl_rx_nbAnt, self.dl_rx_cir_size),
            ('ul_tx', 288, 292, self.ul_tx_nbAnt, self.ul_tx_cir_size),
            ('ul_rx', 296, 300, self.ul_rx_nbAnt, self.ul_rx_cir_size),
        ]:
            _ra = struct.unpack_from('<I', self.shm_mm, _off_ant)[0]
            _rc = struct.unpack_from('<I', self.shm_mm, _off_cir)[0]
            _rb[_name] = (_ra, _rc)
            if _ra != _exp_ant or _rc != _exp_cir:
                print(f"[GPU IPC V7] *** SHM VERIFY FAILED *** {_name}: "
                      f"read ant={_ra},cir={_rc} expected ant={_exp_ant},cir={_exp_cir} "
                      f"(shm_path={self.shm_path})")
        print(f"[GPU IPC V7] SERVER: ready (magic=0x{_rb_magic:08X}, ver={_rb_ver}, "
              f"gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant}, cir_time={self.cir_time}, "
              f"futex=enabled, shm={self.shm_path})")
        print(f"[GPU IPC V7] SHM VERIFY: "
              + " ".join(f"{k}(ant={v[0]},cir={v[1]})" for k, v in _rb.items()))
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

    def set_ul_sync_ts(self, ts):
        """Signal gNB to sync its nextRxTstamp to this UE-timeline timestamp."""
        self.write_shm_field(384, '<Q', ts)


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

            _POST_LIMIT = 28000.0
            out_peak = float(cp.max(cp.abs(self.gpu_out)))
            if out_peak > _POST_LIMIT:
                _rescale = _POST_LIMIT / out_peak
                self.gpu_out *= cp.float64(_rescale)
                n_rx = self.n_rx
                self._tmp_out_2d_final = self.gpu_out.reshape(self.total_cpx, n_rx)
                self._buf_iq_out_3d[:, :, 0] = cp.clip(
                    cp.around(self._tmp_out_2d_final.real), -32768, 32767)
                self._buf_iq_out_3d[:, :, 1] = cp.clip(
                    cp.around(self._tmp_out_2d_final.imag), -32768, 32767)
                self.gpu_iq_out[:] = self._buf_iq_out_3d.ravel().astype(cp.int16)
                if not hasattr(self, '_post_limit_cnt'):
                    self._post_limit_cnt = 0
                self._post_limit_cnt += 1
                if self._post_limit_cnt <= 10 or self._post_limit_cnt % 2000 == 0:
                    print(f"[POST-LIMIT] slot#{self.slot_counter} "
                          f"out_peak={out_peak:.0f} rescale={_rescale:.4f} "
                          f"cnt={self._post_limit_cnt}")

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

            _POST_LIMIT = 28000.0
            out_peak = float(cp.max(cp.abs(self.gpu_out)))
            if out_peak > _POST_LIMIT:
                _rescale = _POST_LIMIT / out_peak
                self.gpu_out *= cp.float64(_rescale)
                n_rx = self.n_rx
                self._tmp_out_2d_final = self.gpu_out.reshape(self.total_cpx, n_rx)
                self._buf_iq_out_3d[:, :, 0] = cp.clip(
                    cp.around(self._tmp_out_2d_final.real), -32768, 32767)
                self._buf_iq_out_3d[:, :, 1] = cp.clip(
                    cp.around(self._tmp_out_2d_final.imag), -32768, 32767)
                self.gpu_iq_out[:] = self._buf_iq_out_3d.ravel().astype(cp.int16)
                if not hasattr(self, '_post_limit_cnt'):
                    self._post_limit_cnt = 0
                self._post_limit_cnt += 1
                if self._post_limit_cnt <= 10 or self._post_limit_cnt % 2000 == 0:
                    print(f"[POST-LIMIT] slot#{self.slot_counter} "
                          f"out_peak={out_peak:.0f} rescale={_rescale:.4f} "
                          f"cnt={self._post_limit_cnt}")

            if do_dual:
                e_gpu_e.record(self.stream)

            _do_diag = (self.slot_counter in (5, 10, 50, 100, 200, 500, 1000)
                        or (self.slot_counter > 0 and self.slot_counter % 500 == 0))
            if _do_diag:
                self.stream.synchronize()
                in_rms = float(cp.sqrt(cp.mean(self.gpu_iq_in.astype(cp.float64)**2)))
                in_max = float(cp.max(cp.abs(self.gpu_iq_in.astype(cp.float64))))
                in_nonzero = int(cp.count_nonzero(self.gpu_iq_in))
                h_mean_sq = float(cp.mean(cp.abs(self.gpu_H)**2))
                h_max = float(cp.max(cp.abs(self.gpu_H)))
                out_rms = float(cp.sqrt(cp.mean(cp.abs(self.gpu_out)**2)))
                out_max = float(cp.max(cp.abs(self.gpu_out)))
                noise_pwr = 0.0
                if noise_on and noise_std_abs is not None:
                    noise_pwr = 2.0 * float(noise_std_abs)**2
                    sig_pwr = max(out_rms**2 - noise_pwr, 1e-30)
                    snr_est = 10.0 * np.log10(sig_pwr / max(noise_pwr, 1e-30))
                else:
                    sig_pwr = out_rms**2
                    snr_est = float('inf')
                expected_out_pwr = in_rms**2 * h_mean_sq * self.n_tx / max(4.0 * self.fft_size, 1)
                print(f"[SIGNAL DIAG] slot#{self.slot_counter} "
                      f"({self.n_tx}tx→{self.n_rx}rx) graph={self.graph_captured} "
                      f"in_rms={in_rms:.1f} in_max={in_max:.0f} in_nz={in_nonzero}/{self.total_int16_in} "
                      f"|H|²={h_mean_sq:.6f} |H|_max={h_max:.4f} "
                      f"out_rms={out_rms:.1f} out_max={out_max:.0f} "
                      f"noise_pwr={noise_pwr:.1f} sig_pwr={sig_pwr:.1f} "
                      f"SNR_est={snr_est:.1f}dB")

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

    def try_put_batch(self, data_batch, timeout=2.0):
        """Wait up to *timeout* seconds for ring buffer space, then drop
        any remaining items.  Returns the number of items inserted."""
        import cupy as _cp
        n = data_batch.shape[0]
        s = self.sync
        inserted = 0
        with s.not_full:
            deadline = time.monotonic() + timeout
            for i in range(n):
                while s.count.value == s.maxlen:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if inserted > 0:
                            _cp.cuda.Device(0).synchronize()
                            s.not_empty.notify_all()
                        return inserted
                    s.not_full.wait(timeout=min(remaining, 0.1))
                self.buffer[s.write_idx.value] = data_batch[i]
                s.write_idx.value = (s.write_idx.value + 1) % s.maxlen
                s.count.value += 1
                inserted += 1
            if inserted > 0:
                _cp.cuda.Device(0).synchronize()
                s.not_empty.notify_all()
        return inserted


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

    def get_batch_view(self, n, timeout=0.1):
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
# UnifiedChannelProducerProcess — single TF context for all UEs (v4)
# ============================================================================

class UnifiedChannelProducerProcess(_mp_ctx.Process):
    """단일 프로세스에서 N_UE개 UE의 채널을 통합 생성.
    N_UE 배치 차원을 활용하여 Sionna 한 번 호출로 전체 UE 채널을 생성하고,
    UE별로 슬라이싱하여 각 링버퍼에 non-blocking으로 분배한다."""

    def __init__(self, config, syncs, handle_queue, stop_event):
        super().__init__(daemon=True)
        self.config = config
        self.syncs = syncs
        self.handle_queue = handle_queue
        self.stop_event = stop_event

    def run(self):
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.get('gpu_num', 0))
        _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        _os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

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
        num_ues = cfg['num_ues']

        if 'ray_data_stacked' in cfg:
            rd = cfg['ray_data_stacked']
            phi_r_rays = _tf.convert_to_tensor(rd['phi_r'])
            phi_t_rays = _tf.convert_to_tensor(rd['phi_t'])
            theta_r_rays = _tf.convert_to_tensor(rd['theta_r'])
            theta_t_rays = _tf.convert_to_tensor(rd['theta_t'])
            power_rays = _tf.convert_to_tensor(rd['power'])
            tau_rays = _tf.convert_to_tensor(rd['tau'])
            print(f"[v4 UnifiedChannelProducerProcess] Stacked P1B ray_data loaded (N_UE={num_ues})")
        elif 'ray_data' in cfg:
            rd = cfg['ray_data']
            phi_r_rays = _tf.convert_to_tensor(rd['phi_r'])
            phi_t_rays = _tf.convert_to_tensor(rd['phi_t'])
            theta_r_rays = _tf.convert_to_tensor(rd['theta_r'])
            theta_t_rays = _tf.convert_to_tensor(rd['theta_t'])
            power_rays = _tf.convert_to_tensor(rd['power'])
            tau_rays = _tf.convert_to_tensor(rd['tau'])
            print(f"[v4 UnifiedChannelProducerProcess] P1B ray_data loaded (N_UE={num_ues})")
        else:
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
        _xpr_mean = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
        _xpr_std  = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}
        _scenario = cfg.get('scenario', 'UMa-NLOS')
        mean_xpr = _xpr_mean.get(_scenario, 7)
        stddev_xpr = _xpr_std.get(_scenario, 4)
        print(f"[ChannelProducer] scenario={_scenario} → XPR mean={mean_xpr}dB std={stddev_xpr}dB")
        pol_mode = cfg.get('polarization', 'single')
        pol_type = "cross" if pol_mode == "dual" else "V"

        BSexample = {
            "num_rows_per_panel": gnb_ny, "num_cols_per_panel": gnb_nx,
            "num_rows": 1, "num_cols": 1,
            "polarization": pol_mode, "polarization_type": pol_type,
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
            polarization=pol_mode, polarization_type=pol_type,
            antenna_pattern='omni',
            carrier_frequency=carrier_freq)

        print(f"[v4 UnifiedChannelProducerProcess] Sionna init start (N_UE={N_UE})")

        _ue_speeds_list = cfg.get('ue_speeds')
        if _ue_speeds_list and len(_ue_speeds_list) >= N_UE:
            _speed_vec = [_ue_speeds_list[i] for i in range(N_UE)]
            _mean_per_ue = _tf.constant(
                [[_speed_vec] * batch_size], dtype=_tf.float32)
            _mean_per_ue = _tf.reshape(_mean_per_ue, [batch_size, N_UE, 1])
            _mean_per_ue = _tf.broadcast_to(_mean_per_ue, [batch_size, N_UE, 3])
            velocities = _tf.abs(_tf.random.normal(
                shape=[batch_size, N_UE, 3], dtype=_tf.float32) * 0.1 + _mean_per_ue)
            print(f"[ChannelProducer] per-UE speeds (m/s): {_speed_vec}")
        else:
            velocities = _tf.abs(_tf.random.normal(
                shape=[batch_size, N_UE, 3], mean=Speed_local, stddev=0.1, dtype=_tf.float32))
        # ── 3GPP TR 38.901 topology computation ────────────────────
        _bs_h = float(cfg.get('bs_height_m', 25.0))
        _ue_h = float(cfg.get('ue_height_m', 1.5))
        _min_d = float(cfg.get('min_ue_dist_m', 35.0))
        _max_d = float(cfg.get('max_ue_dist_m', 500.0))
        _sf_std = float(cfg.get('shadow_fading_std_dB', 6.0))
        _kf_mean = cfg.get('k_factor_mean_dB')
        _kf_std = cfg.get('k_factor_std_dB')

        import math as _math

        # 3GPP scenario defaults (TR 38.901 Table 7.4.1-1)
        _SCENARIO_DEFAULTS = {
            "UMi-LOS":  {"bs_h": 10, "min_d": 10, "max_d": 150, "sf_std": 4.0,
                          "kf_mean": 9.0, "kf_std": 5.0},
            "UMi-NLOS": {"bs_h": 10, "min_d": 10, "max_d": 150, "sf_std": 7.82,
                          "kf_mean": None, "kf_std": None},
            "UMa-LOS":  {"bs_h": 25, "min_d": 35, "max_d": 500, "sf_std": 4.0,
                          "kf_mean": 9.0, "kf_std": 3.5},
            "UMa-NLOS": {"bs_h": 25, "min_d": 35, "max_d": 500, "sf_std": 6.0,
                          "kf_mean": None, "kf_std": None},
        }
        _sdef = _SCENARIO_DEFAULTS.get(_scenario, _SCENARIO_DEFAULTS["UMa-NLOS"])
        if _kf_mean is None:
            _kf_mean = _sdef["kf_mean"]
        if _kf_std is None:
            _kf_std = _sdef["kf_std"]

        # UE horizontal distance: uniform in [min_d, max_d] per UE
        d_2d_np = np.random.uniform(_min_d, _max_d, size=(batch_size, N_UE))
        delta_h = _bs_h - _ue_h
        d_3d_np = np.sqrt(d_2d_np**2 + delta_h**2)

        # Azimuth angle from BS to UE (random in sector)
        _sector_half = float(cfg.get('sector_half_deg', 90.0)) * _math.pi / 180.0
        azimuth_np = np.random.uniform(-_sector_half, _sector_half,
                                        size=(batch_size, N_UE))

        # LOS probability model (TR 38.901 Table 7.4.2-1)
        def _los_prob(d2d, scenario):
            if "LOS" in scenario:
                return np.ones_like(d2d)
            if "NLOS" in scenario:
                return np.zeros_like(d2d)
            if "UMi" in scenario:
                return np.where(d2d <= 18, 1.0, 18.0/d2d * (1 - np.exp(-d2d/36)) + np.exp(-d2d/36))
            else:  # UMa
                c_prime = np.where(_ue_h <= 13, 0.0,
                                   (((_ue_h - 13.0)/10.0)**1.5) * 1.0)
                p_los = (np.where(d2d <= 18, 1.0,
                         (18.0/d2d + np.exp(-d2d/63) * (1 - 18.0/d2d)) *
                         (1 + c_prime * 5.0/4.0 * (d2d/100)**3 * np.exp(-d2d/150))))
                return np.clip(p_los, 0, 1)

        los_prob_np = _los_prob(d_2d_np, _scenario)
        los_draw = np.random.uniform(0, 1, size=(batch_size, N_UE))
        los_np = (los_draw < los_prob_np)  # True = LOS

        # Elevation angles
        elev_rad = np.arctan2(delta_h, d_2d_np)
        los_aoa_np = azimuth_np
        los_aod_np = azimuth_np + _math.pi  # opposite direction
        los_zoa_np = _math.pi / 2 - elev_rad
        los_zod_np = _math.pi / 2 + elev_rad

        # Shadow fading (log-normal per UE)
        sf_dB = np.random.normal(0, _sf_std, size=(batch_size, N_UE))

        # K-factor (LOS only, per UE)
        if _kf_mean is not None and _kf_std is not None:
            kf_dB = np.random.normal(_kf_mean, _kf_std, size=(batch_size, N_UE))
            kf_dB = np.where(los_np, kf_dB, -100.0)  # NLOS → K=-100dB (≈0 linear)
        else:
            kf_dB = np.full((batch_size, N_UE), -100.0)

        print(f"[ChannelProducer] 3GPP topology: BS_h={_bs_h}m UE_h={_ue_h}m "
              f"d_2d=[{_min_d:.0f},{_max_d:.0f}]m SF_σ={_sf_std}dB "
              f"K_μ={_kf_mean}dB K_σ={_kf_std}dB")
        print(f"[ChannelProducer] d_3d: mean={d_3d_np.mean():.1f}m "
              f"min={d_3d_np.min():.1f}m max={d_3d_np.max():.1f}m")
        print(f"[ChannelProducer] LOS ratio: {los_np.mean()*100:.1f}% "
              f"({los_np.sum()}/{los_np.size})")
        print(f"[ChannelProducer] Shadow fading: mean={sf_dB.mean():.2f}dB std={sf_dB.std():.2f}dB")
        if _kf_mean is not None:
            _kf_los = kf_dB[los_np]
            if len(_kf_los) > 0:
                print(f"[ChannelProducer] K-factor (LOS): mean={_kf_los.mean():.1f}dB std={_kf_los.std():.1f}dB")

        # Expand to [batch, N_BS, N_UE] (N_BS=1)
        los_aoa = _tf.constant(los_aoa_np[:, np.newaxis, :], dtype=_tf.float32)
        los_aod = _tf.constant(los_aod_np[:, np.newaxis, :], dtype=_tf.float32)
        los_zoa = _tf.constant(los_zoa_np[:, np.newaxis, :], dtype=_tf.float32)
        los_zod = _tf.constant(los_zod_np[:, np.newaxis, :], dtype=_tf.float32)
        los = _tf.constant(los_np[:, np.newaxis, :])
        distance_3d = _tf.constant(d_3d_np[:, np.newaxis, :], dtype=_tf.float32)

        tx_orientations = _tf.random.normal(
            shape=[batch_size, N_BS, 3], mean=0, stddev=PI/5, dtype=_tf.float32)
        rx_orientations = _tf.random.normal(
            shape=[batch_size, N_UE, 3], mean=0, stddev=PI/5, dtype=_tf.float32)

        # Apply shadow fading to ray powers
        sf_linear = 10.0 ** (-sf_dB / 10.0)  # [batch, N_UE]
        _sf_scale = _tf.constant(
            sf_linear[:, np.newaxis, :, np.newaxis, np.newaxis],
            dtype=power_rays.dtype)
        power_rays_sf = power_rays * _sf_scale

        xpr_pdp = 10**(_tf.random.normal(
            shape=[batch_size, N_BS, N_UE, 1, phi_r_rays.shape[-1]],
            mean=mean_xpr, stddev=stddev_xpr
        )/10)
        PDP = Rays(
            delays=tau_rays, powers=power_rays_sf, aoa=phi_r_rays, aod=phi_t_rays,
            zoa=theta_r_rays, zod=theta_t_rays, xpr=xpr_pdp)

        topology = Topology(
            velocities, "rx", los_aoa, los_aod, los_zoa, los_zod,
            los, distance_3d, tx_orientations, rx_orientations)

        gen = ChannelCoefficientsGeneratorJIN(
            carrier_freq, scs_local, ArrayTX, ArrayRX, False)
        h_field, aoa, zoa = gen._H_PDP_FIX(topology, PDP, N_FFT_local, scs_local)
        h_field = _tf.transpose(h_field, [0, 3, 5, 6, 1, 2, 7, 4])
        aoa = _tf.transpose(aoa, [0, 3, 1, 2, 4])
        zoa = _tf.transpose(zoa, [0, 3, 1, 2, 4])

        print(f"[v4 UnifiedChannelProducerProcess] Sionna init done, allocating {num_ues} GPU ring buffers")

        shape = cfg['shape']
        ring_buffers = []
        for k in range(num_ues):
            rb_k = IPCRingBufferProducer(shape, _cp.complex128, self.syncs[k])
            self.handle_queue.put(rb_k.ipc_handle)
            ring_buffers.append(rb_k)
            print(f"[v4 UnifiedChannelProducerProcess] UE[{k}] IPC handle sent")

        use_xla = cfg.get('use_xla', False)
        xla_tag = " +XLA" if use_xla else ""
        print(f"[v4 UnifiedChannelProducerProcess] Entering generation loop (pid={_os.getpid()}, N_UE={num_ues}{xla_tag})")

        params = dict(Fs=Fs_local, scs=scs_local,
                      N_UE=N_UE, N_BS=N_BS,
                      N_UE_active=cfg['num_rx'], N_BS_serving=cfg['num_tx'])

        sample_times_fixed = _tf.cast(
            _tf.range(buffer_symbol_size), gen.rdtype
        ) / _tf.constant(params['scs'], gen.rdtype)
        ActiveUE_fixed = _tf.constant(
            random_binary_mask_tf_complex64(params['N_UE'], k=params['N_UE_active']),
            dtype=_tf.complex64)
        ServingBS_fixed = _tf.constant(
            random_binary_mask_tf_complex64(params['N_BS'], k=params['N_BS_serving']),
            dtype=_tf.complex64)

        def _generate_eager():
            h_delay, _, _, _ = gen._H_TTI_sequential_fft_o_ELW2_noProfile(
                topology, ActiveUE_fixed, ServingBS_fixed, sample_times_fixed,
                h_field, aoa, zoa)
            h_delay = h_delay[0, :, :, 0, :, :, :]
            h_delay = _tf.transpose(h_delay, [3, 0, 1, 2, 4])
            h_c128 = _tf.cast(h_delay, _tf.complex128)
            energy = _tf.reduce_sum(
                _tf.abs(h_c128) ** 2, axis=[1, 2, 4], keepdims=True)
            n_elem = _tf.cast(
                _tf.shape(h_c128)[1] * _tf.shape(h_c128)[2] * _tf.shape(h_c128)[4],
                _tf.float64)
            scale = _tf.cast(_tf.sqrt(n_elem / (energy + 1e-30)), h_c128.dtype)
            return h_c128 * scale

        if use_xla:
            generate_fn = _tf.function(_generate_eager, jit_compile=True)
            print(f"[v4 XLA] Compiling XLA graph (first call will be slow)...")
            _t_xla = time.time()
            _ = generate_fn()
            print(f"[v4 XLA] Compilation done ({time.time()-_t_xla:.1f}s)")
        else:
            generate_fn = _generate_eager

        symbol_counter = 0
        drop_count = [0] * num_ues

        while not self.stop_event.is_set():
            try:
                h_c128_norm = generate_fn()

                try:
                    h_cp = _cp.from_dlpack(
                        _tf.experimental.dlpack.to_dlpack(h_c128_norm)).copy()
                except Exception:
                    h_cp = _cp.asarray(h_c128_norm.numpy())

                h_cp = _cp.fft.fft(h_cp, axis=-1)

                wideband_energy = float(_cp.mean(_cp.abs(h_cp) ** 2))
                if wideband_energy > 0:
                    h_cp = h_cp / _cp.sqrt(_cp.float64(wideband_energy))

                for ue_k in range(num_ues):
                    h_ue_k = h_cp[:, :, :, ue_k, :]
                    inserted = ring_buffers[ue_k].try_put_batch(h_ue_k)
                    dropped = h_ue_k.shape[0] - inserted
                    if dropped > 0:
                        drop_count[ue_k] += dropped
                        if drop_count[ue_k] % (buffer_symbol_size * 10) < dropped:
                            print(f"[v4 WARN] UE[{ue_k}] ring buffer full, dropped {dropped} symbols (total={drop_count[ue_k]})")

                symbol_counter += buffer_symbol_size

            except _tf.errors.ResourceExhaustedError as e:
                print(f"[v4 UnifiedChannelProducerProcess] GPU OOM — stopping: {e}")
                break
            except Exception as e:
                print(f"[v4 UnifiedChannelProducerProcess] ERROR in generation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        drop_str = ", ".join(f"UE{k}={drop_count[k]}" for k in range(num_ues))
        print(f"[v4 UnifiedChannelProducerProcess] Stopped (symbols={symbol_counter}, dropped: {drop_str})")


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
                 num_ues=1,
                 polarization="single",
                 p1b_npz=None, ue_rx_indices=None,
                 use_xla=False,
                 bs_height_m=25.0, ue_height_m=1.5,
                 isd_m=500, min_ue_dist_m=35, max_ue_dist_m=500,
                 shadow_fading_std_dB=6.0,
                 k_factor_mean_dB=None, k_factor_std_dB=None):
        self.mode = mode
        self.num_ues = num_ues
        self.polarization = polarization
        self.p1b_npz = p1b_npz
        self.ue_rx_indices = ue_rx_indices
        self.use_xla = use_xla
        self.bs_height_m = bs_height_m
        self.ue_height_m = ue_height_m
        self.isd_m = isd_m
        self.min_ue_dist_m = min_ue_dist_m
        self.max_ue_dist_m = max_ue_dist_m
        self.shadow_fading_std_dB = shadow_fading_std_dB
        self.k_factor_mean_dB = k_factor_mean_dB
        self.k_factor_std_dB = k_factor_std_dB
        self._stalled_ue_logged = set()
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
        _use_graph = self.use_cuda_graph
        if _use_graph:
            print("[v4] CUDA Graph DISABLED for DL BLER diagnostic — using _gpu_compute_core directly")
            _use_graph = False
        pipeline_common = dict(
            enable_gpu=self.enable_gpu,
            use_pinned_memory=self.use_pinned_memory,
            use_cuda_graph=_use_graph,
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
        self._last_ch_cache = [None] * self.num_ues
        self._dl_ch_count = 0
        self._dl_bypass_wrap_count = 0
        self._dl_bypass_timeout_count = 0

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
            print(f"[v4] UE[{k}] pipelines created (DL: {self.gnb_ant}tx→{self.ue_ant}rx, UL: {self.ue_ant}tx→{self.gnb_ant}rx)")

        self.pipeline_dl = self.pipelines_dl[0]
        self.pipeline_ul = self.pipelines_ul[0]
        self.gpu_slot_pipeline = self.pipeline_dl
        self._noise_buffers_dl = self._noise_buffers_dl_list[0]
        self._noise_buffers_ul = self._noise_buffers_ul_list[0]

        if not self.custom_channel:
            print(f"[v4] Bypass mode — {N} UE(s), no channel")
            return

        self.N_UE = N
        self.N_BS = 1
        self.num_rx = N
        self.num_tx = 1

        print(f"[v4] Starting UnifiedChannelProducerProcess via spawn (N_UE={N})...")

        scenario = getattr(self, 'scenario', 'UMa-NLOS')
        syncs = [IPCRingBufferSync(maxlen=buffer_len, ctx=_mp_ctx) for _ in range(N)]
        handle_q = _mp_ctx.Queue()
        stop_ev = _mp_ctx.Event()

        config = {
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
            'num_ues': N,
            'Speed': Speed,
            'ue_speeds': ue_speeds,
            'scenario': scenario,
            'gpu_num': gpu_num,
            'use_xla': self.use_xla,
            'polarization': self.polarization,
            'bs_height_m': self.bs_height_m,
            'ue_height_m': self.ue_height_m,
            'isd_m': self.isd_m,
            'min_ue_dist_m': self.min_ue_dist_m,
            'max_ue_dist_m': self.max_ue_dist_m,
            'shadow_fading_std_dB': self.shadow_fading_std_dB,
            'k_factor_mean_dB': self.k_factor_mean_dB,
            'k_factor_std_dB': self.k_factor_std_dB,
        }

        if self.p1b_npz and self.ue_rx_indices:
            config['ray_data_stacked'] = load_p1b_stacked(
                self.p1b_npz, self.ue_rx_indices)
            rx_str = ",".join(str(r) for r in self.ue_rx_indices)
            print(f"[v4] P1B stacked ray loaded: RX=[{rx_str}], N_UE={N}")

        proc = UnifiedChannelProducerProcess(config, syncs, handle_q, stop_ev)
        proc.start()
        print(f"[v4] UnifiedChannelProducerProcess started (pid={proc.pid}, N_UE={N})")

        for k in range(N):
            try:
                ipc_handle = handle_q.get(timeout=120)
            except Exception as e:
                print(f"[v4] ERROR: UE[{k}] IPC handle not received within 120s: {e}")
                if proc.is_alive():
                    proc.terminate()
                raise RuntimeError(f"UnifiedChannelProducerProcess failed to initialize (UE[{k}] handle)") from e

            consumer_k = IPCRingBufferConsumer(
                ipc_handle,
                config['shape'],
                cp.complex128,
                syncs[k])

            self.channel_buffers.append(consumer_k)
            print(f"[v4] UE[{k}] IPCRingBufferConsumer connected (CUDA IPC)")

        self.channel_producers.append(proc)
        self._channel_stop_events.append(stop_ev)

        self.channel_buffer = self.channel_buffers[0]
        self.producer = self.channel_producers[0]

        print(f"[v4] Unified Channel Proxy initialized: {N} UE(s), "
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
        Handles circular buffer wrap via circ_read/circ_write."""
        arr_in = self.ipc.circ_read(src_ptr, ts, nsamps, src_nbAnt, src_cir_size, cp.int16)

        arr_out, wraps_out = self.ipc.get_gpu_array_at(
            dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, cp.int16)
        use_tmp_out = wraps_out
        if use_tmp_out:
            arr_out = cp.zeros(nsamps * dst_nbAnt * 2, dtype=cp.int16)

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
        if use_tmp_out:
            self.ipc.circ_write(dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, arr_out)
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
        Falls back to bypass on timeout (ChannelProducerProcess may be slow/dead).
        Handles circular buffer wrap via circ_read/circ_write."""
        arr_in = src_ipc.circ_read(src_ptr, ts, nsamps, src_nbAnt, src_cir_size, cp.int16)

        arr_out, wraps_out = dst_ipc.get_gpu_array_at(
            dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, cp.int16)
        use_tmp_out = wraps_out
        if use_tmp_out:
            arr_out = cp.zeros(nsamps * dst_nbAnt * 2, dtype=cp.int16)

        n_held = 0
        try:
            channels, n_held = self.channel_buffers[ue_idx].get_batch_view(N_SYM)
            self._last_ch_cache[ue_idx] = channels.copy()
        except TimeoutError:
            cached = self._last_ch_cache[ue_idx]
            if cached is not None:
                channels = cached
            else:
                src_ipc.bypass_copy(dst_ptr, src_ptr, ts, nsamps,
                                    src_nbAnt, src_cir_size, dst_nbAnt, dst_cir_size)
                if direction == "DL" and ue_idx == 0:
                    self._dl_bypass_timeout_count += 1
                return

        if channels.shape[0] < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            n_r, n_t = channels.shape[1], channels.shape[2]
            n_pad = N_SYM - channels.shape[0]
            pad = lib.zeros((n_pad, n_r, n_t, FFT_SIZE), dtype=channels.dtype)
            for j in range(min(n_r, n_t)):
                pad[:, j, j, :] = 1.0
            channels = lib.concatenate([channels, pad])

        if direction == "DL":
            csinet_hook = get_csinet_hook()
            if csinet_hook is not None and csinet_hook.enabled:
                csinet_hook.capture(0, ue_idx, channels)

        if DL_USE_IDENTITY_CHANNEL and direction == "DL":
            n_r, n_t = channels.shape[1], channels.shape[2]
            channels = cp.zeros_like(channels)
            for j in range(min(n_r, n_t)):
                channels[:, j, j, :] = 1.0

        if direction == "UL":
            channels = channels.transpose(0, 2, 1, 3)

        pipeline = self.pipelines_dl[ue_idx] if direction == "DL" else self.pipelines_ul[ue_idx]

        if DL_USE_IDENTITY_CHANNEL and direction == "DL" and ue_idx == 0:
            if not hasattr(self, '_identity_diag_cnt'):
                self._identity_diag_cnt = 0
            self._identity_diag_cnt += 1
            if self._identity_diag_cnt in (1, 5, 50, 500, 2000):
                in_snap = arr_in[:min(len(arr_in), 2048)].copy()

        pipeline.process_slot_ipc(
            arr_in, channels, pathLossLinear, snr_dB, noise_enabled, arr_out, noise_std_abs
        )

        if DL_USE_IDENTITY_CHANNEL and direction == "DL" and ue_idx == 0:
            if self._identity_diag_cnt in (1, 5, 50, 500, 2000):
                out_snap = arr_out[:min(len(arr_out), 2048)].copy()
                cp.cuda.Stream.null.synchronize()
                diff = (in_snap.astype(cp.float64) - out_snap.astype(cp.float64))
                mae = float(cp.mean(cp.abs(diff)))
                max_err = float(cp.max(cp.abs(diff)))
                in_nz = int(cp.count_nonzero(in_snap))
                out_nz = int(cp.count_nonzero(out_snap))
                print(f"[IDENTITY CH DIAG] #{self._identity_diag_cnt} "
                      f"ts={ts} MAE={mae:.3f} MAX_ERR={max_err:.0f} "
                      f"in_nz={in_nz}/2048 out_nz={out_nz}/2048 "
                      f"in[0:8]={in_snap[:8].tolist()} out[0:8]={out_snap[:8].tolist()}")

        if use_tmp_out:
            dst_ipc.circ_write(dst_ptr, ts, nsamps, dst_nbAnt, dst_cir_size, arr_out)
        if n_held > 0:
            self.channel_buffers[ue_idx].release_batch(n_held)

    def _ipc_dl_broadcast(self, start_ts, delta):
        """DL Broadcast: gNB dl_tx → per-UE channel → UE[k] dl_rx."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_dl[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0

        if apply_ch:
            self._dl_ch_count += 1

        for k in range(self.num_ues):
            pos = int(start_ts)
            remaining = int(delta)

            while remaining >= slot_samples and apply_ch:
                if PURE_DL_BYPASS:
                    self.ipc_gnb.bypass_copy(
                        self.ipc_ues[k].gpu_dl_rx_ptr,
                        self.ipc_gnb.gpu_dl_tx_ptr,
                        pos, slot_samples,
                        self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                        self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)
                else:
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

            if remaining > 0 and not apply_ch:
                self.ipc_gnb.bypass_copy(
                    self.ipc_ues[k].gpu_dl_rx_ptr, self.ipc_gnb.gpu_dl_tx_ptr,
                    pos, remaining,
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size,
                    self.ipc_ues[k].dl_rx_nbAnt, self.ipc_ues[k].dl_rx_cir_size)

            actual_end = int(start_ts + delta - 1) if not apply_ch else int(pos - 1)
            if actual_end > 0:
                self.ipc_ues[k].set_last_dl_rx_ts(actual_end)

        if not hasattr(self, '_dl_bcast_diag_count'):
            self._dl_bcast_diag_count = 0
        self._dl_bcast_diag_count += 1
        if self._dl_bcast_diag_count in (1, 2, 5, 10, 50, 100) or self._dl_bcast_diag_count % 500 == 0:
            try:
                gnb_arr = self.ipc_gnb.circ_read(
                    self.ipc_gnb.gpu_dl_tx_ptr, int(start_ts), min(int(delta), slot_samples),
                    self.ipc_gnb.dl_tx_nbAnt, self.ipc_gnb.dl_tx_cir_size)
                gnb_f = gnb_arr.astype(cp.float64)
                gnb_rms = float(cp.sqrt(cp.mean(gnb_f**2)))
                gnb_max = float(cp.max(cp.abs(gnb_f)))
                gnb_nz = int(cp.count_nonzero(gnb_arr))
                ue0_arr = self.ipc_ues[0].circ_read(
                    self.ipc_ues[0].gpu_dl_rx_ptr, int(start_ts), min(int(delta), slot_samples),
                    self.ipc_ues[0].dl_rx_nbAnt, self.ipc_ues[0].dl_rx_cir_size)
                ue0_f = ue0_arr.astype(cp.float64)
                ue0_rms = float(cp.sqrt(cp.mean(ue0_f**2)))
                ue0_max = float(cp.max(cp.abs(ue0_f)))
                ue0_nz = int(cp.count_nonzero(ue0_arr))
                cp.cuda.Stream.null.synchronize()
                print(f"[DL BCAST DIAG] call#{self._dl_bcast_diag_count} "
                      f"ts={start_ts} delta={delta} slots={slots} apply_ch={apply_ch} "
                      f"bypass(wrap={self._dl_bypass_wrap_count},timeout={self._dl_bypass_timeout_count}) "
                      f"gNB_TX: rms={gnb_rms:.1f} max={gnb_max:.0f} nz={gnb_nz} "
                      f"UE0_RX: rms={ue0_rms:.1f} max={ue0_max:.0f} nz={ue0_nz}")
            except Exception as e:
                print(f"[DL BCAST DIAG] call#{self._dl_bcast_diag_count} ERROR: {e}")

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _wait_ul_backpressure(self, write_end_ts):
        """Block until gNB has consumed enough UL data so the circular buffer
        won't be overwritten.  Skips the check before the gNB's first read
        (consumer_ts == 0) and when the gNB hasn't synced yet (gap > cir)."""
        cir_time = self.ipc_gnb.cir_time
        max_ahead = cir_time - cir_time // 4
        deadline = time.perf_counter() + 0.1
        while True:
            cts = self.ipc_gnb.get_ul_consumer_ts()
            if cts == 0:
                return
            gap = write_end_ts - cts
            if gap > cir_time:
                return
            if gap <= max_ahead:
                return
            if time.perf_counter() >= deadline:
                return
            time.sleep(0.001)

    def _ipc_ul_combine(self, start_ts, delta, active_ues=None,
                        ue_ts_offsets=None):
        """UL Combine: UE[k] ul_tx → per-UE channel → superposition → gNB ul_rx.
        start_ts is on the gNB-grid (slot-aligned).  ue_ts_offsets is a dict
        {ue_idx: offset} — per-UE offset to read from each UE's timestamp space."""
        t0 = time.perf_counter()
        slot_samples = self.pipelines_ul[0].total_cpx
        apply_ch = self.ch_en and self.custom_channel
        slots = 0
        ues = active_ues if active_ues is not None else range(self.num_ues)
        first_ue = True

        if not apply_ch:
            for k in ues:
                pos_gnb = int(start_ts)
                remaining = int(delta)
                _k_offset = (ue_ts_offsets.get(k, 0) if ue_ts_offsets else 0)
                while remaining > 0:
                    n = min(remaining, slot_samples)
                    pos_ue = pos_gnb + _k_offset
                    arr_ue = self.ipc_ues[k].circ_read(
                        self.ipc_ues[k].gpu_ul_tx_ptr, pos_ue, n,
                        self.ipc_ues[k].ul_tx_nbAnt,
                        self.ipc_ues[k].ul_tx_cir_size, cp.int16)
                    if UL_GAIN_LINEAR != 1.0:
                        arr_f = arr_ue.astype(cp.float32) * UL_GAIN_LINEAR
                        arr_ue = cp.clip(arr_f, -32768, 32767).astype(cp.int16)
                    self.ipc_gnb.circ_write(
                        self.ipc_gnb.gpu_ul_rx_ptr, pos_gnb, n,
                        self.ipc_gnb.ul_rx_nbAnt,
                        self.ipc_gnb.ul_rx_cir_size, arr_ue)
                    _slot_end_ts = int(pos_gnb + n - 1)
                    if _slot_end_ts > self._ul_rx_ts_high:
                        self._ul_rx_ts_high = _slot_end_ts
                        self.ipc_gnb.set_last_ul_rx_ts(_slot_end_ts)
                    pos_gnb += n
                    remaining -= n
                    if first_ue:
                        slots += 1
                first_ue = False
        else:
            pos = int(start_ts)
            remaining = int(delta)

            while remaining >= slot_samples:
                self._ipc_ul_superposition_slot(pos, slot_samples,
                                                active_ues=active_ues,
                                                ue_ts_offsets=ue_ts_offsets)
                _slot_end_ts = int(pos + slot_samples - 1)
                if _slot_end_ts > self._ul_rx_ts_high:
                    self._ul_rx_ts_high = _slot_end_ts
                    self.ipc_gnb.set_last_ul_rx_ts(_slot_end_ts)
                pos += slot_samples
                remaining -= slot_samples
                slots += 1

        ms = 1000 * (time.perf_counter() - t0)
        return ms, max(slots, 1)

    def _ipc_ul_superposition_slot(self, ts, nsamps, active_ues=None,
                                    ue_ts_offsets=None):
        """Apply per-UE UL channels and sum into gNB ul_rx for one slot.
        ts: gNB-grid timestamp for writing to gNB ul_rx SHM.
        ue_ts_offsets: dict {ue_idx: offset} — per-UE offset added to ts
        when reading from UE ul_tx SHM (each UE has its own timestamp space)."""
        UL_BYPASS_CHANNEL = PURE_DL_BYPASS  # match DL bypass mode
        self._ul_accum[:] = 0
        ues = active_ues if active_ues is not None else range(self.num_ues)
        _ul_diag_slot = getattr(self, '_ul_diag_count', 0)

        for k in ues:
            _k_offset = (ue_ts_offsets.get(k, 0) if ue_ts_offsets else 0)
            ts_ue = ts + _k_offset
            arr_in = self.ipc_ues[k].circ_read(
                self.ipc_ues[k].gpu_ul_tx_ptr, ts_ue, nsamps,
                self.ipc_ues[k].ul_tx_nbAnt, self.ipc_ues[k].ul_tx_cir_size, cp.int16)

            _check_diag = (_ul_diag_slot < 50 or _ul_diag_slot % 200 == 0)
            _check_first = not getattr(self, '_first_ue_energy_logged', False)
            if _check_diag or _check_first:
                _in_energy = float(cp.sum(arr_in.astype(cp.float32) ** 2))
                if _in_energy > 0:
                    if _check_diag:
                        print(f"[UL DIAG] slot#{_ul_diag_slot} UE[{k}] "
                              f"ts_gnb={ts} ts_ue={ts_ue} "
                              f"ul_tx energy={_in_energy:.0f} "
                              f"max_abs={int(cp.max(cp.abs(arr_in)))}")
                    if _check_first:
                        self._first_ue_energy_logged = True
                        print(f"[UL DIAG] *** FIRST UE ENERGY *** "
                              f"slot#{_ul_diag_slot} UE[{k}] "
                              f"ts_gnb={ts} ts_ue={ts_ue} "
                              f"energy={_in_energy:.0f} "
                              f"max_abs={int(cp.max(cp.abs(arr_in)))}")

            if UL_BYPASS_CHANNEL:
                arr_f64 = arr_in.astype(cp.float64)
                arr_cpx = arr_f64.reshape(-1, 2)
                self._ul_accum += (arr_cpx[:, 0] + 1j * arr_cpx[:, 1])
            else:
                n_held = 0
                try:
                    channels, n_held = self.channel_buffers[k].get_batch_view(N_SYM)
                    self._last_ch_cache[k] = channels.copy()
                except TimeoutError:
                    cached = self._last_ch_cache[k]
                    if cached is not None:
                        channels = cached
                    else:
                        channels = cp.zeros(
                            (N_SYM, self.ue_ant, self.gnb_ant, FFT_SIZE),
                            dtype=cp.complex128)
                        for j in range(min(self.ue_ant, self.gnb_ant)):
                            channels[:, j, j, :] = 1.0
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
                if n_held > 0:
                    self.channel_buffers[k].release_batch(n_held)

                self._ul_accum += self.pipelines_ul[k].gpu_out

        n_rx = self.gnb_ant
        total = self.pipelines_ul[0].total_cpx
        n_elem = total * n_rx
        if UL_GAIN_LINEAR != 1.0:
            self._ul_accum *= UL_GAIN_LINEAR
        accum_f64 = self._ul_accum.view(cp.float64)
        if not hasattr(self, '_ul_fused_out') or self._ul_fused_out.shape[0] != n_elem * 2:
            self._ul_fused_out = cp.zeros(n_elem * 2, dtype=cp.int16)
        threads = 256
        blocks = (n_elem + threads - 1) // threads
        _fused_clip_cast_kernel((blocks,), (threads,),
                                (accum_f64, self._ul_fused_out, n_elem))

        _check_out_diag = (_ul_diag_slot < 50 or _ul_diag_slot % 200 == 0)
        _check_out_first = not getattr(self, '_first_gnb_energy_logged', False)
        if _check_out_diag or _check_out_first:
            _out_energy = float(cp.sum(self._ul_fused_out.astype(cp.float32) ** 2))
            _out_max = int(cp.max(cp.abs(self._ul_fused_out)))
            if _out_energy > 0 or (_check_out_diag and _ul_diag_slot < 20):
                if _check_out_diag:
                    print(f"[UL DIAG] slot#{_ul_diag_slot} ts_gnb={ts} "
                          f"ts_ue={ts_ue} "
                          f"gnb_ul_rx energy={_out_energy:.0f} max_abs={_out_max}")
                if _check_out_first and _out_energy > 0:
                    self._first_gnb_energy_logged = True
                    print(f"[UL DIAG] *** FIRST GNB UL_RX ENERGY *** "
                          f"slot#{_ul_diag_slot} ts_gnb={ts} "
                          f"ts_ue={ts_ue} "
                          f"energy={_out_energy:.0f} max_abs={_out_max}")

        self.ipc_gnb.circ_write(
            self.ipc_gnb.gpu_ul_rx_ptr, ts, nsamps,
            self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size,
            self._ul_fused_out)

        self._ul_diag_count = _ul_diag_slot + 1

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
            print(f"[v4] NoiseProducer started (noise_len={np_thread.noise_len}, batch={np_thread.BATCH_SIZE})")

        if not self.custom_channel:
            print("[v4] Bypass mode — channel warmup skipped")
            return
        print(f"[v4] Pre-warming {self.num_ues} UE(s) DL/UL pipelines...")
        t0 = time.time()
        passes = GPUSlotPipeline.WARMUP_SLOTS + 1
        if self.channel_producers and hasattr(self.channel_producers[0], 'is_alive'):
            if not self.channel_producers[0].is_alive():
                print(f"[v4 WARN] UnifiedChannelProducerProcess died during warmup")
        for k in range(self.num_ues):
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
        print(f"[v4] All {self.num_ues} UE(s) pipelines ready ({time.time()-t0:.1f}s)")

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
        print(f"[v4] Entering main loop ({N} UE(s), gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant})...")
        print(f"[v4] PURE_DL_BYPASS={PURE_DL_BYPASS} DL_USE_IDENTITY_CHANNEL={DL_USE_IDENTITY_CHANNEL} UL_GAIN={UL_GAIN_LINEAR}")
        dl_count = 0
        ul_count = 0
        t_start = time.time()
        proxy_dl_head = 0
        apply_ch = self.ch_en and self.custom_channel

        if apply_ch:
            proxy_ul_head_combined = 0
        else:
            proxy_ul_heads = [0] * N

        self._ul_rx_ts_high = 0
        self._ul_aligned_to_ue = False
        self._ul_ue_offsets = {}
        self._prev_ue_heads = {}
        self._ul_log_count = 0
        self._ul_diag_count = 0
        self._ka_log_count = 0
        self._first_ue_energy_logged = False
        self._first_gnb_energy_logged = False

        _ul_rx_total_bytes = self.ipc_gnb.ul_rx_cir_size * GPU_IPC_V6_SAMPLE_SIZE
        _ul_rx_mem = cp.cuda.UnownedMemory(
            self.ipc_gnb.gpu_ul_rx_ptr, _ul_rx_total_bytes, owner=None)
        _ul_rx_arr = cp.ndarray(
            _ul_rx_total_bytes // 2, dtype=cp.int16,
            memptr=cp.cuda.MemoryPointer(_ul_rx_mem, 0))
        _ul_rx_arr[:] = 0
        cp.cuda.Stream.null.synchronize()
        print(f"[INIT] Zero-filled gNB ul_rx buffer "
              f"({_ul_rx_total_bytes} bytes, cir_size={self.ipc_gnb.ul_rx_cir_size})")

        try:
            while True:
                processed = False
                ul_processed = False

                # --- DL Broadcast (rate-limited to prevent burst) ---
                cur_dl_ts = self.ipc_gnb.get_last_dl_tx_ts()
                dl_nsamps = self.ipc_gnb.get_last_dl_tx_nsamps()
                if cur_dl_ts > 0 and dl_nsamps > 0:
                    gnb_dl_head = cur_dl_ts + dl_nsamps
                    if gnb_dl_head > proxy_dl_head:
                        if proxy_dl_head == 0:
                            if apply_ch:
                                proxy_dl_head = (gnb_dl_head // total_cpx) * total_cpx
                            else:
                                proxy_dl_head = gnb_dl_head
                            try:
                                _chk_ts = int(cur_dl_ts)
                                _raw_buf_sz = self.ipc_gnb.dl_tx_cir_size * GPU_IPC_V7_SAMPLE_SIZE
                                _raw_mem = cp.cuda.UnownedMemory(
                                    self.ipc_gnb.gpu_dl_tx_ptr, _raw_buf_sz, owner=None)
                                _raw_arr = cp.ndarray(
                                    _raw_buf_sz // 2, dtype=cp.int16,
                                    memptr=cp.cuda.MemoryPointer(_raw_mem, 0))
                                _raw_nz = int(cp.count_nonzero(_raw_arr))
                                _raw_mx = int(cp.max(cp.abs(_raw_arr)))
                                _test_pat = cp.array([42, -42, 100, -100], dtype=cp.int16)
                                _raw_arr[:4] = _test_pat
                                cp.cuda.Stream.null.synchronize()
                                _rb = _raw_arr[:4].copy()
                                _pat_ok = bool(cp.all(_rb == _test_pat))
                                _raw_arr[:4] = 0
                                cp.cuda.Stream.null.synchronize()
                                import time as _time_mod
                                _time_mod.sleep(0.5)
                                _raw_arr2 = cp.ndarray(
                                    _raw_buf_sz // 2, dtype=cp.int16,
                                    memptr=cp.cuda.MemoryPointer(_raw_mem, 0))
                                _post_nz = int(cp.count_nonzero(_raw_arr2))
                                _post_mx = int(cp.max(cp.abs(_raw_arr2)))
                                print(f"[DL IPC TEST] ptr=0x{self.ipc_gnb.gpu_dl_tx_ptr:x} "
                                      f"last_ts={cur_dl_ts} nsamps={dl_nsamps} "
                                      f"BEFORE: nz={_raw_nz}/{len(_raw_arr)} max={_raw_mx} | "
                                      f"WRITE+READ_BACK: pat_ok={_pat_ok} | "
                                      f"AFTER_500ms: nz={_post_nz}/{len(_raw_arr2)} max={_post_mx}")
                            except Exception as _e:
                                print(f"[DL IPC TEST] ERROR: {_e}")
                        delta = int(gnb_dl_head - proxy_dl_head)
                        max_dl_delta = total_cpx * 4
                        if delta > max_dl_delta:
                            delta = max_dl_delta
                        if apply_ch:
                            delta = (delta // total_cpx) * total_cpx
                        if delta > 0:
                            dl_ms, n_slots = self._ipc_dl_broadcast(proxy_dl_head, delta)
                            proxy_dl_head += delta
                            dl_count += n_slots
                            processed = True

                            self._e2e_proxy_dl_accum_ms += dl_ms
                            self._e2e_dl_in_frame += n_slots
                            for _k in range(N):
                                self._e2e_dl_per_ue[_k] += n_slots
                            self._e2e_slot_count += n_slots
                            self._check_e2e_frame("IPC_G1C+OAI")

                # --- UL Processing (active_set based stall detection) ---
                ue_heads = {}
                if apply_ch:
                    for k in range(N):
                        cur_ul = self.ipc_ues[k].get_last_ul_tx_ts()
                        ul_ns = self.ipc_ues[k].get_last_ul_tx_nsamps()
                        if cur_ul > 0 and ul_ns > 0:
                            ue_heads[k] = cur_ul + ul_ns

                    if ue_heads:
                        max_head = max(ue_heads.values())
                        active_set = {k for k, h in ue_heads.items()
                                      if max_head - h < self.ipc_gnb.cir_time}
                        stalled = set(ue_heads.keys()) - active_set

                        for sk in stalled:
                            if sk not in self._stalled_ue_logged:
                                self._stalled_ue_logged.add(sk)
                                print(f"[v3 WARN] UE[{sk}] stall 감지 — "
                                      f"head={ue_heads[sk]}, max_head={max_head}, "
                                      f"gap={max_head - ue_heads[sk]}, "
                                      f"cir_time={self.ipc_gnb.cir_time}")

                        min_active_head = min(ue_heads[k] for k in active_set)

                        if not self._ul_aligned_to_ue:
                            self._ul_aligned_to_ue = True
                            _spf = total_cpx * 20
                            _slots_per_frame = 20
                            _old_head = proxy_ul_head_combined

                            _ue_write_ts = min(
                                self.ipc_ues[k].get_last_ul_tx_ts()
                                for k in active_set)

                            _gnb_consumer = self.ipc_gnb.get_ul_consumer_ts()
                            _gnb_next_slot = int((_gnb_consumer + 1 + total_cpx - 1) // total_cpx)
                            _gnb_slot_mod = _gnb_next_slot % _slots_per_frame

                            _ue_slot = int(_ue_write_ts) // total_cpx
                            _ue_slot_mod = _ue_slot % _slots_per_frame

                            _slot_adj = (_ue_slot_mod - _gnb_slot_mod) % _slots_per_frame
                            _aligned_slot = _ue_slot - _slot_adj
                            _sync_ts = _aligned_slot * total_cpx
                            if _sync_ts <= _gnb_consumer:
                                _aligned_slot += _slots_per_frame
                                _sync_ts = _aligned_slot * total_cpx

                            proxy_ul_head_combined = _sync_ts

                            self.ipc_gnb.set_ul_sync_ts(
                                int(proxy_ul_head_combined))

                            for k in active_set:
                                self._ul_ue_offsets[k] = 0
                                self._prev_ue_heads[k] = ue_heads.get(k, 0)

                            print(f"[UL ACTIVATE] TDD-aligned sync — "
                                  f"proxy_ul_head={proxy_ul_head_combined} "
                                  f"(was {_old_head}, "
                                  f"ue_write_ts={_ue_write_ts}) "
                                  f"sync_ts={proxy_ul_head_combined} "
                                  f"(frame={proxy_ul_head_combined//_spf}, "
                                  f"slot={proxy_ul_head_combined//total_cpx%_slots_per_frame}) "
                                  f"gnb_consumer={_gnb_consumer} "
                                  f"gnb_next_slot={_gnb_next_slot}(mod{_gnb_slot_mod}) "
                                  f"ue_slot={_ue_slot}(mod{_ue_slot_mod}) "
                                  f"slot_adj={_slot_adj} "
                                  f"proxy_dl_head={proxy_dl_head} "
                                  f"(frame={int(proxy_dl_head)//_spf}) "
                                  f"ue_offsets={self._ul_ue_offsets} "
                                  f"ue_heads={dict(ue_heads)} "
                                  f"last_ul_rx_high={self._ul_rx_ts_high}")

                        if self._ul_aligned_to_ue:
                            for k in active_set:
                                if k not in self._ul_ue_offsets:
                                    if k in ue_heads and ue_heads[k] > 0:
                                        self._ul_ue_offsets[k] = 0
                                        self._prev_ue_heads[k] = ue_heads[k]
                                        print(f"[UL OFFSET] UE[{k}] "
                                              f"offset=0 "
                                              f"(ue_head={ue_heads[k]} "
                                              f"proxy_head={proxy_ul_head_combined})")

                            if proxy_ul_head_combined == 0:
                                _cts = self.ipc_gnb.get_ul_consumer_ts()
                                if _cts > 0:
                                    proxy_ul_head_combined = \
                                        ((_cts + total_cpx) // total_cpx) \
                                        * total_cpx
                                else:
                                    proxy_ul_head_combined = 0
                                for k in active_set:
                                    self._ul_ue_offsets[k] = 0
                                if proxy_ul_head_combined > 0:
                                    self.ipc_gnb.set_last_ul_rx_ts(
                                        int(proxy_ul_head_combined) - 1)
                                    self._ul_rx_ts_high = max(
                                        self._ul_rx_ts_high,
                                        int(proxy_ul_head_combined) - 1)
                                print(f"[UL HEAD INIT] cts={_cts} "
                                      f"proxy_ul_head={proxy_ul_head_combined} "
                                      f"ue_offsets={self._ul_ue_offsets}")
                            min_avail = float('inf')
                            for k in active_set:
                                _ko = self._ul_ue_offsets.get(k, 0)
                                avail_k = ue_heads[k] - \
                                    (proxy_ul_head_combined + _ko)
                                if avail_k < min_avail:
                                    min_avail = avail_k
                            delta = (int(min_avail) // total_cpx) * total_cpx
                            if delta > total_cpx * 4:
                                delta = total_cpx * 4

                            if delta >= total_cpx:
                                _ul_log_cnt = getattr(self, '_ul_log_count', 0)
                                if _ul_log_cnt < 50 or _ul_log_cnt % 500 == 0:
                                    cts = self.ipc_gnb.get_ul_consumer_ts()
                                    _spf = total_cpx * 20
                                    print(f"[UL FLOW] #{_ul_log_cnt} "
                                          f"proxy_head={proxy_ul_head_combined} "
                                          f"(frame={proxy_ul_head_combined//_spf}, "
                                          f"slot={proxy_ul_head_combined//total_cpx%20}) "
                                          f"delta={delta} "
                                          f"({delta/total_cpx:.1f} slots) "
                                          f"ue_offsets={self._ul_ue_offsets} "
                                          f"gnb_consumer={cts} "
                                          f"last_ul_rx={self._ul_rx_ts_high} "
                                          f"apply_ch={apply_ch}")
                                self._ul_log_count = _ul_log_cnt + 1

                                self._wait_ul_backpressure(
                                    int(proxy_ul_head_combined + delta - 1))
                                ul_ms, n_slots = self._ipc_ul_combine(
                                    proxy_ul_head_combined, delta,
                                    active_ues=active_set,
                                    ue_ts_offsets=self._ul_ue_offsets)
                                proxy_ul_head_combined += delta

                                for k in active_set:
                                    if k in self._prev_ue_heads:
                                        _old_h = self._prev_ue_heads[k]
                                        _new_h = self.ipc_ues[k].get_last_ul_tx_ts() + \
                                                 self.ipc_ues[k].get_last_ul_tx_nsamps()
                                        self._prev_ue_heads[k] = _new_h

                                ul_count += n_slots
                                ul_processed = True
                                processed = True

                                self._e2e_proxy_ul_accum_ms += ul_ms
                                self._e2e_ul_in_frame += n_slots
                                for _k in active_set:
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
                                    proxy_ul_heads[k] = ue_head
                                delta = int(ue_head - proxy_ul_heads[k])
                                max_ul_delta = total_cpx * 4
                                if delta > max_ul_delta:
                                    delta = max_ul_delta
                                self._wait_ul_backpressure(
                                    int(proxy_ul_heads[k] + delta - 1))
                                ul_ms_bp, n_bp = self._ipc_ul_combine(
                                    proxy_ul_heads[k], delta)
                                _bp_ue_ts = int(proxy_ul_heads[k] + delta - 1)
                                if _bp_ue_ts > self._ul_rx_ts_high:
                                    self._ul_rx_ts_high = _bp_ue_ts
                                    self.ipc_gnb.set_last_ul_rx_ts(_bp_ue_ts)
                                proxy_ul_heads[k] += delta
                                n_bypass_slots = delta // self.pipelines_ul[0].total_cpx
                                ul_count += n_bypass_slots
                                self._e2e_proxy_ul_accum_ms += 0.0
                                self._e2e_ul_in_frame += n_bypass_slots
                                self._e2e_ul_per_ue[k] += n_bypass_slots
                                self._e2e_slot_count += n_bypass_slots
                                self._check_e2e_frame("IPC_G1C+OAI")
                                ul_processed = True
                                processed = True

                # --- UL keepalive: advance last_ul_rx_ts to prevent gNB blocking ---
                # When UEs are active: advance last_ul_rx_ts but skip zero-fill
                # to preserve positions for UL processing with real UE data.
                # When UEs are NOT active: zero-fill and advance as before.
                _KA_MARGIN_SLOTS = 12
                _ue_active = apply_ch and self._ul_aligned_to_ue

                if proxy_dl_head > 0:
                    cts = self.ipc_gnb.get_ul_consumer_ts()
                    if _ue_active:
                        keepalive_target = int(proxy_ul_head_combined)
                    else:
                        cts_limit = int(cts) + _KA_MARGIN_SLOTS * total_cpx
                        keepalive_target = min(int(proxy_dl_head), cts_limit)
                    if apply_ch:
                        keepalive_start = proxy_ul_head_combined
                    else:
                        keepalive_start = min(proxy_ul_heads[k] for k in range(N)) if N > 0 else 0
                    if keepalive_start < 0:
                        keepalive_start = 0

                    _ka_log = getattr(self, '_ka_log_count', 0)
                    if _ka_log < 10 or _ka_log % 1000 == 0:
                        print(f"[KEEPALIVE] #{_ka_log} "
                              f"start={keepalive_start} ({keepalive_start/total_cpx:.1f} slots) "
                              f"target={keepalive_target} ({keepalive_target/total_cpx:.1f} slots) "
                              f"cts={cts} ({cts/total_cpx:.1f} slots) "
                              f"ue_active={_ue_active}")
                    self._ka_log_count = _ka_log + 1

                    _ka_filled_end = int(keepalive_start)
                    if keepalive_target > keepalive_start:
                        ka_delta = int(keepalive_target - keepalive_start)
                        if _ue_active:
                            ka_delta = min(ka_delta, total_cpx * _KA_MARGIN_SLOTS)
                        else:
                            ka_delta = min(ka_delta, total_cpx * 8)
                            self.ipc_gnb.circ_zero(
                                self.ipc_gnb.gpu_ul_rx_ptr, int(keepalive_start), ka_delta,
                                self.ipc_gnb.ul_rx_nbAnt, self.ipc_gnb.ul_rx_cir_size)
                        _ka_filled_end = int(keepalive_start + ka_delta)
                    if _ka_filled_end > 0:
                        _ka_ts = _ka_filled_end - 1
                        if _ka_ts > self._ul_rx_ts_high:
                            self._ul_rx_ts_high = _ka_ts
                            self.ipc_gnb.set_last_ul_rx_ts(_ka_ts)
                    if not _ue_active:
                        if apply_ch:
                            proxy_ul_head_combined = max(
                                proxy_ul_head_combined, _ka_filled_end)
                        else:
                            for k in range(N):
                                proxy_ul_heads[k] = max(
                                    proxy_ul_heads[k], _ka_filled_end)

                if not processed:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            print(f"\n[v4] Terminated by Ctrl-C (DL: {dl_count}, UL: {ul_count}, UEs: {N})")
        finally:
            self._cleanup_channel_producers()
            self.ipc_gnb.cleanup()
            for ipc in self.ipc_ues:
                ipc.cleanup()

    def _cleanup_channel_producers(self):
        """Gracefully stop UnifiedChannelProducerProcess and release CUDA IPC resources."""
        for i, evt in enumerate(getattr(self, '_channel_stop_events', [])):
            evt.set()

        for i, proc in enumerate(getattr(self, 'channel_producers', [])):
            if hasattr(proc, 'join'):
                proc.join(timeout=3)
                if proc.is_alive():
                    print(f"[v4] UnifiedChannelProducerProcess[{i}] did not exit gracefully, killing")
                    proc.kill()
                    proc.join(timeout=2)

        for i, buf in enumerate(getattr(self, 'channel_buffers', [])):
            if hasattr(buf, 'cleanup'):
                try:
                    buf.cleanup()
                except Exception:
                    pass

        print("[v4] UnifiedChannelProducerProcess cleaned up")

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
        description="G1C v3 Multi-UE MIMO Channel Proxy (P1B ray + stall detection)")

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
    ap.add_argument("--buffer-len", type=int, default=1024,
                    help="Channel IPC ring buffer depth")
    ap.add_argument("--buffer-symbol-size", type=int, default=42)

    ap.add_argument("--enable-gpu", dest='enable_gpu', action="store_true")
    ap.add_argument("--disable-gpu", dest='enable_gpu', action="store_false")
    ap.set_defaults(enable_gpu=True)
    ap.add_argument("--use-pinned-memory", dest='use_pinned_memory', action="store_true")
    ap.add_argument("--no-pinned-memory", dest='use_pinned_memory', action="store_false")
    ap.set_defaults(use_pinned_memory=True)
    ap.add_argument("--use-cuda-graph", dest='use_cuda_graph', action="store_true")
    ap.add_argument("--no-cuda-graph", dest='use_cuda_graph', action="store_false")
    ap.set_defaults(use_cuda_graph=True)

    ap.add_argument("--identity-channel", action="store_true", default=False,
                    help="Force DL identity channel H=I (diagnostic: bypass channel distortion)")
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
                    help="Channel generation mode (v4 always uses dynamic, accepted for compatibility)")
    ap.add_argument("--polarization", choices=["single", "dual"], default="single",
                    help="Antenna polarization: single (V-only) or dual (cross-pol ±45°)")
    ap.add_argument("--sector-half-deg", type=float, default=60.0, dest="sector_half_deg",
                    help="Half-sector span in degrees for UE angular placement (default: 60)")
    ap.add_argument("--jitter-std-deg", type=float, default=10.0, dest="jitter_std_deg",
                    help="AoD/AoA jitter std dev in degrees per UE (default: 10)")

    ap.add_argument("--xla", dest='use_xla', action="store_true",
                    help="Enable XLA JIT compilation for channel generation (kernel fusion)")
    ap.add_argument("--no-xla", dest='use_xla', action="store_false")
    ap.set_defaults(use_xla=False)

    # 3GPP TR 38.901 topology parameters
    ap.add_argument("--bs-height-m", type=float, default=25.0,
                    help="BS antenna height in meters (UMi=10, UMa=25)")
    ap.add_argument("--ue-height-m", type=float, default=1.5,
                    help="UE height in meters (1.5~2.5)")
    ap.add_argument("--isd-m", type=float, default=500,
                    help="Inter-Site Distance in meters (UMi=200, UMa=500)")
    ap.add_argument("--min-ue-dist-m", type=float, default=35,
                    help="Minimum BS-UE horizontal distance in meters")
    ap.add_argument("--max-ue-dist-m", type=float, default=500,
                    help="Maximum BS-UE horizontal distance in meters")
    ap.add_argument("--shadow-fading-std-dB", type=float, default=6.0,
                    help="Shadow fading std dev in dB (UMi-LOS=4, UMi-NLOS=7.82, UMa-LOS=4, UMa-NLOS=6)")
    ap.add_argument("--k-factor-mean-dB", type=float, default=None,
                    help="K-factor mean in dB (LOS only; UMi-LOS=9, UMa-LOS=9)")
    ap.add_argument("--k-factor-std-dB", type=float, default=None,
                    help="K-factor std dev in dB (LOS only; UMi-LOS=5, UMa-LOS=3.5)")

    ap.add_argument("--p1b-npz", type=str, default=None,
                    help="P1B npz file path for per-UE independent ray data")
    ap.add_argument("--ue-rx-indices", type=str, default=None,
                    help="Per-UE RX indices (comma-separated, e.g. '100,500') "
                         "or 'random'. Auto-random if --p1b-npz given without this.")
    ap.add_argument("--ue-speeds", type=str, default=None,
                    help="Per-UE speeds in km/h (comma-separated, e.g. '3,30,120,3'). "
                         "Converted to m/s for Sionna velocity tensor.")

    args = ap.parse_args()

    # ── P1B 경로 해석 (상대경로 → 스크립트 기준 절대경로) ──
    if args.p1b_npz and not os.path.isabs(args.p1b_npz):
        args.p1b_npz = os.path.normpath(os.path.join(_SCRIPT_DIR, args.p1b_npz))
        print(f"[v4] P1B npz path resolved: {args.p1b_npz}")

    # ── P1B RX 인덱스 해석 ──
    resolved_rx_indices = None
    if args.p1b_npz:
        if args.ue_rx_indices is None or args.ue_rx_indices == "random":
            resolved_rx_indices = pick_random_rx_indices(args.p1b_npz, args.num_ues)
        else:
            resolved_rx_indices = [int(x) for x in args.ue_rx_indices.split(",")]
            if len(resolved_rx_indices) != args.num_ues:
                print(f"[ERROR] --ue-rx-indices 개수({len(resolved_rx_indices)})가 "
                      f"--num-ues({args.num_ues})와 불일치")
                sys.exit(1)
            validate_rx_indices(args.p1b_npz, resolved_rx_indices)

    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled, noise_mode, noise_dBFS, noise_std_abs, DL_USE_IDENTITY_CHANNEL, ue_speeds

    if args.ue_speeds:
        ue_speeds = [float(s) / 3.6 for s in args.ue_speeds.split(",")]
        if len(ue_speeds) != args.num_ues:
            print(f"[ERROR] --ue-speeds count ({len(ue_speeds)}) != --num-ues ({args.num_ues})")
            sys.exit(1)
        print(f"[v4] Per-UE speeds: {args.ue_speeds} km/h → {[f'{s:.2f}' for s in ue_speeds]} m/s")
    else:
        ue_speeds = None
    DL_USE_IDENTITY_CHANNEL = True  # DIAG: force identity for BLER root-cause test (was: args.identity_channel)
    if DL_USE_IDENTITY_CHANNEL:
        print("[CONFIG] DL_USE_IDENTITY_CHANNEL=True — forcing H=I for DL (diagnostic mode)")
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
    xla_str = " +XLA" if args.use_xla else ""
    print(f"G1C v4 Unified Multi-UE MIMO Channel Proxy{xla_str}")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"UEs: {args.num_ues}")
    if ue_speeds:
        for i, s in enumerate(ue_speeds):
            print(f"  UE{i} speed: {s*3.6:.1f} km/h ({s:.2f} m/s)")
    else:
        print(f"  Global speed: {Speed} m/s ({Speed*3.6:.1f} km/h)")
    if args.p1b_npz:
        rx_str = ", ".join(f"UE{i}=RX{rx}" for i, rx in enumerate(resolved_rx_indices))
        print(f"P1B Ray Data: {args.p1b_npz}")
        print(f"  RX Indices: {rx_str}")
    else:
        print(f"Ray Data: legacy (saved_rays_data/, all UEs share same rays)")
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
    print(f"3GPP Topology: BS_h={args.bs_height_m}m UE_h={args.ue_height_m}m "
          f"ISD={args.isd_m}m d=[{args.min_ue_dist_m},{args.max_ue_dist_m}]m")
    print(f"  Shadow Fading σ={args.shadow_fading_std_dB}dB  "
          f"K-factor μ={args.k_factor_mean_dB}dB σ={args.k_factor_std_dB}dB")
    print(f"Path Loss: {path_loss_dB} dB (linear={pathLossLinear:.6f})")
    if noise_mode == "relative":
        print(f"AWGN Noise: Relative SNR mode (SNR={snr_dB} dB)")
    elif noise_mode == "absolute":
        _noise_rms = 32767.0 * (10.0 ** (noise_dBFS / 20.0))
        print(f"AWGN Noise: Absolute mode (floor={noise_dBFS} dBFS, rms={_noise_rms:.1f})")
    else:
        print(f"AWGN Noise: Disabled")
    print("=" * 80)

    print("\n[v4 Architecture]")
    if args.enable_gpu and GPU_AVAILABLE:
        print(f"  + Multi-UE: {args.num_ues} UE(s), per-UE IPC/pipeline/channel/noise")
        print("  + ChannelProducer: UnifiedChannelProducerProcess (single TF context, N_UE batch)")
        print("  + RingBuffer: CUDA IPC cross-process GPU ring buffer + non-blocking try_put_batch")
        if args.use_xla:
            print("  + XLA: JIT compilation enabled (kernel fusion)")
        else:
            print("  + XLA: Disabled (eager mode)")
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
        polarization=args.polarization,
        p1b_npz=args.p1b_npz,
        ue_rx_indices=resolved_rx_indices,
        use_xla=args.use_xla,
        bs_height_m=args.bs_height_m,
        ue_height_m=args.ue_height_m,
        isd_m=args.isd_m,
        min_ue_dist_m=args.min_ue_dist_m,
        max_ue_dist_m=args.max_ue_dist_m,
        shadow_fading_std_dB=args.shadow_fading_std_dB,
        k_factor_mean_dB=args.k_factor_mean_dB,
        k_factor_std_dB=args.k_factor_std_dB,
    )

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    proxy.run()


if __name__ == "__main__":
    main()
