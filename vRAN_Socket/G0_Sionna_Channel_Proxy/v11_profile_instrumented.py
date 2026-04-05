"""
================================================================================
v10_cuda_graph.py - CUDA Graph Pipeline 🏆🏆🏆

[성능] v9.2 대비 ~40% 개선 (3.2ms → ~1.9ms/slot 예상)
[속도 원인]
  1. CUDA Graph: 커널 런치 오버헤드 제거 (~15 커널 → 1회 graph launch)

[v9.2 → v10 변경사항]
  1. GPU 연산 정밀도: complex128/float64 유지 (complex64는 PSS 검출 실패)
     - float32 FFT(23비트 mantissa)는 2048-pt FFT convolution에서 정밀도 부족
     - PSS/SSS 상관 피크가 threshold 이하로 저하 → UE sync 불가
  2. CUDA Graph: process_slot GPU 연산 시퀀스 캡처 후 단일 launch로 재생
     - 워밍업 N슬롯 → 그래프 캡처 1회 → 이후 모든 슬롯 그래프 리플레이
     - 커널 런치 오버헤드 ~1.3ms 제거
  3. 채널 H: complex64→complex128 캐스트 (v9과 동일, RingBuffer는 complex64)

[핵심 특징] (v9.2 상속)
- 채널 생성: TensorFlow/Sionna (GPU)
- 신호 처리: CuPy (CUDA) - 모든 데이터 변환/처리를 GPU에서 수행
- 채널 전달: TF → CuPy DLPack (GPU-to-GPU zero-copy, numpy 경유 제거)
- RingBuffer: CuPy GPU 스토리지 (CPU 메모리 사용 안함)
- 최적화: Full GPU Pipeline (int16 입출력, CPU 변환 제거)
- 최적화: Batch FFT (14 심볼/슬롯 동시 처리)
- 최적화: Pinned Memory (int16 전송, 4x 전송량 감소)
- 최적화: GPU AWGN noise (cp.random.randn)
- 최적화: GPU 인덱스 사전계산 (symbol extract/reconstruct)
- 최적화: 커널 배치 (get_batch, put_batch, 2D slice assign)

[v10 추가 최적화]
- CUDA Graph: 커널 런치 오버헤드 제거 (15 커널 → 1 graph launch)
- complex128 유지: PSS/SSS 검출 안정성을 위해 float64 정밀도 유지
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
import ctypes
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import logging
import threading
import json
import matplotlib.pyplot as plt
import time
from sionna.phy import PI, SPEED_OF_LIGHT
from datetime import datetime
import os
from channel_coefficients_JIN import ChannelCoefficientsGeneratorJIN, random_binary_mask_tf_complex64

try:
    from sionna.phy.channel.tr38901 import PanelArray, Topology, Rays
    print("[Sionna Init] sionna.phy.channel.tr38901 모듈 로드 성공")
except ModuleNotFoundError as e:
    print(f"[Sionna Init] ⚠️ sionna 모듈 로드 실패: {e}")
    print("[Sionna Init] sionna 설치 여부를 확인해주세요: pip install sionna")
    import sys
    sys.exit(1)

# CuPy import for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[GPU Init] CuPy 로드 성공 - GPU 가속 활성화")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("[GPU Init] ⚠️ CuPy 없음 - CPU 모드로 실행")

# --- Sionna API 설정 (Client가 직접 제어하기 위함) ---
SIONNA_API_IP = "127.0.0.1"
SIONNA_API_PORT = 7000

# --- 선택적 TF import (사용시만)
try:
    import tensorflow as tf
except ImportError:
    tf = None

gpu_num = 0  # Use "" to use the CPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TENSORBOARD_BINARY"] = "tensorboard"
os.environ["TENSORBOARD_PLUGINS"] = "scalars,images,histograms,graphs,projector,profile"

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

# Path Loss: OAI 규약 (total path gain, 전압 도메인)
path_loss_dB = 0
pathLossLinear = 10**(path_loss_dB / 20.0)

# Noise: SNR 기반 AWGN
snr_dB = None
noise_enabled = False

Speed = 3  #meter/sec

def radian_to_degree(radian):
    return radian * (180.0 / PI)

def degree_to_radian(degree):
    return degree * (PI / 180.0)

def set_BS(location=[0,0,0], rotation=[0,0], num_rows_per_panel=1, num_cols_per_panel=1, num_rows=1, num_cols=1, 
           polarization="single", polarization_type="V", antenna_pattern="38.901", 
           panel_vertical_spacing=2.5, panel_horizontal_spacing=2.5):
    BSexample = {
        "location": location,
        "rotation": rotation,
        "num_rows_per_panel": num_rows_per_panel,
        "num_cols_per_panel": num_cols_per_panel,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "polarization": polarization,
        "polarization_type": polarization_type,
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


class WindowProfiler:
    """고정 길이 윈도우 기반 지연 통계(평균/p95/p99/최대)"""
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
            print(f"  - {key:<10} {self._fmt(self.buffers[key])}")


class GPUSlotPipeline:
    """
    GPU Full Pipeline: int16-to-int16
    Version 11: CUDA Graph Pipeline + 계측 고도화 (complex128)
    
    v9.2 대비 개선:
      1. CUDA Graph: GPU 연산 시퀀스를 캡처하여 단일 launch로 재생 (커널 런치 ~1.3ms 제거)
      2. 노이즈 사전생성: cuRAND을 그래프 밖에서 호출하여 CUDA Graph 호환성 확보
    
    신호 체인 (전체 GPU, complex128):
      H2D(int16, 120KB) → GPU(int16→f64→cpx128→FFT conv→reconstruct→PL→AWGN→clip→int16) → D2H(int16, 120KB)
    
    CUDA Graph 동작:
      - 워밍업 (처음 N슬롯): 일반 실행으로 메모리 풀 확보
      - 캡처 (N+1번째 슬롯): GPU 연산 시퀀스를 그래프로 기록
      - 리플레이 (이후 모든 슬롯): 단일 graph.launch()로 전체 GPU 연산 실행
    """
    WARMUP_SLOTS = 3  # 그래프 캡처 전 워밍업 슬롯 수
    
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
        self.profile_gpu = WindowProfiler(
            "GPU_SLOT",
            ["H2D", "CH_COPY", "NOISE_PREP", "GPU_COMPUTE", "D2H", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval,
        )
        self.profile_gpu_evt = WindowProfiler(
            "GPU_SLOT_EVT",
            ["H2D", "CH_COPY", "NOISE_PREP", "GPU_COMPUTE", "D2H", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval,
        )
        self.profile_gpu_diff = WindowProfiler(
            "GPU_SLOT_CPU_MINUS_EVT",
            ["H2D", "CH_COPY", "NOISE_PREP", "GPU_COMPUTE", "D2H", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval,
        )
        
        # Standard NR slot parameters
        self.n_sym = N_SYM  # 14
        self.total_cpx = sum(SYMBOL_SIZES)  # 30720 complex samples
        self.total_int16 = self.total_cpx * 2  # 61440 int16 values
        
        if not self.enable_gpu:
            print("[GPU Pipeline] GPU 비활성화 - CPU numpy 모드")
            return
        
        print(f"[GPU Pipeline v11] CUDA Graph + complex128 Pipeline 초기화 중...")
        print(f"[GPU Pipeline v11] Pinned Memory: {'활성화' if self.use_pinned_memory else '비활성화'}")
        print(f"[GPU Pipeline v11] CUDA Graph: {'활성화' if use_cuda_graph else '비활성화'}")
        print(f"[GPU Pipeline v11] 정밀도: complex128 (float64 기반, PSS 검출 안정성 확보)")
        
        # Single CUDA Stream
        self.stream = cp.cuda.Stream(non_blocking=True)
        
        # === Pre-compute symbol boundary index arrays (GPU, CuPy 직접) ===
        sym_bounds = get_ofdm_symbol_indices(self.total_cpx)
        
        # Extract: gather indices for each symbol's FFT portion (after CP removal)
        ext_idx = cp.zeros((self.n_sym, fft_size), dtype=cp.int64)
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            ext_idx[i] = cp.arange(s + cp_l, e)
        self.gpu_ext_idx = ext_idx  # (14, 2048) on GPU
        
        # Reconstruct data: flat scatter indices for FFT data portion
        data_dst = []
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            data_dst.append(cp.arange(s + cp_l, e, dtype=cp.int64))
        self.gpu_data_dst = cp.concatenate(data_dst)  # (14*2048=28672,)
        
        # Reconstruct CP: scatter dst indices + gather src indices
        cp_dst_list, cp_src_list = [], []
        for i, (s, e) in enumerate(sym_bounds):
            cp_l = CP1 if i < 12 else CP2
            cp_dst_list.append(cp.arange(s, s + cp_l, dtype=cp.int64))
            cp_src_list.append(cp.arange(i * fft_size + fft_size - cp_l,
                                         i * fft_size + fft_size, dtype=cp.int64))
        self.gpu_cp_dst = cp.concatenate(cp_dst_list)  # (2048,) total CP samples
        self.gpu_cp_src = cp.concatenate(cp_src_list)
        
        print(f"[GPU Pipeline v11] 인덱스 사전계산 완료 (sym={self.n_sym}, fft={fft_size})")
        
        # === Pinned Memory: int16 I/O만 ===
        if self.use_pinned_memory:
            self.pinned_iq_in_buf = cp.cuda.alloc_pinned_memory(self.total_int16 * 2)  # int16=2bytes
            self.pinned_iq_out_buf = cp.cuda.alloc_pinned_memory(self.total_int16 * 2)
            
            self.pinned_iq_in = np.frombuffer(self.pinned_iq_in_buf, dtype=np.int16,
                                              count=self.total_int16)
            self.pinned_iq_out = np.frombuffer(self.pinned_iq_out_buf, dtype=np.int16,
                                               count=self.total_int16)
            
            mem_iq = self.total_int16 * 2 * 2  # in + out
            mem_mb = mem_iq / 1024 / 1024
            print(f"[GPU Pipeline v11] Pinned Memory: {mem_mb:.2f} MB "
                  f"(int16 I/O={self.total_int16*4//1024}KB, 채널 H는 GPU 직접)")
        
        # === Pre-allocated GPU Memory (complex128 = 16 bytes, v9과 동일) ===
        self.gpu_iq_in = cp.zeros(self.total_int16, dtype=cp.int16)
        self.gpu_iq_out = cp.zeros(self.total_int16, dtype=cp.int16)
        self.gpu_x = cp.zeros((self.n_sym, fft_size), dtype=cp.complex128)
        self.gpu_H = cp.zeros((self.n_sym, fft_size), dtype=cp.complex128)
        self.gpu_out = cp.zeros(self.total_cpx, dtype=cp.complex128)
        
        # === Pre-allocated Noise Buffer (CUDA Graph 호환: cuRAND을 캡처 밖에서 호출) ===
        self.gpu_noise_r = cp.zeros(self.total_cpx, dtype=cp.float64)
        self.gpu_noise_i = cp.zeros(self.total_cpx, dtype=cp.float64)
        
        gpu_mb = (self.total_int16 * 2 * 2 + (self.n_sym * fft_size * 16) * 2 + self.total_cpx * 16 + self.total_cpx * 8 * 2) / 1024 / 1024
        print(f"[GPU Pipeline v11] GPU Memory: {gpu_mb:.2f} MB (complex128, v9과 동일)")
        
        # === CUDA Graph 상태 ===
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph = None
        self.graph_captured = False
        self.warmup_count = 0
        # 그래프 캡처 시 PL/noise 설정 저장 (변경 시 재캡처)
        self._graph_pl_linear = None
        self._graph_noise_on = None
        self._graph_snr_db = None
        
        print(f"[GPU Pipeline v11] 초기화 완료 - CUDA Graph + complex128 Pipeline")
        print(f"[GPU Pipeline v11] H2D: {self.total_int16*2//1024}KB(int16) → "
              f"GPU처리(complex128) → D2H: {self.total_int16*2//1024}KB(int16)")
        if use_cuda_graph:
            print(f"[GPU Pipeline v11] CUDA Graph: 워밍업 {self.WARMUP_SLOTS}슬롯 후 캡처")
        print(f"[GPU Pipeline v11] 프로파일링: interval={self.profile_interval}, window={self.profile_window}")
        print(f"[GPU Pipeline v11] 듀얼 타이머 비교(CPU+sync vs CUDA Event): "
              f"{'활성화' if self.dual_timer_compare else '비활성화'}")
    
    def _regenerate_noise(self):
        """
        cuRAND으로 노이즈 사전생성 — CUDA Graph 캡처 밖에서만 호출.
        self.gpu_noise_r, self.gpu_noise_i에 새로운 표준정규 난수를 채움.
        """
        self.gpu_noise_r[:] = cp.random.randn(self.total_cpx).astype(cp.float64)
        self.gpu_noise_i[:] = cp.random.randn(self.total_cpx).astype(cp.float64)
    
    def _gpu_compute_core(self, pl_linear, snr_db, noise_on):
        """
        Pure GPU computation (no H2D/D2H) — CUDA Graph 캡처 가능
        
        모든 중간 결과를 self._tmp_* 에 저장하여 그래프 캡처 시
        메모리가 해제되지 않도록 보장.
        
        노이즈: self.gpu_noise_r/i에 사전생성된 값 사용 (cuRAND 호출 없음)
        
        입력: self.gpu_iq_in (int16), self.gpu_H (complex128)
        출력: self.gpu_iq_out (int16)
        """
        # ─── int16 → float64 → complex128 (IQ deinterleave) ───
        self._tmp_f64 = self.gpu_iq_in.astype(cp.float64)
        self._tmp_cpx = self._tmp_f64[::2] + 1j * self._tmp_f64[1::2]
        self.gpu_x[:] = self._tmp_cpx[self.gpu_ext_idx]
        
        # ─── Batch FFT Convolution (complex128) ───
        self._tmp_Xf = cp.fft.fft(self.gpu_x, axis=1)
        self._tmp_Hf = cp.fft.fft(self.gpu_H, axis=1)
        self._tmp_XfHf = self._tmp_Xf * self._tmp_Hf
        self._tmp_y = cp.fft.ifft(self._tmp_XfHf, axis=1)
        
        # ─── Reconstruct output ───
        self.gpu_out[:] = 0
        self._tmp_y_flat = self._tmp_y.ravel()
        self.gpu_out[self.gpu_data_dst] = self._tmp_y_flat
        self.gpu_out[self.gpu_cp_dst] = self._tmp_y_flat[self.gpu_cp_src]
        
        # ─── Path Loss (float64 스칼라곱) ───
        if pl_linear != 1.0:
            self.gpu_out *= cp.float64(pl_linear)
        
        # ─── AWGN noise (사전생성된 gpu_noise_r/i 사용, cuRAND 없음) ───
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
        
        # ─── complex128 → int16 ───
        self._tmp_clip_r = cp.clip(
            cp.around(self.gpu_out.real), -32768, 32767)
        self._tmp_clip_i = cp.clip(
            cp.around(self.gpu_out.imag), -32768, 32767)
        self.gpu_iq_out[::2] = self._tmp_clip_r.astype(cp.int16)
        self.gpu_iq_out[1::2] = self._tmp_clip_i.astype(cp.int16)
    
    def _try_capture_graph(self, pl_linear, snr_db, noise_on):
        """
        CUDA Graph 캡처 시도.
        
        성공 시: self.cuda_graph에 그래프 저장, self.graph_captured = True
        실패 시: 일반 모드로 폴백, self.use_cuda_graph = False
        """
        try:
            # 캡처 모드 시작 — 이후 GPU 연산은 기록만 되고 실행되지 않음
            self.stream.begin_capture()
            self._gpu_compute_core(pl_linear, snr_db, noise_on)
            self.cuda_graph = self.stream.end_capture()
            
            self.graph_captured = True
            self._graph_pl_linear = pl_linear
            self._graph_noise_on = noise_on
            self._graph_snr_db = snr_db
            
            print(f"[CUDA Graph] ✅ 캡처 성공! "
                  f"(PL={pl_linear}, noise={'ON' if noise_on else 'OFF'}"
                  f"{f', SNR={snr_db}dB' if noise_on and snr_db is not None else ''})")
            print(f"[CUDA Graph] 이후 모든 슬롯은 graph.launch()로 실행")
            
        except Exception as e:
            # 캡처 실패 시 스트림 복원
            try:
                self.stream.end_capture()
            except:
                pass
            
            # ★ 핵심 수정: 디바이스 전체 동기화로 스트림 상태 확실히 복원
            # CUDA 스펙: end_capture 후 스트림은 정상 모드로 돌아가야 하지만,
            # 안전을 위해 디바이스 전체 동기화 수행
            try:
                cp.cuda.Device(0).synchronize()
            except:
                pass
            
            print(f"[CUDA Graph] ❌ 캡처 실패 — 일반 모드로 실행: {e}")
            print(f"[CUDA Graph] 디바이스 동기화 완료 — 이후 정상 GPU 실행")
            self.graph_captured = False
            self.use_cuda_graph = False
    
    def _need_recapture(self, pl_linear, snr_db, noise_on):
        """PL/noise 설정이 변경되었으면 재캡처 필요"""
        if not self.graph_captured:
            return False
        return (self._graph_pl_linear != pl_linear or
                self._graph_noise_on != noise_on or
                self._graph_snr_db != snr_db)
    
    def process_slot(self, iq_bytes, channels_gpu, pl_linear, snr_db, noise_on):
        """
        Full GPU pipeline: raw bytes in → int16 bytes out
        
        v10 동작 모드:
          1. 워밍업 (처음 N슬롯): 일반 GPU 실행 — 메모리 풀 확보
          2. 캡처 (N+1슬롯): CUDA Graph로 GPU 연산 기록 후 즉시 재생
          3. 리플레이 (이후): graph.launch()만으로 전체 GPU 연산 실행
        
        Args:
            iq_bytes: raw IQ bytes
            channels_gpu: (N_SYM, fft_size) CuPy GPU array from get_batch()
            pl_linear: path loss linear gain
            snr_db: SNR in dB (None = no noise)
            noise_on: bool
        """
        n_iq = len(iq_bytes) // 2
        n_cpx = n_iq // 2
        
        if not self.enable_gpu:
            return self._cpu_fallback(iq_bytes, channels_gpu, pl_linear, snr_db, noise_on)
        if n_cpx != self.total_cpx:
            return self._cpu_fallback(iq_bytes, channels_gpu, pl_linear, snr_db, noise_on)
        
        do_profile = (self.slot_counter > 0 and self.slot_counter % self.profile_interval == 0)
        do_dual_profile = do_profile and self.dual_timer_compare
        if do_profile:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        if do_dual_profile:
            e_h2d_s, e_h2d_e = cp.cuda.Event(), cp.cuda.Event()
            e_ch_s, e_ch_e = cp.cuda.Event(), cp.cuda.Event()
            e_noise_s, e_noise_e = cp.cuda.Event(), cp.cuda.Event()
            e_gpu_s, e_gpu_e = cp.cuda.Event(), cp.cuda.Event()
            e_d2h_s, e_d2h_e = cp.cuda.Event(), cp.cuda.Event()
        
        with self.stream:
            # ═══════ Pre-graph: H2D + 채널 복사 (매 슬롯 실행) ═══════
            
            # ─── H2D: int16 IQ data 전송 ───
            if do_dual_profile:
                e_h2d_s.record(self.stream)
            if self.use_pinned_memory:
                ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
                self.gpu_iq_in.set(self.pinned_iq_in, stream=self.stream)
            else:
                iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
                self.gpu_iq_in[:] = cp.asarray(iq_int16)
            if do_dual_profile:
                e_h2d_e.record(self.stream)
            
            if do_profile:
                self.stream.synchronize(); t1 = time.perf_counter()
            
            # ─── 채널 H 복사 (complex128) ───
            if do_dual_profile:
                e_ch_s.record(self.stream)
            n_ch = min(channels_gpu.shape[0], self.n_sym)
            n_w = min(channels_gpu.shape[1], self.fft_size)
            self.gpu_H[:] = 0
            if channels_gpu.dtype != cp.complex128:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w].astype(cp.complex128)
            else:
                self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w]
            if do_dual_profile:
                e_ch_e.record(self.stream)
            
            if do_profile:
                self.stream.synchronize(); t2 = time.perf_counter()
            
            # ═══════ 노이즈 사전생성 (cuRAND — CUDA Graph 캡처 밖) ═══════
            if do_profile:
                t_noise0 = time.perf_counter()
            if do_dual_profile:
                e_noise_s.record(self.stream)
            if noise_on and snr_db is not None:
                self._regenerate_noise()
            if do_dual_profile:
                e_noise_e.record(self.stream)
            if do_profile:
                self.stream.synchronize(); t_noise1 = time.perf_counter()
            
            # ═══════ GPU Compute (CUDA Graph 또는 일반 실행) ═══════
            if do_dual_profile:
                e_gpu_s.record(self.stream)
            
            # 설정 변경 시 재캡처
            if self._need_recapture(pl_linear, snr_db, noise_on):
                print(f"[CUDA Graph] 설정 변경 감지 — 재캡처 시작")
                self.graph_captured = False
                self.warmup_count = self.WARMUP_SLOTS  # 워밍업 건너뛰기
            
            if self.graph_captured:
                # ─── 모드 3: CUDA Graph 리플레이 ───
                self.cuda_graph.launch(self.stream)
            elif self.use_cuda_graph and self.warmup_count >= self.WARMUP_SLOTS:
                # ─── 모드 2: 그래프 캡처 + 즉시 리플레이 ───
                self._try_capture_graph(pl_linear, snr_db, noise_on)
                if self.graph_captured:
                    # 캡처 성공: 캡처 중에는 실행되지 않았으므로 graph launch로 실행
                    self.cuda_graph.launch(self.stream)
                else:
                    # 캡처 실패: 일반 실행
                    self._gpu_compute_core(pl_linear, snr_db, noise_on)
            else:
                # ─── 모드 1: 워밍업 (일반 실행) ───
                self._gpu_compute_core(pl_linear, snr_db, noise_on)
                self.warmup_count += 1
                if self.warmup_count == self.WARMUP_SLOTS and self.use_cuda_graph:
                    print(f"[CUDA Graph] 워밍업 {self.WARMUP_SLOTS}슬롯 완료 — 다음 슬롯에서 캡처 예정")
            if do_dual_profile:
                e_gpu_e.record(self.stream)
            
            if do_profile:
                self.stream.synchronize(); t3 = time.perf_counter()
            
            # ═══════ Post-graph: D2H ═══════
            if do_dual_profile:
                e_d2h_s.record(self.stream)
            
            if self.use_pinned_memory:
                self.gpu_iq_out.get(out=self.pinned_iq_out, stream=self.stream)
                if do_dual_profile:
                    e_d2h_e.record(self.stream)
                self.stream.synchronize()
                result = self.pinned_iq_out.tobytes()
            else:
                out_host = self.gpu_iq_out.get(stream=self.stream)
                if do_dual_profile:
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
            cpu_noise = 1000 * (t_noise1 - t_noise0)
            cpu_gpu = 1000 * (t3 - t_noise1)
            cpu_d2h = 1000 * (t4 - t3)
            cpu_total = 1000 * (t4 - t0)
            self.profile_gpu.add(
                tag=f"mode={mode}",
                H2D=cpu_h2d,
                CH_COPY=cpu_ch,
                NOISE_PREP=cpu_noise,
                GPU_COMPUTE=cpu_gpu,
                D2H=cpu_d2h,
                TOTAL=cpu_total,
            )
            if do_dual_profile:
                evt_h2d = cp.cuda.get_elapsed_time(e_h2d_s, e_h2d_e)
                evt_ch = cp.cuda.get_elapsed_time(e_ch_s, e_ch_e)
                evt_noise = cp.cuda.get_elapsed_time(e_noise_s, e_noise_e)
                evt_gpu = cp.cuda.get_elapsed_time(e_gpu_s, e_gpu_e)
                evt_d2h = cp.cuda.get_elapsed_time(e_d2h_s, e_d2h_e)
                evt_total = evt_h2d + evt_ch + evt_noise + evt_gpu + evt_d2h
                self.profile_gpu_evt.add(
                    tag=f"mode={mode}",
                    H2D=evt_h2d,
                    CH_COPY=evt_ch,
                    NOISE_PREP=evt_noise,
                    GPU_COMPUTE=evt_gpu,
                    D2H=evt_d2h,
                    TOTAL=evt_total,
                )
                self.profile_gpu_diff.add(
                    tag=f"mode={mode}",
                    H2D=cpu_h2d - evt_h2d,
                    CH_COPY=cpu_ch - evt_ch,
                    NOISE_PREP=cpu_noise - evt_noise,
                    GPU_COMPUTE=cpu_gpu - evt_gpu,
                    D2H=cpu_d2h - evt_d2h,
                    TOTAL=cpu_total - evt_total,
                )
        return result
    
    def _cpu_fallback(self, iq_bytes, channels_gpu, pl_linear, snr_db, noise_on):
        """CPU fallback (GPU 비활성화 또는 비표준 슬롯) - numpy 사용"""
        iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
        x_cpx = iq_int16[::2].astype(np.float64) + 1j * iq_int16[1::2].astype(np.float64)
        n_cpx = len(x_cpx)
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        
        # channels_gpu: (N, fft_size) 2D array → numpy로 변환
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


class RingBuffer:
    """GPU RingBuffer: CuPy GPU 배열로 채널 H 저장 (numpy 제거)
    
    v9.2 최적화:
      - put_batch(batch): N개를 1회 lock으로 삽입 (ChannelProducer용)
      - get_batch(n): N개를 1회 lock으로 반환, 연속 슬라이스면 GPU copy 1회
    """
    def __init__(self, shape, dtype=cp.complex64, maxlen=1024):
        if GPU_AVAILABLE:
            self.buffer = cp.zeros((maxlen,) + shape, dtype=dtype)
            self.is_gpu = True
            total_bytes = int(cp.prod(cp.array(self.buffer.shape)).item()) * self.buffer.itemsize
            print(f"[RingBuffer] GPU 모드: {total_bytes / 1024 / 1024:.1f} MB GPU memory "
                  f"(maxlen={maxlen}, shape={shape}, dtype={dtype})")
        else:
            self.buffer = np.zeros((maxlen,) + shape, dtype=np.complex64)
            self.is_gpu = False
            print(f"[RingBuffer] CPU 모드 (GPU 비활성화)")
        self.maxlen = maxlen
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, data):
        """data: CuPy GPU array (or numpy for CPU mode)"""
        with self.not_full:
            while self.count == self.maxlen:
                self.not_full.wait()
            self.buffer[self.write_idx] = data
            self.write_idx = (self.write_idx + 1) % self.maxlen
            self.count += 1
            self.not_empty.notify()

    def put_batch(self, data_batch):
        """data_batch: (N, shape) GPU array — 1회 lock으로 N개 삽입"""
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
        """Returns: CuPy GPU array (or numpy for CPU mode)"""
        with self.not_empty:
            while self.count == 0:
                self.not_empty.wait()
            data = self.buffer[self.read_idx].copy()
            self.read_idx = (self.read_idx + 1) % self.maxlen
            self.count -= 1
            self.not_full.notify()
        return data

    def get_batch(self, n):
        """Returns: (n, shape) GPU array — 1회 lock으로 N개 반환."""
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
            cp.cuda.Device(0).use()  # 스레드에서 CUDA 디바이스 명시 설정
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
            h_delay = tf.squeeze(h_delay)  # (batch_size, fft_size) TF GPU tensor
            
            # === 배치 정규화 (TF 벡터 연산, for-loop 제거) ===
            # v9과 동일: complex128로 에너지 계산 후 정규화 → complex64로 다운캐스트
            # (float32 정규화는 PSS/SSS 검출에 영향을 줄 수 있음)
            h_c128 = tf.cast(h_delay, tf.complex128)
            energy = tf.reduce_sum(tf.abs(h_c128) ** 2, axis=-1, keepdims=True)  # (N, 1)
            h_norm = h_delay / tf.cast(tf.sqrt(energy), h_delay.dtype)
            h_c64 = tf.cast(h_norm, tf.complex64)  # (N, fft_size)
            
            # === 배치 DLPack 전송 (1회 전송) ===
            try:
                h_cp_batch = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c64)).copy()
            except Exception:
                h_cp_batch = cp.asarray(h_c64.numpy())
            
            # === 배치 삽입 (1회 lock) ===
            self.buffer.put_batch(h_cp_batch)
            self.symbol_counter += self.buffer_symbol_size


class Proxy:
    def __init__(self, ue_port, gnb_host, gnb_port, log_level, ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np", custom_channel=False,
                 buffer_len=4096, buffer_symbol_size=42, enable_gpu=True, use_pinned_memory=True,
                 use_cuda_graph=True, profile_interval=100, profile_window=500,
                 dual_timer_compare=True):
        self.prev_ts = None
        self.global_symbol_count = 0
        self.slot_sample_accum = 0
        self.sel = selectors.DefaultSelector()
        self.lis = socket.socket()
        self.lis.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lis.bind(("0.0.0.0", ue_port))
        self.lis.listen()
        self.lis.setblocking(False)
        self.sel.register(self.lis, selectors.EVENT_READ, data="UE_LIS")
        print(f"[INFO] UE listen 0.0.0.0:{ue_port}")
        
        self.gnb_host, self.gnb_port = gnb_host, gnb_port
        self.gnb_ep: Optional[Endpoint] = None
        self.ues: Dict[int, Endpoint] = {}
        self.gnb_hshake: Optional[Tuple[bytes, bytes]] = None
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
        self.profile_interval = max(1, int(profile_interval))
        self.profile_window = max(10, int(profile_window))
        self.dual_timer_compare = bool(dual_timer_compare)
        self.profile_ofdm = WindowProfiler(
            "OFDM_SLOT",
            ["CH_GET", "CH_PAD", "GPU_PROC", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval,
        )
        self.profile_proxy = WindowProfiler(
            "PROXY_E2E",
            ["PROC", "SEND", "TOTAL"],
            window=self.profile_window,
            report_interval=self.profile_interval,
        )
        
        # === E2E 시간 비교: Δwall vs Proxy 처리 → 소켓+OAI 추정 ===
        # TDD 프레임(10 slots = 7D+1S+2U) 단위로 측정
        self._e2e_slot_count = 0            # 전체 슬롯 카운터 (DL+UL)
        self._e2e_frame_slots = 10          # 측정 단위 (= TDD 프레임 크기)
        self._e2e_last_wall = None          # 이전 프레임 끝 wall time
        self._e2e_proxy_dl_accum_ms = 0.0   # 프레임 내 DL Proxy 처리시간 누적 (ms)
        self._e2e_proxy_ul_accum_ms = 0.0   # 프레임 내 UL Proxy 처리시간 누적 (ms)
        self._e2e_dl_in_frame = 0           # 프레임 내 DL 슬롯 수
        self._e2e_ul_in_frame = 0           # 프레임 내 UL 슬롯 수
        
        self.sionna_params = {
            "ch_en": self.ch_en,
            "ch_L": self.ch_L,
            "ch_dd": self.ch_dd,
            "log_plot": self.log_plot,
            "conv_mode": self.conv_mode,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "fft_lib": self.fft_lib,
            "custom_channel": self.custom_channel,
            "buffer_len": self.buffer_len,
            "buffer_symbol_size": self.buffer_symbol_size,
            "enable_gpu": self.enable_gpu,
            "use_pinned_memory": self.use_pinned_memory,
            "use_cuda_graph": self.use_cuda_graph,
            "speed": Speed,
            "path_loss_dB": path_loss_dB,
            "pathLossLinear": pathLossLinear,
            "snr_dB": snr_dB,
            "noise_enabled": noise_enabled
        }

        if self.custom_channel:
            # Load rays data
            phi_r_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/phi_r_rays_for_ChannelBlock.npy"))
            phi_t_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/phi_t_rays_for_ChannelBlock.npy"))
            theta_r_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/theta_r_rays_for_ChannelBlock.npy"))
            theta_t_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/theta_t_rays_for_ChannelBlock.npy"))
            power_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/power_rays_for_ChannelBlock.npy"))
            tau_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory + "/tau_rays_for_ChannelBlock.npy"))
            
            batch_size = 1
            self.N_UE = 1
            self.N_BS = 1
            self.num_rx = 1
            self.num_tx = 1
            BSexample, _ = set_BS()
                
            ArrayRX = PanelArray(
                num_rows_per_panel=1, num_cols_per_panel=1, num_rows=1, num_cols=1,
                polarization='single', polarization_type='V', antenna_pattern='omni',
                carrier_frequency=carrier_frequency
            )
            
            ArrayTX = PanelArray(
                num_rows_per_panel=BSexample["num_rows_per_panel"],
                num_cols_per_panel=BSexample["num_cols_per_panel"],
                num_rows=BSexample["num_rows"],
                num_cols=BSexample["num_cols"],
                polarization=BSexample["polarization"],
                polarization_type=BSexample["polarization_type"],
                antenna_pattern=BSexample["antenna_pattern"],
                carrier_frequency=carrier_frequency,
                panel_vertical_spacing=BSexample["panel_vertical_spacing"],
                panel_horizontal_spacing=BSexample["panel_horizontal_spacing"]
            )

            aoa_pdp = phi_r_rays_for_ChannelBlock
            aod_pdp = phi_t_rays_for_ChannelBlock
            zoa_pdp = theta_r_rays_for_ChannelBlock
            zod_pdp = theta_t_rays_for_ChannelBlock
            power_pdp = power_rays_for_ChannelBlock
            delay_pdp = tau_rays_for_ChannelBlock

            mean_xpr_list = {"UMi-LOS": 9, "UMi-NLOS": 8, "UMa-LOS": 8, "UMa-NLOS": 7}
            stddev_xpr_list = {"UMi-LOS": 3, "UMi-NLOS": 3, "UMa-LOS": 4, "UMa-NLOS": 4}
            mean_xpr = mean_xpr_list["UMa-NLOS"]
            stddev_xpr = stddev_xpr_list["UMa-NLOS"]
            xpr_pdp = 10**(tf.random.normal(
                shape=[batch_size, self.N_BS, self.N_UE, 1, aoa_pdp.shape[-1]], 
                mean=mean_xpr, stddev=stddev_xpr
            )/10)

            PDP = Rays(
                delays=delay_pdp, powers=power_pdp, aoa=aoa_pdp, aod=aod_pdp, 
                zoa=zoa_pdp, zod=zod_pdp, xpr=xpr_pdp
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

            self.channel_buffer = RingBuffer(shape=(FFT_SIZE,), dtype=cp.complex64 if GPU_AVAILABLE else np.complex64, maxlen=buffer_len)
            params = dict(Fs=Fs, scs=scs, N_UE=self.N_UE, N_BS=self.N_BS, N_UE_active=self.num_rx, N_BS_serving=self.num_tx)
            
            self.producer = ChannelProducer(
                self.channel_buffer, self.Channel_Generator, self.topology, params,
                self.h_field_array_power, self.aoa_delay, self.zoa_delay, 
                buffer_symbol_size=buffer_symbol_size
            )
            self.producer.start()
            
            # Initialize GPU Slot Pipeline (v10: CUDA Graph + complex128)
            self.gpu_slot_pipeline = GPUSlotPipeline(
                FFT_SIZE, 
                enable_gpu=self.enable_gpu,
                use_pinned_memory=self.use_pinned_memory,
                use_cuda_graph=self.use_cuda_graph,
                profile_interval=self.profile_interval,
                profile_window=self.profile_window,
                dual_timer_compare=self.dual_timer_compare,
            )
            print(f"[INFO] GPU Slot Pipeline v11 초기화 완료 "
                  f"(GPU={'활성화' if self.enable_gpu and GPU_AVAILABLE else 'CPU 모드'}, "
                  f"CUDA Graph={'활성화' if self.use_cuda_graph else '비활성화'})")
            print(f"[INFO] Profiling interval={self.profile_interval}, window={self.profile_window}")
            print(f"[INFO] Dual timer compare(CPU+sync vs CUDA Event): "
                  f"{'활성화' if self.dual_timer_compare else '비활성화'}")

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
            size, nb, ts, frame, subframe = unpack_header(h)
            log("Proxy → UE", size, nb, ts, frame, subframe, size*nb, "(cached handshake)")
    
    def _handle_ep(self, ep: Endpoint):
        for hdr_raw, hdr_vals, payload in ep.read_blocks():
            t_blk0 = time.perf_counter()
            size, nb, ts, frame, subframe = hdr_vals
            sample_cnt = size * nb
            
            if size > 1 and self.ch_en:
                if self.custom_channel:
                    processed = self.process_ofdm_slot_with_slot_pipeline(
                        payload, ts, log_plot=self.log_plot
                    )
                    ch_note = f" (GPU slot pipeline v11)"
                else:
                    processed = payload
                    ch_note = ""
            else:
                processed = payload
                ch_note = ""
            t_proc1 = time.perf_counter()
            send_ms = 0.0
            
            if ep.role == "gNB":
                log("gNB → Proxy", size, nb, ts, frame, subframe, sample_cnt, 
                    "(handshake)" if size == 1 else ch_note)
                if size == 1: 
                    self.gnb_hshake = (hdr_raw, payload)
                for u in list(self.ues.values()):
                    if u.closed: 
                        continue
                    t_send0 = time.perf_counter()
                    u.send(hdr_raw, processed)
                    send_ms += 1000 * (time.perf_counter() - t_send0)
                    log("Proxy → UE", size, nb, ts, frame, subframe, sample_cnt, 
                        "(handshake)" if size == 1 else ch_note)
            else:
                log("UE → Proxy", size, nb, ts, frame, subframe, sample_cnt, ch_note)
                if self.gnb_ep and not self.gnb_ep.closed:
                    t_send0 = time.perf_counter()
                    self.gnb_ep.send(hdr_raw, processed)
                    send_ms += 1000 * (time.perf_counter() - t_send0)
                    log("Proxy → gNB", size, nb, ts, frame, subframe, sample_cnt, ch_note)

            if size > 1 and self.ch_en and self.custom_channel:
                total_ms = 1000 * (time.perf_counter() - t_blk0)
                proc_ms = 1000 * (t_proc1 - t_blk0)
                self.profile_proxy.add(
                    tag=f"dir={ep.role}",
                    PROC=proc_ms,
                    SEND=send_ms,
                    TOTAL=total_ms,
                )
                
                # === E2E 시간 비교 (TDD 프레임 = 10 total slots 단위) ===
                self._e2e_slot_count += 1
                if ep.role == "gNB":
                    self._e2e_proxy_dl_accum_ms += total_ms
                    self._e2e_dl_in_frame += 1
                else:
                    self._e2e_proxy_ul_accum_ms += total_ms
                    self._e2e_ul_in_frame += 1
                
                if self._e2e_slot_count % self._e2e_frame_slots == 0:
                    now = time.perf_counter()
                    if self._e2e_last_wall is not None:
                        wall_ms = 1000 * (now - self._e2e_last_wall)
                        dl_ms = self._e2e_proxy_dl_accum_ms
                        ul_ms = self._e2e_proxy_ul_accum_ms
                        proxy_ms = dl_ms + ul_ms
                        sock_oai_ms = wall_ms - proxy_ms
                        n = self._e2e_frame_slots
                        nd = self._e2e_dl_in_frame
                        nu = self._e2e_ul_in_frame
                        print(f"\n[E2E frame#{self._e2e_slot_count} "
                              f"({nd}D+{nu}U)] "
                              f"Δwall={wall_ms:.2f}ms  "
                              f"Proxy(DL={dl_ms:.1f}+UL={ul_ms:.1f})"
                              f"={proxy_ms:.2f}ms  "
                              f"Socket+OAI={sock_oai_ms:.2f}ms  "
                              f"| per slot({n}): "
                              f"wall={wall_ms/n:.2f}  "
                              f"proxy={proxy_ms/n:.2f}  "
                              f"sock+oai={sock_oai_ms/n:.2f} ms")
                    else:
                        print(f"\n[E2E frame#{self._e2e_slot_count}] "
                              f"기준점 설정 (다음 프레임부터 비교 시작)")
                    self._e2e_last_wall = now
                    self._e2e_proxy_dl_accum_ms = 0.0
                    self._e2e_proxy_ul_accum_ms = 0.0
                    self._e2e_dl_in_frame = 0
                    self._e2e_ul_in_frame = 0

    def process_ofdm_slot_with_slot_pipeline(self, iq_bytes, ts, log_plot=False):
        """
        Version 11: CUDA Graph + complex128 Pipeline + profile instrumentation (bytes → bytes)
        """
        t_start = time.perf_counter()
        
        n_int16 = len(iq_bytes) // 2
        n_cpx = n_int16 // 2
        sym_idx = get_ofdm_symbol_indices(n_cpx)
        n_sym = len(sym_idx)
        
        # 채널 응답 배치 읽기
        t_ch0 = time.perf_counter()
        channels = self.channel_buffer.get_batch(n_sym)  # (n_sym, FFT_SIZE) CuPy GPU
        t_ch1 = time.perf_counter()
        
        t_pad0 = t_ch1
        if n_sym < N_SYM:
            lib = cp if GPU_AVAILABLE else np
            pad = lib.ones((N_SYM - n_sym, FFT_SIZE), dtype=channels.dtype)
            channels = lib.concatenate([channels, pad])
        t_pad1 = time.perf_counter()
        
        # Full GPU 처리
        t_gpu0 = t_pad1
        result = self.gpu_slot_pipeline.process_slot(
            iq_bytes, channels, pathLossLinear, snr_dB, noise_enabled
        )
        t_end = time.perf_counter()
        
        sc = self.gpu_slot_pipeline.slot_counter
        do_profile = (sc > 0 and sc % self.profile_interval == 0)
        if do_profile:
            mode = "GRAPH" if self.gpu_slot_pipeline.graph_captured else "NORMAL"
            self.profile_ofdm.add(
                tag=f"mode={mode}",
                CH_GET=1000 * (t_ch1 - t_ch0),
                CH_PAD=1000 * (t_pad1 - t_pad0),
                GPU_PROC=1000 * (t_end - t_gpu0),
                TOTAL=1000 * (t_end - t_start),
            )
        
        return result
    
    def run(self):
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


def main():
    ap = argparse.ArgumentParser(description="vRAN Socket Sionna Channel Simulator - Version 11 (Profile Instrumentation)")
    ap.add_argument("--ue-port", type=int, default=6018)
    ap.add_argument("--gnb-host", default="127.0.0.1")
    ap.add_argument("--gnb-port", type=int, default=6017)
    ap.add_argument("--log", choices=["error", "warn", "info", "debug"], default="info")
    ap.add_argument("--ch-en", dest='ch_en', action="store_true", help="Enable channel convolution (FFT H)")
    ap.add_argument("--no-ch-en", dest='ch_en', action="store_false", help="Disable channel convolution")
    ap.set_defaults(ch_en=True)
    ap.add_argument("--ch-dd", type=int, default=0, help="Channel delay (samples, NOT used in this version)")
    ap.add_argument("--ch-L", type=int, default=32, help="Channel impulse response length")
    ap.add_argument("--log-plot", action="store_true", default=False, help="1st OFDM symbol visualization")
    ap.add_argument("--conv-mode", type=str, default="fft", choices=["fft", "oa", "os"], help="Channel convolution mode")
    ap.add_argument("--block-size", type=int, default=4096, help="Block size (for oa, os modes)")
    ap.add_argument("--num-blocks", type=int, default=None, help="Number of blocks limit")
    ap.add_argument("--fft-lib", type=str, default="np", choices=["np", "tf"], help="FFT backend: np or tf")
    ap.add_argument("--custom-channel", dest='custom_channel', action="store_true", help="Enable custom Sionna channel")
    ap.add_argument("--no-custom-channel", dest='custom_channel', action="store_false", help="Disable custom channel")
    ap.set_defaults(custom_channel=True)
    ap.add_argument("--buffer-len", type=int, default=42000, help="Channel buffer length")
    ap.add_argument("--buffer-symbol-size", type=int, default=4200, help="Symbols to generate per batch")
    ap.add_argument("--enable-gpu", dest='enable_gpu', action="store_true", help="Enable GPU acceleration")
    ap.add_argument("--disable-gpu", dest='enable_gpu', action="store_false", help="Disable GPU (use CPU numpy)")
    ap.set_defaults(enable_gpu=True)
    ap.add_argument("--use-pinned-memory", dest='use_pinned_memory', action="store_true", help="Use pinned memory for faster PCIe")
    ap.add_argument("--no-pinned-memory", dest='use_pinned_memory', action="store_false", help="Use regular memory")
    ap.set_defaults(use_pinned_memory=True)
    ap.add_argument("--use-cuda-graph", dest='use_cuda_graph', action="store_true", help="Enable CUDA Graph (kernel launch optimization)")
    ap.add_argument("--no-cuda-graph", dest='use_cuda_graph', action="store_false", help="Disable CUDA Graph (normal kernel execution)")
    ap.set_defaults(use_cuda_graph=True)
    ap.add_argument("--path-loss-dB", type=float, default=0.0,
                    help="Path loss in dB (OAI convention: negative=attenuation, 0=no loss, default=0)")
    ap.add_argument("--snr-dB", type=float, default=None,
                    help="SNR in dB for AWGN noise (None=no noise, 20=20dB SNR, etc.)")
    ap.add_argument("--profile-interval", type=int, default=100,
                    help="Profiling report interval in slots (default: 100)")
    ap.add_argument("--profile-window", type=int, default=500,
                    help="Rolling window size for avg/p95/p99/max stats (default: 500)")
    ap.add_argument("--dual-timer-compare", dest='dual_timer_compare', action="store_true",
                    help="Profile sample slots with both CPU+sync and CUDA Event timers")
    ap.add_argument("--no-dual-timer-compare", dest='dual_timer_compare', action="store_false",
                    help="Disable CUDA Event comparison and keep CPU+sync profiling only")
    ap.set_defaults(dual_timer_compare=True)
    
    args = ap.parse_args()
    
    # 전역 변수 업데이트
    global path_loss_dB, pathLossLinear, snr_dB, noise_enabled
    path_loss_dB = args.path_loss_dB
    pathLossLinear = 10**(path_loss_dB / 20.0)
    snr_dB = args.snr_dB
    noise_enabled = (snr_dB is not None)
    
    print("=" * 80)
    print("vRAN Socket Channel Simulator - Version 11: Profile Instrumentation")
    print("=" * 80)
    print(f"GPU Acceleration: {'Enabled' if args.enable_gpu and GPU_AVAILABLE else 'Disabled (CPU mode)'}")
    print(f"Pinned Memory: {'Enabled' if args.use_pinned_memory else 'Disabled'}")
    print(f"CUDA Graph: {'Enabled' if args.use_cuda_graph else 'Disabled'}")
    print(f"Precision: complex128 (float64, PSS 검출 안정성 확보)")
    print(f"Pipeline: Full GPU int16-to-int16 (delay=0)")
    print(f"Framework: TF/Sionna(채널) + CuPy(신호처리) + DLPack(GPU-to-GPU)")
    print(f"Custom Channel: {'Enabled' if args.custom_channel else 'Disabled'}")
    print(f"Conv Mode: {args.conv_mode.upper()} | FFT Lib: CuPy (GPU)")
    print(f"Buffer Length: {args.buffer_len} symbols ({args.buffer_len / 14:.0f} slots)")
    print(f"Buffer Symbol Size: {args.buffer_symbol_size} symbols/batch")
    print(f"Channel Enabled: {args.ch_en}")
    print(f"Path Loss: {path_loss_dB} dB (linear gain = {pathLossLinear:.6f}, OAI convention)")
    print(f"Profiling: interval={args.profile_interval} slots, window={args.profile_window} samples")
    print(f"Dual Timer Compare: {'Enabled' if args.dual_timer_compare else 'Disabled'}")
    if noise_enabled:
        print(f"AWGN Noise: Enabled (SNR = {snr_dB} dB, GPU cp.random)")
    else:
        print(f"AWGN Noise: Disabled (no noise)")
    print("=" * 80)
    print("\n[최적화 활성화 상태]")
    if args.enable_gpu and GPU_AVAILABLE:
        print("  ✓ Full GPU Pipeline (int16→GPU처리→int16, delay=0)")
        print("  ✓ TF→CuPy DLPack (채널 H GPU-to-GPU, numpy 경유 제거)")
        print("  ✓ GPU RingBuffer (채널 H를 GPU 메모리에 저장)")
        
        if args.use_pinned_memory:
            print("  ✓ Pinned Memory int16 I/O (H2D/D2H 120KB, 채널 H는 GPU 직접)")
        else:
            print("  ✓ Regular Memory (일반 전송)")
        
        print("  ✓ GPU IQ deinterleave + int16 변환 (CPU 데이터 변환 제거)")
        print("  ✓ GPU 인덱스 사전계산 (symbol extract/reconstruct, CuPy)")
        print("  ✓ CuPy Batch FFT (14 symbols 동시)")
        print("  ✓ GPU AWGN noise (cp.random.randn)")
        print("  ✓ Zero pipeline delay (입력 즉시 결과 반환)")
        
        # v11 기준 (v10 연산 유지 + 계측 강화)
        print("  ✓ [v11] complex128 정밀도 유지 (PSS/SSS 검출 안정성)")
        if args.use_cuda_graph:
            print(f"  ✓ [v11] CUDA Graph (워밍업 {GPUSlotPipeline.WARMUP_SLOTS}슬롯 후 캡처, "
                  f"~15 커널 → 1 graph launch)")
        else:
            print("  ✗ [v11] CUDA Graph 비활성화 (일반 커널 런치)")
        print("  ✓ [v11] 계측 고도화: GPU/OFDM/Proxy 구간별 avg/p95/p99/max")
        if args.dual_timer_compare:
            print("  ✓ [v11] G0 듀얼 타이머 비교: CPU+sync vs CUDA Event")
        else:
            print("  ✗ [v11] CUDA Event 비교 비활성화")
    else:
        print("  ✓ CPU NumPy FFT (fallback mode)")
    
    if args.custom_channel:
        print("  ✓ Sionna Channel Model (3GPP TR 38.901)")
        print("  ✓ Background Channel Producer Thread")
        print(f"  ✓ RingBuffer ({args.buffer_len} symbols)")
    
    print("=" * 80)
    print()
    
    Proxy(
        args.ue_port, args.gnb_host, args.gnb_port, args.log,
        ch_en=args.ch_en, ch_dd=args.ch_dd, ch_L=args.ch_L, log_plot=args.log_plot,
        conv_mode=args.conv_mode, block_size=args.block_size, num_blocks=args.num_blocks, 
        fft_lib=args.fft_lib, custom_channel=args.custom_channel, 
        buffer_len=args.buffer_len, buffer_symbol_size=args.buffer_symbol_size,
        enable_gpu=args.enable_gpu, use_pinned_memory=args.use_pinned_memory,
        use_cuda_graph=args.use_cuda_graph,
        profile_interval=args.profile_interval,
        profile_window=args.profile_window,
        dual_timer_compare=args.dual_timer_compare,
    ).run()


if __name__ == "__main__":
    main()

