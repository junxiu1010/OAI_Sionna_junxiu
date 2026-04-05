"""
================================================================================
v6_cupy_stream_per_symbol_slow.py - CuPy + Multi-Stream + 심볼 단위 파이프라인

[성능] ✅ ~130ms (10 DL당) | 실시간 대비 10.8배 느림 ⚠️ CuPy 중 최악
[속도 원인] 심볼 단위 파이프라인으로 인한 오버헤드가 이점을 상쇄

[핵심 특징]
- GPU: CuPy (CUDA)
- 채널: 사전 계산된 H 사용
- 최적화: 3-Stream 파이프라인 (H2D, Compute, D2H)
- 최적화: Pinned Memory 사용

[느린 이유] ⚠️ 심볼 단위 파이프라인 오버헤드
- 매 심볼마다 스트림 동기화 → 오버헤드 누적
- 매 심볼마다 H2D/Compute/D2H 전환 → 컨텍스트 스위칭
- Pinned Memory의 이점이 오버헤드에 묻힘
- 결론: Multi-Stream이 항상 좋은 것은 아님!

[교훈]
- 파이프라인 단위가 너무 작으면 오히려 역효과
- v7, v8처럼 배치 단위로 처리해야 효과적
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
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

PathLoss_dB = -10
PathLoss = 10**(PathLoss_dB/10)
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


class GPUPipelineProcessor:
    """
    GPU Multi-Stream Overlap Processor for Event-based Pipeline
    문서 Section 5.2.4 및 Section 9 구현
    """
    def __init__(self, fft_size=2048, enable_gpu=True):
        self.fft_size = fft_size
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.slot_counter = 0
        
        if not self.enable_gpu:
            print("[GPU Pipeline] GPU 비활성화 - CPU numpy 모드")
            return
        
        print("[GPU Pipeline] GPU Multi-Stream Overlap 초기화 중...")
        
        # Create 3 streams for H2D, Compute, D2H
        self.stream_h2d = cp.cuda.Stream()
        self.stream_compute = cp.cuda.Stream()
        self.stream_d2h = cp.cuda.Stream()
        
        # CUDA Events for stream synchronization
        self.event_h2d_done = cp.cuda.Event()
        self.event_compute_done = cp.cuda.Event()
        
        # Pinned memory for faster transfer (3 slots cycling)
        self.pinned_x = [cp.cuda.alloc_pinned_memory(fft_size * 16) for _ in range(3)]  # complex128
        self.pinned_H = [cp.cuda.alloc_pinned_memory(fft_size * 16) for _ in range(3)]
        self.pinned_y_out = [cp.cuda.alloc_pinned_memory(fft_size * 16) for _ in range(3)]
        
        # GPU buffers (3 ring buffers for pipelining)
        self.gpu_x = [cp.empty(fft_size, dtype=cp.complex128) for _ in range(3)]
        self.gpu_H = [cp.empty(fft_size, dtype=cp.complex128) for _ in range(3)]
        self.gpu_X = [cp.empty(fft_size, dtype=cp.complex128) for _ in range(3)]
        self.gpu_Y = [cp.empty(fft_size, dtype=cp.complex128) for _ in range(3)]
        self.gpu_y = [cp.empty(fft_size, dtype=cp.complex128) for _ in range(3)]
        
        # Pipeline delay buffer (stores results with 2-slot delay)
        self.result_buffer = [None, None, None]
        
        print(f"[GPU Pipeline] 초기화 완료 - Pinned Memory: {fft_size * 16 * 3 * 3 / 1024 / 1024:.2f} MB")
    
    def process_symbol(self, sym_cpu, h_cpu):
        """
        Process one OFDM symbol with GPU pipeline
        
        Pipeline timeline:
        Stream 0 (H2D):     TTI n
        Stream 1 (Compute): TTI n-1
        Stream 2 (D2H):     TTI n-2
        """
        if not self.enable_gpu:
            # Fallback to CPU numpy
            Hf = np.fft.fft(h_cpu, self.fft_size)
            Xf = np.fft.fft(sym_cpu)
            Yf = Xf * Hf
            y = np.fft.ifft(Yf)
            return y
        
        buf_idx = self.slot_counter % 3
        
        # === Stream 0: TTI n - H2D transfer ===
        with self.stream_h2d:
            # Copy to pinned memory
            pinned_x_view = np.frombuffer(self.pinned_x[buf_idx], dtype=np.complex128)
            pinned_H_view = np.frombuffer(self.pinned_H[buf_idx], dtype=np.complex128)
            np.copyto(pinned_x_view[:len(sym_cpu)], sym_cpu.astype(np.complex128))
            np.copyto(pinned_H_view[:len(h_cpu)], h_cpu.astype(np.complex128))
            
            # Transfer to GPU
            self.gpu_x[buf_idx] = cp.asarray(pinned_x_view[:self.fft_size])
            self.gpu_H[buf_idx] = cp.asarray(pinned_H_view[:self.fft_size])
            
            # Record H2D completion event
            self.event_h2d_done.record(self.stream_h2d)
        
        # === Stream 1: TTI n-1 - Channel operations ===
        if self.slot_counter >= 1:
            prev_idx = (self.slot_counter - 1) % 3
            
            with self.stream_compute:
                # Wait for H2D completion
                self.stream_compute.wait_event(self.event_h2d_done)
                
                # FFT + Channel application + IFFT
                self.gpu_X[prev_idx] = cp.fft.fft(self.gpu_x[prev_idx])
                self.gpu_Y[prev_idx] = self.gpu_X[prev_idx] * self.gpu_H[prev_idx]
                self.gpu_y[prev_idx] = cp.fft.ifft(self.gpu_Y[prev_idx])
                
                # Record compute completion event
                self.event_compute_done.record(self.stream_compute)
        
        # === Stream 2: TTI n-2 - D2H transfer ===
        result = None
        if self.slot_counter >= 2:
            result_idx = (self.slot_counter - 2) % 3
            
            with self.stream_d2h:
                # Wait for compute completion
                self.stream_d2h.wait_event(self.event_compute_done)
                
                # Transfer result back to CPU
                result_gpu = self.gpu_y[result_idx].get(stream=self.stream_d2h)
                result = result_gpu
        
        self.slot_counter += 1
        return result
    
    def flush(self):
        """
        Flush pipeline to get remaining 2 results
        """
        if not self.enable_gpu:
            return []
        
        results = []
        
        # Process remaining n-1 slot
        if self.slot_counter >= 1:
            prev_idx = (self.slot_counter - 1) % 3
            
            with self.stream_compute:
                self.stream_compute.wait_event(self.event_h2d_done)
                self.gpu_X[prev_idx] = cp.fft.fft(self.gpu_x[prev_idx])
                self.gpu_Y[prev_idx] = self.gpu_X[prev_idx] * self.gpu_H[prev_idx]
                self.gpu_y[prev_idx] = cp.fft.ifft(self.gpu_Y[prev_idx])
                self.event_compute_done.record(self.stream_compute)
            
            with self.stream_d2h:
                self.stream_d2h.wait_event(self.event_compute_done)
                result = self.gpu_y[prev_idx].get(stream=self.stream_d2h)
                results.append(result)
        
        # Synchronize all streams
        self.stream_h2d.synchronize()
        self.stream_compute.synchronize()
        self.stream_d2h.synchronize()
        
        return results


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
    def __init__(self, shape, dtype=np.complex64, maxlen=1024):
        self.buffer = np.zeros((maxlen,) + shape, dtype=dtype)
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

    def get(self):
        with self.not_empty:
            while self.count == 0:
                self.not_empty.wait()
            data = self.buffer[self.read_idx].copy()
            self.read_idx = (self.read_idx + 1) % self.maxlen
            self.count -= 1
            self.not_full.notify()
        return data


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
        while not self.stop_event.is_set():
            sample_times = tf.cast(
                np.arange(self.buffer_symbol_size), self.channel_generator.rdtype
            ) / tf.constant(self.params['scs'], self.channel_generator.rdtype)
            ActiveUE_component = random_binary_mask_tf_complex64(self.params['N_UE'], k=self.params['N_UE_active'])
            ActiveUE = tf.constant(ActiveUE_component, dtype=tf.complex64)
            ServingBS_component = random_binary_mask_tf_complex64(self.params['N_BS'], k=self.params['N_BS_serving'])
            ServingBS = tf.constant(ServingBS_component, dtype=tf.complex64)
            h_delay, _, _, _ = self.channel_generator._H_TTI_sequential_fft_o_ELW2_noProfile(
                self.topology, ActiveUE, ServingBS, sample_times,
                self.h_field_array_power, self.aoa_delay, self.zoa_delay
            )
            h_delay = tf.squeeze(h_delay).numpy()
            for h in h_delay:
                h_normalized = h / np.sqrt(np.sum(np.abs(h)**2)) / np.sqrt(PathLoss)
                self.buffer.put(h_normalized)
            self.symbol_counter += self.buffer_symbol_size


class Proxy:
    def __init__(self, ue_port, gnb_host, gnb_port, log_level, ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np", custom_channel=False,
                 buffer_len=4096, buffer_symbol_size=42, enable_gpu=True):
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
            "speed": Speed,
            "path_loss_dB": PathLoss_dB
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

            self.channel_buffer = RingBuffer(shape=(FFT_SIZE,), dtype=np.complex64, maxlen=buffer_len)
            params = dict(Fs=Fs, scs=scs, N_UE=self.N_UE, N_BS=self.N_BS, N_UE_active=self.num_rx, N_BS_serving=self.num_tx)
            
            self.producer = ChannelProducer(
                self.channel_buffer, self.Channel_Generator, self.topology, params,
                self.h_field_array_power, self.aoa_delay, self.zoa_delay, 
                buffer_symbol_size=buffer_symbol_size
            )
            self.producer.start()
            
            # Initialize GPU Pipeline Processor
            self.gpu_pipeline = GPUPipelineProcessor(FFT_SIZE, enable_gpu=self.enable_gpu)
            print(f"[INFO] GPU Pipeline 초기화 완료 (GPU={'활성화' if self.enable_gpu and GPU_AVAILABLE else 'CPU 모드'})")

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
        self.ues[ue.fileno()] = ue
        self.sel.register(c, selectors.EVENT_READ, data=ue)
        print(f"[INFO] UE joined {addr}")
        if self.gnb_hshake:
            h, p = self.gnb_hshake
            ue.send(h, p)
            size, nb, ts, frame, subframe = unpack_header(h)
            log("Proxy → UE", size, nb, ts, frame, subframe, size*nb, "(cached handshake)")
    
    def _handle_ep(self, ep: Endpoint):
        for hdr_raw, hdr_vals, payload in ep.read_blocks():
            size, nb, ts, frame, subframe = hdr_vals
            sample_cnt = size * nb
            
            if size > 1 and self.ch_en:
                if self.custom_channel:
                    processed = self.process_ofdm_slot_with_gpu_pipeline(
                        payload, ts, log_plot=self.log_plot
                    )
                    ch_note = f" (GPU pipeline, stream overlap)"
                else:
                    processed = payload
                    ch_note = ""
            else:
                processed = payload
                ch_note = ""
            
            if ep.role == "gNB":
                log("gNB → Proxy", size, nb, ts, frame, subframe, sample_cnt, 
                    "(handshake)" if size == 1 else ch_note)
                if size == 1: 
                    self.gnb_hshake = (hdr_raw, payload)
                for u in list(self.ues.values()):
                    if u.closed: 
                        continue
                    u.send(hdr_raw, processed)
                    log("Proxy → UE", size, nb, ts, frame, subframe, sample_cnt, 
                        "(handshake)" if size == 1 else ch_note)
            else:
                log("UE → Proxy", size, nb, ts, frame, subframe, sample_cnt, ch_note)
                if self.gnb_ep and not self.gnb_ep.closed:
                    self.gnb_ep.send(hdr_raw, processed)
                    log("Proxy → gNB", size, nb, ts, frame, subframe, sample_cnt, ch_note)

    def process_ofdm_slot_with_gpu_pipeline(self, iq_bytes, ts, log_plot=False):
        """
        Process OFDM slot with GPU Multi-Stream Overlap Pipeline
        문서 Section 9.2 구현
        """
        x = np.frombuffer(iq_bytes, dtype='<i2')
        x_cpx = x[::2] + 1j * x[1::2]
        total_samples = len(x_cpx)
        sym_idx = get_ofdm_symbol_indices(total_samples)
        
        # Result buffer with pipeline delay compensation
        symbol_results = []
        pending_results = []
        
        for n, (start, end) in enumerate(sym_idx):
            cp_len = CP1 if n < 12 else CP2
            if end - start != FFT_SIZE + cp_len:
                continue
            
            # Extract symbol
            sym = x_cpx[start + cp_len: end]
            
            # Get channel from buffer
            h_this = self.channel_buffer.get()
            
            # Process through GPU pipeline (returns result with 2-slot delay)
            result = self.gpu_pipeline.process_symbol(sym, h_this)
            
            if result is not None:
                # Result from n-2 symbol
                symbol_results.append((n - 2, result))
        
        # Flush pipeline to get last 2 symbols
        flushed = self.gpu_pipeline.flush()
        for i, result in enumerate(flushed):
            symbol_results.append((len(sym_idx) - 2 + i, result))
        
        # Reconstruct output
        out = np.zeros_like(x_cpx)
        for sym_n, y in symbol_results:
            if sym_n < 0 or sym_n >= len(sym_idx):
                continue
            start, end = sym_idx[sym_n]
            cp_len = CP1 if sym_n < 12 else CP2
            out[start + cp_len: end] = y
            out[start: start + cp_len] = y[-cp_len:]
        
        # Convert to int16
        y_int16 = np.empty(len(out) * 2, dtype='<i2')
        y_int16[::2] = np.clip(np.round(out.real), -32768, 32767).astype('<i2')
        y_int16[1::2] = np.clip(np.round(out.imag), -32768, 32767).astype('<i2')
        
        return y_int16.tobytes()
    
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
    ap = argparse.ArgumentParser(description="vRAN Socket Sionna Channel Simulator - Version 6 (GPU Multi-Stream Overlap)")
    ap.add_argument("--ue-port", type=int, default=6018)
    ap.add_argument("--gnb-host", default="127.0.0.1")
    ap.add_argument("--gnb-port", type=int, default=6017)
    ap.add_argument("--log", choices=["error", "warn", "info", "debug"], default="info")
    ap.add_argument("--ch-en", dest='ch_en', action="store_true", help="Enable channel convolution (FFT H)")
    ap.add_argument("--no-ch-en", dest='ch_en', action="store_false", help="Disable channel convolution")
    ap.set_defaults(ch_en=True)  # Channel enabled by default
    ap.add_argument("--ch-dd", type=int, default=0, help="Channel delay (samples, NOT used in this version)")
    ap.add_argument("--ch-L", type=int, default=32, help="Channel impulse response length")
    ap.add_argument("--log-plot", action="store_true", default=False, help="1st OFDM symbol visualization")
    ap.add_argument("--conv-mode", type=str, default="fft", choices=["fft", "oa", "os"], help="Channel convolution mode")
    ap.add_argument("--block-size", type=int, default=4096, help="Block size (for oa, os modes)")
    ap.add_argument("--num-blocks", type=int, default=None, help="Number of blocks limit")
    ap.add_argument("--fft-lib", type=str, default="np", choices=["np", "tf"], help="FFT backend: np or tf")
    ap.add_argument("--custom-channel", dest='custom_channel', action="store_true", help="Enable custom Sionna channel")
    ap.add_argument("--no-custom-channel", dest='custom_channel', action="store_false", help="Disable custom channel")
    ap.set_defaults(custom_channel=True)  # Custom channel enabled by default
    ap.add_argument("--buffer-len", type=int, default=42000, help="Channel buffer length")
    ap.add_argument("--buffer-symbol-size", type=int, default=4200, help="Symbols to generate per batch")
    ap.add_argument("--enable-gpu", dest='enable_gpu', action="store_true", help="Enable GPU Multi-Stream Overlap")
    ap.add_argument("--disable-gpu", dest='enable_gpu', action="store_false", help="Disable GPU (use CPU numpy)")
    ap.set_defaults(enable_gpu=True)  # GPU enabled by default
    
    args = ap.parse_args()
    
    print("=" * 80)
    print("vRAN Socket Channel Simulator - Version 6: GPU Multi-Stream Overlap")
    print("=" * 80)
    print(f"GPU Acceleration: {'Enabled' if args.enable_gpu and GPU_AVAILABLE else 'Disabled (CPU mode)'}")
    print(f"Custom Channel: {'Enabled' if args.custom_channel else 'Disabled'}")
    print(f"Conv Mode: {args.conv_mode.upper()} | FFT Lib: {args.fft_lib.upper()}")
    print(f"Buffer Length: {args.buffer_len} symbols ({args.buffer_len / 14:.0f} slots)")
    print(f"Buffer Symbol Size: {args.buffer_symbol_size} symbols/batch")
    print(f"Channel Enabled: {args.ch_en}")
    print("=" * 80)
    print("\n[최적화 활성화 상태]")
    if args.enable_gpu and GPU_AVAILABLE:
        print("  ✓ Multi-Stream Pipeline (3 streams: H2D, Compute, D2H)")
        print("  ✓ Pinned Memory (2x faster PCIe transfer)")
        print("  ✓ CUDA Event Synchronization")
        print("  ✓ 3-slot Ring Buffer for Pipeline")
        print("  ✓ CuPy GPU FFT Acceleration")
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
        enable_gpu=args.enable_gpu
    ).run()


if __name__ == "__main__":
    main()

