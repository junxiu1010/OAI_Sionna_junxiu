"""
================================================================================
v2_tf_realtime_slow_fix.py - TensorFlow 실시간 Sionna 채널 생성 (버그 수정)

[성능] ❌ 에러 (CUDA device busy)
[속도 원인] 실시간으로 Sionna 채널 계수를 생성하여 GPU 리소스 충돌 발생

[핵심 특징]
- GPU: TensorFlow
- 채널: 실시간 Sionna 생성 (매 심볼마다 계산)
- 문제점: GPU 리소스 충돌로 실행 불가
- v2 대비: 일부 버그 수정

[느린 이유]
- 실시간 채널 생성은 계산량이 매우 큼
- TensorFlow와 다른 GPU 작업 간 리소스 충돌
================================================================================
"""
import argparse, selectors, socket, struct, numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import time
from sionna.phy.channel.tr38901 import PanelArray, Topology, Rays
from sionna.phy import PI, SPEED_OF_LIGHT
from datetime import datetime
import os

from channel_coefficients_JIN import ChannelCoefficientsGeneratorJIN, random_binary_mask_tf_complex64

# --- 선택적 TF import (사용시만)
try:
    import tensorflow as tf
except ImportError:
    tf = None

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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(os.path.dirname(_SCRIPT_DIR), "saved_rays_data")

def radian_to_degree(radian):
    return radian * (180.0 / PI)

def degree_to_radian(degree):
    return degree * (PI / 180.0)



def set_BS(location=[0,0,0], rotation=[0,0], num_rows_per_panel=1, num_cols_per_panel=1, num_rows=1, num_cols=1, polarization="single", polarization_type="V", antenna_pattern="38.901", panel_vertical_spacing=2.5, panel_horizontal_spacing=2.5):
    BSexample = {"location" : location,
            "rotation" : rotation,
            "num_rows_per_panel" : num_rows_per_panel,
            "num_cols_per_panel" : num_cols_per_panel,
            "num_rows" : num_rows,
            "num_cols" : num_cols,
            "polarization" : polarization,
            "polarization_type" : polarization_type,
            "antenna_pattern" : antenna_pattern,
            "panel_vertical_spacing" : panel_vertical_spacing,
            "panel_horizontal_spacing" : panel_horizontal_spacing}
    tx_antennas = int(BSexample["num_rows_per_panel"]*BSexample["num_cols_per_panel"]*BSexample["num_rows"]*BSexample["num_cols"])
    
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

def gen_time_channel_impulse(L=32, N=FFT_SIZE, seed=0):
    np.random.seed(seed)
    h = (np.random.randn(L) + 1j*np.random.randn(L))
    h = h / np.sqrt(np.sum(np.abs(h)**2))  # Normalize total power = 1
    h_padded = np.zeros(N, dtype=complex)
    h_padded[:L] = h
    return h_padded, h

def get_channel_freq_h_from_time(h_time, N_fft, fft_lib="np"):
    if fft_lib == "tf":
        h_tf = tf.convert_to_tensor(h_time, dtype=tf.complex64)
        H_tf = tf.signal.fft(tf.cast(h_tf, tf.complex64))
        return H_tf.numpy()
    else:
        return np.fft.fft(h_time, N_fft)

def fft_ifft(x, n=None, lib="np"):
    if lib == "tf":
        x_tf = tf.convert_to_tensor(x, dtype=tf.complex64)
        if n is not None:
            # zero pad or slice to n
            pad = n - tf.shape(x_tf)[-1]
            x_tf = tf.cond(pad > 0, lambda: tf.pad(x_tf, [[0, pad]]), lambda: x_tf[:n])
        X_tf = tf.signal.fft(x_tf)
        return X_tf.numpy()
    else:
        return np.fft.fft(x, n)

def ifft(x, n=None, lib="np"):
    if lib == "tf":
        x_tf = tf.convert_to_tensor(x, dtype=tf.complex64)
        if n is not None:
            pad = n - tf.shape(x_tf)[-1]
            x_tf = tf.cond(pad > 0, lambda: tf.pad(x_tf, [[0, pad]]), lambda: x_tf[:n])
        X_tf = tf.signal.ifft(x_tf)
        return X_tf.numpy()
    else:
        return np.fft.ifft(x, n)

def fft_convolution_block(x, h, mode='oa', block_size=None, num_blocks=None, fft_lib="np"):
    L = len(h)
    if block_size is None:
        block_size = 2 ** int(np.ceil(np.log2(L + 1)))
    N = block_size

    h_fft = fft_ifft(h, N, lib=fft_lib)
    x = np.asarray(x)
    n_x = len(x)
    y = np.zeros_like(x, dtype=complex)
    if mode == 'oa':
        step = N - L + 1
        if step <= 0:
            raise ValueError('block_size must be > len(h)')
        num_blocks_eff = (n_x + step - 1) // step
        if num_blocks is not None:
            num_blocks_eff = min(num_blocks_eff, num_blocks)
        for i in range(num_blocks_eff):
            start = i * step
            end = min(start + N, n_x)
            x_blk = np.zeros(N, dtype=complex)
            x_blk[:end-start] = x[start:end]
            X_blk = fft_ifft(x_blk, N, lib=fft_lib)
            y_blk = ifft(X_blk * h_fft, N, lib=fft_lib)
            y[start:start+N] += y_blk[:end-start]
    elif mode == 'os':
        step = N - L + 1
        if step <= 0:
            raise ValueError('block_size must be > len(h)')
        pad = N - step
        x_padded = np.concatenate([np.zeros(pad, dtype=complex), x])
        n_blk = (len(x_padded) - N) // step + 1
        if num_blocks is not None:
            n_blk = min(n_blk, num_blocks)
        out_idx = 0
        for i in range(n_blk):
            start = i * step
            blk = x_padded[start:start+N]
            X_blk = fft_ifft(blk, N, lib=fft_lib)
            y_blk = ifft(X_blk * h_fft, N, lib=fft_lib)
            y[out_idx:out_idx+step] = y_blk[pad:]
            out_idx += step
    else:
        raise ValueError("mode should be 'oa' or 'os'")
    return y

def process_ofdm_slot_with_channel(iq_bytes, h_full, h_nonzero, Hf, log_plot=False,
                                   conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np"):
    x = np.frombuffer(iq_bytes, dtype='<i2')
    x_cpx = x[::2] + 1j * x[1::2]
    total_samples = len(x_cpx)
    sym_idx = get_ofdm_symbol_indices(total_samples)
    out = np.zeros_like(x_cpx)
    symbol_waveforms = []
    out_waveforms = []
    for n, (start, end) in enumerate(sym_idx):
        cp_len = CP1 if n < 12 else CP2
        if end - start != FFT_SIZE + cp_len:
            continue
        sym = x_cpx[start + cp_len: end]
        if conv_mode == "fft":
            if fft_lib == "tf":
                # TF 버전
                sym_tf = tf.convert_to_tensor(sym, dtype=tf.complex64)
                H_tf = tf.convert_to_tensor(Hf, dtype=tf.complex64)
                Xf = tf.signal.fft(sym_tf)
                Yf = Xf * H_tf
                y = tf.signal.ifft(Yf).numpy()
            else:
                Xf = np.fft.fft(sym)
                Yf = Xf * Hf
                y = np.fft.ifft(Yf)
        else:
            y = fft_convolution_block(sym, h_nonzero, mode=conv_mode, block_size=block_size, num_blocks=num_blocks, fft_lib=fft_lib)
        out[start + cp_len: end] = y
        out[start: start + cp_len] = x_cpx[start: start + cp_len]
        symbol_waveforms.append(sym.copy())
        out_waveforms.append(y.copy())
    y_int16 = np.empty(len(out)*2, dtype='<i2')
    y_int16[::2] = np.clip(np.round(out.real), -32768, 32767).astype('<i2')
    y_int16[1::2] = np.clip(np.round(out.imag), -32768, 32767).astype('<i2')
    if log_plot and len(symbol_waveforms) > 0:
        plt.figure(figsize=(12,6))
        plt.subplot(2,1,1)
        plt.title("첫번째 OFDM 심볼 (원본, Real/Imag)")
        plt.plot(np.real(symbol_waveforms[0]), label='Re[x]')
        plt.plot(np.imag(symbol_waveforms[0]), label='Im[x]')
        plt.legend(); plt.grid()
        plt.subplot(2,1,2)
        plt.title("첫번째 OFDM 심볼 (채널 통과 후, Real/Imag)")
        plt.plot(np.real(out_waveforms[0]), label='Re[y]')
        plt.plot(np.imag(out_waveforms[0]), label='Im[y]')
        plt.legend(); plt.grid()
        plt.tight_layout(); plt.show()
        print("원본 심볼 일부:", symbol_waveforms[0][:10])
        print("채널 적용 후 일부:", out_waveforms[0][:10])
    return y_int16.tobytes()



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
    def fileno(self): return self.sock.fileno()
    def close(self):
        if self.closed: return
        self.closed = True
        try: self.sock.close()
        finally: pass
    def read_blocks(self):
        blocks = []
        try: chunk = self.sock.recv(65536)
        except BlockingIOError: return blocks
        except OSError: self.close(); return blocks
        if not chunk: self.close(); return blocks
        self.rx.extend(chunk)
        while True:
            if self.stage == "hdr":
                if len(self.rx) < HDR_LEN: break
                self.hdr_raw = bytes(self.rx[:HDR_LEN]); del self.rx[:HDR_LEN]
                size, nb, ts, frame, subframe = unpack_header(self.hdr_raw)
                self.hdr_vals = (size, nb, ts, frame, subframe)
                self.pay_len = size * nb * 4
                self.stage = "pay"
            if self.stage == "pay":
                if len(self.rx) < self.pay_len: break
                payload = bytes(self.rx[:self.pay_len]); del self.rx[:self.pay_len]
                blocks.append((self.hdr_raw, self.hdr_vals, payload))
                self.stage = "hdr"; self.hdr_raw = b""; self.hdr_vals = None; self.pay_len = 0
        return blocks
    def send(self, h, p):
        try: self.sock.sendall(h+p)
        except OSError: self.close()

class Proxy:
    def __init__(self, ue_port, gnb_host, gnb_port, log_level, ch_en=True, ch_L=32, ch_dd=0, log_plot=False,
                 conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np", custom_channel=False):
        self.prev_ts = None
        self.global_symbol_count = 0
        self.slot_sample_accum = 0  # 전체 누적 sample
        self.sel = selectors.DefaultSelector()
        self.lis = socket.socket(); self.lis.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lis.bind(("0.0.0.0", ue_port)); self.lis.listen(); self.lis.setblocking(False)
        self.sel.register(self.lis, selectors.EVENT_READ, data="UE_LIS")
        print(f"[INFO] UE listen 0.0.0.0:{ue_port}")
        self.gnb_host,self.gnb_port=gnb_host,gnb_port
        self.gnb_ep:Optional[Endpoint]=None
        self.ues:Dict[int,Endpoint]={}
        self.gnb_hshake:Optional[Tuple[bytes,bytes]]=None
        self.ch_en = ch_en
        self.ch_L = ch_L
        self.ch_dd = ch_dd
        self.log_plot = log_plot
        self.conv_mode = conv_mode
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.fft_lib = fft_lib
        self.custom_channel = custom_channel
        if self.custom_channel:
            #Topoplogy
            phi_r_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/phi_r_rays_for_ChannelBlock.npy"))
            phi_t_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/phi_t_rays_for_ChannelBlock.npy"))
            theta_r_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/theta_r_rays_for_ChannelBlock.npy"))
            theta_t_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/theta_t_rays_for_ChannelBlock.npy"))
            power_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/power_rays_for_ChannelBlock.npy"))
            tau_rays_for_ChannelBlock = tf.convert_to_tensor(np.load(directory+f"/tau_rays_for_ChannelBlock.npy"))
            batch_size = 1
            self.N_UE = 1
            self.N_BS = 1
            self.num_rx = 1
            self.num_tx = 1
            BSexample, _ = set_BS()
                
            ArrayRX = PanelArray(num_rows_per_panel = 1,
                                num_cols_per_panel = 1,
                                num_rows = 1,
                                num_cols = 1,
                                polarization = 'single',
                                polarization_type = 'V',
                                antenna_pattern = 'omni', # ["omni","38.901"]
                                carrier_frequency = carrier_frequency)
            
            ArrayTX = PanelArray(num_rows_per_panel = BSexample["num_rows_per_panel"],
                      num_cols_per_panel = BSexample["num_cols_per_panel"],
                      num_rows = BSexample["num_rows"],
                      num_cols = BSexample["num_cols"],
                      polarization = BSexample["polarization"],
                      polarization_type = BSexample["polarization_type"],
                      antenna_pattern = BSexample["antenna_pattern"], # ["omni","38.901"]
                      carrier_frequency = carrier_frequency,
                      panel_vertical_spacing = BSexample["panel_vertical_spacing"],
                      panel_horizontal_spacing = BSexample["panel_horizontal_spacing"])

            aoa_pdp = phi_r_rays_for_ChannelBlock
            aod_pdp = phi_t_rays_for_ChannelBlock
            zoa_pdp = theta_r_rays_for_ChannelBlock
            zod_pdp = theta_t_rays_for_ChannelBlock

            power_pdp = power_rays_for_ChannelBlock
            delay_pdp = tau_rays_for_ChannelBlock

            mean_xpr_list   = {"UMi-LOS":9,"UMi-NLOS":8, "UMa-LOS":8,"UMa-NLOS":7}
            stddev_xpr_list = {"UMi-LOS":3,"UMi-NLOS":3, "UMa-LOS":4,"UMa-NLOS":4}
            mean_xpr = mean_xpr_list["UMa-NLOS"]
            stddev_xpr = stddev_xpr_list["UMa-NLOS"]
            xpr_pdp   = 10**(tf.random.normal(shape=[batch_size,self.N_BS,self.N_UE,1,aoa_pdp.shape[-1]], mean=mean_xpr, stddev=stddev_xpr)/10)

            PDP = Rays(delays=delay_pdp,
                    powers=power_pdp,
                    aoa=aoa_pdp,
                    aod=aod_pdp,
                    zoa=zoa_pdp,
                    zod=zod_pdp,
                    xpr=xpr_pdp)
            velocities = tf.abs(tf.random.normal(shape=[batch_size,self.N_UE,3], mean=1, stddev=0.1, dtype=tf.float32)) # from UE API
            moving_end = "rx"
            los_aoa = tf.zeros([batch_size,self.N_BS,self.N_UE])
            los_aod = tf.zeros([batch_size,self.N_BS,self.N_UE])
            los_zoa = tf.zeros([batch_size,self.N_BS,self.N_UE])
            los_zod = tf.zeros([batch_size,self.N_BS,self.N_UE])
            los     = tf.random.uniform(shape=[batch_size,self.N_BS,self.N_UE], minval=0, maxval=2, dtype=tf.int32 ) > 0
            distance_3d = tf.ones([1,self.N_BS,self.N_UE]) # BS->UE propagation delay 때문에 생기는 phase shift 처리하는 것. B,P 차원이어야 한다
            tx_orientations = tf.random.normal(shape=[batch_size,self.N_BS,3], mean=0, stddev=PI/5, dtype=tf.float32) # radian
            rx_orientations = tf.random.normal(shape=[batch_size,self.N_UE,3], mean=0, stddev=PI/5, dtype=tf.float32) # radian

            self.topology = Topology(velocities,
                                moving_end,
                                los_aoa,
                                los_aod,
                                los_zoa,
                                los_zod,
                                los,
                                distance_3d,
                                tx_orientations,
                                rx_orientations)
            
            self.Channel_Generator = ChannelCoefficientsGeneratorJIN(carrier_frequency, scs, ArrayTX, ArrayRX, False)

            h_field_array_power, aoa_delay, zoa_delay = self.Channel_Generator._H_PDP_FIX(self.topology, PDP, N_FFT, scs) 
            self.h_field_array_power = tf.transpose(h_field_array_power, [0,3,5,6,1,2,7,4]) # [B, N_Rays, N_r, N_t, N_BS, N_UE, N_sym, N_FFT]
            self.aoa_delay = tf.transpose(aoa_delay, [0,3,1,2,4]) # [B, N_BS, N_UE, N_Rays, N_FFT] -> [B, N_Rays, N_BS, N_UE, N_FFT]
            self.zoa_delay = tf.transpose(zoa_delay, [0,3,1,2,4]) # [B, N_BS, N_UE, N_Rays, N_FFT] -> [B, N_Rays, N_BS, N_UE, N_FFT]
        else:
            self.h_time, self.h_nonzero = gen_time_channel_impulse(L=self.ch_L, N=FFT_SIZE)
            self.H_f = get_channel_freq_h_from_time(self.h_time, FFT_SIZE, fft_lib=self.fft_lib)
    def connect_gnb(self):
        try:
            s=socket.create_connection((self.gnb_host,self.gnb_port),timeout=5)
            s.setblocking(False)
            self.gnb_ep=Endpoint(s,"gNB")
            self.sel.register(s,selectors.EVENT_READ,data=self.gnb_ep)
            print(f"[INFO] gNB connected {self.gnb_host}:{self.gnb_port}")
        except OSError as e:
            print(f"[WARN] gNB connect fail: {e}")
    def _reconnect_gnb_if_needed(self):
        if self.gnb_ep and not self.gnb_ep.closed: return
        if self.gnb_ep:
            try: self.sel.unregister(self.gnb_ep.sock)
            except: pass
            self.gnb_ep = None
        self.connect_gnb()
    def _accept_ue(self):
        try: c, addr = self.lis.accept(); c.setblocking(False)
        except OSError: return
        ue=Endpoint(c,"UE"); self.ues[ue.fileno()]=ue
        self.sel.register(c,selectors.EVENT_READ,data=ue)
        print(f"[INFO] UE joined {addr}")
        if self.gnb_hshake:
            h,p=self.gnb_hshake
            ue.send(h,p)
            size,nb,ts,frame,subframe = unpack_header(h)
            log("Proxy → UE", size, nb, ts, frame, subframe, size*nb, "(cached handshake)")
    def _handle_ep(self,ep:Endpoint):
        for hdr_raw, hdr_vals, payload in ep.read_blocks():
            size, nb, ts, frame, subframe = hdr_vals
            sample_cnt = size * nb
            expected_bytes = size * nb * 4
            if len(payload) != expected_bytes:
                print(f"[WARN] bad payload len {len(payload)} != {expected_bytes} (drop)")
                continue
            if getattr(self, "prev_ts", None) is not None and ts < self.prev_ts:
                print(f"[WARN] ts rollback {ts} < {self.prev_ts}")
                # drop하지 않음
            self.prev_ts = ts
            if size > 1 and self.ch_en:
                if self.custom_channel:
                    
                    processed = self.process_ofdm_slot_with_custom_channel(
                        payload,
                        ts,
                        log_plot=self.log_plot,
                        conv_mode=self.conv_mode,
                        block_size=self.block_size,
                        num_blocks=self.num_blocks,
                        fft_lib=self.fft_lib
                    )
                    ch_note = f" (custom ch, ch.conv:{self.conv_mode}/{self.fft_lib})"
                else:
                    processed = process_ofdm_slot_with_channel(
                        payload,
                        self.h_time,
                        self.h_nonzero,
                        self.H_f,
                        log_plot=self.log_plot,
                        conv_mode=self.conv_mode,
                        block_size=self.block_size,
                        num_blocks=self.num_blocks,
                        fft_lib=self.fft_lib
                    )
                    ch_note = f" (ch.conv:{self.conv_mode}/{self.fft_lib})"
            else:
                processed = payload
                ch_note = ""
            if ep.role == "gNB":
                log("gNB → Proxy", size, nb, ts, frame, subframe, sample_cnt, "(handshake)" if size==1 else ch_note)
                for u in list(self.ues.values()):
                    if u.closed: continue
                    if len(processed) != expected_bytes:
                        processed = processed[:expected_bytes] if len(processed) > expected_bytes else processed + b'\x00' * (expected_bytes - len(processed))
                    u.send(hdr_raw, processed)
                    log("Proxy → UE", size, nb, ts, frame, subframe, sample_cnt, "(handshake)" if size==1 else ch_note)
            else:
                log("UE → Proxy", size, nb, ts, frame, subframe, sample_cnt, ch_note)
                if self.gnb_ep and not self.gnb_ep.closed:
                    if len(processed) != expected_bytes:
                        processed = processed[:expected_bytes] if len(processed) > expected_bytes else processed + b'\x00' * (expected_bytes - len(processed))
                    self.gnb_ep.send(hdr_raw, processed)
                    log("Proxy → gNB", size, nb, ts, frame, subframe, sample_cnt, ch_note)

    def process_ofdm_slot_with_custom_channel(self, iq_bytes, ts, log_plot=False,
                                          conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np"):
        x = np.frombuffer(iq_bytes, dtype='<i2')
        x_cpx = x[::2] + 1j * x[1::2]
        total_samples = len(x_cpx)
        sym_idx = get_ofdm_symbol_indices(total_samples)

        sample_times = tf.cast(
        ts + tf.constant([start for (start, _) in sym_idx], dtype=tf.float32),  # 또는 tf.float64
        self.Channel_Generator.rdtype) / tf.constant(Fs, self.Channel_Generator.rdtype)

        N_BS_serving = self.num_tx
        N_UE_active = self.num_rx
        ActiveUE_component = random_binary_mask_tf_complex64(self.N_UE, k=N_UE_active)
        ActiveUE = tf.constant(ActiveUE_component, dtype=tf.complex64)
        ServingBS_component = random_binary_mask_tf_complex64(self.N_BS, k=N_BS_serving)
        ServingBS = tf.constant(ServingBS_component, dtype=tf.complex64)

        h_delay, _, _, _ = self.Channel_Generator._H_TTI_sequential_fft_o_ELW2_noProfile(
                    self.topology, ActiveUE, ServingBS, sample_times, 
                    self.h_field_array_power, self.aoa_delay, self.zoa_delay
                )
        h_delay = tf.squeeze(h_delay).numpy().astype(np.complex64)
        p = np.mean(np.abs(h_delay)**2)
        if p > 0:
            h_delay = h_delay / np.sqrt(p)
        out = np.zeros_like(x_cpx) 
        symbol_waveforms = []
        out_waveforms = []
        N_symbols = min(len(sym_idx), h_delay.shape[0])

        for n, (start, end) in enumerate(sym_idx[:N_symbols]):
            cp_len = CP1 if n < 12 else CP2
            if end - start != FFT_SIZE + cp_len:
                continue
            sym = x_cpx[start + cp_len: end]
            h_this = np.asarray(h_delay[n], dtype=np.complex64)
            if conv_mode == "fft":
                Hf = np.fft.fft(h_this, FFT_SIZE)
                Xf = np.fft.fft(sym)
                Yf = Xf * Hf
                y  = np.fft.ifft(Yf)
            else:
                y = fft_convolution_block(sym, h_this, mode=conv_mode,
                                        block_size=block_size, num_blocks=num_blocks, fft_lib=fft_lib)
            out[start + cp_len: end] = y
            out[start: start + cp_len] = y[-cp_len:]
            symbol_waveforms.append(sym.copy())
            out_waveforms.append(y.copy())

        y_int16 = np.empty(len(out)*2, dtype='<i2')
        y_int16[::2] = np.clip(np.round(out.real), -32768, 32767).astype('<i2')
        y_int16[1::2] = np.clip(np.round(out.imag), -32768, 32767).astype('<i2')
        if log_plot and len(symbol_waveforms) > 0:
            plt.figure(figsize=(12,6))
            plt.subplot(2,1,1)
            plt.title("첫번째 OFDM 심볼 (원본, Real/Imag)")
            plt.plot(np.real(symbol_waveforms[0]), label='Re[x]')
            plt.plot(np.imag(symbol_waveforms[0]), label='Im[x]')
            plt.legend(); plt.grid()
            plt.subplot(2,1,2)
            plt.title("첫번째 OFDM 심볼 (채널 통과 후, Real/Imag)")
            plt.plot(np.real(out_waveforms[0]), label='Re[y]')
            plt.plot(np.imag(out_waveforms[0]), label='Im[y]')
            plt.legend(); plt.grid()
            plt.tight_layout(); plt.show()
            print("원본 심볼 일부:", symbol_waveforms[0][:10])
            print("채널 적용 후 일부:", out_waveforms[0][:10])
        return y_int16.tobytes()
    
    def run(self):
        self.connect_gnb()
        try:
            while True:
                for key,_ in self.sel.select(0.5):
                    if key.data=="UE_LIS": self._accept_ue()
                    else: self._handle_ep(key.data)
                self._reconnect_gnb_if_needed()
        except KeyboardInterrupt:
            print("[INFO] terminated by Ctrl-C")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ue-port",type=int,default=6018)
    ap.add_argument("--gnb-host",default="127.0.0.1")
    ap.add_argument("--gnb-port",type=int,default=6017)
    ap.add_argument("--log",choices=["error","warn","info","debug"],default="info")
    ap.add_argument("--ch-en",action="store_true",default=True,help="Apply channel convolution (FFT H)")
    ap.add_argument("--ch-dd",type=int,default=0,help="Channel delay (samples, NOT used in this version)")
    ap.add_argument("--ch-L",type=int,default=32,help="Channel impulse response length (CP 혹은 블록 길이보다 짧게!)")
    ap.add_argument("--log-plot",action="store_true",default=False,help="1st OFDM symbol 원본/채널 파형 시각화")
    ap.add_argument("--conv-mode",type=str,default="oa",choices=["fft","oa","os"],help="채널 컨볼루션 방식 (fft/oa/os)")
    ap.add_argument("--block-size",type=int,default=4096,help="블록 사이즈 (oa, os모드에서)")
    ap.add_argument("--num-blocks",type=int,default=None,help="블록 개수 제한")
    ap.add_argument("--fft-lib",type=str,default="tf",choices=["np","tf"],help="FFT backend 선택: np or tf")
    ap.add_argument("--custom-channel",action="store_true",default=True,help="Custom channel 사용")
    args=ap.parse_args()
    Proxy(args.ue_port,args.gnb_host,args.gnb_port,args.log,
          ch_en=args.ch_en, ch_dd=args.ch_dd, ch_L=args.ch_L, log_plot=args.log_plot,
          conv_mode=args.conv_mode, block_size=args.block_size, num_blocks=args.num_blocks, fft_lib=args.fft_lib, custom_channel=args.custom_channel).run()

if __name__=="__main__":
    main()
