"""
================================================================================
v3_tf_precomputed.py - TensorFlow + 사전 계산된 채널 (npy 파일)

[성능] ✅ ~73ms (10 DL당) | 실시간 대비 6.1배 느림
[속도 원인] 사전 계산된 채널 사용으로 실시간 생성 오버헤드 제거

[핵심 특징]
- GPU: TensorFlow (선택적)
- 채널: 사전 계산된 .npy 파일 로드
- 장점: 실시간 채널 생성 불필요

[중간 속도 이유]
- 채널 데이터는 미리 계산되어 있음 (빠름)
- TensorFlow FFT가 CuPy보다 느림
- Pinned Memory 미사용으로 PCIe 전송 느림
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
            self.h_delay = np.squeeze(np.load(os.path.join(os.path.dirname(_SCRIPT_DIR), "h_delay_3.5_GHz_RX3.npy")))
            self.h_freq = np.fft.fft(self.h_delay, FFT_SIZE)
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
            
            if size > 1 and self.ch_en:
                if self.custom_channel:
                    processed = self.process_ofdm_slot_with_custom_channel(payload, ts,
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
                if size == 1: self.gnb_hshake = (hdr_raw, payload)
                for u in list(self.ues.values()):
                    if u.closed: continue
                    u.send(hdr_raw, processed)
                    log("Proxy → UE", size, nb, ts, frame, subframe, sample_cnt, "(handshake)" if size==1 else ch_note)
            else:
                log("UE → Proxy", size, nb, ts, frame, subframe, sample_cnt, ch_note)
                if self.gnb_ep and not self.gnb_ep.closed:
                    self.gnb_ep.send(hdr_raw, processed)
                    log("Proxy → gNB", size, nb, ts, frame, subframe, sample_cnt, ch_note)

    def process_ofdm_slot_with_custom_channel(self, iq_bytes, ts, log_plot=False,conv_mode="fft", block_size=4096, num_blocks=None, fft_lib="np"):
        x = np.frombuffer(iq_bytes, dtype='<i2')
        x_cpx = x[::2] + 1j * x[1::2]
        total_samples = len(x_cpx)
        sym_idx = get_ofdm_symbol_indices(total_samples)
        out = np.zeros_like(x_cpx)
        symbol_waveforms = []
        out_waveforms = []

        # ---- 프레임·심볼 인덱스 계산용 파라미터 ----
        N_SLOT_PER_FRAME = 20
        N_SYM_PER_SLOT = N_SYM
        N_SYM_PER_FRAME = N_SLOT_PER_FRAME * N_SYM_PER_SLOT  # 280
        N_FRAMES = 1024
        N_TOTAL_SYMBOLS = N_SYM_PER_FRAME * N_FRAMES  # 286720

        # 한 프레임 내 각 심볼의 샘플 시작 인덱스 테이블
        symbol_starts = []
        idx = 0
        for slot in range(N_SLOT_PER_FRAME):
            for sym in range(N_SYM_PER_SLOT):
                cp = 144 if sym < 12 else 160
                symbol_starts.append(idx)
                idx += cp + 2048
        SAMPLES_PER_FRAME = idx  # 614400

        import bisect

        # ts: 절대 샘플 단위
        frame_idx = ts // SAMPLES_PER_FRAME
        ts_in_frame = ts % SAMPLES_PER_FRAME
        # 이 슬롯에서 시작하는 첫번째 심볼 인덱스 찾기
        sym_base_idx = (frame_idx * N_SYM_PER_FRAME) % N_TOTAL_SYMBOLS

        for n, (start, end) in enumerate(sym_idx):
            cp_len = 144 if n < 12 else 160
            if end - start != 2048 + cp_len:
                continue

            # 실제 심볼 번호 (ts 기준 + 심볼 순서)
            # ts_in_frame: 프레임 내 offset, sym_idx[n][0]: 슬롯 내 offset
            # n은 현재 슬롯 내 OFDM 심볼 순서
            # 절대 인덱스는 sym_base_idx + n (cyclic하게 286720개에서 돌아야 함)
            symbol_index = (sym_base_idx + n) % N_TOTAL_SYMBOLS

            # 채널 계수 할당
            h_full = self.h_delay[symbol_index]    # (N_FFT,)
            h_full = h_full / np.sqrt(np.sum(np.abs(h_full)**2))
            h_nonzero = h_full[:self.ch_L]         # (ch_L,)
            Hf = self.h_freq[symbol_index]         # (N_FFT,)

            sym = x_cpx[start + cp_len: end]
            if conv_mode == "fft":
                if fft_lib == "tf":
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
                y = fft_convolution_block(
                    sym, h_nonzero, mode=conv_mode,
                    block_size=block_size, num_blocks=num_blocks, fft_lib=fft_lib
                )
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
    ap.add_argument("--fft-lib",type=str,default="np",choices=["np","tf"],help="FFT backend 선택: np or tf")
    ap.add_argument("--custom-channel",action="store_true",default=True,help="Custom channel 사용")
    args=ap.parse_args()
    Proxy(args.ue_port,args.gnb_host,args.gnb_port,args.log,
          ch_en=args.ch_en, ch_dd=args.ch_dd, ch_L=args.ch_L, log_plot=args.log_plot,
          conv_mode=args.conv_mode, block_size=args.block_size, num_blocks=args.num_blocks, fft_lib=args.fft_lib, custom_channel=args.custom_channel).run()

if __name__=="__main__":
    main()
