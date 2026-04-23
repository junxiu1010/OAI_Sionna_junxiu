"""
================================================================================
v4_multicell.py — Multi-Cell MU-MIMO Channel Proxy

v4.py를 확장하여 N개 셀을 관리하는 멀티셀 프록시.
하나의 프록시 프로세스에서 모든 셀의 채널을 관리하며,
셀간 간섭(ICI)을 선택적으로 시뮬레이션합니다.

[아키텍처]
  Cell 0: gNB_0 (SHM: gpu_ipc_shm_cell0)
          UE_0_0 (SHM: gpu_ipc_shm_cell0_ue0)
          UE_0_1 (SHM: gpu_ipc_shm_cell0_ue1)
          ...
  Cell 1: gNB_1 (SHM: gpu_ipc_shm_cell1)
          UE_1_0 (SHM: gpu_ipc_shm_cell1_ue0)
          ...

  DL: gNB_c → per-UE channel → UE[c][k] + Σ_c' ICI(gNB_c' → UE[c][k])
  UL: UE[c][k] → per-UE channel → gNB_c + Σ_c' ICI(UE[c'][k'] → gNB_c)

[사용법]
  python3 v4_multicell.py \
      --mode gpu-ipc \
      --num-cells 2 \
      --ues-per-cell 4 \
      --gnb-ant 4 --ue-ant 2 \
      --gnb-nx 2 --gnb-ny 1 \
      --ue-nx 1 --ue-ny 1 \
      --polarization dual \
      --ici-atten-dB 15
================================================================================
"""
import argparse
import os
import signal
import sys
import time

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from v4 import (
    GPUIpcV7Interface,
    GPUSlotPipeline,
    WindowProfiler,
    RingBuffer,
    NoiseProducer,
    IPCRingBufferSync,
    IPCRingBufferProducer,
    IPCRingBufferConsumer,
    UnifiedChannelProducerProcess,
    _fused_clip_cast_kernel,
    GPU_IPC_V7_SHM_PATH,
    GPU_IPC_V7_CIR_TIME,
    GPU_IPC_V7_SAMPLE_SIZE,
    GPU_IPC_V6_SAMPLE_SIZE,
    FFT_SIZE, N_FFT, N_SYM, SYMBOL_SIZES,
    carrier_frequency, scs, Fs, Speed,
    GNB_ANT, UE_ANT, GNB_NX, GNB_NY, UE_NX, UE_NY,
    load_p1b_stacked, pick_random_rx_indices, validate_rx_indices,
    set_BS, get_ofdm_symbol_indices,
)
import v4 as v4_mod

import multiprocessing as _mp
_mp_ctx = _mp.get_context('spawn')

gpu_num = 0

# ── CsiNet Sidecar Hook (Phase 4 Integration) ──
_CSINET_HOOK = None

def get_csinet_hook():
    """Lazy-init the CsiNet channel hook if CSINET_ENABLED env var is set."""
    global _CSINET_HOOK
    if _CSINET_HOOK is not None:
        return _CSINET_HOOK
    if os.environ.get("CSINET_ENABLED", "0") == "1":
        try:
            sys.path.insert(0, os.environ.get("CSINET_PATH",
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


class CellContext:
    """Per-cell state: gNB IPC + UE IPCs + pipelines + channel."""
    def __init__(self, cell_idx, ues_per_cell, gnb_ant, ue_ant,
                 gnb_nx, gnb_ny, ue_nx, ue_ny, polarization,
                 custom_channel, use_cuda_graph, enable_gpu,
                 use_pinned_memory, profile_interval, profile_window,
                 dual_timer_compare, p1b_npz=None, ue_rx_indices=None,
                 use_xla=False, scenario="UMa-NLOS",
                 bs_height_m=25.0, ue_height_m=1.5, isd_m=500,
                 min_ue_dist_m=35, max_ue_dist_m=500,
                 shadow_fading_std_dB=6.0,
                 k_factor_mean_dB=None, k_factor_std_dB=None,
                 sector_half_deg=90.0, ue_speeds=None):
        self.cell_idx = cell_idx
        self.ues_per_cell = ues_per_cell
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.gnb_nx = gnb_nx
        self.gnb_ny = gnb_ny
        self.ue_nx = ue_nx
        self.ue_ny = ue_ny
        self.polarization = polarization
        self.custom_channel = custom_channel
        self.scenario = scenario
        self.bs_height_m = bs_height_m
        self.ue_height_m = ue_height_m
        self.isd_m = isd_m
        self.min_ue_dist_m = min_ue_dist_m
        self.max_ue_dist_m = max_ue_dist_m
        self.shadow_fading_std_dB = shadow_fading_std_dB
        self.k_factor_mean_dB = k_factor_mean_dB
        self.k_factor_std_dB = k_factor_std_dB
        self.sector_half_deg = sector_half_deg
        self.ue_speeds = ue_speeds

        self.ipc_gnb = None
        self.ipc_ues = []
        self.pipelines_dl = []
        self.pipelines_ul = []
        self.channel_buffers = []
        self.channel_producers = []
        self._channel_stop_events = []
        self._last_ch_cache = [None] * ues_per_cell
        self.noise_producers = []

        _use_graph = False  # disabled for stability
        pipeline_common = dict(
            enable_gpu=enable_gpu,
            use_pinned_memory=use_pinned_memory,
            use_cuda_graph=_use_graph,
            profile_interval=profile_interval,
            profile_window=profile_window,
            dual_timer_compare=dual_timer_compare)

        total_cpx = sum(SYMBOL_SIZES)
        noise_len_dl = total_cpx * ue_ant
        noise_len_ul = total_cpx * gnb_ant

        for k in range(ues_per_cell):
            pdl = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=gnb_ant, n_rx_out=ue_ant,
                noise_buffer=None, **pipeline_common)
            pul = GPUSlotPipeline(
                FFT_SIZE, n_tx_in=ue_ant, n_rx_out=gnb_ant,
                noise_buffer=None, **pipeline_common)
            self.pipelines_dl.append(pdl)
            self.pipelines_ul.append(pul)

        self.p1b_npz = p1b_npz
        self.ue_rx_indices = ue_rx_indices
        self.use_xla = use_xla

    def init_ipc(self):
        """Create and initialize gNB + UE IPC interfaces."""
        gnb_shm = f"/tmp/oai_gpu_ipc/gpu_ipc_shm_cell{self.cell_idx}"
        self.ipc_gnb = GPUIpcV7Interface(
            gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
            shm_path=gnb_shm)
        if not self.ipc_gnb.init():
            raise RuntimeError(f"Cell {self.cell_idx}: gNB IPC init failed")
        print(f"[MC] Cell {self.cell_idx} gNB IPC ready: {gnb_shm}")

        for k in range(self.ues_per_cell):
            ue_shm = f"/tmp/oai_gpu_ipc/gpu_ipc_shm_cell{self.cell_idx}_ue{k}"
            ipc_ue = GPUIpcV7Interface(
                gnb_ant=self.gnb_ant, ue_ant=self.ue_ant,
                shm_path=ue_shm)
            if not ipc_ue.init():
                raise RuntimeError(f"Cell {self.cell_idx} UE[{k}]: IPC init failed")
            self.ipc_ues.append(ipc_ue)
            print(f"[MC] Cell {self.cell_idx} UE[{k}] IPC ready: {ue_shm}")

    def init_channel(self, buffer_len=1024, buffer_symbol_size=42):
        """Initialize channel producers for this cell."""
        if not self.custom_channel:
            print(f"[MC] Cell {self.cell_idx}: bypass mode — no channel")
            return

        N = self.ues_per_cell
        npy_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "saved_rays_data")

        config = {
            'gnb_nx': self.gnb_nx, 'gnb_ny': self.gnb_ny,
            'ue_nx': self.ue_nx, 'ue_ny': self.ue_ny,
            'carrier_frequency': carrier_frequency,
            'scs': scs, 'N_FFT': N_FFT, 'Fs': Fs,
            'buffer_symbol_size': buffer_symbol_size,
            'buffer_len': buffer_len,
            'npy_directory': npy_dir,
            'shape': (self.ue_ant, self.gnb_ant, FFT_SIZE),
            'N_UE': N, 'N_BS': 1,
            'num_rx': N, 'num_tx': 1,
            'num_ues': N,
            'Speed': Speed,
            'ue_speeds': getattr(self, 'ue_speeds', None),
            'scenario': self.scenario,
            'gpu_num': gpu_num,
            'use_xla': self.use_xla,
            'polarization': self.polarization,
            'bs_height_m': getattr(self, 'bs_height_m', 25.0),
            'ue_height_m': getattr(self, 'ue_height_m', 1.5),
            'isd_m': getattr(self, 'isd_m', 500),
            'min_ue_dist_m': getattr(self, 'min_ue_dist_m', 35),
            'max_ue_dist_m': getattr(self, 'max_ue_dist_m', 500),
            'shadow_fading_std_dB': getattr(self, 'shadow_fading_std_dB', 6.0),
            'k_factor_mean_dB': getattr(self, 'k_factor_mean_dB', None),
            'k_factor_std_dB': getattr(self, 'k_factor_std_dB', None),
            'sector_half_deg': getattr(self, 'sector_half_deg', 90.0),
        }

        if self.p1b_npz and self.ue_rx_indices:
            config['ray_data_stacked'] = load_p1b_stacked(
                self.p1b_npz, self.ue_rx_indices)

        syncs = [IPCRingBufferSync(maxlen=buffer_len, ctx=_mp_ctx) for _ in range(N)]
        handle_q = _mp_ctx.Queue()
        stop_ev = _mp_ctx.Event()

        proc = UnifiedChannelProducerProcess(config, syncs, handle_q, stop_ev)
        proc.start()
        print(f"[MC] Cell {self.cell_idx} ChannelProducer started (pid={proc.pid}, N_UE={N})")

        for k in range(N):
            try:
                if not proc.is_alive():
                    raise RuntimeError(
                        f"Cell {self.cell_idx}: ChannelProducer died "
                        f"(exit code={proc.exitcode})")
                ipc_handle = handle_q.get(timeout=120)
            except Exception as e:
                if proc.is_alive():
                    proc.terminate()
                raise RuntimeError(
                    f"Cell {self.cell_idx} UE[{k}]: channel handle failed") from e

            consumer_k = IPCRingBufferConsumer(
                ipc_handle, config['shape'], cp.complex128, syncs[k])
            self.channel_buffers.append(consumer_k)

        self.channel_producers.append(proc)
        self._channel_stop_events.append(stop_ev)
        print(f"[MC] Cell {self.cell_idx}: channel initialized ({N} UEs)")

    def cleanup(self):
        for evt in self._channel_stop_events:
            evt.set()
        for proc in self.channel_producers:
            if hasattr(proc, 'join'):
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
        for buf in self.channel_buffers:
            if hasattr(buf, 'cleanup'):
                try:
                    buf.cleanup()
                except Exception:
                    pass
        if self.ipc_gnb:
            self.ipc_gnb.cleanup()
        for ipc in self.ipc_ues:
            ipc.cleanup()


class MultiCellProxy:
    """Multi-cell proxy managing N cells with optional inter-cell interference."""

    def __init__(self, num_cells, ues_per_cell, gnb_ant, ue_ant,
                 gnb_nx, gnb_ny, ue_nx, ue_ny, polarization,
                 custom_channel=True, ici_atten_dB=15.0,
                 use_cuda_graph=True, enable_gpu=True,
                 use_pinned_memory=True,
                 profile_interval=100, profile_window=500,
                 dual_timer_compare=True,
                 p1b_npz=None, ue_rx_indices_all=None,
                 use_xla=False,
                 time_dilation=1.0,
                 bypass_duration=0,
                 scenario="UMa-NLOS",
                 bs_height_m=25.0, ue_height_m=1.5, isd_m=500,
                 min_ue_dist_m=35, max_ue_dist_m=500,
                 shadow_fading_std_dB=6.0,
                 k_factor_mean_dB=None, k_factor_std_dB=None,
                 sector_half_deg=90.0, ue_speeds=None):
        self.num_cells = num_cells
        self.ues_per_cell = ues_per_cell
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.custom_channel = custom_channel
        self.bypass_duration = bypass_duration
        self.scenario = scenario
        self.bs_height_m = bs_height_m
        self.ue_height_m = ue_height_m
        self.isd_m = isd_m
        self.min_ue_dist_m = min_ue_dist_m
        self.max_ue_dist_m = max_ue_dist_m
        self.shadow_fading_std_dB = shadow_fading_std_dB
        self.k_factor_mean_dB = k_factor_mean_dB
        self.k_factor_std_dB = k_factor_std_dB
        self.sector_half_deg = sector_half_deg
        self.ue_speeds = ue_speeds
        self._channel_active = (bypass_duration <= 0 and custom_channel)
        self._phase_switched = False
        self.ici_atten_dB = ici_atten_dB
        self.ici_linear = 10 ** (-ici_atten_dB / 20.0)
        self.enable_ici = (ici_atten_dB < 100.0)
        self.time_dilation = max(1.0, time_dilation)

        self.cells = []
        for c in range(num_cells):
            rx_indices_c = None
            if ue_rx_indices_all and c < len(ue_rx_indices_all):
                rx_indices_c = ue_rx_indices_all[c]

            cell = CellContext(
                cell_idx=c,
                ues_per_cell=ues_per_cell,
                gnb_ant=gnb_ant, ue_ant=ue_ant,
                gnb_nx=gnb_nx, gnb_ny=gnb_ny,
                ue_nx=ue_nx, ue_ny=ue_ny,
                polarization=polarization,
                custom_channel=custom_channel,
                use_cuda_graph=use_cuda_graph,
                enable_gpu=enable_gpu,
                use_pinned_memory=use_pinned_memory,
                profile_interval=profile_interval,
                profile_window=profile_window,
                dual_timer_compare=dual_timer_compare,
                p1b_npz=p1b_npz,
                ue_rx_indices=rx_indices_c,
                use_xla=use_xla,
                scenario=scenario,
                bs_height_m=self.bs_height_m,
                ue_height_m=self.ue_height_m,
                isd_m=self.isd_m,
                min_ue_dist_m=self.min_ue_dist_m,
                max_ue_dist_m=self.max_ue_dist_m,
                shadow_fading_std_dB=self.shadow_fading_std_dB,
                k_factor_mean_dB=self.k_factor_mean_dB,
                k_factor_std_dB=self.k_factor_std_dB,
                sector_half_deg=self.sector_half_deg,
                ue_speeds=self.ue_speeds)
            self.cells.append(cell)

        self._dl_count = 0
        self._ul_count = 0

    def _init_all(self, buffer_len=1024, buffer_symbol_size=42):
        """Initialize IPC and channels for all cells."""
        for cell in self.cells:
            cell.init_ipc()

        for cell in self.cells:
            cell.init_channel(buffer_len, buffer_symbol_size)

        total_cpx = sum(SYMBOL_SIZES)
        if GPU_AVAILABLE:
            for cell in self.cells:
                cell._ul_accum = cp.zeros(
                    total_cpx * self.gnb_ant, dtype=cp.complex128)
                cell._ul_fused_out = cp.zeros(
                    total_cpx * self.gnb_ant * 2, dtype=cp.int16)
                cell._ul_dummy_out = cp.zeros(
                    cell.pipelines_ul[0].total_int16_out, dtype=cp.int16)

        self._warmup_all()
        print(f"[MC] All {self.num_cells} cells initialized")

    def _warmup_all(self):
        """Pre-warm all cell pipelines."""
        for cell in self.cells:
            for np_thread in cell.noise_producers:
                np_thread.start()

            if not cell.custom_channel:
                continue

            for k in range(cell.ues_per_cell):
                for label, pipeline in [("DL", cell.pipelines_dl[k]),
                                        ("UL", cell.pipelines_ul[k])]:
                    dummy_in = cp.zeros(pipeline.total_int16_in, dtype=cp.int16)
                    dummy_out = cp.zeros(pipeline.total_int16_out, dtype=cp.int16)
                    n_r, n_t = pipeline.n_rx, pipeline.n_tx
                    dummy_ch = cp.zeros((N_SYM, n_r, n_t, FFT_SIZE), dtype=cp.complex128)
                    for j in range(min(n_r, n_t)):
                        dummy_ch[:, j, j, :] = 1.0

                    for _ in range(GPUSlotPipeline.WARMUP_SLOTS + 1):
                        pipeline.process_slot_ipc(
                            dummy_in, dummy_ch, 1.0, None, False, dummy_out, None)
            print(f"[MC] Cell {cell.cell_idx}: warmup complete")

    def _dl_apply_channel_slot(self, cell, ue_idx, ts, nsamps):
        """Apply Sionna DL channel for one slot: gNB → H → UE[ue_idx].
        Falls back to bypass_copy on channel buffer timeout."""
        gnb = cell.ipc_gnb
        ue = cell.ipc_ues[ue_idx]

        arr_in = gnb.circ_read(
            gnb.gpu_dl_tx_ptr, ts, nsamps,
            gnb.dl_tx_nbAnt, gnb.dl_tx_cir_size, cp.int16)

        arr_out, wraps = ue.get_gpu_array_at(
            ue.gpu_dl_rx_ptr, ts, nsamps,
            ue.dl_rx_nbAnt, ue.dl_rx_cir_size, cp.int16)
        if wraps:
            arr_out = cp.zeros(
                nsamps * cell.ue_ant * GPU_IPC_V6_SAMPLE_SIZE // 2,
                dtype=cp.int16)

        try:
            channels, n_held = cell.channel_buffers[ue_idx].get_batch_view(N_SYM)
        except Exception:
            gnb.bypass_copy(
                ue.gpu_dl_rx_ptr, gnb.gpu_dl_tx_ptr, ts, nsamps,
                gnb.dl_tx_nbAnt, gnb.dl_tx_cir_size,
                ue.dl_rx_nbAnt, ue.dl_rx_cir_size)
            return

        if channels.shape[0] < N_SYM:
            n_r, n_t = channels.shape[1], channels.shape[2]
            pad = cp.zeros((N_SYM - channels.shape[0], n_r, n_t, FFT_SIZE),
                           dtype=channels.dtype)
            for j in range(min(n_r, n_t)):
                pad[:, j, j, :] = 1.0
            channels = cp.concatenate([channels, pad])

        # CsiNet sidecar: capture H for neural CSI compression
        csinet_hook = get_csinet_hook()
        if csinet_hook is not None and csinet_hook.enabled:
            csinet_hook.capture(cell.cell_idx, ue_idx, channels)

        h_peak = float(cp.max(cp.abs(channels)))
        in_peak_raw = float(cp.max(cp.abs(arr_in.view(cp.int16).astype(cp.float32))))
        AGC_HEADROOM = 28000.0
        n_tx_factor = float(cell.gnb_ant)

        if not hasattr(self, '_dl_in_peak_ema'):
            self._dl_in_peak_ema = {}
        prev_ema = self._dl_in_peak_ema.get(ue_idx, 0.0)
        if in_peak_raw > prev_ema:
            self._dl_in_peak_ema[ue_idx] = in_peak_raw
        else:
            self._dl_in_peak_ema[ue_idx] = max(in_peak_raw, prev_ema * 0.995)
        in_peak = self._dl_in_peak_ema[ue_idx]

        if in_peak < 1.0 and h_peak * n_tx_factor > 1.0:
            in_peak = 4000.0

        estimated_peak = in_peak * h_peak * n_tx_factor
        if estimated_peak > AGC_HEADROOM and estimated_peak > 0:
            dl_gain = AGC_HEADROOM / estimated_peak
        else:
            dl_gain = 1.0

        dl_gain = round(dl_gain * 50) / 50.0

        if not hasattr(self, '_agc_diag_cnt'):
            self._agc_diag_cnt = 0
            self._agc_clip_cnt = 0
        self._agc_diag_cnt += 1
        if dl_gain < 1.0:
            self._agc_clip_cnt += 1
        if self._agc_diag_cnt <= 20 or self._agc_diag_cnt % 5000 == 0:
            print(f"[DL AGC] #{self._agc_diag_cnt} UE{ue_idx} "
                  f"in_peak_raw={in_peak_raw:.0f} in_peak_ema={in_peak:.0f} "
                  f"h_peak={h_peak:.3f} est_peak={estimated_peak:.0f} "
                  f"gain={dl_gain:.4f} "
                  f"clips={self._agc_clip_cnt}/{self._agc_diag_cnt}")

        cell.pipelines_dl[ue_idx].process_slot_ipc(
            arr_in, channels, dl_gain, None, False, arr_out, None)

        if wraps:
            ue.circ_write(ue.gpu_dl_rx_ptr, ts, nsamps,
                          ue.dl_rx_nbAnt, ue.dl_rx_cir_size, arr_out)

        cell.channel_buffers[ue_idx].release_batch(n_held)

    def _dl_broadcast_cell(self, cell, start_ts, delta):
        """DL broadcast for one cell: gNB → per-UE channel → UE[k].
        Applies Sionna channel when _channel_active is True.
        Optionally adds ICI from other cells."""
        slot_samples = cell.pipelines_dl[0].total_cpx
        apply_ch = (self._channel_active
                    and cell.custom_channel
                    and len(cell.channel_buffers) > 0)

        if not hasattr(self, '_dl_diag_cnt'):
            self._dl_diag_cnt = 0
            self._dl_src_nz_total = 0
            self._dl_dst_nz_total = 0

        for k in range(cell.ues_per_cell):
            pos = int(start_ts)
            remaining = int(delta)
            _dl_slot_idx = 0

            while remaining >= slot_samples:
                if apply_ch:
                    self._dl_apply_channel_slot(cell, k, pos, slot_samples)
                else:
                    cell.ipc_gnb.bypass_copy(
                        cell.ipc_ues[k].gpu_dl_rx_ptr,
                        cell.ipc_gnb.gpu_dl_tx_ptr,
                        pos, slot_samples,
                        cell.ipc_gnb.dl_tx_nbAnt, cell.ipc_gnb.dl_tx_cir_size,
                        cell.ipc_ues[k].dl_rx_nbAnt, cell.ipc_ues[k].dl_rx_cir_size)

                    if k == 0:
                        self._dl_diag_cnt += 1
                        do_log = (self._dl_diag_cnt <= 10
                                  or self._dl_diag_cnt % 2000 == 0)
                        if do_log:
                            src_arr = cell.ipc_gnb.circ_read(
                                cell.ipc_gnb.gpu_dl_tx_ptr,
                                pos, slot_samples,
                                cell.ipc_gnb.dl_tx_nbAnt,
                                cell.ipc_gnb.dl_tx_cir_size,
                                cp.int16)
                            dst_arr = cell.ipc_ues[0].circ_read(
                                cell.ipc_ues[0].gpu_dl_rx_ptr,
                                pos, slot_samples,
                                cell.ipc_ues[0].dl_rx_nbAnt,
                                cell.ipc_ues[0].dl_rx_cir_size,
                                cp.int16)
                            src_nz = int(cp.count_nonzero(src_arr))
                            dst_nz = int(cp.count_nonzero(dst_arr))
                            if src_nz > 0:
                                self._dl_src_nz_total += 1
                            if dst_nz > 0:
                                self._dl_dst_nz_total += 1
                            print(f"[DL COPY DIAG] #{self._dl_diag_cnt} "
                                  f"ts={pos} src_nz={src_nz}/{len(src_arr)} "
                                  f"dst_nz={dst_nz}/{len(dst_arr)} "
                                  f"src_total={self._dl_src_nz_total} "
                                  f"dst_total={self._dl_dst_nz_total}")

                if self.enable_ici and self.num_cells > 1:
                    self._add_dl_ici(cell, k, pos, slot_samples)

                pos += slot_samples
                remaining -= slot_samples
                _dl_slot_idx += 1

            if remaining > 0:
                cell.ipc_gnb.bypass_copy(
                    cell.ipc_ues[k].gpu_dl_rx_ptr,
                    cell.ipc_gnb.gpu_dl_tx_ptr,
                    pos, remaining,
                    cell.ipc_gnb.dl_tx_nbAnt, cell.ipc_gnb.dl_tx_cir_size,
                    cell.ipc_ues[k].dl_rx_nbAnt, cell.ipc_ues[k].dl_rx_cir_size)

            actual_end = max(int(start_ts + delta - 1), 0)
            cell.ipc_ues[k].set_last_dl_rx_ts(actual_end)

    def _add_dl_ici(self, victim_cell, ue_idx, ts, nsamps):
        """Add inter-cell interference to UE[ue_idx] in victim_cell.
        Reads other cells' gNB DL signals, attenuates, and adds."""
        if not GPU_AVAILABLE:
            return
        for aggr_cell in self.cells:
            if aggr_cell.cell_idx == victim_cell.cell_idx:
                continue
            try:
                aggr_last_ts = aggr_cell.ipc_gnb.get_last_dl_tx_ts()
                if aggr_last_ts <= 0:
                    continue
                aggr_signal = aggr_cell.ipc_gnb.circ_read(
                    aggr_cell.ipc_gnb.gpu_dl_tx_ptr, ts, nsamps,
                    aggr_cell.ipc_gnb.dl_tx_nbAnt,
                    aggr_cell.ipc_gnb.dl_tx_cir_size, cp.int16)

                ue_rx = victim_cell.ipc_ues[ue_idx].circ_read(
                    victim_cell.ipc_ues[ue_idx].gpu_dl_rx_ptr, ts, nsamps,
                    victim_cell.ipc_ues[ue_idx].dl_rx_nbAnt,
                    victim_cell.ipc_ues[ue_idx].dl_rx_cir_size, cp.int16)

                ici = (aggr_signal.astype(cp.float32) * self.ici_linear).astype(cp.int16)

                min_len = min(len(ue_rx), len(ici))
                combined = ue_rx[:min_len].astype(cp.float32) + ici[:min_len].astype(cp.float32)
                combined = cp.clip(combined, -32768, 32767).astype(cp.int16)

                victim_cell.ipc_ues[ue_idx].circ_write(
                    victim_cell.ipc_ues[ue_idx].gpu_dl_rx_ptr, ts, nsamps,
                    victim_cell.ipc_ues[ue_idx].dl_rx_nbAnt,
                    victim_cell.ipc_ues[ue_idx].dl_rx_cir_size, combined)
            except Exception as e:
                pass

    def _ul_apply_channel_slot(self, cell, ue_idx, ts, nsamps):
        """Apply Sionna UL channel for one UE: UE[ue_idx] → H^T → gNB signal.
        Returns complex128 array for accumulation. Falls back to raw signal."""
        ue = cell.ipc_ues[ue_idx]

        arr_in = ue.circ_read(
            ue.gpu_ul_tx_ptr, ts, nsamps,
            ue.ul_tx_nbAnt, ue.ul_tx_cir_size, cp.int16)

        try:
            channels, n_held = cell.channel_buffers[ue_idx].get_batch_view(N_SYM)
        except Exception:
            arr_f64 = arr_in.astype(cp.float64).reshape(-1, 2)
            return arr_f64[:, 0] + 1j * arr_f64[:, 1]

        if channels.shape[0] < N_SYM:
            n_r, n_t = channels.shape[1], channels.shape[2]
            pad = cp.zeros((N_SYM - channels.shape[0], n_r, n_t, FFT_SIZE),
                           dtype=channels.dtype)
            for j in range(min(n_r, n_t)):
                pad[:, j, j, :] = 1.0
            channels = cp.concatenate([channels, pad])

        channels_ul = channels.transpose(0, 2, 1, 3)

        h_peak = float(cp.max(cp.abs(channels_ul)))
        in_peak_raw = float(cp.max(cp.abs(arr_in.view(cp.int16).astype(cp.float32))))
        UL_AGC_HEADROOM = 28000.0
        n_ues = cell.ues_per_cell
        n_tx_ul = float(cell.ue_ant)

        if not hasattr(self, '_ul_in_peak_ema'):
            self._ul_in_peak_ema = {}
        prev_ema = self._ul_in_peak_ema.get(ue_idx, 0.0)
        if in_peak_raw > prev_ema:
            self._ul_in_peak_ema[ue_idx] = in_peak_raw
        else:
            self._ul_in_peak_ema[ue_idx] = max(in_peak_raw, prev_ema * 0.995)
        in_peak = self._ul_in_peak_ema[ue_idx]

        if in_peak < 1.0 and h_peak * n_tx_ul > 1.0:
            in_peak = 4000.0

        estimated_peak = in_peak * h_peak * n_tx_ul * max(n_ues, 1)
        if estimated_peak > UL_AGC_HEADROOM and estimated_peak > 0:
            ul_gain = UL_AGC_HEADROOM / estimated_peak
        else:
            ul_gain = 1.0

        ul_gain = round(ul_gain * 50) / 50.0

        if not hasattr(self, '_ul_agc_diag_cnt'):
            self._ul_agc_diag_cnt = 0
            self._ul_agc_clip_cnt = 0
        self._ul_agc_diag_cnt += 1
        if ul_gain < 1.0:
            self._ul_agc_clip_cnt += 1
        if self._ul_agc_diag_cnt <= 20 or self._ul_agc_diag_cnt % 5000 == 0:
            print(f"[UL AGC] #{self._ul_agc_diag_cnt} UE{ue_idx} "
                  f"in_peak_raw={in_peak_raw:.0f} in_peak_ema={in_peak:.0f} "
                  f"h_peak={h_peak:.3f} est_peak={estimated_peak:.0f} "
                  f"gain={ul_gain:.4f} "
                  f"clips={self._ul_agc_clip_cnt}/{self._ul_agc_diag_cnt}")

        arr_out = cp.zeros(
            nsamps * cell.gnb_ant * GPU_IPC_V6_SAMPLE_SIZE // 2,
            dtype=cp.int16)

        cell.pipelines_ul[ue_idx].process_slot_ipc(
            arr_in, channels_ul, ul_gain, None, False, arr_out, None)

        cell.channel_buffers[ue_idx].release_batch(n_held)

        arr_f64 = arr_out.astype(cp.float64).reshape(-1, 2)
        return arr_f64[:, 0] + 1j * arr_f64[:, 1]

    def _ul_combine_cell(self, cell, start_ts, delta, active_ues=None):
        """UL superposition for one cell: sum UE signals → gNB ul_rx.
        Applies Sionna channel when _channel_active is True.
        Optionally adds ICI from other cells' UEs."""
        slot_samples = cell.pipelines_ul[0].total_cpx
        apply_ch = (self._channel_active
                    and cell.custom_channel
                    and len(cell.channel_buffers) > 0)
        ues = active_ues if active_ues is not None else range(cell.ues_per_cell)
        pos = int(start_ts)
        remaining = int(delta)

        if not hasattr(self, '_ul_diag_cnt'):
            self._ul_diag_cnt = 0
            self._ul_diag_nz = 0

        while remaining >= slot_samples:
            cell._ul_accum[:] = 0

            for k in ues:
                if apply_ch:
                    ue_cpx = self._ul_apply_channel_slot(cell, k, pos, slot_samples)
                    n_ue = len(ue_cpx)
                    cell._ul_accum[:n_ue] += ue_cpx
                else:
                    ue_ant_k = cell.ipc_ues[k].ul_tx_nbAnt
                    arr_in = cell.ipc_ues[k].circ_read(
                        cell.ipc_ues[k].gpu_ul_tx_ptr, pos, slot_samples,
                        ue_ant_k,
                        cell.ipc_ues[k].ul_tx_cir_size, cp.int16)

                    if self._ul_diag_cnt < 10 or self._ul_diag_cnt % 2000 == 0:
                        _nz = int(cp.count_nonzero(arr_in))
                        _ue_ts = cell.ipc_ues[k].get_last_ul_tx_ts()
                        _ue_ns = cell.ipc_ues[k].get_last_ul_tx_nsamps()
                        _max_v = int(cp.max(cp.abs(arr_in)))
                        print(f"[UL DIAG] slot#{self._ul_diag_cnt} UE{k} pos={pos} "
                              f"nz={_nz}/{len(arr_in)} max={_max_v} "
                              f"ue_ts={_ue_ts} ue_ns={_ue_ns} "
                              f"ue_end={_ue_ts+_ue_ns if _ue_ts>0 and _ue_ns>0 else 0}")
                        if _nz > 0:
                            self._ul_diag_nz += 1

                    arr_f64 = arr_in.astype(cp.float64)
                    arr_cpx = arr_f64.reshape(-1, 2)
                    ue_cpx = arr_cpx[:, 0] + 1j * arr_cpx[:, 1]
                    if ue_ant_k == self.gnb_ant:
                        cell._ul_accum[:len(ue_cpx)] += ue_cpx
                    else:
                        ue_2d = ue_cpx.reshape(slot_samples, ue_ant_k)
                        min_ant = min(ue_ant_k, self.gnb_ant)
                        gnb_2d = cp.zeros((slot_samples, self.gnb_ant),
                                          dtype=cp.complex128)
                        gnb_2d[:, :min_ant] = ue_2d[:, :min_ant]
                        cell._ul_accum += gnb_2d.ravel()

            if self.enable_ici and self.num_cells > 1:
                self._add_ul_ici(cell, pos, slot_samples)

            n_elem = slot_samples * self.gnb_ant * 2
            accum_nz = int(cp.count_nonzero(cell._ul_accum))

            if accum_nz > 0:
                accum_f64 = cell._ul_accum.view(cp.float64)
                pre_max = float(cp.max(cp.abs(accum_f64)))
                UL_SUPERPOS_HEADROOM = 30000.0
                if pre_max > UL_SUPERPOS_HEADROOM:
                    sp_scale = UL_SUPERPOS_HEADROOM / pre_max
                    accum_f64 *= sp_scale
                else:
                    sp_scale = 1.0

                threads = 256
                blocks = (n_elem + threads - 1) // threads
                _fused_clip_cast_kernel(
                    (blocks,), (threads,),
                    (accum_f64, cell._ul_fused_out, n_elem))
                cell.ipc_gnb.circ_write(
                    cell.ipc_gnb.gpu_ul_rx_ptr, pos, slot_samples,
                    cell.ipc_gnb.ul_rx_nbAnt,
                    cell.ipc_gnb.ul_rx_cir_size, cell._ul_fused_out)

            if self._ul_diag_cnt < 10 or self._ul_diag_cnt % 2000 == 0:
                _out_nz = int(cp.count_nonzero(cell._ul_fused_out)) if accum_nz > 0 else 0
                _pre = pre_max if accum_nz > 0 else 0.0
                _sp = sp_scale if accum_nz > 0 else 1.0
                print(f"[UL DIAG] slot#{self._ul_diag_cnt} accum_nz={accum_nz} "
                      f"out_nz={_out_nz}/{n_elem} pre_max={_pre:.0f} "
                      f"sp_scale={_sp:.4f} "
                      f"{'DATA' if accum_nz > 0 else 'SKIP'} "
                      f"total_nz_slots={self._ul_diag_nz}")
            self._ul_diag_cnt += 1

            _slot_end = int(pos + slot_samples - 1)
            cell.ipc_gnb.set_last_ul_rx_ts(_slot_end)

            pos += slot_samples
            remaining -= slot_samples

    def _add_ul_ici(self, victim_cell, ts, nsamps):
        """Add UL inter-cell interference to victim_cell's gNB.
        Other cells' UE signals leak into this gNB's receiver."""
        for aggr_cell in self.cells:
            if aggr_cell.cell_idx == victim_cell.cell_idx:
                continue
            for k in range(aggr_cell.ues_per_cell):
                try:
                    aggr_last = aggr_cell.ipc_ues[k].get_last_ul_tx_ts()
                    if aggr_last <= 0:
                        continue
                    aggr_signal = aggr_cell.ipc_ues[k].circ_read(
                        aggr_cell.ipc_ues[k].gpu_ul_tx_ptr, ts, nsamps,
                        aggr_cell.ipc_ues[k].ul_tx_nbAnt,
                        aggr_cell.ipc_ues[k].ul_tx_cir_size, cp.int16)
                    ici_f64 = aggr_signal.astype(cp.float64) * self.ici_linear
                    ici_cpx = ici_f64.reshape(-1, 2)
                    ici_vals = ici_cpx[:, 0] + 1j * ici_cpx[:, 1]
                    n_ici = len(ici_vals)
                    victim_cell._ul_accum[:n_ici] += ici_vals
                except Exception:
                    pass

    def run(self):
        """Main loop: process all cells' DL and UL."""
        self._init_all()

        total_cpx = sum(SYMBOL_SIZES)
        N_cells = self.num_cells

        proxy_dl_heads = [0] * N_cells
        proxy_ul_heads = [0] * N_cells
        proxy_ul_heads_per_ue = [[0] * cell.ues_per_cell for cell in self.cells]
        ul_rx_ts_highs = [0] * N_cells

        if GPU_AVAILABLE:
            for cell in self.cells:
                _ul_rx_bytes = cell.ipc_gnb.ul_rx_cir_size * GPU_IPC_V6_SAMPLE_SIZE
                _mem = cp.cuda.UnownedMemory(
                    cell.ipc_gnb.gpu_ul_rx_ptr, _ul_rx_bytes, owner=None)
                _arr = cp.ndarray(
                    _ul_rx_bytes // 2, dtype=cp.int16,
                    memptr=cp.cuda.MemoryPointer(_mem, 0))
                _arr[:] = cp.random.randint(-1, 2, size=len(_arr), dtype=cp.int16)
                cp.cuda.Stream.null.synchronize()
                print(f"[MC] Cell {cell.cell_idx}: UL RX buffer pre-filled "
                      f"with noise (size={len(_arr)})")

            _noise_len = total_cpx * self.gnb_ant * 2
            self._ul_noise_slot = cp.random.randint(
                -1, 2, size=_noise_len, dtype=cp.int16)

        dilation = self.time_dilation
        slot_dur_s = total_cpx / 30720000.0
        dilation_sleep_s = slot_dur_s * (dilation - 1.0) if dilation > 1.0 else 0.0

        _ch_mode = "BYPASS" if not self._channel_active else "CHANNEL"
        if self.bypass_duration > 0 and self.custom_channel:
            _ch_mode = f"PHASED (bypass→channel after {self.bypass_duration:.0f}s)"
        print(f"[MC] Entering main loop ({N_cells} cells, "
              f"{self.ues_per_cell} UEs/cell, "
              f"gnb_ant={self.gnb_ant}, ue_ant={self.ue_ant}, "
              f"ICI={'ON '+str(self.ici_atten_dB)+'dB' if self.enable_ici else 'OFF'}, "
              f"mode={_ch_mode}, "
              f"time_dilation={dilation:.1f}x"
              f"{', slot_sleep='+str(round(dilation_sleep_s*1e6))+'us' if dilation_sleep_s > 0 else ''})...")

        t_start = time.time()
        loop_count = 0
        slot_count = 0
        slot_time_sum = 0.0
        slot_time_max = 0.0
        slot_time_min = float('inf')
        last_stats_time = t_start

        dl_init_logged = [False] * N_cells

        try:
            while True:
                processed = False
                t_iter_start = time.time()

                # ── DL: relay gNB DL TX → UE DL RX (per-UE channel) ──
                for ci, cell in enumerate(self.cells):
                    cur_dl_ts = cell.ipc_gnb.get_last_dl_tx_ts()
                    dl_nsamps = cell.ipc_gnb.get_last_dl_tx_nsamps()
                    if cur_dl_ts > 0 and dl_nsamps > 0:
                        gnb_dl_head = cur_dl_ts + dl_nsamps
                        if gnb_dl_head > proxy_dl_heads[ci]:
                            if proxy_dl_heads[ci] == 0:
                                proxy_dl_heads[ci] = gnb_dl_head
                                if not dl_init_logged[ci]:
                                    print(f"[MC] Cell {ci}: first DL at ts={gnb_dl_head}")
                                    dl_init_logged[ci] = True
                            delta = int(gnb_dl_head - proxy_dl_heads[ci])
                            max_dl_delta = total_cpx * 2
                            if delta > max_dl_delta:
                                delta = max_dl_delta
                            if delta > 0:
                                self._dl_broadcast_cell(
                                    cell, proxy_dl_heads[ci], delta)
                                proxy_dl_heads[ci] += delta
                                self._dl_count += 1
                                processed = True

                # ── UL: relay UE UL TX → gNB UL RX ──
                _ul_apply_ch = (self._channel_active
                                and any(c.custom_channel and len(c.channel_buffers) > 0
                                        for c in self.cells))

                for ci, cell in enumerate(self.cells):
                    if _ul_apply_ch:
                        for k in range(cell.ues_per_cell):
                            cur_ul = cell.ipc_ues[k].get_last_ul_tx_ts()
                            ul_ns = cell.ipc_ues[k].get_last_ul_tx_nsamps()
                            if cur_ul > 0 and ul_ns > 0:
                                ue_head = cur_ul + ul_ns
                                if ue_head > proxy_ul_heads[ci]:
                                    if proxy_ul_heads[ci] == 0:
                                        proxy_ul_heads[ci] = ue_head
                                    delta = int(ue_head - proxy_ul_heads[ci])
                                    max_ul_delta = total_cpx * 2
                                    if delta > max_ul_delta:
                                        delta = max_ul_delta
                                    delta = (delta // total_cpx) * total_cpx
                                    if delta >= total_cpx:
                                        self._ul_combine_cell(
                                            cell, proxy_ul_heads[ci], delta)
                                        proxy_ul_heads[ci] += delta
                                        self._ul_count += 1
                                        processed = True
                    else:
                        # UL bypass: relay at each UE's actual UL TX timestamps
                        # Process slot-by-slot, writing noise for empty slots
                        any_ul_written = False
                        for k in range(cell.ues_per_cell):
                            cur_ul = cell.ipc_ues[k].get_last_ul_tx_ts()
                            ul_ns = cell.ipc_ues[k].get_last_ul_tx_nsamps()
                            if cur_ul <= 0 or ul_ns <= 0:
                                continue
                            ue_head = int(cur_ul + ul_ns)
                            per_head = proxy_ul_heads_per_ue[ci][k]
                            if per_head == 0:
                                per_head = int(cur_ul)
                                proxy_ul_heads_per_ue[ci][k] = per_head
                            if ue_head <= per_head:
                                continue
                            delta = int(ue_head - per_head)
                            max_delta = total_cpx * 4
                            if delta > max_delta:
                                per_head = int(ue_head - max_delta)
                                delta = max_delta

                            _bp_pos = per_head
                            _bp_rem = delta
                            _bp_data_slots = 0
                            _bp_noise_slots = 0
                            while _bp_rem >= total_cpx:
                                _slot_arr = cell.ipc_ues[k].circ_read(
                                    cell.ipc_ues[k].gpu_ul_tx_ptr,
                                    _bp_pos, total_cpx,
                                    cell.ipc_ues[k].ul_tx_nbAnt,
                                    cell.ipc_ues[k].ul_tx_cir_size,
                                    cp.int16)
                                if int(cp.count_nonzero(_slot_arr)) > 0:
                                    cell.ipc_gnb.bypass_copy(
                                        cell.ipc_gnb.gpu_ul_rx_ptr,
                                        cell.ipc_ues[k].gpu_ul_tx_ptr,
                                        _bp_pos, total_cpx,
                                        cell.ipc_ues[k].ul_tx_nbAnt,
                                        cell.ipc_ues[k].ul_tx_cir_size,
                                        cell.ipc_gnb.ul_rx_nbAnt,
                                        cell.ipc_gnb.ul_rx_cir_size)
                                    _bp_data_slots += 1
                                else:
                                    _bp_noise_slots += 1
                                _bp_pos += total_cpx
                                _bp_rem -= total_cpx

                            new_head = int(per_head + delta)
                            proxy_ul_heads_per_ue[ci][k] = new_head

                            if not hasattr(self, '_ul_bp_diag_cnt'):
                                self._ul_bp_diag_cnt = 0
                            if self._ul_bp_diag_cnt < 20 or self._ul_bp_diag_cnt % 5000 == 0:
                                print(f"[UL BP] UE{k} head={per_head}->{new_head} "
                                      f"delta={delta} data_slots={_bp_data_slots} "
                                      f"noise_slots={_bp_noise_slots}")
                            self._ul_bp_diag_cnt += 1

                            any_ul_written = True

                        if any_ul_written:
                            new_max = max(proxy_ul_heads_per_ue[ci])
                            if new_max > proxy_ul_heads[ci]:
                                proxy_ul_heads[ci] = new_max
                            cell.ipc_gnb.set_last_ul_rx_ts(
                                int(proxy_ul_heads[ci] - 1))
                            self._ul_count += 1
                            processed = True

                # ── Timing stats + time dilation ──
                if processed:
                    t_iter_end = time.time()
                    dt = t_iter_end - t_iter_start
                    slot_count += 1
                    slot_time_sum += dt
                    if dt > slot_time_max:
                        slot_time_max = dt
                    if dt < slot_time_min:
                        slot_time_min = dt

                    if dilation_sleep_s > 0:
                        time.sleep(dilation_sleep_s)

                # ── Keepalive: advance UL RX ts to keep gNB running ──
                for ci, cell in enumerate(self.cells):
                    if proxy_dl_heads[ci] > 0:
                        if proxy_ul_heads[ci] > 0:
                            # UEs actively writing: relay handles last_ul_rx_ts.
                            # Only advance keepalive if relay pushed beyond high.
                            ka_target = proxy_ul_heads[ci]
                        else:
                            # No UE writing yet: DL-paced keepalive to prevent
                            # gNB deadlock during initial sync phase.
                            ka_target = proxy_dl_heads[ci]
                        if ka_target > ul_rx_ts_highs[ci]:
                            ul_rx_ts_highs[ci] = int(ka_target)
                            cell.ipc_gnb.set_last_ul_rx_ts(int(ka_target))

                # ── Phase switch: bypass → channel after bypass_duration ──
                if (self.bypass_duration > 0
                        and self.custom_channel
                        and not self._phase_switched):
                    _elapsed_phase = time.time() - t_start
                    if _elapsed_phase >= self.bypass_duration:
                        self._channel_active = True
                        self._phase_switched = True
                        print(f"[MC PHASE] ★ Switching to Sionna channel "
                              f"after {_elapsed_phase:.1f}s bypass")

                loop_count += 1
                now = time.time()
                if now - last_stats_time >= 5.0:
                    elapsed = now - t_start
                    if slot_count > 0:
                        avg_us = (slot_time_sum / slot_count) * 1e6
                        min_us = slot_time_min * 1e6
                        max_us = slot_time_max * 1e6
                        rate = slot_count / (now - last_stats_time)
                        _gap_info = []
                        for ci in range(N_cells):
                            _gdl = self.cells[ci].ipc_gnb.get_last_dl_tx_ts()
                            _pdl = proxy_dl_heads[ci]
                            _gap = max(0, _gdl - _pdl) // total_cpx if _pdl > 0 else -1
                            _gap_info.append(f"c{ci}:{_gap}slots")
                        _ul_mode = "CH" if _ul_apply_ch else "BYPASS"
                        print(f"[MC STATS] {elapsed:.0f}s | "
                              f"DL={self._dl_count} UL={self._ul_count}({_ul_mode}) | "
                              f"slot_time: avg={avg_us:.0f}us "
                              f"min={min_us:.0f}us max={max_us:.0f}us | "
                              f"rate={rate:.0f} slots/s | "
                              f"gap=[{','.join(_gap_info)}] | "
                              f"dilation={dilation:.1f}x")
                    else:
                        print(f"[MC STATS] {elapsed:.0f}s | "
                              f"DL={self._dl_count} UL={self._ul_count} | "
                              f"waiting for data...")
                    last_stats_time = now
                    slot_count = 0
                    slot_time_sum = 0.0
                    slot_time_max = 0.0
                    slot_time_min = float('inf')

                if not processed:
                    time.sleep(0.0001)

        except KeyboardInterrupt:
            elapsed = time.time() - t_start
            print(f"\n[MC] Terminated ({elapsed:.1f}s, "
                  f"DL={self._dl_count}, UL={self._ul_count})")
        finally:
            for cell in self.cells:
                cell.cleanup()
            print("[MC] All cells cleaned up")


def main():
    ap = argparse.ArgumentParser(
        description="v4_multicell: Multi-Cell MU-MIMO Channel Proxy")

    ap.add_argument("--mode", choices=["gpu-ipc"], default="gpu-ipc")
    ap.add_argument("--num-cells", type=int, required=True,
                    help="Number of cells")
    ap.add_argument("--ues-per-cell", type=int, required=True,
                    help="Number of UEs per cell")

    ap.add_argument("--gnb-ant", type=int, default=GNB_ANT)
    ap.add_argument("--ue-ant", type=int, default=UE_ANT)
    ap.add_argument("--gnb-nx", type=int, default=GNB_NX)
    ap.add_argument("--gnb-ny", type=int, default=GNB_NY)
    ap.add_argument("--ue-nx", type=int, default=UE_NX)
    ap.add_argument("--ue-ny", type=int, default=UE_NY)
    ap.add_argument("--polarization", choices=["single", "dual"],
                    default="single")

    ap.add_argument("--custom-channel", dest='custom_channel',
                    action="store_true")
    ap.add_argument("--no-custom-channel", dest='custom_channel',
                    action="store_false")
    ap.set_defaults(custom_channel=True)

    ap.add_argument("--ici-atten-dB", type=float, default=15.0,
                    help="Inter-cell interference attenuation in dB "
                         "(default: 15, set >100 to disable)")

    ap.add_argument("--enable-gpu", dest='enable_gpu', action="store_true")
    ap.add_argument("--disable-gpu", dest='enable_gpu', action="store_false")
    ap.set_defaults(enable_gpu=True)
    ap.add_argument("--use-pinned-memory", dest='use_pinned_memory',
                    action="store_true")
    ap.set_defaults(use_pinned_memory=True)
    ap.add_argument("--use-cuda-graph", dest='use_cuda_graph',
                    action="store_true")
    ap.set_defaults(use_cuda_graph=True)
    ap.add_argument("--profile-interval", type=int, default=100)
    ap.add_argument("--profile-window", type=int, default=500)
    ap.add_argument("--dual-timer-compare", dest='dual_timer_compare',
                    action="store_true")
    ap.set_defaults(dual_timer_compare=True)

    ap.add_argument("--scenario", type=str, default="UMa-NLOS",
                    choices=["UMi-LOS", "UMi-NLOS", "UMa-LOS", "UMa-NLOS"],
                    help="3GPP channel scenario (default: UMa-NLOS)")
    ap.add_argument("--sector-half-deg", type=float, default=90.0)
    ap.add_argument("--bs-height-m", type=float, default=25.0)
    ap.add_argument("--ue-height-m", type=float, default=1.5)
    ap.add_argument("--isd-m", type=float, default=500)
    ap.add_argument("--min-ue-dist-m", type=float, default=35)
    ap.add_argument("--max-ue-dist-m", type=float, default=500)
    ap.add_argument("--shadow-fading-std-dB", type=float, default=6.0)
    ap.add_argument("--k-factor-mean-dB", type=float, default=None)
    ap.add_argument("--k-factor-std-dB", type=float, default=None)
    ap.add_argument("--ue-speeds", type=str, default=None,
                    help="Per-UE speeds in km/h (comma-separated)")

    ap.add_argument("--path-loss-dB", type=float, default=0.0)
    ap.add_argument("--snr-dB", type=float, default=None)
    ap.add_argument("--noise-dBFS", type=float, default=None)

    ap.add_argument("--p1b-npz", type=str, default=None)
    ap.add_argument("--ue-rx-indices", type=str, default=None)

    ap.add_argument("--xla", dest='use_xla', action="store_true")
    ap.set_defaults(use_xla=False)

    ap.add_argument("--buffer-len", type=int, default=1024)

    ap.add_argument("--time-dilation", type=float, default=1.0,
                    help="Time dilation factor (default: 1.0 = real speed). "
                         "Set to 10 to run 10x slower for scaling debug.")

    ap.add_argument("--force-bypass", action="store_true",
                    help="Force DL+UL bypass (disable channel entirely)")
    ap.add_argument("--bypass-duration", type=float, default=0,
                    help="Seconds of DL+UL bypass before enabling channel "
                         "(0 = always channel, >0 = phased approach)")

    args = ap.parse_args()

    v4_mod.path_loss_dB = args.path_loss_dB
    v4_mod.pathLossLinear = 10 ** (args.path_loss_dB / 20.0)
    v4_mod.snr_dB = args.snr_dB
    v4_mod.noise_dBFS = args.noise_dBFS
    v4_mod.noise_mode = "none"
    v4_mod.noise_enabled = False
    v4_mod.noise_std_abs = None
    v4_mod.DL_USE_IDENTITY_CHANNEL = False
    v4_mod.PURE_DL_BYPASS = False
    v4_mod.UL_GAIN_LINEAR = 16.0

    if args.ue_speeds:
        v4_mod.ue_speeds = [float(s) / 3.6 for s in args.ue_speeds.split(",")]
    else:
        v4_mod.ue_speeds = None

    if args.snr_dB is not None:
        v4_mod.noise_mode = "relative"
        v4_mod.noise_enabled = True
    elif args.noise_dBFS is not None:
        v4_mod.noise_mode = "absolute"
        v4_mod.noise_enabled = True
        import math
        _rms = 32767.0 * (10.0 ** (args.noise_dBFS / 20.0))
        v4_mod.noise_std_abs = cp.float64(_rms / math.sqrt(2.0)) if GPU_AVAILABLE else None

    N = args.num_cells
    K = args.ues_per_cell
    total_ues = N * K

    if args.p1b_npz and not os.path.isabs(args.p1b_npz):
        args.p1b_npz = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), args.p1b_npz))
        print(f"[MC] P1B npz path resolved: {args.p1b_npz}")

    ue_rx_indices_all = None
    if args.p1b_npz:
        import random as _random
        import numpy as _np
        _data = _np.load(args.p1b_npz, allow_pickle=True)
        _all_rx = _data['rx_indices'].tolist()
        _data.close()

        if args.ue_rx_indices and args.ue_rx_indices != "random":
            flat = [int(x) for x in args.ue_rx_indices.split(",")]
            if len(flat) != total_ues:
                print(f"[ERROR] --ue-rx-indices count ({len(flat)}) != "
                      f"total UEs ({total_ues})")
                sys.exit(1)
            validate_rx_indices(args.p1b_npz, flat)
            ue_rx_indices_all = [flat[c * K:(c + 1) * K] for c in range(N)]
        else:
            if total_ues > len(_all_rx):
                print(f"[ERROR] Need {total_ues} unique RX but P1B has "
                      f"only {len(_all_rx)}")
                sys.exit(1)
            selected = _random.sample(_all_rx, total_ues)
            ue_rx_indices_all = [selected[c * K:(c + 1) * K] for c in range(N)]

        for c in range(N):
            parts = ", ".join(f"UE{k}=RX{rx}"
                              for k, rx in enumerate(ue_rx_indices_all[c]))
            print(f"[MC P1B] Cell {c}: {parts}")

    print("=" * 80)
    print(f"v4_multicell: Multi-Cell MU-MIMO Channel Proxy")
    print("=" * 80)
    print(f"  Cells: {N}")
    print(f"  UEs/cell: {K} (total: {total_ues})")
    print(f"  gNB antenna: {args.gnb_ant} ({args.gnb_ny}x{args.gnb_nx})")
    print(f"  UE antenna: {args.ue_ant} ({args.ue_ny}x{args.ue_nx})")
    print(f"  Polarization: {args.polarization}")
    print(f"  ICI: {args.ici_atten_dB} dB "
          f"({'enabled' if args.ici_atten_dB < 100 else 'disabled'})")
    print(f"  Channel: {'custom' if args.custom_channel else 'bypass'}")
    print(f"  Scenario: {args.scenario}")
    if args.time_dilation > 1.0:
        print(f"  Time dilation: {args.time_dilation:.1f}x "
              f"(slot sleep ≈{sum(SYMBOL_SIZES)/30720000.0*(args.time_dilation-1)*1e6:.0f}us)")
    else:
        print(f"  Time dilation: OFF (real speed)")
    if args.p1b_npz:
        print(f"  P1B Ray Data: {args.p1b_npz}")
    else:
        print(f"  Ray Data: legacy (all UEs share same rays)")
    for c in range(N):
        print(f"  Cell {c} gNB SHM: /tmp/oai_gpu_ipc/gpu_ipc_shm_cell{c}")
        for k in range(K):
            print(f"  Cell {c} UE[{k}] SHM: "
                  f"/tmp/oai_gpu_ipc/gpu_ipc_shm_cell{c}_ue{k}")
    print("=" * 80)

    effective_channel = args.custom_channel and not args.force_bypass

    if args.force_bypass:
        print(f"  Mode: FULL BYPASS (DL+UL, channel disabled)")
    elif args.bypass_duration > 0:
        print(f"  Mode: PHASED — bypass first {args.bypass_duration:.0f}s, "
              f"then Sionna channel")
    else:
        print(f"  Mode: Sionna channel from start")

    proxy = MultiCellProxy(
        num_cells=N,
        ues_per_cell=K,
        gnb_ant=args.gnb_ant,
        ue_ant=args.ue_ant,
        gnb_nx=args.gnb_nx,
        gnb_ny=args.gnb_ny,
        ue_nx=args.ue_nx,
        ue_ny=args.ue_ny,
        polarization=args.polarization,
        custom_channel=effective_channel,
        ici_atten_dB=args.ici_atten_dB,
        use_cuda_graph=args.use_cuda_graph,
        enable_gpu=args.enable_gpu,
        use_pinned_memory=args.use_pinned_memory,
        profile_interval=args.profile_interval,
        profile_window=args.profile_window,
        dual_timer_compare=args.dual_timer_compare,
        p1b_npz=args.p1b_npz,
        ue_rx_indices_all=ue_rx_indices_all,
        use_xla=args.use_xla,
        time_dilation=args.time_dilation,
        bypass_duration=args.bypass_duration,
        scenario=args.scenario,
        bs_height_m=args.bs_height_m,
        ue_height_m=args.ue_height_m,
        isd_m=args.isd_m,
        min_ue_dist_m=args.min_ue_dist_m,
        max_ue_dist_m=args.max_ue_dist_m,
        shadow_fading_std_dB=args.shadow_fading_std_dB,
        k_factor_mean_dB=args.k_factor_mean_dB,
        k_factor_std_dB=args.k_factor_std_dB,
        sector_half_deg=args.sector_half_deg,
        ue_speeds=v4_mod.ue_speeds)

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_handler)

    proxy.run()


if __name__ == "__main__":
    main()
