"""
Phase 4: Channel H 가로채기 훅
===============================
v4_multicell.py의 _dl_apply_channel_slot()에서 채널 행렬 H를 가로채
CsiNet sidecar로 전달하는 모듈.

이 모듈은 v4_multicell.py에 import되어, 채널 행렬이 IQ에 적용되기 직전에
H를 캡처하고 CsiNet 추론 결과로 PMI/CQI를 계산합니다.
"""

import os
import numpy as np

try:
    import cupy as cp
    GPU_OK = True
except ImportError:
    cp = np
    GPU_OK = False


class ChannelHook:
    """Intercepts channel matrices from the Sionna proxy pipeline.

    Usage in v4_multicell.py:
        hook = ChannelHook(enabled=True)
        # inside _dl_apply_channel_slot, after get_batch_view:
        hook.capture(cell_idx, ue_idx, channels)  # channels: (N_SYM, n_rx, n_tx, fft)
    """

    def __init__(self, enabled=False, buffer_size=100, csi_rs_period=20):
        self.enabled = enabled
        self.buffer_size = buffer_size
        self.csi_rs_period = csi_rs_period
        self._slot_counter = {}
        self._buffers = {}
        self._callbacks = []

    def register_callback(self, fn):
        """Register a function to be called when H is captured.
        fn(cell_idx, ue_idx, H_freq) where H_freq: (n_rx, n_tx, n_sc) complex
        """
        self._callbacks.append(fn)

    def capture(self, cell_idx, ue_idx, channels):
        """Capture channel matrix from proxy slot processing.
        channels: cupy array (N_SYM, n_rx, n_tx, FFT_SIZE) complex
        """
        if not self.enabled:
            return

        key = (cell_idx, ue_idx)
        self._slot_counter[key] = self._slot_counter.get(key, 0) + 1

        # Only capture at CSI-RS periodicity
        if self._slot_counter[key] % self.csi_rs_period != 0:
            return

        # Extract first OFDM symbol's channel (representative)
        if GPU_OK and hasattr(channels, 'get'):
            H = channels[0].get()  # (n_rx, n_tx, FFT_SIZE) -> CPU numpy
        else:
            H = np.asarray(channels[0])

        # Store in ring buffer
        if key not in self._buffers:
            self._buffers[key] = []
        buf = self._buffers[key]
        buf.append(H)
        if len(buf) > self.buffer_size:
            buf.pop(0)

        # Notify callbacks
        for fn in self._callbacks:
            try:
                fn(cell_idx, ue_idx, H)
            except Exception as e:
                print(f"[ChannelHook] callback error: {e}")

    def get_latest(self, cell_idx, ue_idx):
        """Get the most recent captured channel matrix."""
        key = (cell_idx, ue_idx)
        buf = self._buffers.get(key, [])
        return buf[-1] if buf else None

    def get_statistics(self, cell_idx, ue_idx, n_samples=None,
                       include_change_metrics=False):
        """Compute covariance and PDP from buffered channels.

        Args:
            cell_idx, ue_idx: cell/UE identifiers.
            n_samples: number of recent samples to use.
            include_change_metrics: if True, also return a dict with
                delta metrics comparing current vs previous statistics.

        Returns:
            (R_H, pdp) if include_change_metrics is False.
            (R_H, pdp, change_metrics) if True, where change_metrics dict
            has keys: r_h_delta_norm, pdp_delta_norm, r_h_relative_change,
            pdp_relative_change.
        """
        key = (cell_idx, ue_idx)
        buf = self._buffers.get(key, [])
        if not buf:
            if include_change_metrics:
                return None, None, None
            return None, None

        if n_samples:
            buf = buf[-n_samples:]

        H_stack = np.stack(buf)  # (N, n_rx, n_tx, FFT_SIZE)
        N, Nr, Nt, Nsc = H_stack.shape

        # Covariance
        h_vec = H_stack.reshape(N, Nr * Nt, Nsc)
        R_H = np.zeros((Nr * Nt, Nr * Nt), dtype=np.complex64)
        for sc in range(Nsc):
            h_sc = h_vec[:, :, sc]
            R_H += (h_sc.conj().T @ h_sc) / N
        R_H /= Nsc

        # PDP
        h_delay = np.fft.ifft(H_stack, axis=-1)
        pdp = np.mean(np.abs(h_delay) ** 2, axis=(0, 1, 2))

        if not include_change_metrics:
            return R_H, pdp

        prev_key = f"_prev_stat_{key}"
        prev = getattr(self, prev_key, None)
        if prev is None:
            change_metrics = {
                "r_h_delta_norm": 0.0, "pdp_delta_norm": 0.0,
                "r_h_relative_change": 0.0, "pdp_relative_change": 0.0,
            }
        else:
            prev_R_H, prev_pdp = prev
            r_delta = np.linalg.norm(R_H - prev_R_H)
            r_prev_norm = max(np.linalg.norm(prev_R_H), 1e-12)
            p_delta = np.linalg.norm(pdp - prev_pdp)
            p_prev_norm = max(np.linalg.norm(prev_pdp), 1e-12)
            change_metrics = {
                "r_h_delta_norm": float(r_delta),
                "pdp_delta_norm": float(p_delta),
                "r_h_relative_change": float(r_delta / r_prev_norm),
                "pdp_relative_change": float(p_delta / p_prev_norm),
            }
        setattr(self, prev_key, (R_H.copy(), pdp.copy()))

        return R_H, pdp, change_metrics
