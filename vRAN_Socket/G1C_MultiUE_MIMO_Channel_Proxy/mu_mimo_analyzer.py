"""
MU-MIMO Precoding Mismatch Analyzer.

Observer-mode module that hooks into the Sionna channel proxy to:
1. Capture true channel matrices H_k per UE from the proxy.
2. Compute ideal precoding (ZF / MMSE) from H.
3. Reconstruct PMI-based precoding from the codebook.
4. Compute per-UE SINR under both schemes.
5. Log all metrics to CSV for post-processing.

Usage:
    analyzer = MuMimoAnalyzer(num_ues=4, gnb_ant=2, ue_ant=2,
                              fft_size=2048, log_dir="./logs/run1")
    # Called by proxy every DL slot:
    analyzer.on_dl_slot(slot_idx, frame_idx, channels_per_ue,
                        noise_power, path_loss_linear)
"""

import csv
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from codebook_reconstruct import get_type1_precoder


class MuMimoAnalyzer:
    """Computes and logs MU-MIMO precoding mismatch metrics."""

    ANALYSIS_SUBCARRIERS = 64  # analyze a subset of SCs to limit overhead

    def __init__(self,
                 num_ues: int,
                 gnb_ant: int,
                 ue_ant: int,
                 fft_size: int,
                 log_dir: str,
                 sample_interval: int = 20,
                 noise_power_default: float = 1e-10):
        """
        Args:
            num_ues: Total number of UEs in the simulation.
            gnb_ant: Number of gNB TX antennas.
            ue_ant: Number of UE RX antennas.
            fft_size: OFDM FFT size.
            log_dir: Directory for output CSV files.
            sample_interval: Analyze every N-th slot to reduce overhead.
            noise_power_default: Default noise power (linear) if not provided.
        """
        self.num_ues = num_ues
        self.gnb_ant = gnb_ant
        self.ue_ant = ue_ant
        self.fft_size = fft_size
        self.log_dir = log_dir
        self.sample_interval = sample_interval
        self.noise_power = noise_power_default
        self.slot_count = 0
        self._lock = threading.Lock()

        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "mu_mimo_analysis.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "frame", "slot", "ue_i", "ue_j",
            "sc_idx_sample",
            "sinr_zf_i_dB", "sinr_zf_j_dB",
            "sinr_mmse_i_dB", "sinr_mmse_j_dB",
            "sinr_pmi1_i_dB", "sinr_pmi1_j_dB",
            "sinr_pmi2_i_dB", "sinr_pmi2_j_dB",
            "sinr_pmi3_i_dB", "sinr_pmi3_j_dB",
            "sinr_pmi4_i_dB", "sinr_pmi4_j_dB",
            "chordal_dist_zf_best_pmi",
            "chordal_dist_mmse_best_pmi",
            "best_pmi_i", "best_pmi_j",
            "channel_corr",
        ])
        self._csv_file.flush()
        self._lines = 0
        print(f"[MuMimoAnalyzer] Initialized. log_dir={log_dir}, "
              f"sample_interval={sample_interval}")

    def update_noise_power(self, noise_power: float):
        """Update noise power (called when hotswap changes noise)."""
        self.noise_power = noise_power

    def on_dl_slot(self,
                   slot_idx: int,
                   frame_idx: int,
                   channels_per_ue: Dict[int, np.ndarray],
                   noise_power: Optional[float] = None,
                   path_loss_linear: float = 1.0):
        """Called by the proxy on each DL slot.

        Args:
            slot_idx: Current slot index.
            frame_idx: Current frame index.
            channels_per_ue: {ue_id: H_k} where H_k has shape
                (N_SYM, ue_ant, gnb_ant, FFT_SIZE) complex.
            noise_power: Noise power (linear). If None, use stored value.
            path_loss_linear: Path loss scaling applied by proxy.
        """
        self.slot_count += 1
        if self.slot_count % self.sample_interval != 0:
            return

        if noise_power is not None:
            self.noise_power = noise_power

        sigma2 = max(self.noise_power, 1e-20)

        try:
            self._analyze_all_pairs(frame_idx, slot_idx,
                                    channels_per_ue, sigma2, path_loss_linear)
        except Exception as e:
            print(f"[MuMimoAnalyzer] Error in analysis: {e}")

    def _analyze_all_pairs(self, frame: int, slot: int,
                           channels: Dict[int, np.ndarray],
                           sigma2: float, pl: float):
        """Analyze all possible MU-MIMO UE pairs."""
        ue_ids = sorted(channels.keys())
        if len(ue_ids) < 2:
            return

        # Pick representative symbol (middle of slot)
        sym_idx = 7

        sc_indices = np.linspace(0, self.fft_size - 1,
                                 self.ANALYSIS_SUBCARRIERS,
                                 dtype=int)

        for i_idx in range(len(ue_ids)):
            for j_idx in range(i_idx + 1, len(ue_ids)):
                ue_i = ue_ids[i_idx]
                ue_j = ue_ids[j_idx]

                H_i_full = channels[ue_i]  # (N_SYM, ue_ant, gnb_ant, FFT)
                H_j_full = channels[ue_j]

                if H_i_full.ndim < 4 or H_j_full.ndim < 4:
                    continue

                H_i_sym = H_i_full[sym_idx]  # (ue_ant, gnb_ant, FFT)
                H_j_sym = H_j_full[sym_idx]

                self._analyze_pair(frame, slot, ue_i, ue_j,
                                   H_i_sym, H_j_sym, sc_indices,
                                   sigma2, pl)

    def _analyze_pair(self, frame: int, slot: int,
                      ue_i: int, ue_j: int,
                      H_i: np.ndarray, H_j: np.ndarray,
                      sc_indices: np.ndarray,
                      sigma2: float, pl: float):
        """Analyze one UE pair across sampled subcarriers."""
        n_sc = len(sc_indices)

        sinr_zf = np.zeros((n_sc, 2))
        sinr_mmse = np.zeros((n_sc, 2))
        sinr_pmi = np.zeros((n_sc, 4, 2))  # 4 PMI combos
        chordal_zf = np.zeros(n_sc)
        chordal_mmse = np.zeros(n_sc)
        best_pmi = np.zeros((n_sc, 2), dtype=int)
        corr = np.zeros(n_sc)

        pmi_combos = [(1, 3), (2, 4), (3, 1), (4, 2)]

        for idx, sc in enumerate(sc_indices):
            Hi = H_i[:, :, sc]  # (ue_ant, gnb_ant)
            Hj = H_j[:, :, sc]

            # Effective channel (rank-1: dominant right singular vector)
            h_eff_i = self._effective_channel(Hi)  # (gnb_ant,)
            h_eff_j = self._effective_channel(Hj)

            H_eff = np.vstack([h_eff_i[np.newaxis, :],
                               h_eff_j[np.newaxis, :]])  # (2, gnb_ant)

            corr[idx] = self._channel_correlation(h_eff_i, h_eff_j)

            # ZF precoding
            W_zf = self._compute_zf(H_eff)
            if W_zf is not None:
                sinr_zf[idx] = self._compute_sinr_pair(
                    Hi, Hj, W_zf[:, 0], W_zf[:, 1], sigma2)
            else:
                sinr_zf[idx] = [-30, -30]

            # MMSE precoding
            W_mmse = self._compute_mmse(H_eff, sigma2)
            if W_mmse is not None:
                sinr_mmse[idx] = self._compute_sinr_pair(
                    Hi, Hj, W_mmse[:, 0], W_mmse[:, 1], sigma2)
            else:
                sinr_mmse[idx] = [-30, -30]

            # PMI-based precoding (all orthogonal pairs)
            for p_idx, (pm_i, pm_j) in enumerate(pmi_combos):
                w_pmi_i = get_type1_precoder(pm_i, n_layers=1)
                w_pmi_j = get_type1_precoder(pm_j, n_layers=1)
                sinr_pmi[idx, p_idx] = self._compute_sinr_pair(
                    Hi, Hj, w_pmi_i, w_pmi_j, sigma2)

            # Best PMI combo (max sum SINR)
            sum_sinr_pmi = sinr_pmi[idx].sum(axis=1)
            best_combo_idx = np.argmax(sum_sinr_pmi)
            best_pmi[idx] = pmi_combos[best_combo_idx]

            # Chordal distance: ZF vs best PMI
            if W_zf is not None:
                bp = pmi_combos[best_combo_idx]
                w_bp_i = get_type1_precoder(bp[0])
                w_bp_j = get_type1_precoder(bp[1])
                W_best_pmi = np.column_stack([w_bp_i, w_bp_j])
                chordal_zf[idx] = self._chordal_distance(W_zf, W_best_pmi)
            if W_mmse is not None:
                bp = pmi_combos[best_combo_idx]
                w_bp_i = get_type1_precoder(bp[0])
                w_bp_j = get_type1_precoder(bp[1])
                W_best_pmi = np.column_stack([w_bp_i, w_bp_j])
                chordal_mmse[idx] = self._chordal_distance(W_mmse, W_best_pmi)

        # Write averaged results
        avg_sinr_zf = self._to_dB(sinr_zf.mean(axis=0))
        avg_sinr_mmse = self._to_dB(sinr_mmse.mean(axis=0))
        avg_sinr_pmi = [self._to_dB(sinr_pmi[:, p].mean(axis=0))
                        for p in range(4)]
        avg_chordal_zf = chordal_zf.mean()
        avg_chordal_mmse = chordal_mmse.mean()
        avg_corr = corr.mean()

        # Most common best PMI
        from collections import Counter
        pmi_counts = Counter(map(tuple, best_pmi.tolist()))
        most_common_pmi = pmi_counts.most_common(1)[0][0]

        with self._lock:
            self._writer.writerow([
                frame, slot, ue_i, ue_j,
                self.ANALYSIS_SUBCARRIERS,
                f"{avg_sinr_zf[0]:.2f}", f"{avg_sinr_zf[1]:.2f}",
                f"{avg_sinr_mmse[0]:.2f}", f"{avg_sinr_mmse[1]:.2f}",
                f"{avg_sinr_pmi[0][0]:.2f}", f"{avg_sinr_pmi[0][1]:.2f}",
                f"{avg_sinr_pmi[1][0]:.2f}", f"{avg_sinr_pmi[1][1]:.2f}",
                f"{avg_sinr_pmi[2][0]:.2f}", f"{avg_sinr_pmi[2][1]:.2f}",
                f"{avg_sinr_pmi[3][0]:.2f}", f"{avg_sinr_pmi[3][1]:.2f}",
                f"{avg_chordal_zf:.4f}",
                f"{avg_chordal_mmse:.4f}",
                most_common_pmi[0], most_common_pmi[1],
                f"{avg_corr:.4f}",
            ])
            self._lines += 1
            if self._lines % 20 == 0:
                self._csv_file.flush()

    # ─── Linear algebra helpers ───

    @staticmethod
    def _effective_channel(H_k: np.ndarray) -> np.ndarray:
        """Extract dominant right singular vector (rank-1 effective channel).

        Args:
            H_k: (n_rx, n_tx) channel matrix for one UE at one subcarrier.

        Returns:
            h_eff: (n_tx,) effective channel direction.
        """
        _, s, Vh = np.linalg.svd(H_k, full_matrices=False)
        return Vh[0, :]  # row 0 of Vh = dominant right singular vector

    @staticmethod
    def _compute_zf(H_eff: np.ndarray) -> Optional[np.ndarray]:
        """Compute ZF precoding: W = H^H (H H^H)^{-1}, power normalized.

        Args:
            H_eff: (K, n_tx) stacked effective channel vectors.

        Returns:
            W: (n_tx, K) normalized precoding matrix, or None if singular.
        """
        try:
            gram = H_eff @ H_eff.conj().T
            if np.linalg.cond(gram) > 1e10:
                return None
            W = H_eff.conj().T @ np.linalg.inv(gram)
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return W / norms
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _compute_mmse(H_eff: np.ndarray, sigma2: float) -> Optional[np.ndarray]:
        """Compute MMSE (regularized ZF) precoding.

        W = H^H (H H^H + sigma^2 I)^{-1}, power normalized.
        """
        try:
            K = H_eff.shape[0]
            gram = H_eff @ H_eff.conj().T + sigma2 * np.eye(K)
            W = H_eff.conj().T @ np.linalg.inv(gram)
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return W / norms
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _compute_sinr_pair(H_i: np.ndarray, H_j: np.ndarray,
                           w_i: np.ndarray, w_j: np.ndarray,
                           sigma2: float) -> np.ndarray:
        """Compute SINR for a 2-UE MU-MIMO pair with matched-filter combining.

        Args:
            H_i: (n_rx, n_tx) channel for UE i.
            H_j: (n_rx, n_tx) channel for UE j.
            w_i: (n_tx,) precoding vector for UE i.
            w_j: (n_tx,) precoding vector for UE j.
            sigma2: Noise power.

        Returns:
            [sinr_i, sinr_j] in linear scale.
        """
        # UE i: desired signal = H_i @ w_i, interference = H_i @ w_j
        sig_i = H_i @ w_i
        intf_i = H_i @ w_j
        # Matched filter combining: u_i = sig_i / ||sig_i||
        sig_pow_i = np.real(np.vdot(sig_i, sig_i))
        intf_pow_i = np.real(np.vdot(intf_i, intf_i))
        sinr_i = sig_pow_i / max(intf_pow_i + sigma2, 1e-20)

        sig_j = H_j @ w_j
        intf_j = H_j @ w_i
        sig_pow_j = np.real(np.vdot(sig_j, sig_j))
        intf_pow_j = np.real(np.vdot(intf_j, intf_j))
        sinr_j = sig_pow_j / max(intf_pow_j + sigma2, 1e-20)

        return np.array([sinr_i, sinr_j])

    @staticmethod
    def _channel_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
        """Normalized channel correlation |h1^H h2|^2 / (||h1||^2 ||h2||^2)."""
        dot = np.abs(np.vdot(h1, h2)) ** 2
        n1 = np.real(np.vdot(h1, h1))
        n2 = np.real(np.vdot(h2, h2))
        denom = max(n1 * n2, 1e-20)
        return dot / denom

    @staticmethod
    def _chordal_distance(W1: np.ndarray, W2: np.ndarray) -> float:
        """Chordal distance between two precoding matrices.

        d_c(W1, W2) = (1/sqrt(2)) * ||W1 W1^H - W2 W2^H||_F
        """
        P1 = W1 @ W1.conj().T
        P2 = W2 @ W2.conj().T
        return np.linalg.norm(P1 - P2, "fro") / np.sqrt(2)

    @staticmethod
    def _to_dB(sinr_linear: np.ndarray) -> np.ndarray:
        """Convert linear SINR to dB, clamped to [-30, 60]."""
        sinr_clamped = np.maximum(sinr_linear, 1e-3)
        return np.clip(10 * np.log10(sinr_clamped), -30, 60)

    def close(self):
        """Flush and close the CSV file."""
        with self._lock:
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
                print(f"[MuMimoAnalyzer] Closed. {self._lines} analysis records "
                      f"written to {self._csv_path}")
