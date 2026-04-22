"""
Phase 4: CSI 주입 모듈
=======================
CsiNet에서 복원된 H_hat으로부터 PMI, CQI, RI를 계산하고
OAI gNB의 CSI report 파이프라인에 주입합니다.

통합 경로:
  H_hat -> SVD -> dominant eigenvectors -> Type 2 PMI 양자화
  H_hat -> SINR 추정 -> CQI 매핑
"""

import numpy as np


# 3GPP TS 38.214 Table 5.2.2.1-2: CQI to SINR mapping (approximate)
CQI_SINR_TABLE = {
    0: -999.0,
    1: -6.7, 2: -4.7, 3: -2.3, 4: 0.2,
    5: 2.4, 6: 4.7, 7: 6.9, 8: 8.9,
    9: 10.8, 10: 12.8, 11: 14.6, 12: 16.6,
    13: 18.6, 14: 20.4, 15: 22.5,
}


def compute_ri_from_H(H_hat, snr_dB=10.0):
    """Compute Rank Indicator from reconstructed channel.
    H_hat: (Nt, Nsc) complex -> RI in {1, 2}
    """
    Nt, Nsc = H_hat.shape

    # SVD per subcarrier, check condition number
    avg_sv_ratio = 0.0
    for sc in range(min(Nsc, 12)):  # sample subcarriers
        U, s, Vh = np.linalg.svd(H_hat[:, sc:sc+1].T, full_matrices=False)
        if len(s) > 1 and s[0] > 0:
            avg_sv_ratio += s[1] / s[0]
    avg_sv_ratio /= min(Nsc, 12)

    # Simple threshold: if second SV is >30% of first, RI=2
    ri = 2 if avg_sv_ratio > 0.3 and snr_dB > 8.0 else 1
    return ri


def compute_pmi_from_H(H_hat, Nt=4, n_beams=2, phase_bits=3):
    """Compute Type 2 PMI indices from reconstructed channel.
    Returns: dict with i1 (wideband), i2 (subband) indices.
    """
    Nsc = H_hat.shape[1]

    # DFT codebook
    codebook = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)

    # Wideband: find strongest beams
    beam_powers = np.abs(codebook.conj().T @ H_hat) ** 2  # (Nt, Nsc)
    beam_energy = beam_powers.sum(axis=1)
    top_beams = np.argsort(-beam_energy)[:n_beams].tolist()

    # Subband: compute phase coefficients
    selected = codebook[:, top_beams]  # (Nt, n_beams)
    coeffs = selected.conj().T @ H_hat  # (n_beams, Nsc)

    n_phase_levels = 2 ** phase_bits
    phases = np.angle(coeffs)
    phase_idx = np.round(phases / (2 * np.pi / n_phase_levels)).astype(int)
    phase_idx = phase_idx % n_phase_levels

    # Amplitude: quantize to 2 levels (0 or 1)
    amps = np.abs(coeffs)
    amp_idx = (amps > 0.5 * np.max(amps, axis=0, keepdims=True)).astype(int)

    return {
        "i1_beam_indices": top_beams,
        "i2_phase": phase_idx.tolist(),
        "i2_amplitude": amp_idx.tolist(),
        "n_beams": n_beams,
        "phase_bits": phase_bits,
    }


def compute_cqi_from_H(H_hat, noise_power_dBm=-100.0, tx_power_dBm=23.0):
    """Compute CQI from reconstructed channel."""
    # Average channel gain
    avg_gain = np.mean(np.abs(H_hat) ** 2)
    rx_power_dBm = tx_power_dBm + 10 * np.log10(max(avg_gain, 1e-15))
    sinr_dB = rx_power_dBm - noise_power_dBm

    # Map SINR to CQI
    best_cqi = 0
    for cqi, threshold in CQI_SINR_TABLE.items():
        if sinr_dB >= threshold:
            best_cqi = cqi
    return min(best_cqi, 15)


def compute_precoding_weights(H_hat, ri=1):
    """Compute ZF precoding weights from H_hat.
    H_hat: (Nt, Nsc) complex
    Returns: W (Nt, ri, Nsc) complex precoding weights
    """
    Nt, Nsc = H_hat.shape

    W = np.zeros((Nt, ri, Nsc), dtype=np.complex64)
    for sc in range(Nsc):
        h = H_hat[:, sc:sc+1]  # (Nt, 1)
        if ri == 1:
            w = h / max(np.linalg.norm(h), 1e-10)
            W[:, 0, sc] = w.flatten()
        else:
            U, s, Vh = np.linalg.svd(h.T, full_matrices=False)
            W[:, :ri, sc] = Vh[:ri].conj().T

    return W


class CSIInjector:
    """Manages CSI injection into OAI pipeline."""

    def __init__(self):
        self._csi_reports = {}  # (cell_idx, ue_idx) -> latest CSI report

    def process_channel(self, cell_idx, ue_idx, H_hat,
                        noise_power_dBm=-100.0, tx_power_dBm=23.0):
        """Process reconstructed channel and prepare CSI report."""
        ri = compute_ri_from_H(H_hat)
        pmi = compute_pmi_from_H(H_hat)
        cqi = compute_cqi_from_H(H_hat, noise_power_dBm, tx_power_dBm)
        W = compute_precoding_weights(H_hat, ri)

        report = {
            "ri": ri,
            "pmi": pmi,
            "cqi": cqi,
            "precoding_weights": W,
            "H_hat": H_hat,
        }
        self._csi_reports[(cell_idx, ue_idx)] = report
        return report

    def get_report(self, cell_idx, ue_idx):
        return self._csi_reports.get((cell_idx, ue_idx))
