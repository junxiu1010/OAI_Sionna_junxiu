"""
3GPP NR Codebook Reconstruction for precoding analysis.

Reconstructs precoding vectors from PMI indices to enable comparison
with ideal (ZF/MMSE) precoding computed from true channel matrices.
"""

import numpy as np

# ─── Type-I Single Panel, 2 ports, 1 layer (38.214 Table 5.2.2.2.1-1) ───
# OAI pm_index 1..4 maps to pmi_x2 = 0..3
TYPE1_2PORT_1LAYER = {
    1: np.array([1,  1],  dtype=np.complex128) / np.sqrt(2),
    2: np.array([1,  1j], dtype=np.complex128) / np.sqrt(2),
    3: np.array([1, -1],  dtype=np.complex128) / np.sqrt(2),
    4: np.array([1, -1j], dtype=np.complex128) / np.sqrt(2),
}

TYPE1_2PORT_2LAYER = {
    1: np.array([[1,  1], [1, -1]],  dtype=np.complex128) / 2.0,
    2: np.array([[1,  1], [1j, -1j]], dtype=np.complex128) / 2.0,
}

# Amplitude dequantization table for Type-II (same as OAI C code)
_TYPE2_WB_AMP = np.array([1.0, 1/np.sqrt(2), 0.5, 1/(2*np.sqrt(2)),
                          0.25, 1/(4*np.sqrt(2)), 0.125, 0.0])
_TYPE2_QPSK_PHASE = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
_TYPE2_8PSK_PHASE = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4,
                              np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])


def get_type1_precoder(pm_index: int, n_layers: int = 1) -> np.ndarray:
    """Return Type-I precoding vector for 2-port codebook.

    Args:
        pm_index: OAI pm_index (1-based). 0 = identity (no precoding).
        n_layers: Number of layers (1 or 2).

    Returns:
        w: (n_tx,) for 1 layer, (n_tx, n_layers) for 2 layers.
           Returns identity-like vector for pm_index=0.
    """
    if pm_index == 0:
        if n_layers == 1:
            return np.array([1, 0], dtype=np.complex128)
        return np.eye(2, dtype=np.complex128)

    if n_layers == 1:
        return TYPE1_2PORT_1LAYER.get(pm_index,
                                       np.array([1, 0], dtype=np.complex128))
    return TYPE1_2PORT_2LAYER.get(pm_index,
                                   np.eye(2, dtype=np.complex128))


def get_type2_precoder(pmi_x1: int, pmi_x2: int,
                       port_sel_indicator: int,
                       wideband_amplitude: np.ndarray,
                       subband_phase: np.ndarray,
                       num_ant_ports: int,
                       num_layers: int,
                       d: int = 1,
                       n_phase: int = 4) -> np.ndarray:
    """Reconstruct Type-II port-selection precoding vector.

    Mirrors OAI's reconstruct_type2_precoding_matrix() logic.

    Args:
        pmi_x1, pmi_x2: PMI indices (informational, not directly used).
        port_sel_indicator: Port selection indicator from CSI report.
        wideband_amplitude: shape (total_coeffs, n_layers), uint8 amp indices.
        subband_phase: shape (n_subbands, total_coeffs, n_layers), uint8 phase indices.
        num_ant_ports: Number of antenna ports.
        num_layers: Number of MIMO layers.
        d: Port selection parameter.
        n_phase: Phase alphabet size (4 = QPSK, 8 = 8PSK).

    Returns:
        W: (num_ant_ports, num_layers) complex precoding matrix.
    """
    L_beams = wideband_amplitude.shape[0] // 2 if wideband_amplitude.ndim > 0 else 1
    total_coeffs = min(2 * L_beams, 2 * d)
    sel_start = port_sel_indicator * d
    phase_table = _TYPE2_8PSK_PHASE if n_phase == 8 else _TYPE2_QPSK_PHASE

    W = np.zeros((num_ant_ports, num_layers), dtype=np.complex128)
    for lay in range(num_layers):
        for c in range(total_coeffs):
            port_local = c % d
            pol = c // d
            ant_idx = sel_start + port_local + pol * (num_ant_ports // 2)
            if ant_idx >= num_ant_ports:
                continue

            amp_idx = min(int(wideband_amplitude[c, lay]), 7)
            amp = _TYPE2_WB_AMP[amp_idx]

            phase = 0.0
            if subband_phase is not None and subband_phase.shape[0] > 0:
                phase_idx = int(subband_phase[0, c, lay])
                phase = phase_table[phase_idx % len(phase_table)]

            w_val = amp * np.exp(1j * phase) / np.sqrt(num_layers)
            W[ant_idx, lay] += w_val

    return W


def reconstruct_precoder_from_csv(pm_index: int, is_type2: bool,
                                  pmi_x1: int = 0, pmi_x2: int = 0,
                                  n_layers: int = 1,
                                  n_tx: int = 2) -> np.ndarray:
    """Reconstruct precoding vector from CSV sideband log fields.

    For post-processing: uses only the fields available in the sideband CSV.
    Type-II reconstruction from CSV is approximate (WB amplitude/phase not in CSV).

    Args:
        pm_index: OAI pm_index from sideband log.
        is_type2: Whether Type-II codebook was used.
        pmi_x1, pmi_x2: PMI indices.
        n_layers: Number of layers.
        n_tx: Number of TX antennas.

    Returns:
        w: (n_tx,) for 1 layer, (n_tx, n_layers) for multi-layer.
    """
    if is_type2:
        # Approximate: use DFT-like vector from pmi_x2 as coarse estimate
        if n_tx == 2 and n_layers == 1:
            phases = _TYPE2_QPSK_PHASE
            phase = phases[pmi_x2 % len(phases)]
            return np.array([1, np.exp(1j * phase)], dtype=np.complex128) / np.sqrt(2)
        return np.ones(n_tx, dtype=np.complex128) / np.sqrt(n_tx)

    return get_type1_precoder(pm_index, n_layers)
