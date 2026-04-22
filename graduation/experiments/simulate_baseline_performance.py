#!/usr/bin/env python3
"""
Baseline 성능 시뮬레이션 — 시스템 모델 기반
==============================================
3GPP TR 38.901 채널 모델 통계 + 5G NR MCS 테이블 기반으로
Type 1 SU / Type 2 SU / Type 2 MU-MIMO 의 기대 성능을 산출합니다.

- 채널: UMi-LOS, UMi-NLOS, UMa-NLOS
- 모드: Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO
- 횡좌표: UE 수 (1, 2, 4, 8, 16)
- 지표: BLER, Average MCS, Cell DL Throughput

모델링 근거:
  1. 채널 SNR: 3GPP TR 38.901 median SNR 기반
  2. CSI 양자화 오차: Type 1 (단일 빔) vs Type 2 (L=2빔, 8PSK)
  3. MU-MIMO 잔여 간섭: CSI 양자화 오차에 의한 inter-UE interference
  4. BLER: SINR → MCS → BLER 매핑 (3GPP MCS Table 1)
  5. Throughput: TBS × (1-BLER) × scheduling_efficiency
"""

import numpy as np
from pathlib import Path
import json

np.random.seed(42)

BW_MHZ = 40
SCS_KHZ = 30
N_PRB = 106
N_SC_PER_PRB = 12
SLOT_DURATION_MS = 0.5
TDD_DL_RATIO = 7 / 10  # 7DL + 2UL + 1 flex
OVERHEAD_RATIO = 0.85   # DMRS, CSI-RS, PDCCH overhead
N_LAYERS_SU_MAX = 4     # SU-MIMO max layers
N_LAYERS_MU = 1         # MU-MIMO: rank-1 per UE
N_TX = 4                # 4T4R
N_RX = 2                # UE 2Rx

# 3GPP MCS Table 1 (QPSK/16QAM/64QAM) — approximate spectral efficiency
MCS_TABLE = {
    0: 0.2344, 1: 0.3066, 2: 0.3770, 3: 0.4902, 4: 0.6016,
    5: 0.7402, 6: 0.8770, 7: 1.0273, 8: 1.1758, 9: 1.3262,
    10: 1.3281, 11: 1.4766, 12: 1.6953, 13: 1.9141, 14: 2.1602,
    15: 2.4063, 16: 2.5703, 17: 2.7305, 18: 2.8438, 19: 3.0293,
    20: 3.3223, 21: 3.6094, 22: 3.9023, 23: 4.2129, 24: 4.5234,
    25: 4.8164, 26: 5.1152, 27: 5.5547,
}

# SINR → MCS 매핑 (approximate, dB)
SINR_TO_MCS_THRESHOLDS = np.array([
    -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6, 7.5,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27
])

def sinr_to_mcs(sinr_dB):
    idx = np.searchsorted(SINR_TO_MCS_THRESHOLDS, sinr_dB, side='right') - 1
    return int(np.clip(idx, 0, 27))

def mcs_to_se(mcs):
    return MCS_TABLE.get(min(mcs, 27), 0.2344)

def sinr_to_bler(sinr_dB, mcs):
    """BLER as function of SINR margin above MCS threshold."""
    if mcs < 0 or mcs > 27:
        return 1.0
    threshold = SINR_TO_MCS_THRESHOLDS[min(mcs, 27)]
    margin = sinr_dB - threshold
    bler = 1.0 / (1.0 + np.exp(2.0 * margin))
    return float(np.clip(bler, 0.001, 0.5))

def link_adapted_mcs(sinr_dB, bler_target=0.10):
    """Find MCS that achieves BLER closest to target (link adaptation)."""
    best_mcs = 0
    for mcs in range(28):
        bler = sinr_to_bler(sinr_dB, mcs)
        if bler <= bler_target:
            best_mcs = mcs
        else:
            break
    return best_mcs


# ── 채널 모델 파라미터 (3GPP TR 38.901) ──────────────────────────

CHANNEL_PARAMS = {
    "UMi-LOS": {
        "median_snr_dB": 22.0,
        "snr_std_dB": 4.0,
        "rician_K_dB": 9.0,
        "angular_spread_deg": 15.0,
        "delay_spread_ns": 30.0,
    },
    "UMi-NLOS": {
        "median_snr_dB": 15.0,
        "snr_std_dB": 6.0,
        "rician_K_dB": 0.0,
        "angular_spread_deg": 40.0,
        "delay_spread_ns": 80.0,
    },
    "UMa-NLOS": {
        "median_snr_dB": 10.0,
        "snr_std_dB": 7.0,
        "rician_K_dB": 0.0,
        "angular_spread_deg": 55.0,
        "delay_spread_ns": 120.0,
    },
}


# ── CSI 양자화 오차 모델 ──────────────────────────────────────────

def csi_quantization_loss_dB(codebook_type, channel_params):
    """
    CSI 피드백 양자화로 인한 SINR 손실 (dB).
    - Type 1: 단일 DFT 빔 → 넓은 angular spread에서 큰 손실
    - Type 2 (L=2, 8PSK): 2빔 결합 → angular spread 적응적
    """
    as_deg = channel_params["angular_spread_deg"]
    K_dB = channel_params["rician_K_dB"]

    if codebook_type == "type1":
        base_loss = 2.0 + 0.06 * as_deg
        if K_dB > 3:
            base_loss *= 0.6
        return base_loss
    elif codebook_type == "type2":
        base_loss = 0.8 + 0.02 * as_deg
        phase_quant_loss = 0.3  # 8PSK ~0.3dB additional
        if K_dB > 3:
            base_loss *= 0.5
        return base_loss + phase_quant_loss
    return 0.0


def mu_mimo_interference_dB(n_co_sched, codebook_type, channel_params):
    """
    MU-MIMO 잔여 간섭 (dB).
    CSI 양자화 오차가 inter-UE 간섭으로 변환됨.
    """
    if n_co_sched <= 1:
        return 0.0
    csi_loss = csi_quantization_loss_dB(codebook_type, channel_params)
    interference_per_ue = 10 ** (-csi_loss / 10)
    total_interference = (n_co_sched - 1) * interference_per_ue
    sinr_degradation = -10 * np.log10(1 + total_interference)
    return sinr_degradation


# ── 성능 시뮬레이션 ──────────────────────────────────────────────

UE_COUNTS = [1, 2, 4, 8, 16]

def simulate_scenario(scenario, mode, n_ues, n_monte_carlo=200):
    ch = CHANNEL_PARAMS[scenario]

    ue_snrs = np.random.normal(ch["median_snr_dB"], ch["snr_std_dB"], (n_monte_carlo, n_ues))
    ue_snrs = np.clip(ue_snrs, -5, 35)

    if mode == "type1_su":
        codebook = "type1"
        is_mu = False
    elif mode == "type2_su":
        codebook = "type2"
        is_mu = False
    else:
        codebook = "type2"
        is_mu = True

    csi_loss = csi_quantization_loss_dB(codebook, ch)

    total_bler = []
    total_mcs = []
    total_tput_mbps = []

    for trial in range(n_monte_carlo):
        trial_bler = []
        trial_mcs = []
        trial_tput = 0.0

        re_per_slot = N_PRB * N_SC_PER_PRB * 14 * OVERHEAD_RATIO
        slots_per_sec = 1000 / SLOT_DURATION_MS

        if is_mu:
            n_co = min(n_ues, N_TX)

            for ue_idx in range(n_ues):
                raw_snr = ue_snrs[trial, ue_idx]
                beamforming_gain = 10 * np.log10(N_TX)
                eff_sinr = raw_snr - csi_loss + beamforming_gain
                eff_sinr += mu_mimo_interference_dB(n_co, codebook, ch)

                mcs = link_adapted_mcs(eff_sinr, bler_target=0.10)
                bler = sinr_to_bler(eff_sinr, mcs)
                se = mcs_to_se(mcs) * N_LAYERS_MU

                bits_per_slot = re_per_slot * se
                raw_tput = bits_per_slot * slots_per_sec * TDD_DL_RATIO / 1e6

                time_frac = min(n_co, n_ues) / max(n_ues, 1)
                ue_tput = raw_tput * (1 - bler) * time_frac

                trial_bler.append(bler)
                trial_mcs.append(mcs)
                trial_tput += ue_tput

        else:
            n_layers = min(N_LAYERS_SU_MAX, N_TX, N_RX)
            if codebook == "type1":
                n_layers = min(n_layers, 2)

            for ue_idx in range(n_ues):
                raw_snr = ue_snrs[trial, ue_idx]
                beamforming_gain = 10 * np.log10(min(N_TX, N_RX))
                eff_sinr = raw_snr - csi_loss + beamforming_gain

                mcs = link_adapted_mcs(eff_sinr, bler_target=0.10)
                bler = sinr_to_bler(eff_sinr, mcs)
                se = mcs_to_se(mcs) * n_layers

                bits_per_slot = re_per_slot * se
                raw_tput = bits_per_slot * slots_per_sec * TDD_DL_RATIO / 1e6

                sched_fraction = 1.0 / n_ues
                ue_tput = raw_tput * (1 - bler) * sched_fraction

                trial_bler.append(bler)
                trial_mcs.append(mcs)
                trial_tput += ue_tput

        total_bler.append(np.mean(trial_bler))
        total_mcs.append(np.mean(trial_mcs))
        total_tput_mbps.append(trial_tput)

    return {
        "bler": float(np.mean(total_bler)),
        "mcs": float(np.mean(total_mcs)),
        "tput_mbps": float(np.mean(total_tput_mbps)),
        "bler_std": float(np.std(total_bler)),
        "mcs_std": float(np.std(total_mcs)),
        "tput_std": float(np.std(total_tput_mbps)),
    }


def run_all():
    scenarios = ["UMi-LOS", "UMi-NLOS", "UMa-NLOS"]
    modes = ["type1_su", "type2_su", "type2_mu"]
    mode_labels = {"type1_su": "Type 1 SU-MIMO",
                   "type2_su": "Type 2 SU-MIMO",
                   "type2_mu": "Type 2 MU-MIMO"}

    results = {}
    for sc in scenarios:
        results[sc] = {}
        for mode in modes:
            ml = mode_labels[mode]
            results[sc][ml] = {
                "bler": [], "mcs": [], "tput": [],
                "bler_std": [], "mcs_std": [], "tput_std": [],
            }
            for n_ue in UE_COUNTS:
                r = simulate_scenario(sc, mode, n_ue)
                results[sc][ml]["bler"].append(r["bler"])
                results[sc][ml]["mcs"].append(r["mcs"])
                results[sc][ml]["tput"].append(r["tput_mbps"])
                results[sc][ml]["bler_std"].append(r["bler_std"])
                results[sc][ml]["mcs_std"].append(r["mcs_std"])
                results[sc][ml]["tput_std"].append(r["tput_std"])
                print(f"  {sc} | {ml:20s} | {n_ue:2d} UE → "
                      f"BLER={r['bler']:.3f}  MCS={r['mcs']:.1f}  "
                      f"Tput={r['tput_mbps']:.1f} Mbps")

    out_path = Path(__file__).parent / "results" / "simulated_baseline_data.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"ue_counts": UE_COUNTS, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    print("Simulating baseline performance (Monte Carlo)...")
    run_all()
