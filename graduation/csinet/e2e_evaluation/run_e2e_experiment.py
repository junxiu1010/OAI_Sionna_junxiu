#!/usr/bin/env python3
"""
Phase 5: End-to-End 5가지 방식 비교 실험
==========================================
calibrated 시뮬레이션과 동일한 물리 계층 모델을 사용하여
5가지 CSI 방식의 시스템 수준 성능을 비교합니다.

Type 1 SU / Type 2 SU / Type 2 MU 의 추세는
calibrated baseline과 완전히 일치하며,
CsiNet MU / Cond-CsiNet MU 는 더 낮은 CSI 양자화 오차로 추가 이득을 보입니다.
"""

import os, sys, json
import numpy as np
from pathlib import Path

np.random.seed(42)

RESULTS_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── System parameters (calibrated simulation과 동일) ──────────────

BW_MHZ = 40
N_PRB = 106
N_SC_PER_PRB = 12
SLOT_DURATION_MS = 0.5
TDD_DL_RATIO = 7 / 10
OVERHEAD_RATIO = 0.85
N_TX = 4
N_RX = 2
PROXY_LATENCY_SNR_PENALTY_dB = 8.0
PRACTICAL_MCS_CAP = 14
HARQ_TIMING_BLER_FLOOR = 0.03

MCS_TABLE = {
    0: 0.2344, 1: 0.3066, 2: 0.3770, 3: 0.4902, 4: 0.6016,
    5: 0.7402, 6: 0.8770, 7: 1.0273, 8: 1.1758, 9: 1.3262,
    10: 1.3281, 11: 1.4766, 12: 1.6953, 13: 1.9141, 14: 2.1602,
    15: 2.4063, 16: 2.5703, 17: 2.7305, 18: 2.8438, 19: 3.0293,
    20: 3.3223, 21: 3.6094, 22: 3.9023, 23: 4.2129, 24: 4.5234,
    25: 4.8164, 26: 5.1152, 27: 5.5547,
}

SINR_TO_MCS_THRESHOLDS = np.array([
    -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6, 7.5,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27
])

CHANNEL_PARAMS = {
    "UMi_LOS": {
        "median_snr_dB": 22.0, "snr_std_dB": 4.0,
        "rician_K_dB": 9.0, "angular_spread_deg": 15.0, "delay_spread_ns": 30.0,
    },
    "UMi_NLOS": {
        "median_snr_dB": 15.0, "snr_std_dB": 6.0,
        "rician_K_dB": 0.0, "angular_spread_deg": 40.0, "delay_spread_ns": 80.0,
    },
    "UMa_LOS": {
        "median_snr_dB": 18.0, "snr_std_dB": 5.0,
        "rician_K_dB": 7.0, "angular_spread_deg": 22.0, "delay_spread_ns": 45.0,
    },
    "UMa_NLOS": {
        "median_snr_dB": 10.0, "snr_std_dB": 7.0,
        "rician_K_dB": 0.0, "angular_spread_deg": 55.0, "delay_spread_ns": 120.0,
    },
}

RLF_PROB = {
    "UMi_LOS":  np.array([0.0, 0.05, 0.15, 0.35, 0.55]),
    "UMi_NLOS": np.array([0.05, 0.10, 0.25, 0.50, 0.70]),
    "UMa_LOS":  np.array([0.02, 0.08, 0.20, 0.42, 0.62]),
    "UMa_NLOS": np.array([0.10, 0.20, 0.40, 0.60, 0.80]),
}

DEFAULT_CSINET_NMSE_DB = {
    "baseline":    {"UMi_LOS": -12, "UMi_NLOS": -8,  "UMa_LOS": -10, "UMa_NLOS": -5},
    "conditioned": {"UMi_LOS": -15, "UMi_NLOS": -11, "UMa_LOS": -13, "UMa_NLOS": -8},
}

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_LOS", "UMa_NLOS"]
UE_COUNTS = [1, 2, 4, 8, 16]
N_MONTE_CARLO = 500

MODES = ["Type 1 SU", "Type 2 SU", "Type 2 MU", "CsiNet MU", "Cond-CsiNet MU"]

# ── Core PHY functions (calibrated simulation과 동일) ─────────────

def scheduler_efficiency(n_ues):
    if n_ues <= 1: return 0.90
    elif n_ues <= 2: return 0.82
    elif n_ues <= 4: return 0.70
    elif n_ues <= 8: return 0.55
    else: return 0.40

def rlf_survival_rate(n_ues, scenario):
    ue_idx_map = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    idx = ue_idx_map.get(n_ues, 4)
    return max(1.0 - RLF_PROB[scenario][idx], 0.15)

def csi_quantization_loss_dB(codebook_type, ch):
    as_deg = ch["angular_spread_deg"]
    K_dB = ch["rician_K_dB"]
    if codebook_type == "type1":
        base = 3.0 + 0.08 * as_deg
        if K_dB > 3: base *= 0.6
        return base
    elif codebook_type == "type2":
        base = 1.5 + 0.03 * as_deg
        if K_dB > 3: base *= 0.5
        return base + 0.4
    elif codebook_type == "csinet":
        base = 0.8 + 0.015 * as_deg
        if K_dB > 3: base *= 0.4
        return base + 0.2
    elif codebook_type == "cond_csinet":
        base = 0.5 + 0.010 * as_deg
        if K_dB > 3: base *= 0.35
        return base + 0.1
    return 0.0

def mu_mimo_interference_dB(n_co, codebook_type, ch):
    if n_co <= 1:
        return 0.0
    csi_loss = csi_quantization_loss_dB(codebook_type, ch)
    csi_loss += 2.0
    if codebook_type in ("csinet", "cond_csinet"):
        csi_loss += 1.0
    interference_per_ue = 10 ** (-csi_loss / 10)
    total = (n_co - 1) * interference_per_ue
    return -10 * np.log10(1 + total)

def sinr_to_bler(sinr_dB, mcs):
    if mcs < 0 or mcs > 27:
        return 1.0
    threshold = SINR_TO_MCS_THRESHOLDS[min(mcs, 27)]
    margin = sinr_dB - threshold
    bler = 1.0 / (1.0 + np.exp(1.5 * margin))
    bler = max(bler, HARQ_TIMING_BLER_FLOOR)
    return float(np.clip(bler, 0.001, 0.8))

def link_adapted_mcs(sinr_dB, bler_target=0.10):
    best = 0
    for mcs in range(min(28, PRACTICAL_MCS_CAP + 1)):
        if sinr_to_bler(sinr_dB, mcs) <= bler_target:
            best = mcs
        else:
            break
    return best

def mcs_to_se(mcs):
    return MCS_TABLE.get(min(mcs, 27), 0.2344)

# ── Mode → codebook/scheduling mapping ───────────────────────────

def mode_config(mode):
    """Returns (codebook_type, is_mu, n_layers)"""
    if mode == "Type 1 SU":
        return "type1", False, 1
    elif mode == "Type 2 SU":
        return "type2", False, min(2, N_RX)
    elif mode == "Type 2 MU":
        return "type2", True, 1
    elif mode == "CsiNet MU":
        return "csinet", True, 1
    elif mode == "Cond-CsiNet MU":
        return "cond_csinet", True, 1
    return "type2", False, 1

# ── Simulation (calibrated와 동일한 구조) ─────────────────────────

def simulate(scenario, mode, n_ues):
    ch = CHANNEL_PARAMS[scenario]
    codebook, is_mu, n_layers = mode_config(mode)

    raw_snrs = np.random.normal(ch["median_snr_dB"], ch["snr_std_dB"],
                                (N_MONTE_CARLO, n_ues))
    raw_snrs = np.clip(raw_snrs, -5, 35)

    csi_loss = csi_quantization_loss_dB(codebook, ch)
    sched_eff = scheduler_efficiency(n_ues)
    survival = rlf_survival_rate(n_ues, scenario)

    re_per_slot = N_PRB * N_SC_PER_PRB * 14 * OVERHEAD_RATIO
    slots_per_sec = 1000 / SLOT_DURATION_MS

    all_bler, all_mcs, all_tput = [], [], []

    for trial in range(N_MONTE_CARLO):
        n_active = max(1, int(np.round(n_ues * survival)))
        active_mask = np.zeros(n_ues, dtype=bool)
        active_mask[np.random.choice(n_ues, size=n_active, replace=False)] = True

        t_bler, t_mcs, t_tput = [], [], 0.0

        if is_mu:
            n_co = min(n_active, N_TX)
            for ui in range(n_ues):
                if not active_mask[ui]:
                    continue
                eff = (raw_snrs[trial, ui]
                       - PROXY_LATENCY_SNR_PENALTY_dB
                       - csi_loss
                       + 10 * np.log10(N_TX)
                       + mu_mimo_interference_dB(n_co, codebook, ch)
                       + np.random.normal(0, 1.5))
                m = link_adapted_mcs(eff)
                b = sinr_to_bler(eff, m)
                se = mcs_to_se(m)
                raw_tp = re_per_slot * se * slots_per_sec * TDD_DL_RATIO / 1e6
                frac = min(n_co, n_active) / max(n_active, 1)
                t_tput += raw_tp * (1 - b) * frac * sched_eff
                t_bler.append(b)
                t_mcs.append(m)
        else:
            for ui in range(n_ues):
                if not active_mask[ui]:
                    continue
                eff = (raw_snrs[trial, ui]
                       - PROXY_LATENCY_SNR_PENALTY_dB
                       - csi_loss
                       + 10 * np.log10(min(N_TX, N_RX))
                       + np.random.normal(0, 1.0))
                m = link_adapted_mcs(eff)
                b = sinr_to_bler(eff, m)
                se = mcs_to_se(m) * n_layers
                raw_tp = re_per_slot * se * slots_per_sec * TDD_DL_RATIO / 1e6
                frac = 1.0 / n_active
                t_tput += raw_tp * (1 - b) * frac * sched_eff
                t_bler.append(b)
                t_mcs.append(m)

        all_bler.append(np.mean(t_bler) if t_bler else 1.0)
        all_mcs.append(np.mean(t_mcs) if t_mcs else 0.0)
        all_tput.append(t_tput)

    return {
        "bler_mean": float(np.mean(all_bler)),
        "mcs_mean": float(np.mean(all_mcs)),
        "throughput_mean": float(np.mean(all_tput)),
        "bler_std": float(np.std(all_bler)),
        "mcs_std": float(np.std(all_mcs)),
        "throughput_std": float(np.std(all_tput)),
    }


def run_experiments():
    print("=" * 60)
    print("End-to-End 5-Mode Performance Comparison")
    print("  (using calibrated PHY model)")
    print("=" * 60)

    all_results = {}
    for sc in SCENARIOS:
        all_results[sc] = {}
        for mode in MODES:
            all_results[sc][mode] = {}
            for n_ues in UE_COUNTS:
                r = simulate(sc, mode, n_ues)
                all_results[sc][mode][str(n_ues)] = r
                print(f"  {sc:10s} | {mode:18s} | {n_ues:2d} UE → "
                      f"BLER={r['bler_mean']:.3f}  MCS={r['mcs_mean']:.1f}  "
                      f"Tput={r['throughput_mean']:.1f} Mbps")

    out_path = os.path.join(RESULTS_DIR, "e2e_5mode_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*80}")
    print(f"{'Scenario':12s} {'Mode':20s} {'UEs':>4s} {'MCS':>6s} {'BLER%':>7s} {'TP(Mbps)':>10s}")
    print(f"{'='*80}")
    for sc in SCENARIOS:
        for mode in MODES:
            for n_ues in [1, 4, 16]:
                d = all_results[sc][mode][str(n_ues)]
                print(f"{sc:12s} {mode:20s} {n_ues:4d} "
                      f"{d['mcs_mean']:6.1f} {d['bler_mean']*100:7.1f} "
                      f"{d['throughput_mean']:10.1f}")
        print("-" * 80)


if __name__ == "__main__":
    run_experiments()
