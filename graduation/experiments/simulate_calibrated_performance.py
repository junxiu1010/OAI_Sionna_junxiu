#!/usr/bin/env python3
"""
OAI-Sionna 시뮬레이터 실측 보정 성능 시뮬레이션
================================================
실제 OAI-Sionna 시뮬레이터 로그에서 관측된 성능 특성을 반영하여
현실적인 기대 성능을 산출합니다.

실측 관찰:
  - MCS: 대부분 0~4, 양호 조건에서 최대 8~12
  - BLER: CQI 15 보고에도 실제 BLER 3%~35%
  - RI: 실질적으로 1~2 (4에 미달)
  - RLF: 4 UE 시 2~3회, 8 UE 이상에서 급증
  - Sionna 프록시 처리 지연 → HARQ 타이밍 불안정
  - GPU IPC → 채널 추정 오차 증가
  - MU-MIMO 페어링 이벤트 매우 드묾

보정 요소:
  1. 처리 지연에 의한 실효 SNR 열화 (~8-12 dB)
  2. CQI/MCS 미스매치에 의한 MCS 상한 제한
  3. 다중 UE 시 스케줄러 오버헤드 및 충돌
  4. RLF에 의한 실효 UE 수 감소
  5. HARQ 재전송 실패에 의한 추가 BLER
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
TDD_DL_RATIO = 7 / 10
OVERHEAD_RATIO = 0.85
N_TX = 4
N_RX = 2

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

# ── OAI-Sionna 실측 보정 파라미터 ─────────────────────────────────

# Sionna 프록시 처리 지연 + GPU IPC에 의한 실효 SNR 열화
PROXY_LATENCY_SNR_PENALTY_dB = 8.0

# CQI-MCS 미스매치: OAI에서 CQI 15를 보고하지만 실제 MCS는 훨씬 낮음
# → 실효 MCS 상한을 제한
PRACTICAL_MCS_CAP = 14

# HARQ 타이밍 불안정에 의한 추가 BLER (재전송 실패)
HARQ_TIMING_BLER_FLOOR = 0.03

# UE 수 증가에 따른 스케줄러 오버헤드 및 충돌 확률
def scheduler_efficiency(n_ues):
    """UE 수에 따른 스케줄링 효율 저하 (실측 기반)"""
    if n_ues <= 1:
        return 0.90
    elif n_ues <= 2:
        return 0.82
    elif n_ues <= 4:
        return 0.70
    elif n_ues <= 8:
        return 0.55
    else:
        return 0.40

# RLF 확률 모델 (실측: 4 UE → ~50% RLF, 8+ UE → ~75%)
def rlf_survival_rate(n_ues, scenario):
    """RLF로 인해 실제 데이터 전송 가능한 UE 비율"""
    base_rlf_prob = {
        "UMi-LOS":  np.array([0.0, 0.05, 0.15, 0.35, 0.55]),
        "UMi-NLOS": np.array([0.05, 0.10, 0.25, 0.50, 0.70]),
        "UMa-LOS":  np.array([0.02, 0.08, 0.20, 0.42, 0.62]),
        "UMa-NLOS": np.array([0.10, 0.20, 0.40, 0.60, 0.80]),
    }
    ue_idx_map = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    idx = ue_idx_map.get(n_ues, 4)
    rlf_prob = base_rlf_prob[scenario][idx]
    survival = 1.0 - rlf_prob
    return max(survival, 0.15)


# ── 채널 모델 (3GPP TR 38.901 기반 + OAI 실측 보정) ───────────────

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
    "UMa-LOS": {
        "median_snr_dB": 18.0,
        "snr_std_dB": 5.0,
        "rician_K_dB": 7.0,
        "angular_spread_deg": 22.0,
        "delay_spread_ns": 45.0,
    },
    "UMa-NLOS": {
        "median_snr_dB": 10.0,
        "snr_std_dB": 7.0,
        "rician_K_dB": 0.0,
        "angular_spread_deg": 55.0,
        "delay_spread_ns": 120.0,
    },
}


def csi_quantization_loss_dB(codebook_type, channel_params):
    as_deg = channel_params["angular_spread_deg"]
    K_dB = channel_params["rician_K_dB"]

    if codebook_type == "type1":
        base_loss = 3.0 + 0.08 * as_deg
        if K_dB > 3:
            base_loss *= 0.6
        return base_loss
    elif codebook_type == "type2":
        base_loss = 1.5 + 0.03 * as_deg
        phase_quant_loss = 0.4
        if K_dB > 3:
            base_loss *= 0.5
        return base_loss + phase_quant_loss
    return 0.0


def mu_mimo_interference_dB(n_co_sched, codebook_type, channel_params):
    if n_co_sched <= 1:
        return 0.0
    csi_loss = csi_quantization_loss_dB(codebook_type, channel_params)
    csi_loss += 2.0  # GPU IPC 추가 양자화 오차
    interference_per_ue = 10 ** (-csi_loss / 10)
    total_interference = (n_co_sched - 1) * interference_per_ue
    sinr_degradation = -10 * np.log10(1 + total_interference)
    return sinr_degradation


def sinr_to_bler(sinr_dB, mcs):
    if mcs < 0 or mcs > 27:
        return 1.0
    threshold = SINR_TO_MCS_THRESHOLDS[min(mcs, 27)]
    margin = sinr_dB - threshold
    # 실측 보정: 더 완만한 BLER 곡선 (OAI에서 BLER 분산이 큼)
    bler = 1.0 / (1.0 + np.exp(1.5 * margin))
    bler = max(bler, HARQ_TIMING_BLER_FLOOR)
    return float(np.clip(bler, 0.001, 0.8))


def link_adapted_mcs(sinr_dB, bler_target=0.10):
    best_mcs = 0
    for mcs in range(min(28, PRACTICAL_MCS_CAP + 1)):
        bler = sinr_to_bler(sinr_dB, mcs)
        if bler <= bler_target:
            best_mcs = mcs
        else:
            break
    return best_mcs


def mcs_to_se(mcs):
    return MCS_TABLE.get(min(mcs, 27), 0.2344)


# ── 성능 시뮬레이션 (실측 보정) ───────────────────────────────────

UE_COUNTS = [1, 2, 4, 8, 16]

def simulate_scenario(scenario, mode, n_ues, n_monte_carlo=500):
    ch = CHANNEL_PARAMS[scenario]

    raw_snrs = np.random.normal(ch["median_snr_dB"], ch["snr_std_dB"],
                                (n_monte_carlo, n_ues))
    raw_snrs = np.clip(raw_snrs, -5, 35)

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
    sched_eff = scheduler_efficiency(n_ues)
    survival = rlf_survival_rate(n_ues, scenario)

    total_bler = []
    total_mcs = []
    total_tput_mbps = []

    for trial in range(n_monte_carlo):
        trial_bler_list = []
        trial_mcs_list = []
        trial_tput = 0.0

        re_per_slot = N_PRB * N_SC_PER_PRB * 14 * OVERHEAD_RATIO
        slots_per_sec = 1000 / SLOT_DURATION_MS

        # RLF로 인해 실제 활성 UE 수
        n_active = max(1, int(np.round(n_ues * survival)))
        # 이번 trial에서 랜덤하게 활성 UE 선택
        active_mask = np.zeros(n_ues, dtype=bool)
        active_indices = np.random.choice(n_ues, size=n_active, replace=False)
        active_mask[active_indices] = True

        if is_mu:
            n_co = min(n_active, N_TX)

            for ue_idx in range(n_ues):
                if not active_mask[ue_idx]:
                    trial_bler_list.append(1.0)
                    trial_mcs_list.append(0)
                    continue

                raw_snr = raw_snrs[trial, ue_idx]
                # 실효 SINR: 프록시 지연 페널티 적용
                eff_sinr = (raw_snr
                            - PROXY_LATENCY_SNR_PENALTY_dB
                            - csi_loss
                            + 10 * np.log10(N_TX)    # BF gain
                            + mu_mimo_interference_dB(n_co, codebook, ch))

                # 다중 UE 간섭 추가 (스케줄러 불완전 분리)
                ue_noise = np.random.normal(0, 1.5)
                eff_sinr += ue_noise

                mcs = link_adapted_mcs(eff_sinr, bler_target=0.10)
                bler = sinr_to_bler(eff_sinr, mcs)
                se = mcs_to_se(mcs) * 1  # MU-MIMO: rank 1

                bits_per_slot = re_per_slot * se
                raw_tput = bits_per_slot * slots_per_sec * TDD_DL_RATIO / 1e6

                time_frac = min(n_co, n_active) / max(n_active, 1)
                ue_tput = raw_tput * (1 - bler) * time_frac * sched_eff

                trial_bler_list.append(bler)
                trial_mcs_list.append(mcs)
                trial_tput += ue_tput

        else:  # SU-MIMO
            # 실측: RI 대부분 1-2
            if codebook == "type1":
                n_layers = 1  # Type 1: 실측에서 대부분 rank 1
            else:
                n_layers = min(2, N_RX)  # Type 2: 실측에서 rank 1-2

            for ue_idx in range(n_ues):
                if not active_mask[ue_idx]:
                    trial_bler_list.append(1.0)
                    trial_mcs_list.append(0)
                    continue

                raw_snr = raw_snrs[trial, ue_idx]
                eff_sinr = (raw_snr
                            - PROXY_LATENCY_SNR_PENALTY_dB
                            - csi_loss
                            + 10 * np.log10(min(N_TX, N_RX)))  # BF gain

                ue_noise = np.random.normal(0, 1.0)
                eff_sinr += ue_noise

                mcs = link_adapted_mcs(eff_sinr, bler_target=0.10)
                bler = sinr_to_bler(eff_sinr, mcs)
                se = mcs_to_se(mcs) * n_layers

                bits_per_slot = re_per_slot * se
                raw_tput = bits_per_slot * slots_per_sec * TDD_DL_RATIO / 1e6

                sched_fraction = 1.0 / n_active
                ue_tput = raw_tput * (1 - bler) * sched_fraction * sched_eff

                trial_bler_list.append(bler)
                trial_mcs_list.append(mcs)
                trial_tput += ue_tput

        # 활성 UE만의 평균 (RLF UE 제외)
        active_blers = [trial_bler_list[i] for i in range(n_ues) if active_mask[i]]
        active_mcss = [trial_mcs_list[i] for i in range(n_ues) if active_mask[i]]

        total_bler.append(np.mean(active_blers) if active_blers else 1.0)
        total_mcs.append(np.mean(active_mcss) if active_mcss else 0.0)
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
    scenarios = ["UMi-LOS", "UMi-NLOS", "UMa-LOS", "UMa-NLOS"]
    modes = ["type1_su", "type2_su", "type2_mu"]
    mode_labels = {
        "type1_su": "Type 1 SU-MIMO",
        "type2_su": "Type 2 SU-MIMO",
        "type2_mu": "Type 2 MU-MIMO",
    }

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

    out_path = Path(__file__).parent / "results" / "calibrated_performance_data.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"ue_counts": UE_COUNTS, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    print("Simulating calibrated performance (OAI-Sionna adjusted)...")
    run_all()
