#!/usr/bin/env python3
"""
E2E 5-Mode Performance with Measured NMSE (gamma=1/4)
=====================================================
기존 run_e2e_experiment.py의 구조를 유지하되,
CsiNet / Cond-CsiNet의 CSI 양자화 손실을 실측 NMSE 기반으로 계산.

NMSE → SNR penalty 변환:
  CSI_loss_dB = -10*log10(1 - 10^(NMSE_dB/10))
"""

import os, sys, json
import numpy as np

np.random.seed(42)

RESULTS_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
FIG_DIR = os.environ.get("CSINET_FIG_DIR",
                         "/workspace/graduation/csinet/figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load measured NMSE from JSON files ────────────────────────────

def load_measured_nmse(gamma=0.25):
    """Load NMSE values from evaluation JSONs for a given gamma."""
    bl_path = os.path.join(RESULTS_DIR, "csinet_evaluation.json")
    cd_path = os.path.join(RESULTS_DIR, "conditioned_evaluation.json")

    bl, cd = {}, {}
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            for r in json.load(f):
                if abs(r["gamma"] - gamma) < 0.001:
                    bl[r["scenario"]] = r["nmse_dB"]
    if os.path.exists(cd_path):
        with open(cd_path) as f:
            for r in json.load(f):
                if abs(r["gamma"] - gamma) < 0.001:
                    cd[r["scenario"]] = r["conditioned_nmse_dB"]

    # UMa_LOS interpolated from UMi_LOS and UMa_NLOS
    for d in [bl, cd]:
        if "UMi_LOS" in d and "UMa_NLOS" in d and "UMa_LOS" not in d:
            d["UMa_LOS"] = (d["UMi_LOS"] + d["UMa_NLOS"]) / 2

    return {"baseline": bl, "conditioned": cd}

MEASURED_NMSE_DB = load_measured_nmse(0.25)  # default, overridden in main

# ── System parameters ────────────────────────────────────────────
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

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_LOS", "UMa_NLOS"]
UE_COUNTS = [1, 2, 4, 8, 16]
N_MONTE_CARLO = 500
MODES = ["Type 1 SU", "Type 2 SU", "Type 2 MU", "CsiNet MU", "Cond-CsiNet MU"]

SCENARIO_DISPLAY = {
    "UMi_LOS": "UMi-LOS", "UMi_NLOS": "UMi-NLOS",
    "UMa_LOS": "UMa-LOS", "UMa_NLOS": "UMa-NLOS",
}

COLORS = {
    "Type 1 SU": "#636363",
    "Type 2 SU": "#2166ac",
    "Type 2 MU": "#4daf4a",
    "CsiNet MU": "#e41a1c",
    "Cond-CsiNet MU": "#ff7f00",
}

# ── Core PHY functions ───────────────────────────────────────────

def nmse_to_csi_loss_dB(nmse_dB):
    """NMSE (dB) → CSI quantization SNR penalty (dB).
    loss = -10*log10(1 - NMSE_linear)
    """
    nmse_lin = 10 ** (nmse_dB / 10)
    nmse_lin = min(nmse_lin, 0.99)
    return -10 * np.log10(1 - nmse_lin)


def csi_quantization_loss_dB(codebook_type, ch, scenario):
    as_deg = ch["angular_spread_deg"]
    K_dB = ch["rician_K_dB"]
    if codebook_type == "type1":
        base = 3.0 + 0.08 * as_deg
        if K_dB > 3:
            base *= 0.6
        return base
    elif codebook_type == "type2":
        base = 1.5 + 0.03 * as_deg
        if K_dB > 3:
            base *= 0.5
        return base + 0.4
    elif codebook_type == "csinet":
        return nmse_to_csi_loss_dB(MEASURED_NMSE_DB["baseline"][scenario])
    elif codebook_type == "cond_csinet":
        return nmse_to_csi_loss_dB(MEASURED_NMSE_DB["conditioned"][scenario])
    return 0.0


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


def mu_mimo_interference_dB(n_co, codebook_type, ch, scenario):
    if n_co <= 1:
        return 0.0
    csi_loss = csi_quantization_loss_dB(codebook_type, ch, scenario)
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


def mode_config(mode):
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


# ── Simulation ───────────────────────────────────────────────────

def simulate(scenario, mode, n_ues):
    ch = CHANNEL_PARAMS[scenario]
    codebook, is_mu, n_layers = mode_config(mode)

    raw_snrs = np.random.normal(ch["median_snr_dB"], ch["snr_std_dB"],
                                (N_MONTE_CARLO, n_ues))
    raw_snrs = np.clip(raw_snrs, -5, 35)

    csi_loss = csi_quantization_loss_dB(codebook, ch, scenario)
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
                       + mu_mimo_interference_dB(n_co, codebook, ch, scenario)
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


# ── Plotting (old single-gamma version removed, see plot_results in Main) ──


def plot_nmse_comparison():
    """NMSE vs compression ratio bar chart (baseline vs conditioned + Type 2 ref)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csinet_path = os.path.join(RESULTS_DIR, "csinet_evaluation.json")
    cond_path = os.path.join(RESULTS_DIR, "conditioned_evaluation.json")
    if not os.path.exists(csinet_path):
        print("  csinet_evaluation.json not found, skip NMSE plot")
        return

    with open(csinet_path) as f:
        bl_all = json.load(f)
    cond_all = []
    if os.path.exists(cond_path):
        with open(cond_path) as f:
            cond_all = json.load(f)

    GAMMAS = [0.25, 0.125, 0.0625, 0.03125]
    GAMMA_LABELS = ["1/4", "1/8", "1/16", "1/32"]
    plot_scenarios = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
    n_sc = len(plot_scenarios)

    fig, axes = plt.subplots(1, n_sc, figsize=(7 * n_sc, 6))
    fig.suptitle("CSI Compression: NMSE vs Compression Ratio",
                 fontsize=14, fontweight="bold")
    x = np.arange(len(GAMMAS))
    bar_w = 0.25

    for idx, sc in enumerate(plot_scenarios):
        ax = axes[idx]
        bl_data = [r for r in bl_all if r["scenario"] == sc]
        bl_nmse, type2_val = [], None
        for g in GAMMAS:
            m = [r for r in bl_data if abs(r["gamma"] - g) < 0.001]
            bl_nmse.append(m[0]["nmse_dB"] if m else 0)
            if type2_val is None and m:
                type2_val = m[0].get("type2_nmse_dB")

        cd_data = [r for r in cond_all if r["scenario"] == sc]
        cd_nmse = []
        for g in GAMMAS:
            m = [r for r in cd_data if abs(r["gamma"] - g) < 0.001]
            cd_nmse.append(m[0]["conditioned_nmse_dB"] if m else 0)

        ax.bar(x - bar_w / 2, bl_nmse, bar_w, color="#2166ac",
               alpha=0.85, label="CsiNet Baseline", edgecolor="white", linewidth=0.5)
        ax.bar(x + bar_w / 2, cd_nmse, bar_w, color="#e41a1c",
               alpha=0.85, label="Conditioned CsiNet", edgecolor="white", linewidth=0.5)

        if type2_val is not None:
            ax.axhline(type2_val, color="#4daf4a", ls="--", lw=2,
                       label=f"Type 2 (L=2, 8PSK): {type2_val:.1f} dB")

        ax.set_xlabel("Compression Ratio (γ)")
        ax.set_ylabel("NMSE (dB)")
        ax.set_title(SCENARIO_DISPLAY[sc], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(GAMMA_LABELS)
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "csinet_nmse_comparison.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Main ─────────────────────────────────────────────────────────

GAMMA_LABEL = {0.25: "1/4", 0.125: "1/8", 0.0625: "1/16", 0.03125: "1/32"}


def run_e2e_for_gamma(gamma):
    """Run full E2E simulation and plotting for a given gamma."""
    global MEASURED_NMSE_DB
    MEASURED_NMSE_DB = load_measured_nmse(gamma)
    gl = GAMMA_LABEL.get(gamma, f"{gamma}")

    print(f"\n{'=' * 70}")
    print(f"E2E 5-Mode with Measured NMSE  (γ = {gl})")
    print(f"{'=' * 70}")

    print(f"\nLoaded NMSE (γ={gl}):")
    for key in ["baseline", "conditioned"]:
        for sc, val in MEASURED_NMSE_DB[key].items():
            print(f"  {key:14s} {sc:10s}: {val:.2f} dB")

    print("\nCSI quantization loss (NMSE → SNR penalty):")
    for sc in SCENARIOS:
        ch = CHANNEL_PARAMS[sc]
        for cb in ["type1", "type2", "csinet", "cond_csinet"]:
            loss = csi_quantization_loss_dB(cb, ch, sc)
            print(f"  {sc:10s} {cb:14s}: {loss:.2f} dB")

    np.random.seed(42)
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

    out_path = os.path.join(RESULTS_DIR, f"e2e_5mode_gamma{gl.replace('/','_')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    plot_results(all_results, gamma)


def plot_results(all_results, gamma=0.25):
    """Generate per-scenario E2E plots for a given gamma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "legend.fontsize": 9, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    gl = GAMMA_LABEL.get(gamma, f"{gamma}")
    n_modes = len(MODES)
    bar_w = 0.15
    x = np.arange(len(UE_COUNTS))
    metrics = ["bler_mean", "mcs_mean", "throughput_mean"]
    ylabels = ["BLER (%)", "Average MCS Index", "Cell Throughput (Mbps)"]

    for sc in SCENARIOS:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        fig.suptitle(
            f"E2E 5-Mode Performance — {SCENARIO_DISPLAY[sc]}  (γ = {gl})",
            fontsize=15, fontweight="bold")

        handles = []
        for col, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[col]
            for mi, mode in enumerate(MODES):
                y = [all_results[sc][mode][str(n)][metric] for n in UE_COUNTS]
                if metric == "bler_mean":
                    y = [v * 100 for v in y]
                bar = ax.bar(x + mi * bar_w, y, bar_w, color=COLORS[mode],
                             alpha=0.85, label=mode, edgecolor="white",
                             linewidth=0.5)
                if col == 0:
                    handles.append(bar)
            ax.set_xlabel("Number of UEs")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x + bar_w * (n_modes - 1) / 2)
            ax.set_xticklabels(UE_COUNTS)

        fig.legend(handles, MODES, loc="upper center",
                   bbox_to_anchor=(0.5, 1.08), ncol=n_modes,
                   framealpha=0.9, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fname = os.path.join(FIG_DIR, f"e2e_5mode_{sc}_g{gl.replace('/','_')}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Throughput gain over Type 2 MU
    gain_modes = ["CsiNet MU", "Cond-CsiNet MU"]
    gain_colors = [COLORS[m] for m in gain_modes]
    gbar_w = 0.30
    for sc in SCENARIOS:
        fig, ax = plt.subplots(figsize=(7, 5))
        base_tp = [all_results[sc]["Type 2 MU"][str(n)]["throughput_mean"]
                   for n in UE_COUNTS]
        for mi, mode in enumerate(gain_modes):
            tp = [all_results[sc][mode][str(n)]["throughput_mean"]
                  for n in UE_COUNTS]
            gain = [(t - b) / max(b, 0.1) * 100 for t, b in zip(tp, base_tp)]
            ax.bar(x + (mi - 0.5) * gbar_w, gain, gbar_w,
                   color=gain_colors[mi], alpha=0.85, label=mode,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Throughput Gain (%)")
        ax.set_title(
            f"Throughput Gain vs Type 2 MU — {SCENARIO_DISPLAY[sc]}  (γ = {gl})",
            fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(UE_COUNTS)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        fname = os.path.join(FIG_DIR, f"e2e_gain_{sc}_g{gl.replace('/','_')}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved: {fname}")


def main():
    for gamma in [0.25, 0.125]:
        run_e2e_for_gamma(gamma)
    plot_nmse_comparison()
    print("\nAll done.")


if __name__ == "__main__":
    main()
