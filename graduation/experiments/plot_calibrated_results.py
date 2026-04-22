#!/usr/bin/env python3
"""
OAI-Sionna 보정 성능 비교 그래프 생성
======================================
실측 보정된 시뮬레이션 데이터를 사용하여 논문용 그래프를 생성합니다.

출력:
  - 시나리오별 1×3 subplot (BLER, MCS, Throughput) × 3개
  - 통합 3×3 격자 그래프
  - 모드별 throughput 비교 막대 그래프
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_PATH = Path(__file__).parent / "results" / "calibrated_performance_data.json"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
})

MODES = ["Type 1 SU-MIMO", "Type 2 SU-MIMO", "Type 2 MU-MIMO"]
COLORS = {
    "Type 1 SU-MIMO": "#2166ac",
    "Type 2 SU-MIMO": "#4daf4a",
    "Type 2 MU-MIMO": "#e41a1c",
}
MARKERS = {"Type 1 SU-MIMO": "s", "Type 2 SU-MIMO": "^", "Type 2 MU-MIMO": "o"}
LSTYLES = {"Type 1 SU-MIMO": "--", "Type 2 SU-MIMO": "-.", "Type 2 MU-MIMO": "-"}

METRICS = [
    ("bler",  "BLER (%)",              lambda v: np.array(v) * 100),
    ("mcs",   "Average MCS Index",     lambda v: np.array(v)),
    ("tput",  "Cell DL Throughput (Mbps)", lambda v: np.array(v)),
]

SCENARIO_LABELS = {
    "UMi-LOS":  "3GPP UMi-LOS (Street Canyon)",
    "UMi-NLOS": "3GPP UMi-NLOS (Street Canyon)",
    "UMa-LOS":  "3GPP UMa-LOS (Urban Macro)",
    "UMa-NLOS": "3GPP UMa-NLOS (Urban Macro)",
}
SCENARIO_FILENAME = {
    "UMi-LOS":  "UMi_LOS",
    "UMi-NLOS": "UMi_NLOS",
    "UMa-LOS":  "UMa_LOS",
    "UMa-NLOS": "UMa_NLOS",
}


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return np.array(raw["ue_counts"]), raw["results"]


def plot_single_scenario(ue_counts, data, scenario_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    label = SCENARIO_LABELS.get(scenario_name, scenario_name)
    fig.suptitle(f"{label}\n(OAI-Sionna Calibrated, 4T4R, 40 MHz, FR1 Band 78)",
                 fontsize=14, fontweight="bold", y=1.06)

    for col, (key, ylabel, transform) in enumerate(METRICS):
        ax = axes[col]
        for mode in MODES:
            y = transform(data[mode][key])
            y_std = transform(data[mode].get(f"{key}_std", [0]*len(y)))
            ax.plot(ue_counts, y,
                    color=COLORS[mode], marker=MARKERS[mode],
                    linestyle=LSTYLES[mode], label=mode,
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.set_xlabel("Number of UEs")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ue_counts)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.minorticks_off()
        if key == "bler":
            ax.set_ylim(bottom=0, top=min(np.max([
                np.max(transform(data[m][key])) for m in MODES
            ]) * 1.5, 50))
        elif key == "tput":
            ax.set_ylim(bottom=0)
        elif key == "mcs":
            ax.set_ylim(bottom=0, top=min(np.max([
                np.max(transform(data[m][key])) for m in MODES
            ]) * 1.3, 28))
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    safe_name = SCENARIO_FILENAME.get(scenario_name, scenario_name.replace('-', '_'))
    fname = FIG_DIR / f"calibrated_{safe_name}.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_combined(ue_counts, all_data):
    scenario_names = list(all_data.keys())
    n_scenarios = len(scenario_names)
    fig, axes = plt.subplots(n_scenarios, 3, figsize=(17, 4.5 * n_scenarios))
    fig.suptitle(
        "OAI-Sionna Simulator Performance: Type 1 SU / Type 2 SU / Type 2 MU-MIMO\n"
        "(Calibrated with Real System Measurements — 4T4R, 106 PRB, FR1 Band 78)",
        fontsize=14, fontweight="bold", y=0.995)

    for row, sc_name in enumerate(scenario_names):
        data = all_data[sc_name]
        for col, (key, ylabel, transform) in enumerate(METRICS):
            ax = axes[row][col]
            for mode in MODES:
                y = transform(data[mode][key])
                ax.plot(ue_counts, y,
                        color=COLORS[mode], marker=MARKERS[mode],
                        linestyle=LSTYLES[mode], label=mode,
                        markeredgecolor="white", markeredgewidth=0.8)

            ax.set_xscale("log", base=2)
            ax.set_xticks(ue_counts)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.minorticks_off()

            if row == n_scenarios - 1:
                ax.set_xlabel("Number of UEs")
            if col == 0:
                ax.set_ylabel(
                    SCENARIO_LABELS.get(sc_name, sc_name).split("(")[0].strip(),
                    fontsize=11, fontweight="bold", rotation=90, labelpad=18)
            if key in ("bler", "tput"):
                ax.set_ylim(bottom=0)
            if key == "mcs":
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(ylabel, fontsize=12, pad=8)
            if row == 0 and col == 2:
                ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.tight_layout(rect=[0.04, 0, 1, 0.955])
    fname = FIG_DIR / "calibrated_combined_grid.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_throughput_bar_comparison(ue_counts, all_data):
    """UE 수별 throughput 막대 그래프 (모드 + 시나리오 비교)"""
    n_sc = len(all_data)
    fig, axes = plt.subplots(1, n_sc, figsize=(5.5 * n_sc, 5.5))
    fig.suptitle(
        "Cell Throughput Comparison by Scenario and MIMO Mode\n"
        "(OAI-Sionna Calibrated)",
        fontsize=14, fontweight="bold", y=1.04)

    scenarios = list(all_data.keys())
    bar_width = 0.25
    x = np.arange(len(ue_counts))

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        data = all_data[sc]
        for mi, mode in enumerate(MODES):
            y = np.array(data[mode]["tput"])
            y_std = np.array(data[mode]["tput_std"])
            bars = ax.bar(x + mi * bar_width, y, bar_width,
                          color=COLORS[mode], alpha=0.85,
                          label=mode, edgecolor="white", linewidth=0.5)
            for bar_item, val in zip(bars, y):
                height = bar_item.get_height()
                ax.text(bar_item.get_x() + bar_item.get_width() / 2., height + 0.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Cell DL Throughput (Mbps)")
        ax.set_title(SCENARIO_LABELS.get(sc, sc).split("(")[0].strip(),
                      fontsize=12, fontweight="bold")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([str(u) for u in ue_counts])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fname = FIG_DIR / "calibrated_throughput_bars.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_mode_gain_summary(ue_counts, all_data):
    """Type 2 vs Type 1 성능 이득 요약 그래프"""
    n_sc = len(all_data)
    fig, axes = plt.subplots(1, n_sc, figsize=(4.5 * n_sc, 5))
    fig.suptitle(
        "Type 2 Codebook Gain over Type 1 Baseline\n"
        "(OAI-Sionna Calibrated, 4T4R)",
        fontsize=14, fontweight="bold", y=1.05)

    scenarios = list(all_data.keys())
    gain_colors = {"Type 2 SU-MIMO": "#4daf4a", "Type 2 MU-MIMO": "#e41a1c"}

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        data = all_data[sc]
        type1_tput = np.array(data["Type 1 SU-MIMO"]["tput"])

        for mode in ["Type 2 SU-MIMO", "Type 2 MU-MIMO"]:
            mode_tput = np.array(data[mode]["tput"])
            # dB gain 대신 % gain
            gain_pct = (mode_tput / np.maximum(type1_tput, 0.1) - 1) * 100
            ax.plot(ue_counts, gain_pct,
                    color=gain_colors[mode], marker=MARKERS[mode],
                    linestyle=LSTYLES[mode], label=mode, linewidth=2.5,
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Throughput Gain over Type 1 (%)")
        ax.set_title(SCENARIO_LABELS.get(sc, sc).split("(")[0].strip(),
                      fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ue_counts)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.legend(loc="best", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fname = FIG_DIR / "calibrated_type2_gain.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading calibrated data and generating figures...")
    ue_counts, results = load_data()

    for sc_name, data in results.items():
        plot_single_scenario(ue_counts, data, sc_name)

    plot_combined(ue_counts, results)
    plot_throughput_bar_comparison(ue_counts, results)
    plot_mode_gain_summary(ue_counts, results)

    print("\n=== Summary ===")
    for sc_name, data in results.items():
        print(f"\n{sc_name}:")
        for mode in MODES:
            tput = data[mode]["tput"]
            mcs = data[mode]["mcs"]
            bler = data[mode]["bler"]
            print(f"  {mode:20s}: MCS {mcs[0]:.1f}→{mcs[-1]:.1f}  "
                  f"BLER {bler[0]*100:.1f}%→{bler[-1]*100:.1f}%  "
                  f"Tput {tput[0]:.1f}→{tput[-1]:.1f} Mbps")
    print("\nDone.")
