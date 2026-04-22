#!/usr/bin/env python3
"""
시뮬레이션 결과 기반 Baseline 성능 비교 그래프 생성

출력:
  - 시나리오별 1×3 subplot (BLER, MCS, Throughput)  × 3개
  - 통합 3×3 격자 그래프
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_PATH = Path(__file__).parent / "results" / "simulated_baseline_data.json"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
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


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return np.array(raw["ue_counts"]), raw["results"]


def plot_single_scenario(ue_counts, data, scenario_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Scenario: {scenario_name}",
                 fontsize=15, fontweight="bold", y=1.03)

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
            ax.set_ylim(bottom=0)
        elif key == "tput":
            ax.set_ylim(bottom=0)
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fname = FIG_DIR / f"sim_baseline_{scenario_name.replace('-', '_')}.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_combined(ue_counts, all_data):
    scenario_names = list(all_data.keys())
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        "Baseline Performance: Type 1 SU / Type 2 SU / Type 2 MU-MIMO\n"
        "(System Model Simulation — 4T4R, 106 PRB, FR1 Band 78)",
        fontsize=14, fontweight="bold", y=0.99)

    for row, sc_name in enumerate(scenario_names):
        data = all_data[sc_name]
        for col, (key, ylabel, transform) in enumerate(METRICS):
            ax = axes[row][col]
            for mode in MODES:
                y = transform(data[mode][key])
                y_std = transform(data[mode].get(f"{key}_std", [0]*len(y)))
                ax.plot(ue_counts, y,
                        color=COLORS[mode], marker=MARKERS[mode],
                        linestyle=LSTYLES[mode], label=mode,
                        markeredgecolor="white", markeredgewidth=0.8)

            ax.set_xscale("log", base=2)
            ax.set_xticks(ue_counts)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.minorticks_off()

            if row == 2:
                ax.set_xlabel("Number of UEs")
            if col == 0:
                ax.set_ylabel(sc_name, fontsize=12, fontweight="bold",
                              rotation=90, labelpad=18)
            if key in ("bler", "tput"):
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(ylabel, fontsize=12, pad=8)
            if row == 0 and col == 2:
                ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fname = FIG_DIR / "sim_baseline_combined_3x3.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading simulation data and generating figures...")
    ue_counts, results = load_data()
    for sc_name, data in results.items():
        plot_single_scenario(ue_counts, data, sc_name)
    plot_combined(ue_counts, results)
    print("Done.")
