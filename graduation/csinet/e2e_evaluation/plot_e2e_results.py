#!/usr/bin/env python3
"""
Phase 5: End-to-End 성능 비교 그래프 생성 (모두 막대 도표)
============================================================
"""

import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
FIG_DIR = os.environ.get("CSINET_FIG_DIR", "/workspace/csinet_figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

MODES = ["Type 1 SU", "Type 2 SU", "Type 2 MU", "CsiNet MU", "Cond-CsiNet MU"]
COLORS = {
    "Type 1 SU": "#636363",
    "Type 2 SU": "#2166ac",
    "Type 2 MU": "#4daf4a",
    "CsiNet MU": "#e41a1c",
    "Cond-CsiNet MU": "#ff7f00",
}
SCENARIOS_DATA = ["UMi_LOS", "UMi_NLOS", "UMa_LOS", "UMa_NLOS"]
SCENARIO_DISPLAY = {"UMi_LOS": "UMi-LOS", "UMi_NLOS": "UMi-NLOS", "UMa_LOS": "UMa-LOS", "UMa_NLOS": "UMa-NLOS"}
UE_COUNTS = [1, 2, 4, 8, 16]
GAMMAS = [0.25, 0.125, 0.0625, 0.03125]
GAMMA_LABELS = ["1/4", "1/8", "1/16", "1/32"]


def load_e2e_data():
    with open(os.path.join(RESULTS_DIR, "e2e_5mode_results.json")) as f:
        return json.load(f)


def plot_5mode_bar():
    """3x3 grid of grouped bar charts: rows=metrics, cols=scenarios."""
    data = load_e2e_data()
    metrics = ["bler_mean", "mcs_mean", "throughput_mean"]
    ylabels = ["BLER (%)", "Average MCS Index", "Cell Throughput (Mbps)"]

    fig, axes = plt.subplots(3, 4, figsize=(26, 15))
    fig.suptitle("End-to-End 5-Mode Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.01)

    n_modes = len(MODES)
    bar_w = 0.15
    x = np.arange(len(UE_COUNTS))

    for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col, sc in enumerate(SCENARIOS_DATA):
            ax = axes[row, col]
            for mi, mode in enumerate(MODES):
                y = [data[sc][mode][str(n)][metric] for n in UE_COUNTS]
                if metric == "bler_mean":
                    y = [v * 100 for v in y]
                ax.bar(x + mi * bar_w, y, bar_w, color=COLORS[mode],
                       alpha=0.85, label=mode, edgecolor="white", linewidth=0.5)

            ax.set_xlabel("Number of UEs")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x + bar_w * (n_modes - 1) / 2)
            ax.set_xticklabels(UE_COUNTS)
            if row == 0:
                ax.set_title(SCENARIO_DISPLAY[sc], fontsize=12, fontweight="bold")
            if col == 0 and row == 0:
                ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "e2e_5mode_combined.png")
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_throughput_bar():
    """Bar chart: throughput comparison per scenario."""
    data = load_e2e_data()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Cell Throughput Comparison (5 CSI Modes)",
                 fontsize=14, fontweight="bold")

    n_modes = len(MODES)
    bar_w = 0.15
    x = np.arange(len(UE_COUNTS))

    for idx, sc in enumerate(SCENARIOS_DATA):
        ax = axes[idx]
        for mi, mode in enumerate(MODES):
            y = [data[sc][mode][str(n)]["throughput_mean"] for n in UE_COUNTS]
            ax.bar(x + mi * bar_w, y, bar_w, color=COLORS[mode],
                   alpha=0.85, label=mode, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Cell Throughput (Mbps)")
        ax.set_title(SCENARIO_DISPLAY[sc], fontsize=12, fontweight="bold")
        ax.set_xticks(x + bar_w * (n_modes - 1) / 2)
        ax.set_xticklabels(UE_COUNTS)
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "e2e_throughput_bar.png")
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_csinet_nmse_bar():
    """Grouped bar chart: NMSE per compression ratio, per scenario."""
    csinet_path = os.path.join(RESULTS_DIR, "csinet_evaluation.json")
    if not os.path.exists(csinet_path):
        print("  No CsiNet evaluation data, skipping NMSE plot")
        return
    with open(csinet_path) as f:
        csinet_results = json.load(f)

    cond_results = []
    cond_path = os.path.join(RESULTS_DIR, "conditioned_evaluation.json")
    if os.path.exists(cond_path):
        try:
            with open(cond_path) as f:
                cond_results = json.load(f)
        except json.JSONDecodeError:
            pass

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("CSI Compression: NMSE vs Compression Ratio",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(GAMMAS))
    bar_w = 0.25

    for idx, sc in enumerate(SCENARIOS_DATA):
        ax = axes[idx]

        bl_data = [r for r in csinet_results if r["scenario"] == sc]
        bl_nmse = []
        type2_val = None
        for g in GAMMAS:
            match = [r for r in bl_data if abs(r["gamma"] - g) < 0.001]
            bl_nmse.append(match[0]["nmse_dB"] if match else 0)
            if type2_val is None and match:
                type2_val = match[0].get("type2_nmse_dB")

        cd_data = [r for r in cond_results if r["scenario"] == sc]
        cd_nmse = []
        for g in GAMMAS:
            match = [r for r in cd_data if abs(r["gamma"] - g) < 0.001]
            cd_nmse.append(match[0]["conditioned_nmse_dB"] if match else 0)

        ax.bar(x - bar_w / 2, bl_nmse, bar_w, color="#2166ac",
               alpha=0.85, label="CsiNet Baseline", edgecolor="white", linewidth=0.5)
        ax.bar(x + bar_w / 2, cd_nmse, bar_w, color="#e41a1c",
               alpha=0.85, label="Conditioned CsiNet", edgecolor="white", linewidth=0.5)

        if type2_val is not None:
            ax.axhline(type2_val, color="gray", ls="--", lw=1.5,
                       label="Type 2 (L=2, 8PSK)")

        ax.set_xlabel("Compression Ratio (γ)")
        ax.set_ylabel("NMSE (dB)")
        ax.set_title(SCENARIO_DISPLAY[sc], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(GAMMA_LABELS)
        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "csinet_nmse_comparison.png")
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_csinet_gain_bar():
    """Grouped bar chart: throughput gain (%) over Type 2 MU per UE count."""
    data = load_e2e_data()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Throughput Gain vs Type 2 MU-MIMO",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(UE_COUNTS))
    bar_w = 0.30
    gain_modes = ["CsiNet MU", "Cond-CsiNet MU"]
    gain_colors = [COLORS["CsiNet MU"], COLORS["Cond-CsiNet MU"]]

    for idx, sc in enumerate(SCENARIOS_DATA):
        ax = axes[idx]
        baseline_tp = [data[sc]["Type 2 MU"][str(n)]["throughput_mean"] for n in UE_COUNTS]

        for mi, mode in enumerate(gain_modes):
            tp = [data[sc][mode][str(n)]["throughput_mean"] for n in UE_COUNTS]
            gain = [(t - b) / max(b, 0.1) * 100 for t, b in zip(tp, baseline_tp)]
            ax.bar(x + (mi - 0.5) * bar_w, gain, bar_w, color=gain_colors[mi],
                   alpha=0.85, label=mode, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Throughput Gain (%)")
        ax.set_title(SCENARIO_DISPLAY[sc], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(UE_COUNTS)
        ax.legend(framealpha=0.9)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "csinet_throughput_gain.png")
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_5mode_per_scenario():
    """Per-scenario 1x3 bar charts (BLER, MCS, Throughput)."""
    data = load_e2e_data()
    metrics = ["bler_mean", "mcs_mean", "throughput_mean"]
    ylabels = ["BLER (%)", "Average MCS Index", "Cell Throughput (Mbps)"]

    n_modes = len(MODES)
    bar_w = 0.15
    x = np.arange(len(UE_COUNTS))

    for sc in SCENARIOS_DATA:
        if sc not in data:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
        fig.suptitle(f"End-to-End 5-Mode Performance — {SCENARIO_DISPLAY[sc]}\n"
                     "(OAI-Sionna Calibrated, 4T4R, 40 MHz, FR1 Band 78)",
                     fontsize=13, fontweight="bold", y=0.99)

        for col, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[col]
            for mi, mode in enumerate(MODES):
                y = [data[sc][mode][str(n)][metric] for n in UE_COUNTS]
                if metric == "bler_mean":
                    y = [v * 100 for v in y]
                ax.bar(x + mi * bar_w, y, bar_w, color=COLORS[mode],
                       alpha=0.85, label=mode, edgecolor="white", linewidth=0.5)

            ax.set_xlabel("Number of UEs")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x + bar_w * (n_modes - 1) / 2)
            ax.set_xticklabels(UE_COUNTS)
            ax.grid(True, alpha=0.3, axis="y")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(MODES),
                   fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.88))

        fig.tight_layout(rect=[0, 0, 1, 0.82])
        safe = sc.replace("-", "_")
        fname = os.path.join(FIG_DIR, f"e2e_5mode_{safe}.png")
        fig.savefig(fname)
        print(f"  Saved: {fname}")
        plt.close(fig)


if __name__ == "__main__":
    print("Generating E2E evaluation plots (bar charts)...")
    plot_5mode_bar()
    plot_5mode_per_scenario()
    plot_throughput_bar()
    plot_csinet_nmse_bar()
    plot_csinet_gain_bar()
    print("All plots generated.")
