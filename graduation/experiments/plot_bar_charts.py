#!/usr/bin/env python3
"""
Convert calibrated performance line charts to grouped bar charts.
Generates one figure per scenario (UMi-LOS, UMa-NLOS, UMa-LOS).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).parent / "results" / "calibrated_performance_data.json"
FIG_DIR = Path(__file__).parent / "figures"

with open(DATA_PATH) as f:
    raw = json.load(f)

ue_counts = raw["ue_counts"]
results = raw["results"]

SCENARIOS = {
    "UMi-LOS":  "3GPP UMi-LOS (Street Canyon)",
    "UMi-NLOS": "3GPP UMi-NLOS (Street Canyon)",
    "UMa-LOS":  "3GPP UMa-LOS (Urban Macro)",
    "UMa-NLOS": "3GPP UMa-NLOS (Urban Macro)",
}

MODES = ["Type 1 SU-MIMO", "Type 2 SU-MIMO", "Type 2 MU-MIMO"]
COLORS = {"Type 1 SU-MIMO": "#2196F3", "Type 2 SU-MIMO": "#4CAF50", "Type 2 MU-MIMO": "#F44336"}
METRICS = [
    ("bler",  "BLER (%)",              100.0),
    ("mcs",   "Average MCS Index",     1.0),
    ("tput",  "Cell DL Throughput (Mbps)", 1.0),
]

x = np.arange(len(ue_counts))
n_modes = len(MODES)
bar_w = 0.25

for sc_key, sc_title in SCENARIOS.items():
    if sc_title is None:
        continue
    sc_data = results[sc_key]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{sc_title}\n(OAI-Sionna Calibrated, 4T4R, 40 MHz, FR1 Band 78)",
                 fontsize=13, fontweight="bold")

    for col, (metric_key, ylabel, scale) in enumerate(METRICS):
        ax = axes[col]
        for i, mode in enumerate(MODES):
            vals = np.array(sc_data[mode][metric_key]) * scale
            offset = (i - (n_modes - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   label=mode, color=COLORS[mode], alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Number of UEs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in ue_counts])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        if metric_key == "bler":
            ax.set_ylim(0, max(ax.get_ylim()[1], 5))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_name = f"calibrated_{sc_key.replace('-','_')}_bar.png"
    out_path = FIG_DIR / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
