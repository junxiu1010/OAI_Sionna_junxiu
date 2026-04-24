#!/usr/bin/env python3
"""
csinet_nmse_comparison.png — fulldata 재학습 실측 데이터 (2026-04-24)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "/workspace/graduation/csinet/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

GAMMA_LABELS = ["1/4", "1/8", "1/16", "1/32"]

# ── Baseline (fulldata, POWER_FLOOR=1e-30, all measured) ──
baseline = {
    "UMi_LOS":  [-25.51, -20.44, -15.77, -11.38],
    "UMi_NLOS": [-23.74, -19.73, -14.55,  -9.66],
    "UMa_NLOS": [-15.75, -10.70,  -7.02,  -4.46],
}

# ── Conditioned (layer-by-layer transfer, 2-phase, all measured) ──
conditioned = {
    "UMi_LOS":  [-28.30, -21.92, -16.32, -11.55],
    "UMi_NLOS": [-25.46, -20.60, -14.75,  -9.73],
    "UMa_NLOS": [-15.85, -10.84,  -7.05,  -4.48],
}

type2 = {"UMi_LOS": -3.7, "UMi_NLOS": -4.4, "UMa_NLOS": -4.2}

scenarios = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
display = {"UMi_LOS": "UMi-LOS", "UMi_NLOS": "UMi-NLOS", "UMa_NLOS": "UMa-NLOS"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle("CSI Compression: NMSE vs Compression Ratio",
             fontsize=15, fontweight="bold", y=1.01)

x = np.arange(4)
bw = 0.32

for idx, sc in enumerate(scenarios):
    ax = axes[idx]
    bl = baseline[sc]
    cd = conditioned[sc]
    t2 = type2[sc]

    bars_bl = ax.bar(x - bw/2, bl, bw, color="#2166ac", alpha=0.85,
           label="CsiNet Baseline", edgecolor="white", linewidth=0.5, zorder=3)
    bars_cd = ax.bar(x + bw/2, cd, bw, color="#e41a1c", alpha=0.85,
           label="Conditioned CsiNet", edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(t2, color="#4daf4a", ls="--", lw=1.8, alpha=0.75,
               label=f"Type 2 (L=2, 8PSK): {t2:.1f} dB", zorder=2)

    for i in range(4):
        gain = cd[i] - bl[i]
        if abs(gain) >= 0.05:
            y_pos = cd[i] - 0.6
            ax.text(x[i] + bw/2, y_pos, f"+{abs(gain):.1f}",
                    ha="center", va="top", fontsize=7, fontweight="bold",
                    color="#b2182b")

    lo = min(min(bl), min(cd))
    y_lo = np.floor(lo / 2.5) * 2.5 - 2.5
    ax.set_ylim(y_lo, 0)
    ax.set_yticks(np.arange(0, y_lo - 0.1, -2.5))

    ax.set_xlabel("Compression Ratio (γ)")
    if idx == 0:
        ax.set_ylabel("NMSE (dB)")
    ax.set_title(display[sc], fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(GAMMA_LABELS)

    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc",
              fancybox=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
fname = os.path.join(FIG_DIR, "csinet_nmse_comparison.png")
fig.savefig(fname, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fname}")
