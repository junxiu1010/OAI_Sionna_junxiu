#!/usr/bin/env python3
"""Part A freshness plot: one figure per scenario (3 speeds as rows)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "part_a_freshness.json")
OUT_DIR = os.path.dirname(__file__)

with open(DATA_PATH) as f:
    results = json.load(f)

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
SPEED_KEYS = ["3kmh", "30kmh", "120kmh"]
SPEED_LABELS = {"3kmh": "3 km/h", "30kmh": "30 km/h", "120kmh": "120 km/h"}

for sc in SCENARIOS:
    sc_data = results.get(sc, {})
    available_speeds = [s for s in SPEED_KEYS if s in sc_data]
    n_speeds = len(available_speeds)
    if n_speeds == 0:
        continue

    fig, axes = plt.subplots(1, n_speeds, figsize=(6 * n_speeds, 5), squeeze=False)
    fig.suptitle(f"Channel Aging Analysis — {sc}\n"
                 f"(Jakes model: NMSE_eff(k) = 1 − (1 − NMSE_fresh)·|ρ(k)|²)",
                 fontsize=13, y=1.02)

    for ci, spd_key in enumerate(available_speeds):
        ax = axes[0][ci]
        r = sc_data[spd_key]
        budgets = sorted(r.keys(), key=lambda x: int(x))
        B_vals = [int(b) for b in budgets]

        nmse_full = [r[b]["nmse_full_dB"] for b in budgets]
        nmse_diff = [r[b]["nmse_diff_dB"] for b in budgets]
        nmse_base = [r[b]["nmse_base_dB"] for b in budgets]

        ax.plot(B_vals, nmse_full, "s-", label="Full-Cond",
                color="#1f77b4", markersize=7, linewidth=2)
        ax.plot(B_vals, nmse_diff, "o-", label="Differential",
                color="#ff7f0e", markersize=7, linewidth=2)
        ax.plot(B_vals, nmse_base, "^--", label="Baseline CsiNet",
                color="#8c564b", markersize=6, alpha=0.7)

        ax.set_xlabel("Overhead Budget (bits/slot)", fontsize=11)
        ax.set_ylabel("Avg Effective NMSE (dB)", fontsize=11)
        ax.set_title(f"{SPEED_LABELS[spd_key]}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"part_a_freshness_{sc}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
