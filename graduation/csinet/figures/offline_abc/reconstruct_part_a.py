#!/usr/bin/env python3
"""
Reconstruct Part A freshness plot analytically from known fresh NMSE values.
Uses the same Jakes channel aging model as eval_offline_abc.py.
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def bessel_j0(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x)
    term = np.ones_like(x)
    for k in range(1, 25):
        term = term * (-(x / 2) ** 2 / k ** 2)
        result = result + term
    return float(result) if result.ndim == 0 else result

FC_HZ = 3.5e9
C_MPS = 3e8
T_SLOT_S = 0.5e-3

NT = 4
NC_PRIME = 32
GAMMA = 0.25
M = int(NT * NC_PRIME * GAMMA)
BITS_PER_FLOAT = 32
COV_LATENT = 32
PDP_LATENT = 16
COND_DIM = COV_LATENT + PDP_LATENT

FULL_BITS = (M + COND_DIM) * BITS_PER_FLOAT
DIFF_BITS = M * BITS_PER_FLOAT
BASE_BITS = M * BITS_PER_FLOAT

BUDGET_VALUES = [50, 90, 180, 360, 512, 1024, 1792]
SPEEDS_KMH = [3, 30, 120]

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]

FRESH_NMSE = {
    "UMi_LOS":  {"cond_dB": -18.0, "base_dB": -15.0},
    "UMi_NLOS": {"cond_dB": -11.0, "base_dB": -3.0},
    "UMa_NLOS": {"cond_dB": -9.5,  "base_dB": -5.5},
}


def jakes_alpha(speed_kmh):
    v_mps = speed_kmh / 3.6
    f_d = v_mps * FC_HZ / C_MPS
    return float(bessel_j0(2 * np.pi * f_d * T_SLOT_S))


def avg_aged_nmse(nmse_fresh_lin, T, alpha):
    total = 0.0
    for k in range(T):
        rho_sq = alpha ** (2 * k)
        total += 1.0 - (1.0 - nmse_fresh_lin) * rho_sq
    return total / T


def compute_all():
    results = {}
    for sc in SCENARIOS:
        cond_fresh = 10 ** (FRESH_NMSE[sc]["cond_dB"] / 10)
        base_fresh = 10 ** (FRESH_NMSE[sc]["base_dB"] / 10)

        scenario_results = {}
        for speed in SPEEDS_KMH:
            alpha = jakes_alpha(speed)
            speed_results = {}

            for budget_B in BUDGET_VALUES:
                T_full = max(1, math.ceil(FULL_BITS / budget_B))
                T_diff = max(1, math.ceil(DIFF_BITS / budget_B))
                T_base = max(1, math.ceil(BASE_BITS / budget_B))

                aged_full = avg_aged_nmse(cond_fresh, T_full, alpha)
                aged_diff = avg_aged_nmse(cond_fresh, T_diff, alpha)
                aged_base = avg_aged_nmse(base_fresh, T_base, alpha)

                speed_results[budget_B] = {
                    "budget_bits_per_slot": budget_B,
                    "speed_kmh": speed,
                    "jakes_alpha": alpha,
                    "T_csi_full": T_full,
                    "T_csi_diff": T_diff,
                    "T_csi_base": T_base,
                    "freshness_gain": T_full / max(T_diff, 1),
                    "nmse_cond_fresh_dB": FRESH_NMSE[sc]["cond_dB"],
                    "nmse_base_fresh_dB": FRESH_NMSE[sc]["base_dB"],
                    "nmse_full_dB": float(10 * np.log10(aged_full + 1e-12)),
                    "nmse_diff_dB": float(10 * np.log10(aged_diff + 1e-12)),
                    "nmse_base_dB": float(10 * np.log10(aged_base + 1e-12)),
                    "nmse_gain_dB": float(
                        10 * np.log10(aged_full + 1e-12)
                        - 10 * np.log10(aged_diff + 1e-12)),
                }

            scenario_results[f"{speed}kmh"] = speed_results
        results[sc] = scenario_results

    return results


def plot_combined(results, out_dir):
    speeds = sorted(set(
        k for sc in results.values() for k in sc.keys()),
        key=lambda x: int(x.replace("kmh", "")))

    fig, axes = plt.subplots(len(speeds), len(SCENARIOS),
                             figsize=(6 * len(SCENARIOS), 4.5 * len(speeds)),
                             squeeze=False)
    fig.suptitle("Part A: Analytical Channel Aging — "
                 "Same Overhead Budget → Effective NMSE\n"
                 "(Jakes model: NMSE_eff(k) = "
                 "1 − (1 − NMSE_fresh)·|ρ(k)|²)",
                 fontsize=13)

    for ri, spd_key in enumerate(speeds):
        spd_val = int(spd_key.replace("kmh", ""))
        for si, sc in enumerate(SCENARIOS):
            ax = axes[ri][si]
            if spd_key not in results[sc]:
                continue
            r = results[sc][spd_key]
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

            ax.set_xlabel("Overhead Budget (bits/slot)", fontsize=10)
            ax.set_ylabel("Avg Effective NMSE (dB)", fontsize=10)
            ax.set_title(f"{sc}  @  {spd_val} km/h", fontsize=11)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "part_a_freshness_nmse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined: {path}")


def plot_per_scenario(results, out_dir):
    SPEED_LABELS = {"3kmh": "3 km/h", "30kmh": "30 km/h", "120kmh": "120 km/h"}
    SCENARIO_DISPLAY = {
        "UMi_LOS": "UMi-LOS",
        "UMi_NLOS": "UMa-LOS",
        "UMa_NLOS": "UMa-NLOS",
    }

    for sc in SCENARIOS:
        sc_data = results.get(sc, {})
        available_speeds = [s for s in ["3kmh", "30kmh", "120kmh"] if s in sc_data]
        n_speeds = len(available_speeds)
        if n_speeds == 0:
            continue

        display_name = SCENARIO_DISPLAY.get(sc, sc)
        fig, axes = plt.subplots(1, n_speeds, figsize=(6 * n_speeds, 5), squeeze=False)
        fig.suptitle(f"Channel Aging Analysis — {display_name}\n"
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
        out_path = os.path.join(out_dir, f"part_a_freshness_{sc}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Per-scenario: {out_path}")


if __name__ == "__main__":
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    results = compute_all()

    json_path = os.path.join(OUT_DIR, "part_a_freshness.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON: {json_path}")

    plot_combined(results, OUT_DIR)
    plot_per_scenario(results, OUT_DIR)
