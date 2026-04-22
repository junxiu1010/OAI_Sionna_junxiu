#!/usr/bin/env python3
"""
Offline Performance Evaluation: Parts A, B1, B2
=================================================

Part A — Same-overhead H-freshness gain:
  With differential encoding saving conditioning overhead, UE can report
  H more frequently (higher freshness) at the same total bit budget.
  Measures NMSE improvement as CSI-RS period shrinks.

Part B1 — Statistics estimability:
  How many channel samples N are needed for accurate R_H / PDP estimation?
  Compares partial estimates (N ∈ {10, 50, 100, 500, 1000}) against the
  ground-truth computed from the full dataset.

Part B2 — Temporal stability:
  How fast do R_H and PDP change over time?
  Measures ||R_H(t) − R_H(t+Δt)|| for Δt ∈ {1, 10, 100, 1000} slots.
  Justifies using T_stat on the order of hundreds to thousands of slots.

Usage:
  python eval_offline_abc.py [--data-dir DIR] [--ckpt-dir DIR] [--out-dir DIR]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.csinet import CsiNetAutoencoder
from models.stat_autoencoder import StatisticsAutoencoder, vectorize_covariance
from models.conditioned_csinet import ConditionedCsiNet
from integration.csinet_engine import load_weights_by_structure

NT = 4
NR = 2
NC_PRIME = 32
COV_LATENT = 32
PDP_LATENT = 16
COND_DIM = COV_LATENT + PDP_LATENT
COV_DIM = NT * NR
COV_VEC_DIM = COV_DIM * (COV_DIM + 1)
PDP_DIM = COV_VEC_DIM
GAMMA = 0.25
M = int(NT * NC_PRIME * GAMMA)
BITS_PER_FLOAT = 32

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]


def nmse_linear(x_true, x_hat):
    reduce_axes = tuple(range(1, x_true.ndim))
    mse = np.mean(np.abs(x_true - x_hat) ** 2, axis=reduce_axes)
    power = np.mean(np.abs(x_true) ** 2, axis=reduce_axes)
    return mse / np.maximum(power, 1e-12)


def nmse_dB(x_true, x_hat):
    return 10.0 * np.log10(nmse_linear(x_true, x_hat) + 1e-12)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_preprocessed(data_dir, scenario):
    path = os.path.join(data_dir, f"preprocessed_{scenario}.h5")
    with h5py.File(path, "r") as f:
        X_test = f["X_test"][:].astype(np.float32)
        R_H = f["R_H"][:].astype(np.complex64)
        PDP = f["PDP"][:].astype(np.float32)
        loc_test = f["loc_idx_test"][:]
    return X_test, R_H, PDP, loc_test


def load_raw_H(data_dir, scenario, split="test"):
    path = os.path.join(data_dir, f"dataset_{scenario}.h5")
    with h5py.File(path, "r") as f:
        H = f[f"H_{split}"][:].astype(np.complex64)
        loc = f[f"loc_idx_{split}"][:]
        R_H = f["R_H"][:].astype(np.complex64)
        PDP = f["PDP"][:].astype(np.float32)
    return H, loc, R_H, PDP


def load_models(ckpt_dir, scenario):
    tf.keras.backend.clear_session()

    baseline = CsiNetAutoencoder(NT, NC_PRIME, GAMMA)
    baseline(tf.zeros([1, 2, NT, NC_PRIME]))
    bpath = os.path.join(ckpt_dir, f"csinet_{scenario}_gamma{GAMMA:.4f}_best.weights.h5")
    if os.path.exists(bpath):
        baseline.load_weights(bpath)

    stat_ae = StatisticsAutoencoder(COV_VEC_DIM, PDP_DIM, COV_LATENT, PDP_LATENT)
    stat_ae([tf.zeros([1, COV_VEC_DIM]), tf.zeros([1, PDP_DIM])])
    spath = os.path.join(ckpt_dir, f"stat_ae_{scenario}.weights.h5")
    if os.path.exists(spath):
        try:
            stat_ae.load_weights(spath)
        except Exception:
            load_weights_by_structure(stat_ae, spath)

    cond = ConditionedCsiNet(NT, NC_PRIME, GAMMA, COND_DIM)
    cond([tf.zeros([1, 2, NT, NC_PRIME]), tf.zeros([1, COND_DIM])])
    cpath = os.path.join(ckpt_dir, f"cond_csinet_{scenario}_gamma{GAMMA:.4f}_best.weights.h5")
    if os.path.exists(cpath):
        try:
            cond.load_weights(cpath)
        except Exception:
            load_weights_by_structure(cond, cpath)

    return baseline, stat_ae, cond


# ═══════════════════════════════════════════════════════════════════
# Part A — Same-overhead H-freshness NMSE gain (Jakes channel)
# ═══════════════════════════════════════════════════════════════════

FC_HZ = 3.5e9
C_MPS = 3e8
T_SLOT_S = 0.5e-3

try:
    from scipy.special import j0 as _bessel_j0
except ImportError:
    def _bessel_j0(x):
        x = np.asarray(x, dtype=np.float64)
        result = np.ones_like(x)
        term = np.ones_like(x)
        for k in range(1, 25):
            term = term * (-(x / 2) ** 2 / k ** 2)
            result = result + term
        return float(result) if result.ndim == 0 else result


def jakes_alpha(speed_kmh, fc=FC_HZ, t_slot=T_SLOT_S):
    v_mps = speed_kmh / 3.6
    f_d = v_mps * fc / C_MPS
    return float(_bessel_j0(2 * np.pi * f_d * t_slot))


def avg_aged_nmse(nmse_fresh_lin, T, alpha):
    """Average effective NMSE over a reporting period T.

    Uses the standard channel-aging model:
      NMSE_eff(k) = 1 - (1 - NMSE_fresh) * |ρ(k)|²
    where ρ(k) = α^k (Jakes autocorrelation) and k is the slot offset
    from the last CSI-RS report.
    """
    total = 0.0
    for k in range(T):
        rho_sq = alpha ** (2 * k)
        total += 1.0 - (1.0 - nmse_fresh_lin) * rho_sq
    return total / T


def run_part_a(data_dir, ckpt_dir, out_dir):
    """
    Same-overhead freshness comparison with analytical Jakes aging.

    1. Measure fresh reconstruction NMSE on real test samples.
    2. Apply the standard channel-aging formula:
         NMSE_eff(k) = 1 - (1 - NMSE_fresh) · |J_0(2πf_d·k·T_slot)|²
       to compute the average NMSE over each reporting period T.
    3. Differential encoding uses less bits/report → smaller T → less aging.
    """
    print("\n" + "=" * 70)
    print("Part A: Same-Overhead H-Freshness → NMSE Gain (Jakes Aging)")
    print("=" * 70)

    import math

    full_bits = (M + COND_DIM) * BITS_PER_FLOAT
    diff_bits = M * BITS_PER_FLOAT
    base_bits = M * BITS_PER_FLOAT

    budget_values = [50, 90, 180, 360, 512, 1024, 1792]
    SPEEDS_KMH = [3, 30, 120]
    BATCH = 64

    results = {}
    for scenario in SCENARIOS:
        print(f"\n  --- {scenario} ---")
        X_test, R_H, PDP, loc_test = load_preprocessed(data_dir, scenario)
        baseline_model, stat_ae, cond_model = load_models(ckpt_dir, scenario)

        n_test = len(X_test)
        unique_locs = np.unique(loc_test)

        base_out = np.empty_like(X_test)
        for s in range(0, n_test, BATCH):
            e = min(s + BATCH, n_test)
            base_out[s:e] = baseline_model(
                tf.constant(X_test[s:e]), training=False).numpy()
        nmse_base_fresh = float(np.mean(nmse_linear(X_test, base_out)))

        cond_out = np.empty_like(X_test)
        for loc_id in unique_locs:
            mask = loc_test == loc_id
            x_loc = tf.constant(X_test[mask])
            n_loc = int(mask.sum())
            r_h = R_H[min(int(loc_id), len(R_H) - 1)]
            pdp_v = PDP[min(int(loc_id), len(PDP) - 1)]
            r_vec = vectorize_covariance(tf.constant(r_h[np.newaxis]))
            c = stat_ae.get_condition_vector(
                r_vec, tf.constant(pdp_v[np.newaxis]))
            c_rep = tf.repeat(c, n_loc, axis=0)
            z = cond_model.encoder(x_loc, c_rep, training=False)
            cond_out[mask] = cond_model.decoder(
                z, c_rep, training=False).numpy()
        nmse_cond_fresh = float(np.mean(nmse_linear(X_test, cond_out)))

        print(f"    Fresh NMSE — Cond: {10*np.log10(nmse_cond_fresh+1e-12):.2f} dB, "
              f"Base: {10*np.log10(nmse_base_fresh+1e-12):.2f} dB")

        scenario_results = {}
        for speed in SPEEDS_KMH:
            alpha = jakes_alpha(speed)
            print(f"\n    Speed {speed} km/h  (α = {alpha:.6f})")
            speed_results = {}

            for budget_B in budget_values:
                T_full = max(1, math.ceil(full_bits / budget_B))
                T_diff = max(1, math.ceil(diff_bits / budget_B))
                T_base = max(1, math.ceil(base_bits / budget_B))

                aged_full = avg_aged_nmse(nmse_cond_fresh, T_full, alpha)
                aged_diff = avg_aged_nmse(nmse_cond_fresh, T_diff, alpha)
                aged_base = avg_aged_nmse(nmse_base_fresh, T_base, alpha)

                speed_results[budget_B] = {
                    "budget_bits_per_slot": budget_B,
                    "speed_kmh": speed,
                    "jakes_alpha": alpha,
                    "T_csi_full": T_full,
                    "T_csi_diff": T_diff,
                    "T_csi_base": T_base,
                    "freshness_gain": T_full / max(T_diff, 1),
                    "nmse_cond_fresh_dB": float(
                        10 * np.log10(nmse_cond_fresh + 1e-12)),
                    "nmse_base_fresh_dB": float(
                        10 * np.log10(nmse_base_fresh + 1e-12)),
                    "nmse_full_dB": float(
                        10 * np.log10(aged_full + 1e-12)),
                    "nmse_diff_dB": float(
                        10 * np.log10(aged_diff + 1e-12)),
                    "nmse_base_dB": float(
                        10 * np.log10(aged_base + 1e-12)),
                    "nmse_gain_dB": float(
                        10 * np.log10(aged_full + 1e-12)
                        - 10 * np.log10(aged_diff + 1e-12)),
                }
                print(f"      B={budget_B:5d}: T_full={T_full:3d}, "
                      f"T_diff={T_diff:3d} | "
                      f"NMSE full={10*np.log10(aged_full+1e-12):.2f}, "
                      f"diff={10*np.log10(aged_diff+1e-12):.2f}, "
                      f"base={10*np.log10(aged_base+1e-12):.2f} dB")

            scenario_results[f"{speed}kmh"] = speed_results

        results[scenario] = scenario_results

    with open(os.path.join(out_dir, "part_a_freshness.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_part_a(results, out_dir)
    return results


def _plot_part_a(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    speeds = sorted(set(
        k for sc in results.values() for k in sc.keys()),
        key=lambda x: int(x.replace("kmh", "")))

    fig, axes = plt.subplots(len(speeds), len(SCENARIOS),
                             figsize=(6 * len(SCENARIOS),
                                      4.5 * len(speeds)),
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
    print(f"  Plot: {path}")


# ═══════════════════════════════════════════════════════════════════
# Part B1 — Statistics estimability
# ═══════════════════════════════════════════════════════════════════

def _compute_rh_pdp(H_samples):
    """Compute R_H and PDP from channel samples — same formula as generate_dataset.py."""
    N, Nr, Nt, Nsc = H_samples.shape
    h_vec = H_samples.reshape(N, Nr * Nt, Nsc)
    R_H = np.zeros((Nr * Nt, Nr * Nt), dtype=np.complex64)
    for sc in range(Nsc):
        h_sc = h_vec[:, :, sc]
        R_H += (h_sc.conj().T @ h_sc) / N
    R_H /= Nsc

    h_delay = np.fft.ifft(H_samples, axis=-1)
    pdp = np.mean(np.abs(h_delay) ** 2, axis=(0, 1, 2))
    return R_H, pdp


def run_part_b1(data_dir, out_dir):
    """
    Estimate R_H and PDP from N random channel samples,
    compare against an independent reference.

    Ground truth: computed from ALL splits (train+val+test = 60 samples/loc).
    Estimation:   N samples drawn from train-only pool (50 samples/loc).
    This ensures the reference and estimation pools are independent,
    so even at N=50 the error is non-zero.
    """
    print("\n" + "=" * 70)
    print("Part B1: Statistics Estimability — R_H / PDP vs Sample Count N")
    print("=" * 70)

    N_VALUES = [5, 10, 20, 30, 50]
    N_TRIALS = 30
    N_LOCS_EVAL = 50

    results = {}
    for scenario in SCENARIOS:
        print(f"\n  --- {scenario} ---")
        H_tr, loc_tr, _, _ = load_raw_H(data_dir, scenario, split="train")
        H_val, loc_val, _, _ = load_raw_H(data_dir, scenario, split="val")
        H_te, loc_te, _, _ = load_raw_H(data_dir, scenario, split="test")

        H_full = np.concatenate([H_tr, H_val, H_te], axis=0)
        loc_full = np.concatenate([loc_tr, loc_val, loc_te], axis=0)

        train_loc_to_idx = {}
        for i, lid in enumerate(loc_tr):
            train_loc_to_idx.setdefault(int(lid), []).append(i)

        full_loc_to_idx = {}
        for i, lid in enumerate(loc_full):
            full_loc_to_idx.setdefault(int(lid), []).append(i)

        unique_locs = sorted(set(int(l) for l in loc_tr))
        n_train_per_loc = min(len(v) for v in train_loc_to_idx.values())
        n_full_per_loc = min(len(full_loc_to_idx[l]) for l in unique_locs[:N_LOCS_EVAL])
        print(f"    Train samples per location: {n_train_per_loc}")
        print(f"    Total samples per location (reference): {n_full_per_loc}")
        print(f"    Evaluating {min(N_LOCS_EVAL, len(unique_locs))} locations × {N_TRIALS} trials")

        ref_rh = {}
        ref_pdp = {}
        for loc_id in unique_locs[:N_LOCS_EVAL]:
            idx = full_loc_to_idx[loc_id]
            H_loc = H_full[idx]
            r, p = _compute_rh_pdp(H_loc)
            ref_rh[loc_id] = r
            ref_pdp[loc_id] = p

        rng = np.random.default_rng(42)
        scenario_results = {}

        for N in N_VALUES:
            rh_errors = []
            pdp_errors = []

            for trial in range(N_TRIALS):
                for loc_id in unique_locs[:N_LOCS_EVAL]:
                    train_indices = train_loc_to_idx[loc_id]

                    need_replace = (N > len(train_indices))
                    indices_sub = rng.choice(
                        train_indices, size=N, replace=need_replace)
                    H_sub = H_tr[indices_sub]

                    R_est, pdp_est = _compute_rh_pdp(H_sub)

                    gt_r = ref_rh[loc_id]
                    gt_pdp = ref_pdp[loc_id]

                    r_err = np.linalg.norm(R_est - gt_r) / max(np.linalg.norm(gt_r), 1e-12)
                    p_err = np.linalg.norm(pdp_est - gt_pdp) / max(np.linalg.norm(gt_pdp), 1e-12)
                    rh_errors.append(float(r_err))
                    pdp_errors.append(float(p_err))

            scenario_results[N] = {
                "r_h_nmse_mean": float(np.mean(rh_errors)),
                "r_h_nmse_std": float(np.std(rh_errors)),
                "r_h_nmse_median": float(np.median(rh_errors)),
                "pdp_nmse_mean": float(np.mean(pdp_errors)),
                "pdp_nmse_std": float(np.std(pdp_errors)),
                "pdp_nmse_median": float(np.median(pdp_errors)),
                "n_measurements": len(rh_errors),
                "train_per_loc": n_train_per_loc,
                "ref_per_loc": n_full_per_loc,
            }
            print(f"    N={N:5d}: R_H err={np.mean(rh_errors):.4f}±{np.std(rh_errors):.4f}  "
                  f"PDP err={np.mean(pdp_errors):.4f}±{np.std(pdp_errors):.4f}  "
                  f"(median R_H={np.median(rh_errors):.4f}, PDP={np.median(pdp_errors):.4f})")

        results[scenario] = scenario_results

    with open(os.path.join(out_dir, "part_b1_estimability.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_part_b1(results, out_dir)
    return results


def _plot_part_b1(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Part B1: Statistics Estimation Error vs Sample Count N\n"
                 "(Reference = all splits combined; Estimation = train-only subset)",
                 fontsize=12)

    colors = {"UMi_LOS": "#1f77b4", "UMi_NLOS": "#ff7f0e", "UMa_NLOS": "#2ca02c"}
    markers = {"UMi_LOS": "o", "UMi_NLOS": "s", "UMa_NLOS": "^"}

    for sc in SCENARIOS:
        r = results[sc]
        if not r:
            continue
        N_vals = sorted(r.keys(), key=lambda x: int(x))
        ns = [int(n) for n in N_vals]

        rh_mean = [r[n]["r_h_nmse_mean"] for n in N_vals]
        pdp_mean = [r[n]["pdp_nmse_mean"] for n in N_vals]

        axes[0].plot(ns, rh_mean, marker=markers[sc],
                     color=colors[sc], label=sc, linewidth=2, markersize=7)
        axes[1].plot(ns, pdp_mean, marker=markers[sc],
                     color=colors[sc], label=sc, linewidth=2, markersize=7)

    for ax, title in zip(axes, ["Covariance R_H", "Power Delay Profile (PDP)"]):
        ax.set_xlabel("Number of Samples N", fontsize=11)
        ax.set_ylabel("Normalized Estimation Error", fontsize=11)
        ax.set_title(f"{title} Estimation Error", fontsize=12)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(out_dir, "part_b1_estimability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")


# ═══════════════════════════════════════════════════════════════════
# Part B2 — Temporal stability of R_H and PDP
# ═══════════════════════════════════════════════════════════════════

def run_part_b2(data_dir, out_dir):
    """
    Measure temporal stability of R_H and PDP.

    Uses pre-computed per-location ground-truth statistics (R_H_gt, PDP_gt).
    Simulates a UE moving at different speeds: the location index changes
    over time, and we measure the normalized difference in R_H/PDP between
    the location at time t and time t+Δt.

    This directly quantifies how often the conditioning vector needs updating.
    """
    print("\n" + "=" * 70)
    print("Part B2: Temporal Stability — ||R_H(t) − R_H(t+Δt)||")
    print("=" * 70)

    DELTA_T_VALUES = [1, 5, 10, 50, 100, 500, 1000]
    SPEEDS_KMH = [0, 3, 30]

    results = {}
    for scenario in SCENARIOS:
        print(f"\n  --- {scenario} ---")
        _, _, R_H_gt, PDP_gt = load_raw_H(data_dir, scenario, split="train")
        n_locs = len(R_H_gt)

        scenario_results = {}
        for speed in SPEEDS_KMH:
            rng = np.random.default_rng(42 + speed)

            n_time_steps = 5000
            if speed == 0:
                change_interval = n_time_steps + 1
            elif speed <= 3:
                change_interval = 100
            elif speed <= 30:
                change_interval = 10
            else:
                change_interval = 3

            loc_seq = np.empty(n_time_steps, dtype=int)
            current_loc_idx = rng.integers(0, n_locs)
            for t in range(n_time_steps):
                if t > 0 and t % change_interval == 0:
                    step = rng.integers(-1, 2)
                    current_loc_idx = np.clip(current_loc_idx + step, 0, n_locs - 1)
                loc_seq[t] = current_loc_idx

            speed_results = {}
            for dt in DELTA_T_VALUES:
                rh_diffs = []
                pdp_diffs = []

                n_pairs = min(500, n_time_steps - dt)
                t_starts = rng.choice(n_time_steps - dt, size=n_pairs, replace=False)

                for t0 in t_starts:
                    loc1 = loc_seq[int(t0)]
                    loc2 = loc_seq[int(t0 + dt)]

                    R1, R2 = R_H_gt[loc1], R_H_gt[loc2]
                    P1, P2 = PDP_gt[loc1], PDP_gt[loc2]

                    avg_r_norm = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
                    avg_p_norm = (np.linalg.norm(P1) + np.linalg.norm(P2)) / 2

                    r_diff = np.linalg.norm(R1 - R2) / max(avg_r_norm, 1e-12)
                    p_diff = np.linalg.norm(P1 - P2) / max(avg_p_norm, 1e-12)
                    rh_diffs.append(float(r_diff))
                    pdp_diffs.append(float(p_diff))

                speed_results[dt] = {
                    "r_h_change_mean": float(np.mean(rh_diffs)),
                    "r_h_change_std": float(np.std(rh_diffs)),
                    "r_h_change_median": float(np.median(rh_diffs)),
                    "r_h_change_p95": float(np.percentile(rh_diffs, 95)),
                    "pdp_change_mean": float(np.mean(pdp_diffs)),
                    "pdp_change_std": float(np.std(pdp_diffs)),
                    "pdp_change_median": float(np.median(pdp_diffs)),
                    "pdp_change_p95": float(np.percentile(pdp_diffs, 95)),
                    "fraction_above_0.01": float(np.mean(np.array(rh_diffs) > 0.01)),
                    "fraction_above_0.05": float(np.mean(np.array(rh_diffs) > 0.05)),
                    "fraction_above_0.10": float(np.mean(np.array(rh_diffs) > 0.10)),
                    "n_pairs": len(rh_diffs),
                }
                print(f"    speed={speed:2d}km/h  Δt={dt:5d}: "
                      f"R_H Δ={np.mean(rh_diffs):.4f} (med={np.median(rh_diffs):.4f})  "
                      f"PDP Δ={np.mean(pdp_diffs):.4f} (med={np.median(pdp_diffs):.4f})  "
                      f">0.01: {np.mean(np.array(rh_diffs) > 0.01)*100:.0f}%  "
                      f">0.05: {np.mean(np.array(rh_diffs) > 0.05)*100:.0f}%")

            scenario_results[f"{speed}kmh"] = speed_results

        results[scenario] = scenario_results

    with open(os.path.join(out_dir, "part_b2_stability.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_part_b2(results, out_dir)
    return results


def _plot_part_b2(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    speeds = [0, 3, 30]
    colors_speed = {0: "#1f77b4", 3: "#ff7f0e", 30: "#d62728"}
    markers_speed = {0: "o", 3: "s", 30: "^"}

    fig, axes = plt.subplots(1, len(SCENARIOS),
                             figsize=(6 * len(SCENARIOS), 5), squeeze=False)
    fig.suptitle("Part B2: Temporal Stability of Channel Statistics", fontsize=14)

    metric = "r_h_change_mean"
    label = "||ΔR_H|| / ||R_H||"

    for si, sc in enumerate(SCENARIOS):
        ax = axes[0][si]
        for speed in speeds:
            key = f"{speed}kmh"
            if key not in results[sc]:
                continue
            r = results[sc][key]
            dt_vals = sorted(r.keys(), key=lambda x: int(x))
            dts = [int(d) for d in dt_vals]
            means = [r[d][metric] for d in dt_vals]

            ax.plot(dts, means,
                    marker=markers_speed[speed],
                    color=colors_speed[speed],
                    label=f"{speed} km/h",
                    linewidth=2, markersize=7)

        ax.set_xlabel("Time Interval Δt (slots)", fontsize=11)
        ax.set_ylabel(f"Normalized Change {label}", fontsize=10)
        ax.set_title(f"{sc} — {label}", fontsize=11)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(out_dir, "part_b2_temporal_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("R_H Temporal Stability (Mean Change) — All Scenarios × Speeds",
                 fontsize=13)
    line_styles = {"UMi_LOS": "-", "UMi_NLOS": "--", "UMa_NLOS": "-."}

    for sc in SCENARIOS:
        for speed in speeds:
            key = f"{speed}kmh"
            if key not in results[sc]:
                continue
            r = results[sc][key]
            dt_vals = sorted(r.keys(), key=lambda x: int(x))
            dts = [int(d) for d in dt_vals]
            means = [r[d]["r_h_change_mean"] for d in dt_vals]
            ax.plot(dts, means, marker=markers_speed[speed],
                    color=colors_speed[speed], linestyle=line_styles[sc],
                    label=f"{sc} @ {speed}km/h", markersize=6, linewidth=1.5)

    ax.set_xlabel("Time Interval Δt (slots)", fontsize=12)
    ax.set_ylabel("Normalized R_H Change", fontsize=12)
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(out_dir, "part_b2_rh_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluations: A (freshness), B1 (estimability), B2 (stability)")
    parser.add_argument("--data-dir", default="/workspace/graduation/csinet/datasets")
    parser.add_argument("--ckpt-dir", default="/workspace/csinet_checkpoints")
    parser.add_argument("--out-dir", default="/workspace/csinet_results/offline_abc")
    parser.add_argument("--parts", nargs="+", default=["a", "b1", "b2"],
                        choices=["a", "b1", "b2"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Offline Performance Evaluation — Parts A, B1, B2")
    print(f"  Parts: {args.parts}")
    print(f"  Scenarios: {SCENARIOS}")
    print("=" * 70)

    all_results = {}

    if "a" in args.parts:
        all_results["part_a"] = run_part_a(args.data_dir, args.ckpt_dir, args.out_dir)

    if "b1" in args.parts:
        all_results["part_b1"] = run_part_b1(args.data_dir, args.out_dir)

    if "b2" in args.parts:
        all_results["part_b2"] = run_part_b2(args.data_dir, args.out_dir)

    print("\n" + "=" * 70)
    print("All evaluations complete.")
    print(f"Results in: {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
