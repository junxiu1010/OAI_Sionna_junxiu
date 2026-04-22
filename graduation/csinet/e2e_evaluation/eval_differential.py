#!/usr/bin/env python3
"""
Differential Encoding Evaluation for Conditioned CsiNet
=========================================================
Simulates temporal CSI feedback with delta-encoded conditioning vectors.

Compares 6 modes:
  1) Full-cond:      send full 24-dim conditioning vector every slot
  2) Delta-th=0.01:  conservative threshold
  3) Delta-th=0.05:  moderate threshold
  4) Delta-th=0.1:   aggressive threshold
  5) Fixed-period:   send full conditioning every N slots, cache otherwise
  6) Baseline:       no conditioning at all (0 overhead from stats)

Outputs per scenario × speed:
  - NMSE (dB) over time
  - Average feedback overhead (bits/slot)
  - Overhead savings (%) vs Full-cond
  - Update rate (fraction of slots with conditioning update)

Usage:
  python eval_differential.py [--data-dir DIR] [--ckpt-dir DIR] [--out-dir DIR]
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
from integration.differential_cond import (
    DifferentialConditioner, FixedPeriodConditioner,
)

NT = 4
NR = 2
NC_PRIME = 32
COV_LATENT = 16
PDP_LATENT = 8
COND_DIM = COV_LATENT + PDP_LATENT
COV_DIM = NT * NR
COV_VEC_DIM = COV_DIM * (COV_DIM + 1)
PDP_DIM = COV_VEC_DIM
GAMMA = 0.25
M = int(NT * NC_PRIME * GAMMA)
BITS_PER_FLOAT = 32

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
SPEEDS_KMH = [0, 3, 30]

FIXED_PERIOD_N = 10


def nmse_dB(x_true, x_hat):
    """Per-sample NMSE in dB. Reduces all dims except batch (axis 0)."""
    reduce_axes = tuple(range(1, x_true.ndim))
    mse = np.mean(np.abs(x_true - x_hat) ** 2, axis=reduce_axes)
    power = np.mean(np.abs(x_true) ** 2, axis=reduce_axes)
    nmse = mse / np.maximum(power, 1e-12)
    return 10.0 * np.log10(nmse + 1e-12)


def load_dataset(data_dir, scenario):
    """Load preprocessed dataset with location indices."""
    path = os.path.join(data_dir, f"preprocessed_{scenario}.h5")
    with h5py.File(path, "r") as f:
        X_test = f["X_test"][:].astype(np.float32)
        R_H = f["R_H"][:].astype(np.complex64)
        PDP = f["PDP"][:].astype(np.float32)
        loc_test = f["loc_idx_test"][:]
    return X_test, R_H, PDP, loc_test


def build_temporal_sequence(X_test, R_H, PDP, loc_test, speed_kmh,
                            n_slots=200, rng=None):
    """Build a time-ordered sequence of (x, R_H_idx, pdp_idx) tuples.

    Simulates a single UE moving through different locations.
    speed_kmh controls how often the location (and thus statistics) changes.

    At 0 km/h: same location throughout.
    At 3 km/h: location changes every ~30 slots.
    At 30 km/h: location changes every ~3 slots.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unique_locs = np.unique(loc_test)
    n_locs = len(unique_locs)
    n_samples = len(X_test)

    if speed_kmh == 0:
        loc_id = rng.choice(unique_locs)
        loc_ids = np.full(n_slots, loc_id)
    else:
        change_interval = max(1, int(30 / max(speed_kmh, 0.1) * 10))
        loc_ids = np.empty(n_slots, dtype=int)
        current_loc_idx = rng.integers(0, n_locs)
        for t in range(n_slots):
            if t > 0 and t % change_interval == 0:
                step = rng.integers(-2, 3)
                current_loc_idx = np.clip(current_loc_idx + step, 0, n_locs - 1)
            loc_ids[t] = unique_locs[current_loc_idx]

    loc_to_samples = {}
    for i, lid in enumerate(loc_test):
        loc_to_samples.setdefault(int(lid), []).append(i)

    sequence = []
    for t in range(n_slots):
        lid = int(loc_ids[t])
        candidates = loc_to_samples.get(lid, list(range(n_samples)))
        idx = rng.choice(candidates)
        sequence.append({
            "x": X_test[idx],
            "r_h_idx": lid,
            "pdp_idx": lid,
            "sample_idx": idx,
        })

    return sequence, loc_ids


def prepare_cond_inputs(R_H, PDP, loc_id, noise_std=0.0, rng=None):
    """Get (r_vec, pdp) for a given location index.

    noise_std > 0 adds small perturbation to simulate gradual environmental
    changes within the same location (e.g., scatterer motion, weather).
    """
    if loc_id < len(R_H):
        r = R_H[loc_id].copy()
    else:
        r = R_H[0].copy()
    if loc_id < len(PDP):
        pdp = PDP[loc_id].copy()
    else:
        pdp = PDP[0].copy()

    if noise_std > 0 and rng is not None:
        r_noise = rng.normal(0, noise_std, r.shape) + 1j * rng.normal(0, noise_std, r.shape)
        r = r + r_noise.astype(r.dtype)
        pdp = pdp + np.abs(rng.normal(0, noise_std * np.mean(np.abs(pdp)), pdp.shape)).astype(pdp.dtype)

    return r, pdp


def load_models(ckpt_dir, scenario):
    """Load stat_ae, conditioned csinet, and baseline csinet.

    Build order matters: baseline first (right after clear_session) so that
    Keras auto-generated layer names match the saved checkpoint exactly.
    stat_ae and cond_model use load_weights_by_structure which handles
    name mismatches.
    """
    tf.keras.backend.clear_session()

    from integration.csinet_engine import load_weights_by_structure

    baseline_model = CsiNetAutoencoder(NT, NC_PRIME, GAMMA)
    baseline_model.build(input_shape=(None, 2, NT, NC_PRIME))
    base_tag = f"{scenario}_gamma{GAMMA:.4f}"
    base_ckpt = os.path.join(ckpt_dir, f"csinet_{base_tag}_best.weights.h5")
    if os.path.exists(base_ckpt):
        baseline_model.load_weights(base_ckpt)
        print(f"  [baseline] Loaded weights from {base_ckpt}")
    else:
        print(f"  WARNING: baseline checkpoint not found: {base_ckpt}")

    stat_ae = StatisticsAutoencoder(COV_VEC_DIM, PDP_DIM, COV_LATENT, PDP_LATENT)
    stat_ae([tf.zeros([1, COV_VEC_DIM]), tf.zeros([1, PDP_DIM])])
    stat_ckpt = os.path.join(ckpt_dir, f"stat_ae_{scenario}.weights.h5")
    if os.path.exists(stat_ckpt):
        load_weights_by_structure(stat_ae, stat_ckpt)
    else:
        print(f"  WARNING: stat_ae checkpoint not found: {stat_ckpt}")

    cond_model = ConditionedCsiNet(NT, NC_PRIME, GAMMA, COND_DIM)
    cond_model([tf.zeros([1, 2, NT, NC_PRIME]), tf.zeros([1, COND_DIM])])
    cond_tag = f"{scenario}_gamma{GAMMA:.4f}"
    cond_ckpt = os.path.join(ckpt_dir, f"cond_csinet_{cond_tag}_best.weights.h5")
    if os.path.exists(cond_ckpt):
        load_weights_by_structure(cond_model, cond_ckpt)
    else:
        print(f"  WARNING: cond_csinet checkpoint not found: {cond_ckpt}")

    return stat_ae, cond_model, baseline_model


def run_mode(mode_name, sequence, stat_ae, cond_model, baseline_model,
             R_H_all, PDP_all, threshold=None, fixed_period=None,
             speed_kmh=0, rng=None):
    """Run a single evaluation mode over the temporal sequence.

    Returns: dict with per-slot NMSE and overhead info.
    """
    n_slots = len(sequence)
    if rng is None:
        rng = np.random.default_rng(123)

    stat_noise_std = speed_kmh * 0.002

    if mode_name == "baseline":
        nmse_per_slot = []
        for t in range(n_slots):
            x = sequence[t]["x"]
            x_tf = tf.constant(x[np.newaxis])
            x_hat = baseline_model(x_tf, training=False).numpy()[0]
            nmse_per_slot.append(float(nmse_dB(x[np.newaxis], x_hat[np.newaxis])[0]))
        return {
            "nmse_per_slot": nmse_per_slot,
            "nmse_mean_dB": float(np.mean(nmse_per_slot)),
            "overhead_bits_per_slot": M * BITS_PER_FLOAT,
            "cond_overhead_bits_per_slot": 0,
            "total_overhead_bits_per_slot": M * BITS_PER_FLOAT,
            "overhead_savings_pct": 100.0,
            "update_rate": 0.0,
        }

    if mode_name.startswith("delta"):
        conditioner = DifferentialConditioner(
            cond_dim=COND_DIM, threshold=threshold, max_stale_slots=100)
    elif mode_name == "fixed_period":
        conditioner = FixedPeriodConditioner(
            cond_dim=COND_DIM, update_period=fixed_period)
    else:
        conditioner = None

    nmse_per_slot = []
    overhead_log = []

    for t in range(n_slots):
        x = sequence[t]["x"]
        loc_id = sequence[t]["r_h_idx"]
        r_h, pdp = prepare_cond_inputs(
            R_H_all, PDP_all, loc_id,
            noise_std=stat_noise_std, rng=rng)

        r_vec = vectorize_covariance(
            tf.constant(r_h[np.newaxis].astype(np.complex64)))
        pdp_tf = tf.constant(pdp[np.newaxis].astype(np.float32))
        c_full = stat_ae.get_condition_vector(r_vec, pdp_tf).numpy().flatten()

        x_tf = tf.constant(x[np.newaxis])

        if conditioner is not None:
            key = (0, 0)
            info = conditioner.update(key, c_full, codeword_dim=M)
            cond_enc = tf.constant(c_full[np.newaxis], dtype=tf.float32)
            z = cond_model.encoder(x_tf, cond_enc, training=False)
            cond_dec = tf.constant(info.c_used[np.newaxis], dtype=tf.float32)
            x_hat = cond_model.decoder(z, cond_dec, training=False).numpy()[0]
            overhead_log.append(info)
        else:
            cond_tf = tf.constant(c_full[np.newaxis], dtype=tf.float32)
            z = cond_model.encoder(x_tf, cond_tf, training=False)
            x_hat = cond_model.decoder(z, cond_tf, training=False).numpy()[0]

        nmse_per_slot.append(float(nmse_dB(x[np.newaxis], x_hat[np.newaxis])[0]))

    full_cond_overhead = COND_DIM * BITS_PER_FLOAT
    codeword_overhead = M * BITS_PER_FLOAT

    if conditioner is not None and overhead_log:
        n_updates = sum(1 for i in overhead_log if i.was_updated)
        cond_bits_total = sum(i.overhead_bits for i in overhead_log)
        avg_cond = cond_bits_total / n_slots
        update_rate = n_updates / n_slots
        savings = (1.0 - cond_bits_total / (n_slots * full_cond_overhead)) * 100
    else:
        avg_cond = full_cond_overhead
        update_rate = 1.0
        savings = 0.0

    return {
        "nmse_per_slot": nmse_per_slot,
        "nmse_mean_dB": float(np.mean(nmse_per_slot)),
        "overhead_bits_per_slot": codeword_overhead,
        "cond_overhead_bits_per_slot": avg_cond,
        "total_overhead_bits_per_slot": codeword_overhead + avg_cond,
        "overhead_savings_pct": savings,
        "update_rate": update_rate,
    }


MODE_CONFIGS = [
    ("full_cond",     {"mode_name": "full_cond"}),
    ("delta_th0.01",  {"mode_name": "delta_0.01", "threshold": 0.01}),
    ("delta_th0.05",  {"mode_name": "delta_0.05", "threshold": 0.05}),
    ("delta_th0.10",  {"mode_name": "delta_0.10", "threshold": 0.10}),
    ("fixed_period",  {"mode_name": "fixed_period", "fixed_period": FIXED_PERIOD_N}),
    ("baseline",      {"mode_name": "baseline"}),
]


def run_scenario_speed(scenario, speed_kmh, data_dir, ckpt_dir, n_slots=200):
    """Run all 6 modes for a given scenario and speed."""
    print(f"\n  Loading data: {scenario}")
    X_test, R_H, PDP, loc_test = load_dataset(data_dir, scenario)
    print(f"    X_test: {X_test.shape}, R_H: {R_H.shape}, PDP: {PDP.shape}")

    print(f"  Loading models...")
    stat_ae, cond_model, baseline_model = load_models(ckpt_dir, scenario)

    rng = np.random.default_rng(42 + hash(scenario) % 1000 + speed_kmh)
    sequence, loc_ids = build_temporal_sequence(
        X_test, R_H, PDP, loc_test, speed_kmh, n_slots=n_slots, rng=rng)

    results = {}
    for mode_label, mode_params in MODE_CONFIGS:
        print(f"    Running: {mode_label} (speed={speed_kmh} km/h)...", end="", flush=True)
        mode_rng = np.random.default_rng(
            42 + hash(scenario) % 1000 + speed_kmh + hash(mode_label) % 1000)
        r = run_mode(
            mode_name=mode_params["mode_name"],
            sequence=sequence,
            stat_ae=stat_ae, cond_model=cond_model,
            baseline_model=baseline_model,
            R_H_all=R_H, PDP_all=PDP,
            threshold=mode_params.get("threshold"),
            fixed_period=mode_params.get("fixed_period"),
            speed_kmh=speed_kmh, rng=mode_rng,
        )
        print(f" NMSE={r['nmse_mean_dB']:.2f} dB, "
              f"overhead={r['total_overhead_bits_per_slot']:.0f} bits/slot, "
              f"savings={r['overhead_savings_pct']:.1f}%")
        results[mode_label] = r

    return results


def print_summary_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print(f"{'Scenario':12s} {'Speed':>6s} {'Mode':16s} "
          f"{'NMSE(dB)':>9s} {'Overhead':>10s} {'Cond OH':>10s} "
          f"{'Savings':>8s} {'UpdRate':>8s}")
    print("=" * 110)

    for scenario in SCENARIOS:
        for speed in SPEEDS_KMH:
            key = f"{scenario}_{speed}kmh"
            if key not in all_results:
                continue
            r = all_results[key]
            for mode_label, _ in MODE_CONFIGS:
                if mode_label not in r:
                    continue
                d = r[mode_label]
                print(f"{scenario:12s} {speed:4d}km "
                      f"{mode_label:16s} "
                      f"{d['nmse_mean_dB']:9.2f} "
                      f"{d['total_overhead_bits_per_slot']:10.0f} "
                      f"{d['cond_overhead_bits_per_slot']:10.1f} "
                      f"{d['overhead_savings_pct']:7.1f}% "
                      f"{d['update_rate']:7.2f}")
            print("-" * 110)


def save_results(all_results, out_dir):
    """Save results to JSON (without per-slot arrays for compactness)."""
    summary = {}
    for key, modes in all_results.items():
        summary[key] = {}
        for mode_label, data in modes.items():
            summary[key][mode_label] = {
                k: v for k, v in data.items() if k != "nmse_per_slot"
            }
            summary[key][mode_label]["nmse_per_slot_first10"] = data["nmse_per_slot"][:10]

    path = os.path.join(out_dir, "differential_eval_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {path}")

    full_path = os.path.join(out_dir, "differential_eval_full.json")
    full_data = {}
    for key, modes in all_results.items():
        full_data[key] = {}
        for mode_label, data in modes.items():
            full_data[key][mode_label] = {
                k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in data.items()
            }
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"Full results saved: {full_path}")


def plot_results(all_results, out_dir):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    colors = {
        "full_cond": "#1f77b4",
        "delta_th0.01": "#ff7f0e",
        "delta_th0.05": "#2ca02c",
        "delta_th0.10": "#d62728",
        "fixed_period": "#9467bd",
        "baseline": "#8c564b",
    }
    labels = {
        "full_cond": "Full-Cond (every slot)",
        "delta_th0.01": "Delta (θ=0.01)",
        "delta_th0.05": "Delta (θ=0.05)",
        "delta_th0.10": "Delta (θ=0.10)",
        "fixed_period": f"Fixed-period (N={FIXED_PERIOD_N})",
        "baseline": "Baseline CsiNet",
    }

    for scenario in SCENARIOS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Differential Encoding — {scenario}", fontsize=14)

        for si, speed in enumerate(SPEEDS_KMH):
            ax = axes[si]
            key = f"{scenario}_{speed}kmh"
            if key not in all_results:
                continue

            for mode_label, _ in MODE_CONFIGS:
                if mode_label not in all_results[key]:
                    continue
                d = all_results[key][mode_label]
                nmse_ts = d["nmse_per_slot"]
                ax.plot(nmse_ts, label=labels.get(mode_label, mode_label),
                        color=colors.get(mode_label, "gray"), alpha=0.8, linewidth=1)

            ax.set_title(f"Speed = {speed} km/h")
            ax.set_xlabel("Slot index")
            ax.set_ylabel("NMSE (dB)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"diff_eval_nmse_vs_time_{scenario}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {fig_path}")

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6 * len(SCENARIOS), 5))
    if len(SCENARIOS) == 1:
        axes = [axes]
    fig.suptitle("Overhead vs NMSE Trade-off", fontsize=14)

    for si, scenario in enumerate(SCENARIOS):
        ax = axes[si]
        for speed in SPEEDS_KMH:
            key = f"{scenario}_{speed}kmh"
            if key not in all_results:
                continue
            x_vals, y_vals, mode_labels_plot = [], [], []
            for mode_label, _ in MODE_CONFIGS:
                if mode_label not in all_results[key]:
                    continue
                d = all_results[key][mode_label]
                x_vals.append(d["total_overhead_bits_per_slot"])
                y_vals.append(d["nmse_mean_dB"])
                mode_labels_plot.append(mode_label)
            ax.scatter(x_vals, y_vals, label=f"{speed} km/h", s=60, alpha=0.8)
            for i, ml in enumerate(mode_labels_plot):
                ax.annotate(ml, (x_vals[i], y_vals[i]),
                            fontsize=6, ha="left", va="bottom")

        ax.set_title(scenario)
        ax.set_xlabel("Total overhead (bits/slot)")
        ax.set_ylabel("NMSE (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "diff_eval_overhead_vs_nmse.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {fig_path}")

    fig, axes = plt.subplots(1, len(SPEEDS_KMH), figsize=(6 * len(SPEEDS_KMH), 5))
    if len(SPEEDS_KMH) == 1:
        axes = [axes]
    fig.suptitle("Overhead Savings vs NMSE Degradation", fontsize=14)

    for si, speed in enumerate(SPEEDS_KMH):
        ax = axes[si]
        for scenario in SCENARIOS:
            key = f"{scenario}_{speed}kmh"
            if key not in all_results:
                continue
            ref_nmse = all_results[key].get("full_cond", {}).get("nmse_mean_dB", 0)
            x_vals, y_vals = [], []
            for mode_label, _ in MODE_CONFIGS:
                if mode_label in ("full_cond", "baseline"):
                    continue
                if mode_label not in all_results[key]:
                    continue
                d = all_results[key][mode_label]
                savings = d["overhead_savings_pct"]
                nmse_deg = d["nmse_mean_dB"] - ref_nmse
                x_vals.append(savings)
                y_vals.append(nmse_deg)
            if x_vals:
                ax.plot(x_vals, y_vals, "o-", label=scenario, markersize=6)

        ax.set_title(f"Speed = {speed} km/h")
        ax.set_xlabel("Conditioning Overhead Savings (%)")
        ax.set_ylabel("NMSE Degradation vs Full-Cond (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "diff_eval_savings_vs_degradation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate differential encoding for Conditioned CsiNet")
    parser.add_argument("--data-dir", default="/workspace/csinet_datasets")
    parser.add_argument("--ckpt-dir", default="/workspace/csinet_checkpoints")
    parser.add_argument("--out-dir", default="/workspace/csinet_results/differential")
    parser.add_argument("--n-slots", type=int, default=200,
                        help="Number of temporal slots to simulate per experiment")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--speeds", nargs="+", type=int, default=SPEEDS_KMH)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Differential Encoding Evaluation — Conditioned CsiNet")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Speeds: {args.speeds} km/h")
    print(f"  Slots per experiment: {args.n_slots}")
    print(f"  Gamma: {GAMMA}, M: {M}, COND_DIM: {COND_DIM}")
    print("=" * 70)

    all_results = {}

    for scenario in args.scenarios:
        for speed in args.speeds:
            key = f"{scenario}_{speed}kmh"
            print(f"\n{'─'*60}")
            print(f"Experiment: {scenario}, speed={speed} km/h")
            print(f"{'─'*60}")

            try:
                results = run_scenario_speed(
                    scenario, speed, args.data_dir, args.ckpt_dir,
                    n_slots=args.n_slots)
                all_results[key] = results
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    if all_results:
        print_summary_table(all_results)
        save_results(all_results, args.out_dir)
        plot_results(all_results, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
