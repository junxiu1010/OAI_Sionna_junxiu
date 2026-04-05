#!/usr/bin/env python3
"""
MU-MIMO Precoding Mismatch Post-Processing Analyzer.

Merges three data sources and produces comparison plots/statistics:
  1. OAI sideband CSV  (mu_mimo_sched.csv, mu_mimo_harq.csv)
  2. Proxy analyzer CSV (mu_mimo_analysis.csv)
  3. nrMAC_stats.log    (BLER / throughput summary)

Usage:
    python analyze_results.py --log-dir ./logs/20260401_run1/
    python analyze_results.py --sched mu_mimo_sched.csv --harq mu_mimo_harq.csv \
                              --analysis mu_mimo_analysis.csv --stats nrMAC_stats.log
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found; plots will be skipped")


# ─── CSV Loaders ───

def load_sched_csv(path: str) -> list:
    """Load OAI sideband scheduling CSV."""
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["frame"] = int(r["frame"])
            r["slot"] = int(r["slot"])
            r["pm_index"] = int(r["pm_index"])
            r["mcs"] = int(r["mcs"])
            r["rb_start"] = int(r["rb_start"])
            r["rb_size"] = int(r["rb_size"])
            r["n_layers"] = int(r["n_layers"])
            r["cqi"] = int(r["cqi"])
            r["ri"] = int(r["ri"])
            r["is_mu_mimo"] = r["is_mu_mimo"] in ("1", "True", "true")
            r["is_secondary"] = r["is_secondary"] in ("1", "True", "true")
            r["is_retx"] = r["is_retx"] in ("1", "True", "true")
            r["harq_round"] = int(r["harq_round"])
            r["tb_size"] = int(r["tb_size"])
            rows.append(r)
    return rows


def load_harq_csv(path: str) -> list:
    """Load OAI sideband HARQ CSV."""
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["frame"] = int(r["frame"])
            r["slot"] = int(r["slot"])
            r["ack"] = r["ack"] in ("1", "True", "true")
            r["harq_round"] = int(r["harq_round"])
            rows.append(r)
    return rows


def load_analysis_csv(path: str) -> list:
    """Load proxy analyzer CSV."""
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in r:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            r["frame"] = int(r["frame"])
            r["slot"] = int(r["slot"])
            rows.append(r)
    return rows


def parse_mac_stats(path: str) -> dict:
    """Parse last snapshot of nrMAC_stats.log for BLER/MCS per UE."""
    text = Path(path).read_text()
    stats = {}
    rnti_pattern = re.compile(r"UE\s+RNTI\s+(0x[0-9a-fA-F]+)")
    dl_pattern = re.compile(
        r"dlsch_rounds\s+([\d/]+),\s+dlsch_errors\s+(\d+).*?"
        r"BLER\s+([\d.]+).*?MCS\s+(\d+)",
        re.DOTALL
    )
    blocks = text.split("UE RNTI")
    for block in blocks[1:]:
        m = re.match(r"\s+(0x[0-9a-fA-F]+)", block)
        if not m:
            continue
        rnti = m.group(1)
        dm = dl_pattern.search(block)
        if dm:
            rounds_str = dm.group(1)
            rounds = [int(x) for x in rounds_str.split("/")]
            stats[rnti] = {
                "dl_rounds": rounds,
                "dl_errors": int(dm.group(2)),
                "dl_bler": float(dm.group(3)),
                "dl_mcs": int(dm.group(4)),
            }
    return stats


# ─── Analysis Functions ───

def compute_sched_statistics(sched: list) -> dict:
    """Compute per-UE and overall scheduling statistics."""
    per_ue = defaultdict(lambda: {
        "total_sched": 0, "mu_mimo_sched": 0,
        "pmi_dist": defaultdict(int),
        "mcs_values": [], "cqi_values": [],
        "total_rbs": 0, "total_tb_bytes": 0,
        "retx_count": 0,
    })

    for r in sched:
        rnti = r["rnti"]
        s = per_ue[rnti]
        s["total_sched"] += 1
        if r["is_mu_mimo"]:
            s["mu_mimo_sched"] += 1
        s["pmi_dist"][r["pm_index"]] += 1
        s["mcs_values"].append(r["mcs"])
        s["cqi_values"].append(r["cqi"])
        s["total_rbs"] += r["rb_size"]
        s["total_tb_bytes"] += r["tb_size"]
        if r["is_retx"]:
            s["retx_count"] += 1

    summary = {}
    for rnti, s in per_ue.items():
        n = s["total_sched"]
        summary[rnti] = {
            "total_sched": n,
            "mu_mimo_ratio": s["mu_mimo_sched"] / max(n, 1),
            "pmi_distribution": dict(s["pmi_dist"]),
            "avg_mcs": np.mean(s["mcs_values"]) if s["mcs_values"] else 0,
            "avg_cqi": np.mean(s["cqi_values"]) if s["cqi_values"] else 0,
            "total_rbs": s["total_rbs"],
            "total_tb_MB": s["total_tb_bytes"] / 1e6,
            "retx_ratio": s["retx_count"] / max(n, 1),
        }
    return summary


def compute_harq_statistics(harq: list) -> dict:
    """Compute per-UE HARQ ACK/NACK statistics."""
    per_ue = defaultdict(lambda: {"ack": 0, "nack": 0, "total": 0})
    for r in harq:
        rnti = r["rnti"]
        per_ue[rnti]["total"] += 1
        if r["ack"]:
            per_ue[rnti]["ack"] += 1
        else:
            per_ue[rnti]["nack"] += 1

    summary = {}
    for rnti, s in per_ue.items():
        summary[rnti] = {
            "total": s["total"],
            "ack": s["ack"],
            "nack": s["nack"],
            "bler_measured": s["nack"] / max(s["total"], 1),
        }
    return summary


def compute_sinr_comparison(analysis: list) -> dict:
    """Compute SINR gap statistics from analyzer CSV."""
    sinr_zf_all = []
    sinr_mmse_all = []
    sinr_best_pmi_all = []
    chordal_zf_all = []
    chordal_mmse_all = []
    corr_all = []

    for r in analysis:
        sinr_zf_all.append((r.get("sinr_zf_i_dB", -30), r.get("sinr_zf_j_dB", -30)))
        sinr_mmse_all.append((r.get("sinr_mmse_i_dB", -30), r.get("sinr_mmse_j_dB", -30)))

        # Find best PMI SINR among the 4 combos
        best_sum = -999
        best_pair = (r.get("sinr_pmi1_i_dB", -30), r.get("sinr_pmi1_j_dB", -30))
        for p in range(1, 5):
            si = r.get(f"sinr_pmi{p}_i_dB", -30)
            sj = r.get(f"sinr_pmi{p}_j_dB", -30)
            if si + sj > best_sum:
                best_sum = si + sj
                best_pair = (si, sj)
        sinr_best_pmi_all.append(best_pair)

        chordal_zf_all.append(r.get("chordal_dist_zf_best_pmi", 0))
        chordal_mmse_all.append(r.get("chordal_dist_mmse_best_pmi", 0))
        corr_all.append(r.get("channel_corr", 0))

    sinr_zf = np.array(sinr_zf_all)
    sinr_mmse = np.array(sinr_mmse_all)
    sinr_pmi = np.array(sinr_best_pmi_all)
    chordal_zf = np.array(chordal_zf_all)
    chordal_mmse = np.array(chordal_mmse_all)
    corr = np.array(corr_all)

    # SINR gap: ideal - PMI (positive = PMI is worse)
    gap_zf = sinr_zf - sinr_pmi
    gap_mmse = sinr_mmse - sinr_pmi

    return {
        "n_samples": len(analysis),
        "sinr_zf_mean_dB": sinr_zf.mean(axis=0).tolist(),
        "sinr_mmse_mean_dB": sinr_mmse.mean(axis=0).tolist(),
        "sinr_pmi_mean_dB": sinr_pmi.mean(axis=0).tolist(),
        "sinr_gap_zf_mean_dB": gap_zf.mean(axis=0).tolist(),
        "sinr_gap_mmse_mean_dB": gap_mmse.mean(axis=0).tolist(),
        "sinr_gap_zf_p95_dB": np.percentile(gap_zf, 95, axis=0).tolist(),
        "sinr_gap_mmse_p95_dB": np.percentile(gap_mmse, 95, axis=0).tolist(),
        "chordal_dist_zf_mean": float(chordal_zf.mean()),
        "chordal_dist_mmse_mean": float(chordal_mmse.mean()),
        "channel_correlation_mean": float(corr.mean()),
        "channel_correlation_p90": float(np.percentile(corr, 90)),
        "_raw": {
            "sinr_zf": sinr_zf,
            "sinr_mmse": sinr_mmse,
            "sinr_pmi": sinr_pmi,
            "gap_zf": gap_zf,
            "gap_mmse": gap_mmse,
            "chordal_zf": chordal_zf,
            "chordal_mmse": chordal_mmse,
            "corr": corr,
        }
    }


# ─── Plotting ───

def plot_sinr_cdf(raw: dict, output_dir: str):
    """Plot CDF of SINR for ZF, MMSE, and best PMI."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, label in enumerate(["UE_i", "UE_j"]):
        ax = axes[idx]
        for key, name, color in [
            ("sinr_zf", "ZF (Ideal)", "tab:blue"),
            ("sinr_mmse", "MMSE (Ideal)", "tab:green"),
            ("sinr_pmi", "Best PMI (Type-I)", "tab:red"),
        ]:
            vals = np.sort(raw[key][:, idx])
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("CDF")
        ax.set_title(f"SINR CDF - {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sinr_cdf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


def plot_sinr_gap(raw: dict, output_dir: str):
    """Plot SINR gap (ideal - PMI) CDF."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, label in enumerate(["UE_i", "UE_j"]):
        ax = axes[idx]
        for key, name, color in [
            ("gap_zf", "ZF - PMI", "tab:blue"),
            ("gap_mmse", "MMSE - PMI", "tab:green"),
        ]:
            vals = np.sort(raw[key][:, idx])
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=name, color=color, linewidth=1.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("SINR Gap (dB)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Precoding Mismatch SINR Gap - {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sinr_gap_cdf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


def plot_chordal_distance(raw: dict, output_dir: str):
    """Plot chordal distance histogram."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw["chordal_zf"], bins=50, alpha=0.7, label="ZF vs Best PMI",
            color="tab:blue", density=True)
    ax.hist(raw["chordal_mmse"], bins=50, alpha=0.7, label="MMSE vs Best PMI",
            color="tab:green", density=True)
    ax.set_xlabel("Chordal Distance")
    ax.set_ylabel("Density")
    ax.set_title("Precoding Matrix Chordal Distance (Ideal vs PMI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "chordal_distance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


def plot_correlation_vs_gap(raw: dict, output_dir: str):
    """Scatter plot of channel correlation vs SINR gap."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    gap_mean = raw["gap_zf"].mean(axis=1)
    ax.scatter(raw["corr"], gap_mean, alpha=0.3, s=10, color="tab:blue")
    ax.set_xlabel("Channel Correlation (|h1^H h2|^2 / ||h1||^2||h2||^2)")
    ax.set_ylabel("Mean SINR Gap: ZF - PMI (dB)")
    ax.set_title("Channel Correlation vs Precoding Mismatch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_vs_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


def plot_mcs_cqi_timeline(sched: list, output_dir: str):
    """Time series of MCS and CQI per UE."""
    if not HAS_MPL or not sched:
        return

    per_ue = defaultdict(lambda: {"t": [], "mcs": [], "cqi": [], "mu": []})
    for r in sched:
        rnti = r["rnti"]
        t = r["frame"] * 20 + r["slot"]
        per_ue[rnti]["t"].append(t)
        per_ue[rnti]["mcs"].append(r["mcs"])
        per_ue[rnti]["cqi"].append(r["cqi"])
        per_ue[rnti]["mu"].append(1 if r["is_mu_mimo"] else 0)

    n_ues = len(per_ue)
    fig, axes = plt.subplots(n_ues, 1, figsize=(14, 3 * n_ues), squeeze=False)

    for i, (rnti, data) in enumerate(sorted(per_ue.items())):
        ax = axes[i, 0]
        t = np.array(data["t"])
        ax.plot(t, data["mcs"], label="MCS", color="tab:blue", linewidth=0.8)
        ax.plot(t, data["cqi"], label="CQI", color="tab:orange", linewidth=0.8)

        mu_t = t[np.array(data["mu"]) == 1]
        if len(mu_t) > 0:
            ax.scatter(mu_t, [0] * len(mu_t), color="tab:red", s=5,
                       label="MU-MIMO", zorder=5)
        ax.set_ylabel("MCS / CQI")
        ax.set_title(f"UE {rnti}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Slot Index (frame*20 + slot)")
    plt.tight_layout()
    path = os.path.join(output_dir, "mcs_cqi_timeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] {path}")


# ─── Main ───

def main():
    ap = argparse.ArgumentParser(description="MU-MIMO Precoding Mismatch Post-Processor")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="Directory containing all log files (auto-detect)")
    ap.add_argument("--sched", type=str, default=None, help="mu_mimo_sched.csv path")
    ap.add_argument("--harq", type=str, default=None, help="mu_mimo_harq.csv path")
    ap.add_argument("--analysis", type=str, default=None, help="mu_mimo_analysis.csv path")
    ap.add_argument("--stats", type=str, default=None, help="nrMAC_stats.log path")
    ap.add_argument("--output-dir", type=str, default=None, help="Output directory for plots/JSON")
    args = ap.parse_args()

    if args.log_dir:
        ld = args.log_dir
        if not args.sched:
            args.sched = os.path.join(ld, "mu_mimo_sched.csv")
        if not args.harq:
            args.harq = os.path.join(ld, "mu_mimo_harq.csv")
        if not args.analysis:
            args.analysis = os.path.join(ld, "mu_mimo_analysis.csv")
        if not args.stats:
            args.stats = os.path.join(ld, "nrMAC_stats.log")

    output_dir = args.output_dir or args.log_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    report = {"generated_at": str(np.datetime64("now"))}

    # 1. Scheduling statistics
    if args.sched and os.path.exists(args.sched):
        print(f"\n[1/4] Loading scheduling data: {args.sched}")
        sched = load_sched_csv(args.sched)
        sched_stats = compute_sched_statistics(sched)
        report["scheduling"] = sched_stats
        print(f"  Loaded {len(sched)} scheduling entries")
        for rnti, s in sched_stats.items():
            print(f"  {rnti}: sched={s['total_sched']}, MU-MIMO={s['mu_mimo_ratio']:.1%}, "
                  f"avgMCS={s['avg_mcs']:.1f}, avgCQI={s['avg_cqi']:.1f}, "
                  f"retx={s['retx_ratio']:.1%}")
        plot_mcs_cqi_timeline(sched, output_dir)
    else:
        sched = []
        print("[1/4] Scheduling CSV not found, skipping")

    # 2. HARQ statistics
    if args.harq and os.path.exists(args.harq):
        print(f"\n[2/4] Loading HARQ data: {args.harq}")
        harq = load_harq_csv(args.harq)
        harq_stats = compute_harq_statistics(harq)
        report["harq"] = harq_stats
        print(f"  Loaded {len(harq)} HARQ entries")
        for rnti, s in harq_stats.items():
            print(f"  {rnti}: total={s['total']}, ACK={s['ack']}, NACK={s['nack']}, "
                  f"BLER={s['bler_measured']:.3f}")
    else:
        print("[2/4] HARQ CSV not found, skipping")

    # 3. SINR comparison
    if args.analysis and os.path.exists(args.analysis):
        print(f"\n[3/4] Loading analyzer data: {args.analysis}")
        analysis = load_analysis_csv(args.analysis)
        if analysis:
            sinr_stats = compute_sinr_comparison(analysis)
            raw = sinr_stats.pop("_raw")
            report["sinr_comparison"] = sinr_stats
            print(f"  Loaded {sinr_stats['n_samples']} analysis samples")
            print(f"  SINR ZF mean  (dB): {sinr_stats['sinr_zf_mean_dB']}")
            print(f"  SINR MMSE mean(dB): {sinr_stats['sinr_mmse_mean_dB']}")
            print(f"  SINR PMI mean (dB): {sinr_stats['sinr_pmi_mean_dB']}")
            print(f"  Gap ZF-PMI mean(dB): {sinr_stats['sinr_gap_zf_mean_dB']}")
            print(f"  Gap ZF-PMI p95 (dB): {sinr_stats['sinr_gap_zf_p95_dB']}")
            print(f"  Chordal dist ZF: {sinr_stats['chordal_dist_zf_mean']:.4f}")
            print(f"  Channel corr mean: {sinr_stats['channel_correlation_mean']:.4f}")

            plot_sinr_cdf(raw, output_dir)
            plot_sinr_gap(raw, output_dir)
            plot_chordal_distance(raw, output_dir)
            plot_correlation_vs_gap(raw, output_dir)
        else:
            print("  No analysis data found")
    else:
        print("[3/4] Analysis CSV not found, skipping")

    # 4. MAC stats
    if args.stats and os.path.exists(args.stats):
        print(f"\n[4/4] Loading MAC stats: {args.stats}")
        mac_stats = parse_mac_stats(args.stats)
        report["mac_stats"] = mac_stats
        for rnti, s in mac_stats.items():
            print(f"  {rnti}: BLER={s['dl_bler']:.3f}, MCS={s['dl_mcs']}, "
                  f"errors={s['dl_errors']}")
    else:
        print("[4/4] MAC stats not found, skipping")

    # Save JSON report
    json_path = os.path.join(output_dir, "mu_mimo_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Done] Report saved to {json_path}")
    print(f"[Done] Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
