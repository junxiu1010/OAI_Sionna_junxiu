#!/usr/bin/env python3
"""
Type-I vs Type-II PMI Verification via Normalized Channel Correlation.

Reads binary CSI channel logs produced by the instrumented OAI UE
(csi_rx.c with CSI_CHANNEL_LOG env set), reconstructs the precoding
vector W from the reported PMI, and computes the normalized correlation
    rho = |h^H w| / (||h|| ||w||)
averaged over active subcarriers and RX antennas.

Usage:
    python verify_type2_pmi.py <log1.bin> [log2.bin ...] [--output fig.png]
    python verify_type2_pmi.py results/type1_snr*.bin results/type2_snr*.bin

The script auto-detects Type-I / Type-II from each record and groups
results by (pmi_type, snr_tag) extracted from the filename.
"""

import argparse
import struct
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

MAGIC = 0x43534932  # "CSI2"
HEADER_SIZE = 64

WB_AMP_TABLE = np.array([
    1.0,
    1.0 / np.sqrt(2),
    0.5,
    1.0 / (2.0 * np.sqrt(2)),
    0.25,
    1.0 / (4.0 * np.sqrt(2)),
    0.125,
    0.0,
])

QPSK_PHASE = np.array([0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
PSK8_PHASE = np.array([i * np.pi / 4 for i in range(8)])


def parse_header(buf):
    """Parse a 64-byte record header into a dict."""
    if len(buf) < HEADER_SIZE:
        return None
    magic = struct.unpack_from("<I", buf, 0)[0]
    if magic != MAGIC:
        return None

    h = {}
    h["record_size"] = struct.unpack_from("<I", buf, 4)[0]
    h["frame"] = struct.unpack_from("<I", buf, 8)[0]
    h["slot"] = struct.unpack_from("<I", buf, 12)[0]
    h["N_ports"] = struct.unpack_from("<H", buf, 16)[0]
    h["N_rx"] = struct.unpack_from("<H", buf, 18)[0]
    h["ofdm_size"] = struct.unpack_from("<I", buf, 20)[0]
    h["first_carrier"] = struct.unpack_from("<I", buf, 24)[0]
    h["start_rb"] = struct.unpack_from("<H", buf, 28)[0]
    h["nr_of_rbs"] = struct.unpack_from("<H", buf, 30)[0]
    h["mem_offset"] = buf[32]
    h["rank"] = buf[33]
    h["cqi"] = buf[34]
    h["pmi_type"] = buf[35]  # 0 = Type-I, 1 = Type-II

    h["i1"] = buf[36]
    h["i2"] = buf[37]
    h["port_sel_indicator"] = buf[38]
    h["strongest_coeff0"] = buf[39]
    h["strongest_coeff1"] = buf[40]
    h["num_beams"] = buf[41]
    h["port_sel_d"] = buf[42]
    h["phase_alphabet"] = buf[43]
    h["wb_amp_l0"] = list(buf[44:52])
    h["wb_amp_l1"] = list(buf[52:60])
    h["num_subbands"] = buf[60]
    return h


def read_records(filepath):
    """Generator yielding (header_dict, H_est, subband_phases) per record."""
    data = Path(filepath).read_bytes()
    pos = 0
    while pos + HEADER_SIZE <= len(data):
        hdr = parse_header(data[pos : pos + HEADER_SIZE])
        if hdr is None:
            pos += 1
            continue

        rec_size = hdr["record_size"]
        if pos + rec_size > len(data):
            break

        N_rx = hdr["N_rx"]
        N_ports = hdr["N_ports"]
        ofdm_size = hdr["ofdm_size"]
        ch_offset = pos + HEADER_SIZE
        ch_count = N_rx * N_ports * ofdm_size
        ch_bytes = ch_count * 4  # c16_t = 2 x int16

        if ch_offset + ch_bytes > len(data):
            break

        raw = np.frombuffer(data[ch_offset : ch_offset + ch_bytes], dtype=np.int16)
        raw_c = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
        H_est = raw_c.reshape(N_rx, N_ports, ofdm_size)

        sb_phases = None
        if hdr["pmi_type"] == 1 and hdr["num_subbands"] > 0:
            tc = 2 * hdr["num_beams"]
            nl = hdr["rank"] + 1
            if nl > 2:
                nl = 2
            sb_count = hdr["num_subbands"] * tc * nl
            sb_start = ch_offset + ch_bytes
            if sb_start + sb_count <= len(data):
                sb_raw = np.frombuffer(
                    data[sb_start : sb_start + sb_count], dtype=np.uint8
                )
                sb_phases = sb_raw.reshape(hdr["num_subbands"], tc, nl)

        pos += rec_size
        yield hdr, H_est, sb_phases


def reconstruct_type1(hdr):
    """Build (N_ports,) precoding vector for Type-I (2-port codebook, rank 1).

    OAI Type-I uses only ports 0 and 1 even if N_ports > 2.
    The remaining ports are zero-padded.
    """
    N_ports = hdr["N_ports"]
    i2 = hdr["i2"]
    phi = QPSK_PHASE[i2 & 0x3]
    W = np.zeros(N_ports, dtype=np.complex128)
    W[0] = 1.0
    W[1] = np.exp(1j * phi)
    W[:2] /= np.sqrt(2.0)
    return W


def reconstruct_type2(hdr, sb_phases):
    """Build (N_ports,) precoding vector for Type-II Port Selection (rank 1, WB)."""
    N_ports = hdr["N_ports"]
    d = hdr["port_sel_d"]
    L = hdr["num_beams"]
    pa = hdr["phase_alphabet"]
    sel_start = hdr["port_sel_indicator"] * d

    total_coeffs = 2 * L
    if total_coeffs > 2 * d:
        total_coeffs = 2 * d

    phase_table = PSK8_PHASE if pa == 8 else QPSK_PHASE

    W = np.zeros(N_ports, dtype=np.complex128)
    for c in range(total_coeffs):
        port_local = c % d
        pol = c // d
        ant_idx = sel_start + port_local + pol * (N_ports // 2)
        if ant_idx >= N_ports:
            continue

        amp_idx = hdr["wb_amp_l0"][c] if c < 8 else 0
        if amp_idx > 7:
            amp_idx = 7
        amp = WB_AMP_TABLE[amp_idx]

        phase = 0.0
        if sb_phases is not None and sb_phases.shape[0] > 0:
            pidx = int(sb_phases[0, c, 0])
            phase = phase_table[pidx % len(phase_table)]

        W[ant_idx] += amp * np.exp(1j * phase)

    norm = np.linalg.norm(W)
    if norm > 0:
        W /= norm
    return W


def compute_correlation(H_est, W, hdr):
    """Compute average normalized correlation across active subcarriers and RX antennas."""
    N_rx = hdr["N_rx"]
    N_ports = hdr["N_ports"]
    ofdm_size = hdr["ofdm_size"]
    first_carrier = hdr["first_carrier"]
    start_rb = hdr["start_rb"]
    nr_of_rbs = hdr["nr_of_rbs"]
    mem_offset = hdr["mem_offset"]

    w_vec = W[:N_ports]

    rho_sum = 0.0
    count = 0

    for rb in range(start_rb, start_rb + nr_of_rbs):
        k_base = (first_carrier + rb * 12) % ofdm_size
        for sc in range(12):
            k = (k_base + sc) % ofdm_size
            for rx in range(N_rx):
                h = H_est[rx, :, k]
                h_norm = np.linalg.norm(h)
                w_norm = np.linalg.norm(w_vec)
                if h_norm > 0 and w_norm > 0:
                    rho = np.abs(np.vdot(h, w_vec)) / (h_norm * w_norm)
                    rho_sum += rho
                    count += 1

    return rho_sum / count if count > 0 else 0.0


def extract_snr_tag(filepath):
    """Try to extract SNR value from filename like 'type1_snr10.bin'."""
    name = Path(filepath).stem
    for part in name.split("_"):
        if part.startswith("snr"):
            try:
                return float(part[3:])
            except ValueError:
                pass
    return None


def process_files(filepaths):
    """Process all log files and return correlation results grouped by (pmi_type, snr)."""
    results = defaultdict(list)

    for fp in filepaths:
        snr_tag = extract_snr_tag(fp)
        print(f"Processing {fp} (SNR tag: {snr_tag}) ...")

        rec_count = 0
        for hdr, H_est, sb_phases in read_records(fp):
            if hdr["pmi_type"] == 0:
                W = reconstruct_type1(hdr)
                label = "Type-I"
            else:
                W = reconstruct_type2(hdr, sb_phases)
                label = "Type-II"

            rho = compute_correlation(H_est, W, hdr)
            results[(label, snr_tag)].append(rho)
            rec_count += 1

        file_rhos = []
        for key, vals in results.items():
            if key[1] == snr_tag and len(vals) > 0:
                file_rhos.extend(vals[-rec_count:] if rec_count <= len(vals) else vals)
        avg = np.mean(file_rhos) if file_rhos else 0.0
        print(f"  -> {rec_count} records, avg rho = {avg:.4f}")

    return results


def print_summary(results):
    """Print a tabular summary of results."""
    by_type = defaultdict(dict)
    for (label, snr), vals in sorted(results.items()):
        by_type[label][snr] = (np.mean(vals), np.std(vals), len(vals))

    print("\n" + "=" * 70)
    print(f"{'PMI Type':<12} {'SNR':>8} {'Avg rho':>10} {'Std':>10} {'N':>6}")
    print("-" * 70)
    for label in sorted(by_type.keys()):
        for snr in sorted(by_type[label].keys(), key=lambda x: x if x is not None else -999):
            mean_r, std_r, n = by_type[label][snr]
            snr_str = f"{snr:.0f} dB" if snr is not None else "N/A"
            print(f"{label:<12} {snr_str:>8} {mean_r:>10.4f} {std_r:>10.4f} {n:>6}")
    print("=" * 70)


def plot_results(results, output_path):
    """Generate comparison plots: correlation vs SNR, CDF, and improvement."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    by_type = defaultdict(dict)
    for (label, snr), vals in results.items():
        if snr is not None:
            by_type[label][snr] = vals

    has_snr_data = any(len(d) > 1 for d in by_type.values())

    if has_snr_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = [axes[0], axes[1], None]

    colors = {"Type-I": "#2196F3", "Type-II": "#F44336"}

    # Plot 1: Average Correlation vs SNR (grouped bar chart)
    ax = axes[0]
    sorted_labels = sorted(by_type.keys())
    all_snr_vals = sorted({s for d in by_type.values() for s in d.keys()})
    n_groups = len(all_snr_vals)
    n_bars = len(sorted_labels)
    bar_width = 0.35
    x_pos = np.arange(n_groups)

    for i, label in enumerate(sorted_labels):
        means = [np.mean(by_type[label].get(s, [0])) for s in all_snr_vals]
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(x_pos + offset, means, bar_width,
               label=label, color=colors.get(label, None), alpha=0.85)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Avg Normalized Correlation")
    ax.set_title("Channel Correlation vs SNR")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s:.0f}" for s in all_snr_vals])
    if any(ax.get_legend_handles_labels()[1]):
        ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: CDF at selected SNR points
    ax = axes[1]
    all_snrs = set()
    for d in by_type.values():
        all_snrs.update(d.keys())
    selected_snrs = sorted(all_snrs)
    if len(selected_snrs) > 3:
        mid = len(selected_snrs) // 2
        selected_snrs = [selected_snrs[0], selected_snrs[mid], selected_snrs[-1]]
    elif len(selected_snrs) == 0:
        for (label, _snr), vals in results.items():
            selected_snrs = [_snr]
            break

    linestyles = ["-", "--", ":"]
    for i, snr_val in enumerate(selected_snrs):
        ls = linestyles[i % len(linestyles)]
        for label in sorted(by_type.keys()):
            vals = by_type[label].get(snr_val, [])
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            snr_str = f"{snr_val:.0f}dB" if snr_val is not None else "N/A"
            ax.plot(sorted_vals, cdf, ls, label=f"{label} @ {snr_str}",
                    color=colors.get(label, None))
    ax.set_xlabel("Normalized Correlation")
    ax.set_ylabel("CDF")
    ax.set_title("Correlation CDF")
    ax.set_xlim(0, 1.05)
    if any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Percentage improvement (only if multi-SNR data)
    if axes[2] is not None and "Type-I" in by_type and "Type-II" in by_type:
        ax = axes[2]
        common_snrs = sorted(set(by_type["Type-I"].keys()) & set(by_type["Type-II"].keys()))
        if len(common_snrs) > 0:
            improvements = []
            for s in common_snrs:
                m1 = np.mean(by_type["Type-I"][s])
                m2 = np.mean(by_type["Type-II"][s])
                imp = ((m2 - m1) / m1 * 100) if m1 > 0 else 0
                improvements.append(imp)
            ax.bar(range(len(common_snrs)), improvements, color="#4CAF50", alpha=0.8)
            ax.set_xticks(range(len(common_snrs)))
            ax.set_xticklabels([f"{s:.0f}" for s in common_snrs])
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Improvement (%)")
            ax.set_title("Type-II vs Type-I Improvement")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Type-II PMI implementation via channel correlation analysis."
    )
    parser.add_argument("files", nargs="+", help="Binary CSI log files (.bin)")
    default_out = str(Path(__file__).resolve().parent / "results" / "pmi_correlation.png")
    parser.add_argument("--output", "-o", default=default_out,
                        help=f"Output plot filename (default: {default_out})")
    args = parser.parse_args()

    results = process_files(args.files)
    if not results:
        print("No records found in any input file.", file=sys.stderr)
        sys.exit(1)

    print_summary(results)
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
