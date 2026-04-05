#!/usr/bin/env python3
"""
plot_mimo_sweep.py — gnb.conf 모드 변경 + nrMAC_stats 파싱 + 그래프 생성

CLI:
  python3 plot_mimo_sweep.py apply-config <mode> <conf_path>
  python3 plot_mimo_sweep.py plot <manifest.csv>
"""
import sys, re, os, csv

MODES = ["type2_mu", "type2_su", "type1_su"]
MODE_LABELS = {
    "type2_mu": "Type-II MU-MIMO",
    "type2_su": "Type-II SU-MIMO",
    "type1_su": "Type-I SU-MIMO",
}
COLORS = {
    "type2_mu": "#1f77b4",
    "type2_su": "#2ca02c",
    "type1_su": "#d62728",
}
MARKERS = {
    "type2_mu": "o",
    "type2_su": "s",
    "type1_su": "^",
}

# ── gnb.conf 수정 ────────────────────────────────────────────────
def apply_config(mode, conf_path):
    """gnb.conf의 codebook_type 및 mu_mimo를 mode에 맞게 수정한다."""
    with open(conf_path) as f:
        lines = f.readlines()

    in_type1 = False
    in_type2 = False
    new_lines = []

    for line in lines:
        stripped = line.lstrip()

        if "### TYPE1_BEGIN" in line:
            in_type1 = True
            new_lines.append(line)
            continue
        if "### TYPE1_END" in line:
            in_type1 = False
            new_lines.append(line)
            continue
        if "### TYPE2_BEGIN" in line:
            in_type2 = True
            new_lines.append(line)
            continue
        if "### TYPE2_END" in line:
            in_type2 = False
            new_lines.append(line)
            continue

        if in_type1:
            content = stripped.lstrip("#").strip()
            indent = "        "
            if mode == "type1_su":
                new_lines.append(f"{indent}{content}\n")
            else:
                new_lines.append(f"{indent}#{content}\n")
            continue

        if in_type2:
            content = stripped.lstrip("#").strip()
            indent = "        "
            if mode in ("type2_mu", "type2_su"):
                new_lines.append(f"{indent}{content}\n")
            else:
                new_lines.append(f"{indent}#{content}\n")
            continue

        if re.match(r"\s*mu_mimo\s*=", line):
            val = 1 if mode == "type2_mu" else 0
            new_lines.append(f"  mu_mimo                     = {val};\n")
            continue

        new_lines.append(line)

    with open(conf_path, "w") as f:
        f.writelines(new_lines)
    print(f"[config] Applied mode={mode} to {conf_path}")


# ── nrMAC_stats.log 파싱 ─────────────────────────────────────────
RE_UE = re.compile(r"UE RNTI (\w+) CU-UE-ID (\d+) (in|out)-of-sync")
RE_DL = re.compile(
    r"UE \w+: dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+), dlsch_errors (\d+), "
    r"pucch0_DTX (\d+), BLER ([\d.]+) MCS \(\d+\) (\d+)"
)
RE_UL = re.compile(
    r"UE \w+: ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+), "
    r"ulsch_DTX (\d+), BLER ([\d.]+) MCS .* SNR ([\d.]+) dB"
)
RE_TX = re.compile(r"UE \w+: MAC:\s+TX\s+(\d+)\s+RX\s+(\d+)")


def parse_mac_stats(path):
    entries = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m_ue = RE_UE.search(lines[i])
        if not m_ue:
            i += 1
            continue
        ue = {
            "rnti": m_ue.group(1),
            "cu_id": int(m_ue.group(2)),
            "in_sync": m_ue.group(3) == "in",
        }
        for j in range(1, 6):
            if i + j >= len(lines):
                break
            ln = lines[i + j]
            m_dl = RE_DL.search(ln)
            if m_dl:
                r = list(map(int, m_dl.group(1, 2, 3, 4)))
                ue["dl_rounds_total"] = sum(r)
                ue["dl_rounds"] = r
                ue["dl_errors"] = int(m_dl.group(5))
                ue["pucch0_dtx"] = int(m_dl.group(6))
                ue["dl_bler"] = float(m_dl.group(7))
                ue["mcs"] = int(m_dl.group(8))
            m_ul = RE_UL.search(ln)
            if m_ul:
                ue["ul_snr"] = float(m_ul.group(8))
                ue["ul_bler"] = float(m_ul.group(7))
            m_tx = RE_TX.search(ln)
            if m_tx:
                ue["tx_bytes"] = int(m_tx.group(1))
                ue["rx_bytes"] = int(m_tx.group(2))
        entries.append(ue)
        i += 1
    return entries


def summarize(entries, expected_ues):
    """한 실행의 파싱 결과를 요약 딕셔너리로 반환."""
    if not entries:
        return {
            "avg_dl_bler": 1.0, "avg_ul_bler": 1.0,
            "total_tx_kb": 0, "avg_mcs": 0, "avg_pucch_dtx": 0,
            "avg_ul_snr": 0, "num_rnti": 0, "reconnect_ratio": 0,
            "avg_dl_retx_ratio": 0,
        }
    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0

    n = len(entries)
    avg_dl_bler = _mean([e.get("dl_bler", 1.0) for e in entries])
    avg_ul_bler = _mean([e.get("ul_bler", 1.0) for e in entries])
    total_tx_kb = sum(e.get("tx_bytes", 0) for e in entries) / 1024
    avg_mcs = _mean([e.get("mcs", 0) for e in entries])
    avg_dtx = _mean([e.get("pucch0_dtx", 0) for e in entries])
    snr_vals = [e["ul_snr"] for e in entries if "ul_snr" in e]
    avg_snr = _mean(snr_vals)
    reconnect = n / expected_ues if expected_ues > 0 else 0

    dl_1st = sum(e.get("dl_rounds", [0])[0] for e in entries)
    dl_all = sum(e.get("dl_rounds_total", 0) for e in entries)
    retx_ratio = (dl_all - dl_1st) / dl_all if dl_all > 0 else 0

    return {
        "avg_dl_bler": avg_dl_bler,
        "avg_ul_bler": avg_ul_bler,
        "total_tx_kb": total_tx_kb,
        "avg_mcs": avg_mcs,
        "avg_pucch_dtx": avg_dtx,
        "avg_ul_snr": avg_snr,
        "num_rnti": n,
        "reconnect_ratio": reconnect,
        "avg_dl_retx_ratio": retx_ratio,
    }


# ── 그래프 생성 ──────────────────────────────────────────────────
def _init_matplotlib():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return np, plt


def generate_plots(manifest_path):
    np, plt = _init_matplotlib()

    out_dir = os.path.join(os.path.expanduser("~"), "oai_sionna_junxiu", "figures_sweep")
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)

    ue_counts = sorted(set(int(r["num_ues"]) for r in runs))
    modes_seen = sorted(set(r["mode"] for r in runs), key=lambda m: MODES.index(m))

    results = {}
    for r in runs:
        n = int(r["num_ues"])
        mode = r["mode"]
        log_path = os.path.join(r["log_dir"], "nrMAC_stats.log")
        if not os.path.exists(log_path):
            print(f"  WARN: {log_path} not found, skipping")
            continue
        entries = parse_mac_stats(log_path)
        s = summarize(entries, n)
        results[(n, mode)] = s
        print(f"  UE={n:2d}  mode={mode:10s}  BLER={s['avg_dl_bler']:.3f}  "
              f"TX={s['total_tx_kb']:.0f}KB  MCS={s['avg_mcs']:.1f}  "
              f"RNTI={s['num_rnti']}  DTX={s['avg_pucch_dtx']:.0f}")

    x = np.array(ue_counts)

    def get_metric(mode, key):
        return np.array([results.get((n, mode), {}).get(key, 0) for n in ue_counts])

    def save(fig, name):
        fig.savefig(os.path.join(out_dir, f"{name}.png"))
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close(fig)
        print(f"  Saved {name}")

    # ── Fig 1: Average DL BLER ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "avg_dl_bler")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Average DL BLER")
    ax.set_title("(a) Average Downlink BLER vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig1_avg_dl_bler")

    # ── Fig 2: Aggregate DL Throughput ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "total_tx_kb")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Aggregate DL Throughput (KB)")
    ax.set_title("(b) Aggregate Downlink Throughput vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig2_aggregate_throughput")

    # ── Fig 3: Average MCS ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "avg_mcs")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Average MCS Index")
    ax.set_title("(c) Average MCS Index vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig3_avg_mcs")

    # ── Fig 4: PUCCH DTX ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "avg_pucch_dtx")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Average PUCCH DTX Count")
    ax.set_title("(d) Average PUCCH DTX vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig4_pucch_dtx")

    # ── Fig 5: Connection Stability ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "reconnect_ratio")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal (no reconnection)")
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("RNTI Count / Expected UEs")
    ax.set_title("(e) Connection Stability vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "fig5_connection_stability")

    # ── Fig 6: DL HARQ Retransmission Ratio ───────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "avg_dl_retx_ratio")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("DL HARQ Retransmission Ratio")
    ax.set_title("(f) DL HARQ Retransmission Ratio vs. Number of UEs")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, "fig6_dl_retx_ratio")

    # ── Fig 7: Average UL SNR ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in modes_seen:
        y = get_metric(mode, "avg_ul_snr")
        ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                label=MODE_LABELS[mode], linewidth=2, markersize=8)
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Average UL SNR (dB)")
    ax.set_title("(g) Average Uplink SNR vs. Number of UEs")
    ax.set_xticks(x)
    ax.legend()
    save(fig, "fig7_ul_snr")

    # ── Fig 8: Combined summary (2x2 subplot) ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    metric_defs = [
        ("avg_dl_bler", "Average DL BLER", "(a)"),
        ("total_tx_kb", "Aggregate DL Throughput (KB)", "(b)"),
        ("avg_mcs", "Average MCS Index", "(c)"),
        ("avg_pucch_dtx", "Average PUCCH DTX Count", "(d)"),
    ]
    for ax, (key, ylabel, prefix) in zip(axes.flat, metric_defs):
        for mode in modes_seen:
            y = get_metric(mode, key)
            ax.plot(x, y, marker=MARKERS[mode], color=COLORS[mode],
                    label=MODE_LABELS[mode], linewidth=2, markersize=7)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix} {ylabel}")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle("MIMO Mode Performance vs. Number of UEs", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "fig8_combined_summary")

    print(f"\nAll figures saved to: {out_dir}/")
    return out_dir


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 plot_mimo_sweep.py apply-config <mode> <conf_path>")
        print("  python3 plot_mimo_sweep.py plot <manifest.csv>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "apply-config":
        if len(sys.argv) != 4:
            print("Usage: python3 plot_mimo_sweep.py apply-config <mode> <conf_path>")
            sys.exit(1)
        apply_config(sys.argv[2], sys.argv[3])

    elif cmd == "plot":
        if len(sys.argv) != 3:
            print("Usage: python3 plot_mimo_sweep.py plot <manifest.csv>")
            sys.exit(1)
        generate_plots(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
