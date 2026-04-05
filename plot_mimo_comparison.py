#!/usr/bin/env python3
"""
MIMO Mode Performance Comparison — Publication-Quality Figures
Configs:
  A: Type-II MU-MIMO  (20260401_172044)
  B: Type-II SU-MIMO  (20260401_172550)
  C: Type-I  SU-MIMO  (20260401_172944)
"""
import re, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

LABELS = {
    "A": "Type-II MU-MIMO",
    "B": "Type-II SU-MIMO",
    "C": "Type-I SU-MIMO",
}
COLORS = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
MARKERS = {"A": "o", "B": "s", "C": "^"}

LOG_BASE = "/home/dclserver78/oai_sionna_junxiu/logs"
LOG_DIRS = {
    "A": "20260401_172044_G1C_v0_ipc_8ue_ga2x1_ua2x1_xp2",
    "B": "20260401_172550_G1C_v0_ipc_8ue_ga2x1_ua2x1_xp2",
    "C": "20260401_172944_G1C_v0_ipc_8ue_ga2x1_ua2x1_xp2",
}
OUT = os.path.join(os.path.expanduser("~"), "oai_sionna_junxiu", "figures")
os.makedirs(OUT, exist_ok=True)

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
        ue = {"rnti": m_ue.group(1), "cu_id": int(m_ue.group(2)),
              "in_sync": m_ue.group(3) == "in"}
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
        entries.append(ue)
        i += 1
    return entries

data = {}
for k, d in LOG_DIRS.items():
    p = os.path.join(LOG_BASE, d, "nrMAC_stats.log")
    data[k] = parse_mac_stats(p)
    print(f"Config {k} ({LABELS[k]}): {len(data[k])} UEs parsed")

N_ORIG = 8

# ═══════════════════════════════════════════════════════════════
# Fig 1: DL BLER per original UE (bar chart)
# ═══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 4.2))
x = np.arange(1, N_ORIG + 1)
width = 0.25
for i, k in enumerate(["A", "B", "C"]):
    vals = [data[k][j]["dl_bler"] for j in range(min(N_ORIG, len(data[k])))]
    ax1.bar(x + (i - 1) * width, vals, width, label=LABELS[k],
            color=COLORS[k], edgecolor="white", linewidth=0.5)
ax1.set_xlabel("UE Index")
ax1.set_ylabel("DL BLER")
ax1.set_title("(a) Downlink BLER per UE (Initial 8 UEs)")
ax1.set_xticks(x)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower right")
fig1.savefig(os.path.join(OUT, "fig1_dl_bler_per_ue.png"))
fig1.savefig(os.path.join(OUT, "fig1_dl_bler_per_ue.pdf"))
print("Saved fig1")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════
# Fig 2: CDF of DL BLER (all UEs)
# ═══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(6, 4.2))
for k in ["A", "B", "C"]:
    blers = sorted([u["dl_bler"] for u in data[k]])
    cdf = np.arange(1, len(blers) + 1) / len(blers)
    ax2.step(blers, cdf, where="post", label=LABELS[k],
             color=COLORS[k], linewidth=2)
ax2.set_xlabel("DL BLER")
ax2.set_ylabel("CDF")
ax2.set_title("(b) CDF of DL BLER (All UE Instances)")
ax2.set_xlim(0, 1)
ax2.legend(loc="upper left")
fig2.savefig(os.path.join(OUT, "fig2_dl_bler_cdf.png"))
fig2.savefig(os.path.join(OUT, "fig2_dl_bler_cdf.pdf"))
print("Saved fig2")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════
# Fig 3: Aggregate KPI bar chart
# ═══════════════════════════════════════════════════════════════
def avg_bler_first8(d):
    return np.mean([d[j]["dl_bler"] for j in range(min(8, len(d)))])

def total_tx_kb(d):
    return sum(u.get("tx_bytes", 0) for u in d) / 1024

def avg_dtx_first8(d):
    return np.mean([d[j]["pucch0_dtx"] for j in range(min(8, len(d)))])

metrics = {
    "Avg DL BLER\n(UE 1\u20138)": {k: avg_bler_first8(data[k]) for k in "ABC"},
    "Unique RNTI\nCount": {k: float(len(data[k])) for k in "ABC"},
    "Total DL TX\n(KB)": {k: total_tx_kb(data[k]) for k in "ABC"},
    "Avg PUCCH DTX\n(UE 1\u20138)": {k: avg_dtx_first8(data[k]) for k in "ABC"},
}

fig3, axes3 = plt.subplots(1, 4, figsize=(14, 4))
for ax, (title, vals) in zip(axes3, metrics.items()):
    x_pos = np.arange(3)
    bars = [vals[k] for k in "ABC"]
    colors_l = [COLORS[k] for k in "ABC"]
    b = ax.bar(x_pos, bars, color=colors_l, edgecolor="white", width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[k].replace(" ", "\n") for k in "ABC"], fontsize=8)
    ax.set_title(title, fontsize=11)
    for bar, val in zip(b, bars):
        if val < 1:
            fmt = f"{val:.3f}"
        elif val < 100:
            fmt = f"{val:.0f}"
        else:
            fmt = f"{val:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(bars) * 1.25)
fig3.suptitle("(c) Aggregate Performance Comparison", fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, "fig3_aggregate_kpi.png"))
fig3.savefig(os.path.join(OUT, "fig3_aggregate_kpi.pdf"))
print("Saved fig3")
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════
# Fig 4: HARQ retransmission distribution (first 8 UEs)
# ═══════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(7, 4.2))
harq_labels = ["1st TX", "2nd TX\n(Retx-1)", "3rd TX\n(Retx-2)", "4th TX\n(Retx-3)"]
x4 = np.arange(4)
width4 = 0.25
for i, k in enumerate(["A", "B", "C"]):
    rounds_sum = np.zeros(4)
    for j in range(min(8, len(data[k]))):
        r = data[k][j].get("dl_rounds", [0, 0, 0, 0])
        rounds_sum += np.array(r)
    rounds_norm = rounds_sum / rounds_sum[0] if rounds_sum[0] > 0 else rounds_sum
    ax4.bar(x4 + (i - 1) * width4, rounds_norm, width4,
            label=LABELS[k], color=COLORS[k], edgecolor="white")
ax4.set_xticks(x4)
ax4.set_xticklabels(harq_labels)
ax4.set_ylabel("Normalized Ratio (to 1st TX)")
ax4.set_title("(d) HARQ Retransmission Distribution (UE 1\u20138 Aggregate)")
ax4.set_ylim(0, 1.15)
ax4.legend()
fig4.savefig(os.path.join(OUT, "fig4_harq_distribution.png"))
fig4.savefig(os.path.join(OUT, "fig4_harq_distribution.pdf"))
print("Saved fig4")
plt.close(fig4)

# ═══════════════════════════════════════════════════════════════
# Fig 5: PUCCH DTX over connection order
# ═══════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(8, 4.2))
for k in ["A", "B", "C"]:
    dtx_vals = [u["pucch0_dtx"] for u in data[k]]
    indices = np.arange(1, len(dtx_vals) + 1)
    ax5.plot(indices, dtx_vals, marker=MARKERS[k], markersize=3,
             label=LABELS[k], color=COLORS[k], linewidth=1.2, alpha=0.85)
ax5.set_xlabel("UE Instance (Connection Order)")
ax5.set_ylabel("PUCCH Format-0 DTX Count")
ax5.set_title("(e) PUCCH DTX Accumulation per UE Instance")
ax5.legend()
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
fig5.savefig(os.path.join(OUT, "fig5_pucch_dtx_per_ue.png"))
fig5.savefig(os.path.join(OUT, "fig5_pucch_dtx_per_ue.pdf"))
print("Saved fig5")
plt.close(fig5)

# ═══════════════════════════════════════════════════════════════
# Fig 6: DL throughput per UE instance
# ═══════════════════════════════════════════════════════════════
fig6, ax6 = plt.subplots(figsize=(8, 4.2))
for k in ["A", "B", "C"]:
    tx_vals = [u.get("tx_bytes", 0) / 1024 for u in data[k]]
    indices = np.arange(1, len(tx_vals) + 1)
    ax6.plot(indices, tx_vals, marker=MARKERS[k], markersize=3,
             label=LABELS[k], color=COLORS[k], linewidth=1.2, alpha=0.85)
ax6.set_xlabel("UE Instance (Connection Order)")
ax6.set_ylabel("DL MAC TX (KB)")
ax6.set_title("(f) Downlink Throughput per UE Instance")
ax6.legend()
ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
fig6.savefig(os.path.join(OUT, "fig6_dl_throughput_per_ue.png"))
fig6.savefig(os.path.join(OUT, "fig6_dl_throughput_per_ue.pdf"))
print("Saved fig6")
plt.close(fig6)

print(f"\nAll figures saved to: {OUT}/")
print("Files:", sorted(os.listdir(OUT)))
