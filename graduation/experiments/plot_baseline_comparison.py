#!/usr/bin/env python3
"""
Baseline 성능 비교 그래프 생성
  - 3개 시나리오(UMi-LOS, UMi-NLOS, UMa-NLOS)별 3×1 subplot 그룹
  - 각 그룹: BLER, Average MCS, Cell Throughput
  - 횡좌표: UE 수 (1, 2, 4, 8, 16)
  - 3개 곡선: Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO

실험 데이터가 없는 경우 3GPP 채널 모델 통계 기반 이론적 추정값을 사용합니다.
실험 후 DATA 섹션의 값을 교체하면 실측 그래프가 됩니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

UE_COUNTS = np.array([1, 2, 4, 8, 16])

# ──────────────────────────────────────────────────────────────────
# DATA — 이론적 추정값 (실험 후 실측값으로 교체)
#
# 모델링 근거:
#   - UMi-LOS:  높은 SNR (~22 dB median), 강한 LOS 성분 → 높은 빔포밍 이득
#   - UMi-NLOS: 중간 SNR (~15 dB), 다중 클러스터 → Type 2의 이점이 부각
#   - UMa-NLOS: 낮은 SNR (~10 dB), 넓은 지연/각도 확산 → CSI 정확도가 핵심
#
#   SU-MIMO: UE 증가 시 RR 스케줄링으로 per-UE 시간 감소 → cell tput 포화
#   MU-MIMO: UE 증가 시 공간 다중화 이득 → cell tput 선형에 가깝게 증가
#            단, CSI 양자화 오차로 인한 잔여 간섭이 UE 수에 비례하여 증가
# ──────────────────────────────────────────────────────────────────

scenarios = {
    "UMi-LOS": {
        "Type 1 SU-MIMO": {
            "bler":  [0.08, 0.09, 0.10, 0.12, 0.15],
            "mcs":   [14,   13,   12,   11,   10  ],
            "tput":  [18.0, 28.5, 38.0, 44.0, 47.0],
        },
        "Type 2 SU-MIMO": {
            "bler":  [0.05, 0.06, 0.07, 0.08, 0.10],
            "mcs":   [17,   16,   15,   14,   13  ],
            "tput":  [22.0, 35.0, 48.0, 56.0, 60.0],
        },
        "Type 2 MU-MIMO": {
            "bler":  [0.06, 0.07, 0.08, 0.10, 0.13],
            "mcs":   [16,   15,   14,   13,   11  ],
            "tput":  [20.0, 38.0, 68.0, 96.0, 110.0],
        },
    },
    "UMi-NLOS": {
        "Type 1 SU-MIMO": {
            "bler":  [0.12, 0.14, 0.16, 0.20, 0.25],
            "mcs":   [10,   9,    8,    7,    6   ],
            "tput":  [12.0, 18.5, 24.0, 27.0, 28.5],
        },
        "Type 2 SU-MIMO": {
            "bler":  [0.07, 0.08, 0.10, 0.12, 0.16],
            "mcs":   [13,   12,   11,   10,   9   ],
            "tput":  [16.0, 26.0, 36.0, 42.0, 45.0],
        },
        "Type 2 MU-MIMO": {
            "bler":  [0.08, 0.09, 0.11, 0.14, 0.19],
            "mcs":   [12,   11,   10,   9,    7   ],
            "tput":  [15.0, 28.0, 50.0, 68.0, 76.0],
        },
    },
    "UMa-NLOS": {
        "Type 1 SU-MIMO": {
            "bler":  [0.18, 0.20, 0.24, 0.30, 0.38],
            "mcs":   [7,    6,    5,    4,    3   ],
            "tput":  [8.0,  12.0, 15.0, 16.5, 17.0],
        },
        "Type 2 SU-MIMO": {
            "bler":  [0.10, 0.12, 0.14, 0.18, 0.24],
            "mcs":   [10,   9,    8,    7,    6   ],
            "tput":  [12.0, 19.0, 26.0, 30.0, 32.0],
        },
        "Type 2 MU-MIMO": {
            "bler":  [0.11, 0.13, 0.16, 0.21, 0.28],
            "mcs":   [9,    8,    7,    6,    4   ],
            "tput":  [11.0, 21.0, 36.0, 46.0, 50.0],
        },
    },
}

MODES = ["Type 1 SU-MIMO", "Type 2 SU-MIMO", "Type 2 MU-MIMO"]
COLORS = {"Type 1 SU-MIMO": "#2166ac",
           "Type 2 SU-MIMO": "#4daf4a",
           "Type 2 MU-MIMO": "#e41a1c"}
MARKERS = {"Type 1 SU-MIMO": "s",
            "Type 2 SU-MIMO": "^",
            "Type 2 MU-MIMO": "o"}
LINESTYLES = {"Type 1 SU-MIMO": "--",
               "Type 2 SU-MIMO": "-.",
               "Type 2 MU-MIMO": "-"}

METRICS = [
    ("bler",  "BLER",                    "%",    lambda v: v * 100),
    ("mcs",   "Average MCS Index",       "",     lambda v: v),
    ("tput",  "Cell DL Throughput",      "Mbps", lambda v: v),
]


def plot_scenario_group(scenario_name, data, fig_idx):
    """한 시나리오(채널)에 대해 3개 metric subplot 생성."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Scenario: {scenario_name}", fontsize=15, fontweight="bold", y=1.02)

    for col, (key, ylabel, unit, transform) in enumerate(METRICS):
        ax = axes[col]
        for mode in MODES:
            y = np.array(data[mode][key], dtype=float)
            y = transform(y)
            ax.plot(UE_COUNTS, y,
                    color=COLORS[mode],
                    marker=MARKERS[mode],
                    linestyle=LINESTYLES[mode],
                    label=mode,
                    markeredgecolor="white",
                    markeredgewidth=0.8)
        ax.set_xlabel("Number of UEs")
        label = f"{ylabel} ({unit})" if unit else ylabel
        ax.set_ylabel(label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(UE_COUNTS)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.minorticks_off()

        if key == "bler":
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        elif key == "tput":
            ax.set_ylim(bottom=0)

        ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fname = OUT_DIR / f"baseline_{scenario_name.replace('-', '_')}.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname


def plot_combined():
    """3개 시나리오를 3×3 격자 (row=scenario, col=metric)로 통합."""
    scenario_names = list(scenarios.keys())
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    fig.suptitle("Baseline Performance Comparison: Type 1 SU / Type 2 SU / Type 2 MU-MIMO",
                 fontsize=15, fontweight="bold", y=0.98)

    for row, sc_name in enumerate(scenario_names):
        data = scenarios[sc_name]
        for col, (key, ylabel, unit, transform) in enumerate(METRICS):
            ax = axes[row][col]
            for mode in MODES:
                y = np.array(data[mode][key], dtype=float)
                y = transform(y)
                ax.plot(UE_COUNTS, y,
                        color=COLORS[mode],
                        marker=MARKERS[mode],
                        linestyle=LINESTYLES[mode],
                        label=mode,
                        markeredgecolor="white",
                        markeredgewidth=0.8)

            ax.set_xscale("log", base=2)
            ax.set_xticks(UE_COUNTS)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.minorticks_off()

            if row == 2:
                ax.set_xlabel("Number of UEs")
            if col == 0:
                ax.set_ylabel(sc_name, fontsize=12, fontweight="bold",
                              rotation=90, labelpad=15)

            if key == "bler":
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
            elif key == "tput":
                ax.set_ylim(bottom=0)

            if row == 0:
                label = f"{ylabel} ({unit})" if unit else ylabel
                ax.set_title(label, fontsize=12, pad=8)

            if row == 0 and col == 2:
                ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fname = OUT_DIR / "baseline_combined_3x3.png"
    fig.savefig(fname)
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname


if __name__ == "__main__":
    print("Generating baseline comparison figures...")
    for sc_name, data in scenarios.items():
        plot_scenario_group(sc_name, data, 0)
    plot_combined()
    print("Done.")
