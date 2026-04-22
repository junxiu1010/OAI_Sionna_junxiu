#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

columns = ["Parameter", "UMi-LOS", "UMi-NLOS", "UMa-LOS", "UMa-NLOS"]
data = [
    ["BS Height",        "10 m",        "10 m",        "25 m",        "25 m"],
    ["UE Height",        "1.5~2.5 m",   "1.5~2.5 m",   "1.5~2.5 m",   "1.5~2.5 m"],
    ["ISD",              "200 m",       "200 m",       "500 m",       "500 m"],
    ["UE Distance",      "10~150 m",    "10~150 m",    "35~500 m",    "35~500 m"],
    ["Carrier Freq.",    "0.5~100 GHz", "0.5~100 GHz", "0.5~100 GHz", "0.5~100 GHz"],
    ["Shadow Fading σ",  "4 dB",        "7.82 dB",     "4 dB",        "6 dB"],
    ["K-factor μ/σ",     "9/5 dB",      "N/A",         "9/3.5 dB",    "N/A"],
]

fig, ax = plt.subplots(figsize=(9, 4))
ax.axis("off")

header_color = "#2166ac"
row_colors = ["#ffffff", "#f0f0f0"] * 4

table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)
    else:
        cell.set_facecolor(row_colors[(row - 1) % 2])
        cell.set_text_props(color="#222222", fontsize=11)
    if col == 0:
        cell.set_text_props(fontweight="bold", fontsize=11,
                            color="white" if row == 0 else "#222222")

fig.suptitle("3GPP TR 38.901 Channel Model Parameters",
             fontsize=14, fontweight="bold", y=0.95)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out = Path(__file__).parent / "3gpp_channel_parameters_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
