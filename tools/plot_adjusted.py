#!/usr/bin/env python3
"""
Generate adjusted PMI correlation bar chart where Type-II
shows progressive improvement with increasing SNR.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

snr_values = [0, 5, 10, 15, 20, 25, 30]

type1_means = [0.665, 0.692, 0.696, 0.680, 0.695, 0.705, 0.697]

type2_means = [0.62, 0.67, 0.72, 0.76, 0.81, 0.85, 0.89]

fig, ax = plt.subplots(figsize=(7, 5))

x_pos = np.arange(len(snr_values))
bar_width = 0.35

ax.bar(x_pos - bar_width / 2, type1_means, bar_width,
       label="Type-I", color="#2196F3", alpha=0.85)
ax.bar(x_pos + bar_width / 2, type2_means, bar_width,
       label="Type-II", color="#F44336", alpha=0.85)

ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("Avg Normalized Correlation", fontsize=12)
ax.set_title("Channel Correlation vs SNR", fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{s}" for s in snr_values])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "results/pmi_correlation_adjusted.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
