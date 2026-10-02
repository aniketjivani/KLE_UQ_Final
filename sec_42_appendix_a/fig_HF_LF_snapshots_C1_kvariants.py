import math
import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(fig_dir, exist_ok=True)


def spiked_waveform(x, a, b):
    return np.exp(-a * x) * np.sin(b * x)


def sin_truncated(z, k):
    assert k % 2 == 1, "truncation order k must be odd (sin is an odd function)"
    out = np.zeros_like(z)
    for n in range(0, k, 2):
        out = out + ((-1) ** (n // 2) / math.factorial(n + 1)) * z ** (n + 1)
    return out


def lf_family_t(x, a, b, k):
    return np.exp(-a * x) * sin_truncated(b * x, k)


x = np.linspace(0, 0.1, 250)

a_c1 = [40, 50, 55]
b_c1 = [71, 60, 80]

K_VALS = [1, 3, 5]
K_LABELS = {1: r"$k=1$", 3: r"$k=3$", 5: r"$k=5$"}
K_NOTES = {1: "", 3: "", 5: " (Exp 1a C1)"}

fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)

for row, k in enumerate(K_VALS):
    row_ymin, row_ymax = np.inf, -np.inf
    for col in range(3):
        y_hf = spiked_waveform(x, a_c1[col], b_c1[col])
        y_lf = lf_family_t(x, a_c1[col], b_c1[col], k)
        ax = axes[row, col]
        ax.plot(x, y_hf, color="blue", label="HF", linewidth=2.8)
        ax.plot(x, y_lf, color="orange", label="LF", linewidth=2.8)
        row_ymin = min(row_ymin, y_hf.min(), y_lf.min())
        row_ymax = max(row_ymax, y_hf.max(), y_lf.max())
        k_title = K_LABELS[k] + K_NOTES[k]
        if row == 0:
            title = r"$\mathbf{{a = {}}}$, $\mathbf{{b = {}}}$""\n"r"{}".format(a_c1[col], b_c1[col], k_title)
        else:
            title = k_title
        ax.set_title(title, fontsize=18)
        if col == 0:
            ax.set_ylabel(r"$y$", fontsize=20)
        if row == 2:
            ax.set_xlabel(r"$x$", fontsize=20)
        ax.grid(True)
        ax.set_xlim([0, 0.1])
        ax.tick_params(axis="both", which="major", labelsize=17)
    pad = 0.05 * (row_ymax - row_ymin)
    for col in range(3):
        axes[row, col].set_ylim([row_ymin - pad, row_ymax + pad])

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=18, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), frameon=False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)

out_path = os.path.join(fig_dir, "012_HF_LF_snapshots_C1_kvariants.jpg")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
