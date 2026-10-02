import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(fig_dir, exist_ok=True)


def spiked_waveform(x, a, b):
    return np.exp(-a * x) * np.sin(b * x)


def lf_family_bs(x, a, b, s):
    bx_d = x * (180 / np.pi) * b
    A = 4.0 - 0.5 * s
    D = 40500.0 - 25500.0 * s
    return np.exp(-a * x) * (A * bx_d * (180 - bx_d)) / (D - bx_d * (180 - bx_d))


x = np.linspace(0, 0.1, 250)

a_c2 = [35, 38, 40]
b_c2 = [75, 72, 70]

S_VALS = [0.0, 0.5, 1.0]
S_LABELS = {0.0: r"$s=0.0$", 0.5: r"$s=0.5$", 1.0: r"$s=1.0$"}
S_NOTES = {0.0: " (Stroethoff 2014)", 0.5: "", 1.0: " (Exp 1a C2)"}

fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)

for row, s in enumerate(S_VALS):
    row_ymin, row_ymax = np.inf, -np.inf
    for col in range(3):
        y_hf = spiked_waveform(x, a_c2[col], b_c2[col])
        y_lf = lf_family_bs(x, a_c2[col], b_c2[col], s)
        ax = axes[row, col]
        ax.plot(x, y_hf, color="blue", label="HF", linewidth=2.8)
        ax.plot(x, y_lf, color="orange", label="LF", linewidth=2.8)
        row_ymin = min(row_ymin, y_hf.min(), y_lf.min())
        row_ymax = max(row_ymax, y_hf.max(), y_lf.max())
        s_title = S_LABELS[s] + S_NOTES[s]
        if row == 0:
            title = r"$\mathbf{{a = {}}}$, $\mathbf{{b = {}}}$""\n"r"{}".format(a_c2[col], b_c2[col], s_title)
        else:
            title = s_title
        ax.set_title(title, fontsize=16)
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
plt.subplots_adjust(wspace=0.25, hspace=0.35, bottom=0.1)

out_path = os.path.join(fig_dir, "013_HF_LF_snapshots_C2_svariants.jpg")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
