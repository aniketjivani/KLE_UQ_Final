import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", family="serif")
plt.rc("axes.spines", top=True, right=True)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('axes', labelsize=15)
plt.rc('figure', titlesize=15)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.8)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bifi_regime_study"))
data_dir = os.path.join(ROOT_DIR, "outputs", "lf_pilot_cost_accounted", "data")
fig_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(fig_dir, exist_ok=True)

S_VALUES = [0.0, 0.5, 1.0]
BUDGETS = [100, 125, 150]
TAG_SUFFIX = {100: "", 125: "_budget125", 150: "_budget150"}
R_COLS = [4.0, 8.0, 16.0, np.inf]

bf_color = '#9467bd'
hf_color = '#c36027'
line_styles = ['-', '--', ':', '-.']


def load_cell(budget, r, s):
    suffix = "freeLF" if np.isinf(r) else f"r{int(r)}"
    tag = f"BS_s{s:.3f}_case2{TAG_SUFFIX[budget]}"
    fname = os.path.join(data_dir, f"lf_pilot_cost_accounted_{tag}_{suffix}.npz")
    return np.load(fname)


fig, axes = plt.subplots(len(S_VALUES), len(BUDGETS), figsize=(17, 13))

for row, s in enumerate(S_VALUES):
    row_min, row_max = np.inf, -np.inf
    for col, budget in enumerate(BUDGETS):
        ax = axes[row, col]
        for r, ls in zip(R_COLS, line_styles):
            d = load_cell(budget, r, s)
            bf_errors = d["bf_errors"]
            hf_errors = d["hf_errors"]
            n_h_sweep = d["N_H_sweep"]
            bf_mean, bf_std = bf_errors.mean(axis=0), bf_errors.std(axis=0)
            hf_mean, hf_std = hf_errors.mean(axis=0), hf_errors.std(axis=0)
            row_min = min(row_min, np.min(bf_mean - bf_std), np.min(hf_mean - hf_std))
            row_max = max(row_max, np.max(bf_mean + bf_std), np.max(hf_mean + hf_std))
            r_label = "free-LF" if np.isinf(r) else f"r={int(r)}"

            ax.plot(n_h_sweep, bf_mean, ls, marker="o", color=bf_color, markersize=3.8,
                    linewidth=1.6, label=f"BF-KLE ({r_label})")
            ax.fill_between(n_h_sweep, bf_mean - bf_std, bf_mean + bf_std, color=bf_color, alpha=0.15)
            ax.plot(n_h_sweep, hf_mean, ls, marker="D", color=hf_color, markersize=3.4,
                    linewidth=1.3, alpha=0.9, label=f"HF-KLE ({r_label})")
            ax.fill_between(n_h_sweep, hf_mean - hf_std, hf_mean + hf_std, color=hf_color, alpha=0.08)

        if row == 0:
            ax.set_title(r"$N_{\mathrm{LF}} = %d$" % budget, fontsize=14)
        if col == 0:
            ax.set_ylabel(rf"$s={s:.1f}$", fontsize=13, labelpad=15)
        if row < len(S_VALUES) - 1:
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis="both", labelsize=10)
    row_padding = 0.05 * (row_max - row_min)
    for ax in axes[row, :]:
        ax.set_ylim(row_min - row_padding, row_max + row_padding)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.005), frameon=False)
fig.suptitle("C2", fontsize=18, fontweight="bold", fontname="Times New Roman", y=0.98)
fig.supxlabel(r"$N_{\mathrm{HF}}^{(\ell)}$", y=0.095, fontsize=15)
fig.supylabel(r"$\mu_{\varepsilon,\ell}$", x=0.015, fontsize=15)
fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.16, hspace=0.20, wspace=0.16)

out_path = os.path.join(fig_dir, "007_alt_regime_BS_convergence.jpg")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
