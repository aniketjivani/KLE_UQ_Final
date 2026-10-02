\
\
\
\
\
\
\
\
   

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="serif")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APPENDIX_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(APPENDIX_DIR)
DATA_DIR = os.path.join(APPENDIX_DIR, "data")
RESULT_DIR = DATA_DIR
FIGDIR = os.path.join(APPENDIX_DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

COLOR_KLE = "#2a78d6"
COLOR_DON = "#eb6834"
NREPS = 10


def mean_std(x):
    return float(np.mean(x)), float(np.std(x))


def format_theta(val):
\
                               
    return f"{val:.0f}" if float(val).is_integer() else f"{val:.2f}"


def fig1_bar_panel(kle, don):
    metrics = [
        ("bf_errors", "Composite\n(primary)"),
        ("sf_errors", "SF ablation\n(secondary)"),
        ("lf_term_errors", "LF-term"),
        ("correction_term_errors", "Correction-term\n(diagnostic)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (key, label) in zip(axes, metrics):
        km, ks = mean_std(kle[key])
        dm, ds = mean_std(don[key])
        x = [0, 1]
        heights = [km, dm]
        errs = [ks, ds]
        colors = [COLOR_KLE, COLOR_DON]
        xticklabels = ["SF-KLE", "SF-DeepONet"] if key == "sf_errors" else ["BF-KLE", "BF-DeepONet"]
        ax.bar(x, heights, yerr=errs, width=0.55, color=colors, capsize=4,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("relative $L_2$ error" if ax is axes[0] else "")
        for xi, (h, e) in enumerate(zip(heights, errs)):
            ax.text(xi, h + e + 0.02 * max(heights), f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("BF-KLE vs BF-DeepONet -- jump-function benchmark\n"
                  "(mean $\\pm$ std over 10 replications; N_LF=100, N_HF=10, N_x=101)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    outpath = os.path.join(FIGDIR, "fig1_bar_panel_kle_deeponet_10reps.jpg")
    fig.savefig(outpath, dpi=300)
    print(f"Saved {outpath}")
    plt.close(fig)


def fig1_single_panel_bf_composite(kle, don):
                                                                          
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    km, ks = mean_std(kle["bf_errors"])
    dm, ds = mean_std(don["bf_errors"])
    x = [0, 1]
    heights = [km, dm]
    errs = [ks, ds]
    colors = [COLOR_KLE, COLOR_DON]
    ax.bar(x, heights, yerr=errs, width=0.55, color=colors, capsize=4,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["BF-KLE", "BF-DeepONet"], fontsize=10)
    ax.set_ylabel(r"$\mu_\varepsilon$")
    for xi, (h, e) in enumerate(zip(heights, errs)):
        ax.text(xi, h + e + 0.02 * max(heights), f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, "fig1_single_panel_bfcomposite_kle_deeponet_10reps.jpg")
    fig.savefig(outpath, dpi=300)
    print(f"Saved {outpath}")
    plt.close(fig)


def fig2_per_replication(kle, don):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    reps = np.arange(1, NREPS + 1)
    ax.plot(reps, kle["bf_errors"], "o-", color=COLOR_KLE, label="BF-KLE")
    ax.plot(reps, don["bf_errors"], "^-", color=COLOR_DON, label="BF-DeepONet")
    ax.axvline(5.5, color="gray", lw=1, ls=":")
    ax.text(5.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1, "  new reps ->",
            fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("replication")
    ax.set_ylabel("relative $L_2$ error (composite)")
    ax.set_xticks(reps)
    ax.set_title("Per-replication composite error (10 reps)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, "fig2_per_replication_kle_deeponet_10reps.jpg")
    fig.savefig(outpath, dpi=300)
    print(f"Saved {outpath}")
    plt.close(fig)


def fig3_5vs10_summary(kle, don, kle5, don5):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    labels = ["BF-KLE\n(5 reps)", "BF-KLE\n(10 reps)", "BF-DeepONet\n(5 reps)", "BF-DeepONet\n(10 reps)"]
    means = [mean_std(kle5["bf_errors"])[0], mean_std(kle["bf_errors"])[0],
             mean_std(don5["bf_errors"])[0], mean_std(don["bf_errors"])[0]]
    stds = [mean_std(kle5["bf_errors"])[1], mean_std(kle["bf_errors"])[1],
            mean_std(don5["bf_errors"])[1], mean_std(don["bf_errors"])[1]]
    colors = [COLOR_KLE, COLOR_KLE, COLOR_DON, COLOR_DON]
    x = np.arange(4)
    ax.bar(x, means, yerr=stds, width=0.6, color=colors, capsize=4,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative $L_2$ error (composite)")
    ax.set_title("Composite error: 5-rep vs 10-rep aggregate")
    for xi, (h, e) in enumerate(zip(means, stds)):
        ax.text(xi, h + e + 0.02 * max(means), f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, "fig3_5vs10rep_summary.jpg")
    fig.savefig(outpath, dpi=300)
    print(f"Saved {outpath}")
    plt.close(fig)


def fig4_predictions_samples():
\
\
\
\
\
\
\
       
    N_SAMPLES = 5
    COLOR_LF_TRUE = "#999999"
    COLOR_HF_TRUE = "#5e3c99"
    COLOR_LF_PRED = "#2a78d6"
    COLOR_BF_PRED = "#eb6834"
    LW_TRUE = 1.6
    LW_PRED = 2.8

    base = np.load(os.path.join(DATA_DIR, "jump_bifi_rep1.npz"))
    x = base["x"]
    a_grid = base["a_grid"]

    kle = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_kle.npz"))
    kle_LF_pred = kle["y_pred_LF"].T
    kle_BF_pred = kle["y_pred_BF"].T
    kle_LF_true = kle["LF_oracle"].T
    kle_HF_true = kle["HF_oracle"].T

    don = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_deeponet_jax.npz"))
    don_LF_pred = don["F_LF_o"]
    don_BF_pred = don["F_comp_o"]
    don_LF_true = don["LF_oracle"]
    don_HF_true = don["HF_oracle"]

    n_oracle = len(a_grid)
    sample_idx = np.linspace(0, n_oracle - 1, N_SAMPLES).astype(int)

    fig, axes = plt.subplots(2, N_SAMPLES, figsize=(3.1 * N_SAMPLES, 6.4), sharex=True)

    row_data = [
        (kle_LF_true, kle_HF_true, kle_LF_pred, kle_BF_pred),
        (don_LF_true, don_HF_true, don_LF_pred, don_BF_pred),
    ]

    for row, (lf_true, hf_true, lf_pred, bf_pred) in enumerate(row_data):
                                                                             
        for col, idx in enumerate(sample_idx):
            ax = axes[row, col]
            ax.plot(x, lf_true[idx], "--", color=COLOR_LF_TRUE, lw=LW_TRUE, label="Ground-truth LF")
            ax.plot(x, hf_true[idx], "--", color=COLOR_HF_TRUE, lw=LW_TRUE, label="Ground-truth HF")
            ax.plot(x, lf_pred[idx], "-", color=COLOR_LF_PRED, lw=LW_PRED, label="LF Surrogate")
            ax.plot(x, bf_pred[idx], "-", color=COLOR_BF_PRED, lw=LW_PRED, label="BF Surrogate")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                a_val = a_grid[idx]
                ax.set_title(rf"$\theta = {format_theta(a_val)}$", fontsize=11, fontname="Times New Roman")
            if row == 1:
                ax.set_xlabel("$x$", fontsize=10)
            if col == 0:
                ax.set_ylabel("$y$", fontsize=11)

                                                                                           
    for col in range(N_SAMPLES):
        ylo = min(axes[0, col].get_ylim()[0], axes[1, col].get_ylim()[0])
        yhi = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
        axes[0, col].set_ylim(ylo, yhi)
        axes[1, col].set_ylim(ylo, yhi)

    row_labels = ["BF-KLE", "BF-DeepONet"]
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.38, 0.5, label, transform=axes[row, 0].transAxes,
                           ha="right", va="center", fontsize=13, fontweight="bold",
                           fontname="Times New Roman", clip_on=False)

    fig.tight_layout(rect=[0.035, 0.03, 1, 0.94])

                                                                        
                                                     
    pos_left = axes[0, 0].get_position()
    pos_right = axes[0, -1].get_position()
    center_x = (pos_left.x0 + pos_right.x1) / 2

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(center_x, -0.03),
               prop={"family": "Times New Roman", "size": 11})
    fig.suptitle("Test Predictions (BF-KLE v/s BF-DeepONet)", fontsize=16,
                 fontweight="bold", fontname="Times New Roman", x=center_x)

    outpath = os.path.join(FIGDIR, "compare_predictions_samples.jpg")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def fig5_joint_predictions_and_summary(kle, don):
\
\
\
\
\
\
       
    N_SAMPLES = 5
    COLOR_LF_TRUE = "#999999"
    COLOR_HF_TRUE = "#5e3c99"
    LW_TRUE = 1.6
    LW_PRED = 2.6

    base = np.load(os.path.join(DATA_DIR, "jump_bifi_rep1.npz"))
    x = base["x"]
    a_grid = base["a_grid"]

    kle_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_kle.npz"))
    kle_BF_pred = kle_diag["y_pred_BF"].T
    LF_true = kle_diag["LF_oracle"].T
    HF_true = kle_diag["HF_oracle"].T

    don_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_deeponet_jax.npz"))
    don_BF_pred = don_diag["F_comp_o"]

    n_oracle = len(a_grid)
    sample_idx = np.linspace(0, n_oracle - 1, N_SAMPLES).astype(int)

    fig = plt.figure(figsize=(3.0 * N_SAMPLES, 7.6))
    gs_lines = fig.add_gridspec(1, N_SAMPLES, left=0.06, right=0.98, top=0.92, bottom=0.62, wspace=0.35)
    line_axes = [fig.add_subplot(gs_lines[0, i]) for i in range(N_SAMPLES)]

    for col, idx in enumerate(sample_idx):
        ax = line_axes[col]
        ax.plot(x, LF_true[idx], "--", color=COLOR_LF_TRUE, lw=LW_TRUE, label="Ground-truth LF")
        ax.plot(x, HF_true[idx], "--", color=COLOR_HF_TRUE, lw=LW_TRUE, label="Ground-truth HF")
        ax.plot(x, kle_BF_pred[idx], "-", color=COLOR_KLE, lw=LW_PRED, label="BF-KLE")
        ax.plot(x, don_BF_pred[idx], "-", color=COLOR_DON, lw=LW_PRED, label="BF-DeepONet")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        a_val = a_grid[idx]
        ax.set_title(rf"$\theta = {format_theta(a_val)}$", fontsize=10)
        ax.set_xlabel("$x$", fontsize=10)
        if col == 0:
            ax.set_ylabel("$y$", fontsize=11)

                                                                       
                                                                     
                                                        
    gs_bar = fig.add_gridspec(1, 1, left=0.38, right=0.62, top=0.46, bottom=0.08)
    ax_bar = fig.add_subplot(gs_bar[0, 0])
    km, ks = mean_std(kle["bf_errors"])
    dm, ds = mean_std(don["bf_errors"])
    ax_bar.bar([0, 1], [km, dm], yerr=[ks, ds], width=0.55, color=[COLOR_KLE, COLOR_DON],
               capsize=4, edgecolor="black", linewidth=0.6)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["BF-KLE", "BF-DeepONet"], fontsize=10)
    ax_bar.set_ylabel(r"$\mu_\varepsilon$")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    handles, labels = line_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.545))

    outpath = os.path.join(FIGDIR, "fig5_joint_predictions_and_composite_summary.jpg")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def fig6_joint_predictions_and_summary_altcolors(kle, don):
\
\
\
\
\
\
       
    N_SAMPLES = 5
    COLOR_HF_TRUE = "#5e3c99"
    COLOR_KLE_LINE = "#1b9e77"
    COLOR_DON_LINE = "#e7298a"
    LW_TRUE = 1.6
    LW_PRED = 2.6

    base = np.load(os.path.join(DATA_DIR, "jump_bifi_rep1.npz"))
    x = base["x"]
    a_grid = base["a_grid"]

    kle_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_kle.npz"))
    kle_BF_pred = kle_diag["y_pred_BF"].T
    HF_true = kle_diag["HF_oracle"].T

    don_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_deeponet_jax.npz"))
    don_BF_pred = don_diag["F_comp_o"]

    n_oracle = len(a_grid)
    sample_idx = np.linspace(0, n_oracle - 1, N_SAMPLES).astype(int)

    fig = plt.figure(figsize=(3.0 * N_SAMPLES, 10.2))
    gs_lines = fig.add_gridspec(2, N_SAMPLES, left=0.07, right=0.98, top=0.93, bottom=0.50,
                                 wspace=0.35, hspace=0.32)
    line_axes = [[fig.add_subplot(gs_lines[row, col]) for col in range(N_SAMPLES)] for row in range(2)]

    row_data = [
        (COLOR_KLE_LINE, kle_BF_pred, "BF-KLE"),
        (COLOR_DON_LINE, don_BF_pred, "BF-DeepONet"),
    ]

    for row, (color_pred, bf_pred, row_label) in enumerate(row_data):
        for col, idx in enumerate(sample_idx):
            ax = line_axes[row][col]
            ax.plot(x, HF_true[idx], "--", color=COLOR_HF_TRUE, lw=LW_TRUE, label="Ground-truth HF")
            ax.plot(x, bf_pred[idx], "-", color=color_pred, lw=LW_PRED, label=row_label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=11)
            if row == 0:
                a_val = a_grid[idx]
                ax.set_title(rf"$\theta = {format_theta(a_val)}$", fontsize=13)
            if row == 1:
                ax.set_xlabel("$x$", fontsize=13)
            if col == 0:
                ax.set_ylabel("$y$", fontsize=13)

                                                             
    for col in range(N_SAMPLES):
        ylo = min(line_axes[0][col].get_ylim()[0], line_axes[1][col].get_ylim()[0])
        yhi = max(line_axes[0][col].get_ylim()[1], line_axes[1][col].get_ylim()[1])
        line_axes[0][col].set_ylim(ylo, yhi)
        line_axes[1][col].set_ylim(ylo, yhi)

    handles0, labels0 = line_axes[0][0].get_legend_handles_labels()
    handles1, labels1 = line_axes[1][0].get_legend_handles_labels()
    handles = [handles0[0], handles0[1], handles1[1]]
    labels = [labels0[0], labels0[1], labels1[1]]
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False,
               fontsize=14, bbox_to_anchor=(0.53, 0.41))

                                                                         
                                                                          
    gs_bar = fig.add_gridspec(1, 1, left=0.38, right=0.62, top=0.32, bottom=0.06)
    ax_bar = fig.add_subplot(gs_bar[0, 0])
    km, ks = mean_std(kle["bf_errors"])
    dm, ds = mean_std(don["bf_errors"])
    heights = [km, dm]
    errs = [ks, ds]
    ax_bar.bar([0, 1], heights, yerr=errs, width=0.55, color=COLOR_KLE,
               capsize=4, edgecolor="black", linewidth=0.6)
    for xi, (h, e) in enumerate(zip(heights, errs)):
        ax_bar.text(xi, h + e + 0.02 * max(heights), f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["BF-KLE", "BF-DeepONet"], fontsize=13)
    ax_bar.set_ylabel(r"$\mu_\varepsilon$", fontsize=13)
    ax_bar.tick_params(axis="y", labelsize=11)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    outpath = os.path.join(FIGDIR, "fig6_joint_predictions_and_composite_summary_altcolors.jpg")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def fig7_joint_predictions_and_summary_kle_order3(kle_order3, don):
\
\
\
\
\
\
\
\
\
       
    N_SAMPLES = 5
    COLOR_HF_TRUE = "#5e3c99"
    COLOR_KLE_LINE = "#1b9e77"
    COLOR_DON_LINE = "#e7298a"
    LW_TRUE = 1.6
    LW_PRED = 2.6

    base = np.load(os.path.join(DATA_DIR, "jump_bifi_rep1.npz"))
    x = base["x"]
    a_grid = base["a_grid"]

    kle_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_kle_order3.npz"))
    kle_BF_pred = kle_diag["y_pred_BF"].T
    HF_true = kle_diag["HF_oracle"].T

    don_diag = np.load(os.path.join(DATA_DIR, "diag_fields_rep1_deeponet_jax.npz"))
    don_BF_pred = don_diag["F_comp_o"]

    n_oracle = len(a_grid)
    sample_idx = np.linspace(0, n_oracle - 1, N_SAMPLES).astype(int)

    fig = plt.figure(figsize=(3.0 * N_SAMPLES, 10.2))
    gs_lines = fig.add_gridspec(2, N_SAMPLES, left=0.07, right=0.98, top=0.93, bottom=0.50,
                                 wspace=0.35, hspace=0.32)
    line_axes = [[fig.add_subplot(gs_lines[row, col]) for col in range(N_SAMPLES)] for row in range(2)]

    row_data = [
        (COLOR_KLE_LINE, kle_BF_pred, "BF-KLE (order=3)"),
        (COLOR_DON_LINE, don_BF_pred, "BF-DeepONet"),
    ]

    for row, (color_pred, bf_pred, row_label) in enumerate(row_data):
        for col, idx in enumerate(sample_idx):
            ax = line_axes[row][col]
            ax.plot(x, HF_true[idx], "--", color=COLOR_HF_TRUE, lw=LW_TRUE, label="Ground-truth HF")
            ax.plot(x, bf_pred[idx], "-", color=color_pred, lw=LW_PRED, label=row_label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=11)
            if row == 0:
                a_val = a_grid[idx]
                ax.set_title(rf"$\theta = {format_theta(a_val)}$", fontsize=13)
            if row == 1:
                ax.set_xlabel("$x$", fontsize=13)
            if col == 0:
                ax.set_ylabel("$y$", fontsize=13)

                                                             
    for col in range(N_SAMPLES):
        ylo = min(line_axes[0][col].get_ylim()[0], line_axes[1][col].get_ylim()[0])
        yhi = max(line_axes[0][col].get_ylim()[1], line_axes[1][col].get_ylim()[1])
        line_axes[0][col].set_ylim(ylo, yhi)
        line_axes[1][col].set_ylim(ylo, yhi)

    handles0, labels0 = line_axes[0][0].get_legend_handles_labels()
    handles1, labels1 = line_axes[1][0].get_legend_handles_labels()
    handles = [handles0[0], handles0[1], handles1[1]]
    labels = [labels0[0], labels0[1], labels1[1]]
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False,
               fontsize=14, bbox_to_anchor=(0.53, 0.41))

                                                                         
                                                                          
    gs_bar = fig.add_gridspec(1, 1, left=0.38, right=0.62, top=0.32, bottom=0.06)
    ax_bar = fig.add_subplot(gs_bar[0, 0])
    km, ks = mean_std(kle_order3["bf_errors"])
    dm, ds = mean_std(don["bf_errors"])
    heights = [km, dm]
    errs = [ks, ds]
    ax_bar.bar([0, 1], heights, yerr=errs, width=0.55, color=COLOR_KLE,
               capsize=4, edgecolor="black", linewidth=0.6)
    for xi, (h, e) in enumerate(zip(heights, errs)):
        ax_bar.text(xi, h + e + 0.02 * max(heights), f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["BF-KLE\n(order=3)", "BF-DeepONet"], fontsize=13)
    ax_bar.set_ylabel(r"$\mu_\varepsilon$", fontsize=13)
    ax_bar.tick_params(axis="y", labelsize=11)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    outpath = os.path.join(FIGDIR, "fig7_joint_predictions_and_composite_summary_kle_order3.jpg")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def main():
    kle = np.load(os.path.join(RESULT_DIR, "results_bfkle_10reps.npz"))
    don = np.load(os.path.join(RESULT_DIR, "results_deeponet_jax_10reps.npz"))
    fig6_joint_predictions_and_summary_altcolors(kle, don)


if __name__ == "__main__":
    main()
