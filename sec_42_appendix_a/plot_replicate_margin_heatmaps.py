from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BIFI_ROOT = SCRIPT_DIR.parent / "bifi_regime_study"
BUDGETS = [100, 125, 150]
R_COLS = [4, 8, 16, "freeLF"]
R_LABELS = ["r=4", "r=8", "r=16", "free-LF"]


def nonfrozen_bs_tag(value, budget):
    suffix = {100: "", 125: "_budget125", 150: "_budget150"}[budget]
    return f"BS_s{value:.3f}_case2{suffix}"


def nonfrozen_t_tag(value, budget, xmax005):
    suffix = {100: "", 125: "_budget125", 150: "_budget150"}[budget]
    xmax = "_xmax005" if xmax005 else ""
    return f"T_k{value}_b6080{xmax}{suffix}"


def preexisting_bs_tag(value, budget):
    return f"BS_s{value:.3f}_case2_preexistingLF_budget{budget}"


def preexisting_t_tag(value, budget, xmax005):
    xmax = "_xmax005" if xmax005 else ""
    return f"T_k{value}_b6080{xmax}_budget{budget}"


SPECS = [
    ("lf_pilot_cost_accounted", "non_frozen_LF", "007_alt_regime_BS_original_compare_replicate_margin_summary.jpg", [0.0, 0.5, 1.0], ["s=0.00", "s=0.50", "s=1.00"], nonfrozen_bs_tag),
    ("lf_pilot_cost_accounted", "non_frozen_LF", "006_alt_regime_T_original_compare_replicate_margin_summary.jpg", [3, 5, 7], ["k=3", "k=5", "k=7"], lambda value, budget: nonfrozen_t_tag(value, budget, False)),
    ("lf_pilot_cost_accounted", "non_frozen_LF", "008_alt_regime_T_xmax005_compare_replicate_margin_summary.jpg", [3, 5, 7], ["k=3", "k=5", "k=7"], lambda value, budget: nonfrozen_t_tag(value, budget, True)),
    ("preexisting_lf_pilot", "frozen_LF", "005_alt_regime_BS_frozenLF_compare_replicate_margin_summary.jpg", [0.0, 0.5, 1.0], ["s=0.00", "s=0.50", "s=1.00"], preexisting_bs_tag),
    ("preexisting_lf_pilot", "frozen_LF", "004_alt_regime_T_frozenLF_compare_replicate_margin_summary.jpg", [3, 5, 7], ["k=3", "k=5", "k=7"], lambda value, budget: preexisting_t_tag(value, budget, False)),
    ("preexisting_lf_pilot", "frozen_LF", "008_alt_regime_T_frozenLF_xmax005_compare_replicate_margin_summary.jpg", [3, 5, 7], ["k=3", "k=5", "k=7"], lambda value, budget: preexisting_t_tag(value, budget, True)),
]


def replication_margins(path):
    data = np.load(path)
    bf_last = np.asarray(data["bf_errors"][:, -1], dtype=float)
    hf_last = np.asarray(data["hf_errors"][:, -1], dtype=float)
    if bf_last.shape != hf_last.shape or np.any(hf_last == 0.0):
        raise ValueError(f"Invalid final-stage errors in {path}")
    return (hf_last - bf_last) / hf_last


def mean_matrices(study, row_values, tag_function):
    data_dir = BIFI_ROOT / "outputs" / study / "data"
    means = {}
    for budget in BUDGETS:
        mean_matrix = np.empty((len(row_values), len(R_COLS)))
        for row, value in enumerate(row_values):
            for col, r_col in enumerate(R_COLS):
                suffix = "freeLF" if r_col == "freeLF" else f"r{r_col}"
                path = data_dir / f"{study}_{tag_function(value, budget)}_{suffix}.npz"
                margins = replication_margins(path)
                mean_matrix[row, col] = margins.mean()
        means[budget] = mean_matrix
    return means


def text_color(image, value):
    red, green, blue, _ = image.cmap(image.norm(value))
    return "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"


def draw_matrix(ax, matrix, cmap, vmin, vmax, row_labels, title, show_ylabels):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(R_LABELS)))
    ax.set_xticklabels(R_LABELS, fontsize=13, fontweight="bold", fontname="Times New Roman")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels if show_ylabels else [], fontsize=14, fontweight="bold", fontname="Times New Roman")
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=16)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=12, color=text_color(image, value))
    return image


def plot_spec(spec):
    study, _, output_name, row_values, row_labels, tag_function = spec
    means = mean_matrices(study, row_values, tag_function)
    mean_limit = np.nanmax(np.abs(np.concatenate([matrix.ravel() for matrix in means.values()])))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for col, budget in enumerate(BUDGETS):
        title = r"$N_{\mathrm{LF}} = %d$" % budget
        mean_image = draw_matrix(axes[col], means[budget], "RdBu", -mean_limit, mean_limit, row_labels, title, col == 0)

    fig.subplots_adjust(left=0.12, right=0.86, top=0.87, bottom=0.13, wspace=0.22)
    mean_cax = fig.add_axes([0.89, 0.18, 0.018, 0.58])
    mean_bar = fig.colorbar(mean_image, cax=mean_cax)
    mean_cax.tick_params(labelsize=12)

    output = SCRIPT_DIR / output_name
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    plt.style.use("seaborn-v0_8-notebook")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    for spec in SPECS:
        print(plot_spec(spec))


if __name__ == "__main__":
    main()
