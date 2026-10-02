
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = (
    ROOT
    / "data/exp11/rep005_gp_bf_error_increase_scan_stage024_025.npz"
)
OUT_FILE = (
    ROOT
    / "figures/exp11b_rep005_gp_bf_error_increase_stage024_rank2_vs_stage025_rank1.png"
)
LF_OUT_FILE = (
    ROOT
    / "figures/exp11b_rep005_gp_lf_predictions_bf_error_increase_points_stage024_025.png"
)
DELTA_OUT_FILE = (
    ROOT
    / "figures/exp11b_rep005_gp_delta_predictions_bf_error_increase_points_stage024_025.png"
)


def main() -> None:
    d = np.load(RESULT_FILE)
    stages = d["stages"].astype(int)
    lf_ranks = d["lf_ranks"].astype(int)
    delta_ranks = d["delta_ranks"].astype(int)
    x = d["x"]
    points = d["ab_points"]
    hf_truth = d["hf_oracle"]
    lf_truth = d["lf_oracle"]
    delta_truth = d["delta_oracle"]
    lf_prediction = d["lf_prediction"]
    delta_prediction = d["delta_prediction"]
    bf_prediction = d["bf_prediction"]
    lf_errors = d["lf_relative_l2_error"]
    delta_errors = d["delta_relative_l2_error"]
    errors = d["bf_relative_l2_error"]
    increases = d["bf_error_increase"]

    assert stages.tolist() == [24, 25]
    assert lf_ranks.tolist() == [2, 1]
    assert np.all(increases > 0)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey="col")
    for stage_i, (stage, lf_rank, delta_rank) in enumerate(zip(stages, lf_ranks, delta_ranks)):
        for point_i, (a, b) in enumerate(points):
            ax = axes[stage_i, point_i]
            ax.plot(
                x,
                lf_truth[point_i],
                color="black",
                linestyle="--",
                linewidth=1.7,
                label="Ground-truth LF",
            )
            ax.plot(
                x,
                lf_prediction[stage_i, point_i],
                color="tab:red",
                linewidth=1.0,
                alpha=0.65,
                label="LF-KLE",
            )
            ax.plot(
                x,
                hf_truth[point_i],
                color="black",
                linestyle=":",
                linewidth=1.9,
                label="Ground-truth HF",
            )
            ax.plot(
                x,
                bf_prediction[stage_i, point_i],
                color="tab:blue",
                linewidth=1.9,
                label="BF-KLE",
            )
            if stage_i == 0:
                ax.set_title(rf"$a={a:.1f},\ b={b:.1f}$")
            if point_i == 0:
                ax.set_ylabel(f"stage {stage}\nLF rank {lf_rank}, Delta rank {delta_rank}")
            if stage_i == 1:
                ax.set_xlabel(r"$x$")
            ax.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_FILE}")

    def plot_component(*, truth, prediction, errors_component, color, truth_label, prediction_label, title, out_file):
        fig_component, axes_component = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey="col")
        for stage_i, (stage, lf_rank, delta_rank) in enumerate(zip(stages, lf_ranks, delta_ranks)):
            for point_i, (a, b) in enumerate(points):
                ax = axes_component[stage_i, point_i]
                ax.plot(x, truth[point_i], color="black", linestyle="--", linewidth=1.7, label=truth_label)
                ax.plot(x, prediction[stage_i, point_i], color=color, linewidth=1.9, label=prediction_label)
                if stage_i == 0:
                    ax.set_title(rf"$a={a:.1f},\ b={b:.1f}$")
                if point_i == 0:
                    ax.set_ylabel(f"stage {stage}\nLF rank {lf_rank}, Delta rank {delta_rank}")
                if stage_i == 1:
                    ax.set_xlabel("x")
                change = errors_component[1, point_i] - errors_component[0, point_i]
                annotation = rf"rel. $L^2$={errors_component[stage_i, point_i]:.3e}"
                if stage_i == 1:
                    annotation += "\n" + rf"change={change:+.3e}"
                ax.text(
                    0.03,
                    0.06,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
                )
                ax.grid(alpha=0.25, linestyle="--")

        handles_component, labels_component = axes_component[0, 0].get_legend_handles_labels()
        fig_component.legend(
            handles_component,
            labels_component,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
        fig_component.suptitle(title, y=1.06)
        fig_component.tight_layout(rect=(0, 0, 1, 0.92))
        fig_component.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig_component)
        print(f"saved: {out_file}")

    plot_component(
        truth=lf_truth,
        prediction=lf_prediction,
        errors_component=lf_errors,
        color="tab:red",
        truth_label="Taylor LF truth",
        prediction_label="LF-KLE prediction",
        title="Rep 5 GP/AL: LF-KLE predictions at scan-selected BF-error-increase points",
        out_file=LF_OUT_FILE,
    )
    plot_component(
        truth=delta_truth,
        prediction=delta_prediction,
        errors_component=delta_errors,
        color="tab:purple",
        truth_label="true HF − LF",
        prediction_label="Delta-KLE prediction",
        title="Rep 5 GP/AL: Delta-KLE predictions at scan-selected BF-error-increase points",
        out_file=DELTA_OUT_FILE,
    )


if __name__ == "__main__":
    main()
