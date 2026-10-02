                                                                            

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
APPENDIX_DIR = SCRIPT_DIR.parent
DEFAULT_RESULT_DIR = APPENDIX_DIR / "data" / "results_accuracy_compute_codex"
DEFAULT_FIGURE_DIR = APPENDIX_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figures", type=pathlib.Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--compilation-inclusive-only",
        action="store_true",
        help=(
            "Generate only the companion accuracy--time figure with the "
            "one-time Julia and XLA compilation costs included."
        ),
    )
    return parser.parse_args()


def median_iqr(values):
    median = np.median(values)
    lower, upper = np.quantile(values, [0.25, 0.75])
    return median, median - lower, upper - median


def save_figure(fig, path: pathlib.Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95, "optimize": True, "subsampling": 0},
    )
    plt.close(fig)
    print(f"Saved {path}")


def plot_fixed_recipe(bfkle, deeponet, figure_dir):
    n = min(len(bfkle["fit_seconds"]), len(deeponet["fit_seconds"]))
    labels = ["BF-KLE\nwarm fit", f"BF-DeepONet\n{int(deeponet['niter'])} steps"]
    samples = [bfkle["fit_seconds"][:n], deeponet["fit_seconds"][:n]]
    medians = np.asarray([np.median(values) for values in samples])

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(labels, medians, color=["#4477AA", "#CC6677"], alpha=0.78)
    offsets = np.linspace(-0.055, 0.055, n) if n > 1 else np.zeros(1)
    for method, values in enumerate(samples):
        ax.scatter(
            method + offsets,
            values,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            s=38,
            zorder=4,
            label="individual replications" if method == 0 else None,
        )
    ax.scatter(
        0,
        float(bfkle["cold_fit_seconds"]),
        marker="D",
        color="#228833",
        edgecolor="black",
        s=48,
        zorder=5,
        label="BF-KLE cold first fit",
    )
    ax.scatter(
        1,
        float(deeponet["compile_seconds"]),
        marker="s",
        color="#AA3377",
        edgecolor="black",
        s=48,
        zorder=5,
        label="DeepONet XLA compile only",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Offline fitting time [s]")
    ax.set_title(f"Fixed-recipe offline fitting cost (n={n} matched runs)")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    save_figure(fig, figure_dir / "fixed_recipe_accuracy_compute_codex.jpg")


def plot_matched_accuracy(bfkle, deeponet, figure_dir):
    n = min(len(bfkle["composite_errors"]), len(deeponet["matched_errors"]))
    labels = ["BF-KLE\ncomplete fit", "BF-DeepONet\ntime matched"]
    samples = [bfkle["composite_errors"][:n], deeponet["matched_errors"][:n]]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for rep in range(n):
        ax.plot([0, 1], [samples[0][rep], samples[1][rep]], color="0.7", linewidth=1)
        ax.scatter(0, samples[0][rep], color="#4477AA", edgecolor="black", zorder=4)
        ax.scatter(1, samples[1][rep], color="#CC6677", edgecolor="black", zorder=4)
    for method, values in enumerate(samples):
        median = np.median(values)
        ax.plot(
            [method - 0.15, method + 0.15],
            [median, median],
            color="black",
            linewidth=2.2,
        )
    ax.set_xticks([0, 1], labels)
    ax.set_yscale("log")
    ax.set_ylabel(r"Relative $L^2$ error")
    budget = float(deeponet["matched_budget_seconds"])
    ax.set_title(
        f"Accuracy at BF-KLE's {budget:.4g} s budget\n"
        f"DeepONet shown at first completed step (n={n})"
    )
    ax.grid(axis="y", which="both", alpha=0.25)
    save_figure(fig, figure_dir / "wallclock_matched_accuracy_compute_codex.jpg")


def plot_accuracy_curve(bfkle, deeponet, figure_dir):
    n = min(len(bfkle["composite_errors"]), len(deeponet["curve_errors"]))
    times = deeponet["curve_seconds"][:n]
    errors = deeponet["curve_errors"][:n]
    median_times = np.median(times, axis=0)
    median_errors = np.median(errors, axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for rep in range(n):
        ax.plot(
            times[rep],
            errors[rep],
            "o-",
            linewidth=1.0,
            markersize=3.5,
            alpha=0.72,
            label=f"BF-DeepONet replication {rep + 1}",
        )
    ax.plot(
        median_times,
        median_errors,
        "k--",
        linewidth=1.3,
        label="two-run median (visual guide)",
    )
    ax.scatter(
        bfkle["fit_seconds"][:n],
        bfkle["composite_errors"][:n],
        s=65,
        marker="D",
        color="#4477AA",
        edgecolor="black",
        zorder=5,
        label="BF-KLE matched replications",
    )
    steps = np.asarray(deeponet["curve_steps"])
    for step in [1, 10, 100, 1000, int(steps[-1])]:
        locations = np.flatnonzero(steps == step)
        if locations.size:
            index = int(locations[0])
            ax.annotate(
                str(step),
                (median_times[index], median_errors[index]),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative warm fitting time [s]")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_title(f"Accuracy versus offline fitting time (n={n}; labels are iterations)")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=7.5)
    save_figure(fig, figure_dir / "error_vs_fit_time_accuracy_compute_codex.jpg")


def plot_accuracy_curve_including_compilation(bfkle, deeponet, figure_dir):
                                                                                 
    n = min(len(bfkle["composite_errors"]), len(deeponet["curve_errors"]))
    xla_seconds = float(deeponet["compile_seconds"])
    julia_seconds = float(bfkle["compile_estimate_seconds"])
    times = deeponet["curve_seconds"][:n] + xla_seconds
    errors = deeponet["curve_errors"][:n]
    bfkle_times = bfkle["fit_seconds"][:n] + julia_seconds
    median_times = np.median(times, axis=0)
    median_errors = np.median(errors, axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    run_colors = ["#D62728", "#17BECF"]
    for rep in range(n):
        ax.plot(
            times[rep],
            errors[rep],
            "o-",
            color=run_colors[rep % len(run_colors)],
            linewidth=2.0,
            markersize=3.5,
            alpha=0.85,
            label=f"BF-DeepONet replication {rep + 1}",
        )
    ax.scatter(
        bfkle_times,
        bfkle["composite_errors"][:n],
        s=65,
        marker="D",
        color="#4D4D4D",
        edgecolor="black",
        zorder=5,
        label="BF-KLE matched replications",
    )
    steps = np.asarray(deeponet["curve_steps"])
    for step in [10, 100, 1000, int(steps[-1])]:
        locations = np.flatnonzero(steps == step)
        if locations.size:
            index = int(locations[0])
            annotation = f"{step} iterations" if step in {10, 1000} else str(step)
            ax.annotate(
                annotation,
                (median_times[index], median_errors[index]),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative fitting time [s]")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=7.5)
    save_figure(
        fig,
        figure_dir
        / "error_vs_total_time_including_compilation_accuracy_compute_codex.jpg",
    )


def write_budget_table(deeponet, path: pathlib.Path) -> None:
                                                                              
    steps = np.asarray(deeponet["curve_steps"], dtype=int)
    times = np.asarray(deeponet["curve_seconds"])
    errors = np.asarray(deeponet["curve_errors"])
    matched_steps = np.asarray(deeponet["matched_steps"], dtype=int)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "replication",
                "iteration",
                "cumulative_fit_seconds",
                "relative_l2_error",
                "is_first_step_exceeding_bfkle_budget",
            ]
        )
        for rep in range(times.shape[0]):
            for column, step in enumerate(steps):
                writer.writerow(
                    [
                        rep + 1,
                        int(step),
                        f"{times[rep, column]:.9f}",
                        f"{errors[rep, column]:.9e}",
                        int(step == matched_steps[rep]),
                    ]
                )
    print(f"Saved {path}")


def write_summary(bfkle, deeponet, path: pathlib.Path) -> None:
    n = min(len(bfkle["composite_errors"]), len(deeponet["final_errors"]))
    steps = np.asarray(deeponet["curve_steps"], dtype=int)
    times = np.asarray(deeponet["curve_seconds"][:n])
    errors = np.asarray(deeponet["curve_errors"][:n])
    budget = float(deeponet["matched_budget_seconds"])
    lines = [
        "# BF-KLE versus BF-DeepONet accuracy--compute results",
        "",
        (
            "This report records the measured offline fitting costs and relative "
            "errors at selected DeepONet iteration budgets for comparison with "
            "BF-KLE."
        ),
        "",
        "## Scope and interpretation",
        "",
        (
            f"This is a preliminary two-run sensitivity comparison using the first "
            f"{n} matched data replications. It is not an uncertainty estimate; the "
            "figures therefore show individual runs and do not use standard-deviation "
            "or interquartile uncertainty bands."
        ),
        "",
        (
            f"Both methods use the fixed 100-LF/10-HF training-data budget. "
            f"DeepONet uses {int(deeponet['niter'])} optimizer steps and FP64. "
            "Fitting timers exclude data loading, prediction/error evaluation, "
            "plotting, and checkpoint I/O."
        ),
        "",
        "## Timing and final accuracy",
        "",
        "| Replication | BF-KLE warm fit (s) | BF-KLE error | DeepONet fit (s) | DeepONet final error | DeepONet/BF-KLE time |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rep in range(n):
        kle_time = float(bfkle["fit_seconds"][rep])
        deep_time = float(deeponet["fit_seconds"][rep])
        lines.append(
            f"| {rep + 1} | {kle_time:.6f} | "
            f"{float(bfkle['composite_errors'][rep]):.6f} | {deep_time:.3f} | "
            f"{float(deeponet['final_errors'][rep]):.6f} | "
            f"{deep_time / kle_time:,.0f}x |"
        )
    lines.extend(
        [
            "",
            f"BF-KLE cold first fit: {float(bfkle['cold_fit_seconds']):.3f} s; "
            f"estimated Julia compilation contribution: "
            f"{float(bfkle['compile_estimate_seconds']):.3f} s.",
            "",
            f"DeepONet XLA compilation: {float(deeponet['compile_seconds']):.3f} s.",
            "",
            "## Accuracy at the BF-KLE warm fitting-time budget",
            "",
            f"The median BF-KLE warm-fit budget is {budget:.9f} s. Even the first "
            "completed DeepONet step exceeds it, so no sub-step accuracy is inferred "
            "or interpolated.",
            "",
            "| Replication | BF-KLE error | DeepONet step | DeepONet time (s) | Budget overrun | DeepONet error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rep in range(n):
        matched_time = float(deeponet["matched_seconds"][rep])
        lines.append(
            f"| {rep + 1} | {float(bfkle['composite_errors'][rep]):.6f} | "
            f"{int(deeponet['matched_steps'][rep])} | {matched_time:.6f} | "
            f"{matched_time / budget:.1f}x | "
            f"{float(deeponet['matched_errors'][rep]):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Logged DeepONet iteration budgets",
            "",
            "| Iteration | Rep. 1 time (s) | Rep. 1 error | Rep. 2 time (s) | Rep. 2 error |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for column, step in enumerate(steps):
        rep_values = []
        for rep in range(min(n, 2)):
            rep_values.extend(
                [f"{times[rep, column]:.6f}", f"{errors[rep, column]:.6e}"]
            )
        while len(rep_values) < 4:
            rep_values.extend(["--", "--"])
        lines.append(f"| {step} | " + " | ".join(rep_values) + " |")
    lines.extend(
        [
            "",
            "The iteration-2 spike in replication 1 is a measured optimization "
            "transient, not a plotting error. With only two runs, differences between "
            "replications should be read as sensitivity evidence rather than as a "
            "stable population statistic.",
            "",
            (
                f"*Generated {dt.date.today().isoformat()} from "
                "`bfkle_accuracy_compute_codex.npz` and "
                "`deeponet_accuracy_compute_codex.npz`.*"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    bfkle_path = args.results / "bfkle_accuracy_compute_codex.npz"
    deeponet_path = args.results / "deeponet_accuracy_compute_codex.npz"
    if not bfkle_path.exists() or not deeponet_path.exists():
        raise FileNotFoundError(
            "Run both benchmark scripts before plotting: "
            f"missing {[str(p) for p in (bfkle_path, deeponet_path) if not p.exists()]}"
        )

    args.figures.mkdir(parents=True, exist_ok=True)
    with np.load(bfkle_path) as bfkle, np.load(deeponet_path) as deeponet:
        if args.compilation_inclusive_only:
            plot_accuracy_curve_including_compilation(bfkle, deeponet, args.figures)
            return
        plot_fixed_recipe(bfkle, deeponet, args.figures)
        plot_matched_accuracy(bfkle, deeponet, args.figures)
        plot_accuracy_curve(bfkle, deeponet, args.figures)
        write_budget_table(
            deeponet, args.results / "training_budget_accuracy_compute_codex.csv"
        )
        write_summary(
            bfkle, deeponet, args.results / "accuracy_compute_summary_codex.md"
        )


if __name__ == "__main__":
    main()
