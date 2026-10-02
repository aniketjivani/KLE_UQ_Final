from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_LF = 100
N_DELTA = 60
N_HF_ONLY = 85  # ceil(100 / 4) + 60
N_MASTER = N_LF + N_HF_ONLY
SEED = 20260102  # BASE_SEED + first replication


def latin_hypercube(n_points: int, rng: np.random.Generator) -> np.ndarray:
    unit_design = np.column_stack(
        [(rng.permutation(n_points) + rng.random(n_points)) / n_points for _ in range(2)]
    )
    return 2.0 * unit_design - 1.0


def maximin_subset(points: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    selected = np.empty(n_select, dtype=int)
    selected[0] = rng.integers(len(points))
    nearest_sqdist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    nearest_sqdist[selected[0]] = -np.inf

    for i in range(1, n_select):
        selected[i] = np.argmax(nearest_sqdist)
        candidate_sqdist = np.sum((points - points[selected[i]]) ** 2, axis=1)
        nearest_sqdist = np.minimum(nearest_sqdist, candidate_sqdist)
        nearest_sqdist[selected[: i + 1]] = -np.inf
    return selected


def main() -> None:
    rng = np.random.default_rng(SEED)

    master_plan = latin_hypercube(N_MASTER, rng)
    lf_plan = master_plan[maximin_subset(master_plan, N_LF, rng)]
    delta_plan = lf_plan[maximin_subset(lf_plan, N_DELTA, rng)]
    hf_only_plan = master_plan[maximin_subset(master_plan, N_HF_ONLY, rng)]

    plt.style.use("seaborn-v0_8-notebook")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    lf = ax.scatter(
        lf_plan[:, 0],
        lf_plan[:, 1],
        s=48,
        marker="o",
        facecolor="#c9c9c9",
        edgecolor="#555555",
        linewidth=0.45,
        label="LF-KLE",
        zorder=1,
    )
    hf = ax.scatter(
        hf_only_plan[:, 0],
        hf_only_plan[:, 1],
        s=82,
        marker="^",
        facecolor="#e17c05",
        edgecolor="#2c2c2c",
        linewidth=0.55,
        label="HF-KLE",
        zorder=2,
    )
    delta = ax.scatter(
        delta_plan[:, 0],
        delta_plan[:, 1],
        s=132,
        marker="*",
        facecolor="#0072b2",
        edgecolor="#1f1f1f",
        linewidth=0.4,
        label="Discrepancy KLE",
        zorder=3,
    )

    ax.set_xlabel(r"$\xi_1$", fontsize=20)
    ax.set_ylabel(r"$\xi_2$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8d8d8", linewidth=0.7)
    ax.legend(
        handles=[lf, delta, hf],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=14,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    output = Path(__file__).resolve().parent / "014_lf_delta_hf_design_r4.jpg"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
