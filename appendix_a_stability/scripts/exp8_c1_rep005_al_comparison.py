
from pathlib import Path
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies"))
import register_parula as rp  # noqa: E402


REP_ID = 5
N_PILOT_HF = 5
N_ACQUIRED = 60

BASELINE_FILE = ROOT / "data/exp8/err_heatmap_all_reps_case_002_EI_nolog.npz"
FIXED_FILE = ROOT / "data/exp8/exp8_heatmap_arrays.npz"
FROZEN_FILE = ROOT / "data/exp8/exp8_02_heatmap_arrays.npz"
OUT_DIR = ROOT / "figures"
OUT_FILE = OUT_DIR / "exp8_c1_rep005_bf_kle_al_rank_comparison.png"


def rep_index(data: np.lib.npyio.NpzFile, rep_id: int) -> int:
    rep_ids = np.asarray(data["rep_ids"], dtype=int)
    matches = np.flatnonzero(rep_ids == rep_id)
    if matches.size != 1:
        raise ValueError(f"expected one entry for Rep {rep_id}, found {matches.size}")
    return int(matches[0])


def stacked_al_panel(data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    gp_on_gp = np.asarray(data["gp_on_gp"], dtype=float)[: N_PILOT_HF + N_ACQUIRED, :N_ACQUIRED, index]
    gp_on_ra = np.asarray(data["gp_on_ra"], dtype=float)[:N_ACQUIRED, :N_ACQUIRED, index]
    panel = np.concatenate((gp_on_gp, gp_on_ra), axis=0)
    if panel.shape != (125, 60) or not np.isfinite(panel).all() or np.any(panel <= 0):
        raise ValueError(f"invalid BF-KLE-AL panel: shape={panel.shape}")
    return panel


def main() -> None:
    with np.load(BASELINE_FILE) as baseline, np.load(FIXED_FILE) as fixed, np.load(FROZEN_FILE) as frozen:
        panels = [
            stacked_al_panel(baseline, REP_ID - 1),
            stacked_al_panel(fixed, rep_index(fixed, REP_ID)),
            stacked_al_panel(frozen, rep_index(frozen, REP_ID)),
        ]

    titles = ("Original", "Fixed-rank LF", "Frozen LF-KLE")
    positive = np.concatenate([panel.ravel() for panel in panels])
    norm = colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    cmap = ListedColormap(rp._parula_data)

    step_x, step_y = [], []
    for stage in range(N_ACQUIRED):
        step_x.extend([stage - 0.5, stage + 0.5])
        step_y.extend([
            stage + N_PILOT_HF - 0.5,
            stage + N_PILOT_HF - 0.5,
        ])

    plt.rc("font", family="serif")
    fig = plt.figure(figsize=(16.5, 7.2))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.055), wspace=0.14)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[0, col], sharex=axes[0], sharey=axes[0]) for col in (1, 2))
    colorbar_axis = fig.add_subplot(grid[0, 3])
    image = None
    for axis, panel, title in zip(axes, panels, titles):
        image = axis.imshow(panel, cmap=cmap, norm=norm, aspect="auto")
        axis.plot(step_x, step_y, color="black", linewidth=2.5)
        axis.axhline(N_PILOT_HF + N_ACQUIRED - 0.5, color="red", linewidth=3.5)
        axis.set_title(title, fontsize=19)
        axis.set_xticks([0, 9, 19, 29, 39, 49, 59])
        axis.set_xticklabels([1, 10, 20, 30, 40, 50, 60], fontsize=14)
        axis.tick_params(axis="y", labelsize=14)

    axes[0].set_yticks([0, 9, 19, 24, 29, 34, 44, 54])
    axes[0].set_yticklabels([1, 10, 20, 25, 30, 35, 45, 55])
    axes[0].set_ylabel("High-fidelity Run IDs (65 onwards via RS)", fontsize=18)
    for axis in axes[1:]:
        axis.tick_params(axis="y", which="both", left=False, labelleft=False)
    fig.supxlabel("Acquisition Stage", fontsize=19, y=0.04)

    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Relative L1 error", fontsize=16)
    colorbar.ax.tick_params(labelsize=13)

    fig.subplots_adjust(left=0.085, right=0.94, bottom=0.13, top=0.91)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
