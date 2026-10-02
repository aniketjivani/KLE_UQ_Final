
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies"))
import register_parula as rp

plt.rc("font", family="serif")
parula_colors = rp._parula_data

RESULTS_DIR = ROOT / "data/exp8"
FIGS_DIR = ROOT / "figures/exp8_02_frozenlf"
os.makedirs(FIGS_DIR, exist_ok=True)

npz_name = sys.argv[1] if len(sys.argv) > 1 else "exp8_02_heatmap_arrays.npz"
d = np.load(os.path.join(RESULTS_DIR, npz_name))
is_bsine = "bsine" in npz_name.lower()
case_label = "C2 (bSine LF)" if is_bsine else "C1 (Taylor LF, k=5)"
fname_suffix = "_c1bsine" if is_bsine else ""
N_PILOT_HF = int(d["N_PILOT_HF"])
N_ACQUIRED = int(d["N_ACQUIRED"])
rep_ids = d["rep_ids"].astype(int)

gp_stacked = np.concatenate((d["gp_on_gp"], d["gp_on_ra"]), axis=0)
ra_stacked = np.concatenate((d["ra_on_ra"], d["ra_on_gp"]), axis=0)

step_x, step_y = [], []
for i in range(N_ACQUIRED):
    step_x.extend([i - 0.5, i + 1 - 0.5])
    step_y.extend([i + N_PILOT_HF - 0.5, i + N_PILOT_HF - 0.5])

xticks = np.array([0, 9, 19, 29, 39, 49, 59])
xticklabels = [1, 10, 20, 30, 40, 50, 60]
yticks = np.array([0, 9, 19, 24, 29, 34, 44, 54])
yticklabels = [1, 10, 20, 25, 30, 35, 45, 55]

for rep_i, repID in enumerate(rep_ids):
    gp_results = gp_stacked[:, :, rep_i]
    ra_results = ra_stacked[:, :, rep_i]

    min_cbar = min(gp_results.min(), ra_results.min())
    max_cbar = max(gp_results.max(), ra_results.max())

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    im0 = ax[0].imshow(gp_results, cmap=ListedColormap(parula_colors),
                        norm=colors.LogNorm(vmin=min_cbar, vmax=max_cbar))
    ax[0].set_xticks(xticks)
    ax[0].set_yticks(yticks)
    ax[0].set_xticklabels(xticklabels, fontsize=16)
    ax[0].set_yticklabels(yticklabels, fontsize=16)
    ax[0].plot(step_x, step_y, 'k', linewidth=3)
    ax[0].axhline(y=(N_PILOT_HF + N_ACQUIRED - 0.5), color='r', linewidth=4.5)
    ax[0].set_xlabel("Acquisition Stage", fontsize=20)
    ax[0].set_ylabel("High-fidelity Run IDs (65 onwards via RS)", fontsize=20)
    ax[0].set_title("BF-KLE-AL (frozen LF, rank=2)", fontsize=18)
    cbar0 = fig.colorbar(im0, fraction=0.046, pad=0.04, ax=ax[0])
    cbar0.ax.tick_params(labelsize=16)

    im1 = ax[1].imshow(ra_results, cmap=ListedColormap(parula_colors),
                        norm=colors.LogNorm(vmin=min_cbar, vmax=max_cbar))
    ax[1].set_xticks(xticks)
    ax[1].set_yticks(yticks)
    ax[1].set_xticklabels(xticklabels, fontsize=16)
    ax[1].set_yticklabels(yticklabels, fontsize=16)
    ax[1].plot(step_x, step_y, 'k', linewidth=3)
    ax[1].axhline(y=(N_PILOT_HF + N_ACQUIRED - 0.5), color='r', linewidth=4.5)
    ax[1].set_xlabel("Acquisition Stage", fontsize=20)
    ax[1].set_ylabel("High-fidelity Run IDs (65 onwards via AL)", fontsize=20)
    ax[1].set_title("BF-KLE-RS (frozen LF, rank=2)", fontsize=18)
    cbar1 = fig.colorbar(im1, fraction=0.046, pad=0.04, ax=ax[1])
    cbar1.ax.tick_params(labelsize=16)

    fig.suptitle("rep_{:03d} -- {}, frozen-LF-surrogate intervention".format(repID, case_label), fontsize=16, y=0.98)
    fig.tight_layout()

    outpath = os.path.join(FIGS_DIR, "exp8_02_frozenlf_heatmap_rep{:03d}{}.png".format(repID, fname_suffix))
    fig.savefig(outpath, dpi=200)
    print("saved:", outpath)
    plt.close(fig)
