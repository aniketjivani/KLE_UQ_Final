# %%
# show heatmaps that summarize how relative error trends change as more points are acquired in each case.
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import os
import numpy as np
import sys

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/")

import register_parula as rp
parula_colors = rp._parula_data
N_ACQUIRED = 40
N_PILOT_HF = 10
all_modes_delta = 0
acqFunc = "EI"
saveFig = True

err_comparison_reps_file = os.path.join("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/err_heatmaps", "err_heatmap_all_reps_2d_{}_nolog.npz".format(acqFunc))

err_all_reps = np.load(err_comparison_reps_file)
print(err_all_reps.files)

gp_on_gp = err_all_reps['gp_on_gp'][:(N_ACQUIRED + N_PILOT_HF), :N_ACQUIRED, :]
gp_on_ra = err_all_reps['gp_on_ra'][:N_ACQUIRED, :N_ACQUIRED, :]
ra_on_ra = err_all_reps['ra_on_ra'][:(N_ACQUIRED + N_PILOT_HF), :N_ACQUIRED, :]
ra_on_gp = err_all_reps['ra_on_gp'][:N_ACQUIRED, :N_ACQUIRED, :]

print(gp_on_gp.shape)
print(gp_on_ra.shape)

gp_stacked = np.concatenate((gp_on_gp, gp_on_ra), axis=0)
ra_stacked = np.concatenate((ra_on_ra, ra_on_gp), axis=0)
print(gp_stacked.shape)
print(ra_stacked.shape)

step_x = []
step_y = []
for i in range(N_ACQUIRED):
    step_x.extend([i - 0.5, i + 1 - 0.5])
    step_y.extend([i + 10 - 0.5, i + 10 - 0.5])

# %%
# err_all_reps_comparison = np.load(err_comparison_reps_file)
# gp_on_gp_comparison = err_all_reps_comparison['gp_on_gp']
# gp_on_ra_comparison = err_all_reps_comparison['gp_on_ra']
# ra_on_ra_comparison = err_all_reps_comparison['ra_on_ra']
# ra_on_gp_comparison = err_all_reps_comparison['ra_on_gp']

# gp_stacked_comparison = np.concatenate((gp_on_gp_comparison, gp_on_ra_comparison), axis=0)
# ra_stacked_comparison = np.concatenate((ra_on_ra_comparison, ra_on_gp_comparison), axis=0)

# %%

reps_to_use = [0, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29]
jrepID = 15
repID = reps_to_use[jrepID]

gp_results = gp_stacked[:, :, jrepID]
ra_results = ra_stacked[:, :, jrepID]

gp_results_comparison = gp_stacked[:, :, jrepID]
ra_results_comparison = ra_stacked[:, :, jrepID]

min_cbar = min(gp_results.min(), ra_results.min(), 
               gp_results_comparison.min(), ra_results_comparison.min())
max_cbar = max(gp_results.max(), ra_results.max(), 
               gp_results_comparison.max(), ra_results_comparison.max())
logBar = True


fig, ax = plt.subplots(1, 2, figsize=(12, 8))

if logBar:
    im0 = ax[0].imshow(gp_results, 
                   cmap=ListedColormap(parula_colors),
                   norm=colors.LogNorm(vmin=min_cbar, 
                  vmax=max_cbar)
                  )
else:
    im0 = ax[0].imshow(gp_results, 
                   cmap=ListedColormap(parula_colors),
                   vmin=min_cbar,
                   vmax=max_cbar
                  )
ax[0].set_xticks(np.array([0, 4, 9, 14, 19, 24, 34, 44]))
ax[0].set_yticks(np.array([0, 4, 9, 14, 19, 24, 29, 34, 44]))
ax[0].set_xticklabels([1, 5, 10, 15, 20, 25, 35, 45], fontsize=12)
ax[0].set_yticklabels([1, 5, 10, 15, 20, 25, 30, 35, 45], fontsize=12)
ax[0].plot(step_x, step_y, 'k', linewidth=3)
ax[0].axhline(y=(N_PILOT_HF + N_ACQUIRED - 0.5), color='r', linewidth=4.5)
ax[0].set_xlabel("Acquisition Step", fontsize=16)
ax[0].set_ylabel("High-fidelity Run IDs (65 onwards Randomly Selected)", fontsize=16)
ax[0].set_title("Predictions from AL", fontsize=22)

fig.colorbar(im0, fraction=0.046, pad=0.04, ax=ax[0])


if logBar:
    im1 = ax[1].imshow(ra_results, 
                cmap=ListedColormap(parula_colors),
                norm=colors.LogNorm(vmin=min_cbar, 
                vmax=max_cbar)
                  )
    
else:
    im1 = ax[1].imshow(ra_results, 
                cmap=ListedColormap(parula_colors),
                vmin=min_cbar,
                vmax=max_cbar
                )

ax[1].set_xticks(np.array([0, 4, 9, 14, 19, 24, 34, 44]))
ax[1].set_yticks(np.array([0, 4, 9, 14, 19, 24, 29, 34, 44]))
ax[1].set_xticklabels([1, 5, 10, 15, 20, 25, 35, 45], fontsize=12)
ax[1].set_yticklabels([1, 5, 10, 15, 20, 25, 30, 35, 45], fontsize=12)
ax[1].plot(step_x, step_y, 'k', linewidth=3)

ax[1].axhline(y=(N_PILOT_HF + N_ACQUIRED - 0.5), color='r', linewidth=4.5)
ax[1].set_xlabel("Acquisition Step", fontsize=16)
ax[1].set_ylabel("High-fidelity Run IDs (65 onwards via AL)", fontsize=16)
ax[1].set_title("Predictions from RS", fontsize=22)

fig.colorbar(im1, fraction=0.046, pad=0.04, ax=ax[1])

# fig.suptitle("Replication {:02d}".format(repID + 1), fontsize=24, y=0.95)

fig.tight_layout()

if saveFig:
    plt.savefig("./2d_pde/figs_2d/heatmap_log_scale_rep{:03d}_2d_{}.jpg".format(repID + 1, acqFunc), dpi=200, bbox_inches='tight')

# %%
plt.hist(gp_on_ra[:, -1, jrepID], alpha=0.5, color="royalblue", label="Holdout from RS")
plt.hist(ra_on_gp[:, -1, jrepID], alpha=0.3, color="darkorange", label="Holdout from AL")
plt.legend(fontsize=16)
plt.xlabel("Relative Error", fontsize=16)
plt.title(r"Predictive Performance at $N_{HF}=B$", fontsize=20)
if saveFig:
    plt.savefig("./2d_pde/figs_2d/hist_rel_error_rep{:03d}_2d_{}.jpg".format(repID + 1, acqFunc), dpi=200)
# %%
