# %%
# use data generated from `fig_bf_surrogate_uncertainty.jl` to visualize forward UQ based on different surrogate predictions.

# to viz mean predictions errors, use data generated from `fig_bf_surrogate_uncertainty_stagewise.jl` (not used for paper)

# for HF KLE use KLE built with both sets of points (generated via `fig_bf_hf_kle_oracle_performance.jl`)

import os
import numpy as np
import math
import sys
import copy
import argparse
import pickle
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/")
import register_parula as rp
parula_colors = rp._parula_data
saveFig = False
use_cumulative_HF = True

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc("font", family="serif")
plt.rc("axes.spines", top=True, right=True) # Remove top and right spines for a cleaner look

# Define consistent font sizes
TITLE_FS = 22
LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 16

# Define consistent plotting parameters
LINE_WIDTH = 3.0
FILL_ALPHA = 0.25

# Define a professional, high-contrast color palette
COLORS = {
    'BF': 'crimson',      # A strong blue
    'LF': 'teal',      # A distinct orange
    'HF': 'goldenrod'       # A vibrant green
}


# %%

chosen_case = 2
# chosen_case = 1
chosen_rep = 1
chosen_acq = "EI"

data_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c{:d}_{}".format(chosen_case, chosen_acq)

surr_pred_filepath = os.path.join(data_dir, "rep_{:03d}".format(chosen_rep), "all_surr_oracle_predictions_final.npz")

surr_preds = np.load(surr_pred_filepath, allow_pickle=True)

hf_cumulative_filepath = os.path.join(data_dir, "rep_{:03d}".format(chosen_rep), "hf_surr_oracle_predictions_cumulative.npz")

hf_cumulative_preds = np.load(hf_cumulative_filepath, allow_pickle=True)['HF_cumulative_oracle']

BF_oracle_gp = surr_preds["BF_oracle_gp"]
LF_oracle_gp = surr_preds["LF_oracle_gp"]
# HF_oracle_gp = surr_preds["HF_oracle_gp"]
BF_oracle_ra = surr_preds["BF_oracle_ra"]
LF_oracle_ra = surr_preds["LF_oracle_ra"]
# HF_oracle_ra = surr_preds["HF_oracle_ra"]

if use_cumulative_HF:
    HF_oracle_gp = np.copy(hf_cumulative_preds)
    HF_oracle_ra = np.copy(hf_cumulative_preds)
else:
    HF_oracle_gp = surr_preds["HF_oracle_gp"]
    HF_oracle_ra = surr_preds["HF_oracle_ra"]


# %%

mean_BF_gp = np.mean(BF_oracle_gp, axis=(0,1))
std_BF_gp = np.std(BF_oracle_gp, axis=(0,1))
mean_LF_gp = np.mean(LF_oracle_gp, axis=(0,1))
std_LF_gp = np.std(LF_oracle_gp, axis=(0,1))
mean_HF_gp = np.mean(HF_oracle_gp, axis=(0,1))
std_HF_gp = np.std(HF_oracle_gp, axis=(0,1))
mean_BF_ra = np.mean(BF_oracle_ra, axis=(0,1))
std_BF_ra = np.std(BF_oracle_ra, axis=(0,1))
mean_LF_ra = np.mean(LF_oracle_ra, axis=(0,1))
std_LF_ra = np.std(LF_oracle_ra, axis=(0,1))
mean_HF_ra = np.mean(HF_oracle_ra, axis=(0,1))
std_HF_ra = np.std(HF_oracle_ra, axis=(0,1))

# %%

# load oracle mean.

hf_oracle_file = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/HF_Oracle_case_{:02d}_mean_std.npz".format(chosen_case))
HF_oracle_mean = hf_oracle_file["mean"]

# %%
xgrid = np.linspace(0, 0.1, 250)

def plot_gp_confidence_interval(ax, x, mean, std, color, label, alpha=None):
    """
    Plots a line with its corresponding shaded confidence interval on a given axis.
    """
    # Plot the mean line
    ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, label=label)
    
    # Plot the confidence interval
    if alpha is None:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=FILL_ALPHA,
            label="_nolegend_" # Hide fill from legend
        )
    else:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha,
            label="_nolegend_" # Hide fill from legend
    )
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)



plot_gp_confidence_interval(ax[0], xgrid, mean_BF_gp, std_BF_gp, COLORS['BF'], 'BF Surrogate')
plot_gp_confidence_interval(ax[0], xgrid, mean_LF_gp, std_LF_gp, COLORS['LF'], 'LF-KLE')
plot_gp_confidence_interval(ax[0], xgrid, mean_HF_gp, std_HF_gp, COLORS['HF'], 'HF-KLE', alpha=0.25)
ax[0].plot(xgrid, HF_oracle_mean, color='black', linestyle='--', linewidth=3.5, label='HF Mean')

# plot oracle mean

if chosen_case == 2:
    ax[0].set_title("Active Learning (EI)", fontsize=TITLE_FS)
    ax[1].set_title("Random Sampling", fontsize=TITLE_FS)

if chosen_case == 1:
    ax[0].set_xlabel(r"$x$", fontsize=LABEL_FS)
    ax[1].set_xlabel(r"$x$", fontsize=LABEL_FS)

ax[0].set_ylabel(r"$\widetilde{y}$", fontsize=LABEL_FS)


plot_gp_confidence_interval(ax[1], xgrid, mean_BF_ra, std_BF_ra, COLORS['BF'], 'BF Surrogate')
plot_gp_confidence_interval(ax[1], xgrid, mean_LF_ra, std_LF_ra, COLORS['LF'], 'LF-KLE')
plot_gp_confidence_interval(ax[1], xgrid, mean_HF_ra, std_HF_ra, COLORS['HF'], 'HF-KLE', alpha=0.25)
ax[1].plot(xgrid, HF_oracle_mean, color='black', linestyle='--', linewidth=3.5, label='HF Mean')

for a in ax:
    if chosen_case == 1:
        a.legend(fontsize=LEGEND_FS, loc='upper right')
    elif chosen_case == 2:
        a.legend(fontsize=LEGEND_FS, loc='upper left')
    a.tick_params(axis='both', which='major', labelsize=TICK_FS)
    a.set_xlim(xgrid.min(), xgrid.max())
    # Override default grid to make it slightly more visible if needed
    a.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.6)

# if chosen_case == 2:
#     # turn off x ticks on each plot
#     ax[0].set_xticks([])
#     ax[1].set_xticks([])

#     # still keep vertical grid lines
#     ax[0].grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.6)
#     ax[1].grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.6)



# fetch ylims of plot
ylims_surr = ax[0].get_ylim()
print("Y-limits of the plots: ", ylims_surr)

# Use tight_layout to ensure labels, titles, etc., fit nicely.
# fig.tight_layout()
# plt.show()
if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/surrogate_uncertainty_case{:d}_{}.png".format(chosen_case, chosen_acq), bbox_inches='tight')

# %%

# err_data = np.load(os.path.join(data_dir, "rep_{:03d}".format(chosen_rep), "all_surr_predictions_err.npz"), allow_pickle=True)

# mean_HF_err_gp = err_data["mean_HF_err_gp"]
# mean_HF_err_ra = err_data["mean_HF_err_ra"]
# mean_BF_err_gp = err_data["mean_BF_err_gp"]
# mean_BF_err_ra = err_data["mean_BF_err_ra"]

# %%
# plot imshow for prediction errors on grid.



