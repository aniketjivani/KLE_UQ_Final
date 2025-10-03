# make scatterplot of points colored by stage of acquisition.

# %%

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

# LOAD VERY LAST INPUT FILE FROM EACH CASE. SCATTER BASED ON THIS. USE DIFFERENT MARKER FOR PILOT POINTS. 1x3 SUBPLOTS. Random points with very small markersize, light color.

# chosen_case = 1

reps_to_use = np.array([0, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29]) + 1
jrepID = 15

lb = [0.01, 0.05, 0.3, 0.55]
ub = [0.05, 0.08, 0.7, 0.85]

chosen_acq = "EI"
BUDGET_HF = 40
N_REPS = 20
N_PILOT_HF = 10
N_PILOT_LF = 300

inputs_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_inputs_trunc_{}".format(chosen_acq)

input_scaled_pilot = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_HF_Pilot_scaled_trunc.txt") # -1 to 1
inputs_original_pilot = 0.5 * (input_scaled_pilot + 1) * (np.array(ub) - np.array(lb)) + np.array(lb)

all_inputs_gp = np.zeros((BUDGET_HF, N_REPS * 4))
all_inputs_ra = np.zeros((BUDGET_HF, N_REPS * 4))

for i, repID in enumerate(reps_to_use):
    inputs_path = os.path.join(inputs_dir, "rep_{:03d}".format(repID), "HF_Batch_{:03d}_Final.txt".format(BUDGET_HF))

    inputs = np.loadtxt(inputs_path)

    all_inputs_gp[:, (i*4):((i*4) + 4)] = inputs[(N_PILOT_HF):(N_PILOT_HF + BUDGET_HF), :]

    all_inputs_ra[:, (i*4):((i*4) + 4)] = inputs[(2 * N_PILOT_HF + BUDGET_HF):, :]

# %%
# set colormap to plot points based on acquisition order
# colormap_acq = plt.cm.viridis
colormap_acq = plt.cm.viridis
fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True, sharex=True)

# scatter all points in a given column. If shape(all_inputs_min_gp) = (40, 80), then 20 scatter calls, but with the same color based on i value.

# tsth = False # plot theta_s and theta_h
tsth = True
# for i in range(BUDGET_HF):

chosen_rep_ID = 15
chosen_rep = reps_to_use[chosen_rep_ID]
for i in range(BUDGET_HF):
    if tsth:
        xidx = 0
        yidx = 1
    else:
        xidx = 2
        yidx = 3

    a0 = axs[0].scatter(all_inputs_gp[i, 4 * chosen_rep_ID + xidx],
                    all_inputs_gp[i, 4 * chosen_rep_ID + yidx],
                    s=55,
                    alpha=0.7,
                    edgecolor='black',
                    color=colormap_acq(i / BUDGET_HF)
    )
    a1 = axs[1].scatter(all_inputs_ra[i, 4 * chosen_rep_ID + xidx],
                        all_inputs_ra[i, 4 * chosen_rep_ID + yidx],
                        s=55,
                        alpha=0.7,
                        edgecolor='black',
                        color='salmon'
        )
        
for a in axs:
    if tsth:
        xidx = 0
        yidx = 1
    else:
        xidx = 2
        yidx = 3

    a.scatter(inputs_original_pilot[:, xidx],
               inputs_original_pilot[:, yidx],
               s=250,
               marker='*',
               color='goldenrod',
               edgecolor='black',
               linewidth=1.5,
               )
    if tsth:
        a.set_xlabel(r"$\xi_s$", fontsize=LABEL_FS)
        a.set_ylabel(r"$\xi_h$", fontsize=LABEL_FS)
    else:
        a.set_xlabel(r"$\xi_x$", fontsize=LABEL_FS)
        a.set_ylabel(r"$\xi_y$", fontsize=LABEL_FS)
    a.grid(False)

if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/figs_2d/scatter_2d_acq_{}.jpg".format(chosen_acq), bbox_inches='tight', dpi=200)

# %%
# plot a horizontal colorbar in its own plot.
fig2, ax2 = plt.subplots(figsize=(6, 1))
norm = plt.Normalize(1, BUDGET_HF)
cb1 = cm.ScalarMappable(norm=norm, cmap=colormap_acq)
cb1.set_array([])
cbar = fig2.colorbar(cb1, cax=ax2, orientation='horizontal')
cbar.set_label('Acquisition Order', fontsize=LABEL_FS)
cbar.ax.tick_params(labelsize=TICK_FS)

# %% additional viz:



