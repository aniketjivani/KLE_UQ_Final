# %%
# Plot snapshots of HF and LF data for 3 QoIs extracted along the centerline.

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from cycler import cycler
import os
import numpy as np
import sys
import h5py

saveFig = False

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/")

import register_parula as rp
parula_colors = rp._parula_data

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", 
       family="serif")
plt.rc("axes.spines", top=True, right=True)
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=16)
# plt.rc('axes', labelsize=18)
# plt.rc('legend', fontsize=12)
# plt.rc('figure', titlesize=18)
# set gridlines and grid alpha and grid linestyle
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.8)

# %%
mu_l = 1.84592e-5
rhoInf = 0.0722618

# %%
f = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/PilotBatchData.jld", "r")
# keys = ['HFLFID','_creator','xbyD','xiHF1','xiLF1','yHFUU','yHFUW','yHFV','yLFUU','yLFUW','yLFV']
xi_LF1 = f["xiLF1"][:].T
xi_HF1 = f["xiHF1"][:].T
xbyD = f["xbyD"][:].T

yLF_v_pilot = f["yLFV"][:].T
yHF_v_pilot = f["yHFV"][:].T

yLF_uu_pilot = f["yLFUU"][:].T
yHF_uu_pilot = f["yHFUU"][:].T

yLF_uw_pilot = f["yLFUW"][:].T
yHF_uw_pilot = f["yHFUW"][:].T

n_lines_lf = yLF_v_pilot.shape[1]
n_lines_hf = yHF_v_pilot.shape[1]

common_mask_lf = np.isin(xi_LF1, xi_HF1).all(axis=1)
common_lf_indices = np.where(common_mask_lf)[0]
# array([  1,  20,  21,  28,  29,  54,  69, 109, 126, 155, 160, 175, 180,
    #    188, 194])

colorblind_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
]


common_hf_lf_colors = colorblind_colors[:n_lines_hf]
lf_single_color = colorblind_colors[-1]

# %%
# use a single color (with some alpha) for all LF lines. But also highlight the common lines in the same set of colors as the HF lines.
fig, ax = plt.subplots(2, 3, figsize=(16, 9), sharex=True)

qoi_list_hf = [yHF_v_pilot, yHF_uu_pilot, yHF_uw_pilot]
qoi_list_lf = [yLF_v_pilot, yLF_uu_pilot, yLF_uw_pilot]
qoi_labels = [r"$\overline{v}$", r"$\overline{u'u'}$", r"$\overline{u'w'}$"]


for i in range(3):
    # top row is HF lines.
    ax[0, i].set_prop_cycle(cycler('color', common_hf_lf_colors))
    ax[0, i].plot(xbyD, qoi_list_hf[i], linewidth=3.5)

    if i == 0:
        ax[0, i].set_ylabel('HF Pilot\n\n' + qoi_labels[i], fontsize=20, fontweight='bold')
        ax[1, i].set_ylabel('LF Pilot\n\n' + qoi_labels[i], fontsize=20, fontweight='bold')
    else:
        ax[0, i].set_ylabel(qoi_labels[i], fontsize=20)
        ax[1, i].set_ylabel(qoi_labels[i], fontsize=20)


    # bottom row. First plot all LF lines in a single color with some alpha.
    ax[1, i].plot(xbyD, qoi_list_lf[i], color=lf_single_color, alpha=0.3, linewidth=0.8)
    ax[1, i].set_prop_cycle(cycler('color', common_hf_lf_colors))
    ax[1, i].plot(xbyD, qoi_list_lf[i][:, common_lf_indices], linewidth=3.5)

    ax[1, i].set_xlabel(r"$x/D$", fontsize=20)

    # set tick label size
    ax[0, i].tick_params(axis='both', which='major', labelsize=14)
    ax[1, i].tick_params(axis='both', which='major', labelsize=14)

for a in ax.ravel():
    a.set_xlim(xbyD.min(), xbyD.max())

fig.tight_layout()

# if saveFig:
#     plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/HF_LF_snapshots_3QoIs.jpg", dpi=200, bbox_inches='tight')

# %%

fig_corr, ax_corr = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# Calculate correlations at each x/D location
correlations = []
for i in range(3):
    qoi_hf = qoi_list_hf[i]
    qoi_lf = qoi_list_lf[i][:, common_lf_indices]
    corr_at_x = []
    for j in range(len(xbyD)):
        corr = np.corrcoef(qoi_hf[j, :], qoi_lf[j, :])[0, 1]
        corr_at_x.append(corr)
    correlations.append(corr_at_x)
    
    # Plot correlation trend
    ax_corr[i].plot(xbyD, corr_at_x, linewidth=3, color='darkblue', marker='o', markersize=4)
    ax_corr[i].set_title(f'HF-LF Correlation: {qoi_labels[i]}', fontsize=18)
    ax_corr[i].set_xlabel(r"$x/D$", fontsize=20)
    ax_corr[i].grid(True, alpha=0.3)
    ax_corr[i].tick_params(axis='both', which='major', labelsize=14)
    
    # Add horizontal line at correlation = 0.8 for reference
    ax_corr[i].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='r = 0.8')
    ax_corr[i].legend(fontsize=12)

ax_corr[0].set_ylabel('Correlation Coefficient', fontsize=20)

# Set y-axis limits to show full correlation range
for ax in ax_corr:
    ax.set_ylim(0, 1)
    ax.set_xlim(xbyD.min(), xbyD.max())

fig_corr.tight_layout()

# Print summary statistics
print("Correlation Summary:")
for i, qoi in enumerate(qoi_labels):
    corr_array = np.array(correlations[i])
    print(f"{qoi}: Mean = {np.mean(corr_array):.3f}, Min = {np.min(corr_array):.3f}, Max = {np.max(corr_array):.3f}")

if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/HF_LF_correlation_trends.jpg", dpi=200, bbox_inches='tight')

# %%