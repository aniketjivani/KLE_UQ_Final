# %%
# Plot progression of LOO error for new HF data points.

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import os
import numpy as np
import sys
import h5py

saveFig = True

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/")

import register_parula as rp
parula_colors = rp._parula_data

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", 
       family="serif")
plt.rc("axes.spines", top=True, right=True)
# set gridlines and grid alpha and grid linestyle
plt.rc('axes', grid=False)
# plt.rc('grid', linestyle='--')
# plt.rc('grid', alpha=0.8)

# %%
mu_l = 1.84592e-5
rhoInf = 0.0722618
lb = np.array([293.24, 0.1, 1.531])
ub = np.array([312.94, 0.3, 4.6055])

ip_param_names = [r"$U_c$", r"$\kappa$", r"$\log(\nu/\tilde{\nu})$"]

nPilotLF = 200
nPilotHF = 15

# %%
f = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/PilotBatchData.jld", "r")
# keys = ['HFLFID','_creator','xbyD','xiHF1','xiLF1','yHFUU','yHFUW','yHFV','yLFUU','yLFUW','yLFV']
xi_LF_Pilot = f["xiLF1"][:].T
xi_HF_Pilot = f["xiHF1"][:].T
xbyD = f["xbyD"][:].T

yLF_v_pilot = f["yLFV"][:].T
yHF_v_pilot = f["yHFV"][:].T

yLF_uu_pilot = f["yLFUU"][:].T
yHF_uu_pilot = f["yHFUU"][:].T

yLF_uw_pilot = f["yLFUW"][:].T
yHF_uw_pilot = f["yHFUW"][:].T

n_lines_lf = yLF_v_pilot.shape[1]
n_lines_hf = yHF_v_pilot.shape[1]

common_mask_lf = np.isin(xi_LF_Pilot, xi_HF_Pilot).all(axis=1)
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
test_pts_mat = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/JetTestPts.mat", "r")

test_pts = test_pts_mat['test_points'][:].T
test_pts_scaled = 0.5 * test_pts * (ub - lb) + 0.5 * (ub + lb)
with np.printoptions(suppress=True, precision=4):
    print(test_pts_scaled)


# %% Pilot + 35 points (7 batches of 5 each)

hf_all = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HF_AllPoints_Scaled.txt")
lf_all = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/LF_AllPoints_Scaled.txt")

hf_all_scaled = 0.5 * hf_all * (ub - lb) + 0.5 * (ub + lb)
lf_all_scaled = 0.5 * lf_all * (ub - lb) + 0.5 * (ub + lb)

# %% err across batches.
pilot_batch_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch01.txt")
batch_01_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch02.txt")
batch_02_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch03.txt")
batch_03_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch04.txt")
batch_04_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch05.txt")
batch_05_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch06.txt")
batch_06_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch07.txt")
batch_07_mean_err = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorFinal.txt")

all_batch_errs = [batch_01_mean_err, batch_02_mean_err, batch_03_mean_err,
                  batch_04_mean_err, batch_05_mean_err, batch_06_mean_err,
                  batch_07_mean_err]

# the line plot of err progression will read something like plt.plot(1:15, pilot_batch_mean_err), plt.plot(16:20, batch_01_mean_err), etc.)

err_line_colors = colorblind_colors[5:13]
err_line_markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

# %% 3 subplots: two for scatter plot of HF points in acquisition order, one for err progression.

n_batches_acq = 7
n_batch_total = n_batches_acq + 1
batch_colors = plt.cm.cool(np.linspace(0, 1, n_batches_acq + 1))

hf_pilot = hf_all_scaled[:nPilotHF, :]
hf_batches_acq = [hf_all_scaled[(nPilotHF + i*5):(nPilotHF + (i+1)*5), :] for i in range(n_batches_acq)]

hf_all_batches = [hf_pilot] + hf_batches_acq

# %%
# plt.figure()

# for i in range(1, n_batch_total):
#     plt.scatter(hf_all_batches[i][:, 0],
#                 hf_all_batches[i][:, 1],
#                 s=80,
#                 marker='o',
#                 alpha=0.7,
#                 color=batch_colors[i-1],
#                 label=""
#     )

# plt.scatter(hf_all_batches[0][:, 0],
#             hf_all_batches[0][:, 1],
#             s=80,
#             marker='*',
#             color='goldenrod',
#             edgecolor='black',
#             linewidth=1.5,
#             )

# plt.figure()
# plt.scatter(hf_all_batches[0][:, 1],
#             hf_all_batches[0][:, 2],
#             s=80,
#             marker='*',
#             color='goldenrod',
#             edgecolor='black',
#             linewidth=1.5,
#             )
# for i in range(1, n_batch_total):
#     plt.scatter(hf_all_batches[i][:, 1],
#                 hf_all_batches[i][:, 2],
#                 s=80,
#                 marker='o',
#                 alpha=0.7,
#                 color=batch_colors[i-1],
#                 label=""
#     )

# plt.scatter(hf_all_batches[0][:, 0],
#             hf_all_batches[0][:, 1],
#             s=80,
#             marker='*',
#             color='goldenrod',
#             edgecolor='black',
#             linewidth=1.5,
#             )

# %% 3D scatter plot

fig = plt.figure(figsize=(14, 12)) # Increased height slightly for better spacing
ax = fig.add_subplot(111, projection='3d')

# Plot pilot batch first (as stars)
ax.scatter(hf_all_batches[0][:, 0],
           hf_all_batches[0][:, 1], 
           hf_all_batches[0][:, 2],
           s=200, # Increased size for visibility
           marker='*',
           facecolor=batch_colors[0],
           edgecolor='black',
           linewidth=1.5,
           label="Pilot HF Simulations")

for x, y, z in zip(hf_all_batches[0][:, 0], hf_all_batches[0][:, 1], hf_all_batches[0][:, 2]):
        ax.plot([x, x], [y, y], [0, z], color='black', alpha=0.9, linewidth=1.2)

# Plot acquisition batches
for i in range(1, n_batch_total):
    ax.scatter(hf_all_batches[i][:, 0],
               hf_all_batches[i][:, 1],
               hf_all_batches[i][:, 2],
               s=95, # Increased size
               marker='o',
               alpha=0.8,
               color=batch_colors[i],
               linewidth=0.5,
               edgecolor='black',
               label="")

    # Projection lines for less ambiguity
    for x, y, z in zip(hf_all_batches[i][:, 0], hf_all_batches[i][:, 1], hf_all_batches[i][:, 2]):
        ax.plot([x, x], [y, y], [0, z], color='black', alpha=0.9, linewidth=1.2)

ax.set_title("Selected Points (Batch AL)", fontsize=28, pad=20)
ax.legend(bbox_to_anchor=(0.25, 0.96), loc='upper left', fontsize=16)
# ax.view_init(elev=20, azim=45)
ax.grid(True, alpha=0.1)
# ax.grid(False)
# Increase tick label size and padding
ax.tick_params(axis='x', which='major', pad=10, labelsize=14)
ax.tick_params(axis='y', which='major', pad=10, labelsize=14)
ax.tick_params(axis='z', which='major', pad=10, labelsize=14)

if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/3D_scatter_HF_acq_order.png", bbox_inches='tight', dpi=200)

# %% line plots of err progression

line_cols = plt.cm.cool(np.linspace(0, 1, n_batch_total))

plt.figure(figsize=(10, 6))
plt.plot(range(1, nPilotHF + 1), pilot_batch_mean_err,
         marker=err_line_markers[0],
        #  color=err_line_colors[0],
            color=line_cols[0],
         linewidth=3.5,
         markeredgecolor='black',
        markeredgewidth=1.0,
         markersize=8,
        label="")
for i in range(n_batches_acq):
    plt.plot(range(nPilotHF + i*5 + 1, nPilotHF + (i+1)*5 + 1),
             all_batch_errs[i][(nPilotHF + i*5):(nPilotHF + (i+1)*5)],
             marker=err_line_markers[i + 1],
             color=line_cols[i + 1],
             markeredgecolor='black',
             markeredgewidth=1.0,
             linewidth=3.5,
             markersize=8,
             label="Batch {}".format(i + 1))

plt.xlim(1, nPilotHF + n_batches_acq * 5)
plt.xlabel(r"$N_{HF}$", fontsize=22)
plt.ylabel("Average LOO Error", fontsize=22)
# set tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/err_progression_HF_AL.jpg", bbox_inches='tight', dpi=200)

# %%
fig2, ax2 = plt.subplots(figsize=(6, 1))
norm = plt.Normalize(1, 8)
cb1 = cm.ScalarMappable(norm=norm, cmap=plt.cm.cool)
cb1.set_array([])
cbar = fig2.colorbar(cb1, cax=ax2, orientation='horizontal')
cbar.set_label('Acquisition Order', fontsize=22)
cbar.ax.tick_params(labelsize=14)

if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/3D_scatter_HF_acq_order.jpg", bbox_inches='tight', dpi=200)

# %%
