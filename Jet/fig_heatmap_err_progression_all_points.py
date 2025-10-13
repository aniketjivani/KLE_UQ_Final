# %%
# Plot progression of LOO error for all points (past and future) as a heatmap similar to the 1D case and 2D case. Skip a separate heatmap for random as training set errors for random do not necessarily show any ordering, instead show a single heatmap followed by histograms of error distribution for fixed 500 pt test set common to each surrogate.(hfpred mode, comparisons vs HF-KLE)

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

saveFig = False

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

# %%
train_budget_errs = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataTrainBudget_AL.npy")

rem_budget_errs = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataRemBudget_AL.npy")

holdout_rs_errs = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_RS_SurrAL.npy")

holdout_rs_bf_preds = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/BFPredictionsHoldout_RS_SurrAL.npy")

holdout_rs_hf_true = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HFDataHoldout_RS.npy")

holdout_al_errs = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_AL_SurrRS.npy")

holdout_test_errs = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrAL.npy")

holdout_test_errs_random = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrRandom.npy")

# %% 
rem_budget_errs2 = np.zeros_like(rem_budget_errs)

n_batches = rem_budget_errs.shape[1]
N_TOTAL_HF = 50
batch_size = 5

for bID in range(n_batches - 1):
    print((nPilotHF + bID * batch_size), (N_TOTAL_HF - nPilotHF - bID * batch_size), bID)
    rem_budget_errs2[(nPilotHF + bID * batch_size):, bID] = rem_budget_errs[:(N_TOTAL_HF - nPilotHF - bID * batch_size), bID]

train_budget_errs_final = train_budget_errs + rem_budget_errs2

# %% hists - distribution of errors (all QoIs averaged)

plt.hist(holdout_test_errs[:, 5], alpha=0.5, color="royalblue", label="AL Prediction Error")
plt.hist(holdout_test_errs_random[:, 5], alpha=0.3, color="darkorange", label="RS Prediction Error")
plt.legend(fontsize=16)
plt.xlabel("Relative Error (all QoIs)", fontsize=16)
# if saveFig:
#     plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/hist_avg_rel_err_all_qois.jpg", bbox_inches='tight', dpi=200)



# %%

cmap_lines = plt.cm.cool(np.linspace(0, 1, 8))
fig = plt.figure(figsize=(13, 5))
qoi_labels = [r"$\overline{v}$", r"$\overline{u'u'}$", r"$\overline{u'w'}$"]
fig.add_subplot(131)
for i in range(8):
    plt.plot(xbyD, holdout_rs_bf_preds[:, 1, i, 0], color=cmap_lines[i])
plt.plot(xbyD, holdout_rs_hf_true[:, 1, 0], color="black", linewidth=2.5)
plt.xlabel(r"$x/D$", fontsize=16)
plt.ylabel(qoi_labels[0], fontsize=16)
plt.xlim(xbyD.min(), xbyD.max())

fig.add_subplot(132)
for i in range(8):
    plt.plot(xbyD, holdout_rs_bf_preds[:, 1, i, 1], color=cmap_lines[i])
plt.plot(xbyD, holdout_rs_hf_true[:, 1, 1], color="black", linewidth=2.5)
plt.xlabel(r"$x/D$", fontsize=16)
plt.ylabel(qoi_labels[1], fontsize=16)
plt.xlim(xbyD.min(), xbyD.max())

fig.add_subplot(133)
for i in range(8):
    plt.plot(xbyD, holdout_rs_bf_preds[:, 1, i, 2], color=cmap_lines[i])
plt.plot(xbyD, holdout_rs_hf_true[:, 1, 2], color="black", linewidth=2.5)
plt.xlabel(r"$x/D$", fontsize=16)
plt.ylabel(qoi_labels[2], fontsize=16)
plt.xlim(xbyD.min(), xbyD.max())
plt.tight_layout()

plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/BFPredictionsHoldout_RS_Run2_SurrAL.jpg", bbox_inches='tight', dpi=200)

# %%
