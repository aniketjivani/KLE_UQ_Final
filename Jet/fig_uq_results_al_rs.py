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

saveFig = True

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/")

import register_parula as rp
parula_colors = rp._parula_data

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", 
       family="serif")
plt.rc("axes.spines", top=True, right=True)

plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.8)
TITLE_FS = 22
LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 22
LINE_WIDTH = 3.0
FILL_ALPHA = 0.2

COLORS = {
    'BF': 'royalblue',
    'LF': 'teal',
    'HF': 'orangered'
}


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

colorblind_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
]
qoi_labels = [r"$\overline{v}$", r"$\overline{u'u'}$", r"$\overline{u'w'}$"]


common_hf_lf_colors = colorblind_colors[:n_lines_hf]
lf_single_color = colorblind_colors[-1]

# %%
test_pts_mat = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/JetTestPts.mat", "r")

test_pts = test_pts_mat['test_points'][:].T
test_pts_scaled = 0.5 * test_pts * (ub - lb) + 0.5 * (ub + lb)
with np.printoptions(suppress=True, precision=4):
    print(test_pts_scaled)

# %%

uq_al_pred = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/UQPredsAL.npz")

uq_random_pred = np.load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/UQPredsRandom.npz")

yBFPredALV, yBFPredALUU, yBFPredALUW = uq_al_pred["yBFPredALV"], uq_al_pred["yBFPredALUU"], uq_al_pred["yBFPredALUW"]

yLFPredALV, yLFPredALUU, yLFPredALUW = uq_al_pred["yLFPredALV"], uq_al_pred["yLFPredALUU"], uq_al_pred["yLFPredALUW"]

yHFPredALV, yHFPredALUU, yHFPredALUW = uq_al_pred["yHFPredALV"], uq_al_pred["yHFPredALUU"], uq_al_pred["yHFPredALUW"]

yBFPredRSV, yBFPredRSUU, yBFPredRSUW = uq_random_pred["yBFPredRSV"], uq_random_pred["yBFPredRSUU"], uq_random_pred["yBFPredRSUW"]

yLFPredRSV, yLFPredRSUU, yLFPredRSUW = uq_random_pred["yLFPredRSV"], uq_random_pred["yLFPredRSUU"], uq_random_pred["yLFPredRSUW"]

yHFPredRSV, yHFPredRSUU, yHFPredRSUW = uq_random_pred["yHFPredRSV"], uq_random_pred["yHFPredRSUU"], uq_random_pred["yHFPredRSUW"]

# %%

def get_summary(pred_tensor_bf, pred_tensor_lf, pred_tensor_hf):
    mean_bf = np.mean(pred_tensor_bf, axis=1)
    std_bf = np.std(pred_tensor_bf, axis=1)

    mean_lf = np.mean(pred_tensor_lf, axis=1)
    std_lf = np.std(pred_tensor_lf, axis=1)

    mean_hf = np.mean(pred_tensor_hf, axis=1)
    std_hf = np.std(pred_tensor_hf, axis=1)

    return (mean_bf, std_bf), (mean_lf, std_lf), (mean_hf, std_hf)

stats_V_AL = get_summary(yBFPredALV, yLFPredALV, yHFPredALV)
stats_UU_AL = get_summary(yBFPredALUU, yLFPredALUU, yHFPredALUU)
stats_UW_AL = get_summary(yBFPredALUW, yLFPredALUW, yHFPredALUW)

stats_V_RS = get_summary(yBFPredRSV, yLFPredRSV, yHFPredRSV)
stats_UU_RS = get_summary(yBFPredRSUU, yLFPredRSUU, yHFPredRSUU)
stats_UW_RS = get_summary(yBFPredRSUW, yLFPredRSUW, yHFPredRSUW)


# %% Plot mean +/- 1 \sigma, with separate lines for each quantity (no fill between). Separate linestyles, line thickness etc. for each surrogate type.



# %%
def plot_gp_confidence_interval(ax, x, mean, std, color, label, alpha=None, 
                                hatching=None,
                                ls=None,
                                lw=None):
    """
    Plots a line with its corresponding shaded confidence interval on a given axis.
    """

    if lw is None:
        lw = LINE_WIDTH
    
    ax.plot(x, mean, color=color, linewidth=lw, 
            linestyle=ls,
            label=label)

    if alpha is None:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=FILL_ALPHA,
            label="_nolegend_",
            hatch=hatching,
        )
    else:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha,
            label="_nolegend_",
            hatch=hatching
    )
        
    ax.set_xlim([x[0], x[-1]])


# %%
fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

# plot_gp_confidence_interval(ax[0, 0], xbyD, stats_V_AL[0][0], stats_V_AL[0][1], COLORS['BF'], 'BF Surrogate')

# plot line with error bars for BF surrogate
ax[0, 0].errorbar(xbyD[::10], stats_V_AL[0][0][::10], yerr=stats_V_AL[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)


plot_gp_confidence_interval(ax[0, 0], xbyD, stats_V_AL[1][0], stats_V_AL[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[0, 0], xbyD, stats_V_AL[2][0], stats_V_AL[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

# plot_gp_confidence_interval(ax[0, 1], xbyD, stats_UU_AL[0][0], stats_UU_AL[0][1], COLORS['BF'], 'BF Surrogate')
ax[0, 1].errorbar(xbyD[::10], stats_UU_AL[0][0][::10], yerr=stats_UU_AL[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)
plot_gp_confidence_interval(ax[0, 1], xbyD, stats_UU_AL[1][0], stats_UU_AL[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[0, 1], xbyD, stats_UU_AL[2][0], stats_UU_AL[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

# plot_gp_confidence_interval(ax[0, 2], xbyD, stats_UW_AL[0][0], stats_UW_AL[0][1], COLORS['BF'], 'BF Surrogate')
ax[0, 2].errorbar(xbyD[::10], stats_UW_AL[0][0][::10], yerr=stats_UW_AL[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)
plot_gp_confidence_interval(ax[0, 2], xbyD, stats_UW_AL[1][0], stats_UW_AL[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[0, 2], xbyD, stats_UW_AL[2][0], stats_UW_AL[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

for i, a in enumerate(ax[0, :]):
    a.set_title(qoi_labels[i], fontsize=24)
    a.tick_params(axis='both', which='major', labelsize=TICK_FS)
    a.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.6)


# plot_gp_confidence_interval(ax[1, 0], xbyD, stats_V_RS[0][0], stats_V_RS[0][1], COLORS['BF'], 'BF Surrogate')
ax[1, 0].errorbar(xbyD[::10], stats_V_RS[0][0][::10], yerr=stats_V_RS[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)
plot_gp_confidence_interval(ax[1, 0], xbyD, stats_V_RS[1][0], stats_V_RS[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[1, 0], xbyD, stats_V_RS[2][0], stats_V_RS[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

ax[1, 1].errorbar(xbyD[::10], stats_UU_RS[0][0][::10], yerr=stats_UU_RS[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)
# plot_gp_confidence_interval(ax[1, 1], xbyD, stats_UU_RS[0][0], stats_UU_RS[0][1], COLORS['BF'], 'BF Surrogate')
plot_gp_confidence_interval(ax[1, 1], xbyD, stats_UU_RS[1][0], stats_UU_RS[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[1, 1], xbyD, stats_UU_RS[2][0], stats_UU_RS[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

ax[1, 2].errorbar(xbyD[::10], stats_UW_RS[0][0][::10], yerr=stats_UW_RS[0][1][::10], color=COLORS['BF'], 
                  linewidth=LINE_WIDTH, linestyle=':', label='BF Surrogate', capsize=5)
# plot_gp_confidence_interval(ax[1, 2], xbyD, stats_UW_RS[0][0], stats_UW_RS[0][1], COLORS['BF'], 'BF Surrogate')
plot_gp_confidence_interval(ax[1, 2], xbyD, stats_UW_RS[1][0], stats_UW_RS[1][1], COLORS['LF'], 'LF-KLE', ls='--')
plot_gp_confidence_interval(ax[1, 2], xbyD, stats_UW_RS[2][0], stats_UW_RS[2][1], COLORS['HF'], 'HF-KLE', hatching='//', ls='-.')

# fig.suptitle("Forward UQ Results", fontsize=TITLE_FS, y=1.05)
for i, a in enumerate(ax[1, :]):
    a.set_xlabel(r"$x/D$", fontsize=LABEL_FS)
    a.tick_params(axis='both', which='major', labelsize=TICK_FS)
    a.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.6)


for a in ax.ravel():
    for spine in a.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor('black')


for a in ax[:, 0]:
    a.set_ylim(-0.5, 140)
for a in ax[:, 1]:
    a.set_ylim(-40, 2200)
for a in ax[:, 2]:
    a.set_ylim(-5, 800)


handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=LEGEND_FS, ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.tight_layout()
# if saveFig:
#     plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/fig_uq_results_al_jet.png", bbox_inches='tight', dpi=200)
if saveFig:
    plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/fig_uq_results_al_rs_jet_qois.png", bbox_inches='tight', dpi=200)

# %%



# %%