# %%
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np
import torch
import pickle
import os
import sys
from rich.progress import track

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", 
       family="serif")
plt.rc("axes.spines", top=True, right=True)
# set explicit fontsizes for ticks, lables, legend and title
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)
# set gridlines and grid alpha and grid linestyle
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.8)

# case objects batch ...jld

# %%
def summOracleObj(data_dir,
                  batchID, 
                  repFormat="rep_{:03d}_logEI_NEW2", 
                  repID = 1, 
                  ):
    data_dict = np.load(os.path.join(data_dir, 
              #       "rep_{:03d}".format(repID), 
                     repFormat.format(repID),
                    "case_objects_batch_{:03d}.npz".format(batchID)))
    oracle_gp = data_dict["oracle_gp"]
    oracle_ra = data_dict["oracle_ra"]
    
    mean_gp = oracle_gp.mean()
    mean_ra = oracle_ra.mean()
    
    return oracle_gp, oracle_ra, mean_gp, mean_ra


# %%
# N_ACQUIRED = 100
# N_ACQUIRED = 46
# N_REPS = 5

N_ACQUIRED = 60
N_REPS = 5

mean_gp_all_reps = np.zeros((N_ACQUIRED, N_REPS))
# mean_gp_all_reps_min = np.zeros((N_ACQUIRED, N_REPS))
mean_ra_all_reps = np.zeros((N_ACQUIRED, N_REPS))

chosen_case = 2
# chosen_case = 2
allModes = 0
chosen_acq = "EI"
saveFig = False
loo = True

if allModes:
    if chosen_case == 1:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_all_modes".format(chosen_acq)
    elif chosen_case == 2:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_all_modes".format(chosen_acq)
else:
    # if chosen_case == 1:
    #     data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}".format(chosen_acq)
    # elif chosen_case == 2:
    #     data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}".format(chosen_acq)
    if chosen_case == 1:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_nolog".format(chosen_acq)
        data_dir_min = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_nolog_min".format(chosen_acq)
    elif chosen_case == 2:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_nolog".format(chosen_acq)
        data_dir_min = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_nolog_min".format(chosen_acq)

if loo:
    if chosen_case == 1:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_nolog_loo".format(chosen_acq)
        data_dir_min = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_nolog_min_loo".format(chosen_acq)
    elif chosen_case == 2:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_nolog_loo".format(chosen_acq)
        data_dir_min = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_nolog_min_loo".format(chosen_acq)



for rID in range(N_REPS):
    for bID in range(N_ACQUIRED):
       _, _, mean_gp, mean_ra = summOracleObj(data_dir_all,
                  bID, 
                  repFormat="rep_{:03d}",
                  repID=rID + 1)
    #    _, _, mean_gp_min, _ = summOracleObj(data_dir_min,
    #                 bID, 
    #                 repFormat="rep_{:03d}",
    #                 repID=rID + 1)
    
       mean_gp_all_reps[bID, rID] = mean_gp
    #    mean_gp_all_reps_min[bID, rID] = mean_gp_min
       mean_ra_all_reps[bID, rID] = mean_ra


# rID = 1
# for bID in range(N_ACQUIRED):
#     _, _, mean_gp, mean_ra = summOracleObj(data_dir_all,
#                 bID, 
#                 repFormat="rep_{:03d}",
#                 repID=rID + 1)

#     mean_gp_all_reps[bID, 0] = mean_gp
#     mean_ra_all_reps[bID, 0] = mean_ra


mean_gp_mean = mean_gp_all_reps.mean(axis=1)
mean_gp_std = mean_gp_all_reps.std(axis=1)
# mean_gp_mean_min = mean_gp_all_reps_min.mean(axis=1)
# mean_gp_std_min = mean_gp_all_reps_min.std(axis=1)
mean_ra_mean = mean_ra_all_reps.mean(axis=1)
mean_ra_std = mean_ra_all_reps.std(axis=1)

# use different colors other than blue or orange. 
line_colors = ['#9467bd', 
               '#d46197', 
               "#c36027"]
fig, ax = plt.subplots()

# make boxplots of the oracle errors at every 10 acquisitions?!
set_log = False
if set_log:
    ax.set_yscale('log')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_gp_mean, 
            '-o',
            color=line_colors[0],
            markersize=8,
            linewidth=2.5,
            label="GP ({})".format(chosen_acq))

ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
                mean_gp_mean +  mean_gp_std,
                mean_gp_mean -  mean_gp_std,
                color=line_colors[0],
                alpha=0.2,
                label="")

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
#             mean_gp_mean_min, 
#             '-s',
#             color=line_colors[1],
#             markersize=8,
#             linewidth=2.5,
#             label="Least Error ({})".format(chosen_acq))

# ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
#                 mean_gp_mean_min +  mean_gp_std_min,
#                 mean_gp_mean_min -  mean_gp_std_min,
#                 color=line_colors[1],
#                 alpha=0.1,
#                 label="")

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_ra_mean, 
            '-*',
            color=line_colors[2],
            linewidth=2.5,
            markersize=8,
            label='Random')


ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
                mean_ra_mean + mean_ra_std,
                mean_ra_mean - mean_ra_std,
                color=line_colors[2],
                alpha=0.2,
                label="")


# ax.set_title('Active Learning Performance', fontsize=20)
ax.set_xlabel(r'Design stage $\ell$', fontsize=16)
ax.set_ylabel(r'$(\mu_{\varepsilon}^{(\ell)})$', fontsize=16)
ax.set_xlim([1, N_ACQUIRED])


# set integer ticklabels
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
if chosen_case == 1:
    ax.legend(loc='lower left')
elif chosen_case == 2:
    ax.legend(loc='upper right')
plt.tight_layout()

if saveFig:
    if allModes:
        if chosen_case == 1:
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/exp_oracle_err_bSine_all_modes.png")
            plt.close()
        elif chosen_case == 2:
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/exp_oracle_err_taylor_all_modes.png")
            plt.close()
    else:
        if chosen_case == 1:
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d//exp_oracle_err_bSine_{}_all_cases.png".format(chosen_acq))
            plt.close()
        elif chosen_case == 2:
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/exp_oracle_err_taylor_{}_all_cases.png".format(chosen_acq))
            plt.close()
# %%

# %%