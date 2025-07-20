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
N_ACQUIRED = 50
N_REPS = 5
mean_gp_all_reps = np.zeros((N_ACQUIRED, N_REPS))
mean_ra_all_reps = np.zeros((N_ACQUIRED, N_REPS))

chosen_case = 1
# chosen_case = 2
allModes = 1
chosen_acq = "EI"
# chosen_acq = "logEI"
saveFig = False


if allModes:
    if chosen_case == 1:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}_all_modes".format(chosen_acq)
    elif chosen_case == 2:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}_all_modes".format(chosen_acq)
else:
    if chosen_case == 1:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c1_{}".format(chosen_acq)
    elif chosen_case == 2:
        data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_{}".format(chosen_acq)



for rID in range(5):
    for bID in range(50):
       _, _, mean_gp, mean_ra = summOracleObj(data_dir_all,
                  bID, 
                  repFormat="rep_{:03d}",
                  repID=rID + 1)
    
       mean_gp_all_reps[bID, rID] = mean_gp
       mean_ra_all_reps[bID, rID] = mean_ra


mean_gp_mean = mean_gp_all_reps.mean(axis=1)
mean_gp_std = mean_gp_all_reps.std(axis=1)

mean_ra_mean = mean_ra_all_reps.mean(axis=1)
mean_ra_std = mean_ra_all_reps.std(axis=1)

fig, ax = plt.subplots()
ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_gp_mean, 
            '-o',
            markersize=15,
            linewidth=2,
            label='GP (EI)')

ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
                mean_gp_mean + 2 * mean_gp_std,
                mean_gp_mean - 2 * mean_gp_std,
                alpha=0.2,
                label="")

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_gp_ei, 
#             '-o',
#             markersize=15,
#             linewidth=2,
#             label='GP (EI)')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_ra_mean, 
            '-*',
            linewidth=2,
            markersize=15,
            label='Random')


ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
                mean_ra_mean + 2 * mean_ra_std,
                mean_ra_mean - 2 * mean_ra_std,
                alpha=0.2,
                label="")

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_ra_lei, 
#             '-*',
#             linewidth=2,
#             markersize=15,
#             label='Random (logEI)')

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_ra_ei, 
#             '-*',
#             linewidth=2,
#             markersize=15,
#             label='Random (EI)')



# ax.set_title('Active Learning Performance', fontsize=20)
ax.set_xlabel('Batch ID')
ax.set_ylabel('Expected Oracle Error')

# set integer ticklabels
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(loc='best')
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
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/exp_oracle_err_bSine_{}.png".format(chosen_acq))
            plt.close()
        elif chosen_case == 2:
            plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/exp_oracle_err_taylor_{}.png".format(chosen_acq))
            plt.close()
# %%
