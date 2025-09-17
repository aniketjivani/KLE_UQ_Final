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
                     repFormat.format(repID),
                    "case_objects_batch_{:03d}.npz".format(batchID)))
    oracle_gp = data_dict["oracle_err_gp"]
    oracle_ra = data_dict["oracle_err_ra"]
    
    mean_gp = oracle_gp.mean()
    mean_ra = oracle_ra.mean()
    
    return oracle_gp, oracle_ra, mean_gp, mean_ra


# %% 
N_ACQUIRED = 27
# N_ACQUIRED = 20
N_REPS = 3
mean_gp_all_reps = np.zeros((N_ACQUIRED, N_REPS))
mean_ra_all_reps = np.zeros((N_ACQUIRED, N_REPS))

allModes = 0
chosen_acq = "EI"
saveFig = False


if allModes:
    data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_pred_data_all_modes_{}".format(chosen_acq)
else:
    # data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_pred_data_{}".format(chosen_acq)
    # data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_pred_data_log_err_{}".format(chosen_acq)
    data_dir_all = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_pred_data_trunc_{}".format(chosen_acq)


oracle_gp_all = np.zeros((N_ACQUIRED, N_REPS, 1000))
oracle_ra_all = np.zeros((N_ACQUIRED, N_REPS, 1000))

# for rID in range(N_REPS):
for i, rID in enumerate([0, 2, 5]):
    for bID in range(N_ACQUIRED):
       oracle_gp, oracle_ra, mean_gp, mean_ra = summOracleObj(data_dir_all,
                  bID, 
                  repFormat="rep_{:03d}",
                  repID=rID + 1)
    
       mean_gp_all_reps[bID, i] = mean_gp
       mean_ra_all_reps[bID, i] = mean_ra
       oracle_gp_all[bID, i, :] = oracle_gp
       oracle_ra_all[bID, i, :] = oracle_ra    


set_log=False
if set_log:
    mean_gp_mean = np.log10(mean_gp_all_reps).mean(axis=1)
    mean_ra_mean = np.log10(mean_ra_all_reps).mean(axis=1)

    mean_gp_std = np.log10(mean_gp_all_reps).std(axis=1)
    mean_ra_std = np.log10(mean_ra_all_reps).std(axis=1)
else:
    mean_gp_mean = mean_gp_all_reps.mean(axis=1)
    mean_gp_std = mean_gp_all_reps.std(axis=1)

    mean_ra_mean = mean_ra_all_reps.mean(axis=1)
    mean_ra_std = mean_ra_all_reps.std(axis=1)

fig, ax = plt.subplots()
ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_gp_mean, 
            # '-o',
            # markersize=15,
            linewidth=2,
            label="GP ({})".format(chosen_acq))

ax.fill_between(np.linspace(1, N_ACQUIRED, N_ACQUIRED),
                mean_gp_mean + 2 * mean_gp_std,
                mean_gp_mean - 2 * mean_gp_std,
                alpha=0.2,
                label="")



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

ax.set_xlabel('Batch ID')
ax.set_ylabel('Expected Oracle Error')

# set integer ticklabels
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(loc='best')
plt.tight_layout()


# 2d_pde does not have cases
# %%
chosen_rep = 1
chosen_batches = [10, 20, 30, 50]

# plot histograms of each strategy on a single plot
fig, axs = plt.subplots(2, 2, figsize=(12, 5))


for i, a in enumerate(axs.ravel()):
    chosen_batch = chosen_batches[i]
    a.set_title('Batch ID: {}'.format(chosen_batch))

    a.hist(oracle_ra_all[chosen_batch - 1, chosen_rep - 1, :], 
            alpha=0.5,
            label="Random",
            color='cyan',
            density=True)

    a.hist(oracle_gp_all[chosen_batch - 1, chosen_rep - 1, :], 
            alpha=0.5, 
            label="GP ({})".format(chosen_acq),
            color='goldenrod',
            density=True)


    a.set_xlabel('Oracle Error')
    a.set_xlim([0, 5])

    a.legend(loc='best')
plt.tight_layout()
# %%
