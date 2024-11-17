# %%
# oracle errors summary
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np
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
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)
# set gridlines and grid alpha and grid linestyle
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.8)

# case objects batch ...jld

def summOracleObj(data_dir,
                  batchID, 
                  repFormat="rep_{:03d}_logEI_NEW2",
                  nReps=1, repID = 1, repStart=1, repEnd=1):
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
data_dir="/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data"
plot_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots"

# N_ACQUIRED = 20
N_ACQUIRED = 50
N_REPS = 1
REP_START = 1
REP_END = 1
mean_errs_gp_lei = np.zeros((N_ACQUIRED, N_REPS))
mean_errs_ra_lei = np.zeros((N_ACQUIRED, N_REPS))

mean_errs_gp_ei = np.zeros((N_ACQUIRED, N_REPS))
mean_errs_ra_ei = np.zeros((N_ACQUIRED, N_REPS))

for jIdx, j in enumerate(range(REP_START, REP_END+1)):
       for i in track(range(N_ACQUIRED), description="Processing each batch"):
             _, _, mean_err_gp, mean_err_ra = summOracleObj(data_dir, i, repFormat="rep_{:03d}_logEI_NEW2_50", nReps=N_REPS, repID=j, repStart=REP_START, repEnd=REP_END)
             mean_errs_gp_lei[i, jIdx] = mean_err_gp
             mean_errs_ra_lei[i, jIdx] = mean_err_ra
             _, _, mean_err_gp, mean_err_ra = summOracleObj(data_dir, i, repFormat="rep_{:03d}_EI_NEW2_50", nReps=N_REPS, repID=j, repStart=REP_START, repEnd=REP_END)
             mean_errs_gp_ei[i, jIdx] = mean_err_gp
             mean_errs_ra_ei[i, jIdx] = mean_err_ra


# %%
fig, ax = plt.subplots()
# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_gp, 
#             '-o',
#             markersize=15,
#             linewidth=2,
#             label='GP')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_errs_gp_lei, 
            '-o',
            markersize=15,
            linewidth=2,
            label='GP (logEI)')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_errs_gp_ei, 
            '-o',
            markersize=15,
            linewidth=2,
            label='GP (EI)')

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_ra, 
#             '-*',
#             linewidth=2,
#             markersize=15,
#             label='Random')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_errs_ra_lei, 
            '-*',
            linewidth=2,
            markersize=15,
            label='Random (logEI)')

ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
            mean_errs_ra_ei, 
            '-*',
            linewidth=2,
            markersize=15,
            label='Random (EI)')

ax.set_title('Oracle Error summary', fontsize=20)
ax.set_xlabel('Batch ID')
ax.set_ylabel('Average error over grid')

# set integer ticklabels
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.legend(loc='best')
plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "oracle_error_summary.png"))
plt.savefig(os.path.join(plot_dir, "oracle_error_summary_logei_ei_50batch.png"))

# %%
