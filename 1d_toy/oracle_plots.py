# %%
# oracle errors summary
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np
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
plt.rc('legend', fontsize=8)
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
# data_dir="/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data"
# plot_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots"

# # N_ACQUIRED = 20
# N_ACQUIRED = 50
# N_REPS = 1
# REP_START = 1
# REP_END = 1
# mean_errs_gp_lei = np.zeros((N_ACQUIRED, N_REPS))
# mean_errs_ra_lei = np.zeros((N_ACQUIRED, N_REPS))

# mean_errs_gp_ei = np.zeros((N_ACQUIRED, N_REPS))
# mean_errs_ra_ei = np.zeros((N_ACQUIRED, N_REPS))

# for jIdx, j in enumerate(range(REP_START, REP_END+1)):
#        for i in track(range(N_ACQUIRED), description="Processing each batch"):
#              _, _, mean_err_gp, mean_err_ra = summOracleObj(data_dir, i, repFormat="rep_{:03d}_logEI_NEW2_50", nReps=N_REPS, repID=j, repStart=REP_START, repEnd=REP_END)
#              mean_errs_gp_lei[i, jIdx] = mean_err_gp
#              mean_errs_ra_lei[i, jIdx] = mean_err_ra
#              _, _, mean_err_gp, mean_err_ra = summOracleObj(data_dir, i, repFormat="rep_{:03d}_EI_NEW2_50", nReps=N_REPS, repID=j, repStart=REP_START, repEnd=REP_END)
#              mean_errs_gp_ei[i, jIdx] = mean_err_gp
#              mean_errs_ra_ei[i, jIdx] = mean_err_ra


# %%
# fig, ax = plt.subplots()
# # ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
# #             mean_errs_gp, 
# #             '-o',
# #             markersize=15,
# #             linewidth=2,
# #             label='GP')

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_gp_lei, 
#             '-o',
#             markersize=15,
#             linewidth=2,
#             label='GP (logEI)')

# ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
#             mean_errs_gp_ei, 
#             '-o',
#             markersize=15,
#             linewidth=2,
#             label='GP (EI)')

# # ax.plot(np.linspace(1, N_ACQUIRED, N_ACQUIRED), 
# #             mean_errs_ra, 
# #             '-*',
# #             linewidth=2,
# #             markersize=15,
# #             label='Random')

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

# ax.set_title('Oracle Error summary', fontsize=20)
# ax.set_xlabel('Batch ID')
# ax.set_ylabel('Average error over grid')

# # set integer ticklabels
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ax.legend(loc='best')
# plt.tight_layout()
# # plt.savefig(os.path.join(plot_dir, "oracle_error_summary.png"))
# plt.savefig(os.path.join(plot_dir, "oracle_error_summary_logei_ei_50batch.png"))

# %%

# %%

# for matern and rk kernels - plot covariance matrix of oracle and cv errors side-by-side for all active learning stages. (checking if choice of kernel corresponds to what is observed w.r.t cov structure, stationarity etc. of errors!)

acq_mat_suffix = "matern"
acq_rq_suffix = "rkkernel"

data_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data"
inputs_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_inputs"
NREPS = 5
NPILOTHF = 5
NBATCH = 50
chosen_rep = 1

rep_data_mat = os.path.join(data_dir, "rep_{:03d}_{}".format(chosen_rep, acq_mat_suffix))
rep_data_rq = os.path.join(data_dir, "rep_{:03d}_{}".format(chosen_rep, acq_rq_suffix))

rep_inputs_mat = os.path.join(inputs_dir, "rep_{:03d}_{}".format(chosen_rep, acq_mat_suffix))
rep_inputs_rq = os.path.join(inputs_dir, "rep_{:03d}_{}".format(chosen_rep, acq_rq_suffix))

# %%

def oracle_cv_comparisons(acq_data_dir, acq_inputs_dir):
      fig, ax = plt.subplots(NBATCH, 2, figsize=np.array([225, 3400])/25.4)
      for nb in range(NBATCH):
       batch_obj = np.load(os.path.join(acq_data_dir,
                              "case_objects_batch_{:03d}.npz".format(nb)))
          
       with open(os.path.join(acq_data_dir, "gp_batch_{:02d}.pkl".format(nb)), 'rb') as f:
            gp_data = pickle.load(f)


       oracle_err_gp = batch_obj["oracle_gp"]
       oracle_err_ra = batch_obj["oracle_ra"]

       c0 = ax[nb, 0].imshow(oracle_err_gp,
                             origin="lower",
                             cmap='viridis')
       
       ct0 = fig.colorbar(c0, ax=ax[nb, 0], fraction=0.046, pad=0.04)

       c1 = ax[nb, 1].imshow(oracle_err_ra,
                             origin="lower",
                             cmap='viridis')
       
       ct1 = fig.colorbar(c1, ax=ax[nb, 1], fraction=0.046, pad=0.04)

       ct0.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
       ct1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

       ax[nb, 0].set_ylabel(r"$b$")
       for ia, a in enumerate(ax[nb, :]):
            a.set_aspect('equal')


      for ia, a in enumerate(ax[-1, :]):
        a.set_xlabel(r"$a$")

      fig.tight_layout()
      #       fig.suptitle("Oracle GP vs RA Errors")
    



oracle_cv_comparisons(rep_data_rq, rep_inputs_rq)


       

       
            






# %%
