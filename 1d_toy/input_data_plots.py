# %%
# Plot data from an arbitrary batch (HF Points and LF Points Scatter). Save to file.
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap

import numpy as np
import os
import sys
from rich.progress import track
import torch
import pickle

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf

# %%
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


# %%
# 40 pts acquired, no repeats.
acq_ei_suffix = "EI_long"
acq_lei_suffix = "logEI_long"

data_dir = "./1d_toy/1d_pred_data"
inputs_dir = "./1d_toy/1d_inputs"

NREPS = 5
chosen_rep = 1

rep_data_ei = os.path.join(data_dir, "rep_{:03d}_{}".format(chosen_rep, acq_ei_suffix))
rep_data_lei = os.path.join(data_dir, "rep_{:03d}_{}".format(chosen_rep, acq_lei_suffix))

rep_inputs_ei = os.path.join(inputs_dir, "rep_{:03d}_{}".format(chosen_rep, acq_ei_suffix))
rep_inputs_lei = os.path.join(inputs_dir, "rep_{:03d}_{}".format(chosen_rep, acq_lei_suffix))


with open("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data/rep_001/gp_batch_01.pkl", 'rb') as f:
    gp_data = pickle.load(f)

# %%
ntest = 40

lb = [40, 30]
ub = [60, 50]

test_X1 = torch.linspace(lb[0], ub[0], ntest)
test_X2 = torch.linspace(lb[1], ub[1], ntest)

# create tensor of all combinations of test_X1 and test_X2 (meshgrid and flatten)

test_X1_grid, test_X2_grid = torch.meshgrid(test_X1, test_X2)
test_X = torch.stack([test_X1_grid.flatten(), test_X2_grid.flatten()], dim=-1)


# %%

# GP Acquired and Random Points side-by-side.
# Pilot points are in the same color. New points follow a colormap !!
# (plot with two colormaps)
# if colormap is used, line traces connecting new points are not needed.

# compute the predictive mean and variance
# with torch.no_grad():
    # posterior_unscaled = gp.outcome_transform.untransform_posterior(gp.posterior(test_X))
#     posterior_unscaled = gp.posterior(test_X)
#     mean_gp = posterior_unscaled.mean
#     variance_gp = posterior_unscaled.variance

    # lower_gp, upper_gp = posterior_dist.mvn.confidence_region()
    # utilities = logEI(test_X.unsqueeze(dim=1))
    
mean_gp = gp_data["mean_gp"]
variance_gp = gp_data["var_gp"]
utilities = gp_data["utilities"]
# with torch.no_grad():
#        utilities = gp_data["utilities"](test_X.unsqueeze(dim=1))

fig, ax = plt.subplots(1, 3, figsize=(11, 4))

# plot the predictive mean
ct1 = ax[0].contourf(test_X1_grid, test_X2_grid, 
                     mean_gp.reshape(ntest, ntest),
                    #  np.exp(mean_gp.reshape(ntest, ntest)), 
                     cmap='viridis')
ax[0].set_title('Mean')
c1 = fig.colorbar(ct1, ax=ax[0], fraction=0.046, pad=0.04)
# plot the predictive variance
ct2 = ax[1].contourf(test_X1_grid, test_X2_grid, torch.sqrt(variance_gp.reshape(ntest, ntest)), cmap='viridis')
ax[1].set_title('Std. Dev.')
c2 = fig.colorbar(ct2, ax=ax[1], fraction=0.046, pad=0.04)
# plot the utility
ct3 = ax[2].contourf(test_X1_grid, test_X2_grid, utilities.reshape(ntest, ntest), cmap='viridis')
ax[2].set_title('Utility')
c3 = fig.colorbar(ct3, ax=ax[2], fraction=0.046, pad=0.04)
# Now overlay selected candidate on utility plot
# ax[2].scatter(candidate[0][0], candidate[0][1], 
#             s=100,
#             c='gold',
#             marker='*',
#             edgecolors='k',
#             linewidths=1.5,
#             clip_on=False,
# )

def sci_notation_formatter(x, pos):
    # format numbers based on their exponent
    if np.abs(x) < 1e-5:
       return f'{x:.2f}'
    elif np.abs(x) < 1e-2:
       return f'{x:.2f}'
    else:
       s = f'{x:.2e}'
       return f"{s.split('e')[0]}e{int(s.split('e')[1])}"

c1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
c2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
c3.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

# c3.ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_notation_formatter))
# c3.ax.tick_params(axis='y', which='major', pad=1)

# change aspect ratio of the plots
for ia, a in enumerate(ax):
    a.set_aspect('equal')
                # c='red', marker='x', s=100)
    a.set_xlabel(r"$a$")
    if ia == 0:
        a.set_ylabel(r"$b$")
plt.tight_layout()

# # save and remove margins
# plt.savefig(os.path.join(plot_dir, 
#             "rep_{:03d}".format(rep_num), 
#             "batch_{:02d}_contours_{}_1d.png".format(batch_num, acq_func)), 
#             bbox_inches='tight')

# plt.close()


# %% 

