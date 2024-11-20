import sys
# %%

# print(sys.argv)

torch_seed = sys.argv[1]
lb = [sys.argv[2], sys.argv[3]]
ub = [sys.argv[4], sys.argv[5]]
plot_dir = sys.argv[6]
data_dir = sys.argv[7]
input_dir = sys.argv[8]
rep_num = sys.argv[9]
batch_num = sys.argv[10]
cv_errors = sys.argv[11]
hf_inputs = sys.argv[12] 
lf_inputs = sys.argv[13]
lf_subset = sys.argv[14]
acq_func = sys.argv[15]

# print arguments with which the script is called to verify correct ordering.

print("Running Bayesian Optimization with the following arguments: ")
print("torch_seed: ", torch_seed)
print("lb: ", lb)
print("ub: ", ub)
print("plot_dir: ", plot_dir)
print("data_dir: ", data_dir)
print("input_dir: ", input_dir)
print("rep_num: ", rep_num)
print("batch_num: ", batch_num)

# %%
import numpy as np
import torch
import gpytorch
import botorch
import os
import sys
import pickle
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.analytic import LogExpectedImprovement, ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", family="serif")
plt.rc("axes.spines", top=True, right=True)
# set explicit fontsizes for ticks, lables, legend and title
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=16)


# torch.manual_seed(torch_seed)

# %%
random_acq = torch.rand(2)
# print(random_acq)

# rescale the random_acq to the domain of the function
# elementwise multiply lb and ub to the random_acq
random_acq_rescaled = torch.unsqueeze(random_acq * (torch.tensor(ub) - torch.tensor(lb)) + torch.tensor(lb), 1)
# print(random_acq_rescaled)

# %%

nHF = int(hf_inputs.shape[0] / 2)

train_X = torch.tensor(hf_inputs[:nHF, :], dtype=torch.double)
train_Y = torch.tensor(cv_errors, dtype=torch.double).unsqueeze(-1)

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
# acq = LogExpectedImprovement(model=gp, best_f=train_Y.max())

# to-do: implement constraint for non-repeated points in acquisition function.

# alternately , strategies with q > 1 for quicker evaluations of oracle performance.


if acq_func == "EI":
    acq = ExpectedImprovement(model=gp, best_f=train_Y.max())
elif acq_func == "logEI":
    acq = LogExpectedImprovement(model=gp, best_f=train_Y.max())


bounds = torch.stack([torch.tensor(lb), 
                      torch.tensor(ub)]).to(torch.double)

print("Running Acquisition.")
candidate, acq_value = optimize_acqf(
    # logEI, 
    acq,
    bounds=bounds, 
    q=1, 
    num_restarts=5, 
    raw_samples=20,
)

print("Acquisition complete.")
# %% Save GP object to file:
# with open(os.path.join(data_dir, "rep_{:03d}".format(rep_num), "batch_{:02d}_gp.pkl".format(batch_num),
#                        "wb")) as f:
#     pickle.dump({"gp": gp, 
#                  "utility": logEI}, 
#                  f, 
#                  protocol=pickle.HIGHEST_PROTOCOL)

print("Random acquisition: ", random_acq_rescaled.T)
print("Active learnt acquisition: ", candidate)

# %%
nLF = int(lf_inputs.shape[0]/2)
# nHF = hf_inputs.shape[0]

augmented_LF = np.zeros((2 * (nLF + 1), 2))
augmented_LF[:nLF, :] = lf_inputs[:nLF, :]
augmented_LF[nLF:(nLF + 1), :] = candidate.detach().numpy()

augmented_LF[(nLF + 1):(2 * (nLF + 1) - 1), :] = lf_inputs[nLF:, :]
augmented_LF[(2 * (nLF + 1) - 1), :] = random_acq_rescaled.T.detach().numpy()

augmented_HF = np.zeros((2 * (nHF + 1), 2))
augmented_HF[:nHF, :] = hf_inputs[:nHF, :]
augmented_HF[nHF:(nHF + 1), :] = candidate.detach().numpy()
augmented_HF[(nHF + 1):(2 * (nHF + 1) - 1), :] = hf_inputs[nHF:, :]
augmented_HF[(2 * (nHF + 1) - 1), :] = random_acq_rescaled.T.detach().numpy()

augmented_LF_subset = np.zeros((2 * (nHF + 1)))
augmented_LF_subset[:nHF] = lf_subset[:nHF]
augmented_LF_subset[nHF:(nHF + 1)] = nLF + 1
augmented_LF_subset[(nHF + 1):(2 * (nHF + 1) - 1)] = lf_subset[nHF:]
augmented_LF_subset[(2 * (nHF + 1) - 1)] = nLF + 1


# save to input dir with appropriate rep and batch numbering.
np.savetxt(os.path.join(input_dir,
                            "rep_{:03d}".format(rep_num),
                            "HF_Batch_{:03d}_Final.txt".format(batch_num + 1)), augmented_HF)

np.savetxt(os.path.join(input_dir,
                            "rep_{:03d}".format(rep_num),
                            "LF_Batch_{:03d}_Final.txt".format(batch_num + 1)), augmented_LF)

np.savetxt(os.path.join(input_dir,
                        "rep_{:03d}".format(rep_num),
                        "HF_Batch_{:03d}_Subset_Final.txt".format(batch_num + 1)), augmented_LF_subset, fmt="%d")

#%% Test points for contour plots.

ntest = 30
test_X1 = torch.linspace(lb[0], ub[0], ntest)
test_X2 = torch.linspace(lb[1], ub[1], ntest)

# create tensor of all combinations of test_X1 and test_X2 (meshgrid and flatten)

test_X1_grid, test_X2_grid = torch.meshgrid(test_X1, test_X2)
test_X = torch.stack([test_X1_grid.flatten(), test_X2_grid.flatten()], dim=-1)

# compute the predictive mean and variance
with torch.no_grad():
    # posterior_unscaled = gp.outcome_transform.untransform_posterior(gp.posterior(test_X))
    posterior_unscaled = gp.posterior(test_X)
    mean_gp = posterior_unscaled.mean
    variance_gp = posterior_unscaled.variance

    # lower_gp, upper_gp = posterior_dist.mvn.confidence_region()
    # utilities = logEI(test_X.unsqueeze(dim=1))
    utilities = acq(test_X.unsqueeze(dim=1))


fig, ax = plt.subplots(1, 3, figsize=(11, 4))

# plot the predictive mean
ct1 = ax[0].contourf(test_X1_grid, test_X2_grid, np.exp(mean_gp.reshape(ntest, ntest)), levels=25, cmap='viridis')
ax[0].set_title('Mean')
c1 = fig.colorbar(ct1, ax=ax[0], fraction=0.046, pad=0.04)
# plot the predictive variance
ct2 = ax[1].contourf(test_X1_grid, test_X2_grid, torch.sqrt(variance_gp.reshape(ntest, ntest)), levels=25, cmap='viridis')
ax[1].set_title('Std. Dev.')
c2 = fig.colorbar(ct2, ax=ax[1], fraction=0.046, pad=0.04)
# plot the utility
ct3 = ax[2].contourf(test_X1_grid, test_X2_grid, utilities.reshape(ntest, ntest), levels=25, cmap='viridis')
ax[2].set_title('Utility')
c3 = fig.colorbar(ct3, ax=ax[2], fraction=0.046, pad=0.04)
# Now overlay selected candidate on utility plot
ax[2].scatter(candidate[0][0], candidate[0][1], 
            s=100,
            c='gold',
            marker='*',
            edgecolors='k',
            linewidths=1.5,
            clip_on=False,
)

c1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
c2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
c3.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# change aspect ratio of the plots
for ia, a in enumerate(ax):
    a.set_aspect('equal')
                # c='red', marker='x', s=100)
    a.set_xlabel(r"$a$")
    if ia == 0:
        a.set_ylabel(r"$b$")
plt.tight_layout()

# save and remove margins
plt.savefig(os.path.join(plot_dir, 
            "rep_{:03d}".format(rep_num), 
            "batch_{:02d}_contours_{}_1d.png".format(batch_num, acq_func)), 
            bbox_inches='tight')

plt.close()