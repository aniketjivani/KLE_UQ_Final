# load errors, process them through a Single or Multitask GP Object and select the next point to evaluate.
# %%
import numpy as np
import torch
import gpytorch
import botorch
import os
import pickle
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt
# set dpi
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

errors_test_round1 = np.array([0.367452, -0.023629, 0.274604, -0.212587, 0.126809])

lb = [40, 30]
ub = [60, 50]

# %%
# we will do both random and active learning point selection in the same script simultaneously.
torch.manual_seed(20240987)
random_acq = torch.rand(2)
print(random_acq)

# rescale the random_acq to the domain of the function
# elementwise multiply lb and ub to the random_acq
random_acq_rescaled = torch.unsqueeze(random_acq * (torch.tensor(ub) - torch.tensor(lb)) + torch.tensor(lb), 1)
print(random_acq_rescaled)


# %%
# Define GP with Matern Kernel, fit to data and select next point based on LogExpectedImprovement

train_X =  torch.tensor([[59.397, 30.5025],
                              [40.804, 30.201],
                              [58.2915, 49.4975],
                              [40.402, 49.196],
                              [50.1508, 39.1457]],
                              dtype=torch.double)

train_Y = torch.tensor(errors_test_round1, dtype=torch.double).unsqueeze(-1)    

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
    # outcome_transform=Standardize(m=1),
)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())

bounds = torch.stack([torch.tensor(lb), 
                      torch.tensor(ub)]).to(torch.double)
candidate, acq_value = optimize_acqf(
    logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)

print(candidate.T, acq_value)

# %% Plot GP mean, variance and utility contours!



# create a fine grid of test points for contour plots.
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
    utilities = logEI(test_X.unsqueeze(dim=1))


# Contour Plots

fig, ax = plt.subplots(1, 3, figsize=(11, 4))

# now ensure that everything is in the correct shape when we reshaped.


# plot the predictive mean
ct1 = ax[0].contourf(test_X1_grid, test_X2_grid, np.exp(mean_gp.reshape(ntest, ntest)), levels=25, cmap='viridis')
ax[0].set_title('Mean')
fig.colorbar(ct1, ax=ax[0], fraction=0.046, pad=0.04)
# plot the predictive variance
ct2 = ax[1].contourf(test_X1_grid, test_X2_grid, torch.sqrt(variance_gp.reshape(ntest, ntest)), levels=25, cmap='viridis')
ax[1].set_title('Std. Dev.')
fig.colorbar(ct2, ax=ax[1], fraction=0.046, pad=0.04)
# plot the utility
ct3 = ax[2].contourf(test_X1_grid, test_X2_grid, utilities.reshape(ntest, ntest), levels=25, cmap='viridis')
ax[2].set_title('Utility')
fig.colorbar(ct3, ax=ax[2], fraction=0.046, pad=0.04)
# Now overlay selected candidate on utility plot
ax[2].scatter(candidate[0][0], candidate[0][1], 
            s=100,
            c='gold',
            marker='*',
            edgecolors='k',
            linewidths=1.5,
            clip_on=False,
)
# change aspect ratio of the plots
for ia, a in enumerate(ax):
    a.set_aspect('equal')
                # c='red', marker='x', s=100)
    a.set_xlabel(r"$a$")
    if ia == 0:
        a.set_ylabel(r"$b$")
plt.tight_layout()

# save and remove margins
plt.savefig("../Plots/1d_toy_plots/scratch_bo_contour_logEI_1d.png", bbox_inches='tight')


# %%
#   mean = prediction.mean
#             lower, upper = prediction.confidence_region()

#             tr_x = submodel.train_inputs[0].detach().numpy()
#             tr_y = submodel.train_targets.detach().numpy()

#             # Plot training data as black stars
#             ax.scatter(tr_x, tr_y, c='k', marker='*',
#                        s=25 , label='Training Data')
#             # changed from std_obj[i]
#             ax.errorbar(
#                 tr_x, tr_y, yerr=std_obj[:, i], fmt="*", color='black', linestyle='none')
#             # Predictive mean
#             ax.plot(test_x.numpy(), mean.numpy(),
#                     color_plots[i], label='GP Mean')
#             # oracle information
#             ax.plot(test_x.numpy(), betas[i, :],
#                     color_plots[i],
#                     linestyle='dashed',
#                     # color='skyblue',
#                     label='True Value')
#             # Shade in confidence with same color as predictive mean
#             ax.fill_between(test_x.numpy(),
#                             lower.detach().numpy(),
#                             upper.detach().numpy(),
#                             alpha=0.3,
#                             color=color_plots[i],
#                             label='GP CI')
#             ax.set_xlim([test_lb - 0.25, test_ub + 0.25])
#             ax.set_ylim([-1, 3])


                # ax.plot(test_x.numpy(), utils_to_plot[:, acq_iter],
                #         color='purple',
                #         linewidth=2,
                #         label='')

                # # Overlay the selected candidate
                # cd_to_plot = acq_to_plot[acq_iter]
                # ax.scatter(cd_to_plot,
                #            obj_to_plot[acq_iter],
                #            s=100,
                #            c='gold',
                #            marker='*',
                #            edgecolors='k',
                #            linewidths=1.5,
                #            label="Selected Candidate")
                # ax.set_ylim([0, 1])
                # ax.set_xlim([test_lb - 0.25, test_ub + 0.25])
                # ax.set_title(
                #     'Acquition Function Utility') # r'$a_{\gamma,\mu_{0}}(\xi)')
