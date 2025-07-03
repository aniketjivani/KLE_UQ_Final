import os
import numpy as np
from sklearn import gaussian_process as gp
import bayes_opt as bo
import copy
import argparse
import os
import pickle
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm


import matplotlib
font = {'family' : 'serif',
        'size'   : 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


random_realizations_idx = np.random.choice(200*200, 20, replace=False)
# random_realizations_idx = np.array([250, 350, 400, 450, 500, 750, 18000])
random_realizations_idx

all_cases = [1, 2]



for chosen_case in all_cases:
    if chosen_case == 1:
        hf_oracle_file = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/HF_Oracle_case_01.jld", "r")
    elif chosen_case == 2:
        hf_oracle_file = h5py.File("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/HF_Oracle_case_02.jld", "r")
    HF_oracle_raw = hf_oracle_file["HF_oracle"][:]
    HF_oracle = np.zeros((200, 200, 250))
    for i in range(HF_oracle_raw.shape[0]):
        HF_oracle[:, :, i] = HF_oracle_raw[i, :, :].T


    # rearrange so its simply 2D
    HF_oracle_ft = np.zeros((200*200, 250))
    cart_ind_y, cart_ind_x = np.unravel_index(np.arange(200*200), (200, 200))
    for idx, (j, i) in enumerate(zip(cart_ind_y, cart_ind_x)):
        HF_oracle_ft[idx, :] = HF_oracle[j, i, :]


    HF_oracle_mean = np.mean(HF_oracle_ft, axis=0)
    HF_oracle_std = np.std(HF_oracle_ft, axis=0)
    xgrid = np.linspace(0, 0.1, 250)

    fig, ax = plt.subplots(figsize=(8, 6))


    for i, idx in enumerate(random_realizations_idx):
        ax.plot(xgrid, HF_oracle_ft[idx, :], 
                color="red", linewidth=1.0, alpha=0.4, 
                label="Realizations" if i == 0 else None, 
                zorder=1)



    ax.plot(xgrid, HF_oracle_mean, color='navy', linewidth=4.5, label=r"")
    ax.fill_between(xgrid, HF_oracle_mean - HF_oracle_std, HF_oracle_mean + HF_oracle_std,
                color='cornflowerblue', alpha=0.5, 
                label=r"$\mu \pm \sigma$")

    ax.fill_between(xgrid, HF_oracle_mean - 2 * HF_oracle_std, HF_oracle_mean + 2 * HF_oracle_std,
                color='dodgerblue', alpha=0.15, label=r"$\mu \pm 2\sigma$")

    ax.set_xlabel(r'$x$', fontsize=22)
    ax.set_ylabel(r"$y_{\mathrm{HF}}$", fontsize=22)
    ax.set_xlim([0.0, 0.1])

    if chosen_case == 1:
        ax.legend(loc='best', fontsize=22)
    
    ax.grid('True', linestyle='--', linewidth=1, alpha=0.7)


    if chosen_case == 1:
        plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/HF_oracle_uncertainty_bSineRange.jpg", bbox_inches='tight')
    elif chosen_case == 2:
        plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/1d_toy_plots/HF_oracle_uncertainty_taylorRange.jpg", bbox_inches='tight')