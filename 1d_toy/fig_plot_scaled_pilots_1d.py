# %%
# scaled pilot plot (\xi_1, \xi_2) for 1d toy problem.

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

from matplotlib.colors import ListedColormap
import os
import numpy as np
import math
import sys

sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/")

import register_parula as rp
parula_colors = rp._parula_data

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", family="serif")
plt.rc("axes.spines", top=True, right=True)

# %%

input_list_HF_scaled = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_HF_Pilot_scaled.txt")[:5, :]

input_list_LF_scaled = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LF_Pilot_scaled.txt")[:200, :]
# %% single scatter plot. 

# LF - red circles
# HF - blue stars

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(input_list_LF_scaled[:, 0], input_list_LF_scaled[:, 1], s=80, marker='o', color='red', edgecolors='k', label="LF")
ax.scatter(input_list_HF_scaled[:, 0], input_list_HF_scaled[:, 1], s=200, marker='*', color='blue', edgecolors='k', linewidths=1.5, label="HF")
ax.set_xlabel(r"$\xi_1$", fontsize=20)
ax.set_ylabel(r"$\xi_2$", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_title("Pilot Design", fontsize=22)
ax.grid(True)
ax.set_xlim([-1.05, 1.05])
ax.set_ylim([-1.05, 1.05])

plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/scaled_pilot_design_1d_toy.png", bbox_inches='tight')


# %%