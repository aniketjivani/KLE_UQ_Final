# %%
# Plot snapshots of HF and LF data for C1, C2 in 1D toy problem.

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
# set explicit fontsizes for ticks, lables, legend and title
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# plt.rc('axes', labelsize=16)
# plt.rc('legend', fontsize=14, edgecolor="none", frameon=True)
# plt.rc('figure', titlesize=16)

# %%
def spiked_waveform(x, a, b):
    """
    Function that takes in two parameters, a and b and returns a spiked waveform
    of the form exp(-ax)*sin(bx).
    """
    return np.exp(-a * x) * np.sin(b * x)

def taylor_approx(x, a, b):
    """
    Function that returns Taylor series approximation of the spiked waveform exp(-ax)*sin(bx).
    This acts as our low-fidelity model approximation.
    """
    sin_taylor = b * x - (b**3 / math.factorial(3)) * x**3 + (b**5 / math.factorial(5)) * x**5
    return np.exp(-a * x) * sin_taylor

def b_sine_approx(x, a, b):
    """
    Function that returns a degraded version of Bhaskara's sine approximation on the interval [0, π].
    """
    bx_d = x * (180 / np.pi) * b
    return np.exp(-a * x) * (3.5 * bx_d * (180 - bx_d)) / (15000 - bx_d * (180 - bx_d))


# %%
a_c1 = [40, 50, 55]
b_c1 = [71, 60, 80]
b_c2 = [41, 30, 50]

x = np.linspace(0, 0.1, 250)

y_hf1 = [spiked_waveform(x, a_c1[i], b_c1[i]) for i in range(3)]
y_lf1 = [taylor_approx(x, a_c1[i], b_c1[i]) for i in range(3)]

y_hf2 = [spiked_waveform(x, a_c1[i], b_c2[i]) for i in range(3)]
y_lf2 = [b_sine_approx(x, a_c1[i], b_c2[i]) for i in range(3)]


# %%

fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)

# plot y_hf1 and y_lf1 in the top row.
for i in range(3):
    ax[0, i].plot(x, y_hf1[i], color='blue', label="HF", linewidth=2)
    ax[0, i].plot(x, y_lf1[i], color='orange', label="LF", linewidth=2)
    ax[0, i].set_title(r"C1: $a$ = {}, $b$ = {}".format(a_c1[i], b_c1[i]), fontsize=20)
    if i == 0:
        ax[0, i].set_ylabel(r"$y$", fontsize=20)
    ax[0, i].legend(fontsize=13)
    ax[0, i].grid()
    ax[0, i].set_xlim([0, 0.1])
    ax[0, i].tick_params(axis='both', which='major', labelsize=14)


for i in range(3):
    ax[1, i].plot(x, y_hf2[i], color='blue', label="HF", linewidth=2)
    ax[1, i].plot(x, y_lf2[i], color='orange', label="LF", linewidth=2)
    ax[1, i].set_title(r"C2: $a$ = {}, $b$ = {}".format(a_c1[i], b_c2[i]), fontsize=20)
    if i == 0:
        ax[1, i].set_ylabel(r"$y$", fontsize=20)
    ax[1, i].set_xlabel(r"$x$", fontsize=20)
    ax[1, i].legend(fontsize=13)
    ax[1, i].grid()
    ax[1, i].set_xlim([0, 0.1])
    ax[1, i].tick_params(axis='both', which='major', labelsize=14)

plt.subplots_adjust(wspace=0.2, hspace=0.3)
# fig.tight_layout()

plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/HF_LF_snapshots_C1_C2.png", bbox_inches='tight')

# %%
