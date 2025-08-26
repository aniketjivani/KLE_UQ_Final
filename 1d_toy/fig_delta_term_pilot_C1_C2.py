# Delta term pilot samples.
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


# load as dtype int
input_list_HF_Idx = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LFIdx.txt")[:5].astype(int) - 1

input_list_LF_scaled = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LF_Pilot_scaled.txt")[:200, :][input_list_HF_Idx, :]

# %%
lb_c1 = np.array([40, 60])
ub_c1 = np.array([60, 80])

lb_c2 = np.array([40, 30])
ub_c2 = np.array([60, 50])

input_list_HF_c1 = lb_c1 + (input_list_HF_scaled + 1) * (ub_c1 - lb_c1) / 2

input_list_HF_c2 = lb_c2 + (input_list_HF_scaled + 1) * (ub_c2 - lb_c2) / 2

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

x = np.linspace(0, 0.1, 250)
y_LF_c1 = np.stack([taylor_approx(x, input_list_HF_c1[i, 0], input_list_HF_c1[i, 1]) for i in range(5)]).T
y_LF_c2 = np.stack([b_sine_approx(x, input_list_HF_c2[i, 0], input_list_HF_c2[i, 1]) for i in range(5)]).T

y_HF_c1 = np.stack([spiked_waveform(x, input_list_HF_c1[i, 0], input_list_HF_c1[i, 1]) for i in range(5)]).T
y_HF_c2 = np.stack([spiked_waveform(x, input_list_HF_c2[i, 0], input_list_HF_c2[i, 1]) for i in range(5)]).T

yd_c1 = y_HF_c1 - y_LF_c1
yd_c2 = y_HF_c2 - y_LF_c2


# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

ax[0].plot(x, yd_c1, linewidth=2.5)
ax[0].set_xlabel(r"$x$", fontsize=20)
ax[0].set_ylabel(r"$y_{\Delta}$", fontsize=20)
ax[0].set_title("C1", fontsize=24)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[0].grid(True)
ax[0].set_xlim([0, 0.1])

ax[1].plot(x, yd_c2, linewidth=2.5)
ax[1].set_xlabel(r"$x$", fontsize=20)
ax[1].set_title("C2", fontsize=24)
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].grid(True)
ax[1].set_xlim([0, 0.1])

plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/delta_term_pilot_samples_1d_toy_C1_C2.png", bbox_inches='tight')
# %%
