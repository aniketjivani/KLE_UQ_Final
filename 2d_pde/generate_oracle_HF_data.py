# %%
# use designs generated via `generate_oracle_design_2d.jl` to generate HF oracle data using functions from `run_hf_lf_pde.py`

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
import os
import copy
from rich.progress import track
import sys
sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde")
# import register_parula as rp
# parula_colors = rp._parula_data

from run_hf_lf_pde import getGridQuantities, getVelocitiesGeneric, getPhiForThetaFOU, plotPhiForThetaGeneric

# %%
nx = 64
ny = 64

gridQuantities = getGridQuantities(nx, ny)
u_vel, v_vel = getVelocitiesGeneric(gridQuantities)

oracle_input_list = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_oracle_HF.txt")

n_oracle = oracle_input_list.shape[0]

dir_save = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_oracle_HF_data"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

for sid in range(1, n_oracle + 1):
    plotPhiForThetaGeneric(gridQuantities,
                           sid,
                           u_vel,
                           v_vel,
                           theta_s = oracle_input_list[sid - 1, 0],
                           theta_h = oracle_input_list[sid - 1, 1],
                           theta_x = oracle_input_list[sid - 1, 2],
                           theta_y = oracle_input_list[sid - 1, 3],
                           savefig=False,
                           savedata=True,
                           savedatadir=dir_save)


# %%
