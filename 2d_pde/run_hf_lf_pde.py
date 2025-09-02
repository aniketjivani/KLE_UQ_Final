# %%
# HF and LF runs for PDE, verify correctness and generate pilot sets. Also make plots for sample outputs.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import copy
from rich.progress import track
import sys
sys.path.insert(0, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde")
import register_parula as rp
parula_colors = rp._parula_data

# %%
def getGridQuantities(nx, ny):
    x = np.linspace(0, 1, nx + 1)  # x-Grid
    y = np.linspace(0, 1, ny + 1)  # y-Grid
    xm = x[:-1] + (x[1] - x[0]) / 2  # x-Grid
    ym = y[:-1] + (y[1] - y[0]) / 2  # y-Grid
    XM, YM = np.meshgrid(xm, ym)
    dx = xm[1] - xm[0]
    dy = ym[1] - ym[0]
    dxi = 1 / dx
    dyi = 1 / dy
    dxi2 = 1 / dx ** 2
    dyi2 = 1 / dy ** 2
    return x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2
# %%
def getVelocitiesGeneric(gridQuantities):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    nx = x.shape[0] - 1
    ny = y.shape[0] - 1
    
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))
    for i in range(nx + 1):
        for j in range(ny):
            u[i, j] = 1 / 10 - (np.sin(np.pi * x[i])) ** 2 * (
                        np.sin(np.pi * (ym[j] - 0.05)) * np.cos(np.pi * (ym[j] - 0.05)) -
                        np.sin(np.pi * (ym[j] + 0.05)) * np.cos(np.pi * (ym[j] + 0.05)))
    for i in range(nx):
        for j in range(ny + 1):
            v[i, j] = np.sin(np.pi * xm[i]) * np.cos(np.pi * xm[i]) * (
                        (np.sin(np.pi * (y[j] - 0.05))) ** 2 -
                        (np.sin(np.pi * (y[j] + 0.05))) ** 2)
            
    return u, v
# %% 
def getPhiForThetaFOU(gridQuantities, u_vel, v_vel, theta_s=0.01, theta_h=0.05, theta_x=0.3, theta_y=0.55, alpha=1e-2):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    nx = x.shape[0] - 1
    ny = y.shape[0] - 1

    omega = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            omega[i, j] = ((theta_s) / (2 * np.pi * theta_h**2)) * (np.exp(-((theta_x - xm[i]) ** 2 + (theta_y - ym[j]) ** 2) / (2 * theta_h ** 2)) - np.exp(-((xm[i] - theta_x + 0.05) ** 2 + (ym[j] - theta_y + 0.05) ** 2) / (2 * theta_h ** 2)))
    # phi_sols = []
    CFL = 0.8
    maxU = np.max(np.abs(u_vel))
    maxU = np.max([maxU, np.max(np.abs(v_vel))])
    dt_c = CFL * dx / maxU
    dt_v = CFL * dx ** 2 / alpha / 4
    dt = min(dt_c, dt_v)
    # print("Timestep chosen: ", dt)
    tf = 2.5               
    phi = np.zeros((nx, ny))
    phi_old = phi.copy()
    # Loop through time
    # t = 0
    n_steps = int(np.ceil(tf/dt))
    # for _ in track(range(n_steps), description="Processing for n timesteps"):
    for _ in range(n_steps):
        # Loop through space
        # for i in range(nx):
            # for j in range(ny):

        phi_im1 = np.roll(phi_old, 1, axis=0)
        phi_ip1 = np.roll(phi_old, -1, axis=0)
        phi_jm1 = np.roll(phi_old, 1, axis=1)
        phi_jp1 = np.roll(phi_old, -1, axis=1)

        # Diffusion (explicit)
        diff = alpha * dxi2 * (phi_ip1 - 2 * phi_old + phi_im1) + alpha * dyi2 * (phi_jm1 - 2 * phi_old + phi_jp1)

        ue = u_vel[1:, :]
        uw = u_vel[:-1, :]
        un = v_vel[:, 1:]
        us = v_vel[:, :-1]

        phi_e = np.where(ue > 0, phi_old, phi_jp1)
        phi_w = np.where(uw > 0, phi_jm1, phi_old)

        phi_n = np.where(un > 0, phi_old, phi_ip1)
        phi_s = np.where(us > 0, phi_im1, phi_old)

        conv_x = dxi * (ue * phi_e - uw * phi_w)
        conv_y = dyi * (un * phi_n - us * phi_s)
        conv = - (conv_x + conv_y)

        phi = phi_old + dt * (conv + diff + omega)

        phi_old = phi.copy()
        # phi_sols.append(phi_old.T)
    # return omega, phi.T, phi_sols
    return omega, phi.T

# %%
def plotPhiForThetaGeneric(gridQuantities, sid, u_vel, v_vel, 
                           theta_s=0.01, 
                           theta_h=0.05, 
                           theta_x=0.3, 
                           theta_y=0.55, 
                           savedatadir=None, 
                           savefigdir=None, 
                           savefig=False, 
                           savedata=False,
                           alpha=1e-2,
                           colors_to_use=None,
                           minv = None,
                           maxv = None):
    
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    fidelity = ""
    if x.shape[0] - 1 == 64:
        fidelity += "HF"
    elif x.shape[0] - 1 == 32:
        fidelity += "LF"

    _, phi_data = getPhiForThetaFOU(gridQuantities, u_vel, v_vel, 
                                     theta_s=theta_s, 
                                     theta_h=theta_h, 
                                     theta_x=theta_x, 
                                     theta_y=theta_y,
                                     alpha=alpha)
    
    if savedata:
        if fidelity == "LF":
            np.savetxt(os.path.join(savedatadir, "LF_2D_Run{:04d}.txt".format(sid)), phi_data)
        elif fidelity == "HF":
            np.savetxt(os.path.join(savedatadir, "HF_2D_Run{:04d}.txt".format(sid)), phi_data)
    
    if savefig:
        plt.figure(figsize=(6, 6))
        if (minv is not None) and (maxv is not None):
            plt.imshow(phi_data,
                     origin="lower",
                     extent=[0, 1, 0, 1],
                     vmin=minv,
                     vmax=maxv,
                     cmap=colors_to_use)
        else:
            plt.imshow(phi_data,
                 origin="lower",
                 extent=[0, 1, 0, 1],
                 cmap=colors_to_use)
        if fidelity == "HF":
            plt.xlabel(r"$x$", fontsize=24)
        else:
            # set color to white
            plt.xlabel(r"$x$", fontsize=24, color='white')
        plt.ylabel(r"$y$", fontsize=24)

        # set colorbar tick fontsize

        cbar = plt.colorbar()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(20)

        # set tick labels format (2 decimal places)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # turn off x and y ticks
        plt.xticks([])
        plt.yticks([])

        # plt.title(
        #     r"$\theta_s$={:.2f} \\n $\theta_h$={:.2f} \\n $\theta_x$={:.2f}, $\theta_y$={:.2f}".format(
        #         theta_s, theta_h, theta_x, theta_y
        #     ), fontsize=22
        # )
        if fidelity == "LF":
            plt.title(
                "$\\theta_s$={:.2f}, $\\theta_h$={:.2f}\n$\\theta_x$={:.2f}, $\\theta_y$={:.2f}".format(
                    theta_s, theta_h, theta_x, theta_y
                ), fontsize=22
            )
        plt.tight_layout()
        plt.savefig(os.path.join(savefigdir, "{}_2D_Run{:04d}.jpg".format(fidelity, sid)), dpi=300, bbox_inches='tight')
        plt.close()
        
    print("processed data for Run {:04d}".format(sid))
    # print ylims
    # print("Phi min: ", np.min(phi_data), " Phi max: ", np.max(phi_data))
# %% Example Usage
nx = 32
ny = 32
gridQuantities = getGridQuantities(nx, ny)
runPilot = False
u_vel, v_vel = getVelocitiesGeneric(gridQuantities)
if runPilot:
    if nx == 32:
        pilot_scaled = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LF_Pilot_scaled.txt")[:1000, :]
        dir_save = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirLF/pilotLF"
    elif nx == 64:
        pilot_scaled = np.loadtxt("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_HF_Pilot_scaled.txt")[:50, :]
        dir_save = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirHF/pilotHF"
        
    lb = np.array([0.01, 0.05, 0.3, 0.55])
    ub = np.array([0.05, 0.08, 0.7, 0.85])

    inputs_orig = lb + (pilot_scaled + 1) * 0.5 * (ub - lb)
    u_vel, v_vel = getVelocitiesGeneric(gridQuantities)

    n_pilot = inputs_orig.shape[0]



    for sid in range(1, n_pilot + 1):
        plotPhiForThetaGeneric(gridQuantities, 
                            sid,
                            u_vel, v_vel,
                            theta_s=inputs_orig[sid - 1, 0],
                            theta_h=inputs_orig[sid - 1, 1],
                            theta_x=inputs_orig[sid - 1, 2],
                            theta_y=inputs_orig[sid - 1, 3],
                            savefig=False,
                            savedata=True,
                            savedatadir=dir_save,
                            colors_to_use=ListedColormap(parula_colors)
                            )

ts_vals = [0.05, 0.05, 0.01, 0.02]
th_vals = [0.07, 0.08, 0.05, 0.07]
tx_vals = [0.41, 0.48, 0.69, 0.69]
ty_vals = [0.56, 0.77, 0.67, 0.83]

minv_sample = [-0.2081, -0.2929, -0.054, -0.1312]
maxv_sample = [0.3219, 0.1355, 0.1006, 0.0807]

for sid, (ts, th, tx, ty) in enumerate(zip(ts_vals, th_vals, tx_vals, ty_vals)):
    plotPhiForThetaGeneric(gridQuantities, 
                           sid,
                           u_vel, v_vel,
                           theta_s=ts,
                           theta_h=th,
                           theta_x=tx,
                           theta_y=ty,
                           savefig=True,
                           savefigdir="/Users/ajivani/Desktop/Research/KLE_UQ_Final/Plots/2d_pde_plots",
                           colors_to_use=ListedColormap(parula_colors),
                            minv=minv_sample[sid],
                            maxv=maxv_sample[sid]
                           )
# %%

