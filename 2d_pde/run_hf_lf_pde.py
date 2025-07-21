# %%
# HF and LF runs for PDE, verify correctness and generate pilot sets.
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from rich.progress import track
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
    phi_sols = []
    CFL = 0.8
    maxU = np.max(np.abs(u_vel))
    maxU = np.max([maxU, np.max(np.abs(v_vel))])
    dt_c = CFL * dx / maxU
    dt_v = CFL * dx ** 2 / alpha / 4
    dt = min(dt_c, dt_v)
    tf = 2.5               
    phi = np.zeros((nx, ny))
    phi_old = phi.copy()
    # Loop through time
    # t = 0
    n_steps = int(np.ceil(tf/dt))
    for _ in track(range(n_steps), description="Processing for n timesteps"):
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
        phi_sols.append(phi_old.T)
    return omega, phi.T, phi_sols

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
                        #    fidelity="LF",
                           alpha=1e-2):
    
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    if x.shape[0] - 1 == 64:
        fidelity = "HF"
    elif x.shape[0] - 1 == 16:
        fidelity = "LF"

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
        fig, ax = plt.subplots()
        im = ax.imshow(phi_data,
                 origin="lower",
                 extent=[0, 1, 0, 1],
                 cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im)
        ax.set_title(r"θₛ={:.2f} θₕ={:.2f} θₓ={:.2f} θᵧ={:.2f}".format(theta_s, theta_h, theta_x, theta_y))
        fig.savefig(os.path.join(savefigdir, "{}_2D_Run{:04d}.png".format(fidelity, sid)))
        plt.close()
        
    print("processed data for Run {:04d}".format(sid))


# %% First order upwind?



# %% Example Usage
# nx = 64
# ny = 64
# gridQuantities = getGridQuantities(nx, ny)
# u_vel, v_vel = getVelocitiesGeneric(gridQuantities)
# phi_data = getPhiForThetaGeneric(gridQuantities, u_vel, v_vel)
# %%