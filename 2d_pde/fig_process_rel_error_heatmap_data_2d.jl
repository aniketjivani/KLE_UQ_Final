using LatinHypercubeSampling
using Random
using LinearAlgebra
using SpecialFunctions
using Printf
using JLD
using NPZ
using PyCall
using DelimitedFiles
using DataFrames
using Combinatorics
using Serialization
using Plots
np = pyimport("numpy")
sp = pyimport("scipy")
sys = pyimport("sys")
pkl = pyimport("pickle")

include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/utils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/kleUtils.jl")
computeErrors = 1

kle_cases = ["gp", "ra"]
# we will evaluate case of getAllModes = 0 and 1, where 1 is turned on only for the delta term.

kle_kwargs_Δ = (useFullGrid=1,
            getAllModes=0,
            order=3,
            dims=4,
            weightFunction=getWeights2D,
            family="Legendre",
            solver="Tikhonov-L2"
            )

kle_kwargs = (order=3,
            dims=4,
            family="Legendre",
            useFullGrid=1,
            getAllModes=0,
            weightFunction=getWeights2D,
            solver="Tikhonov-L2"
            )

kle_kwargs_HF = (order=3,
            dims=4,
            family="Legendre",
            useFullGrid=1,
            getAllModes=0,
            weightFunction=getWeights2D,
            solver="Tikhonov-L2"
            )

args_dict = Dict("root_dir"=>"/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde",
    "plot_dir"=> "./Plots/2d_pde_plots_all_modes",
            "data_dir"=> "./2d_pde/2d_pred_data_all_modes",
            "input_dir"=> "./2d_pde/2d_inputs_all_modes",
            "dirLFData"=>"./2d_pde/dirLF",
            "dirHFData"=>"./2d_pde/dirHF",
            "NREPS"=> 20,
            "NFOLDS"=> 10,
            "BUDGET_HF"=>40,
            "N_PILOT_LF"=>300,
            "N_PILOT_HF"=>10,
            "acqFunc" => "EI",
            "lb" => [0.01, 0.05, 0.3, 0.55],
            "ub" => [0.05, 0.08, 0.7, 0.85]
       )

if kle_kwargs_Δ.getAllModes == 1
    args_dict["data_dir"] = @sprintf("./2d_pde/2d_pred_data_%s_all_modes", args_dict["acqFunc"])
    args_dict["input_dir"] = @sprintf("./2d_pde/2d_inputs_%s_all_modes", args_dict["acqFunc"])
    args_dict["plot_dir"] = @sprintf("./Plots/2d_pde_plots_%s_all_modes", args_dict["acqFunc"])
elseif kle_kwargs_Δ.getAllModes == 0
    args_dict["data_dir"] = @sprintf("./2d_pde/2d_pred_data_trunc_%s", args_dict["acqFunc"])
    args_dict["input_dir"] = @sprintf("./2d_pde/2d_inputs_trunc_%s", args_dict["acqFunc"])
    args_dict["plot_dir"] = @sprintf("./Plots/2d_pde_plots_trunc_%s", args_dict["acqFunc"])
end



py"""
import numpy as np
import os
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

def getPhiForThetaFOU(gridQuantities, u_vel, v_vel, theta_s=0.01, theta_h=0.05, theta_x=0.3, theta_y=0.55, alpha=1e-2):
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    nx = x.shape[0] - 1
    ny = y.shape[0] - 1

    omega = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            omega[i, j] = ((theta_s) / (2 * np.pi * theta_h**2)) * (np.exp(-((theta_x - xm[i]) ** 2 + (theta_y - ym[j]) ** 2) / (2 * theta_h ** 2)) - np.exp(-((xm[i] - theta_x + 0.05) ** 2 + (ym[j] - theta_y + 0.05) ** 2) / (2 * theta_h ** 2)))
    CFL = 0.8
    maxU = np.max(np.abs(u_vel))
    maxU = np.max([maxU, np.max(np.abs(v_vel))])
    dt_c = CFL * dx / maxU
    dt_v = CFL * dx ** 2 / alpha / 4
    dt = min(dt_c, dt_v)
    tf = 2.5               
    phi = np.zeros((nx, ny))
    phi_old = phi.copy()
    n_steps = int(np.ceil(tf/dt))
    for _ in range(n_steps):
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
    return omega, phi.T

def plotPhiForThetaGeneric(gridQuantities, 
                           u_vel, 
                           v_vel, 
                           theta_s=0.01, 
                           theta_h=0.05, 
                           theta_x=0.3, 
                           theta_y=0.55, 
                           alpha=1e-2,
                           ):
    
    x, y, xm, ym, XM, YM, dx, dy, dxi, dyi, dxi2, dyi2 = gridQuantities

    fidelity = ""
    if x.shape[0] - 1 == 128:
        fidelity += "HF"
    elif x.shape[0] - 1 == 32:
        fidelity += "LF"

    _, phi_data = getPhiForThetaFOU(gridQuantities, u_vel, v_vel, 
                                     theta_s=theta_s, 
                                     theta_h=theta_h, 
                                     theta_x=theta_x, 
                                     theta_y=theta_y,
                                     alpha=alpha)
        
    print("processed data for run")
    return phi_data
"""

reps_to_use = [0, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29] .+ 1

pilot_dir_HF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirHF/pilotHF_trunc"
pilot_dir_LF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirLF/pilotLF_trunc"

nxLF = 32
nyLF = 32
nxHF = 128
nyHF = 128

gridQuantitiesLF = py"getGridQuantities"(nxLF, nyLF)
gridQuantitiesHF = py"getGridQuantities"(nxHF, nyHF)

xLF, yLF, xmLF, ymLF, XMLF, YMLF, dxLF, dyLF, dxiLF, dyiLF, dxi2LF, dyi2LF = gridQuantitiesLF
xHF, yHF, xmHF, ymHF, XMHF, YMHF, dxHF, dyHF, dxiHF, dyiHF, dxi2HF, dyi2HF = gridQuantitiesHF

u_vel_LF, v_vel_LF = py"getVelocitiesGeneric"(gridQuantitiesLF)
u_vel_HF, v_vel_HF = py"getVelocitiesGeneric"(gridQuantitiesHF)

_, pilot_HF_data = readHFFromDir(pilot_dir_HF, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)
pilot_LF_data = readLFFromDir(pilot_dir_LF, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)

N_PILOT_LF = args_dict["N_PILOT_LF"]
N_PILOT_HF = args_dict["N_PILOT_HF"]
N_ACQUIRED = args_dict["BUDGET_HF"]
N_REPS = args_dict["NREPS"]
acqFunc = args_dict["acqFunc"]
lb = args_dict["lb"]
ub = args_dict["ub"]

gp_on_gp = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS);
gp_on_ra = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS);
ra_on_ra = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS);
ra_on_gp = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS);
kle_cases = ["gp", "ra"]

nxLF = 32
nyLF = 32
nxHF = 128
nyHF = 128

gridQuantitiesLF = py"getGridQuantities"(nxLF, nyLF)
gridQuantitiesHF = py"getGridQuantities"(nxHF, nyHF)

xLF, yLF, xmLF, ymLF, XMLF, YMLF, dxLF, dyLF, dxiLF, dyiLF, dxi2LF, dyi2LF = gridQuantitiesLF
xHF, yHF, xmHF, ymHF, XMHF, YMHF, dxHF, dyHF, dxiHF, dyiHF, dxi2HF, dyi2HF = gridQuantitiesHF

u_vel_LF, v_vel_LF = py"getVelocitiesGeneric"(gridQuantitiesLF)
u_vel_HF, v_vel_HF = py"getVelocitiesGeneric"(gridQuantitiesHF)

run_hf_all_acquired = false;
run_lf_all_acquired = false;

hf_gp_flat_all = zeros(nxLF * nyLF, N_ACQUIRED, length(reps_to_use));
hf_ra_flat_all = zeros(nxLF * nyLF, N_ACQUIRED, length(reps_to_use));

lf_gp_flat_all = zeros(nxLF * nyLF, N_ACQUIRED, length(reps_to_use));
lf_ra_flat_all = zeros(nxLF * nyLF, N_ACQUIRED, length(reps_to_use));
# for all reps to use, generate HF data and save to file.
# for (j, repID) in enumerate(reps_to_use[16:end])

# for (j, repID) in zip(16:length(reps_to_use), reps_to_use[16:end])
for (j, repID) in enumerate(reps_to_use)
    println("Starting repetition $repID")
    input_file = joinpath(args_dict["input_dir"], 
    @sprintf("rep_%03d", repID),
    @sprintf("HF_Batch_%03d_Final.txt", N_ACQUIRED))
    
    all_inputs = readdlm(input_file)

    xi_pred_all_gp = all_inputs[(N_PILOT_HF + 1):(N_PILOT_HF + N_ACQUIRED), :]
    xi_pred_all_ra = all_inputs[(N_PILOT_HF + N_ACQUIRED + 1 + N_PILOT_HF):end, :]
    for (i, xi) in enumerate(eachrow(xi_pred_all_gp))
        if run_lf_all_acquired
            phi_lf_gp_temp = py"plotPhiForThetaGeneric"(gridQuantitiesLF,
            u_vel_LF,
            v_vel_LF,
            theta_s = xi[1],
            theta_h = xi[2],
            theta_x = xi[3],
            theta_y = xi[4],
            alpha = 1e-2,
            )
            lf_gp_flat = phi_lf_gp_temp[:]
            lf_gp_flat_all[:, i, j] .= lf_gp_flat;
        end
        if run_hf_all_acquired
            phi_hf_gp_temp = py"plotPhiForThetaGeneric"(gridQuantitiesHF,
            u_vel_HF,
            v_vel_HF,
            theta_s = xi[1],
            theta_h = xi[2],
            theta_x = xi[3],
            theta_y = xi[4],
            alpha = 1e-2,
            )
            _, hf_gp_flat = readAndInterpolateHFFromArray(phi_hf_gp_temp, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)
            hf_gp_flat_all[:, i, j] .= hf_gp_flat;
        end
    end
    for (i, xi) in enumerate(eachrow(xi_pred_all_ra))
        if run_lf_all_acquired
            phi_lf_ra_temp = py"plotPhiForThetaGeneric"(gridQuantitiesLF,
            u_vel_LF,
            v_vel_LF,
            theta_s = xi[1],
            theta_h = xi[2],
            theta_x = xi[3],
            theta_y = xi[4],
            alpha = 1e-2,
            )
            lf_ra_flat = phi_lf_ra_temp[:]
            lf_ra_flat_all[:, i, j] .= lf_ra_flat;
        end
        if run_hf_all_acquired
            phi_hf_ra_temp = py"plotPhiForThetaGeneric"(gridQuantitiesHF,
            u_vel_HF,
            v_vel_HF,
            theta_s = xi[1],
            theta_h = xi[2],
            theta_x = xi[3],
            theta_y = xi[4],
            alpha = 1e-2,
            )
            _, hf_ra_flat = readAndInterpolateHFFromArray(phi_hf_ra_temp, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)
            hf_ra_flat_all[:, i, j] .= hf_ra_flat;
        end
    end
end
if run_hf_all_acquired
    println("Saving HF data for all reps to file")
    open("./2d_pde/err_comparison_HF_data_20reps_2d.jls", "w") do io
        serialize(io, (hf_gp_flat_all, hf_ra_flat_all));
    end
end
if run_lf_all_acquired
    println("Saving LF data for all reps to file")
    open("./2d_pde/err_comparison_LF_data_20reps_2d.jls", "w") do io
        serialize(io, (lf_gp_flat_all, lf_ra_flat_all));
    end
end

hf_data_all = deserialize("./2d_pde/err_comparison_HF_data_20reps_2d.jls");
hf_gp_flat_all = hf_data_all[1];
hf_ra_flat_all = hf_data_all[2];
lf_data_all = deserialize("./2d_pde/err_comparison_LF_data_20reps_2d.jls");
lf_gp_flat_all = lf_data_all[1];
lf_ra_flat_all = lf_data_all[2];

if computeErrors == 1
    # for (j, repID) in enumerate(reps_to_use)
    for (j, repID) in zip(16:length(reps_to_use), reps_to_use[16:end])
        println("Starting repetition $repID")
        rd_seed = 20250531 + repID
        Random.seed!(rd_seed)
        inputs_dir = joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID))
        data_dir = joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID))
        input_file = joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Final.txt", N_ACQUIRED))

        all_inputs = readdlm(input_file)
        xi_pred_all_gp = all_inputs[(N_PILOT_HF + 1):(N_PILOT_HF + N_ACQUIRED), :]
        xi_pred_all_ra = all_inputs[(N_PILOT_HF + N_ACQUIRED + 1 + N_PILOT_HF):end, :]

        inputs_hf_gp_all = 2 * (xi_pred_all_gp .- (1/2)*(lb + ub)') ./ (ub - lb)'
        inputs_hf_ra_all = 2 * (xi_pred_all_ra .- (1/2)*(lb + ub)') ./ (ub - lb)'

        y_hf_gp_all = hf_gp_flat_all[:, :, j];
        y_hf_ra_all = hf_ra_flat_all[:, :, j];

        y_lf_gp_all = lf_gp_flat_all[:, :, j];
        y_lf_ra_all = lf_ra_flat_all[:, :, j];

        for batchID in 0:(N_ACQUIRED - 1)
            println("Starting Batch  $batchID")
            case_objects = npzread(joinpath(data_dir, @sprintf("case_objects_batch_%03d.npz", batchID)));
            kle_data = deserialize(joinpath(data_dir, @sprintf("case_objects_batch_%03d.jls", batchID)));

            cv_gp = case_objects["cv_gp"];
            cv_ra = case_objects["cv_ra"];

            kle_gp_obj = kle_data[1];
            kle_ra_obj = kle_data[2];

            xi_HF_gp_pred = all_inputs[(N_PILOT_HF + batchID + 1):(N_PILOT_HF + N_ACQUIRED), :]
            xi_HF_ra_pred = all_inputs[(N_PILOT_HF + N_ACQUIRED + batchID + 1 + N_PILOT_HF):end, :]

            LF_data, HF_data = nothing, nothing
            inputsHFSubsetIdx = nothing
            xi_lf, xi_hf = nothing, nothing
            xi_lf_scaled, xi_hf_scaled = nothing, nothing
            y_hf_gp_pred = y_hf_gp_all[:, (batchID + 1):end]
            y_hf_ra_pred = y_hf_ra_all[:, (batchID + 1):end]
            if batchID == 0
                xi_hf = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_HF_Pilot_scaled_trunc.txt")
                xi_lf = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LF_Pilot_scaled_trunc.txt")
                xi_hf_scaled = 0.5 * xi_hf .* (ub - lb)' .+ 0.5 * (ub + lb)'
                xi_lf_scaled = 0.5 * xi_lf .* (ub - lb)' .+ 0.5 * (ub + lb)'
                inputsHFSubsetIdx = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LFIdx_trunc.txt", Int64)[:]
            else
                xi_hf = readdlm(joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Final.txt", batchID)))
                xi_lf = readdlm(joinpath(inputs_dir, @sprintf("LF_Batch_%03d_Final.txt", batchID)))
                # here orig == scaled inputs.
                xi_hf_scaled = 2 * (xi_hf .- (1/2)*(lb + ub)') ./ (ub - lb)'
                xi_lf_scaled = 2 * (xi_lf .- (1/2)*(lb + ub)') ./ (ub - lb)'
                inputsHFSubsetIdx = readdlm(joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Subset_Final.txt", batchID)), Int64)[:]
            end
            nLF = Int(size(xi_lf, 1) / 2)
            nHF = Int(size(xi_hf, 1) / 2)
            if batchID == 0
                HF_data = [pilot_HF_data pilot_HF_data]
                LF_data = [pilot_LF_data pilot_LF_data]
            else
                LF_data = zeros(nxLF * nyLF, 2 * nLF)
                HF_data = zeros(nxLF * nyLF, 2 * nHF)
                LF_data[:, 1:N_PILOT_LF] = pilot_LF_data
                HF_data[:, 1:N_PILOT_HF] = pilot_HF_data
                LF_data[:, (nLF + 1):(nLF + N_PILOT_LF)] = pilot_LF_data
                HF_data[:, (nHF + 1):(nHF + N_PILOT_HF)] = pilot_HF_data

                LF_data[:, (N_PILOT_LF + 1):nLF] = y_lf_gp_all[:, 1:(batchID)]
                LF_data[:, (nLF + N_PILOT_LF + 1):end] = y_lf_ra_all[:, 1:(batchID)]

                HF_data[:, (N_PILOT_HF + 1):nHF] = y_hf_gp_all[:, 1:(batchID)]
                HF_data[:, (nHF + N_PILOT_HF + 1):end] = y_hf_ra_all[:, 1:(batchID)]
            end

            if batchID > 0
                delta_idx = extend_vector(inputsHFSubsetIdx[1:N_PILOT_HF], inputsHFSubsetIdx[(N_PILOT_HF + 1):nHF])
                Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, delta_idx]]
            else
                Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]
            end


            QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = nothing, nothing, nothing, nothing, nothing
            QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = nothing, nothing, nothing, nothing, nothing
            QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = nothing, nothing, nothing, nothing, nothing
            QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = nothing, nothing, nothing, nothing, nothing

            if batchID == 0
                for case in kle_cases
                    println("Rebuilding surrogate for case $case")
                    if case == "gp"
                        QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = buildKLE(xi_lf[1:nLF, :], LF_data[:, 1:nLF], xmLF; kle_kwargs...)

                        QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = buildKLE(xi_hf[1:nHF, :], Y_Delta[:, 1:nHF], xmLF; kle_kwargs_Δ...)
                    elseif case == "ra"
                        QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = buildKLE(xi_lf[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], xmLF; kle_kwargs...)

                        QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = buildKLE(xi_hf[(nHF + 1):end, :], Y_Delta[:, (nHF + 1):end], xmLF; kle_kwargs_Δ...)
                    end
                end
            else
                for case in kle_cases
                    println("Rebuilding surrogate for case $case")
                    if case == "gp"
                        QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = buildKLE(xi_lf_scaled[1:nLF, :], LF_data[:, 1:nLF], xmLF; kle_kwargs...)

                        QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = buildKLE(xi_hf_scaled[1:nHF, :], Y_Delta[:, 1:nHF], xmLF; kle_kwargs_Δ...)
                    elseif case == "ra"
                        QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = buildKLE(xi_lf_scaled[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], xmLF; kle_kwargs...)

                        QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = buildKLE(xi_hf_scaled[(nHF + 1):end, :], Y_Delta[:, (nHF + 1):end], xmLF; kle_kwargs_Δ...)
                    end
                end
            end

            inputs_hf_gp_pred = 2 * (xi_HF_gp_pred .- (1/2)*(lb + ub)') ./ (ub - lb)'
            inputs_hf_ra_pred = 2 * (xi_HF_ra_pred .- (1/2)*(lb + ub)') ./ (ub - lb)'
            inputs_pred_all_gp = 2 * (xi_pred_all_gp .- (1/2)*(lb + ub)') ./ (ub - lb)'
            inputs_pred_all_ra = 2 * (xi_pred_all_ra .- (1/2)*(lb + ub)') ./ (ub - lb)'

            gp_gp_pred = predictOnGrid(QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp, QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp, inputs_hf_gp_pred, XMLF[:])
            gp_ra_pred = predictOnGrid(QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp, QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp, inputs_pred_all_ra, XMLF[:])
        
            ra_ra_pred = predictOnGrid(QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra, QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra, inputs_hf_ra_pred, XMLF[:])
            ra_gp_pred = predictOnGrid(QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra, QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra, inputs_pred_all_gp, XMLF[:])
            
            gp_on_gp[1:(N_PILOT_HF + batchID), batchID + 1, j] = cv_gp
            gp_on_gp[(N_PILOT_HF + batchID + 1):end, batchID + 1, j] = [ϵ1(y_hf_gp_pred[:, i], gp_gp_pred[:, i]) for i in 1:size(xi_HF_gp_pred, 1)]
            gp_on_ra[:, batchID + 1, j] = [ϵ1(y_hf_ra_all[:, i], gp_ra_pred[:, i]) for i in 1:size(xi_pred_all_ra, 1)]

            ra_on_ra[1:(N_PILOT_HF + batchID), batchID + 1, j] = cv_ra
            ra_on_ra[(N_PILOT_HF + batchID + 1):end, batchID + 1, j] = [ϵ1(y_hf_ra_pred[:, i], ra_ra_pred[:, i]) for i in 1:size(xi_HF_ra_pred, 1)]
            ra_on_gp[:, batchID + 1, j] = [ϵ1(y_hf_gp_all[:, i], ra_gp_pred[:, i]) for i in 1:size(xi_pred_all_gp, 1)]
        end
    end


    if kle_kwargs_Δ.getAllModes == 1
        # save results for all reps using Serialization 
        open(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_all_modes_nolog.jls", acqFunc)), "w") do io
            serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
        end
    elseif kle_kwargs_Δ.getAllModes == 0
        # save results for all reps using Serialization 
        open(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_nolog.jls", acqFunc)), "w") do io
            serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
        end
    end
elseif computeErrors == 0
    if kle_kwargs_Δ.getAllModes == 1
        all_error_data = deserialize(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_all_modes_nolog.jls", acqFunc)))
        gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp = all_error_data[1], all_error_data[2], all_error_data[3], all_error_data[4]

        npzwrite(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_all_modes_nolog.npz", acqFunc)),
        Dict("gp_on_gp"=>gp_on_gp,
        "gp_on_ra"=>gp_on_ra,
        "ra_on_ra"=>ra_on_ra, 
        "ra_on_gp"=>ra_on_gp))
    elseif kle_kwargs_Δ.getAllModes == 0
        all_error_data = deserialize(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_nolog.jls", acqFunc)))
        gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp = all_error_data[1], all_error_data[2], all_error_data[3], all_error_data[4]

        npzwrite(joinpath("./2d_pde/", @sprintf("err_heatmaps/err_heatmap_all_reps_2d_%s_nolog.npz", acqFunc)),
        Dict("gp_on_gp"=>gp_on_gp,
        "gp_on_ra"=>gp_on_ra,
        "ra_on_ra"=>ra_on_ra, 
        "ra_on_gp"=>ra_on_gp))
    end
end

# %%
# if kle_kwargs_Δ.getAllModes == 1
#     # save results for all reps using Serialization 
#     open(joinpath("./1d_toy/", @sprintf("err_heatmap_all_reps_case_%03d_all_modes.jls", chosen_case)), "w") do io
#         serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
#     end
# elseif kle_kwargs_Δ.getAllModes == 0
#     # save results for all reps using Serialization 
#     open(joinpath("./1d_toy/", @sprintf("err_heatmap_all_reps_case_%03d.jls", chosen_case)), "w") do io
#         serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
#     end
# end

# gpgp2 = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS)
# gpra2 = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS)
# rara2 = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS)
# ragp2 = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS)


# for (j, repID) in enumerate(reps_to_use[1:15])
#     gpgp2[:, :, j] = gp_on_gp[:, :, repID]
#     gpra2[:, :, j] = gp_on_ra[:, :, repID]
#     rara2[:, :, j] = ra_on_ra[:, :, repID]
#     ragp2[:, :, j] = ra_on_gp[:, :, repID]
# end



