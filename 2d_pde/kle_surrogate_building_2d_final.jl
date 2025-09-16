# %%
## Load packages and modules, define args.

# Evaluate performance under greater number of k folds and modified acquisition function. (optionally) use FixedNoiseGaussianLikelihood based on noise level from repeated k-fold CV in a single replication.

using LatinHypercubeSampling
using Random
using LinearAlgebra
using SpecialFunctions
using Distributions
using ArgParse
using CSV
using DataFrames
using NPZ
using PyCall
using DelimitedFiles
using Term.Progress
using Printf
using JLD
using Combinatorics
using Serialization
using Plots
using Pickle

np = pyimport("numpy")
sp = pyimport("scipy")
sys = pyimport("sys")
pkl = pyimport("pickle")

include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/utils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/kleUtils.jl")

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
            "NREPS"=> 5,
            "NFOLDS"=> 10,
            # "N_PILOT_LF"=>500,
            # "N_PILOT_HF"=>25,
            "BUDGET_HF"=>40,
            # "BUDGET_HF"=>75,
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
    # args_dict["data_dir"] = @sprintf("./2d_pde/2d_pred_data_%s", args_dict["acqFunc"])
    # args_dict["input_dir"] = @sprintf("./2d_pde/2d_inputs_%s", args_dict["acqFunc"])
    # args_dict["plot_dir"] = @sprintf("./Plots/2d_pde_plots_%s", args_dict["acqFunc"])

    # args_dict["data_dir"] = @sprintf("./2d_pde/2d_pred_data_%s", args_dict["acqFunc"])
    # args_dict["input_dir"] = @sprintf("./2d_pde/2d_inputs_%s", args_dict["acqFunc"])
    # args_dict["plot_dir"] = @sprintf("./Plots/2d_pde_plots_%s", args_dict["acqFunc"])

    args_dict["data_dir"] = @sprintf("./2d_pde/2d_pred_data_trunc_%s", args_dict["acqFunc"])
    args_dict["input_dir"] = @sprintf("./2d_pde/2d_inputs_trunc_%s", args_dict["acqFunc"])
    args_dict["plot_dir"] = @sprintf("./Plots/2d_pde_plots_trunc_%s", args_dict["acqFunc"])
end

nxLF = 32
nyLF = 32

# nxHF = 64
# nyHF = 64

nxHF = 128
nyHF = 128


# %%
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

gridQuantitiesLF = py"getGridQuantities"(nxLF, nyLF)
gridQuantitiesHF = py"getGridQuantities"(nxHF, nyHF)

xLF, yLF, xmLF, ymLF, XMLF, YMLF, dxLF, dyLF, dxiLF, dyiLF, dxi2LF, dyi2LF = gridQuantitiesLF
xHF, yHF, xmHF, ymHF, XMHF, YMHF, dxHF, dyHF, dxiHF, dyiHF, dxi2HF, dyi2HF = gridQuantitiesHF

u_vel_LF, v_vel_LF = py"getVelocitiesGeneric"(gridQuantitiesLF)
u_vel_HF, v_vel_HF = py"getVelocitiesGeneric"(gridQuantitiesHF)

kle_cases = ["gp", "ra"]
lb = args_dict["lb"]
ub = args_dict["ub"]
acqFunc = args_dict["acqFunc"]
n_pilot_lf = args_dict["N_PILOT_LF"]
n_pilot_hf = args_dict["N_PILOT_HF"]

oracle_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_oracle_HF_data"
# _, HF_oracle = readHFFromDir(oracle_dir, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)
# npzwrite("./2d_pde/HF_oracle_flattened_interp.npy", HF_oracle)
HF_oracle_data = np.load("./2d_pde/HF_oracle_flattened_interp.npy", allow_pickle=true)
HF_oracle_design = np.loadtxt("./2d_pde/input_list_oracle_HF_scaled.txt")

# first read and interpolate pilot data, also save it to array directly.
# pilot_dir_HF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirHF/pilotHF"
# pilot_dir_LF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirLF/pilotLF"

output_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_output_dir/"

pilot_dir_HF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirHF/pilotHF_trunc"
pilot_dir_LF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/dirLF/pilotLF_trunc"

# if !(isfile(joinpath(args_dict["root_dir"],"LF_pilot_data.jls")))
_, pilot_HF_data = readHFFromDir(pilot_dir_HF, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)

pilot_LF_data = readLFFromDir(pilot_dir_LF, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)

open(joinpath(args_dict["root_dir"], "HF_pilot_data.jls"), "w") do io
    serialize(io, pilot_HF_data)
end

open(joinpath(args_dict["root_dir"], "LF_pilot_data.jls"), "w") do io
    serialize(io, pilot_LF_data)
end
# else
#     pilot_HF_data = deserialize(joinpath(args_dict["root_dir"], "HF_pilot_data.jls"))
#     pilot_LF_data = deserialize(joinpath(args_dict["root_dir"], "LF_pilot_data.jls"))
# end

# for repID in 1:args_dict["NREPS"]
for repID in 3:4
    println("Starting repetition $repID")
    # Specify a new random seed for each repetition
    rd_seed = 20250531 + repID
    # rd_seed = 20241201 + repID
    Random.seed!(rd_seed)

    # make input directory for this replication
    # we will save separate input files and data files for the randomly acq and the active learnt points but within the same rep dir.
    mkpath(joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(args_dict["plot_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(output_dir, @sprintf("rep_%03d", repID)))

    batchID = 0
    for batchID in 0:(args_dict["BUDGET_HF"] - 1)
    # for batchID in 15:(args_dict["BUDGET_HF"] - 1)
    # for batchID in 0:15
        if batchID == 0
            isPilot = true
            # fileLF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LF_Pilot_scaled.txt"
            # fileHF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_HF_Pilot_scaled.txt"
            # fileHFIdx = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LFIdx.txt"    
            fileLF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LF_Pilot_scaled_trunc.txt"
            fileHF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_HF_Pilot_scaled_trunc.txt"
            fileHFIdx = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/input_list_LFIdx_trunc.txt"    
        else
            isPilot = false
            fileLF = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", repID), 
            @sprintf("LF_Batch_%03d_Final", batchID) * ".txt"
            )
            fileHF = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", repID),
            @sprintf("HF_Batch_%03d_Final", batchID) * ".txt"
            )
            fileHFIdx = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", repID),
            @sprintf("HF_Batch_%03d_Subset_Final", batchID) * ".txt"
            )
        end
        inputsLF = readdlm(fileLF)
        inputsHF = readdlm(fileHF)
        inputsHFSubsetIdx = readdlm(fileHFIdx, Int64)[:]
        # @assert size(inputsLF, 1) - nPilotLF == size(inputsHF, 1) - nPilotHF
        nLF = Int(size(inputsLF, 1) / 2)
        nHF = Int(size(inputsHF, 1) / 2)
        nPilotLF = deepcopy(n_pilot_lf)
        nPilotHF = deepcopy(n_pilot_hf)
        batch_num = Int(size(inputsLF, 1) / 2) - nPilotLF

        @assert batchID == batch_num
        @assert nPilotLF + batch_num == nLF
        @assert nPilotHF + batch_num == nHF
        println("Batch number: $batch_num", " LF Points:  $nLF", " HF Points: $nHF")

        # Rescale acquired points which are between (-1, 1) to original domain.
        LF_data, HF_data = nothing, nothing
        inputsLF_orig, inputsHF_orig = nothing, nothing
        if batchID == 0
            inputsLF_orig = 0.5 * inputsLF .* (ub - lb)' .+ 0.5 * (ub + lb)'
            inputsHF_orig = 0.5 * inputsHF .* (ub - lb)' .+ 0.5 * (ub + lb)'
            # input data is split into two parts: first half is GP points, second half is points through random acquisition. (The first k points in each are identical). We run the surrogate building loop twice, once for each half.

            open(joinpath(output_dir, @sprintf("rep_%03d", repID), @sprintf("output_data_batch_%03d", batchID) * ".jls"), "w") do io
                serialize(io, Dict("lf_gp_flat"=>pilot_LF_data, "hf_gp_flat"=>pilot_HF_data, "lf_ra_flat"=>pilot_LF_data, "hf_ra_flat"=>pilot_HF_data))
            end

            LF_data = [pilot_LF_data pilot_LF_data]
            HF_data = [pilot_HF_data pilot_HF_data]
        else
            inputsLF_orig = 2 * (inputsLF .- (1/2)*(lb + ub)') ./ (ub - lb)'
            inputsHF_orig = 2 * (inputsHF .- (1/2)*(lb + ub)') ./ (ub - lb)'


            # generate data for only the latest run and save to file. Load previous data from file and concatenate both with pilot data.

            new_gp = inputsLF[nLF, :]
            new_ra = inputsLF[end, :]


            phi_hf_gp = py"plotPhiForThetaGeneric"(gridQuantitiesHF,
            u_vel_HF,
            v_vel_HF,
            theta_s = new_gp[1],
            theta_h = new_gp[2],
            theta_x = new_gp[3],
            theta_y = new_gp[4],
            alpha = 1e-2,
            )

            phi_lf_gp = py"plotPhiForThetaGeneric"(gridQuantitiesLF,
            u_vel_LF,
            v_vel_LF,
            theta_s = new_gp[1],
            theta_h = new_gp[2],
            theta_x = new_gp[3],
            theta_y = new_gp[4],
            alpha = 1e-2,
            )

            phi_hf_ra = py"plotPhiForThetaGeneric"(gridQuantitiesHF,
            u_vel_HF,
            v_vel_HF,
            theta_s = new_ra[1],
            theta_h = new_ra[2],
            theta_x = new_ra[3],
            theta_y = new_ra[4],
            alpha = 1e-2,
            )

            phi_lf_ra = py"plotPhiForThetaGeneric"(gridQuantitiesLF,
            u_vel_LF,
            v_vel_LF,
            theta_s = new_ra[1],
            theta_h = new_ra[2],
            theta_x = new_ra[3],
            theta_y = new_ra[4],
            alpha = 1e-2,
            )

            lf_gp_flat = phi_lf_gp[:]
            lf_ra_flat = phi_lf_ra[:]

            _, hf_gp_flat = readAndInterpolateHFFromArray(phi_hf_gp, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)
            _, hf_ra_flat = readAndInterpolateHFFromArray(phi_hf_ra, gridQuantitiesHF, gridQuantitiesLF; nxHF = nxHF, nyHF = nyHF, nxLF = nxLF, nyLF = nyLF)

            # save to file.
            open(joinpath(output_dir, @sprintf("rep_%03d", repID), @sprintf("output_data_batch_%03d", batchID) * ".jls"), "w") do io
                serialize(io, Dict("lf_gp_flat"=>lf_gp_flat, "hf_gp_flat"=>hf_gp_flat, "lf_ra_flat"=>lf_ra_flat, "hf_ra_flat"=>hf_ra_flat))
            end


            # also load previously generated HF data on the fly from file. (for all previous batchID)

            LF_data = zeros(nxLF * nyLF, 2 * nLF)
            HF_data = zeros(nxLF * nyLF, 2 * nHF)

            LF_data[:, 1:nPilotLF] = pilot_LF_data
            HF_data[:, 1:nPilotHF] = pilot_HF_data

            LF_data[:, (nLF + 1):(nLF + nPilotLF)] = pilot_LF_data
            HF_data[:, (nHF + 1):(nHF + nPilotHF)] = pilot_HF_data


            if batchID > 1
                for bID in 1:(batchID - 1)
                    prev_data = deserialize(joinpath(output_dir, @sprintf("rep_%03d", repID), @sprintf("output_data_batch_%03d", bID) * ".jls"))

                    lf_gp = prev_data["lf_gp_flat"]
                    hf_gp = prev_data["hf_gp_flat"]
                    lf_ra = prev_data["lf_ra_flat"]
                    hf_ra = prev_data["hf_ra_flat"]

                    LF_data[:, (nPilotLF + bID)] = lf_gp
                    LF_data[:, (nLF + nPilotLF + bID)] = lf_ra

                    HF_data[:, (nPilotHF + bID)] = hf_gp
                    HF_data[:, (nHF + nPilotHF + bID)] = hf_ra
                end
                LF_data[:, nLF] = lf_gp_flat
                LF_data[:, end] = lf_ra_flat
                HF_data[:, nHF] = hf_gp_flat
                HF_data[:, end] = hf_ra_flat
            else
                LF_data[:, (nPilotLF + 1):nLF] = lf_gp_flat
                LF_data[:, (nLF + nPilotLF + 1):end] = lf_ra_flat

                HF_data[:, (nPilotHF + 1):nHF] = hf_gp_flat
                HF_data[:, (nHF + nPilotHF + 1):end] = hf_ra_flat
            end
        end

        # Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]
        delta_idx = nothing
        Y_Delta = nothing
        if batchID > 0
            delta_idx = extend_vector(inputsHFSubsetIdx[1:n_pilot_hf], inputsHFSubsetIdx[(n_pilot_hf + 1):nHF])

            Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, delta_idx]]
        else
            Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]
        end
        # Generate k-fold indices (for half the dataset, repeat)

        # k_folds_batch = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed))

        k_folds_batch_1 = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed))
        k_folds_batch_2 = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed + 200))

        case_objects = []
        cv_gp, oracle_err_gp, oracle_err_hf_gp, kle_gp, kle_oracle_gp = nothing, nothing, nothing, nothing, nothing
        cv_ra, oracle_err_ra, oracle_err_hf_ra, kle_ra, kle_oracle_ra = nothing, nothing, nothing, nothing, nothing
        
        if batchID == 0
            for case in kle_cases
                println("Rebuilding surrogate for case $case")
                if case == "gp"
                    cv_gp, oracle_err_gp, oracle_err_hf_gp, kle_gp, kle_oracle_gp = evaluateKLE(inputsLF[1:nLF, :], LF_data[:, 1:nLF], inputsHFSubsetIdx[1:nHF], inputsHF[1:nHF, :], HF_data[:, 1:nHF], Y_Delta[:, 1:nHF], xmLF; 
                    useAbsErr=0, 
                    all_folds=k_folds_batch_1, 
                    HF_oracle_data=HF_oracle_data, 
                    HF_oracle_design=HF_oracle_design)
                elseif case == "ra"
                    cv_ra, oracle_err_ra, oracle_err_hf_ra, kle_ra, kle_oracle_ra = evaluateKLE(inputsLF[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end], inputsHF[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], xmLF; 
                    useAbsErr=0,
                    all_folds=k_folds_batch_2, 
                    HF_oracle_data=HF_oracle_data, 
                    HF_oracle_design=HF_oracle_design)
                end
            end
        else
            for case in kle_cases
                println("Rebuilding surrogate for case $case")
                if case == "gp"
                    cv_gp, oracle_err_gp, oracle_err_hf_gp, kle_gp, kle_oracle_gp = evaluateKLE(inputsLF_orig[1:nLF, :], LF_data[:, 1:nLF], inputsHFSubsetIdx[1:nHF], inputsHF_orig[1:nHF, :], HF_data[:, 1:nHF], Y_Delta[:, 1:nHF], xmLF; 
                    useAbsErr=0, 
                    all_folds=k_folds_batch_1, 
                    HF_oracle_data=HF_oracle_data, 
                    HF_oracle_design=HF_oracle_design)
                elseif case == "ra"
                    cv_ra, oracle_err_ra, oracle_err_hf_ra, kle_ra, kle_oracle_ra = evaluateKLE(inputsLF_orig[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end], inputsHF_orig[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], xmLF; 
                    useAbsErr=0,
                    all_folds=k_folds_batch_2, 
                    HF_oracle_data=HF_oracle_data, 
                    HF_oracle_design=HF_oracle_design)
                end
            end
        end

        npzwrite(joinpath(args_dict["data_dir"],
        @sprintf("rep_%03d", repID), 
        @sprintf("case_objects_batch_%03d.npz", batchID)), Dict("cv_gp"=> cv_gp,
            "oracle_err_gp"=> oracle_err_gp,
            "oracle_err_hf_gp"=> oracle_err_hf_gp,
            "cv_ra"=> cv_ra,
            "oracle_err_ra"=> oracle_err_ra,
            "oracle_err_hf_ra"=> oracle_err_hf_ra,
            ))

        open(joinpath(args_dict["data_dir"],
            @sprintf("rep_%03d", repID),
            @sprintf("case_objects_batch_%03d.jls", batchID)), "w") do io
        serialize(io, (kle_gp, kle_ra, kle_oracle_gp, kle_oracle_ra))
        end


        if batchID == 0
            sys.argv = ["./2d_pde/botorch_optim_point_selection_2d_final.py",
                rd_seed,
                lb[1],
                lb[2],
                lb[3],
                lb[4],
                ub[1],
                ub[2],
                ub[3],
                ub[4],
                args_dict["plot_dir"],
                args_dict["data_dir"],
                args_dict["input_dir"],
                repID,
                batch_num,
                # log.(cv_gp),
                cv_gp,
                inputsHF_orig,
                inputsLF_orig,
                inputsHFSubsetIdx,
                acqFunc
                ]


            # save arguments as dict to pickle
            # Pickle.store("/Users/ajivani/Downloads/args_bo.pkl", Dict("filename"=>"./2d_pde/botorch_optim_point_selection_2d_final.py",
            #     "rd_seed"=>rd_seed,
            #     "lb1"=>lb[1],
            #     "lb2"=>lb[2],
            #     "lb3"=>lb[3],
            #     "lb4"=>lb[4],
            #     "ub1"=>ub[1],
            #     "ub2"=>ub[2],
            #     "ub3"=>ub[3],
            #     "ub4"=>ub[4],
            #     "plot_dir"=>args_dict["plot_dir"],
            #     "data_dir"=>args_dict["data_dir"],
            #     "input_dir"=>args_dict["input_dir"],
            #     "repID"=>repID,
            #     "batch_num"=>batch_num,
            #     # "log_cv_gp"=>log.(cv_gp),
            #     "cv_gp"=>cv_gp,
            #     "inputsHF_orig"=>inputsHF_orig[:],
            #     "inputsLF_orig"=>inputsLF_orig[:],
            #     "inputsHFSubsetIdx"=>inputsHFSubsetIdx,
            #     "acqFunc"=>acqFunc
            #     ))
        else
            sys.argv = ["./2d_pde/botorch_optim_point_selection_2d_final.py",
            rd_seed,
            lb[1],
            lb[2],
            lb[3],
            lb[4],
            ub[1],
            ub[2],
            ub[3],
            ub[4],
            args_dict["plot_dir"],
            args_dict["data_dir"],
            args_dict["input_dir"],
            repID,
            batch_num,
            # log.(cv_gp),
            cv_gp,
            inputsHF,
            inputsLF,
            inputsHFSubsetIdx,
            acqFunc
            ]
        end

        @pyinclude "./2d_pde/botorch_optim_point_selection_2d_final.py"

        gp = py"gp"
        # utilities = py"utilities"
        # mean_gp = py"mean_gp"
        # var_gp = py"variance_gp"

        open(joinpath(args_dict["data_dir"], 
            # args_dict["rep_dir"],
            @sprintf("rep_%03d", repID), 
            @sprintf("gp_batch_%02d.pkl", batchID)), "w") do f
            pkl.dump(Dict("gp"=>gp,
                        # "utilities"=>utilities,
                        # "mean_gp"=>mean_gp,
                        # "var_gp"=>var_gp
                        ), f)
        end

    end
    println("Finished replication $repID")
end

