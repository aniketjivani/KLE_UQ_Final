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

nxLF = 32
nyLF = 32
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
HF_oracle_data = np.load("./2d_pde/HF_oracle_flattened_interp.npy", allow_pickle=true)
HF_oracle_design = np.loadtxt("./2d_pde/input_list_oracle_HF_scaled.txt")

output_dir = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/2d_pde/2d_output_dir/"

function to_dict(obj::KLEObject)
    return Dict(
        "YmLF" => obj.YmLF,
        "QLF" => obj.QLF,
        "λLF" => obj.λLF,
        "bβLF" => obj.bβLF,
        "regLF" => obj.regLF,
        "YmDelta" => obj.YmDelta,
        "QDelta" => obj.QDelta,
        "λDelta" => obj.λDelta,
        "bβDelta" => obj.bβDelta,
        "regDelta" => obj.regDelta
    )
end

counts_gp_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"], args_dict["NFOLDS"])
counts_ra_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"], args_dict["NFOLDS"])
counts_gp_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"], args_dict["NFOLDS"])
counts_ra_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"], args_dict["NFOLDS"])

max_gp_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
max_ra_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
max_gp_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
max_ra_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])

min_gp_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
min_ra_LF = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
min_gp_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])
min_ra_Delta = zeros(args_dict["NREPS"], args_dict["BUDGET_HF"])


reps_to_use = [1, 3, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 30]

function count_modes(lambdas::Vector{Any})
    counts = []
    for lbd in lambdas
        push!(counts, length(lbd))
    end
    return counts
end

for (jrep, repID) in enumerate(reps_to_use)
    println("Processing rep = $jrep")
    for batchID in 0:(args_dict["BUDGET_HF"] - 1)
        kle_obj = deserialize(joinpath(args_dict["data_dir"],
            @sprintf("rep_%03d", repID),
            @sprintf("case_objects_batch_%03d.jls", batchID)))

        kle_gp = to_dict(kle_obj[1])
        kle_ra = to_dict(kle_obj[2])

        count_gp_LF = count_modes(kle_gp["λLF"])
        count_ra_LF = count_modes(kle_ra["λLF"])
        count_gp_Delta = count_modes(kle_gp["λDelta"])
        count_ra_Delta = count_modes(kle_ra["λDelta"])

        counts_gp_LF[jrep, batchID + 1, :] = count_gp_LF
        counts_ra_LF[jrep, batchID + 1, :] = count_ra_LF
        counts_gp_Delta[jrep, batchID + 1, :] = count_gp_Delta
        counts_ra_Delta[jrep, batchID + 1, :] = count_ra_Delta
    end
end

for (jrep, repID) in enumerate(reps_to_use)
    for batchID in 0:args_dict["BUDGET_HF"] - 1
        max_gp_LF[jrep, batchID + 1] = maximum(counts_gp_LF[jrep, batchID + 1, :])
        max_ra_LF[jrep, batchID + 1] = maximum(counts_ra_LF[jrep, batchID + 1, :])
        max_gp_Delta[jrep, batchID + 1] = maximum(counts_gp_Delta[jrep, batchID + 1, :])
        max_ra_Delta[jrep, batchID + 1] = maximum(counts_ra_Delta[jrep, batchID + 1, :])

        min_gp_LF[jrep, batchID + 1] = minimum(counts_gp_LF[jrep, batchID + 1, :])
        min_ra_LF[jrep, batchID + 1] = minimum(counts_ra_LF[jrep, batchID + 1, :])
        min_gp_Delta[jrep, batchID + 1] = minimum(counts_gp_Delta[jrep, batchID + 1, :])
        min_ra_Delta[jrep, batchID + 1] = minimum(counts_ra_Delta[jrep, batchID + 1, :])
    end
end

println("2D PDE - Max LF modes for GP:" , maximum(max_gp_LF), " Min LF modes for GP:" , minimum(min_gp_LF), " Max Delta modes for GP:" , maximum(max_gp_Delta), " Min Delta modes for GP:" , minimum(min_gp_Delta))

println("2D PDE - Max LF modes for RS:" , maximum(max_ra_LF), " Min LF modes for RA:" , minimum(min_ra_LF), " Max Delta modes for RA:" , maximum(max_ra_Delta), " Min Delta modes for RA:" , minimum(min_ra_Delta))