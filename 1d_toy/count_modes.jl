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

include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/kleUtils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/utils.jl")

kle_kwargs_Δ = (useFullGrid=1,
            getAllModes=0,
            order=3,
            dims=2,
            weightFunction=getWeights,
            family="Legendre",
            solver="Tikhonov-L2"
            )

kle_kwargs = (order=3,
            dims=2,
            family="Legendre",
            useFullGrid=1,
            getAllModes=0,
            weightFunction=getWeights,
            solver="Tikhonov-L2"
            )

dict_case1 = Dict("plot_dir"=> "./Plots/1d_toy_plots_long_c1_all_modes",
            "data_dir"=> "./1d_toy/1d_pred_data_c1_all_modes",
            "input_dir"=> "./1d_toy/1d_inputs_c1_all_modes",
            # "NREPS"=> 5,
            "NREPS"=> 20,
            "NFOLDS"=> 5,
            "BUDGET_HF"=>60,
            "lb" => [40, 30],
            "ub" => [60, 50],
            "acqFunc" => "EI",
            "chosen_lf" => "bSine",
            "case" => "max",
       )

dict_case2 = Dict("plot_dir"=> "./Plots/1d_toy_plots_long_c2",
            "data_dir"=> "./1d_toy/1d_pred_data_c2",
            "input_dir"=> "./1d_toy/1d_inputs_c2",
            "NREPS"=> 20,
            # "NREPS"=> 5,
            "NFOLDS"=> 5,
            # "BUDGET_HF"=>60,
            "BUDGET_HF"=>60,
            "lb" => [40, 60],
            "ub" => [60, 80],
            "chosen_lf" => "taylor",
            "acqFunc" => "EI",
            "case" => "max",
       )

chosen_case = 2
if chosen_case == 1
    args_dict = dict_case1
elseif chosen_case == 2
    args_dict = dict_case2
end

if kle_kwargs_Δ.getAllModes == 1
    dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_all_modes_%s", dict_case1["acqFunc"], args_dict["case"])
    dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_all_modes_%s", dict_case1["acqFunc"], args_dict["case"])
    dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_all_modes_%s", dict_case2["acqFunc"],
    args_dict["case"])
    dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_all_modes_%s", dict_case2["acqFunc"],
    args_dict["case"])
    dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_all_modes_%s", dict_case1["acqFunc"], args_dict["case"])
    dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_all_modes_%s", dict_case2["acqFunc"], args_dict["case"])
elseif kle_kwargs_Δ.getAllModes == 0
    if args_dict["case"] == "min"
        dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_nolog_%s", dict_case1["acqFunc"], args_dict["case"])
        dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_nolog_%s", dict_case1["acqFunc"], args_dict["case"])
        dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_nolog_%s", dict_case2["acqFunc"], args_dict["case"])
        dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_nolog_%s", dict_case2["acqFunc"], args_dict["case"])
        dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_nolog_%s", dict_case1["acqFunc"], args_dict["case"])
        dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_nolog_%s", dict_case2["acqFunc"], args_dict["case"])
    else
        dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_nolog", dict_case1["acqFunc"])
        dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_nolog", dict_case1["acqFunc"])
        dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_nolog", dict_case2["acqFunc"])
        dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_nolog", dict_case2["acqFunc"])
        dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_nolog", dict_case1["acqFunc"])
        dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_nolog", dict_case2["acqFunc"])
    end
end

sys = pyimport("sys")
pkl = pyimport("pickle")
kle_cases = ["gp", "ra"]

lb = args_dict["lb"]
ub = args_dict["ub"]
acqFunc = args_dict["acqFunc"]

## Define grid 
x = collect(range(0, 0.1; length=250))
ng = length(x)

# Define grid of a and b values to generate oracle data on
a_grid = collect(range(lb[1], ub[1]; length=200))
b_grid = collect(range(lb[2], ub[2]; length=200))

# scale a_grid and b_grid to lie between -1 and +1.
a_grid_scaled = 2 * (a_grid .- (1/2)*(lb[1] + ub[1])) ./ (ub[1] - lb[1])
b_grid_scaled = 2 * (b_grid .- (1/2)*(lb[2] + ub[2])) ./ (ub[2] - lb[2])

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

function count_modes(lambdas::Vector{Any})
    counts = []
    for lbd in lambdas
        push!(counts, length(lbd))
    end
    return counts
end


for repID in 1:args_dict["NREPS"]
    println("Processing repID = $repID")
    for batchID in 0:args_dict["BUDGET_HF"] - 1
        kle_obj = deserialize(joinpath(args_dict["data_dir"],
        @sprintf("rep_%03d", repID),
        @sprintf("case_objects_batch_%03d.jls", batchID)))
    
        kle_gp = to_dict(kle_obj[1])
        kle_ra = to_dict(kle_obj[2])

        count_gp_LF = count_modes(kle_gp["λLF"])
        count_ra_LF = count_modes(kle_ra["λLF"])
        count_gp_Delta = count_modes(kle_gp["λDelta"])
        count_ra_Delta = count_modes(kle_ra["λDelta"])

        counts_gp_LF[repID, batchID + 1, :] = count_gp_LF
        counts_ra_LF[repID, batchID + 1, :] = count_ra_LF
        counts_gp_Delta[repID, batchID + 1, :] = count_gp_Delta
        counts_ra_Delta[repID, batchID + 1, :] = count_ra_Delta

        
    end
end

for repID in 1:args_dict["NREPS"]
    for batchID in 0:args_dict["BUDGET_HF"] - 1
        max_gp_LF[repID, batchID + 1] = maximum(counts_gp_LF[repID, batchID + 1, :])
        max_ra_LF[repID, batchID + 1] = maximum(counts_ra_LF[repID, batchID + 1, :])
        max_gp_Delta[repID, batchID + 1] = maximum(counts_gp_Delta[repID, batchID + 1, :])
        max_ra_Delta[repID, batchID + 1] = maximum(counts_ra_Delta[repID, batchID + 1, :])

        min_gp_LF[repID, batchID + 1] = minimum(counts_gp_LF[repID, batchID + 1, :])
        min_ra_LF[repID, batchID + 1] = minimum(counts_ra_LF[repID, batchID + 1, :])
        min_gp_Delta[repID, batchID + 1] = minimum(counts_gp_Delta[repID, batchID + 1, :])
        min_ra_Delta[repID, batchID + 1] = minimum(counts_ra_Delta[repID, batchID + 1, :])
    end
end

if chosen_case == 1
    println("bSine LF: Max LF modes for GP:" , maximum(max_gp_LF), " Min LF modes for GP:" , minimum(min_gp_LF), " Max Delta modes for GP:" , maximum(max_gp_Delta), " Min Delta modes for GP:" , minimum(min_gp_Delta))

    println("bSine LF: Max LF modes for RS:" , maximum(max_ra_LF), " Min LF modes for RS:" , minimum(min_ra_LF), " Max Delta modes for RS:" , maximum(max_ra_Delta), " Min Delta modes for RS:" , minimum(min_ra_Delta))
elseif chosen_case == 2
    println("Taylor LF: Max LF modes for GP:" , maximum(max_gp_LF), " Min LF modes for GP:" , minimum(min_gp_LF), " Max Delta modes for GP:" , maximum(max_gp_Delta), " Min Delta modes for GP:" , minimum(min_gp_Delta))

    println("Taylor LF: Max LF modes for RS:" , maximum(max_ra_LF), " Min LF modes for RS:" , minimum(min_ra_LF), " Max Delta modes for RS:" , maximum(max_ra_Delta), " Min Delta modes for RS:" , minimum(min_ra_Delta))
end