# Build HF-KLE from 105 points (5 pilot, 50 each via AL and RS), and compare oracle performance over grid with individual Bifidelity KLEs.

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


include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/kleUtils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/utils.jl")


generate_KLE_oracle = false
generate_surr_predictions = true
chosen_case = 2  # 1 or 2

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

kle_kwargs_HF = (order=3,
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
            "NREPS"=> 5,
            "NFOLDS"=> 5,
            # "BUDGET_HF"=>20,
            "BUDGET_HF"=>50,
            "lb" => [40, 30],
            "ub" => [60, 50],
            # "acqFunc" => "EI",
            "chosen_lf" => "bSine",
            "acqFunc" => "EI",
       )

dict_case2 = Dict("plot_dir"=> "./Plots/1d_toy_plots_long_c2",
            "data_dir"=> "./1d_toy/1d_pred_data_c2",
            "input_dir"=> "./1d_toy/1d_inputs_c2",
            "NREPS"=> 5,
            "NFOLDS"=> 5,
            "BUDGET_HF"=>50,
            "lb" => [40, 60],
            "ub" => [60, 80],
            "chosen_lf" => "taylor",
            "acqFunc" => "EI",
       )

@assert dict_case1["acqFunc"] == dict_case2["acqFunc"]

if kle_kwargs_Δ.getAllModes == 1
    dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_all_modes", dict_case2["acqFunc"])
    dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_all_modes", dict_case2["acqFunc"])
    dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_all_modes", dict_case2["acqFunc"])
elseif kle_kwargs_Δ.getAllModes == 0
    dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s", dict_case1["acqFunc"])
    dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s", dict_case1["acqFunc"])
    dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s", dict_case2["acqFunc"])
    dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s", dict_case2["acqFunc"])
    dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s", dict_case1["acqFunc"])
    dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s", dict_case2["acqFunc"])
end

if chosen_case == 1
    args_dict = dict_case1
elseif chosen_case == 2
    args_dict = dict_case2
end

sys = pyimport("sys")
pkl = pyimport("pickle")
np = pyimport("numpy")

kle_cases = ["gp", "ra"]

lb = args_dict["lb"]
ub = args_dict["ub"]
acqFunc = args_dict["acqFunc"]
BUDGET_HF = args_dict["BUDGET_HF"]

## Define grid 
x = collect(range(0, 0.1; length=250))
ng = length(x)
data_dir = args_dict["data_dir"]
input_dir = args_dict["input_dir"]

a_grid = collect(range(lb[1], ub[1]; length=200))
b_grid = collect(range(lb[2], ub[2]; length=200))

# scale a_grid and b_grid to lie between -1 and +1.
grid_a_scaled = 2 * (a_grid .- (1/2)*(lb[1] + ub[1])) ./ (ub[1] - lb[1])
grid_b_scaled = 2 * (b_grid .- (1/2)*(lb[2] + ub[2])) ./ (ub[2] - lb[2])

if generate_KLE_oracle
    for repID in 1:args_dict["NREPS"]
    # for repID in 1:1
        println("Starting repetition $repID ...")
        rd_seed = 20250531 + repID
        Random.seed!(rd_seed)

        mkpath(joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID)))
        mkpath(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID)))

        fileLF = joinpath(args_dict["input_dir"], 
                @sprintf("rep_%03d", repID), 
                @sprintf("LF_Batch_%03d_Final", BUDGET_HF) * ".txt"
                )
        fileHF = joinpath(args_dict["input_dir"], 
                @sprintf("rep_%03d", repID),
                @sprintf("HF_Batch_%03d_Final", BUDGET_HF) * ".txt"
                )
        fileHFIdx = joinpath(args_dict["input_dir"], 
                @sprintf("rep_%03d", repID),
                @sprintf("HF_Batch_%03d_Subset_Final", BUDGET_HF) * ".txt"
                )

        inputsLF = readdlm(fileLF)
        inputsHF = readdlm(fileHF)
        inputsHFSubsetIdx = readdlm(fileHFIdx, Int64)[:]

        nLF = Int(size(inputsLF, 1) / 2)
        nHF = Int(size(inputsHF, 1) / 2)

        nPilotLF = 200
        nPilotHF = 5

        inputsLF_orig = 2 * (inputsLF .- (1/2)*(lb + ub)') ./ (ub - lb)'
        inputsHF_orig = 2 * (inputsHF .- (1/2)*(lb + ub)') ./ (ub - lb)'
        LF_data = generateLF(x, inputsLF; chosen_lf=args_dict["chosen_lf"])
        HF_data = generateHF(x, inputsHF)

        Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]

        kle_gp_oracle = nothing
        kle_ra_oracle = nothing

        unique_HF_inputs = [inputsHF_orig[1:nHF, :]; inputsHF_orig[(nHF + nPilotHF + 1):end, :]]

        unique_HF_data = [HF_data[:, 1:nHF] HF_data[:, (nHF + nPilotHF + 1):end]]

        QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = buildKLE(unique_HF_inputs, unique_HF_data, x; kle_kwargs_HF...)

        kle_oracle_HF = KLEObjectSingle(YMeanHF_oracle, QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle)

        open(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "kle_object_hf_cumulative.jls"), "w") do io
            serialize(io, kle_oracle_HF)
        end

    end
else
    println("KLE object already generated. Sample Filepaths: ", joinpath(args_dict["data_dir"], @sprintf("rep_%03d", 1), "kle_object_hf_cumulative.jls"))
end


if generate_surr_predictions
    for repID in 1:args_dict["NREPS"]
        HF_cumulative_oracle = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)

        kle_oracle_HF = deserialize(open(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "kle_object_hf_cumulative.jls")))

        QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = kle_oracle_HF.Q, kle_oracle_HF.λ, kle_oracle_HF.bβ, kle_oracle_HF.reg, kle_oracle_HF.Ym


        for i in 1:length(grid_a_scaled)
            for j in 1:length(grid_b_scaled)
                ΨTest_HF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'

                klModes_HF_oracle = QHF_oracle .* sqrt.(λHF_oracle)'

                HF_cumulative_oracle[i, j, :] = (klModes_HF_oracle * bβHF_oracle * ΨTest_HF_oracle) + YMeanHF_oracle
            end
        end

        np.savez_compressed(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "hf_surr_oracle_predictions_cumulative.npz"), HF_cumulative_oracle=HF_cumulative_oracle)

        println("Completed surrogate predictions for repetition $repID ...")
    end
else
    println("Surrogate predictions already generated. Sample Filepaths: ", joinpath(args_dict["data_dir"], @sprintf("rep_%03d", 1), "hf_surr_oracle_predictions_cumulative.npz"))
end