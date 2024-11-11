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

include("./kleUtils.jl")
include("./utils.jl")


kle_kwargs_Δ = (useFullGrid=1,
            getAllModes=0,
            order=2,
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

args_dict = Dict("plot_dir"=> "./Plots/1d_toy_plots",
            "data_dir"=> "./1d_toy/1d_pred_data",
            "input_dir"=> "./1d_toy/1d_inputs",
            "NREPS"=> 1,
            "NFOLDS"=> 3,
            "BUDGET_HF"=>20,
            "lb" => [40, 30],
            "ub" => [60, 50],
       )

sys = pyimport("sys")
sys.argv = ["./botorch_optim_point_selection_1d_final.py",
            args_dict["plot_dir"],
            args_dict["data_dir"],
            args_dict["input_dir"],
            args_dict["NREPS"],
            args_dict["NFOLDS"],
            args_dict["BUDGET_HF"],
            args_dict["lb"][1],
            args_dict["lb"][2],
            args_dict["ub"][1],
            args_dict["ub"][2]
            ]

kle_cases = ["gp", "ra"]

lb = args_dict["lb"]
ub = args_dict["ub"]

## Define grid 
x = collect(range(0, 0.1; length=250))
ng = length(x)


# Define grid of a and b values to generate oracle data on
a_grid = collect(range(lb[1], ub[1]; length=200))
b_grid = collect(range(lb[2], ub[2]; length=200))

# scale a_grid and b_grid to lie between -1 and +1.
a_grid_scaled = 2 * (a_grid .- (1/2)*(lb[1] + ub[1])) ./ (ub[1] - lb[1])
b_grid_scaled = 2 * (b_grid .- (1/2)*(lb[2] + ub[2])) ./ (ub[2] - lb[2])

# check if HF oracle file exists, if not, generate data and save to file.

if !isfile("./1d_toy/HF_Oracle.jld")
    HF_oracle = generateOracleData(a_grid, b_grid, x)
    JLD.save("./1d_toy/HF_Oracle.jld", "HF_oracle", HF_oracle)
else
    HF_oracle = JLD.load("./1d_toy/HF_Oracle.jld")["HF_oracle"]
end



## Loop for building surrogate


for repID in 1:args_dict["NREPS"]
    println("Starting repetition $repID")
    # Specify a new random seed for each repetition
    rd_seed = 20241031 + repID
    Random.seed!(rd_seed)

    # make input directory for this replication
    # we will save separate input files and data files for the randomly acq and the active learnt points but within the same rep dir.
    mkpath(joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID)))

    for batchID in 0:(args_dict["BUDGET_HF"] - 1)
        if batchID == 0
            isPilot = true
            fileLF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LF_Pilot_scaled.txt"
            fileHF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_HF_Pilot_scaled.txt"
            fileHFIdx = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LFIdx.txt"    
        else
            isPilot = false
            fileLF = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", rep_num), 
            @sprintf("LF_Batch_%03d_Final", CURRENT_BT) * ".txt"
            )
            fileHF = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", rep_num),
            @sprintf("HF_Batch_%03d_Final", CURRENT_BT) * ".txt"
            )
            fileHFIdx = joinpath(args_dict["input_dir"], 
            @sprintf("rep_%03d", rep_num),
            @sprintf("HF_Batch_%03d_Subset_Final", CURRENT_BT) * ".txt"
            )
        end
    

        inputsLF = readdlm(fileLF)
        inputsHF = readdlm(fileHF)
        inputsHFSubsetIdx = readdlm(fileHFIdx, Int64)[:]

        # @assert size(inputsLF, 1) - nPilotLF == size(inputsHF, 1) - nPilotHF

        nLF = Int(size(inputsLF, 1) / 2)
        nHF = Int(size(inputsHF, 1) / 2)

        batch_num = size(inputsLF, 1) - nPilotLF

        @assert CURRENT_BT == batch_num

        println("Batch number: $batch_num", "LF Points: $nLF", "HF Points: $nHF")


        # Rescale acquired points which are between (-1, 1) to original domain.
        inputsLF_orig = 0.5 * inputsLF .* (ub - lb)' .+ 0.5 * (ub + lb)'
        inputsHF_orig = 0.5 * inputsHF .* (ub - lb)' .+ 0.5 * (ub + lb)'


        # input data is split into two parts: first half is GP points, second half is points through random acquisition. (The first k points in each are identical). We run the surrogate building loop twice, once for each half.



        LF_data = generateLF(x, inputsLF_orig, nPilotLF, batch_num)
        HF_data = generateHF(x, inputsHF_orig, nPilotHF, batch_num)

        Y_Delta = HF_data - LF_data[:, inputsHFSubsetIdx]

        # Generate k-fold indices (for half the dataset, repeat)

        k_folds_batch = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed))

        case_objects = Dict()
        for case in kle_cases
            println("Rebuilding surrogate for case $case")
            if case == "gp"

            elseif case == "ra"

            end

        end



        @pyinclude "./botorch_optim_point_selection_1d_final.py"

        acq_point = py"acq_point"

        println("Next training point for KLE:  $acq_point")

    end
    println("Finished replication $repID")
end







