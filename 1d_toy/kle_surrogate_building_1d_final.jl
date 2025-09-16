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
            "NREPS"=> 5,
            "NFOLDS"=> 5,
            # "BUDGET_HF"=>20,
            "BUDGET_HF"=>60,
            "lb" => [40, 30],
            "ub" => [60, 50],
            "acqFunc" => "EI",
            "chosen_lf" => "bSine",
            # "acqFunc" => "UCB",
            # "acqFunc" => "logEI"
       )

dict_case2 = Dict("plot_dir"=> "./Plots/1d_toy_plots_long_c2",
            "data_dir"=> "./1d_toy/1d_pred_data_c2",
            "input_dir"=> "./1d_toy/1d_inputs_c2",
            "NREPS"=> 5,
            "NFOLDS"=> 5,
            "BUDGET_HF"=>60,
            "lb" => [40, 60],
            "ub" => [60, 80],
            "chosen_lf" => "taylor",
            "acqFunc" => "EI",
       )

# @assert dict_case1["acqFunc"] == dict_case2["acqFunc"]

if kle_kwargs_Δ.getAllModes == 1
    dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_all_modes", dict_case2["acqFunc"])
    dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_all_modes", dict_case2["acqFunc"])
    dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_all_modes", dict_case1["acqFunc"])
    dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_all_modes", dict_case2["acqFunc"])
elseif kle_kwargs_Δ.getAllModes == 0
    dict_case1["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c1_%s_nolog", dict_case1["acqFunc"])
    dict_case1["input_dir"] = @sprintf("./1d_toy/1d_inputs_c1_%s_nolog", dict_case1["acqFunc"])
    dict_case2["data_dir"] = @sprintf("./1d_toy/1d_pred_data_c2_%s_nolog", dict_case2["acqFunc"])
    dict_case2["input_dir"] = @sprintf("./1d_toy/1d_inputs_c2_%s_nolog", dict_case2["acqFunc"])
    dict_case1["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c1_%s_nolog", dict_case1["acqFunc"])
    dict_case2["plot_dir"] = @sprintf("./Plots/1d_toy_plots_long_c2_%s_nolog", dict_case2["acqFunc"])
end


chosen_case = 2
if chosen_case == 1
    args_dict = dict_case1
elseif chosen_case == 2
    args_dict = dict_case2
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

# check if HF oracle file exists, if not, generate data and save to file.

if !isfile(@sprintf("./1d_toy/HF_Oracle_case_%02d.jld", chosen_case))
    HF_oracle = generateOracleData(a_grid, b_grid, x)
    JLD.save(@sprintf("./1d_toy/HF_Oracle_case_%02d.jld", chosen_case), "HF_oracle", HF_oracle)
else
    HF_oracle = JLD.load(@sprintf("./1d_toy/HF_Oracle_case_%02d.jld", chosen_case))["HF_oracle"]
end

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


## Loop for building surrogate
# for repID in 1:args_dict["NREPS"]
for repID in 1:args_dict["NREPS"]
# for repID in 2:2
# for repID in 1:1
    println("Starting repetition $repID")
    # Specify a new random seed for each repetition
    rd_seed = 20250431 + repID
    Random.seed!(rd_seed)

    # make input directory for this replication
    # we will save separate input files and data files for the randomly acq and the active learnt points but within the same rep dir.
    mkpath(joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID)))
    mkpath(joinpath(args_dict["plot_dir"], @sprintf("rep_%03d", repID)))

    batchID = 0
    # for batchID in 0:(args_dict["BUDGET_HF"] - 1)
    # for batchID in 0:25
    for batchID in 0:(args_dict["BUDGET_HF"] - 1)
        if batchID == 0
            isPilot = true
            fileLF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LF_Pilot_scaled.txt"
            fileHF = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_HF_Pilot_scaled.txt"
            fileHFIdx = "/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LFIdx.txt"    
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

        nPilotLF = 200
        nPilotHF = 5

        batch_num = Int(size(inputsLF, 1) / 2) - nPilotLF

        @assert batchID == batch_num

        println("Batch number: $batch_num", " LF Points:  $nLF", " HF Points: $nHF")


        # Rescale acquired points which are between (-1, 1) to original domain.
        LF_data, HF_data = nothing, nothing
        inputsLF_orig, inputsHF_orig = nothing, nothing
        if batchID == 0
            inputsLF_orig = 0.5 * inputsLF .* (ub - lb)' .+ 0.5 * (ub + lb)'
            inputsHF_orig = 0.5 * inputsHF .* (ub - lb)' .+ 0.5 * (ub + lb)'
            # input data is split into two parts: first half is GP points, second half is points through random acquisition. (The first k points in each are identical). We run the surrogate building loop twice, once for each half.

            LF_data = generateLF(x, inputsLF_orig; chosen_lf=args_dict["chosen_lf"])
            HF_data = generateHF(x, inputsHF_orig)
        else
            inputsLF_orig = 2 * (inputsLF .- (1/2)*(lb + ub)') ./ (ub - lb)'
            inputsHF_orig = 2 * (inputsHF .- (1/2)*(lb + ub)') ./ (ub - lb)'
            LF_data = generateLF(x, inputsLF; chosen_lf=args_dict["chosen_lf"])
            HF_data = generateHF(x, inputsHF)
        end

        # Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]

        delta_idx = nothing
        Y_Delta = nothing
        if batchID > 0
            delta_idx = extend_vector(inputsHFSubsetIdx[1:nPilotHF], inputsHFSubsetIdx[(nPilotHF + 1):nHF])

            print(delta_idx)

            Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, delta_idx]]
        else
            Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]
        end

        # Generate k-fold indices (for half the dataset, repeat)

        k_folds_batch_1 = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed))
        k_folds_batch_2 = k_folds(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed + 200))

        # k_folds_batch_1 = k_folds_expanded(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed))
        # k_folds_batch_2 = k_folds_expanded(inputsHFSubsetIdx[1:nHF], args_dict["NFOLDS"]; rng_gen=MersenneTwister(rd_seed + 200))


        case_objects = []
        cv_gp, oracle_gp, kle_gp = nothing, nothing, nothing
        cv_ra, oracle_ra, kle_ra = nothing, nothing, nothing
        
        if batchID == 0
            for case in kle_cases
                println("Rebuilding surrogate for case $case")
                if case == "gp"
                    cv_gp, oracle_gp, kle_gp = evaluateKLE(inputsLF[1:nLF, :], LF_data[:, 1:nLF], inputsHFSubsetIdx[1:nHF], inputsHF[1:nHF, :], HF_data[:, 1:nHF], Y_Delta[:, 1:nHF], x; 
                    useAbsErr=0, 
                    all_folds=k_folds_batch_1, 
                    grid_a_scaled=a_grid_scaled, 
                    grid_b_scaled=b_grid_scaled)
                elseif case == "ra"
                    cv_ra, oracle_ra, kle_ra = evaluateKLE(inputsLF[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end], inputsHF[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], x; 
                    useAbsErr=0,
                    all_folds=k_folds_batch_2, 
                    grid_a_scaled=a_grid_scaled, 
                    grid_b_scaled=b_grid_scaled)
                end
            end
        else
            for case in kle_cases
                println("Rebuilding surrogate for case $case")
                if case == "gp"
                    cv_gp, oracle_gp, kle_gp = evaluateKLE(inputsLF_orig[1:nLF, :], LF_data[:, 1:nLF], inputsHFSubsetIdx[1:nHF], inputsHF_orig[1:nHF, :], HF_data[:, 1:nHF], Y_Delta[:, 1:nHF], x; 
                    useAbsErr=0, 
                    all_folds=k_folds_batch_1, 
                    grid_a_scaled=a_grid_scaled, 
                    grid_b_scaled=b_grid_scaled)
                elseif case == "ra"
                    cv_ra, oracle_ra, kle_ra = evaluateKLE(inputsLF_orig[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end], inputsHF_orig[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], x; 
                    useAbsErr=0,
                    all_folds=k_folds_batch_2, 
                    grid_a_scaled=a_grid_scaled, 
                    grid_b_scaled=b_grid_scaled)
                end
            end
        end

        npzwrite(joinpath(args_dict["data_dir"], 
        # args_dict["rep_dir"],
        @sprintf("rep_%03d", repID), 
        @sprintf("case_objects_batch_%03d.npz", batchID)), 
        Dict("cv_gp"=> cv_gp,
            "oracle_gp"=> oracle_gp,
            # "kle_gp"=> kle_gp,
            "cv_ra"=> cv_ra,
            "oracle_ra"=> oracle_ra,
            # "kle_ra"=> kle_ra
            ))

        # JLD.save(joinpath(args_dict["data_dir"], 
        # # args_dict["rep_dir"], @sprintf("rep_%03d", repID), 
        # @sprintf("case_objects_batch_%03d.jld", batchID)), Dict("kle_gp" => kle_gp, "kle_ra" => kle_ra))

        open(joinpath(args_dict["data_dir"],
            # args_dict["rep_dir"],
            @sprintf("rep_%03d", repID),
              @sprintf("case_objects_batch_%03d.jls", batchID)), "w") do io
        serialize(io, (kle_gp, kle_ra))
        end

        # push!(case_objects, Dict("gp"=>(cv_gp, oracle_gp, kle_gp)))
        # push!(case_objects, Dict("ra"=>(cv_ra, oracle_ra, kle_ra)))

        # # Save the case objects to file
        # JLD.save(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), @sprintf("case_objects_batch_%03d.jld", batchID)), "case_objects", case_objects)


        if batchID == 0
            sys.argv = ["./1d_toy/botorch_optim_point_selection_1d_final.py",
                rd_seed,
                lb[1],
                lb[2],
                ub[1],
                ub[2],
                args_dict["plot_dir"],
                args_dict["data_dir"],
                args_dict["input_dir"],
                repID,
                batch_num,
                # log.(cv_gp[2]),
                # log.(cv_gp),
                cv_gp,
                inputsHF_orig,
                inputsLF_orig,
                inputsHFSubsetIdx,
                acqFunc
                ]
        else
            sys.argv = ["./1d_toy/botorch_optim_point_selection_1d_final.py",
            rd_seed,
            lb[1],
            lb[2],
            ub[1],
            ub[2],
            args_dict["plot_dir"],
            args_dict["data_dir"],
            args_dict["input_dir"],
            repID,
            batch_num,
            # log.(cv_gp[2]),
            # log.(cv_gp),
            cv_gp,
            inputsHF,
            inputsLF,
            inputsHFSubsetIdx,
            acqFunc
            ]
        end
        

            # torch_seed = sys.argv[1]
            # lb = [sys.argv[2], sys.argv[3]]
            # ub = [sys.argv[4], sys.argv[5]]
            # plot_dir = sys.argv[6]
            # data_dir = sys.argv[7]
            # input_dir = sys.argv[8]
            # rep_num = sys.argv[9]
            # batch_num = sys.argv[10]
            # cv_errors = sys.argv[11]
            # hf_inputs = sys.argv[12] 
            # lf_inputs = sys.argv[13]
            # lf_subset = sys.argv[14]

        @pyinclude "./1d_toy/botorch_optim_point_selection_1d_final.py"

        gp = py"gp"
        utilities = py"utilities"
        mean_gp = py"mean_gp"
        var_gp = py"variance_gp"

        open(joinpath(args_dict["data_dir"], 
            # args_dict["rep_dir"],
            @sprintf("rep_%03d", repID), 
            @sprintf("gp_batch_%02d.pkl", batchID)), "w") do f
            pkl.dump(Dict("gp"=>gp,
                        "utilities"=>utilities,
                        "mean_gp"=>mean_gp,
                        "var_gp"=>var_gp), f)
        end

    end
    println("Finished replication $repID")
end







