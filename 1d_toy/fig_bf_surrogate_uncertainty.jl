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


# # convert deserialized data to numpy arrays
# kle_data_gp, kle_data_ra = deserialize("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/1d_pred_data_c2_EI/rep_001/case_objects_batch_000.jls")

# create KLE objects at final budget for both cases and save them in respective data directories.

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

        for case in kle_cases
            println("Rebuilding surrogate for case: $case ...")
            if case == "gp"
                QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle = buildKLE(inputsLF_orig[1:nLF, :], LF_data[:, 1:nLF], x; kle_kwargs...)
                QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle = buildKLE(inputsHF_orig[1:nHF, :], Y_Delta[:, 1:nHF], x; kle_kwargs_Δ...)
                QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = buildKLE(inputsHF_orig[1:nHF, :], HF_data[:, 1:nHF], x; kle_kwargs_HF...)

                kle_gp_oracle = KLEObjectAll(YMeanLF_oracle, QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanDelta_oracle, QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanHF_oracle, QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle)

            elseif case == "ra"
                QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle = buildKLE(inputsLF_orig[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], x; kle_kwargs...)
                QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle = buildKLE(inputsHF_orig[(nHF + 1):end, :], Y_Delta[:, (nHF + 1):end], x; kle_kwargs_Δ...)
                QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = buildKLE(inputsHF_orig[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], x; kle_kwargs_HF...)

                kle_ra_oracle = KLEObjectAll(YMeanLF_oracle, QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanDelta_oracle, QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanHF_oracle, QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle)
            end

            open(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "kle_object_oracle_final.jls"), "w") do io
                serialize(io, (kle_gp_oracle, kle_ra_oracle))
            end
            println("Finished replication $repID")
        end
    end
else
    println("KLE object already generated. Sample Filepaths: ", joinpath(args_dict["data_dir"], @sprintf("rep_%03d", 1), "kle_object_oracle_final.jls"))
end

# make predictions on 200 x 200 grid for each surrogate and save the results.


if generate_surr_predictions
    for repID in 1:args_dict["NREPS"]
    # for repID in 1:1
        BF_oracle_gp, LF_oracle_gp, HF_oracle_gp, BF_oracle_ra, LF_oracle_ra, HF_oracle_ra = nothing, nothing, nothing, nothing, nothing, nothing
        println("Starting repetition $repID ...")
        rd_seed = 20250531 + repID
        Random.seed!(rd_seed)

        # deserialize and assign KLE objects to variables.
        kle_gp_oracle, kle_ra_oracle = deserialize(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "kle_object_oracle_final.jls"))

        for case in kle_cases
            println("Predicting on grid for case: $case ...")
            if case == "gp"
                QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle = kle_gp_oracle.QLF, kle_gp_oracle.λLF, kle_gp_oracle.bβLF, kle_gp_oracle.regLF, kle_gp_oracle.YmLF
                QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle = kle_gp_oracle.QDelta, kle_gp_oracle.λDelta, kle_gp_oracle.bβDelta, kle_gp_oracle.regDelta, kle_gp_oracle.YmDelta
                QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = kle_gp_oracle.QHF, kle_gp_oracle.λHF, kle_gp_oracle.bβHF, kle_gp_oracle.regHF, kle_gp_oracle.YmHF

                BF_oracle_gp = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)
                LF_oracle_gp = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)
                HF_oracle_gp = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)

                for i in 1:length(grid_a_scaled)
                    for j in 1:length(grid_b_scaled)
                        ΨTest_LF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
                        ΨTest_Δ_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
                        ΨTest_HF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'

                        klModes_LF_oracle = QLF_oracle .* sqrt.(λLF_oracle)'
                        klModes_Δ_oracle = QDelta_oracle .* sqrt.(λDelta_oracle)'
                        klModes_HF_oracle = QHF_oracle .* sqrt.(λHF_oracle)'

                        BF_oracle_gp[i, j, :] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle + (klModes_Δ_oracle * bβDelta_oracle * ΨTest_Δ_oracle) + YMeanDelta_oracle

                        LF_oracle_gp[i, j, :] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle

                        HF_oracle_gp[i, j, :] = (klModes_HF_oracle * bβHF_oracle * ΨTest_HF_oracle) + YMeanHF_oracle
                    end
                end
            elseif case == "ra"
                QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle = kle_ra_oracle.QLF, kle_ra_oracle.λLF, kle_ra_oracle.bβLF, kle_ra_oracle.regLF, kle_ra_oracle.YmLF
                QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle = kle_ra_oracle.QDelta, kle_ra_oracle.λDelta, kle_ra_oracle.bβDelta, kle_ra_oracle.regDelta, kle_ra_oracle.YmDelta
                QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = kle_ra_oracle.QHF, kle_ra_oracle.λHF, kle_ra_oracle.bβHF, kle_ra_oracle.regHF, kle_ra_oracle.YmHF

                BF_oracle_ra = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)
                LF_oracle_ra = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)
                HF_oracle_ra = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)

                for i in 1:length(grid_a_scaled)
                    for j in 1:length(grid_b_scaled)
                        ΨTest_LF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
                        ΨTest_Δ_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
                        ΨTest_HF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'

                        klModes_LF_oracle = QLF_oracle .* sqrt.(λLF_oracle)'
                        klModes_Δ_oracle = QDelta_oracle .* sqrt.(λDelta_oracle)'
                        klModes_HF_oracle = QHF_oracle .* sqrt.(λHF_oracle)'

                        BF_oracle_ra[i, j, :] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle + (klModes_Δ_oracle * bβDelta_oracle * ΨTest_Δ_oracle) + YMeanDelta_oracle

                        LF_oracle_ra[i, j, :] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle
                        
                        HF_oracle_ra[i, j, :] = (klModes_HF_oracle * bβHF_oracle * ΨTest_HF_oracle) + YMeanHF_oracle
                    end
                end
            end
        end
        # npzwrite(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "all_surr_oracle_predictions_final.npz"), Dict("BF_oracle_gp"=>BF_oracle_gp, "LF_oracle_gp"=>LF_oracle_gp, "HF_oracle_gp"=>HF_oracle_gp, "BF_oracle_ra"=>BF_oracle_ra, "LF_oracle_ra"=>LF_oracle_ra, "HF_oracle_ra"=>HF_oracle_ra))

        # open(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "all_surr_oracle_predictions_final.jls"), "w") do io
        #     serialize(io, (BF_oracle_gp, LF_oracle_gp, HF_oracle_gp, BF_oracle_ra, LF_oracle_ra, HF_oracle_ra))
        # end
        np.savez_compressed(joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID), "all_surr_oracle_predictions_final.npz"), BF_oracle_gp=BF_oracle_gp, LF_oracle_gp=LF_oracle_gp, HF_oracle_gp=HF_oracle_gp, BF_oracle_ra=BF_oracle_ra, LF_oracle_ra=LF_oracle_ra, HF_oracle_ra=HF_oracle_ra)


        println("Finished replication $repID")
    end
else
    println("Surrogate predictions already generated. Sample Filepaths: ", joinpath(args_dict["data_dir"], @sprintf("rep_%03d", 1), "all_surr_oracle_predictions_final.npz"))
end








# cv_ra, oracle_ra, kle_ra = evaluateKLE(inputsLF_orig[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end], inputsHF_orig[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], x; 
# useAbsErr=0,
# all_folds=k_folds_batch, 
# grid_a_scaled=a_grid_scaled, 
# grid_b_scaled=b_grid_scaled)
