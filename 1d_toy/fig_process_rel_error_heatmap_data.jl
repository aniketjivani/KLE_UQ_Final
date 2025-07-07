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

include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/kleUtils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/utils.jl")

sys = pyimport("sys")
pkl = pyimport("pickle")

kle_cases = ["gp", "ra"]
x = collect(range(0, 0.1; length=250))
ng = length(x)

# we will evaluate case of getAllModes = 0 and 1, where 1 is turned on only for the delta term.

kle_kwargs_Δ = (useFullGrid=1,
            getAllModes=1,
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
            "BUDGET_HF"=>50,
            "lb" => [40, 30],
            "ub" => [60, 50],
            "acqFunc" => "EI",
            "chosen_lf" => "bSine",
            "N_PILOT_HF" => 5
       )

dict_case2 = Dict("plot_dir"=> "./Plots/1d_toy_plots_long_c2",
            "data_dir"=> "./1d_toy/1d_pred_data_c2",
            "input_dir"=> "./1d_toy/1d_inputs_c2",
            "NREPS"=> 5,
            "NFOLDS"=> 5,
            "BUDGET_HF"=>50,
            "lb" => [40, 60],
            "ub" => [60, 80],
            "acqFunc" => "EI",
            "chosen_lf" => "taylor",
            "N_PILOT_HF" => 5
       )

if kle_kwargs_Δ.getAllModes == 1
    dict_case1["data_dir"] = "./1d_toy/1d_pred_data_c1_all_modes"
    dict_case1["input_dir"] = "./1d_toy/1d_inputs_c1_all_modes"
    dict_case2["data_dir"] = "./1d_toy/1d_pred_data_c2_all_modes"
    dict_case2["input_dir"] = "./1d_toy/1d_inputs_c2_all_modes"
    dict_case1["plot_dir"] = "./Plots/1d_toy_plots_long_c1_all_modes"
    dict_case2["plot_dir"] = "./Plots/1d_toy_plots_long_c2_all_modes"
elseif kle_kwargs_Δ.getAllModes == 0
    dict_case1["data_dir"] = "./1d_toy/1d_pred_data_c1"
    dict_case1["input_dir"] = "./1d_toy/1d_inputs_c1"
    dict_case2["data_dir"] = "./1d_toy/1d_pred_data_c2"
    dict_case2["input_dir"] = "./1d_toy/1d_inputs_c2"
    dict_case1["plot_dir"] = "./Plots/1d_toy_plots_long_c1"
    dict_case2["plot_dir"] = "./Plots/1d_toy_plots_long_c2"
end



chosen_case = 2
if chosen_case == 1
    args_dict = dict_case1
elseif chosen_case == 2
    args_dict = dict_case2
end

N_PILOT_HF = args_dict["N_PILOT_HF"]
N_ACQUIRED = args_dict["BUDGET_HF"]
N_REPS = args_dict["NREPS"]

lb = args_dict["lb"]
ub = args_dict["ub"]
acqFunc = args_dict["acqFunc"]

gp_on_gp = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS);
gp_on_ra = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS);
ra_on_ra = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, N_REPS);
ra_on_gp = zeros(N_ACQUIRED, N_ACQUIRED, N_REPS);


for repID in 1:args_dict["NREPS"]
    println("Starting repetition $repID")
    rd_seed = 20250531 + repID
    Random.seed!(rd_seed)
    inputs_dir = joinpath(args_dict["input_dir"], @sprintf("rep_%03d", repID))
	data_dir = joinpath(args_dict["data_dir"], @sprintf("rep_%03d", repID))
    input_file = joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Final.txt", N_ACQUIRED))
    
    all_inputs = readdlm(input_file)

    xi_pred_all_gp = all_inputs[(N_PILOT_HF + 1):(N_PILOT_HF + N_ACQUIRED), :]
    xi_pred_all_ra = all_inputs[(N_PILOT_HF + N_ACQUIRED + 1 + N_PILOT_HF):end, :]

    # inputs_hf_gp_all = 2 * (xi_pred_all_gp .- (1/2)*(lb + ub)') ./ (ub - lb)'
    # inputs_hf_ra_all = 2 * (xi_pred_all_ra .- (1/2)*(lb + ub)') ./ (ub - lb)'

    y_hf_gp_all = generateHF(x, xi_pred_all_gp)
	y_hf_ra_all = generateHF(x, xi_pred_all_ra)

    for batchID in 0:(N_ACQUIRED - 1)
        # load KLE params for each case

        println("Starting Batch  $batchID")
        case_objects = npzread(joinpath(data_dir, @sprintf("case_objects_batch_%03d.npz", batchID)))

        kle_data = deserialize(joinpath(data_dir, @sprintf("case_objects_batch_%03d.jls", batchID)))

        cv_gp = case_objects["cv_gp"]
        cv_ra = case_objects["cv_ra"]

        kle_gp_obj = kle_data[1]
        kle_ra_obj = kle_data[2]

        # cross-predictions across GP and RA inputs
        # yPredicted[:, holdout_indices[1]] = (klModes_LF * bβLF * ΨTest_LF) .+ YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) .+ YMeanDelta

        xi_HF_gp_pred = all_inputs[(N_PILOT_HF + batchID + 1):(N_PILOT_HF + N_ACQUIRED), :]
		xi_HF_ra_pred = all_inputs[(N_PILOT_HF + N_ACQUIRED + batchID + 1 + N_PILOT_HF):end, :]

        # y_hf_gp_all = generateHF(x, xi_HF_gp_pred)
        # y_hf_ra_all = generateHF(x, xi_HF_ra_pred)


        LF_data, HF_data = nothing, nothing
        # inputsLF_orig, inputsHF_orig = nothing, nothing
        inputsHFSubsetIdx = nothing
        xi_lf, xi_hf = nothing, nothing

        y_hf_gp_pred = generateHF(x, xi_HF_gp_pred)
		y_hf_ra_pred = generateHF(x, xi_HF_ra_pred)

        if batchID == 0
            xi_hf = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_HF_Pilot_scaled.txt")

            xi_lf = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LF_Pilot_scaled.txt")

            xi_hf = 0.5 * xi_hf .* (ub - lb)' .+ 0.5 * (ub + lb)'

            xi_lf = 0.5 * xi_lf .* (ub - lb)' .+ 0.5 * (ub + lb)'

            HF_data = generateHF(x, xi_hf)
            LF_data = generateLF(x, xi_lf; chosen_lf=args_dict["chosen_lf"])

            inputsHFSubsetIdx = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/input_list_LFIdx.txt", Int64)[:]
        else
            xi_hf = readdlm(joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Final.txt", batchID)))

            xi_lf = readdlm(joinpath(inputs_dir, @sprintf("LF_Batch_%03d_Final.txt", batchID)))

            # here orig == scaled inputs.

            xi_hf = 2 * (xi_hf .- (1/2)*(lb + ub)') ./ (ub - lb)'

            xi_lf = 2 * (xi_lf .- (1/2)*(lb + ub)') ./ (ub - lb)'


            inputsHFSubsetIdx = readdlm(joinpath(inputs_dir, @sprintf("HF_Batch_%03d_Subset_Final.txt", batchID)), Int64)[:]

            HF_data = generateHF(x, xi_hf)

            LF_data = generateLF(x, xi_lf; chosen_lf=args_dict["chosen_lf"])
        end

        nLF = Int(size(xi_lf, 1) / 2)
        nHF = Int(size(xi_hf, 1) / 2)

        Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, inputsHFSubsetIdx[(nHF + 1):end]]]


        QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = nothing, nothing, nothing, nothing, nothing
        QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = nothing, nothing, nothing, nothing, nothing
        QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = nothing, nothing, nothing, nothing, nothing
        QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = nothing, nothing, nothing, nothing, nothing
        for case in kle_cases
            println("Rebuilding surrogate for case $case")
            if case == "gp"
                QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = buildKLE(xi_lf[1:nLF, :], LF_data[:, 1:nLF], x; kle_kwargs...)

	            QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = buildKLE(xi_hf[1:nHF, :], Y_Delta[:, 1:nHF], x; kle_kwargs_Δ...)
            elseif case == "ra"
                QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = buildKLE(xi_lf[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], x; kle_kwargs...)

	            QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = buildKLE(xi_hf[1:nHF, :], Y_Delta[:, (nHF + 1):end], x; kle_kwargs_Δ...)
            end
        end
        
        # inputs_hf_gp_pred = 0.5 * xi_HF_gp_pred .* (ub - lb)' .+ 0.5 * (ub + lb)'
		# inputs_hf_ra_pred = 0.5 * xi_HF_ra_pred .* (ub - lb)' .+ 0.5 * (ub + lb)'


        inputs_hf_gp_pred = 2 * (xi_HF_gp_pred .- (1/2)*(lb + ub)') ./ (ub - lb)'

        inputs_hf_ra_pred = 2 * (xi_HF_ra_pred .- (1/2)*(lb + ub)') ./ (ub - lb)'

        inputs_pred_all_gp = 2 * (xi_pred_all_gp .- (1/2)*(lb + ub)') ./ (ub - lb)'

        inputs_pred_all_ra = 2 * (xi_pred_all_ra .- (1/2)*(lb + ub)') ./ (ub - lb)'



        gp_gp_pred = predictOnGrid(QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp, QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp, inputs_hf_gp_pred, x)
		gp_ra_pred = predictOnGrid(QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp, QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp, inputs_pred_all_ra, x)
	
		ra_ra_pred = predictOnGrid(QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra, QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra, inputs_hf_ra_pred, x)
		ra_gp_pred = predictOnGrid(QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra, QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra, inputs_pred_all_gp, x)
        
        gp_on_gp[1:(N_PILOT_HF + batchID), batchID + 1, repID] = cv_gp
		gp_on_gp[(N_PILOT_HF + batchID + 1):end, batchID + 1, repID] = [ϵ1(y_hf_gp_pred[:, i], gp_gp_pred[:, i]) for i in 1:size(xi_HF_gp_pred, 1)]
		gp_on_ra[:, batchID + 1, repID] = [ϵ1(y_hf_ra_all[:, i], gp_ra_pred[:, i]) for i in 1:size(xi_pred_all_ra, 1)]
			
		ra_on_ra[1:(N_PILOT_HF + batchID), batchID + 1, repID] = cv_ra
		ra_on_ra[(N_PILOT_HF + batchID + 1):end, batchID + 1, repID] = [ϵ1(y_hf_ra_pred[:, i], ra_ra_pred[:, i]) for i in 1:size(xi_HF_ra_pred, 1)]
		ra_on_gp[:, batchID + 1, repID] = [ϵ1(y_hf_gp_all[:, i], ra_gp_pred[:, i]) for i in 1:size(xi_pred_all_gp, 1)]

    end
end



if kle_kwargs_Δ.getAllModes == 1
    # save results for all reps using Serialization 
    open(joinpath("./1d_toy/", @sprintf("err_heatmap_all_reps_case_%03d_all_modes.jls", chosen_case)), "w") do io
        serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
    end
elseif kle_kwargs_Δ.getAllModes == 0
    # save results for all reps using Serialization 
    open(joinpath("./1d_toy/", @sprintf("err_heatmap_all_reps_case_%03d.jls", chosen_case)), "w") do io
        serialize(io, (gp_on_gp, gp_on_ra, ra_on_ra, ra_on_gp))
    end
end





