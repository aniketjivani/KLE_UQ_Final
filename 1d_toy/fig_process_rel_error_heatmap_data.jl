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
            "N_PILOT_HF" => 10
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
            "N_PILOT_HF" => 10
       )

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

    xi_pred_all_gp = all_inputs[(N_PILOT_HF + 1):N_ACQUIRED, :]
    xi_pred_all_ra = all_hf_ra[(N_PILOT_HF + 1):end, :]











