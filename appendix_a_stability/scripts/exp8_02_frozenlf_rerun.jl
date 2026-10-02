
using Random
using LinearAlgebra
using SpecialFunctions
using Distributions
using NPZ
using Printf
using JLD
using Combinatorics
using Serialization
using DelimitedFiles

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEPENDENCY_DIR = joinpath(ROOT, "dependencies")
include(joinpath(DEPENDENCY_DIR, "kleUtils.jl"))
include(joinpath(DEPENDENCY_DIR, "utils.jl"))

const FIXED_RANK_LF = 2
kle_kwargs = (order=3,
              dims=2,
              family="Legendre",
              useFullGrid=1,
              getAllModes=0,
              fixedRank=FIXED_RANK_LF,
              weightFunction=getWeights,
              solver="Tikhonov-L2")

const FIXED_RANK_DELTA = 1
kle_kwargs_Δ = (useFullGrid=1,
                getAllModes=0,
                fixedRank=FIXED_RANK_DELTA,
                order=3,
                dims=2,
                weightFunction=getWeights,
                family="Legendre",
                solver="Tikhonov-L2")

const CHOSEN_CASE = 2   # code's dict_case2 = paper's C1 (Taylor LF)
const LB = [40.0, 60.0]
const UB = [60.0, 80.0]
const CHOSEN_LF = "taylor"
const NFOLDS = 5
const N_PILOT_HF = 5
const N_ACQUIRED = 60   # matches Figure 6's N_ACQUIRED window

const INPUT_DIR = joinpath(ROOT, "data", "1d_inputs_c2_EI_nolog")
const ORACLE_CACHE = joinpath(ROOT, "data", "exp8", "exp8_HF_Oracle_case_02.jld")

const REP_IDS = isempty(ARGS) ? [5, 2] : parse.(Int, split(ARGS[1], ","))
const REP_TAG = isempty(ARGS) ? "" : "_reps" * join(REP_IDS, "")
const OUT_DIR = joinpath(ROOT, "data", "exp8", "exp8_frozenlf_c2$(REP_TAG)")
const OUTPUT_NPZ = joinpath(ROOT, "data", "exp8", "exp8_02_heatmap_arrays$(REP_TAG).npz")

mkpath(OUT_DIR)

x = collect(range(0, 0.1; length=250))
ng = length(x)

a_grid = collect(range(LB[1], UB[1]; length=200))
b_grid = collect(range(LB[2], UB[2]; length=200))
a_grid_scaled = 2 * (a_grid .- (1/2)*(LB[1] + UB[1])) ./ (UB[1] - LB[1])
b_grid_scaled = 2 * (b_grid .- (1/2)*(LB[2] + UB[2])) ./ (UB[2] - LB[2])

if !isfile(ORACLE_CACHE)
    println("Generating HF oracle grid (one-time, cached to $(ORACLE_CACHE))")
    global HF_oracle = generateOracleData(a_grid, b_grid, x)
    JLD.save(ORACLE_CACHE, "HF_oracle", HF_oracle)
else
    global HF_oracle = JLD.load(ORACLE_CACHE)["HF_oracle"]
end

function run_batch(repID, batchID, rd_seed, frozenLF_gp, frozenLF_ra)
    fileLF = joinpath(INPUT_DIR, @sprintf("rep_%03d", repID), @sprintf("LF_Batch_%03d_Final.txt", batchID))
    fileHF = joinpath(INPUT_DIR, @sprintf("rep_%03d", repID), @sprintf("HF_Batch_%03d_Final.txt", batchID))
    fileHFIdx = joinpath(INPUT_DIR, @sprintf("rep_%03d", repID), @sprintf("HF_Batch_%03d_Subset_Final.txt", batchID))

    inputsLF = readdlm(fileLF)
    inputsHF = readdlm(fileHF)
    inputsHFSubsetIdx = readdlm(fileHFIdx, Int64)[:]

    nLF = Int(size(inputsLF, 1) / 2)
    nHF = Int(size(inputsHF, 1) / 2)
    nPilotHF = N_PILOT_HF

    inputsLF_scaled = 2 * (inputsLF .- (1/2)*(LB + UB)') ./ (UB - LB)'
    inputsHF_scaled = 2 * (inputsHF .- (1/2)*(LB + UB)') ./ (UB - LB)'

    LF_data = generateLF(x, inputsLF; chosen_lf=CHOSEN_LF)
    HF_data = generateHF(x, inputsHF)

    delta_idx = extend_vector(inputsHFSubsetIdx[1:nPilotHF], inputsHFSubsetIdx[(nPilotHF + 1):nHF])
    Y_Delta = [HF_data[:, 1:nHF] - LF_data[:, inputsHFSubsetIdx[1:nHF]] HF_data[:, (nHF + 1):end] - LF_data[:, delta_idx]]

    k_folds_batch_1 = k_folds(inputsHFSubsetIdx[1:nHF], NFOLDS; rng_gen=MersenneTwister(rd_seed))
    k_folds_batch_2 = k_folds(inputsHFSubsetIdx[1:nHF], NFOLDS; rng_gen=MersenneTwister(rd_seed + 200))

    if frozenLF_gp === nothing
        frozenLF_gp = buildKLE(inputsLF_scaled[1:nLF, :], LF_data[:, 1:nLF], x; kle_kwargs...)
    end
    if frozenLF_ra === nothing
        frozenLF_ra = buildKLE(inputsLF_scaled[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], x; kle_kwargs...)
    end

    cv_gp, oracle_gp, kle_gp = evaluateKLE(inputsLF_scaled[1:nLF, :], LF_data[:, 1:nLF], inputsHFSubsetIdx[1:nHF],
        inputsHF_scaled[1:nHF, :], HF_data[:, 1:nHF], Y_Delta[:, 1:nHF], x;
        useAbsErr=0, all_folds=k_folds_batch_1, grid_a_scaled=a_grid_scaled, grid_b_scaled=b_grid_scaled,
        frozenLF=frozenLF_gp)
    cv_ra, oracle_ra, kle_ra = evaluateKLE(inputsLF_scaled[(nLF + 1):end, :], LF_data[:, (nLF + 1):end], inputsHFSubsetIdx[(nHF + 1):end],
        inputsHF_scaled[(nHF + 1):end, :], HF_data[:, (nHF + 1):end], Y_Delta[:, (nHF + 1):end], x;
        useAbsErr=0, all_folds=k_folds_batch_2, grid_a_scaled=a_grid_scaled, grid_b_scaled=b_grid_scaled,
        frozenLF=frozenLF_ra)

    QLF_gp, λLF_gp, bβLF_gp, regLF_gp, YMeanLF_gp = frozenLF_gp
    QDelta_gp, λDelta_gp, bβDelta_gp, regDelta_gp, YMeanDelta_gp = buildKLE(inputsHF_scaled[1:nHF, :], Y_Delta[:, 1:nHF], x; kle_kwargs_Δ...)
    QLF_ra, λLF_ra, bβLF_ra, regLF_ra, YMeanLF_ra = frozenLF_ra
    QDelta_ra, λDelta_ra, bβDelta_ra, regDelta_ra, YMeanDelta_ra = buildKLE(inputsHF_scaled[(nHF + 1):end, :], Y_Delta[:, (nHF + 1):end], x; kle_kwargs_Δ...)

    return (cv_gp=cv_gp, oracle_gp=oracle_gp, kle_gp=kle_gp, cv_ra=cv_ra, oracle_ra=oracle_ra, kle_ra=kle_ra,
            QLF_gp=QLF_gp, λLF_gp=λLF_gp, bβLF_gp=bβLF_gp, regLF_gp=regLF_gp, YMeanLF_gp=YMeanLF_gp,
            QDelta_gp=QDelta_gp, λDelta_gp=λDelta_gp, bβDelta_gp=bβDelta_gp, regDelta_gp=regDelta_gp, YMeanDelta_gp=YMeanDelta_gp,
            QLF_ra=QLF_ra, λLF_ra=λLF_ra, bβLF_ra=bβLF_ra, regLF_ra=regLF_ra, YMeanLF_ra=YMeanLF_ra,
            QDelta_ra=QDelta_ra, λDelta_ra=λDelta_ra, bβDelta_ra=bβDelta_ra, regDelta_ra=regDelta_ra, YMeanDelta_ra=YMeanDelta_ra,
            nLF=nLF, nHF=nHF, frozenLF_gp=frozenLF_gp, frozenLF_ra=frozenLF_ra)
end

gp_on_gp = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, length(REP_IDS))
gp_on_ra = zeros(N_ACQUIRED, N_ACQUIRED, length(REP_IDS))
ra_on_ra = zeros(N_PILOT_HF + N_ACQUIRED, N_ACQUIRED, length(REP_IDS))
ra_on_gp = zeros(N_ACQUIRED, N_ACQUIRED, length(REP_IDS))
rank_LF_gp_trace = zeros(N_ACQUIRED, length(REP_IDS))
rank_LF_ra_trace = zeros(N_ACQUIRED, length(REP_IDS))

for (rep_i, repID) in enumerate(REP_IDS)
    println("=== rep_$(lpad(repID,3,'0')) ($(rep_i)/$(length(REP_IDS))) ===")
    rd_seed = 20250431 + repID
    Random.seed!(rd_seed)
    mkpath(joinpath(OUT_DIR, @sprintf("rep_%03d", repID)))

    final_file = joinpath(INPUT_DIR, @sprintf("rep_%03d", repID), @sprintf("HF_Batch_%03d_Final.txt", N_ACQUIRED))
    all_inputs = readdlm(final_file)
    xi_pred_all_gp = all_inputs[(N_PILOT_HF + 1):(N_PILOT_HF + N_ACQUIRED), :]
    xi_pred_all_ra = all_inputs[(N_PILOT_HF + N_ACQUIRED + 1 + N_PILOT_HF):end, :]
    y_hf_gp_all = generateHF(x, xi_pred_all_gp)
    y_hf_ra_all = generateHF(x, xi_pred_all_ra)
    inputs_pred_all_gp = 2 * (xi_pred_all_gp .- (1/2)*(LB + UB)') ./ (UB - LB)'
    inputs_pred_all_ra = 2 * (xi_pred_all_ra .- (1/2)*(LB + UB)') ./ (UB - LB)'

    frozenLF_gp = nothing
    frozenLF_ra = nothing

    for batchID in 1:N_ACQUIRED
        println("  batch $batchID/$(N_ACQUIRED)")
        res = run_batch(repID, batchID, rd_seed, frozenLF_gp, frozenLF_ra)
        frozenLF_gp = res.frozenLF_gp
        frozenLF_ra = res.frozenLF_ra

        npzwrite(joinpath(OUT_DIR, @sprintf("rep_%03d", repID), @sprintf("case_objects_batch_%03d.npz", batchID)),
                 Dict("cv_gp" => res.cv_gp, "oracle_gp" => res.oracle_gp,
                      "cv_ra" => res.cv_ra, "oracle_ra" => res.oracle_ra,
                      "rank_LF_gp" => Float64(length(res.λLF_gp)), "rank_LF_ra" => Float64(length(res.λLF_ra))))
        open(joinpath(OUT_DIR, @sprintf("rep_%03d", repID), @sprintf("case_objects_batch_%03d.jls", batchID)), "w") do io
            serialize(io, (res.kle_gp, res.kle_ra))
        end

        rank_LF_gp_trace[batchID, rep_i] = length(res.λLF_gp)
        rank_LF_ra_trace[batchID, rep_i] = length(res.λLF_ra)

        xi_HF_gp_pred = all_inputs[(N_PILOT_HF + batchID + 1):(N_PILOT_HF + N_ACQUIRED), :]
        xi_HF_ra_pred = all_inputs[(N_PILOT_HF + N_ACQUIRED + batchID + 1 + N_PILOT_HF):end, :]
        y_hf_gp_pred = generateHF(x, xi_HF_gp_pred)
        y_hf_ra_pred = generateHF(x, xi_HF_ra_pred)
        inputs_hf_gp_pred = 2 * (xi_HF_gp_pred .- (1/2)*(LB + UB)') ./ (UB - LB)'
        inputs_hf_ra_pred = 2 * (xi_HF_ra_pred .- (1/2)*(LB + UB)') ./ (UB - LB)'

        gp_gp_pred = predictOnGrid(res.QLF_gp, res.λLF_gp, res.bβLF_gp, res.regLF_gp, res.YMeanLF_gp,
                                    res.QDelta_gp, res.λDelta_gp, res.bβDelta_gp, res.regDelta_gp, res.YMeanDelta_gp,
                                    inputs_hf_gp_pred, x)
        gp_ra_pred = predictOnGrid(res.QLF_gp, res.λLF_gp, res.bβLF_gp, res.regLF_gp, res.YMeanLF_gp,
                                    res.QDelta_gp, res.λDelta_gp, res.bβDelta_gp, res.regDelta_gp, res.YMeanDelta_gp,
                                    inputs_pred_all_ra, x)
        ra_ra_pred = predictOnGrid(res.QLF_ra, res.λLF_ra, res.bβLF_ra, res.regLF_ra, res.YMeanLF_ra,
                                    res.QDelta_ra, res.λDelta_ra, res.bβDelta_ra, res.regDelta_ra, res.YMeanDelta_ra,
                                    inputs_hf_ra_pred, x)
        ra_gp_pred = predictOnGrid(res.QLF_ra, res.λLF_ra, res.bβLF_ra, res.regLF_ra, res.YMeanLF_ra,
                                    res.QDelta_ra, res.λDelta_ra, res.bβDelta_ra, res.regDelta_ra, res.YMeanDelta_ra,
                                    inputs_pred_all_gp, x)

        gp_on_gp[1:(N_PILOT_HF + batchID), batchID, rep_i] = res.cv_gp
        gp_on_gp[(N_PILOT_HF + batchID + 1):end, batchID, rep_i] = [ϵ1(y_hf_gp_pred[:, i], gp_gp_pred[:, i]) for i in 1:size(xi_HF_gp_pred, 1)]
        gp_on_ra[:, batchID, rep_i] = [ϵ1(y_hf_ra_all[:, i], gp_ra_pred[:, i]) for i in 1:size(xi_pred_all_ra, 1)]

        ra_on_ra[1:(N_PILOT_HF + batchID), batchID, rep_i] = res.cv_ra
        ra_on_ra[(N_PILOT_HF + batchID + 1):end, batchID, rep_i] = [ϵ1(y_hf_ra_pred[:, i], ra_ra_pred[:, i]) for i in 1:size(xi_HF_ra_pred, 1)]
        ra_on_gp[:, batchID, rep_i] = [ϵ1(y_hf_gp_all[:, i], ra_gp_pred[:, i]) for i in 1:size(xi_pred_all_gp, 1)]
    end
end

npzwrite(OUTPUT_NPZ,
         Dict("gp_on_gp" => gp_on_gp, "gp_on_ra" => gp_on_ra,
              "ra_on_ra" => ra_on_ra, "ra_on_gp" => ra_on_gp,
              "rep_ids" => Float64.(REP_IDS),
              "rank_LF_gp_trace" => rank_LF_gp_trace, "rank_LF_ra_trace" => rank_LF_ra_trace,
              "N_PILOT_HF" => Float64(N_PILOT_HF), "N_ACQUIRED" => Float64(N_ACQUIRED)))

println("\nExp 8_02 done. Heatmap arrays saved to $(OUTPUT_NPZ)")
