# get error data on B - NHF points for every batch.
# Plot as a heatmap, highlight certain batches.
using LatinHypercubeSampling
using Random
using LinearAlgebra
using GaussianProcesses
using Optim
using SpecialFunctions
using Distributions
using PyCall
using JLD
using NPZ
using CSV
using MAT
using DataFrames
using DelimitedFiles
using Printf
using Statistics
using Serialization
# ϵ2(y1::Vector, y2::Vector) = norm((y1 - y2)) / norm(y1)
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/kleUtils.jl")
include("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/utils.jl")

kle_kwargs_Δ = (useFullGrid=1,
            getAllModes=0,
            order=3,
            dims=3,
            weightFunction=getWeights,
            family="Legendre",
			solver="Tikhonov-L2"
)

kle_kwargs = (order=3,
			dims=3,
			family="Legendre",
			useFullGrid=1,
			getAllModes=0,
            weightFunction=getWeights,
			solver="Tikhonov-L2"
			)

lb = [293.24, 0.1, 1.531]
ub = [312.94, 0.3, 4.6055]
d = load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/PilotBatchData.jld")
xbyD = d["xbyD"]
HFLF_ID_Pilot = d["HFLFID"]
nLF_Pilot = 200
nHF_Pilot = 15
BTotal = 50
xi_LF_all = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/LF_AllPoints_Scaled.txt")
xi_HF_all = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HF_AllPoints_Scaled.txt")
xi_LF_all_scaled = 0.5 * xi_LF_all .* (ub - lb)' .+ 0.5 * (ub + lb)'
xi_HF_all_scaled = 0.5 * xi_HF_all .* (ub - lb)' .+ 0.5 * (ub + lb)'
HFLF_ID_all = vcat(HFLF_ID_Pilot, nLF_Pilot .+ collect(1:(BTotal - nHF_Pilot)))

test_points_all = matread("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/JetTestPts.mat")["test_points"]

test_points_all_scaled = 0.5 * test_points_all .* (ub - lb)' .+ 0.5 * (ub + lb)'

dlf_all = load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/LFDataAll.jld")
yLF_V_All = dlf_all["yLFVAll"]
yLF_UU_All = dlf_all["yLFUUAll"]
yLF_UW_All = dlf_all["yLFUWAll"]

dhf_all = load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HFDataAll.jld")
yHF_V_All = dhf_all["yHFVAll"]
yHF_UU_All = dhf_all["yHFUUAll"]
yHF_UW_All = dhf_all["yHFUWAll"]

yDelta_V_All = yHF_V_All - yLF_V_All[:, HFLF_ID_all]
yDelta_UU_All = yHF_UU_All - yLF_UU_All[:, HFLF_ID_all]
yDelta_UW_All = yHF_UW_All - yLF_UW_All[:, HFLF_ID_all]

mutable struct JetData
	MV
	UU
	UW
    JetData(MV, UU, UW) = new(MV, UU, UW)
end

random_data = load("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/LFHFRandomData.jld")

yLF_V_Random = random_data["yLFV"]
yLF_UU_Random = random_data["yLFUU"]
yLF_UW_Random = random_data["yLFUW"]
yHF_V_Random = random_data["yHFV"]
yHF_UU_Random = random_data["yHFUU"]
yHF_UW_Random = random_data["yHFUW"]

xi_HF_Random = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HFBatchRandomPoints.txt")

xi_HF_Random_scaled = 0.5 * xi_HF_Random .* (ub - lb)' .+ 0.5 * (ub + lb)'


xi_LF_all_random = [xi_LF_all[1:nLF_Pilot, :]; xi_HF_Random]
xi_HF_all_random = [xi_HF_all[1:nHF_Pilot, :]; xi_HF_Random]
yLF_V_all_random = [yLF_V_All[:, 1:nLF_Pilot] yLF_V_Random]
yLF_UU_all_random = [yLF_UU_All[:, 1:nLF_Pilot] yLF_UU_Random]
yLF_UW_all_random = [yLF_UW_All[:, 1:nLF_Pilot] yLF_UW_Random]
yHF_V_all_random = [yHF_V_All[:, 1:nHF_Pilot] yHF_V_Random]
yHF_UU_all_random = [yHF_UU_All[:, 1:nHF_Pilot] yHF_UU_Random]
yHF_UW_all_random = [yHF_UW_All[:, 1:nHF_Pilot] yHF_UW_Random]
HFLF_ID_all_random = vcat(HFLF_ID_Pilot, nLF_Pilot .+ collect(1:size(xi_HF_Random, 1)))

yDelta_V_all_random = yHF_V_all_random - yLF_V_all_random[:, HFLF_ID_all_random]
yDelta_UU_all_random = yHF_UU_all_random - yLF_UU_all_random[:, HFLF_ID_all_random]
yDelta_UW_all_random = yHF_UW_all_random - yLF_UW_all_random[:, HFLF_ID_all_random]

# function to get test errors (just relative difference no LOO for each batch)
function getTestErrors(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; type="hfpred", hf_oracle=nothing, test_points=nothing)
	yPredicted = zeros(size(yLF_data, 1), size(test_points, 1))
	yHFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	yLFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	
	QLF, λLF, bβLF, regLF, YMeanLF = buildKLE(xi_LF, yLF_data, grid; kle_kwargs...)
	QDelta, λDelta, bβDelta, regDelta, YMeanDelta = buildKLE(xi_HF, yDelta_data, grid; kle_kwargs_Δ...)
	QHF, λHF, bβHF, regHF, YMeanHF = buildKLE(xi_HF, yHF_data, grid; kle_kwargs...)

    # print eigenvalues.
    # println("LF eigenvalues: ", λLF)
    # println("Delta eigenvalues: ", λDelta)
    # println("HF eigenvalues: ", λHF)

	testErrors = []
	for i in 1:size(test_points, 1)
		ΨTest_LF = PrepCaseA(test_points[i, :]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
		ΨTest_Δ = PrepCaseA(test_points[i, :]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

		klModes_LF = QLF .* sqrt.(λLF)'
		klModes_HF = QHF .* sqrt.(λHF)'
		klModes_Δ  = QDelta .* sqrt.(λDelta)'
		
		yPredicted[:, i] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) + YMeanDelta

		yLFPred[:, i] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF
		yHFPred[:, i] = (klModes_HF * bβHF * ΨTest_LF) + YMeanHF # psi test will be same for LF and HF in this case.
        if type == "hfpred"
		    push!(testErrors, ϵ2(yHFPred[:, i], yPredicted[:, i]))
        # elseif type == "hfpred" and hf_oracle != nothing
        #     push!(testErrors, ϵ2(hf_oracle[:, i], yPredicted[:, i]))
        elseif type == "hforacle" # where true HF data is available.
            push!(testErrors, ϵ2(hf_oracle[:, i], yPredicted[:, i]))
        end
	end
	return yPredicted, yLFPred, yHFPred, testErrors, length(λLF), length(λDelta)
end

function getLOOErrors(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid)
	yPredicted = zeros(size(yLF_data, 1), size(xi_HF, 1))
	for (idx, HFID) in enumerate(HF_LF_ID)
		QLF, λLF, bβLF, regLF, YMeanLF = buildKLE(xi_LF[1:end .!= HFID, :], yLF_data[:, 1:end .!= HFID], grid; kle_kwargs...)
		QDelta, λDelta, bβDelta, regDelta, YMeanDelta = buildKLE(xi_HF[1:end .!= idx, :], yDelta_data[:, 1:end .!= idx], grid; kle_kwargs_Δ...)

		ΨTest_LF = PrepCaseA(xi_LF[HFID, :]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
		ΨTest_Δ = PrepCaseA(xi_HF[idx, :]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

		klModes_LF = QLF .* sqrt.(λLF)'
		klModes_Δ  = QDelta .* sqrt.(λDelta)'

		yPredicted[:, idx] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) + YMeanDelta
	end
	LOOErrors = [ϵ2(yHF_data[:, i], yPredicted[:, i]) for i in 1:size(xi_HF, 1)]

	return yPredicted, LOOErrors
end

# use getTestErrors on (B - NHF) points for every batch.
n_batches = 7
n_batches_random = 5
n_batch_total = n_batches + 1
batch_size = 5



# xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; type="hfpred", hf_oracle=nothing

# specify type Vector{Vector{Float64}}
all_batch_rem_budget_errs = []

modesLF_V_all_gp = []
modesDelta_V_all_gp = []
modesLF_UU_all_gp = []
modesDelta_UU_all_gp = []
modesLF_UW_all_gp = []
modesDelta_UW_all_gp = []

for bID in 0:(n_batches)
    println("Batch ID: $bID")
    println("Using first $(nLF_Pilot + bID * batch_size) LF points and first $(nHF_Pilot + bID * batch_size) HF points.")
    _, _, _, testErrors_V, modesLF_V, modesDelta_V = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_V_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_V_All[:, (nHF_Pilot + bID * batch_size + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + bID * batch_size + 1):end, :]);

    _, _, _, testErrors_UU, modesLF_UU, modesDelta_UU = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UU_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UU_All[:, (nHF_Pilot + bID * batch_size + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + bID * batch_size + 1):end, :]);

    _, _, _, testErrors_UW, modesLF_UW, modesDelta_UW = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UW_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UW_All[:, (nHF_Pilot + bID * batch_size + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + bID * batch_size + 1):end, :]);

    if bID <= (n_batches - 1)
        mean_test_errs = mean([testErrors_V testErrors_UU testErrors_UW], dims=2)
        push!(all_batch_rem_budget_errs, mean_test_errs)
    end

    push!(modesLF_V_all_gp, modesLF_V)
    push!(modesDelta_V_all_gp, modesDelta_V)
    push!(modesLF_UU_all_gp, modesLF_UU)
    push!(modesDelta_UU_all_gp, modesDelta_UU)
    push!(modesLF_UW_all_gp, modesLF_UW)
    push!(modesDelta_UW_all_gp, modesDelta_UW)
end

# open("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataRemBudget_AL.jls", "w") do io
#     serialize(io, all_batch_rem_budget_errs)
# end

rem_budget_errs = zeros(50, 8)
for bID in 0:(n_batches - 1)
    rem_budget_errs[1:length(all_batch_rem_budget_errs[bID + 1]), bID + 1] = all_batch_rem_budget_errs[bID + 1]
end

npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataRemBudget_AL.npy", rem_budget_errs)

# get errors on random test set, in "hforacle" mode.
all_batch_holdout_rs_errs = []
bf_predictions_holdout_rs = []
for bID in 0:(n_batches)
    println("Batch ID: $bID")
    println("Using first $(nLF_Pilot + bID * batch_size) LF points and first $(nHF_Pilot + bID * batch_size) HF points.")
    yVBF_holdout_rs, _, _, testErrors_V = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_V_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_V_Random,
    test_points=xi_HF_Random);

    yUUBF_holdout_rs, _, _, testErrors_UU = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UU_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UU_Random,
    test_points=xi_HF_Random);

    yUWBF_holdout_rs, _, _, testErrors_UW = getTestErrors(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UW_All[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UW_Random,
    test_points=xi_HF_Random);


    mean_test_errs = mean([testErrors_V testErrors_UU testErrors_UW], dims=2)
    push!(all_batch_holdout_rs_errs, mean_test_errs)
    push!(bf_predictions_holdout_rs,
    (yVBF_holdout_rs, yUUBF_holdout_rs, yUWBF_holdout_rs))
end

yHoldoutRS_all = zeros(size(yHF_V_Random, 1), size(yHF_V_Random, 2), n_batch_total, 3)
holdout_rs_errs = zeros(size(yHF_V_Random, 2), n_batch_total)
for bID in 0:(n_batches)
    holdout_rs_errs[:, bID + 1] = all_batch_holdout_rs_errs[bID + 1]
    yHoldoutRS_all[:, :, bID + 1, 1] = bf_predictions_holdout_rs[bID + 1][1]
    yHoldoutRS_all[:, :, bID + 1, 2] = bf_predictions_holdout_rs[bID + 1][2]
    yHoldoutRS_all[:, :, bID + 1, 3] = bf_predictions_holdout_rs[bID + 1][3]
end
npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_RS_SurrAL.npy", holdout_rs_errs)
npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/BFPredictionsHoldout_RS_SurrAL.npy", yHoldoutRS_all)
yRandom_True = zeros(size(yHF_V_Random, 1), size(yHF_V_Random, 2), 3)
yRandom_True[:, :, 1] = yHF_V_Random
yRandom_True[:, :, 2] = yHF_UU_Random
yRandom_True[:, :, 3] = yHF_UW_Random
npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HFDataHoldout_RS.npy", yRandom_True)




# set hifi-KLE dataset for holdout test points to combination of random and AL simulations.
xi_HF_combined = [xi_HF_all; xi_HF_Random]
yHF_V_combined = [yHF_V_All yHF_V_Random]
yHF_UU_combined = [yHF_UU_All yHF_UU_Random]
yHF_UW_combined = [yHF_UW_All yHF_UW_Random]

function getTestErrorsPred(xi_LF, yLF_data, HF_LF_ID, xi_HF, xi_HF_KLE, yHF_data, yDelta_data, grid; type="hfpred", test_points=nothing)
	yPredicted = zeros(size(yLF_data, 1), size(test_points, 1))
	yHFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	yLFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	
	QLF, λLF, bβLF, regLF, YMeanLF = buildKLE(xi_LF, yLF_data, grid; kle_kwargs...)
	QDelta, λDelta, bβDelta, regDelta, YMeanDelta = buildKLE(xi_HF, yDelta_data, grid; kle_kwargs_Δ...)
	QHF, λHF, bβHF, regHF, YMeanHF = buildKLE(xi_HF_KLE, yHF_data, grid; kle_kwargs...)
	testErrors = []
	for i in 1:size(test_points, 1)
		ΨTest_LF = PrepCaseA(test_points[i, :]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
		ΨTest_Δ = PrepCaseA(test_points[i, :]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

		klModes_LF = QLF .* sqrt.(λLF)'
		klModes_HF = QHF .* sqrt.(λHF)'
		klModes_Δ  = QDelta .* sqrt.(λDelta)'
		
		yPredicted[:, i] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) + YMeanDelta

		yLFPred[:, i] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF
		yHFPred[:, i] = (klModes_HF * bβHF * ΨTest_LF) + YMeanHF # psi test will be same for LF and HF in this case.
        if type == "hfpred"
		    push!(testErrors, ϵ2(yHFPred[:, i], yPredicted[:, i]))
        end
	end
	return yPredicted, yLFPred, yHFPred, testErrors
end


yBFPredALV, yLFPredALV, yHFPredALV, _ = getTestErrorsPred(xi_LF_all,
yLF_V_All[:, 1:(nLF_Pilot + 5 * batch_size)],
HFLF_ID_all[1:(nHF_Pilot + 5 * batch_size)],
xi_HF_all,
xi_HF_all,
yHF_V_All,
yDelta_V_All,
xbyD; type="hfpred",
test_points=test_points_all);

yBFPredALUU, yLFPredALUU, yHFPredALUU, _ = getTestErrorsPred(xi_LF_all,
yLF_UU_All,
HFLF_ID_all,
xi_HF_all,
xi_HF_all,
yHF_UU_All,
yDelta_UU_All,
xbyD; type="hfpred",
test_points=test_points_all);

yBFPredALUW, yLFPredALUW, yHFPredALUW, _ = getTestErrorsPred(xi_LF_all,
yLF_UW_All,
HFLF_ID_all,
xi_HF_all,
xi_HF_all,
yHF_UW_All,
yDelta_UW_All,
xbyD; type="hfpred",
test_points=test_points_all);

yBFPredRSV, yLFPredRSV, yHFPredRSV, _ = getTestErrorsPred(xi_LF_all_random,
yLF_V_all_random,
HFLF_ID_all_random,
xi_HF_all_random,
xi_HF_all_random,
yHF_V_all_random,
yDelta_V_all_random,
xbyD; type="hfpred",
test_points=test_points_all);

yBFPredRSUU, yLFPredRSUU, yHFPredRSUU, _ = getTestErrorsPred(xi_LF_all_random,
yLF_UU_all_random,
HFLF_ID_all_random,
xi_HF_all_random,
xi_HF_all_random,
yHF_UU_all_random,
yDelta_UU_all_random,
xbyD; type="hfpred",
test_points=test_points_all);

yBFPredRSUW, yLFPredRSUW, yHFPredRSUW, _ = getTestErrorsPred(xi_LF_all_random,
yLF_UW_all_random,
HFLF_ID_all_random,
xi_HF_all_random,
xi_HF_all_random,
yHF_UW_all_random,
yDelta_UW_all_random,
xbyD; type="hfpred",
test_points=test_points_all);

npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/UQPredsAL.npz",
Dict("yBFPredALV" => yBFPredALV,
"yLFPredALV" => yLFPredALV,
"yHFPredALV" => yHFPredALV,
"yBFPredALUU" => yBFPredALUU,
"yLFPredALUU" => yLFPredALUU,
"yHFPredALUU" => yHFPredALUU,
"yBFPredALUW" => yBFPredALUW,
"yLFPredALUW" => yLFPredALUW,
"yHFPredALUW" => yHFPredALUW))

npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/UQPredsRandom.npz",
Dict("yBFPredRSV" => yBFPredRSV,
"yLFPredRSV" => yLFPredRSV,
"yHFPredRSV" => yHFPredRSV,
"yBFPredRSUU" => yBFPredRSUU,
"yLFPredRSUU" => yLFPredRSUU,
"yHFPredRSUU" => yHFPredRSUU,
"yBFPredRSUW" => yBFPredRSUW,
"yLFPredRSUW" => yLFPredRSUW,
"yHFPredRSUW" => yHFPredRSUW))







# # get errors on fixed test set, in "hfpred" mode.
# all_batch_holdout_test_errs = []
# for bID in 0:(n_batches)
#     println("Batch ID: $bID")
#     println("Using first $(nLF_Pilot + bID * batch_size) LF points and first $(nHF_Pilot + bID * batch_size) HF points.")
#     _, _, _, testErrors_V = getTestErrorsPred(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_V_All[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
#     # xi_HF_combined,
#     # # yHF_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     # yHF_V_combined,
#     xi_HF_all,
#     yHF_V_All,
#     yDelta_V_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     _, _, _, testErrors_UU = getTestErrorsPred(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_UU_All[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
#     # xi_HF_combined,
#     # # yHF_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     # yHF_UU_combined,
#     xi_HF_all,
#     yHF_UU_All,
#     yDelta_UU_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     _, _, _, testErrors_UW = getTestErrorsPred(xi_LF_all[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_UW_All[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all[1:(nHF_Pilot + bID * batch_size), :],
#     # xi_HF_combined,
#     # # yHF_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     # yHF_UW_combined,
#     xi_HF_all,
#     yHF_UW_All,
#     yDelta_UW_All[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     mean_test_errs = mean([testErrors_V testErrors_UU testErrors_UW], dims=2)
#     push!(all_batch_holdout_test_errs, mean_test_errs)
# end

# # open("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrAL.jls", "w") do io
# #     serialize(io, all_batch_holdout_test_errs)
# # end

# holdout_test_errs = zeros(size(test_points_all, 1), n_batch_total)
# for bID in 0:(n_batches)
#     holdout_test_errs[:, bID + 1] = all_batch_holdout_test_errs[bID + 1]
# end
# npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrAL.npy", holdout_test_errs)

# # obtain similar holdout test err with surrogate built via random sampling. (one shot)
# all_batch_holdout_test_errs_random = []
# for bID in 0:(n_batches_random)
#     println("Batch ID: $bID")
#     println("Using first $(nLF_Pilot + bID * batch_size) LF points and first $(nHF_Pilot + bID * batch_size) HF points.")
#     _, _, _, testErrors_V = getTestErrorsPred(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_V_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all_random[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
#     # # xi_HF_combined,
#     # # # yHF_V_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     # # yHF_V_combined,
#     # xi_HF_all_random,
#     # yHF_V_all_random,
#     xi_HF_all,
#     yHF_V_All,
#     yDelta_V_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     _, _, _, testErrors_UU = getTestErrorsPred(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_UU_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all_random[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
#     # # xi_HF_combined,
#     # # # yHF_UU_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     # # yHF_UU_combined,
#     # xi_HF_all_random,
#     # yHF_UU_all_random,
#     xi_HF_all,
#     yHF_UU_All,
#     yDelta_UU_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     _, _, _, testErrors_UW = getTestErrorsPred(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
#     yLF_UW_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
#     HFLF_ID_all_random[1:(nHF_Pilot + bID * batch_size)],
#     xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
#     # # xi_HF_combined,
#     # # # yHF_UW_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     # # yHF_UW_combined,
#     # xi_HF_all_random,
#     # yHF_UW_all_random,
#     xi_HF_all,
#     yHF_UW_All,
#     yDelta_UW_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
#     xbyD; type="hfpred",
#     test_points=test_points_all);

#     mean_test_errs = mean([testErrors_V testErrors_UU testErrors_UW], dims=2)

#     push!(all_batch_holdout_test_errs_random, mean_test_errs)
# end

# # open("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrRandom.jls", "w") do io
# #     serialize(io, all_batch_holdout_test_errs_random)
# # end

# holdout_test_errs_random = zeros(size(test_points_all, 1), n_batches_random + 1)
# for bID in 0:(n_batches_random)
#     holdout_test_errs_random[:, bID + 1] = all_batch_holdout_test_errs_random[bID + 1]
# end
# npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_TestPts_SurrRandom.npy", holdout_test_errs_random)


# load mean errors per batch.
all_batch_train_budget_errs = []

for bID in 0:(n_batches)
    println("Batch ID: $bID")
    
    if bID != n_batches
        mean_errs = readdlm(@sprintf("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch%02d.txt", bID + 1))
    else
        mean_errs = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorFinal.txt")
    end

    push!(all_batch_train_budget_errs, mean_errs)
end

train_budget_errs = zeros(50, 8)

for bID in 0:(n_batches)
    if bID != n_batches
        mean_errs = readdlm(@sprintf("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorBatch%02d.txt", bID + 1))
    else
        mean_errs = readdlm("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/MeanErrorFinal.txt")
    end
    train_budget_errs[1:length(mean_errs), bID + 1] = mean_errs
end


npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataTrainBudget_AL.npy", train_budget_errs)


# get errors on AL test set in "hforacle" mode.
all_batch_holdout_al_errs = []
# bf_predictions_holdout_al = []
for bID in 0:(n_batches_random)
    println("Batch ID: $bID")
    println("Using first $(nLF_Pilot + bID * batch_size) LF points and first $(nHF_Pilot + bID * batch_size) HF points.")
    yVBF_holdout_al, _, _, testErrors_V = getTestErrors(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
    yLF_V_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
    yHF_V_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_V_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_V_All[:, (nHF_Pilot + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + 1):end, :]);

    yUUBF_holdout_al, _, _, testErrors_UU = getTestErrors(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UU_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UU_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UU_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UU_All[:, (nHF_Pilot + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + 1):end, :]);

    yUWBF_holdout_al, _, _, testErrors_UW = getTestErrors(xi_LF_all_random[1:(nLF_Pilot + bID * batch_size), :],
    yLF_UW_all_random[:, 1:(nLF_Pilot + bID * batch_size)],
    HFLF_ID_all[1:(nHF_Pilot + bID * batch_size)],
    xi_HF_all_random[1:(nHF_Pilot + bID * batch_size), :],
    yHF_UW_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    yDelta_UW_all_random[:, 1:(nHF_Pilot + bID * batch_size)],
    xbyD; type="hforacle",
    hf_oracle=yHF_UW_All[:, (nHF_Pilot + 1):end],
    test_points=xi_HF_all[(nHF_Pilot + 1):end, :]);


    mean_test_errs = mean([testErrors_V testErrors_UU testErrors_UW], dims=2)
    push!(all_batch_holdout_al_errs, mean_test_errs)
end

holdout_al_errs = zeros(size(yHF_V_All, 2) - nHF_Pilot, n_batches_random + 1)
for bID in 0:(n_batches_random)
    holdout_al_errs[:, bID + 1] = all_batch_holdout_al_errs[bID + 1]
end
npzwrite("/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/data/HeatmapErrDataHoldout_AL_SurrRS.npy", holdout_al_errs)
