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
using CSV
using MAT
using DataFrames
using DelimitedFiles
using Printf
using Statistics

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

# function to get test errors (just relative difference no LOO for each batch)
function getTestErrors(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; type="hfpred", hf_oracle=nothing)
	yPredicted = zeros(size(yLF_data, 1), size(test_points, 1))
	yHFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	yLFPred = zeros(size(yLF_data, 1), size(test_points, 1))
	
	QLF, λLF, bβLF, regLF, YMeanLF = buildKLE(xi_LF, yLF_data, grid; kle_kwargs...)
	QDelta, λDelta, bβDelta, regDelta, YMeanDelta = buildKLE(xi_HF, yDelta_data, grid; kle_kwargs_Δ...)
	QHF, λHF, bβHF, regHF, YMeanHF = buildKLE(xi_HF_all, yHF_data, grid; kle_kwargs...)
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
        elseif type == "hforacle" # where true HF data is available.
            push!(testErrors, ϵ2(hf_oracle[:, i], yPredicted[:, i]))
        end
	end
	return yPredicted, yLFPred, yHFPred, testErrors
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
n_batch_total = n_batches + 1
batch_size = 5



# _, _, _, testErrors_V = getTestErrors()