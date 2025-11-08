# Other utility functions to generate waveforms, calculate errors, build surrogate with cross-validation strategies etc.
using Random
using LinearAlgebra

"""
Function that takes in two parameters, a and b and returns a spiked waveform 
of the form exp(-ax)*sin(bx)
"""
function spikedWaveform(x, a, b)
    return exp.(-a * x) .* sin.(b * x)
end

"""
Function that returns Taylor series approximation of the spiked waveform exp(-ax)*sin(bx).
This acts as our low-fidelity model approximation.
"""
function taylorApprox(x, a, b)
    sinTaylor = b*x - (b^3/factorial(3))*x.^3 + (b^5/factorial(5))*x.^5;
    return exp.(-a*x) .* sinTaylor
end

"""
Function that returns a degraded version of Bhaskara's sine approximation on the interval [0, π]
"""
function bSineApprox(x, a, b)
    bx_d = x*(180/pi).*b
    return exp.(-a .* x) .* (3.5 * bx_d .* (180 .- bx_d)) ./ (15000 .- bx_d .* (180 .- bx_d))
end

function generateLF(x, inputsLF; chosen_lf="taylor")
    ng = length(x)
    nData = size(inputsLF, 1)

    LF_raw_data = zeros(ng, nData)

    # assume correct inputsLF have already been loaded in the main() function below.
    # if loading scaled inputs, convert first before passing to data generating function.

    for i in 1:nData
        if chosen_lf == "taylor"
            LF_raw_data[:, i] = taylorApprox(x, inputsLF[i, 1], inputsLF[i, 2])
        
        elseif chosen_lf == "bSine"
            LF_raw_data[:, i] = bSineApprox(x, inputsLF[i, 1], inputsLF[i, 2])
        end
    end

    return LF_raw_data
end

function generateHF(x, inputsHF)
    ng = length(x)
    nData = size(inputsHF, 1)

    HF_raw_data = zeros(ng, nData)

    # assume correct inputsHF have already been loaded in the main() function below.
    # if loading scaled inputs, convert first before passing to data generating function.

    for i in 1:nData
        HF_raw_data[:, i] = spikedWaveform(x, inputsHF[i, 1], inputsHF[i, 2])
    end

    return HF_raw_data
end

ϵ2(y1::Vector, y2::Vector) = norm((y1 - y2)) / norm(y1)
ϵ1(y1::Vector, y2::Vector) = norm((y1 - y2), 1) / norm(y1, 1)
ϵAbs2(y1::Vector, y2::Vector) = norm((y1 - y2))
ϵAbs1(y1::Vector, y2::Vector) = norm((y1 - y2), 1)

mutable struct KLEObjectSingle
    Ym
    Q
    λ
    bβ
    reg

    KLEObjectSingle(Ym, Q, λ, bβ, reg) = new(Ym, Q, λ, bβ, reg)
end

mutable struct KLEObject
    YmLF
    QLF
    λLF
    bβLF
    regLF
    YmDelta
    QDelta
    λDelta
    bβDelta
    regDelta

    KLEObject(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta) = new(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta)
end

mutable struct KLEObjectAll
    YmLF
    QLF
    λLF
    bβLF
    regLF
    YmDelta
    QDelta
    λDelta
    bβDelta
    regDelta
    YmHF
    QHF
    λHF
    bβHF
    regHF

    KLEObjectAll(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, QHF, λHF, bβHF, regHF) = new(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, QHF, λHF, bβHF, regHF)
end

"""
Return the train set indices and the holdout set indices for each of the k folds that we randomly partition our original pool into.
"""
function k_folds(a, k; rng_gen=MersenneTwister(202410))
    # Shuffle the indices for randomness
    indices = collect(1:length(a))
    Random.shuffle!(rng_gen, indices)

    # Determine fold sizes
    fold_size = div(length(a), k)
    remainder = mod(length(a), k)
    
    folds = []
    start_idx = 1

    for i in 1:k
        # Adjust fold size for uneven splits
        current_fold_size = fold_size + (i <= remainder ? 1 : 0)
        end_idx = start_idx + current_fold_size - 1

		holdout_indices = indices[start_idx:end_idx]
		train_indices = indices[setdiff(1:length(a), start_idx:end_idx)]
		
        push!(folds, (train_indices, holdout_indices))
        start_idx = end_idx + 1
    end

    return folds
end

function k_folds_expanded(a, k; rng_gen=MersenneTwister(202410))
    n = length(a)

    # allow symbolic request for leave-one-out
    if k === :loo || k === :leave_one_out
        k = n
    end

    # validate k
    if !(isa(k, Integer) && 1 <= k <= n)
        throw(ArgumentError("k must be an integer in 1:length(a) or :loo / :leave_one_out"))
    end

    # Shuffle the indices for randomness
    indices = collect(1:n)
    Random.shuffle!(rng_gen, indices)

    # Determine fold sizes (distribute remainder to first `remainder` folds)
    fold_size = div(n, k)
    remainder = rem(n, k)

    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    start_idx = 1

    for i in 1:k
        current_fold_size = fold_size + (i <= remainder ? 1 : 0)
        end_idx = start_idx + current_fold_size - 1

        holdout_indices = indices[start_idx:end_idx]

        # build train indices from the shuffled indices while handling edge slices
        left = start_idx > 1 ? indices[1:start_idx-1] : Int[]
        right = end_idx < n ? indices[end_idx+1:end] : Int[]
        train_indices = vcat(left, right)

        push!(folds, (train_indices, holdout_indices))
        start_idx = end_idx + 1
    end

    return folds
end

function generateOracleData(grid_a, grid_b, x_data)
    ng = length(x_data)

    # array of zeros for HF oracle only I guess.
    # LF_oracle = zeros(length(grid_a), length(grid_b), ng)
    HF_oracle = zeros(length(grid_a), length(grid_b), ng)

    for i in 1:length(grid_a)
        for j in 1:length(grid_b)
            # LF_oracle[i, j, :] = taylorApprox(x_data, grid_a[i], grid_b[j])
            HF_oracle[i, j, :] = spikedWaveform(x_data, grid_a[i], grid_b[j])
        end
    end

    # return LF_oracle, HF_oracle
    return HF_oracle
end

function generateOracleDataLF(grid_a, grid_b, x_data; chosen_case=1)
    ng = length(x_data)

    # array of zeros for HF oracle only I guess.
    # LF_oracle = zeros(length(grid_a), length(grid_b), ng)
    LF_oracle = zeros(length(grid_a), length(grid_b), ng)

    for i in 1:length(grid_a)
        for j in 1:length(grid_b)
            if chosen_case == 2
                LF_oracle[i, j, :] = taylorApprox(x_data, grid_a[i], grid_b[j])
            elseif chosen_case == 1
                LF_oracle[i, j, :] = bSineApprox(x_data, grid_a[i], grid_b[j])
            end
        end
    end

    return LF_oracle
end


function evaluateKLE(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; 
    all_folds=nothing, 
    useAbsErr=0, 
    grid_a_scaled=nothing, 
    grid_b_scaled=nothing)

	yPredicted = zeros(size(yLF_data, 1), size(xi_HF, 1))

    YmLF_all = []
    QLF_all = []
	λLF_all = []
	bβLF_all = []
	regLF_all = []

    YmDelta_all = []
	QDelta_all = []
	λDelta_all = []
	bβDelta_all = []
	regDelta_all = []

	# @assert length(HF_LF_ID) == size(xi_HF, 1)

	ng = length(grid)
		
	# Loop for general as well as LOO cross-validation model building
	# all_folds = k_folds(HF_LF_ID, nFolds)

	for (idx, (train_indices, holdout_indices)) in enumerate(all_folds)
		# println(holdout_indices)
		holdout_HF_ID = HF_LF_ID[holdout_indices]
		
		QLF, λLF, bβLF, regLF, YMeanLF = buildKLE(xi_LF[setdiff(1:end, holdout_HF_ID), :], yLF_data[:, setdiff(1:end, holdout_HF_ID)], grid; kle_kwargs...)
		QDelta, λDelta, bβDelta, regDelta, YMeanDelta = buildKLE(xi_HF[setdiff(1:end, holdout_indices), :], yDelta_data[:, setdiff(1:end, holdout_indices)], grid; kle_kwargs_Δ...)

		if length(holdout_indices) == 1
			ΨTest_LF = PrepCaseA(xi_LF[holdout_HF_ID[1], :]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
			ΨTest_Δ = PrepCaseA(xi_HF[holdout_indices[1], :]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
		else
			ΨTest_LF = PrepCaseA(xi_LF[holdout_HF_ID, :]; order=kle_kwargs.order, dims=kle_kwargs.dims)'
			ΨTest_Δ = PrepCaseA(xi_HF[holdout_indices, :]; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
		end
		# println(size(ΨTest_LF))
		# println(size(ΨTest_Δ))

		klModes_LF = QLF .* sqrt.(λLF)'
		klModes_Δ  = QDelta .* sqrt.(λDelta)'

		if length(holdout_indices) == 1
			yPredicted[:, holdout_indices[1]] = (klModes_LF * bβLF * ΨTest_LF) .+ YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) .+ YMeanDelta
		else
			yPredicted[:, holdout_indices] = (klModes_LF * bβLF * ΨTest_LF) .+ YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) .+ YMeanDelta
		end

		push!(YmLF_all, YMeanLF)
        push!(QLF_all, QLF)
		push!(λLF_all, λLF)
		push!(bβLF_all, bβLF)
		push!(regLF_all, regLF)

        push!(YmDelta_all, YMeanDelta)
		push!(QDelta_all, QDelta)
		push!(λDelta_all, λDelta)
		push!(bβDelta_all, bβDelta)
		push!(regDelta_all, regDelta)
	end
	

    LOOErrors = []
    
    if useAbsErr==1
        LOOErrors = [ϵAbs1(yHF_data[:, i], yPredicted[:, i]) for i in 1:size(xi_HF, 1)]
	else
        LOOErrors = [ϵ1(yHF_data[:, i], yPredicted[:, i]) for i in 1:size(xi_HF, 1)]
    end

	# Also generate predictions from the full model at multiple test points
	QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle = buildKLE(xi_LF, yLF_data, grid; kle_kwargs...)
	QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle = buildKLE(xi_HF, yDelta_data, grid; kle_kwargs_Δ...)	

	# predict on grid
	BF_oracle = zeros(length(grid_a_scaled), length(grid_b_scaled), ng)
	oracle_errors = zeros(length(grid_a_scaled), length(grid_b_scaled))
	
	for i in 1:length(grid_a_scaled)
		for j in 1:length(grid_b_scaled)
			ΨTest_LF_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
		    ΨTest_Δ_oracle = PrepCaseA([grid_a_scaled[i], grid_b_scaled[j]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

			klModes_LF_oracle = QLF_oracle .* sqrt.(λLF_oracle)'
			klModes_Δ_oracle = QDelta_oracle .* sqrt.(λDelta_oracle)'

			BF_oracle[i, j, :] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle + (klModes_Δ_oracle * bβDelta_oracle * ΨTest_Δ_oracle) + YMeanDelta_oracle
			if useAbsErr == 1
				oracle_errors[i, j] = ϵAbs1(HF_oracle[i, j, :], BF_oracle[i, j, :])
			else
				oracle_errors[i, j] = ϵ1(HF_oracle[i, j, :], BF_oracle[i, j, :])
			end
		end
	end
	
    kleObject = KLEObject(YmLF_all, QLF_all, λLF_all, bβLF_all, regLF_all, YmDelta_all, QDelta_all, λDelta_all, bβDelta_all, regDelta_all)
	# return (yPredicted, LOOErrors), (BF_oracle, oracle_errors), kleObject
    return LOOErrors, oracle_errors, kleObject
end


function predictOnGrid(QLF, λLF, bβLF, regLF, YMeanLF, QDelta, λDelta, bβDelta, regDelta, YMeanDelta, theta_pred, x_data)
    # QLF = kleParams["QLF"]
    # λLF = kleParams["λLF"]
    # bβLF = kleParams["bβLF"]
    # regLF = kleParams["regLF"]
    # YMeanLF = kleParams["YMeanLF"]

    # QDelta = kleParams["QDelta"]
    # λDelta = kleParams["λDelta"]
    # bβDelta = kleParams["bβDelta"]
    # regDelta = kleParams["regDelta"]
    # YMeanDelta = kleParams["YMeanDelta"]

    ng = length(x_data)

	BFPredictions = zeros(ng, size(theta_pred, 1))

    for i in 1:size(theta_pred, 1)
		ΨTest_LF = PrepCaseA([theta_pred[i, 1], theta_pred[i, 2]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
		ΨTest_Δ = PrepCaseA([theta_pred[i, 1], theta_pred[i, 2]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

		klModes_LF = QLF .* sqrt.(λLF)'
		klModes_Δ  = QDelta .* sqrt.(λDelta)'

		BFPredictions[:, i] = (klModes_LF * bβLF * ΨTest_LF) + YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) + YMeanDelta
    end

    return BFPredictions
end

function extend_vector(base, extras)
    # First half = base plus extras
    first_half = vcat(base, extras)
    lbase_extended = extras[1] - 1
    total_shift = 2 * lbase_extended + length(extras)
    shift_idx = [total_shift + i for i in 1:length(extras)]

    second_half = vcat(base, shift_idx)

    return second_half
end