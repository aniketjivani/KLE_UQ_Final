# Other utility functions to generate waveforms, calculate errors, build surrogate with cross-validation strategies etc.
using Random
using LinearAlgebra
using PyCall
spi = sp.interpolate

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

mutable struct KLEObjectStd
    YmLF
    YstdLF
    QLF
    λLF
    bβLF
    regLF
    YmDelta
    YstdDelta
    QDelta
    λDelta
    bβDelta
    regDelta

    KLEObjectStd(YmLF, YstdLF, QLF, λLF, bβLF, regLF, YmDelta, YstdDelta, QDelta, λDelta, bβDelta, regDelta) = new(YmLF, YstdLF, QLF, λLF, bβLF, regLF, YmDelta, YstdDelta, QDelta, λDelta, bβDelta, regDelta)
end

mutable struct KLEObjectAllStd
    YmLF
    YstdLF
    QLF
    λLF
    bβLF
    regLF
    YmDelta
    YstdDelta
    QDelta
    λDelta
    bβDelta
    regDelta
    YmHF
    YstdHF
    QHF
    λHF
    bβHF
    regHF

    # KLEObjectAll(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, QHF, λHF, bβHF, regHF) = new(YmLF, QLF, λLF, bβLF, regLF, YmDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, QHF, λHF, bβHF, regHF)
    KLEObjectAllStd(YmLF, YstdLF, QLF, λLF, bβLF, regLF, YmDelta, YstdDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, YstdHF, QHF, λHF, bβHF, regHF) = new(YmLF, YstdLF, QLF, λLF, bβLF, regLF, YmDelta, YstdDelta, QDelta, λDelta, bβDelta, regDelta, YmHF, YstdHF, QHF, λHF, bβHF, regHF)
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

"""
for 2D evaluateKLE, supply HF oracle data generated via space-filling design. Also build HF KLE oracle and save oracle objects.
"""
function evaluateKLE(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; 
    all_folds=nothing, 
    useAbsErr=0,
    HF_oracle_data=nothing,
    HF_oracle_design=nothing)

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

	ng = length(grid)^2
		
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
    QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle = buildKLE(xi_HF, yHF_data, grid; kle_kwargs_HF...)



	# predict on grid
	BF_oracle = zeros(ng, size(HF_oracle_design, 1))
    HF_KLE_oracle = zeros(ng, size(HF_oracle_design, 1))
	oracle_errors = zeros(size(HF_oracle_design, 1))
    oracle_errors_HF = zeros(size(HF_oracle_design, 1))
	
    nOracle = size(HF_oracle_design, 1)
    for i in 1:nOracle
        ΨTest_LF_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
        ΨTest_Δ_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
        ΨTest_HF_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'

        klModes_LF_oracle = QLF_oracle .* sqrt.(λLF_oracle)'
        klModes_Δ_oracle = QDelta_oracle .* sqrt.(λDelta_oracle)'
        klModes_HF_oracle = QHF_oracle .* sqrt.(λHF_oracle)'

        BF_oracle[:, i] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle) + YMeanLF_oracle + (klModes_Δ_oracle * bβDelta_oracle * ΨTest_Δ_oracle) + YMeanDelta_oracle

        HF_KLE_oracle[:, i] = (klModes_HF_oracle * bβHF_oracle * ΨTest_HF_oracle) + YMeanHF_oracle

        if useAbsErr == 1
            oracle_errors[i] = ϵAbs1(HF_oracle_data[:, i], BF_oracle[:, i])
            oracle_errors_HF[i] = ϵAbs1(HF_oracle_data[:, i], HF_KLE_oracle[:, i])
        else
            oracle_errors[i] = ϵ1(HF_oracle_data[:, i], BF_oracle[:, i])
            oracle_errors_HF[i] = ϵ1(HF_oracle_data[:, i], HF_KLE_oracle[:, i])
        end
    end
	
    kleObject = KLEObject(YmLF_all, QLF_all, λLF_all, bβLF_all, regLF_all, YmDelta_all, QDelta_all, λDelta_all, bβDelta_all, regDelta_all)

    kle_oracle = KLEObjectAll(YMeanLF_oracle, QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanDelta_oracle, QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanHF_oracle, QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle)

    return LOOErrors, oracle_errors, oracle_errors_HF, kleObject, kle_oracle
end

"""
for 2D evaluateKLE, supply HF oracle data generated via space-filling design. Also build HF KLE oracle and save oracle objects. Optionally, divide by std dev before building KLEs.
"""
function evaluateKLEStandardized(xi_LF, yLF_data, HF_LF_ID, xi_HF, yHF_data, yDelta_data, grid; 
    all_folds=nothing, 
    useAbsErr=0,
    HF_oracle_data=nothing,
    HF_oracle_design=nothing)
	yPredicted = zeros(size(yLF_data, 1), size(xi_HF, 1))
    YmLF_all = []
    YstdLF_all = []
    QLF_all = []
	λLF_all = []
	bβLF_all = []
	regLF_all = []
    YmDelta_all = []
    YstdDelta_all = []
	QDelta_all = []
	λDelta_all = []
	bβDelta_all = []
	regDelta_all = []
	# @assert length(HF_LF_ID) == size(xi_HF, 1)
	ng = length(grid)^2
		
	# Loop for general as well as LOO cross-validation model building
	# all_folds = k_folds(HF_LF_ID, nFolds)

	for (idx, (train_indices, holdout_indices)) in enumerate(all_folds)
		# println(holdout_indices)
		holdout_HF_ID = HF_LF_ID[holdout_indices]
		
		QLF, λLF, bβLF, regLF, YMeanLF, YStdLF = buildKLEStandardized(xi_LF[setdiff(1:end, holdout_HF_ID), :], yLF_data[:, setdiff(1:end, holdout_HF_ID)], grid; kle_kwargs...)
		QDelta, λDelta, bβDelta, regDelta, YMeanDelta, YStdDelta = buildKLEStandardized(xi_HF[setdiff(1:end, holdout_indices), :], yDelta_data[:, setdiff(1:end, holdout_indices)], grid; kle_kwargs_Δ...)

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
            yPredicted[:, holdout_indices[1]] = (klModes_LF * bβLF * ΨTest_LF) .* YStdLF .+ YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) .* YStdDelta .+ YMeanDelta
		else
			yPredicted[:, holdout_indices] = (klModes_LF * bβLF * ΨTest_LF) .* YStdLF .+ YMeanLF + (klModes_Δ * bβDelta * ΨTest_Δ) .* YStdDelta .+ YMeanDelta
		end

		push!(YmLF_all, YMeanLF)
        push!(YstdLF_all, YStdLF)
        push!(QLF_all, QLF)
		push!(λLF_all, λLF)
		push!(bβLF_all, bβLF)
		push!(regLF_all, regLF)

        push!(YmDelta_all, YMeanDelta)
        push!(YstdDelta_all, YStdDelta)
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
	QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanLF_oracle, YStdLF_oracle = buildKLEStandardized(xi_LF, yLF_data, grid; kle_kwargs...)
	QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanDelta_oracle, YStdDelta_oracle = buildKLEStandardized(xi_HF, yDelta_data, grid; kle_kwargs_Δ...)
    QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle, YMeanHF_oracle, YStdHF_oracle = buildKLEStandardized(xi_HF, yHF_data, grid; kle_kwargs_HF...)


	# predict on grid
	BF_oracle = zeros(ng, size(HF_oracle_design, 1))
    HF_KLE_oracle = zeros(ng, size(HF_oracle_design, 1))
	oracle_errors = zeros(size(HF_oracle_design, 1))
    oracle_errors_HF = zeros(size(HF_oracle_design, 1))
	
    nOracle = size(HF_oracle_design, 1)
    for i in 1:nOracle
        ΨTest_LF_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs.order, dims=kle_kwargs.dims)'
        ΨTest_Δ_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
        ΨTest_HF_oracle = PrepCaseA([HF_oracle_design[i, 1], HF_oracle_design[i, 2], HF_oracle_design[i, 3], HF_oracle_design[i, 4]]'; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'

        klModes_LF_oracle = QLF_oracle .* sqrt.(λLF_oracle)'
        klModes_Δ_oracle = QDelta_oracle .* sqrt.(λDelta_oracle)'
        klModes_HF_oracle = QHF_oracle .* sqrt.(λHF_oracle)'

        BF_oracle[:, i] = (klModes_LF_oracle * bβLF_oracle * ΨTest_LF_oracle).* YStdLF_oracle + YMeanLF_oracle + (klModes_Δ_oracle * bβDelta_oracle * ΨTest_Δ_oracle).* YStdDelta_oracle + YMeanDelta_oracle

        HF_KLE_oracle[:, i] = (klModes_HF_oracle * bβHF_oracle * ΨTest_HF_oracle).*YStdHF_oracle + YMeanHF_oracle

        if useAbsErr == 1
            oracle_errors[i] = ϵAbs1(HF_oracle_data[:, i], BF_oracle[:, i])
            oracle_errors_HF[i] = ϵAbs1(HF_oracle_data[:, i], HF_KLE_oracle[:, i])
        else
            oracle_errors[i] = ϵ1(HF_oracle_data[:, i], BF_oracle[:, i])
            oracle_errors_HF[i] = ϵ1(HF_oracle_data[:, i], HF_KLE_oracle[:, i])
        end
    end
	
    kleObject = KLEObjectStd(YmLF_all, YstdLF_all, QLF_all, λLF_all, bβLF_all, regLF_all, YmDelta_all, YstdDelta_all, QDelta_all, λDelta_all, bβDelta_all, regDelta_all)
    kle_oracle = KLEObjectAllStd(YMeanLF_oracle, YStdLF_oracle, QLF_oracle, λLF_oracle, bβLF_oracle, regLF_oracle, YMeanDelta_oracle, YStdDelta_oracle, QDelta_oracle, λDelta_oracle, bβDelta_oracle, regDelta_oracle, YMeanHF_oracle, YStdHF_oracle, QHF_oracle, λHF_oracle, bβHF_oracle, regHF_oracle)

    return LOOErrors, oracle_errors, oracle_errors_HF, kleObject, kle_oracle
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

function readAndInterpolateHFFromArray(HF_array, gridQuantitiesHF, gridQuantitiesLF;nxHF=64, nyHF=64, nxLF=32, nyLF=32)
    m, n = size(HF_array)
    @assert nxHF == m
    @assert nyHF == n
    
    xmHF = gridQuantitiesHF[3]
    ymHF = gridQuantitiesHF[4]
    XMLF = gridQuantitiesLF[5]
    YMLF = gridQuantitiesLF[6]

    nRuns = 1

    HF_raw_data = HF_array[:]
    interpHF = spi.RegularGridInterpolator((xmHF, xmHF), HF_array)

    HF_interp_data = interpHF((XMLF, YMLF))'[:]
    return HF_raw_data, HF_interp_data
end


function readAndInterpolateHF(HF_data_dir, nPilot, batch_num, rep_num, gridQuantitiesHF, gridQuantitiesLF;nxHF=64, nyHF=64, nxLF=32, nyLF=32)
    sample_file = joinpath(HF_data_dir, "HF_2D_Run0001.txt")
    sample_data = readdlm(sample_file)
    m, n = size(sample_data)
    @assert nxHF == m
    @assert nyHF == n

    xmHF = gridQuantitiesHF[3]
    ymHF = gridQuantitiesHF[4]
    XMLF = gridQuantitiesLF[5]
    YMLF = gridQuantitiesLF[6]

    HF_raw_data = zeros(nxHF * nyHF, nPilot + batch_num)
    HF_interp_data = zeros(nxLF * nyLF, nPilot + batch_num)

    for i in 1:(nPilot + batch_num)
        filename_i = []
        if i <= nPilot
            filename_i = joinpath(HF_data_dir, "HF_2D_Run" * @sprintf("%04d", i) * ".txt")
        else
            filename_i = joinpath(HF_data_dir, 
                                @sprintf("rep_%03d", rep_num),
            "HF_2D_Run" * @sprintf("%04d", i) * ".txt")
        end
        # filename_i = joinpath(HF_data_dir, "HF_2D_Run" * @sprintf("%04d", i) * ".txt")

        HF_raw_data[:, i] = readdlm(filename_i)[:]
        interpHF = spi.RegularGridInterpolator((xmHF, xmHF), readdlm(filename_i))

        HF_interp_data[:, i] = interpHF((XMLF, YMLF))'[:]

    end

    return HF_raw_data, HF_interp_data
end

function readHFFromDir(HF_data_dir, gridQuantitiesHF, gridQuantitiesLF;nxHF=64, nyHF=64, nxLF=32, nyLF=32)
    sample_file = joinpath(HF_data_dir, "HF_2D_Run0001.txt")
    sample_data = readdlm(sample_file)
    m, n = size(sample_data)
    @assert nxHF == m
    @assert nyHF == n

    xmHF = gridQuantitiesHF[3]
    ymHF = gridQuantitiesHF[4]
    XMLF = gridQuantitiesLF[5]
    YMLF = gridQuantitiesLF[6]

    nRuns = length(readdir(HF_data_dir))

    HF_raw_data = zeros(nxHF * nyHF, nRuns)
    HF_interp_data = zeros(nxLF * nyLF, nRuns)

    for i in 1:nRuns
        filename_i = joinpath(HF_data_dir,
            "HF_2D_Run" * @sprintf("%04d", i) * ".txt")

        HF_raw_data[:, i] = readdlm(filename_i)[:]
        interpHF = spi.RegularGridInterpolator((xmHF, xmHF), readdlm(filename_i))

        HF_interp_data[:, i] = interpHF((XMLF, YMLF))'[:]

    end
    return HF_raw_data, HF_interp_data
end

function readLFFromDir(LF_data_dir, gridQuantitiesHF, gridQuantitiesLF;nxHF=64, nyHF=64, nxLF=32, nyLF=32)
    sample_file = joinpath(LF_data_dir, "LF_2D_Run0001.txt")
    sample_data = readdlm(sample_file)
    m, n = size(sample_data)
    @assert nxLF == m
    @assert nyLF == n

    nRuns = length(readdir(LF_data_dir))
    LF_raw_data = zeros(nxLF * nyLF, nRuns)

    for i in 1:nRuns
        filename_i = joinpath(LF_data_dir,
            "LF_2D_Run" * @sprintf("%04d", i) * ".txt")

        LF_raw_data[:, i] = readdlm(filename_i)[:]
    end
    return LF_raw_data
end

function readLF(LF_data_dir, nPilot, batch_num, rep_num; nxLF=32, nyLF=32)
    sample_file = joinpath(LF_data_dir, "LF_2D_Run0001.txt")
    sample_data = readdlm(sample_file)
    m, n = size(sample_data)
    @assert nxLF == m
    @assert nyLF == n

    LF_raw_data = zeros(nxLF * nyLF, nPilot + batch_num)

    for i in 1:(nPilot + batch_num)
        filename_i = []
        if i <= nPilot
            filename_i = joinpath(LF_data_dir, "LF_2D_Run" * @sprintf("%04d", i) * ".txt")
        else
            filename_i = joinpath(LF_data_dir, 
            @sprintf("rep_%03d", rep_num),
            "LF_2D_Run" * @sprintf("%04d", i) * ".txt")
        end
        # filename_i = joinpath(LF_data_dir, "LF_2D_Run" * @sprintf("%04d", i) * ".txt")

        LF_raw_data[:, i] = readdlm(filename_i)[:]
    end

    return LF_raw_data
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

function extend_vector(base, extras)
    # First half = base plus extras
    first_half = vcat(base, extras)
    lbase_extended = extras[1] - 1
    total_shift = 2 * lbase_extended + length(extras)
    shift_idx = [total_shift + i for i in 1:length(extras)]

    second_half = vcat(base, shift_idx)

    return second_half
end