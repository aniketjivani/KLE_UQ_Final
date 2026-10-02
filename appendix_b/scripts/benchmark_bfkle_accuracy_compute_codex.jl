"""
Benchmark the offline fitting time of the BF-KLE surrogate used in the
jump-function comparison.

Only the LF and discrepancy surrogate fits are timed. Data loading,
single-fidelity fitting, oracle prediction, error calculation, and result I/O
are deliberately outside the timer.

Example
-------
julia --project=bifi_regime_study \
    deeponet_comparisons/base_experiment/benchmark_bfkle_accuracy_compute_codex.jl

Optional arguments
------------------
--reps N             Number of data replications, default 10.
--timing-repeats N   Repeated fits per data replication, default 3.
--blas-threads N     BLAS threads, default 12.
--out PATH           Output NPZ path.
"""

using LinearAlgebra
using NPZ
using Printf
using Random
using Statistics

const APPENDIX_DIR = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = normpath(joinpath(APPENDIX_DIR, ".."))
const DEEPONET_DIR = joinpath(PROJECT_ROOT, "deeponet_comparisons")

include(joinpath(PROJECT_ROOT, "1d_toy", "kleUtils.jl"))
include(joinpath(DEEPONET_DIR, "utils_jump.jl"))

const DATA_DIR = joinpath(APPENDIX_DIR, "data")
const DEFAULT_RESULT_DIR = joinpath(DATA_DIR, "results_accuracy_compute_codex")

function cli_value(flag::String, default, convert)
    idx = findfirst(==(flag), ARGS)
    idx === nothing && return default
    idx == length(ARGS) && error("Missing value after $flag")
    return convert(ARGS[idx + 1])
end

struct BFKLEModel
    Q_LF
    lambda_LF
    beta_LF
    mean_LF
    Q_delta
    lambda_delta
    beta_delta
    mean_delta
end

function fit_bfkle(d, kle_kwargs)
    x = d["x"]
    xi_LF = d["xi_LF_scaled"]
    xi_HF = d["xi_HF_scaled"]
    LF_data = d["LF_data"]
    HF_data = d["HF_data"]
    HF_idx_in_LF = round.(Int, d["HF_idx_in_LF"])

    Q_LF, lambda_LF, beta_LF, _, mean_LF =
        buildKLE(xi_LF, LF_data, x; kle_kwargs...)

    delta_data = HF_data .- LF_data[:, HF_idx_in_LF]
    Q_delta, lambda_delta, beta_delta, _, mean_delta =
        buildKLE(xi_HF, delta_data, x; kle_kwargs...)

    return BFKLEModel(
        Q_LF, lambda_LF, beta_LF, mean_LF,
        Q_delta, lambda_delta, beta_delta, mean_delta,
    )
end

function predict_component(xi, Q, lambda, beta, field_mean; order::Int, dims::Int)
    basis = PrepCaseA(xi; order=order, dims=dims)'
    modes = Q .* sqrt.(lambda)'
    return modes * beta * basis .+ field_mean
end

function composite_error(model::BFKLEModel, d, kle_kwargs)
    a_grid = d["a_grid"]
    lower, upper = d["lb"], d["ub"]
    a_scaled = 2.0 .* (a_grid .- 0.5 * (lower + upper)) ./ (upper - lower)
    xi_test = reshape(a_scaled, :, 1)

    y_LF = predict_component(
        xi_test, model.Q_LF, model.lambda_LF, model.beta_LF, model.mean_LF;
        order=kle_kwargs.order, dims=kle_kwargs.dims,
    )
    y_delta = predict_component(
        xi_test, model.Q_delta, model.lambda_delta, model.beta_delta,
        model.mean_delta; order=kle_kwargs.order, dims=kle_kwargs.dims,
    )
    prediction = y_LF .+ y_delta
    oracle = d["HF_oracle"]

    return mean(
        ϵ2_jump(oracle[:, i], prediction[:, i]) for i in axes(oracle, 2)
    )
end

function main()
    n_reps = cli_value("--reps", 10, x -> parse(Int, x))
    timing_repeats = cli_value("--timing-repeats", 3, x -> parse(Int, x))
    blas_threads = cli_value("--blas-threads", 12, x -> parse(Int, x))
    default_out = joinpath(DEFAULT_RESULT_DIR, "bfkle_accuracy_compute_codex.npz")
    out_file = cli_value("--out", default_out, String)

    n_reps > 0 || error("--reps must be positive")
    timing_repeats > 0 || error("--timing-repeats must be positive")
    BLAS.set_num_threads(blas_threads)

    kle_kwargs = (
        order=6,
        dims=1,
        family="Legendre",
        useFullGrid=1,
        getAllModes=0,
        weightFunction=getWeights,
        solver="Tikhonov-L2",
    )

    data = [
        npzread(joinpath(DATA_DIR, @sprintf("jump_bifi_rep%d.npz", rep)))
        for rep in 1:n_reps
    ]

    cold_model = nothing
    cold_seconds = @elapsed cold_model = fit_bfkle(data[1], kle_kwargs)
    @assert cold_model !== nothing

    repeat_seconds = zeros(n_reps, timing_repeats)
    errors = zeros(n_reps)
    ranks_LF = zeros(Int, n_reps)
    ranks_delta = zeros(Int, n_reps)

    for rep in 1:n_reps
        model = nothing
        for trial in 1:timing_repeats
            GC.gc()
            repeat_seconds[rep, trial] = @elapsed begin
                model = fit_bfkle(data[rep], kle_kwargs)
            end
        end

        @assert model !== nothing
        errors[rep] = composite_error(model, data[rep], kle_kwargs)
        ranks_LF[rep] = length(model.lambda_LF)
        ranks_delta[rep] = length(model.lambda_delta)

        @printf(
            "rep %2d/%d  median fit %.6f s  error %.6e  ranks=(%d,%d)\n",
            rep, n_reps, median(repeat_seconds[rep, :]), errors[rep],
            ranks_LF[rep], ranks_delta[rep],
        )
    end

    fit_seconds = vec(median(repeat_seconds, dims=2))
    compile_estimate = max(cold_seconds - median(fit_seconds), 0.0)

    mkpath(dirname(out_file))
    npzwrite(out_file, Dict(
        "fit_seconds" => fit_seconds,
        "repeat_seconds" => repeat_seconds,
        "composite_errors" => errors,
        "ranks_LF" => ranks_LF,
        "ranks_delta" => ranks_delta,
        "cold_fit_seconds" => cold_seconds,
        "compile_estimate_seconds" => compile_estimate,
        "blas_threads" => blas_threads,
        "n_reps" => n_reps,
        "timing_repeats" => timing_repeats,
    ))

    @printf("BF-KLE warm fit median: %.6f s\n", median(fit_seconds))
    @printf("BF-KLE warm fit IQR:    [%.6f, %.6f] s\n",
            quantile(fit_seconds, 0.25), quantile(fit_seconds, 0.75))
    @printf("BF-KLE cold fit:        %.6f s\n", cold_seconds)
    println("Saved $out_file")
end

main()
