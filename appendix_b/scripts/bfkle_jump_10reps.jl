using LinearAlgebra
using NPZ
using Printf
using Statistics
using Random

const appendix_dir = normpath(joinpath(@__DIR__, ".."))
const project_root = normpath(joinpath(appendix_dir, ".."))
const deeponet_dir = joinpath(project_root, "deeponet_comparisons")

include(joinpath(project_root, "1d_toy", "kleUtils.jl"))
include(joinpath(deeponet_dir, "utils_jump.jl"))

const data_dir = joinpath(appendix_dir, "data")
const NREPS = 10

kle_kwargs = (order=6, dims=1, family="Legendre", useFullGrid=1, getAllModes=0,
              weightFunction=getWeights, solver="Tikhonov-L2")
kle_kwargs_Δ = kle_kwargs
kle_kwargs_HF = kle_kwargs

bf_errors = zeros(NREPS)
sf_errors = zeros(NREPS)
lf_term_errors = zeros(NREPS)
correction_term_errors = zeros(NREPS)

for rep in 1:NREPS
    npz_file = joinpath(data_dir, @sprintf("jump_bifi_rep%d.npz", rep))
    d = npzread(npz_file)

    x = d["x"]
    a_grid = d["a_grid"]
    lb, ub = d["lb"], d["ub"]
    xi_LF_scaled = d["xi_LF_scaled"]
    xi_HF_scaled = d["xi_HF_scaled"]
    HF_idx_in_LF = round.(Int, d["HF_idx_in_LF"])
    LF_data = d["LF_data"]
    HF_data = d["HF_data"]
    LF_oracle = d["LF_oracle"]
    HF_oracle = d["HF_oracle"]

    N_ORACLE = length(a_grid)
    a_grid_scaled = 2.0 .* (a_grid .- 0.5 * (lb + ub)) ./ (ub - lb)
    xi_oracle_scaled = reshape(a_grid_scaled, :, 1)

    QLF, λLF, bβLF, _, YMeanLF = buildKLE(xi_LF_scaled, LF_data, x; kle_kwargs...)
    klModes_LF = QLF .* sqrt.(λLF)'
    ΨLF = PrepCaseA(xi_oracle_scaled; order=kle_kwargs.order, dims=kle_kwargs.dims)'
    y_pred_LF = klModes_LF * bβLF * ΨLF .+ YMeanLF

    Delta_data = HF_data .- LF_data[:, HF_idx_in_LF]
    QΔ, λΔ, bβΔ, _, YMeanΔ = buildKLE(xi_HF_scaled, Delta_data, x; kle_kwargs_Δ...)
    klModes_Δ = QΔ .* sqrt.(λΔ)'
    ΨΔ = PrepCaseA(xi_oracle_scaled; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'
    y_pred_Δ = klModes_Δ * bβΔ * ΨΔ .+ YMeanΔ

    y_pred_BF = y_pred_LF .+ y_pred_Δ

    QHF, λHF, bβHF, _, YMeanHF = buildKLE(xi_HF_scaled, HF_data, x; kle_kwargs_HF...)
    klModes_HF = QHF .* sqrt.(λHF)'
    ΨHF = PrepCaseA(xi_oracle_scaled; order=kle_kwargs_HF.order, dims=kle_kwargs_HF.dims)'
    y_pred_HF = klModes_HF * bβHF * ΨHF .+ YMeanHF

    bf_errors[rep] = mean(ϵ2_jump(HF_oracle[:, k], y_pred_BF[:, k]) for k in 1:N_ORACLE)
    sf_errors[rep] = mean(ϵ2_jump(HF_oracle[:, k], y_pred_HF[:, k]) for k in 1:N_ORACLE)
    lf_term_errors[rep] = mean(ϵ2_jump(LF_oracle[:, k], y_pred_LF[:, k]) for k in 1:N_ORACLE)
    correction_term_errors[rep] = mean(
        ϵ2_jump(HF_oracle[:, k] .- LF_oracle[:, k], y_pred_Δ[:, k]) for k in 1:N_ORACLE)

    @assert all(isfinite, y_pred_BF) "rep $rep: BF prediction contains NaN/Inf"
    @assert all(isfinite, y_pred_HF) "rep $rep: SF-KLE prediction contains NaN/Inf"

    println(@sprintf("rep %d/%d: BF=%.4e  SF=%.4e  LF-term=%.4e  correction-term=%.4e",
                      rep, NREPS, bf_errors[rep], sf_errors[rep],
                      lf_term_errors[rep], correction_term_errors[rep]))
end

println(@sprintf("BF-KLE:  mean=%.4e  std=%.4e", mean(bf_errors), std(bf_errors)))
println(@sprintf("SF-KLE:  mean=%.4e  std=%.4e", mean(sf_errors), std(sf_errors)))
println(@sprintf("LF-term: mean=%.4e  std=%.4e", mean(lf_term_errors), std(lf_term_errors)))
println(@sprintf("Correction-term: mean=%.4e  std=%.4e",
                  mean(correction_term_errors), std(correction_term_errors)))

out_file = joinpath(data_dir, "results_bfkle_10reps.npz")
npzwrite(out_file,
    Dict("bf_errors" => bf_errors,
         "sf_errors" => sf_errors,
         "lf_term_errors" => lf_term_errors,
         "correction_term_errors" => correction_term_errors))
println(@sprintf("Saved %s", out_file))
