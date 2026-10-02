using LatinHypercubeSampling
using Random
using LinearAlgebra
using JLD
using NPZ
using DelimitedFiles
using Printf
using Statistics

script_dir = @__DIR__
jet_dir = normpath(joinpath(script_dir, "..", "Jet"))
data_dir = joinpath(jet_dir, "data")
old_dir = joinpath(jet_dir, "data_jet_all_old", "data")

include(joinpath(jet_dir, "kleUtils.jl"))
include(joinpath(jet_dir, "utils.jl"))

kle_kwargs = (order=3, dims=3, family="Legendre", useFullGrid=1, getAllModes=0,
              weightFunction=getWeights, solver="Tikhonov-L2")

kle_kwargs_Δ = (useFullGrid=1, getAllModes=0, order=3, dims=3,
                weightFunction=getWeights, family="Legendre", solver="Tikhonov-L2")

pilot_batch_data_og = JLD.load(joinpath(data_dir, "PilotBatchData.jld"))
xiLF_og = pilot_batch_data_og["xiLF1"]
xiHF_og = pilot_batch_data_og["xiHF1"]
xbyD    = pilot_batch_data_og["xbyD"]

xiLF_og_scaled = (xiLF_og .+ 1) ./ 2
NP_LF_new = 100
n_HF = size(xiHF_og, 1)

hf_indices_in_lf = Vector{Int}(undef, n_HF)
for j in 1:n_HF
    match_idx = findfirst(i -> all(isapprox.(xiLF_og[i, :], xiHF_og[j, :]; atol=1e-10)), 1:size(xiLF_og, 1))
    @assert !isnothing(match_idx) "No matching LF point found for HF point $j"
    hf_indices_in_lf[j] = match_idx
end
hf_indices_in_lf_kept = hf_indices_in_lf
remaining_indices = setdiff(1:size(xiLF_og, 1), hf_indices_in_lf)

rng = MersenneTwister(2026)
sampled_remaining = shuffle(rng, remaining_indices)[1:(NP_LF_new - n_HF)]
final_LF_indices = vcat(hf_indices_in_lf_kept, sampled_remaining)

xiLF_pilot = xiLF_og_scaled[final_LF_indices, :] .* 2 .- 1
xiHF_pilot = xiLF_og_scaled[hf_indices_in_lf_kept, :] .* 2 .- 1

yLFV_og  = pilot_batch_data_og["yLFV"];  yLFUU_og = pilot_batch_data_og["yLFUU"];  yLFUW_og = pilot_batch_data_og["yLFUW"]
yLFV_new  = yLFV_og[:, final_LF_indices];  yLFUU_new = yLFUU_og[:, final_LF_indices];  yLFUW_new = yLFUW_og[:, final_LF_indices]

yHFV_new  = pilot_batch_data_og["yHFV"][:, 1:n_HF]
yHFUU_new = pilot_batch_data_og["yHFUU"][:, 1:n_HF]
yHFUW_new = pilot_batch_data_og["yHFUW"][:, 1:n_HF]

yDeltaV_new  = yHFV_new  .- yLFV_new[:, 1:n_HF]
yDeltaUU_new = yHFUU_new .- yLFUU_new[:, 1:n_HF]
yDeltaUW_new = yHFUW_new .- yLFUW_new[:, 1:n_HF]

QLF_V,  λLF_V,  bβLF_V,  _, YmLF_V  = buildKLE(xiLF_pilot, yLFV_new,  xbyD; kle_kwargs...)
QLF_UU, λLF_UU, bβLF_UU, _, YmLF_UU = buildKLE(xiLF_pilot, yLFUU_new, xbyD; kle_kwargs...)
QLF_UW, λLF_UW, bβLF_UW, _, YmLF_UW = buildKLE(xiLF_pilot, yLFUW_new, xbyD; kle_kwargs...)

klModes_LF_V  = QLF_V  .* sqrt.(λLF_V)'
klModes_LF_UU = QLF_UU .* sqrt.(λLF_UU)'
klModes_LF_UW = QLF_UW .* sqrt.(λLF_UW)'

hf_all_scaled = readdlm(joinpath(data_dir, "HF_AllPoints_Scaled.txt"))
@assert isapprox(hf_all_scaled[1:15, :], xiHF_og; atol=1e-6) "Row 1:15 mismatch"

hf_data_all = JLD.load(joinpath(data_dir, "HFDataAll.jld"))
lf_data_all = JLD.load(joinpath(data_dir, "LFDataAll.jld"))
yHFVAll  = hf_data_all["yHFVAll"];  yHFUUAll = hf_data_all["yHFUUAll"]; yHFUWAll = hf_data_all["yHFUWAll"]
yLFVAll  = lf_data_all["yLFVAll"];  yLFUUAll = lf_data_all["yLFUUAll"]; yLFUWAll = lf_data_all["yLFUWAll"]

n_HF_40 = 40
xiHF_40 = hf_all_scaled[1:n_HF_40, :]
lf_match_idx_40 = vcat(hf_indices_in_lf_kept, (16:n_HF_40) .+ 185)

yDeltaV_40  = yHFVAll[:, 1:n_HF_40]  .- yLFVAll[:, lf_match_idx_40]
yDeltaUU_40 = yHFUUAll[:, 1:n_HF_40] .- yLFUUAll[:, lf_match_idx_40]
yDeltaUW_40 = yHFUWAll[:, 1:n_HF_40] .- yLFUWAll[:, lf_match_idx_40]

QΔ_V_40,  λΔ_V_40,  bβΔ_V_40,  _, YmΔ_V_40  = buildKLE(xiHF_40, yDeltaV_40,  xbyD; kle_kwargs_Δ...)
QΔ_UU_40, λΔ_UU_40, bβΔ_UU_40, _, YmΔ_UU_40 = buildKLE(xiHF_40, yDeltaUU_40, xbyD; kle_kwargs_Δ...)
QΔ_UW_40, λΔ_UW_40, bβΔ_UW_40, _, YmΔ_UW_40 = buildKLE(xiHF_40, yDeltaUW_40, xbyD; kle_kwargs_Δ...)

klModes_Δ40_V  = QΔ_V_40  .* sqrt.(λΔ_V_40)'
klModes_Δ40_UU = QΔ_UU_40 .* sqrt.(λΔ_UU_40)'
klModes_Δ40_UW = QΔ_UW_40 .* sqrt.(λΔ_UW_40)'

hf_batch_random = readdlm(joinpath(old_dir, "HFBatchRandomPoints.txt"))
n_HF_pilot_plus_b0207 = 45
xiHF_sf_70 = vcat(hf_all_scaled[1:n_HF_pilot_plus_b0207, :], hf_batch_random)
n_HF_sf = size(xiHF_sf_70, 1)
@assert n_HF_sf == 70

lf_hf_random_data = JLD.load(joinpath(old_dir, "LFHFRandomData.jld"))
yHFV_random  = lf_hf_random_data["yHFV"];  yHFUU_random = lf_hf_random_data["yHFUU"];  yHFUW_random = lf_hf_random_data["yHFUW"]
yLFV_random  = lf_hf_random_data["yLFV"];  yLFUU_random = lf_hf_random_data["yLFUU"];  yLFUW_random = lf_hf_random_data["yLFUW"]

yHFV_sf_70  = hcat(yHFVAll[:, 1:n_HF_pilot_plus_b0207],  yHFV_random)
yHFUU_sf_70 = hcat(yHFUUAll[:, 1:n_HF_pilot_plus_b0207], yHFUU_random)
yHFUW_sf_70 = hcat(yHFUWAll[:, 1:n_HF_pilot_plus_b0207], yHFUW_random)

QHF_sf_V,  λHF_sf_V,  bβHF_sf_V,  _, YmHF_sf_V  = buildKLE(xiHF_sf_70, yHFV_sf_70,  xbyD; kle_kwargs...)
QHF_sf_UU, λHF_sf_UU, bβHF_sf_UU, _, YmHF_sf_UU = buildKLE(xiHF_sf_70, yHFUU_sf_70, xbyD; kle_kwargs...)
QHF_sf_UW, λHF_sf_UW, bβHF_sf_UW, _, YmHF_sf_UW = buildKLE(xiHF_sf_70, yHFUW_sf_70, xbyD; kle_kwargs...)

klModes_HFsf_V  = QHF_sf_V  .* sqrt.(λHF_sf_V)'
klModes_HFsf_UU = QHF_sf_UU .* sqrt.(λHF_sf_UU)'
klModes_HFsf_UW = QHF_sf_UW .* sqrt.(λHF_sf_UW)'

xiHF_pilot_random_40 = vcat(xiHF_pilot, hf_batch_random)

yDeltaV_pr_40  = hcat(yDeltaV_new,  yHFV_random  .- yLFV_random)
yDeltaUU_pr_40 = hcat(yDeltaUU_new, yHFUU_random .- yLFUU_random)
yDeltaUW_pr_40 = hcat(yDeltaUW_new, yHFUW_random .- yLFUW_random)

QΔ_V_pr,  λΔ_V_pr,  bβΔ_V_pr,  _, YmΔ_V_pr  = buildKLE(xiHF_pilot_random_40, yDeltaV_pr_40,  xbyD; kle_kwargs_Δ...)
QΔ_UU_pr, λΔ_UU_pr, bβΔ_UU_pr, _, YmΔ_UU_pr = buildKLE(xiHF_pilot_random_40, yDeltaUU_pr_40, xbyD; kle_kwargs_Δ...)
QΔ_UW_pr, λΔ_UW_pr, bβΔ_UW_pr, _, YmΔ_UW_pr = buildKLE(xiHF_pilot_random_40, yDeltaUW_pr_40, xbyD; kle_kwargs_Δ...)

klModes_Δpr_V  = QΔ_V_pr  .* sqrt.(λΔ_V_pr)'
klModes_Δpr_UU = QΔ_UU_pr .* sqrt.(λΔ_UU_pr)'
klModes_Δpr_UW = QΔ_UW_pr .* sqrt.(λΔ_UW_pr)'

xiHF_holdout = hf_all_scaled[46:50, :]
yHFV_holdout  = yHFVAll[:, 46:50]
yHFUU_holdout = yHFUUAll[:, 46:50]
yHFUW_holdout = yHFUWAll[:, 46:50]
n_holdout = size(xiHF_holdout, 1)
n_grid = length(xbyD)

yPredField_bf_V  = zeros(n_grid, n_holdout); yPredField_bf_UU  = zeros(n_grid, n_holdout); yPredField_bf_UW  = zeros(n_grid, n_holdout)
yPredField_sf_V  = zeros(n_grid, n_holdout); yPredField_sf_UU  = zeros(n_grid, n_holdout); yPredField_sf_UW  = zeros(n_grid, n_holdout)
yPredField_bfr_V = zeros(n_grid, n_holdout); yPredField_bfr_UU = zeros(n_grid, n_holdout); yPredField_bfr_UW = zeros(n_grid, n_holdout)
yPredField_lf_V  = zeros(n_grid, n_holdout); yPredField_lf_UU  = zeros(n_grid, n_holdout); yPredField_lf_UW  = zeros(n_grid, n_holdout)

for i in 1:n_holdout
    xi_pt = xiHF_holdout[i, :]'

    ΨTest_LF = PrepCaseA(xi_pt; order=kle_kwargs.order, dims=kle_kwargs.dims)'
    ΨTest_Δ  = PrepCaseA(xi_pt; order=kle_kwargs_Δ.order, dims=kle_kwargs_Δ.dims)'

    yPred_lf_V  = klModes_LF_V  * bβLF_V  * ΨTest_LF .+ YmLF_V
    yPred_lf_UU = klModes_LF_UU * bβLF_UU * ΨTest_LF .+ YmLF_UU
    yPred_lf_UW = klModes_LF_UW * bβLF_UW * ΨTest_LF .+ YmLF_UW

    yPred_bf_V  = yPred_lf_V  .+ klModes_Δ40_V  * bβΔ_V_40  * ΨTest_Δ .+ YmΔ_V_40
    yPred_bf_UU = yPred_lf_UU .+ klModes_Δ40_UU * bβΔ_UU_40 * ΨTest_Δ .+ YmΔ_UU_40
    yPred_bf_UW = yPred_lf_UW .+ klModes_Δ40_UW * bβΔ_UW_40 * ΨTest_Δ .+ YmΔ_UW_40

    yPred_sf_V  = klModes_HFsf_V  * bβHF_sf_V  * ΨTest_LF .+ YmHF_sf_V
    yPred_sf_UU = klModes_HFsf_UU * bβHF_sf_UU * ΨTest_LF .+ YmHF_sf_UU
    yPred_sf_UW = klModes_HFsf_UW * bβHF_sf_UW * ΨTest_LF .+ YmHF_sf_UW

    yPred_bfr_V  = yPred_lf_V  .+ klModes_Δpr_V  * bβΔ_V_pr  * ΨTest_Δ .+ YmΔ_V_pr
    yPred_bfr_UU = yPred_lf_UU .+ klModes_Δpr_UU * bβΔ_UU_pr * ΨTest_Δ .+ YmΔ_UU_pr
    yPred_bfr_UW = yPred_lf_UW .+ klModes_Δpr_UW * bβΔ_UW_pr * ΨTest_Δ .+ YmΔ_UW_pr

    yPredField_bf_V[:, i]  = vec(yPred_bf_V);  yPredField_bf_UU[:, i]  = vec(yPred_bf_UU);  yPredField_bf_UW[:, i]  = vec(yPred_bf_UW)
    yPredField_sf_V[:, i]  = vec(yPred_sf_V);  yPredField_sf_UU[:, i]  = vec(yPred_sf_UU);  yPredField_sf_UW[:, i]  = vec(yPred_sf_UW)
    yPredField_bfr_V[:, i] = vec(yPred_bfr_V); yPredField_bfr_UU[:, i] = vec(yPred_bfr_UU); yPredField_bfr_UW[:, i] = vec(yPred_bfr_UW)
    yPredField_lf_V[:, i]  = vec(yPred_lf_V);  yPredField_lf_UU[:, i]  = vec(yPred_lf_UU);  yPredField_lf_UW[:, i]  = vec(yPred_lf_UW)
end

absL2_bf_mean  = [(norm(yHFV_holdout[:,i].-yPredField_bf_V[:,i])  + norm(yHFUU_holdout[:,i].-yPredField_bf_UU[:,i])  + norm(yHFUW_holdout[:,i].-yPredField_bf_UW[:,i]))/3  for i in 1:n_holdout]
absL2_sf_mean  = [(norm(yHFV_holdout[:,i].-yPredField_sf_V[:,i])  + norm(yHFUU_holdout[:,i].-yPredField_sf_UU[:,i])  + norm(yHFUW_holdout[:,i].-yPredField_sf_UW[:,i]))/3  for i in 1:n_holdout]
absL2_bfr_mean = [(norm(yHFV_holdout[:,i].-yPredField_bfr_V[:,i]) + norm(yHFUU_holdout[:,i].-yPredField_bfr_UU[:,i]) + norm(yHFUW_holdout[:,i].-yPredField_bfr_UW[:,i]))/3 for i in 1:n_holdout]
absL2_lf_mean  = [(norm(yHFV_holdout[:,i].-yPredField_lf_V[:,i])  + norm(yHFUU_holdout[:,i].-yPredField_lf_UU[:,i])  + norm(yHFUW_holdout[:,i].-yPredField_lf_UW[:,i]))/3  for i in 1:n_holdout]

@printf("Mean absolute L2 error over %d holdout points -- BF(AL): %.4f  SF: %.4f  BF(random): %.4f  LF-only: %.4f\n",
        n_holdout, mean(absL2_bf_mean), mean(absL2_sf_mean), mean(absL2_bfr_mean), mean(absL2_lf_mean))

output_path = joinpath(script_dir, "Holdout_Predictions.npz")
npzwrite(output_path,
         Dict("xbyD" => xbyD,
              "xiHF_holdout" => xiHF_holdout,
              "yHFV_true"  => yHFV_holdout,  "yHFUU_true"  => yHFUU_holdout,  "yHFUW_true"  => yHFUW_holdout,
              "yV_bf"   => yPredField_bf_V,   "yUU_bf"   => yPredField_bf_UU,   "yUW_bf"   => yPredField_bf_UW,
              "yV_sf"   => yPredField_sf_V,   "yUU_sf"   => yPredField_sf_UU,   "yUW_sf"   => yPredField_sf_UW,
              "yV_bfr"  => yPredField_bfr_V,  "yUU_bfr"  => yPredField_bfr_UU,  "yUW_bfr"  => yPredField_bfr_UW,
              "yV_lf"   => yPredField_lf_V,   "yUU_lf"   => yPredField_lf_UU,   "yUW_lf"   => yPredField_lf_UW,
              "absL2_bf_mean" => absL2_bf_mean, "absL2_sf_mean" => absL2_sf_mean,
              "absL2_bfr_mean" => absL2_bfr_mean, "absL2_lf_mean" => absL2_lf_mean))

@printf("\nWrote %s (with LF-only predictions added)\n", output_path)
