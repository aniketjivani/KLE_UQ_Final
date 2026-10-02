
using DelimitedFiles
using LinearAlgebra
using NPZ
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEPENDENCY_DIR = joinpath(ROOT, "dependencies")
include(joinpath(DEPENDENCY_DIR, "kleUtils.jl"))
include(joinpath(DEPENDENCY_DIR, "utils.jl"))

const REP_ID = 5
const STAGES = [24, 25]
const EXPECTED_LF_RANKS = [2, 1]
const LB = [40.0, 60.0]
const UB = [60.0, 80.0]
const X = collect(range(0, 0.1; length=250))
const INPUT_DIR = joinpath(ROOT, "data", "1d_inputs_c2_EI_nolog", @sprintf("rep_%03d", REP_ID))
const OUT_DIR = joinpath(ROOT, "data", "exp11")
const OUT_FILE = joinpath(OUT_DIR, "rep005_gp_bf_error_increase_scan_stage024_025.npz")
const PREVIOUS_POINTS = Set([(45.0, 65.0), (45.0, 75.0), (55.0, 65.0), (55.0, 75.0)])
const MIN_SCALED_SEPARATION = 0.18
const N_SELECT = 4

const KLE_KWARGS = (
    order=3, dims=2, family="Legendre", useFullGrid=1, getAllModes=0,
    fixedRank=nothing, weightFunction=getWeights, solver="Tikhonov-L2",
)

scale_ab(ab) = 2 .* (ab .- (1 / 2) .* (LB + UB)') ./ (UB - LB)'

function stage_snapshots(stage)
    lf_path = joinpath(INPUT_DIR, @sprintf("LF_Batch_%03d_Final.txt", stage))
    hf_path = joinpath(INPUT_DIR, @sprintf("HF_Batch_%03d_Final.txt", stage))
    subset_path = joinpath(INPUT_DIR, @sprintf("HF_Batch_%03d_Subset_Final.txt", stage))

    lf_all = readdlm(lf_path)
    hf_all = readdlm(hf_path)
    subset_all = readdlm(subset_path, Int)[:]
    iseven(size(lf_all, 1)) || error("odd LF row count in $lf_path")
    iseven(size(hf_all, 1)) || error("odd HF row count in $hf_path")
    n_lf = div(size(lf_all, 1), 2)
    n_hf = div(size(hf_all, 1), 2)
    length(subset_all) == 2 * n_hf || error("subset-index mismatch in $subset_path")

    inputs_lf = Matrix{Float64}(lf_all[1:n_lf, :])
    inputs_hf = Matrix{Float64}(hf_all[1:n_hf, :])
    subset = Vector{Int}(subset_all[1:n_hf])
    lf_data = generateLF(X, inputs_lf; chosen_lf="taylor")
    hf_data = generateHF(X, inputs_hf)
    delta_data = hf_data .- lf_data[:, subset]
    return inputs_lf, inputs_hf, lf_data, delta_data
end

function build_stage_models(stage)
    inputs_lf, inputs_hf, lf_data, delta_data = stage_snapshots(stage)
    lf_model = buildKLE(scale_ab(inputs_lf), lf_data, X; KLE_KWARGS...)
    delta_model = buildKLE(scale_ab(inputs_hf), delta_data, X; KLE_KWARGS...)
    return lf_model, delta_model
end

function predict_component(model, ab_scaled)
    Q, lambda, beta, _, mean_field = model
    return predictKLE(ab_scaled, Q, lambda, beta, mean_field; order=3, dims=2)
end

a_grid = collect(range(LB[1], UB[1]; length=41))
b_grid = collect(range(LB[2], UB[2]; length=41))
candidate_points = zeros(length(a_grid) * length(b_grid), 2)
idx = 1
for a in a_grid, b in b_grid
    candidate_points[idx, :] = [a, b]
    global idx += 1
end
candidate_scaled = scale_ab(candidate_points)
hf_truth_all = generateHF(X, candidate_points)

n_candidates = size(candidate_points, 1)
bf_predictions_all = zeros(length(STAGES), length(X), n_candidates)
lf_predictions_all = zeros(length(STAGES), length(X), n_candidates)
bf_errors_all = zeros(length(STAGES), n_candidates)
lf_ranks = zeros(Int, length(STAGES))
delta_ranks = zeros(Int, length(STAGES))

for (stage_i, stage) in enumerate(STAGES)
    lf_model, delta_model = build_stage_models(stage)
    lf_pred = predict_component(lf_model, candidate_scaled)
    delta_pred = predict_component(delta_model, candidate_scaled)
    bf_pred = lf_pred + delta_pred
    lf_ranks[stage_i] = length(lf_model[2])
    delta_ranks[stage_i] = length(delta_model[2])
    lf_ranks[stage_i] == EXPECTED_LF_RANKS[stage_i] || error(
        "stage $stage LF rank $(lf_ranks[stage_i]) != expected $(EXPECTED_LF_RANKS[stage_i])"
    )
    lf_predictions_all[stage_i, :, :] = lf_pred
    bf_predictions_all[stage_i, :, :] = bf_pred
    for k in 1:n_candidates
        bf_errors_all[stage_i, k] = norm(bf_pred[:, k] - hf_truth_all[:, k]) / norm(hf_truth_all[:, k])
    end
end

error_increase = bf_errors_all[2, :] - bf_errors_all[1, :]
order = sortperm(error_increase; rev=true)
selected = Int[]

function scaled_distance(i, j)
    return norm(candidate_scaled[i, :] - candidate_scaled[j, :]) / 2
end

for candidate in order
    point_tuple = (candidate_points[candidate, 1], candidate_points[candidate, 2])
    point_tuple in PREVIOUS_POINTS && continue
    error_increase[candidate] > 0 || break
    all(scaled_distance(candidate, chosen) >= MIN_SCALED_SEPARATION for chosen in selected) || continue
    push!(selected, candidate)
    length(selected) == N_SELECT && break
end
length(selected) == N_SELECT || error("found only $(length(selected)) separated positive-increase points")

selected_points = candidate_points[selected, :]
selected_hf_truth = permutedims(hf_truth_all[:, selected], (2, 1))
selected_lf_predictions = permutedims(lf_predictions_all[:, :, selected], (1, 3, 2))
selected_bf_predictions = permutedims(bf_predictions_all[:, :, selected], (1, 3, 2))
selected_lf_truth = permutedims(generateLF(X, selected_points; chosen_lf="taylor"), (2, 1))
selected_delta_truth = selected_hf_truth .- selected_lf_truth
selected_delta_predictions = selected_bf_predictions .- selected_lf_predictions
selected_bf_errors = bf_errors_all[:, selected]
selected_increase = error_increase[selected]
selected_lf_errors = zeros(length(STAGES), N_SELECT)
selected_delta_errors = zeros(length(STAGES), N_SELECT)
for stage_i in eachindex(STAGES), point_i in 1:N_SELECT
    selected_lf_errors[stage_i, point_i] = (
        norm(selected_lf_predictions[stage_i, point_i, :] - selected_lf_truth[point_i, :]) /
        norm(selected_lf_truth[point_i, :])
    )
    selected_delta_errors[stage_i, point_i] = (
        norm(selected_delta_predictions[stage_i, point_i, :] - selected_delta_truth[point_i, :]) /
        norm(selected_delta_truth[point_i, :])
    )
end

mkpath(OUT_DIR)
npzwrite(OUT_FILE, Dict(
    "rep_id" => REP_ID,
    "stages" => STAGES,
    "lf_ranks" => lf_ranks,
    "delta_ranks" => delta_ranks,
    "x" => X,
    "ab_points" => selected_points,
    "hf_oracle" => selected_hf_truth,
    "lf_oracle" => selected_lf_truth,
    "delta_oracle" => selected_delta_truth,
    "lf_prediction" => selected_lf_predictions,
    "delta_prediction" => selected_delta_predictions,
    "bf_prediction" => selected_bf_predictions,
    "lf_relative_l2_error" => selected_lf_errors,
    "delta_relative_l2_error" => selected_delta_errors,
    "bf_relative_l2_error" => selected_bf_errors,
    "bf_error_increase" => selected_increase,
    "scan_shape" => [length(a_grid), length(b_grid)],
    "positive_increase_count" => count(>(0), error_increase),
))

println("scanned $n_candidates points; $(count(>(0), error_increase)) have larger BF error at stage 25")
println("selected separated points, ordered by absolute BF-error increase:")
for (j, candidate) in enumerate(selected)
    println(@sprintf(
        "  %d: (a=%.1f, b=%.1f): BF stage24=%.6f, stage25=%.6f, increase=%+.6f, ratio=%.3f",
        j, candidate_points[candidate, 1], candidate_points[candidate, 2],
        bf_errors_all[1, candidate], bf_errors_all[2, candidate], error_increase[candidate],
        bf_errors_all[2, candidate] / bf_errors_all[1, candidate],
    ))
    println(@sprintf(
        "     LF stage24=%.6f, stage25=%.6f; Delta stage24=%.6f, stage25=%.6f",
        selected_lf_errors[1, j], selected_lf_errors[2, j],
        selected_delta_errors[1, j], selected_delta_errors[2, j],
    ))
end
println("saved: $OUT_FILE")
