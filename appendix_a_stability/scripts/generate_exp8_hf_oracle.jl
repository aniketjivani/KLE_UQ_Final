using JLD

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "dependencies", "utils.jl"))

const OUTPUT = joinpath(ROOT, "data", "exp8", "exp8_HF_Oracle_case_02.jld")

x = collect(range(0, 0.1; length=250))
a_grid = collect(range(40.0, 60.0; length=200))
b_grid = collect(range(60.0, 80.0; length=200))

mkpath(dirname(OUTPUT))
JLD.save(OUTPUT, "HF_oracle", generateOracleData(a_grid, b_grid, x))
println("Saved $(OUTPUT)")
