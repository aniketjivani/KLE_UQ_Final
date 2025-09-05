# generate oracle design. Since it is too expensive to do discretize and use all combinations from the 4D grid, we will use a Latin Hypercube design much like the one used for the pilot set.
using LatinHypercubeSampling
using Random
using LinearAlgebra
using DelimitedFiles

NRUNS = 1000

oraclePlan, _ = LHCoptim(NRUNS, 4, 1000, 
                rng=MersenneTwister(6026)
                )

lb = [0.01, 0.05, 0.3, 0.55]
ub = [0.05, 0.08, 0.7, 0.85]

oraclePoints = scaleLHC(oraclePlan, [(lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3]), (lb[4], ub[4])])
oraclePointsScaled = scaleLHC(oraclePlan, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)])


open("./2d_pde/input_list_oracle_HF.txt", "w") do io
    writedlm(io, oraclePoints)
end
open("./2d_pde/input_list_oracle_HF_scaled.txt", "w") do io
    writedlm(io, oraclePointsScaled)
end