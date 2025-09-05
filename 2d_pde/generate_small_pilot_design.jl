using DelimitedFiles
using Random
using Distributions
using ArgParse
using Dates
using LatinHypercubeSampling

# four parameters: theta_s, theta_h, theta_x, theta_y

lower_bounds = [0.01, 0.05, 0.3, 0.55]
upper_bounds = [0.05, 0.08, 0.7, 0.85]

NLF = 500
NHF = 25

input_list = zeros(NLF, 4)

LFPlan, _ = LHCoptim(NLF, 4, NLF,
                rng=MersenneTwister(8026)
                )
input_list = scaleLHC(LFPlan, [(lower_bounds[1], upper_bounds[1]),
                               (lower_bounds[2], upper_bounds[2]),
                               (lower_bounds[3], upper_bounds[3]),
                               (lower_bounds[4], upper_bounds[4])
                               ])

input_list_scaled = scaleLHC(LFPlan, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)])

HFPlan, _ = subLHCoptim(LFPlan, NHF, 1000)
HFIdx = sort(subLHCindex(LFPlan, HFPlan))
HFPlanFinal = LFPlan[HFIdx, :]

input_listHF = input_list[HFIdx, :]
input_listHF_scaled = input_list_scaled[HFIdx, :]

open("./2d_pde/input_list_LF_Pilot_scaled.txt", "w") do io
    writedlm(io, input_list_scaled)
end
open("./2d_pde/input_list_LF_Pilot.txt", "w") do io
    writedlm(io, input_list)
end
open("./2d_pde/input_list_HF_Pilot_scaled.txt", "w") do io
    writedlm(io, input_listHF_scaled)
end
open("./2d_pde/input_list_HF_Pilot.txt", "w") do io
    writedlm(io, input_listHF)
end
open("./2d_pde/input_list_LFIdx.txt", "w") do io
    writedlm(io, HFIdx)
end