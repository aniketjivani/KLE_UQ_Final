# Execution order

```bash
julia scripts/generate_exp8_hf_oracle.jl
julia scripts/exp8_lfallmodes_rerun.jl
julia scripts/exp8_02_frozenlf_rerun.jl
python scripts/exp8_01_fixedrank_heatmaps.py
python scripts/exp8_02_heatmaps.py
python scripts/exp8_c1_rep005_al_comparison.py
julia scripts/exp11b_rep005_bf_error_increase_scan.jl
```
