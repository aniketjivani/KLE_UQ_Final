### Bifidelity KLEs with Active Learning for Random Fields

We present a bifidelity Karhunen--Loève expansion (KLE) surrogate model for field-valued quantities of interest (QoIs) under uncertain inputs. The QoIs considered here are scalar fields. The approach combines the spectral efficiency of the KLE with polynomial chaos expansions (PCEs) to preserve an explicit mapping between input uncertainties and output fields. By coupling inexpensive low-fidelity (LF) simulations that capture dominant response trends with a limited number of high-fidelity (HF) simulations that correct for systematic bias, the proposed method can enable accurate and computationally affordable surrogate construction.
To further improve surrogate accuracy, we develop an active learning strategy that adaptively selects new HF evaluations based on the surrogate's generalization error, estimated via cross-validation and modeled using Gaussian process regression. New HF samples are then acquired by maximizing an expected improvement criterion, targeting regions of high surrogate error.
The resulting BF-KLE-AL framework is demonstrated on three examples of increasing complexity: a one-dimensional analytical benchmark, a two-dimensional convection-diffusion system, and a three-dimensional turbulent round jet simulation based on Reynolds-averaged Navier--Stokes (RANS) and enhanced delayed detached-eddy simulations (EDDES). The experiments show that bifidelity gains depend on LF accuracy, discrepancy approximation, and the allocation of simulation cost. Active learning improves prediction over random sampling in several settings, while the cost-matched comparisons identify both favorable regimes and cases where an HF-only surrogate is more accurate.

**Preprint**: https://arxiv.org/abs/2511.03756

**Code**:
Selection of new points during the active learning process was performed with [BoTorch v0.8.5](https://pypi.org/project/botorch/0.8.5/).
Using later versions of BoTorch may require modified function arguments in the active learning routines, optimization calls etc. (e.g. `fit_gpytorch_mll` vs `fit_gpytorch_model`).

To re-train DeepONet in `appendix_b`:

`python -m pip install --upgrade "jax[cpu]" optax`

Install Julia using the instructions at [julialang.org/downloads](https://julialang.org/downloads/).

To set up PyCall with the Python environment containing BoTorch:

1. Activate the environment and run `which python` to find its Python executable.
2. Start Julia and run:

```julia
using Pkg
ENV["PYTHON"] = "/path/from/which/python"
Pkg.add("PyCall")
Pkg.build("PyCall")
```

3. Restart Julia and run `using PyCall`.

We present results for 3 problems:

- 1D QoIs from a synthetic pulse function (`1d_toy`)
- 2D QoIs from convection-diffusion based PDE (`2d_pde`)
- 1D QoIs from 3D simulations of a turbulent round jet (`Jet`, `ground_truth_jet_comparisons`) All simulations were conducted using SU2 (details in preprint).

Results for appendices:

- `sec_42_appendix_a`
- `appendix_a_stability`
- `appendix_b`
