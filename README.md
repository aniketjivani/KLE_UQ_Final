### Bifidelity KLEs with Active Learning for Random Fields

We present a bifidelity Karhunen--Lo\`{e}ve expansion (KLE) surrogate model for field-valued quantities of interest (QoIs) under uncertain inputs. The approach combines the spectral efficiency of the KLE with polynomial chaos expansions (PCEs) to preserve an explicit mapping between input uncertainties and output fields. By coupling inexpensive low-fidelity (LF) simulations that capture dominant response trends with a limited number of high-fidelity (HF) simulations that correct for systematic bias, the proposed method enables accurate and computationally affordable surrogate construction. To further improve surrogate accuracy, we form an active learning strategy that adaptively selects new HF evaluations based on the surrogate's generalization error, estimated via cross-validation and modeled using Gaussian process regression. New HF samples are then acquired by maximizing an expected improvement criterion, targeting regions of high surrogate error. The resulting BF-KLE-AL framework is demonstrated on three examples of increasing complexity: a one-dimensional analytical benchmark, a two-dimensional convection-diffusion system, and a three-dimensional turbulent round jet simulation based on Reynolds-averaged Navier--Stokes (RANS) and enhanced delayed detached-eddy simulations (EDDES). Across these cases, the method achieves consistent improvements in predictive accuracy and sample efficiency relative to single-fidelity and random-sampling approaches.



**Code**:
Selection of new points during the active learning process was performed with BoTorch (v0.8.5). 
Using later versions of BoTorch may require modified function arguments in the active learning routines, optimization calls etc. (e.g. `fit_gpytorch_mll` vs `fit_gpytorch_model`).

We present results for 3 problems:

- 1D QoIs from a synthetic pulse function (`1d_toy`)
- 2D QoIs from convection-diffusion based PDE (`2d_pde`)
- 1D QoIs from 3D simulations of a turbulent round jet. (`Jet`) All simulations were conducted using SU2 (details in preprint).