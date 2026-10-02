\
\
\
\
\
\
   

import pathlib
import pickle
import sys

import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
APPENDIX_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APPENDIX_DIR.parent
DEEPONET_DIR = PROJECT_ROOT / "deeponet_comparisons"
SOURCE_BASE_EXPERIMENT = DEEPONET_DIR / "base_experiment"
sys.path.insert(0, str(DEEPONET_DIR))

from deeponet_model_jax import composite_forward
from deeponet_utils import u_of, rel_l2_batch

DATA_DIR = APPENDIX_DIR / "data"
CKPT_DIR = APPENDIX_DIR / "checkpoints_jax"
NREPS = 10


def main():
    bf_errors, sf_errors, lf_term_errors, correction_term_errors = [], [], [], []

    for rep in range(1, NREPS + 1):
        ckpt_path = CKPT_DIR / f"composite_rep{rep}.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        params = ckpt["params"]

        d = np.load(DATA_DIR / f"jump_bifi_rep{rep}.npz")
        x = d["x"]
        a_grid = d["a_grid"]
        LF_oracle = d["LF_oracle"].T
        HF_oracle = d["HF_oracle"].T
        u_oracle = u_of(a_grid, x)

        x_grid_j = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)
        u_oracle_j = jnp.asarray(u_oracle, dtype=jnp.float32)
        F_LF_o, F_lin_o, F_nl_o, F_comp_o = composite_forward(params, u_oracle_j, x_grid_j)
        F_LF_o = np.asarray(F_LF_o)
        F_comp_o = np.asarray(F_comp_o)
        F_correction_o = F_comp_o - F_LF_o

        bf_err = rel_l2_batch(F_comp_o, HF_oracle)
        lf_err = rel_l2_batch(F_LF_o, LF_oracle)
        corr_err = rel_l2_batch(F_correction_o, HF_oracle - LF_oracle)

                                                                             
                                                                                         
                                                                                
        bf_errors.append(bf_err)
        lf_term_errors.append(lf_err)
        correction_term_errors.append(corr_err)

        print(f"rep {rep:2d}: BF={bf_err:.4e}  LF-term={lf_err:.4e}  correction-term={corr_err:.4e}")

    sf_first5 = np.load(DATA_DIR / "results_deeponet_jax.npz")["sf_errors"]
    sf_last5 = np.load(DATA_DIR / "results_deeponet_jax_reps6_10.npz")["sf_errors"]
    sf_errors = np.concatenate([sf_first5, sf_last5])

    bf_errors = np.array(bf_errors)
    lf_term_errors = np.array(lf_term_errors)
    correction_term_errors = np.array(correction_term_errors)

    print(f"BF-DeepONet (JAX, 10 reps): mean={bf_errors.mean():.4e}  std={bf_errors.std():.4e}")
    print(f"SF-DeepONet (JAX, 10 reps): mean={sf_errors.mean():.4e}  std={sf_errors.std():.4e}")
    print(f"LF-term (JAX, 10 reps):     mean={lf_term_errors.mean():.4e}  std={lf_term_errors.std():.4e}")
    print(f"Correction-term (JAX, 10 reps): mean={correction_term_errors.mean():.4e}  "
          f"std={correction_term_errors.std():.4e}")

    out_file = DATA_DIR / "results_deeponet_jax_10reps.npz"
    np.savez(out_file, bf_errors=bf_errors, sf_errors=sf_errors,
              lf_term_errors=lf_term_errors, correction_term_errors=correction_term_errors)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
