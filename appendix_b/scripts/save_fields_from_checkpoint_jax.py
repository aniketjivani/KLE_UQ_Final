\
\
\
\
\
\
\
\
\
   

import argparse
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
OUTPUT_DIR = DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for rep in args.rep:
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
        print(f"rep {rep}: (from checkpoint, final_iter={ckpt['final_iter']}) "
              f"BF={bf_err:.4e}  LF-term={lf_err:.4e}  correction-term={corr_err:.4e}")

        outpath = OUTPUT_DIR / f"diag_fields_rep{rep}_deeponet_jax.npz"
        np.savez(outpath, F_LF_o=F_LF_o, F_comp_o=F_comp_o, F_correction_o=F_correction_o,
                  LF_oracle=LF_oracle, HF_oracle=HF_oracle)
        print(f"  saved {outpath}")


if __name__ == "__main__":
    main()
