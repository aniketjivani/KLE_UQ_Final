
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)

n_x = 200
x = np.linspace(0, 1, n_x)
ell = 0.05
sigma2 = 1.0


def rbf(x1, x2, ell, s2):
    r2 = (x1[:, None] - x2[None, :]) ** 2
    return s2 * np.exp(-0.5 * r2 / ell**2)


K_prior = rbf(x, x, ell, sigma2)
K_prior += 1e-10 * np.eye(n_x)

m = 6
obs_idx = np.sort(rng.choice(n_x, size=m, replace=False))
x_obs = x[obs_idx]

K_oo = rbf(x_obs, x_obs, ell, sigma2)
L_oo = np.linalg.cholesky(K_oo + 1e-8 * np.eye(m))
y_obs = L_oo @ rng.standard_normal(m)
noise = 1e-4

K_xo = rbf(x, x_obs, ell, sigma2)
K_oo_reg = K_oo + noise * np.eye(m)
K_oo_inv_y = np.linalg.solve(K_oo_reg, y_obs)
K_oo_inv_Kox = np.linalg.solve(K_oo_reg, K_xo.T)

mu_post = K_xo @ K_oo_inv_y
K_post = K_prior - K_xo @ K_oo_inv_Kox
K_post = 0.5 * (K_post + K_post.T) + 1e-10 * np.eye(n_x)

L_post = np.linalg.cholesky(K_post)


def sample_posterior(n_samples):
    z = rng.standard_normal((n_x, n_samples))
    return mu_post[:, None] + L_post @ z


N_max = 800
X_full = sample_posterior(N_max)


def modes_for_threshold(X, threshold=0.99):
    N = X.shape[1]
    Xc = X - X.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (s**2) / max(N - 1, 1)
    cum = np.cumsum(eigvals) / eigvals.sum()
    n_modes = int(np.searchsorted(cum, threshold) + 1)
    return eigvals, cum, n_modes


threshold = 0.99
N_values = np.arange(5, N_max + 1)
mode_counts = np.array(
    [modes_for_threshold(X_full[:, :N], threshold)[2] for N in N_values]
)

eigvals_ref, cum_ref, n_modes_ref = modes_for_threshold(X_full, threshold)
print(f"RBF ell={ell}, {m} conditioning points")
print(f"reference (N={N_max}) modes at 99%: {n_modes_ref}")
print(f"mode count range over sweep: {mode_counts.min()} to {mode_counts.max()}")
diffs = np.diff(mode_counts)
print(
    f"transitions: {int(np.sum(diffs != 0))} "
    f"(up: {int(np.sum(diffs > 0))}, down: {int(np.sum(diffs < 0))})"
)
down_idx = np.where(diffs < 0)[0]
print("down-transitions at N:", [int(N_values[i]) for i in down_idx[:10]])

fig, ax = plt.subplots(figsize=(8, 4.5))
mask = (N_values >= 490) & (N_values <= 600)
ax.step(N_values[mask], mode_counts[mask], where="post", lw=1.3, marker=".")
ax.set_xlabel("Number of snapshots, N")
ax.set_ylabel(r"$k_t$")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mode_count_vs_N_rbf_zoom.jpg", dpi=250, format="jpg")
plt.close(fig)

print(f"saved: {OUTPUT_DIR / 'mode_count_vs_N_rbf_zoom.jpg'}")
