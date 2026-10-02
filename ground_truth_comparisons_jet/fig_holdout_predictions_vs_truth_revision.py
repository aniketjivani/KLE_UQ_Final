import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

saveFig = True

plt.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-v0_8-notebook')
plt.rc("font", family="serif")
plt.rc("axes.spines", top=True, right=True)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('grid', alpha=0.6)

script_dir = Path(__file__).resolve().parent
d = np.load(script_dir / "Holdout_Predictions.npz")

xbyD = d["xbyD"]
n_holdout = d["yHFV_true"].shape[1]

qoi_labels = [r"$\overline{v}$", r"$\overline{u'u'}$", r"$\overline{u'w'}$"]
qoi_keys = ["V", "UU", "UW"]

true_fields = {"V": d["yHFV_true"], "UU": d["yHFUU_true"], "UW": d["yHFUW_true"]}

surrogates = {
    "HF-KLE":    {"V": d["yV_sf"],  "UU": d["yUU_sf"],  "UW": d["yUW_sf"],  "color": "#d62728", "ls": "--"},
    "LF-KLE":    {"V": d["yV_lf"],  "UU": d["yUU_lf"],  "UW": d["yUW_lf"],  "color": "#9467bd", "ls": ":"},
    "BF-KLE-RS": {"V": d["yV_bfr"], "UU": d["yUU_bfr"], "UW": d["yUW_bfr"], "color": "#2ca02c", "ls": "-."},
    "BF-KLE-AL": {"V": d["yV_bf"],  "UU": d["yUU_bf"],  "UW": d["yUW_bf"],  "color": "#1f77b4", "ls": "--"},
}

SURROGATE_LINEWIDTH = 5.0

hf_point_labels = [f"Holdout Sim {i + 1}" for i in range(n_holdout)]

fig, ax = plt.subplots(3, n_holdout, figsize=(4.2 * n_holdout, 11), sharex=True)

for row, qoi in enumerate(qoi_keys):
    for col in range(n_holdout):
        a = ax[row, col]

        a.plot(xbyD, true_fields[qoi][:, col], color="black", linewidth=4.0,
               label="Ground truth HF", zorder=5)

        for name, s in surrogates.items():
            a.plot(xbyD, s[qoi][:, col], color=s["color"], linestyle=s["ls"],
                   linewidth=SURROGATE_LINEWIDTH, alpha=0.9, label=name)

        a.tick_params(axis='both', which='major', labelsize=14)
        a.set_xlim(xbyD.min(), xbyD.max())

        if row == 0:
            a.set_title(hf_point_labels[col], fontsize=20, fontweight='normal')
        if row == 2:
            a.set_xlabel(r"$x/D$", fontsize=20)
        if col == 0:
            a.set_ylabel(qoi_labels[row], fontsize=22, fontweight='bold')

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=16,
           bbox_to_anchor=(0.5, 1.03), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])

if saveFig:
    plt.savefig(script_dir / "holdout_predictions_vs_truth.jpg",
                dpi=200, bbox_inches='tight')

print("Mean absolute L2 error over 5 holdout points:")
for name in ["BF-KLE-AL", "HF-KLE", "LF-KLE", "BF-KLE-RS"]:
    key = {"BF-KLE-AL": "absL2_bf_mean", "HF-KLE": "absL2_sf_mean",
           "LF-KLE": "absL2_lf_mean", "BF-KLE-RS": "absL2_bfr_mean"}[name]
    print(f"  {name:10s}: {np.mean(d[key]):.4f}")

surrogate_order = ["BF-KLE-AL", "HF-KLE", "LF-KLE", "BF-KLE-RS"]
rel_err_table = {name: {} for name in surrogate_order}

for name in surrogate_order:
    s = surrogates[name]
    for qoi in qoi_keys:
        true_q = true_fields[qoi]
        pred_q = s[qoi]
        rel_errs = np.linalg.norm(true_q - pred_q, axis=0) / np.linalg.norm(true_q, axis=0)
        rel_err_table[name][qoi] = np.mean(rel_errs)

print("\nMean relative L2 error over 5 holdout points, by QoI:")
header = "| QoI | " + " | ".join(surrogate_order) + " |"
sep = "|---|" + "---|" * len(surrogate_order)
print(header)
print(sep)
for qoi, label in zip(qoi_keys, qoi_labels):
    row = f"| {label} | " + " | ".join(f"{rel_err_table[name][qoi]:.4f}" for name in surrogate_order) + " |"
    print(row)
