import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


OUTPUT = Path(__file__).resolve().parent / "003c_correlation_C1_C1restricted_C2.jpg"
N_SAMPLES = 1000
N_POINTS = 250

K_VALUES = [3, 5, 7]
K_COLORS = {3: "#00BFFF", 5: "#1E90FF", 7: "blue"}
S_VALUES = [0.0, 0.5, 1.0]
S_COLORS = {0.0: "#00BFFF", 0.5: "#1E90FF", 1.0: "blue"}


def spiked_waveform(x, a, b):
    return np.exp(-a * x) * np.sin(b * x)


def sin_truncated(z, k):
    if k % 2 != 1:
        raise ValueError("Taylor truncation order must be odd")
    result = np.zeros_like(z)
    for n in range(0, k, 2):
        result += (-1) ** (n // 2) * z ** (n + 1) / math.factorial(n + 1)
    return result


def lf_taylor(x, a, b, k):
    return np.exp(-a * x) * sin_truncated(b * x, k)


def lf_bhaskara_sine(x, a, b, s):
    bx_degrees = x * (180 / np.pi) * b
    numerator_scale = 4.0 - 0.5 * s
    denominator_scale = 40500.0 - 25500.0 * s
    return np.exp(-a * x) * numerator_scale * bx_degrees * (180 - bx_degrees) / (
        denominator_scale - bx_degrees * (180 - bx_degrees)
    )


def correlation_curve(x_values, a_samples, b_samples, low_fidelity, knob):
    correlation = np.empty_like(x_values)
    for index, x in enumerate(x_values):
        if x == 0.0:
            correlation[index] = 1.0
            continue
        hf_values = spiked_waveform(x, a_samples, b_samples)
        lf_values = low_fidelity(x, a_samples, b_samples, knob)
        correlation[index] = np.corrcoef(hf_values, lf_values)[0, 1]
    return correlation


def style_panel(axis, title, x_limit, show_ylabel=False):
    axis.set_xlabel(r"$x$", fontsize=26)
    if show_ylabel:
        axis.set_ylabel("Correlation", fontsize=26)
    axis.set_title(title, fontsize=24, fontweight="bold", fontname="Times New Roman")
    axis.set_xlim(x_limit)
    axis.tick_params(axis="both", which="major", labelsize=22)
    axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def add_legend(axis):
    axis.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                ncol=3, frameon=False)


def main():
    x_full = np.linspace(0, 0.1, N_POINTS)
    x_restricted = np.linspace(0, 0.05, N_POINTS)
    rng = np.random.default_rng(0)
    a_taylor = rng.uniform(40, 60, N_SAMPLES)
    b_taylor = rng.uniform(60, 80, N_SAMPLES)
    a_bhaskara = rng.uniform(40, 60, N_SAMPLES)
    b_bhaskara = rng.uniform(30, 50, N_SAMPLES)

    fig, axes = plt.subplots(1, 3, figsize=(27, 8))

    for k in K_VALUES:
        correlation = correlation_curve(x_full, a_taylor, b_taylor, lf_taylor, k)
        axes[0].plot(x_full, correlation, linewidth=3.5, color=K_COLORS[k], label=rf"$k={k}$")
    style_panel(axes[0], "C1", (0, 0.1), show_ylabel=True)
    add_legend(axes[0])

    for k in K_VALUES:
        correlation = correlation_curve(x_restricted, a_taylor, b_taylor, lf_taylor, k)
        axes[1].plot(x_restricted, correlation, linewidth=3.5, color=K_COLORS[k], label=rf"$k={k}$")
    style_panel(axes[1], r"C1 restricted, $x \in [0, 0.05]$", (0, 0.05))
    add_legend(axes[1])

    for s in S_VALUES:
        correlation = correlation_curve(x_full, a_bhaskara, b_bhaskara, lf_bhaskara_sine, s)
        axes[2].plot(x_full, correlation, linewidth=3.5, color=S_COLORS[s], label=rf"$s={s:.1f}$")
    style_panel(axes[2], "C2", (0, 0.1))
    add_legend(axes[2])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
