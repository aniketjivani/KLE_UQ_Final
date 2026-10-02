import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTFILE = Path(__file__).resolve().parent / "013c_LF_sine_approximations_C1_C1restricted_C2.jpg"

TAYLOR_COLORS = {3: "#00BFFF", 5: "#1E90FF", 7: "blue"}
BS_COLORS = {0.0: "#00BFFF", 0.5: "#1E90FF", 1.0: "blue"}


def taylor_sine(x: np.ndarray, b: float, order: int) -> np.ndarray:
    z = b * x
    approximation = np.zeros_like(z)
    for power in range(1, order + 1, 2):
        approximation += (-1) ** ((power - 1) // 2) * z**power / math.factorial(power)
    return approximation


def bs_sine(x: np.ndarray, b: float, s: float) -> np.ndarray:
    bx_degrees = np.degrees(b * x)
    numerator_scale = 4.0 - 0.5 * s
    denominator_constant = 40500.0 - 25500.0 * s
    return numerator_scale * bx_degrees * (180.0 - bx_degrees) / (
        denominator_constant - bx_degrees * (180.0 - bx_degrees)
    )


def add_original_sine(axis: plt.Axes, x: np.ndarray, b: float) -> None:
    axis.plot(
        x,
        np.sin(b * x),
        color="#222222",
        linewidth=5.2,
        alpha=0.62,
        zorder=1,
        label=r"Original $\sin(bx)$",
    )


def style_axis(axis: plt.Axes, title: str, x_limits: tuple[float, float], y_label: bool) -> None:
    axis.set_title(title, fontsize=24, fontweight="bold", fontname="Times New Roman")
    axis.set_xlabel(r"$x$", fontsize=24)
    if y_label:
        axis.set_ylabel(r"$y_{\mathrm{LF}}$", fontsize=24)
    axis.set_xlim(*x_limits)
    axis.tick_params(axis="both", which="major", labelsize=19)
    axis.grid(True, color="#a6a6a6", alpha=0.65, linewidth=0.9)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=16,
        columnspacing=1.35,
    )


def main() -> None:
    b_taylor = 70.0  # C1 midpoint: b in [60, 80].
    b_bs = 40.0  # C2 midpoint: b in [30, 50].
    x_full = np.linspace(0.0, 0.1, 600)
    x_restricted = np.linspace(0.0, 0.05, 600)

    plt.rcParams.update(
        {
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))

    add_original_sine(axes[0], x_full, b_taylor)
    for order in (3, 5, 7):
        axes[0].plot(
            x_full,
            taylor_sine(x_full, b_taylor, order),
            color=TAYLOR_COLORS[order],
            linewidth=3.4,
            zorder=2,
            label=rf"Taylor, $k={order}$",
        )
    style_axis(axes[0], r"C1, $b=70$", (0.0, 0.1), y_label=True)

    add_original_sine(axes[1], x_restricted, b_taylor)
    for order in (3, 5, 7):
        axes[1].plot(
            x_restricted,
            taylor_sine(x_restricted, b_taylor, order),
            color=TAYLOR_COLORS[order],
            linewidth=3.4,
            zorder=2,
            label=rf"Taylor, $k={order}$",
        )
    style_axis(axes[1], r"C1 restricted, $x \in [0, 0.05]$, $b=70$", (0.0, 0.05), y_label=False)

    add_original_sine(axes[2], x_full, b_bs)
    for s in (0.0, 0.5, 1.0):
        axes[2].plot(
            x_full,
            bs_sine(x_full, b_bs, s),
            color=BS_COLORS[s],
            linewidth=3.4,
            zorder=2,
            label=rf"BS, $s={s:.1f}$",
        )
    style_axis(axes[2], r"C2, $b=40$", (0.0, 0.1), y_label=False)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.35)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTFILE)


if __name__ == "__main__":
    main()
