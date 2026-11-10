"""Funções auxiliares de visualização dos experimentos."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Paleta acessível (Okabe-Ito) e estilo consistente.
COLORS = {
    "frozenlake_4x4": "#0072B2",
    "frozenlake_4x4_dry": "#009E73",
    "frozenlake_8x8": "#56B4E9",
    "cliffwalking": "#D55E00",
    "taxi": "#CC79A7",
}
SWEEP_COLORS = ["#000000", "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def load_curve(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def stack_curves(paths: list[Path], key: str, window: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empilha curvas de vários seeds e devolve média, desvio e eixo x."""
    curves = []
    for path in paths:
        data = load_curve(path)
        curves.append(smooth(data[key], window))
    matrix = np.vstack(curves)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return mean, std, np.arange(1, matrix.shape[1] + 1)


def band(ax, x, mean, std, color, label=None, alpha=0.18):
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"figura salva: {path}")
