"""Gera as figuras do relatório a partir dos resultados salvos."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments import ENVIRONMENTS, MAIN_ENVIRONMENTS  # noqa: E402
from src.experiment import train_agent  # noqa: E402
from src.plotting import (  # noqa: E402
    COLORS,
    SWEEP_COLORS,
    band,
    load_curve,
    save,
    set_style,
    smooth,
    stack_curves,
)
from src.qlearning import QLearningConfig  # noqa: E402

METRICS = ROOT / "results" / "metrics"
CURVES = ROOT / "results" / "curves"
FIGURES = ROOT / "results" / "figures"

ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def curve_path(row: pd.Series, prefix: str) -> Path:
    filename = f"{prefix}{row['env_key']}__seed{int(row['seed'])}"
    tag = row.get("run_tag", "")
    if isinstance(tag, str) and tag:
        filename += f"__{tag}"
    return CURVES / f"{filename}.npz"


def paths_for(summary: pd.DataFrame, prefix: str, value: str | None = None) -> list[Path]:
    frame = summary if value is None else summary[summary["sweep_value"] == value]
    return [curve_path(row, prefix) for _, row in frame.iterrows()]


def figure_training_curves() -> None:
    summary = pd.read_csv(METRICS / "main_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, key in zip(axes.flat, MAIN_ENVIRONMENTS):
        spec = ENVIRONMENTS[key]
        paths = paths_for(summary[summary["env_key"] == key], "main__")
        mean, std, x = stack_curves(paths, "train_rewards", window=100)
        band(ax, x, mean, std, COLORS[key])
        ax.set_title(spec.label)
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Recompensa média\n(janela móvel = 100)")
        if spec.success_rule == "positive_reward":
            ax.set_ylim(-0.02, 1.02)
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Curvas de aprendizado por ambiente (média ± desvio-padrão entre 10 sementes)",
        fontsize=11,
    )
    save(fig, FIGURES / "fig_01_curvas_treino.png")


def figure_greedy_evaluation() -> None:
    summary = pd.read_csv(METRICS / "main_summary.csv")
    fig, (ax_success, ax_reward) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for key in MAIN_ENVIRONMENTS:
        spec = ENVIRONMENTS[key]
        paths = paths_for(summary[summary["env_key"] == key], "main__")
        success_curves = []
        reward_curves = []
        episodes = None
        for path in paths:
            data = load_curve(path)
            episodes = data["eval_episodes"]
            success_curves.append(data["eval_success"])
            reward_curves.append(data["eval_rewards"])
        success = np.vstack(success_curves)
        reward = np.vstack(reward_curves)
        ax_success.plot(episodes, success.mean(axis=0), color=COLORS[key], label=spec.label)
        ax_success.fill_between(
            episodes,
            success.mean(axis=0) - success.std(axis=0),
            success.mean(axis=0) + success.std(axis=0),
            color=COLORS[key],
            alpha=0.15,
            linewidth=0,
        )
        if key in {"cliffwalking", "taxi"}:
            ax_reward.plot(episodes, reward.mean(axis=0), color=COLORS[key], label=spec.label)
            ax_reward.fill_between(
                episodes,
                reward.mean(axis=0) - reward.std(axis=0),
                reward.mean(axis=0) + reward.std(axis=0),
                color=COLORS[key],
                alpha=0.15,
                linewidth=0,
            )

    ax_success.set_title("Taxa de sucesso da política gulosa")
    ax_success.set_xlabel("Episódio de treinamento (escala log)")
    ax_success.set_ylabel("Taxa de sucesso")
    ax_success.set_ylim(-0.03, 1.03)
    ax_success.set_xscale("log")
    ax_success.legend(loc="lower right")

    ax_reward.set_title("Recompensa média da política gulosa")
    ax_reward.set_xlabel("Episódio de treinamento (escala log)")
    ax_reward.set_ylabel("Recompensa média por episódio")
    ax_reward.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_reward.set_xscale("log")
    ax_reward.legend(loc="lower right")

    save(fig, FIGURES / "fig_02_avaliacao_gulosa.png")


def figure_environment_comparison() -> None:
    agg = pd.read_csv(METRICS / "main_aggregated.csv")
    order = [key for key in MAIN_ENVIRONMENTS if key in set(agg["env_key"])]
    agg = agg.set_index("env_key").loc[order].reset_index()
    labels = [ENVIRONMENTS[key].short for key in agg["env_key"]]
    colors = [COLORS[key] for key in agg["env_key"]]
    positions = np.arange(len(agg))

    fig, (ax_success, ax_reward) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax_success.bar(
        positions,
        agg["success_mean"],
        yerr=agg["success_std"],
        color=colors,
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_success.set_xticks(positions, labels)
    ax_success.set_ylim(0, 1.08)
    ax_success.set_ylabel("Taxa de sucesso final")
    ax_success.set_title("Desempenho final da política gulosa")
    for x, value in zip(positions, agg["success_mean"]):
        ax_success.text(x, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)

    ax_reward.bar(
        positions,
        agg["reward_mean"],
        yerr=agg["reward_std"],
        color=colors,
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_reward.axhline(0, color="gray", linewidth=0.8)
    ax_reward.set_xticks(positions, labels)
    ax_reward.set_ylabel("Recompensa média final")
    ax_reward.set_title("Recompensa média final (± desvio entre sementes)")
    for x, value in zip(positions, agg["reward_mean"]):
        offset = 0.5 if value >= 0 else -0.9
        ax_reward.text(x, value + offset, f"{value:.2f}", ha="center", fontsize=8)

    save(fig, FIGURES / "fig_03_comparacao_ambientes.png")


def _numeric_value(label: str, prefix: str) -> float:
    match = re.search(rf"{prefix}=([0-9.]+)", label)
    return float(match.group(1)) if match else float("nan")


def _sweep_figure(kind: str, prefix: str, title: str, filename: str, xlabel: str) -> None:
    summary = pd.read_csv(METRICS / f"{kind}_summary.csv")
    agg = pd.read_csv(METRICS / f"{kind}_aggregated.csv")
    values = list(agg["sweep_value"])
    values.sort(key=lambda label: _numeric_value(label, prefix))

    fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for color, value in zip(SWEEP_COLORS, values):
        paths = paths_for(summary, f"{kind}__", value=value)
        mean, std, x = stack_curves(paths, "train_rewards", window=100)
        number = _numeric_value(value, prefix)
        ax_curve.plot(x, mean, color=color, label=f"{prefix} = {number:g}")
    ax_curve.set_xlabel("Episódio")
    ax_curve.set_ylabel("Recompensa média\n(janela móvel = 100)")
    ax_curve.set_title(title)
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.legend(loc="lower right")

    ordered = agg.set_index("sweep_value").loc[values].reset_index()
    xs = [_numeric_value(value, prefix) for value in ordered["sweep_value"]]
    ax_final.errorbar(
        xs,
        ordered["success_mean"],
        yerr=ordered["success_std"],
        marker="o",
        color="#333333",
        capsize=3,
    )
    ax_final.set_xlabel(xlabel)
    ax_final.set_ylabel("Taxa de sucesso final")
    ax_final.set_title("Desempenho final em função do hiperparâmetro")
    ax_final.set_ylim(-0.03, 1.03)
    save(fig, FIGURES / filename)


def figure_alpha() -> None:
    _sweep_figure(
        "alpha",
        "alpha",
        "Curvas de aprendizado variando a taxa de aprendizado (α)",
        "fig_04_alpha.png",
        "Taxa de aprendizado (α)",
    )


def figure_gamma() -> None:
    _sweep_figure(
        "gamma",
        "gamma",
        "Curvas de aprendizado variando o fator de desconto (γ)",
        "fig_05_gamma.png",
        "Fator de desconto (γ)",
    )


def _epsilon_label(value: str) -> str:
    if value.startswith("eps_fixo"):
        number = value.split("=")[1]
        return f"ε fixo = {float(number):g}"
    number = value.split("=")[1]
    return f"ε decai (fator {float(number):g})"


def figure_epsilon() -> None:
    summary = pd.read_csv(METRICS / "epsilon_summary.csv")
    agg = pd.read_csv(METRICS / "epsilon_aggregated.csv")
    values = [
        "eps_fixo=0.05",
        "eps_fixo=0.1",
        "eps_fixo=0.3",
        "decay=0.99",
        "decay=0.995",
        "decay=0.999",
    ]
    values = [value for value in values if value in set(agg["sweep_value"])]
    fig, (ax_schedule, ax_curve, ax_final) = plt.subplots(1, 3, figsize=(12.5, 4.0))

    for color, value in zip(SWEEP_COLORS, values):
        paths = paths_for(summary, "epsilon__", value=value)
        mean, std, x = stack_curves(paths, "train_rewards", window=100)
        label = _epsilon_label(value)
        ax_curve.plot(x, mean, color=color, label=label)

        episodes = np.arange(1, 4001)
        if value.startswith("eps_fixo"):
            epsilon = float(value.split("=")[1])
            schedule = np.full_like(episodes, epsilon, dtype=float)
        else:
            decay = float(value.split("=")[1])
            schedule = np.maximum(0.01, 1.0 * decay**episodes)
        ax_schedule.plot(episodes, schedule, color=color, label=label)

    ax_schedule.set_xlabel("Episódio")
    ax_schedule.set_ylabel("ε")
    ax_schedule.set_title("Cronogramas de exploração")
    ax_schedule.set_ylim(0, 1.02)
    ax_schedule.legend(loc="upper right")

    ax_curve.set_xlabel("Episódio")
    ax_curve.set_ylabel("Recompensa média\n(janela móvel = 100)")
    ax_curve.set_title("Curvas de aprendizado por cronograma de ε")
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.legend(loc="lower right")

    ordered = agg.set_index("sweep_value").loc[values].reset_index()
    positions = np.arange(len(ordered))
    ax_final.bar(
        positions,
        ordered["success_mean"],
        yerr=ordered["success_std"],
        color=SWEEP_COLORS[: len(ordered)],
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_final.set_xticks(positions, [_epsilon_label(v) for v in ordered["sweep_value"]], rotation=35, ha="right")
    ax_final.set_ylabel("Taxa de sucesso final")
    ax_final.set_title("Desempenho final por cronograma")
    ax_final.set_ylim(0, 1.08)
    for x, value in zip(positions, ordered["success_mean"]):
        ax_final.text(x, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)

    save(fig, FIGURES / "fig_06_epsilon.png")


def _draw_policy(ax, agent, spec, title: str) -> None:
    env = spec.make()
    try:
        desc = env.unwrapped.desc
        letters = np.array([[cell.decode() for cell in row] for row in desc])
    finally:
        env.close()

    size = letters.shape[0]
    values = np.full((size, size), np.nan)
    arrows = np.full((size, size), "", dtype=object)
    for row in range(size):
        for col in range(size):
            state = row * size + col
            if state in agent.q:
                q_row = agent.q[state]
                values[row, col] = q_row.max()
                arrows[row, col] = ACTION_ARROWS[int(np.argmax(q_row))]

    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(masked, cmap=cmap)
    for row in range(size):
        for col in range(size):
            letter = letters[row, col]
            if letter in {"H", "G"}:
                ax.text(col, row, letter, ha="center", va="center", fontsize=11, fontweight="bold")
            else:
                ax.text(col, row - 0.18, letter, ha="center", va="center", fontsize=7, color="white")
                ax.text(col, row + 0.20, arrows[row, col], ha="center", va="center", fontsize=13, color="white")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_title(title)
    ax.grid(False)
    return image


def figure_policy_maps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    specs = [ENVIRONMENTS["frozenlake_4x4_dry"], ENVIRONMENTS["frozenlake_4x4"]]
    for ax, spec in zip(axes, specs):
        config = QLearningConfig(
            alpha=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            episodes=5000,
            max_steps=100,
            seed=0,
            eval_every=500,
            eval_episodes=10,
        )
        _, agent = train_agent(spec, config, return_agent=True)
        image = _draw_policy(ax, agent, spec, spec.label)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="max Q(s, a)")
    fig.suptitle(
        "Valores de Q e política gulosa aprendida (seed = 0; S = início, F = gelo, H = buraco, G = objetivo)",
        fontsize=10,
    )
    save(fig, FIGURES / "fig_07_politica.png")


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_training_curves()
    figure_greedy_evaluation()
    figure_environment_comparison()
    figure_alpha()
    figure_gamma()
    figure_epsilon()
    figure_policy_maps()


if __name__ == "__main__":
    main()
