"""Agrega os resultados por ambiente/configuração e calcula convergência."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import load_curve  # noqa: E402

METRICS = ROOT / "results" / "metrics"
CURVES = ROOT / "results" / "curves"


def convergence_episode(eval_episodes: np.ndarray, eval_success: np.ndarray, final_success: float) -> float:
    """Primeiro ponto (com confirmação no ponto seguinte) que atinge 90% do desempenho final."""
    if final_success <= 0:
        return float("nan")
    target = 0.9 * final_success
    reached = eval_success >= target
    for index in range(len(reached) - 1):
        if reached[index] and reached[index + 1]:
            return float(eval_episodes[index])
    if reached.any():
        return float(eval_episodes[int(np.argmax(reached))])
    return float("nan")


def add_convergence(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    values = []
    for _, row in summary.iterrows():
        filename = f"{prefix}{row['env_key']}__seed{int(row['seed'])}"
        tag = row.get("run_tag", "")
        if isinstance(tag, str) and tag:
            filename += f"__{tag}"
        path = CURVES / f"{filename}.npz"
        if not path.exists():
            values.append(float("nan"))
            continue
        data = load_curve(path)
        values.append(
            convergence_episode(
                data["eval_episodes"],
                data["eval_success"],
                float(row["final_eval_success_rate"]),
            )
        )
    summary = summary.copy()
    summary["convergence_episode"] = values
    return summary


def aggregate(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = summary.groupby(group_cols, dropna=False)
    result = grouped.agg(
        runs=("seed", "count"),
        success_mean=("final_eval_success_rate", "mean"),
        success_std=("final_eval_success_rate", "std"),
        reward_mean=("final_eval_reward_mean", "mean"),
        reward_std=("final_eval_reward_mean", "std"),
        steps_mean=("final_eval_steps_mean", "mean"),
        steps_std=("final_eval_steps_mean", "std"),
        convergence_mean=("convergence_episode", "mean"),
        convergence_std=("convergence_episode", "std"),
        time_mean=("training_time_s", "mean"),
        qsize_mean=("q_table_size", "mean"),
        episodes=("episodes", "first"),
    ).reset_index()
    # Média das recompensas finais de cada execução também é útil para tabela.
    result["reward_overall_std"] = grouped["final_eval_reward_mean"].std().to_numpy()
    return result


def _pick_best(agg: pd.DataFrame) -> pd.Series:
    """Escolhe a melhor configuração: sucesso, depois convergência e recompensa.

    Empates na taxa de sucesso são comuns com apenas cinco sementes, então a
    convergência (menor episódio) é usada como critério de desempate.
    """
    frame = agg.copy()
    frame["_convergence"] = frame["convergence_mean"].fillna(float("inf"))
    frame = frame.sort_values(
        ["success_mean", "_convergence", "reward_mean"],
        ascending=[False, True, False],
    )
    return frame.iloc[0]


def _best_alpha(agg: pd.DataFrame) -> float:
    row = _pick_best(agg)
    return float(str(row["sweep_value"]).split("=")[1])


def _best_gamma(agg: pd.DataFrame) -> float:
    row = _pick_best(agg)
    return float(str(row["sweep_value"]).split("=")[1])


def _best_epsilon(agg: pd.DataFrame) -> dict[str, float]:
    row = _pick_best(agg)
    value = str(row["sweep_value"])
    if value.startswith("eps_fixo"):
        epsilon = float(value.split("=")[1])
        return {
            "epsilon_start": epsilon,
            "epsilon_end": epsilon,
            "epsilon_decay": 1.0,
        }
    decay = float(value.split("=")[1])
    return {
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": decay,
    }


def main() -> None:
    main_summary = pd.read_csv(METRICS / "main_summary.csv")
    main_summary = add_convergence(main_summary, prefix="main__")
    main_summary.to_csv(METRICS / "main_runs_with_convergence.csv", index=False)

    main_agg = aggregate(main_summary, ["env_key", "env_label"])
    main_agg.to_csv(METRICS / "main_aggregated.csv", index=False)
    print("=== Agregado principal ===")
    print(main_agg.to_string(index=False))

    best: dict[str, float] = {}
    for kind in ["alpha", "gamma", "epsilon"]:
        summary = pd.read_csv(METRICS / f"{kind}_summary.csv")
        summary = add_convergence(summary, prefix=f"{kind}__")
        summary.to_csv(METRICS / f"{kind}_runs_with_convergence.csv", index=False)
        agg = aggregate(summary, ["sweep_value"])
        agg.to_csv(METRICS / f"{kind}_aggregated.csv", index=False)
        print(f"\n=== Agregado {kind} ===")
        print(agg.to_string(index=False))

        if kind == "alpha":
            best["alpha"] = _best_alpha(agg)
        elif kind == "gamma":
            best["gamma"] = _best_gamma(agg)
        else:
            best.update(_best_epsilon(agg))

    import json

    (METRICS / "best_config.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print("\n=== Melhor configuração (estudos isolados) ===")
    print(best)

    tuned_path = METRICS / "tuned_summary.csv"
    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path)
        tuned = add_convergence(tuned, prefix="tuned__")
        tuned.to_csv(METRICS / "tuned_runs_with_convergence.csv", index=False)
        tuned_agg = aggregate(tuned, ["env_key", "env_label"])
        tuned_agg.to_csv(METRICS / "tuned_aggregated.csv", index=False)
        print("\n=== Agregado ajustado ===")
        print(tuned_agg.to_string(index=False))


if __name__ == "__main__":
    main()
