"""Executa todos os experimentos de Q-learning nos ambientes Toy Text.

Uso:
    python scripts/run_experiments.py            # executa tudo
    python scripts/run_experiments.py --quick    # validação rápida
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments import ENVIRONMENTS, MAIN_ENVIRONMENTS  # noqa: E402
from src.experiment import save_run, save_summary, summary_row, train_agent  # noqa: E402
from src.qlearning import QLearningConfig  # noqa: E402

METRICS_DIR = ROOT / "results" / "metrics"
CURVES_DIR = ROOT / "results" / "curves"

MAIN_SEEDS = list(range(10))
SWEEP_SEEDS = list(range(100, 105))


def _config_for(spec, seed: int, **overrides) -> QLearningConfig:
    values = dict(
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        episodes=spec.episodes,
        max_steps=spec.max_steps,
        seed=seed,
        eval_every=100,
        eval_episodes=10,
    )
    values.update(overrides)
    return QLearningConfig(**values)


def run_grid(
    name: str,
    jobs: list[tuple[object, QLearningConfig]],
    curve_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    total = len(jobs)
    started = time.perf_counter()
    for index, (spec, config) in enumerate(jobs, start=1):
        print(
            f"[{name}] {index}/{total} | {spec.label} | seed={config.seed} | "
            f"alpha={config.alpha} gamma={config.gamma} "
            f"eps={config.epsilon_start}->{config.epsilon_end} "
            f"(decay={config.epsilon_decay})",
            flush=True,
        )
        result = train_agent(spec, config)
        tag = "" if name in {"main", "tuned"} else _slug(_sweep_label(config, name))
        save_run(result, curve_dir, prefix=f"{name}__", tag=tag)
        row = summary_row(result, experiment=name)
        row["sweep_value"] = _sweep_label(config, name)
        row["run_tag"] = tag
        rows.append(row)
        print(
            f"    sucesso={result['final_eval_success_rate']:.3f} "
            f"recompensa={result['final_eval_reward_mean']:.2f} "
            f"passos={result['final_eval_steps_mean']:.1f} "
            f"tempo={result['training_time_s']:.1f}s",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(f"[{name}] concluído em {elapsed:.1f}s ({total} execuções)", flush=True)
    return rows


def _sweep_label(config: QLearningConfig, kind: str) -> str:
    if kind == "alpha":
        return f"alpha={config.alpha:g}"
    if kind == "gamma":
        return f"gamma={config.gamma:g}"
    if config.epsilon_decay == 1.0:
        return f"eps_fixo={config.epsilon_start:g}"
    return f"decay={config.epsilon_decay:g}"


def _slug(text: str) -> str:
    """Converte o rótulo da configuração em um nome de arquivo seguro."""
    return (
        text.replace("=", "-")
        .replace(".", "p")
        .replace(" ", "_")
        .replace(",", "-")
    )


def build_main_jobs(quick: bool) -> list[tuple[object, QLearningConfig]]:
    jobs = []
    seeds = MAIN_SEEDS[:1] if quick else MAIN_SEEDS
    for key in MAIN_ENVIRONMENTS:
        spec = ENVIRONMENTS[key]
        episodes = 300 if quick else spec.episodes
        for seed in seeds:
            jobs.append((spec, _config_for(spec, seed, episodes=episodes)))
    return jobs


def build_tuned_jobs(quick: bool) -> list[tuple[object, QLearningConfig]]:
    """Repete o treinamento com a melhor configuração encontrada nos estudos."""
    best_path = METRICS_DIR / "best_config.json"
    if not best_path.exists():
        raise FileNotFoundError(
            "best_config.json não encontrado; execute analyze_results.py após os estudos."
        )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    jobs = []
    seeds = MAIN_SEEDS[:1] if quick else MAIN_SEEDS
    for key in MAIN_ENVIRONMENTS:
        spec = ENVIRONMENTS[key]
        episodes = 300 if quick else spec.episodes
        for seed in seeds:
            jobs.append(
                (
                    spec,
                    _config_for(
                        spec,
                        seed,
                        alpha=best["alpha"],
                        gamma=best["gamma"],
                        epsilon_start=best["epsilon_start"],
                        epsilon_end=best["epsilon_end"],
                        epsilon_decay=best["epsilon_decay"],
                        episodes=episodes,
                    ),
                )
            )
    return jobs


def build_sweep_jobs(kind: str, quick: bool) -> list[tuple[object, QLearningConfig]]:
    spec = ENVIRONMENTS["frozenlake_4x4"]
    seeds = SWEEP_SEEDS[:1] if quick else SWEEP_SEEDS
    episodes = 300 if quick else 4000
    jobs = []
    if kind == "alpha":
        values = [0.01, 0.05, 0.1, 0.5, 1.0]
        for alpha in values:
            for seed in seeds:
                jobs.append((spec, _config_for(spec, seed, alpha=alpha, episodes=episodes)))
    elif kind == "gamma":
        values = [0.8, 0.9, 0.95, 0.99, 0.999]
        for gamma in values:
            for seed in seeds:
                jobs.append((spec, _config_for(spec, seed, gamma=gamma, episodes=episodes)))
    elif kind == "epsilon":
        schedules = [
            dict(epsilon_start=0.05, epsilon_end=0.05, epsilon_decay=1.0),
            dict(epsilon_start=0.10, epsilon_end=0.10, epsilon_decay=1.0),
            dict(epsilon_start=0.30, epsilon_end=0.30, epsilon_decay=1.0),
            dict(epsilon_start=1.00, epsilon_end=0.01, epsilon_decay=0.990),
            dict(epsilon_start=1.00, epsilon_end=0.01, epsilon_decay=0.995),
            dict(epsilon_start=1.00, epsilon_end=0.01, epsilon_decay=0.999),
        ]
        for schedule in schedules:
            for seed in seeds:
                jobs.append((spec, _config_for(spec, seed, episodes=episodes, **schedule)))
    else:  # pragma: no cover
        raise ValueError(kind)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="execução curta de validação")
    parser.add_argument(
        "--only",
        choices=["main", "alpha", "gamma", "epsilon", "tuned"],
        help="executa apenas um grupo de experimentos",
    )
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "main": build_main_jobs(args.quick),
        "alpha": build_sweep_jobs("alpha", args.quick),
        "gamma": build_sweep_jobs("gamma", args.quick),
        "epsilon": build_sweep_jobs("epsilon", args.quick),
    }
    if args.only == "tuned":
        groups = {"tuned": build_tuned_jobs(args.quick)}
        selected = ["tuned"]
    else:
        selected = [args.only] if args.only else list(groups)

    metadata = {
        "quick": args.quick,
        "main_seeds": MAIN_SEEDS[:1] if args.quick else MAIN_SEEDS,
        "sweep_seeds": SWEEP_SEEDS[:1] if args.quick else SWEEP_SEEDS,
        "groups": selected,
    }
    (METRICS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for name in selected:
        rows = run_grid(name, groups[name], CURVES_DIR)
        save_summary(rows, METRICS_DIR / f"{name}_summary.csv")


if __name__ == "__main__":
    main()
