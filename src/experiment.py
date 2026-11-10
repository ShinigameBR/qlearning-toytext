"""Laço de treinamento, avaliação e persistência dos resultados."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .environments import EnvironmentSpec, is_success
from .qlearning import QLearningAgent, QLearningConfig


def evaluate_policy(
    env,
    agent: QLearningAgent,
    n_episodes: int,
    max_steps: int,
    success_rule: str,
) -> dict[str, float]:
    """Executa a política gulosa sem exploração e agrega as métricas."""
    rewards: list[float] = []
    successes: list[float] = []
    steps: list[int] = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        t = 0
        while not (terminated or truncated) and t < max_steps:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            t += 1
        rewards.append(total_reward)
        steps.append(t)
        if success_rule == "positive_reward":
            successes.append(1.0 if (terminated and total_reward > 0) else 0.0)
        else:
            successes.append(1.0 if terminated else 0.0)

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_steps_mean": float(np.mean(steps)),
    }


def train_agent(
    spec: EnvironmentSpec,
    config: QLearningConfig,
    verbose: bool = False,
    return_agent: bool = False,
) -> Any:
    """Treina um agente em um ambiente e devolve curvas e métricas finais."""
    train_env = spec.make()
    eval_env = spec.make()
    try:
        n_actions = int(train_env.action_space.n)
        agent = QLearningAgent(n_actions=n_actions, config=config)

        train_rewards: list[float] = []
        train_steps: list[int] = []
        epsilons: list[float] = []
        eval_episodes: list[int] = []
        eval_rewards: list[float] = []
        eval_reward_stds: list[float] = []
        eval_success: list[float] = []
        eval_steps: list[float] = []

        # O mesmo seed garante que a sequência de transições seja reproduzível.
        state, _ = train_env.reset(seed=config.seed)
        eval_env.reset(seed=config.seed + 10_000)

        start = time.perf_counter()
        for episode in range(1, config.episodes + 1):
            if episode > 1:
                state, _ = train_env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            t = 0
            while not (terminated or truncated) and t < config.max_steps:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += float(reward)
                t += 1

            train_rewards.append(total_reward)
            train_steps.append(t)
            epsilons.append(agent.epsilon)
            agent.decay_epsilon()

            if episode % config.eval_every == 0 or episode == config.episodes:
                metrics = evaluate_policy(
                    eval_env,
                    agent,
                    n_episodes=config.eval_episodes,
                    max_steps=spec.max_steps,
                    success_rule=spec.success_rule,
                )
                eval_episodes.append(episode)
                eval_rewards.append(metrics["eval_reward_mean"])
                eval_reward_stds.append(metrics["eval_reward_std"])
                eval_success.append(metrics["eval_success_rate"])
                eval_steps.append(metrics["eval_steps_mean"])

            if verbose and episode % max(1, config.episodes // 10) == 0:
                print(
                    f"  {spec.label}: episódio {episode}/{config.episodes} "
                    f"(recompensa recente={np.mean(train_rewards[-100:]):.2f})"
                )

        elapsed = time.perf_counter() - start
    finally:
        train_env.close()
        eval_env.close()

    result = {
        "env_key": spec.key,
        "env_label": spec.label,
        "env_id": spec.env_id,
        "seed": config.seed,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay": config.epsilon_decay,
        "episodes": config.episodes,
        "max_steps": config.max_steps,
        "final_epsilon": float(epsilons[-1]) if epsilons else config.epsilon_start,
        "train_rewards": np.asarray(train_rewards, dtype=np.float64),
        "train_steps": np.asarray(train_steps, dtype=np.int32),
        "epsilons": np.asarray(epsilons, dtype=np.float64),
        "eval_episodes": np.asarray(eval_episodes, dtype=np.int32),
        "eval_rewards": np.asarray(eval_rewards, dtype=np.float64),
        "eval_reward_stds": np.asarray(eval_reward_stds, dtype=np.float64),
        "eval_success": np.asarray(eval_success, dtype=np.float64),
        "eval_steps": np.asarray(eval_steps, dtype=np.float64),
        "final_eval_reward_mean": float(eval_rewards[-1]),
        "final_eval_reward_std": float(eval_reward_stds[-1]),
        "final_eval_success_rate": float(eval_success[-1]),
        "final_eval_steps_mean": float(eval_steps[-1]),
        "q_table_size": int(len(agent.q)),
        "training_time_s": float(elapsed),
        "config_json": json.dumps(asdict(config), sort_keys=True),
    }
    if return_agent:
        return result, agent
    return result


def save_run(
    result: dict[str, Any],
    directory: Path,
    prefix: str = "",
    tag: str = "",
) -> Path:
    """Salva as curvas de um experimento em um arquivo .npz."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    filename = f"{prefix}{result['env_key']}__seed{result['seed']}"
    if tag:
        filename += f"__{tag}"
    path = directory / f"{filename}.npz"
    np.savez_compressed(path, **arrays)
    return path


def summary_row(result: dict[str, Any], experiment: str) -> dict[str, Any]:
    """Extrai apenas as métricas escalares de um resultado."""
    row = {
        key: value
        for key, value in result.items()
        if not isinstance(value, np.ndarray)
    }
    row["experiment"] = experiment
    return row


def save_summary(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return frame
