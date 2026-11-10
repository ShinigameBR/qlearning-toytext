"""Especificação dos ambientes Toy Text usados nos experimentos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class EnvironmentSpec:
    """Metadados e parâmetros de um ambiente."""

    key: str
    env_id: str
    label: str
    max_steps: int
    episodes: int
    success_rule: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    short: str = ""

    def make(self) -> gym.Env:
        return gym.make(self.env_id, **self.kwargs)


def is_success(spec: EnvironmentSpec, terminated: bool, episode_reward: float) -> bool:
    """Define sucesso de forma explícita para cada família de ambiente.

    No FrozenLake os buracos também encerram o episódio, mas com recompensa
    zero; por isso o sucesso exige recompensa positiva. No CliffWalking e no
    Taxi a terminação indica que a meta foi alcançada.
    """
    if spec.success_rule == "positive_reward":
        return bool(terminated and episode_reward > 0)
    return bool(terminated)


ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    "frozenlake_4x4": EnvironmentSpec(
        key="frozenlake_4x4",
        env_id="FrozenLake-v1",
        label="FrozenLake 4x4 (escorregadio)",
        short="FL 4x4\n(slippery)",
        max_steps=100,
        episodes=5000,
        success_rule="positive_reward",
        kwargs={"is_slippery": True},
    ),
    "frozenlake_4x4_dry": EnvironmentSpec(
        key="frozenlake_4x4_dry",
        env_id="FrozenLake-v1",
        label="FrozenLake 4x4 (determinístico)",
        short="FL 4x4\n(determinístico)",
        max_steps=100,
        episodes=5000,
        success_rule="positive_reward",
        kwargs={"is_slippery": False},
    ),
    "frozenlake_8x8": EnvironmentSpec(
        key="frozenlake_8x8",
        env_id="FrozenLake8x8-v1",
        label="FrozenLake 8x8 (escorregadio)",
        short="FL 8x8\n(slippery)",
        max_steps=200,
        episodes=10000,
        success_rule="positive_reward",
        kwargs={"is_slippery": True},
    ),
    "cliffwalking": EnvironmentSpec(
        key="cliffwalking",
        env_id="CliffWalking-v1",
        label="CliffWalking",
        short="CliffWalking",
        max_steps=100,
        episodes=3000,
        success_rule="terminated",
    ),
    "taxi": EnvironmentSpec(
        key="taxi",
        env_id="Taxi-v4",
        label="Taxi",
        short="Taxi",
        max_steps=200,
        episodes=5000,
        success_rule="terminated",
    ),
}

MAIN_ENVIRONMENTS = [
    "frozenlake_4x4",
    "frozenlake_4x4_dry",
    "frozenlake_8x8",
    "cliffwalking",
    "taxi",
]
