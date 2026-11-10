"""Implementação tabular do algoritmo Q-learning.

O código segue a formulação clássica de Watkins (1989):

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

A tabela Q é armazenada em um dicionário ``estado -> vetor de valores``. Essa
escolha mantém a implementação independente do tipo de observação devolvido
pelo ambiente (inteiros no FrozenLake, CliffWalking e Taxi) e permite reutilizar
o mesmo agente em qualquer ambiente discreto do Gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QLearningConfig:
    """Hiperparâmetros e configuração de execução do Q-learning."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    episodes: int = 5000
    max_steps: int = 200
    seed: int = 0
    eval_every: int = 100
    eval_episodes: int = 10


class QLearningAgent:
    """Agente tabular com política epsilon-gulosa."""

    def __init__(self, n_actions: int, config: QLearningConfig) -> None:
        self.n_actions = int(n_actions)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.q: dict[Any, np.ndarray] = {}
        self.epsilon = float(config.epsilon_start)

    def q_values(self, state: Any) -> np.ndarray:
        """Devolve (criando se necessário) o vetor Q do estado."""
        values = self.q.get(state)
        if values is None:
            values = np.zeros(self.n_actions, dtype=np.float64)
            self.q[state] = values
        return values

    def act(self, state: Any, greedy: bool = False) -> int:
        """Seleciona uma ação.

        Durante o treinamento usa epsilon-guloso; na avaliação usa a política
        gulosa pura. Empates são desfeitos de forma aleatória para não enviesar
        a exploração.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        values = self.q_values(state)
        best = np.flatnonzero(values == values.max())
        if best.size == 1:
            return int(best[0])
        return int(self.rng.choice(best))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
    ) -> float:
        """Aplica a regra de atualização e devolve o erro de diferença temporal."""
        current = self.q_values(state)[action]
        if terminated:
            target = reward
        else:
            target = reward + self.config.gamma * float(np.max(self.q_values(next_state)))
        td_error = target - current
        self.q_values(state)[action] = current + self.config.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        """Decaimento multiplicativo do epsilon ao final de cada episódio."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def greedy_policy(self) -> dict[Any, int]:
        """Política gulosa derivada da tabela Q."""
        return {state: int(np.argmax(values)) for state, values in self.q.items()}
