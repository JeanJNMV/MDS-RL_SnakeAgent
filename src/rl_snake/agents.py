import pickle
from collections import defaultdict

import numpy as np


class BaseAgent:
    def choose_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        pass


class RandomAgent(BaseAgent):
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions

    def choose_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(0, self.n_actions))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        state_round_decimals: int = 4,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_round_decimals = state_round_decimals
        self.rng = np.random.default_rng(seed)

        self.q_table = defaultdict(self._new_q_values)

    def _new_q_values(self) -> np.ndarray:
        return np.zeros(self.n_actions, dtype=np.float32)

    def _state_key(self, state: np.ndarray) -> tuple[float, ...]:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        rounded = np.round(flat, decimals=self.state_round_decimals)
        return tuple(float(v) for v in rounded)

    def choose_action(self, state: np.ndarray) -> int:
        state_key = self._state_key(state)
        if float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        current_q = float(self.q_table[state_key][action])
        next_max = 0.0 if done else float(np.max(self.q_table[next_state_key]))
        target = float(reward) + self.gamma * next_max
        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "state_round_decimals": self.state_round_decimals,
            "q_table": dict(self.q_table),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            state_round_decimals=payload["state_round_decimals"],
        )
        for key, values in payload["q_table"].items():
            agent.q_table[key] = np.asarray(values, dtype=np.float32)
        return agent
