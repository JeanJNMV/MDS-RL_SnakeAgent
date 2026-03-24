from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class SnakeEnv:
    """Simple Snake environment using NumPy.

    Grid encoding in observation:
        0 = empty
        1 = snake body
        2 = snake head
        3 = food
        4 = obstacle

    Actions:
        0 = up
        1 = right
        2 = down
        3 = left
    """

    ACTION_TO_DELTA = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1),
    }

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        init_length: int = 3,
        seed: int | None = None,
        food_reward: float = 1.0,
        death_reward: float = -1.0,
        step_reward: float = 0.0,
        max_steps: int | None = None,
        allow_reverse: bool = False,
        obstacles: list[tuple[int, int]] | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.init_length = init_length
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.allow_reverse = allow_reverse

        self.rng = np.random.default_rng(seed)

        self.obstacles = set(obstacles or [])

        self.snake: list[tuple[int, int]] = []
        self.direction = RIGHT
        self.food: tuple[int, int] | None = None
        self.done = False
        self.score = 0
        self.steps = 0

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment and return the first observation."""
        self.done = False
        self.score = 0
        self.steps = 0
        self.direction = RIGHT

        center_row = self.height // 2
        center_col = self.width // 2

        self.snake = [(center_row, center_col - i) for i in range(self.init_length)]

        if any(pos in self.obstacles for pos in self.snake):
            raise ValueError("Initial snake overlaps an obstacle.")

        self._spawn_food()
        return self._get_observation()

    def step(self, action: int) -> StepResult:
        """Apply one action and return observation, reward, done, info."""
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"score": self.score, "steps": self.steps},
            )

        self.steps += 1

        action = int(action)
        if action not in self.ACTION_TO_DELTA:
            raise ValueError(f"Invalid action: {action}")

        action = self._sanitize_action(action)
        self.direction = action

        head_row, head_col = self.snake[0]
        d_row, d_col = self.ACTION_TO_DELTA[action]
        new_head = (head_row + d_row, head_col + d_col)

        ate_food = new_head == self.food

        if self._is_collision(new_head, grow=ate_food):
            self.done = True
            return StepResult(
                observation=self._get_observation(),
                reward=self.death_reward,
                done=True,
                info={"score": self.score, "steps": self.steps},
            )

        self.snake.insert(0, new_head)

        reward = self.step_reward

        if ate_food:
            self.score += 1
            reward += self.food_reward
            self._spawn_food()
        else:
            self.snake.pop()

        if self.max_steps is not None and self.steps >= self.max_steps:
            self.done = True
            return StepResult(
                observation=self._get_observation(),
                reward=reward,
                done=True,
                info={
                    "score": self.score,
                    "steps": self.steps,
                    "termination_reason": "max_steps",
                },
            )

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=False,
            info={"score": self.score, "steps": self.steps},
        )

    def render(self) -> None:
        """Very simple text rendering for debugging."""
        obs = self._get_observation()
        symbols = {
            0: ".",
            1: "o",
            2: "H",
            3: "F",
            4: "#",
        }

        for row in obs:
            print(" ".join(symbols[int(cell)] for cell in row))
        print()

    def sample_action(self) -> int:
        """Random action, useful for testing."""
        return int(self.rng.integers(0, 4))

    def _sanitize_action(self, action: int) -> int:
        """Prevent direct reversal unless explicitly allowed."""
        if self.allow_reverse:
            return action

        opposite = {
            UP: DOWN,
            DOWN: UP,
            LEFT: RIGHT,
            RIGHT: LEFT,
        }

        if action == opposite[self.direction]:
            return self.direction

        return action

    def _is_collision(self, position: tuple[int, int], grow: bool) -> bool:
        """Check wall, obstacle, and self-collision.

        If grow=False, moving into the current tail is allowed,
        because the tail will move away this step.
        """
        row, col = position

        # Wall collision
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True

        # Obstacle collision
        if position in self.obstacles:
            return True

        # Self collision
        body_to_check = self.snake if grow else self.snake[:-1]
        if position in body_to_check:
            return True

        return False

    def _spawn_food(self) -> None:
        """Place food in a random free cell."""
        free_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.snake and (r, c) not in self.obstacles
        ]

        if not free_cells:
            self.food = None
            self.done = True
            return

        idx = int(self.rng.integers(len(free_cells)))
        self.food = free_cells[idx]

    def _get_observation(self) -> np.ndarray:
        """Return a grid observation as a NumPy array."""
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        for r, c in self.obstacles:
            grid[r, c] = 4

        for r, c in self.snake[1:]:
            grid[r, c] = 1

        if self.snake:
            head_r, head_c = self.snake[0]
            grid[head_r, head_c] = 2

        if self.food is not None:
            food_r, food_c = self.food
            grid[food_r, food_c] = 3

        return grid.copy()
