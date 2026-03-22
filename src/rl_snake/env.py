from dataclasses import dataclass

import gymnasium as gym

from rl_snake.agents import BaseAgent
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder


@dataclass
class EpisodeStats:
    raw_return: float
    shaped_return: float
    steps: int
    terminated: bool
    truncated: bool
    iteration_limit_reached: bool = False


class SnakeEnv:
    """Thin environment wrapper to centralize Snake-v1 creation and lifecycle."""

    def __init__(self, render_mode: str | None = None, **options):
        self.env = gym.make("Snake-v1", render_mode=render_mode, **options)

    def close(self) -> None:
        self.env.close()

    def run_episode(
        self,
        agent: BaseAgent,
        encoder: BaseStateEncoder,
        reward_wrapper: BaseReward,
        seed: int | None = None,
        *,
        train: bool = True,
        max_iterations: int | None = None,
    ) -> "EpisodeStats":
        obs, info = self.env.reset(seed=seed)
        reward_wrapper.reset()
        state = encoder.encode(obs, info)

        raw_return = 0.0
        shaped_return = 0.0
        steps = 0
        terminated = False
        truncated = False

        done = False
        iteration_limit_reached = False
        while not done and (max_iterations is None or steps < max_iterations):
            action = agent.choose_action(state)
            next_obs, raw_reward, terminated, truncated, next_info = self.env.step(
                action
            )
            next_state = encoder.encode(next_obs, next_info)
            # Raw reward measures true environment performance.
            # Shaped reward is only the optimization signal.
            shaped_reward = reward_wrapper.compute(
                state=state,
                action=action,
                next_state=next_state,
                raw_reward=raw_reward,
                terminated=terminated,
                truncated=truncated,
                info=next_info,
            )

            done = terminated or truncated

            if train:
                agent.update(
                    state=state,
                    action=action,
                    reward=shaped_reward,
                    next_state=next_state,
                    done=done,
                )

            state = next_state
            raw_return += float(raw_reward)
            shaped_return += float(shaped_reward)
            steps += 1

        if not done and max_iterations is not None and steps >= max_iterations:
            iteration_limit_reached = True

        return EpisodeStats(
            raw_return=raw_return,
            shaped_return=shaped_return,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            iteration_limit_reached=iteration_limit_reached,
        )
