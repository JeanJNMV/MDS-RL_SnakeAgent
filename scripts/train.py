import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rl_snake.agents import BaseAgent
from rl_snake.env import SnakeEnv
from rl_snake.rewards import BaseReward
from rl_snake.states import BaseStateEncoder

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )

    env = SnakeEnv()

    agent: BaseAgent = instantiate(cfg.agent)
    encoder: BaseStateEncoder = instantiate(cfg.encoder)
    reward_wrapper: BaseReward = instantiate(cfg.reward)

    shaped_episode_rewards = []
    raw_episode_rewards = []
    episode_lengths = []

    for episode in range(cfg.episodes):
        episode_seed = None if cfg.seed is None else cfg.seed + episode
        stats = env.run_episode(
            agent=agent,
            encoder=encoder,
            reward_wrapper=reward_wrapper,
            seed=episode_seed,
            train=True,
            max_iterations=cfg.get("episode_max_iterations", None),
        )

        shaped_episode_rewards.append(stats.shaped_return)
        raw_episode_rewards.append(stats.raw_return)
        episode_lengths.append(stats.steps)

        if (episode + 1) % cfg.log_every == 0:
            mean_shaped_reward = (
                sum(shaped_episode_rewards[-cfg.log_every :]) / cfg.log_every
            )
            mean_raw_reward = sum(raw_episode_rewards[-cfg.log_every :]) / cfg.log_every
            mean_steps = sum(episode_lengths[-cfg.log_every :]) / cfg.log_every
            logger.info(
                "Episode %4d/%d | Avg Raw Reward (%d ep): %8.3f | Avg Shaped Reward (%d ep): %8.3f | Avg Steps (%d ep): %6.1f",
                episode + 1,
                cfg.episodes,
                cfg.log_every,
                mean_raw_reward,
                cfg.log_every,
                mean_shaped_reward,
                cfg.log_every,
                mean_steps,
            )

    if shaped_episode_rewards:
        overall_shaped_reward = sum(shaped_episode_rewards) / len(
            shaped_episode_rewards
        )
        overall_raw_reward = sum(raw_episode_rewards) / len(raw_episode_rewards)
        overall_steps = sum(episode_lengths) / len(episode_lengths)
        logger.info(
            "Training complete | Episodes: %d | Mean Raw Reward: %8.3f | Mean Shaped Reward: %8.3f | Mean Steps: %6.1f",
            len(shaped_episode_rewards),
            overall_raw_reward,
            overall_shaped_reward,
            overall_steps,
        )

    if cfg.save.enabled:
        agent.save(cfg.save.path)
        logger.info("Agent saved to %s", cfg.save.path)

    env.close()


if __name__ == "__main__":
    main()
