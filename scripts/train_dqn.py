"""Train a DQN agent on the Snake environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import wandb
from rl_snake.agent import DQNAgent, get_state
from rl_snake.env import SnakeEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Snake")

    # Environment
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--food-reward", type=float, default=10.0)
    p.add_argument("--death-reward", type=float, default=-10.0)
    p.add_argument("--step-reward", type=float, default=0)

    # Agent
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--target-update", type=int, default=1_000)

    # Training
    p.add_argument("--episodes", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=500)

    # W&B
    p.add_argument("--wandb-project", type=str, default="rl-snake")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
    )

    env = SnakeEnv(
        height=args.height,
        width=args.width,
        seed=args.seed,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
        step_reward=args.step_reward,
        max_steps=args.max_steps,
    )

    agent = DQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, args.episodes + 1):
        env.reset()
        state = get_state(env)
        total_reward = 0.0
        episode_losses: list[float] = []

        while not env.done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state = get_state(env)

            agent.store_transition(state, action, result.reward, next_state, result.done)
            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += result.reward

        wandb.log(
            {
                "episode": episode,
                "score": env.score,
                "total_reward": total_reward,
                "steps": env.steps,
                "epsilon": agent.epsilon,
                "loss": float(np.mean(episode_losses)) if episode_losses else None,
            }
        )

        if episode % args.save_every == 0:
            ckpt_path = save_dir / f"dqn_ep{episode}.pt"
            agent.save(str(ckpt_path))
            wandb.save(str(ckpt_path))
            print(
                f"[{episode}/{args.episodes}]  score={env.score}  ε={agent.epsilon:.3f}  saved → {ckpt_path}"
            )

    final_path = save_dir / "dqn_final.pt"
    agent.save(str(final_path))
    wandb.save(str(final_path))
    wandb.finish()
    print(f"Training complete. Final checkpoint → {final_path}")


if __name__ == "__main__":
    train(parse_args())
