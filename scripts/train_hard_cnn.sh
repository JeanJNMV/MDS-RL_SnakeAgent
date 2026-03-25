#!/bin/bash
# ============================================================
# Step 5 — Hard env, CNN agent  (expected: SUCCEEDS via curriculum)
# ============================================================
# REQUIRES: checkpoints/simple_cnn/final.pt  (run train_simple_cnn.sh first)
#
# Why training from scratch on the hard env fails:
#   - The agent must learn two skills simultaneously: basic navigation AND
#     dynamic-obstacle avoidance.
#   - With a random policy, the agent almost never eats food (obstacle kills
#     it first) -> reward signal is entirely "death" -> agent stalls.
#   - Epsilon_decay=0.9995 @ ~100 steps/ep reaches the floor in ~60 episodes,
#     locking in a nearly-random policy for the remaining 14 940 episodes.
#
# Curriculum fix:
#   1. Load simple_cnn weights — the agent already knows food navigation.
#   2. Re-explore at epsilon=0.5 so it can discover obstacle-avoidance.
#   3. Use a much slower epsilon decay (0.99995 reaches 0.05 after ~600 eps).
#   4. Lower LR (1e-4) to fine-tune without overwriting the navigation policy.
#   5. The CNN's 4-frame stack gives it temporal context to infer obstacle
#      velocity — exactly what the MLP cannot do.
#
# Usage:
#   bash scripts/train_hard_cnn.sh \
#     --load-checkpoint checkpoints/simple_cnn/final.pt

set -euo pipefail
cd "$(dirname "$0")/.."

uv run -m rl_snake.train \
  --agent-type   cnn     \
  --episodes     10000   \
  --max-steps    500     \
  --n-gold       1       \
  --n-silver     2       \
  --n-poison     1       \
  --n-dynamic-obstacles  1 \
  --obstacle-wiggle-range 3 \
  --gold-reward   10.0   \
  --silver-reward  5.0   \
  --poison-reward -5.0   \
  --death-reward  -10.0  \
  --step-reward   -0.01  \
  --distance-reward-scale 0.1 \
  --n-frames     4       \
  --double-dqn           \
  --dueling              \
  --n-step       3       \
  --lr           1e-4    \
  --gamma        0.99    \
  --epsilon-start  0.5   \
  --epsilon-end    0.05  \
  --epsilon-decay  0.99995 \
  --batch-size   128     \
  --buffer-capacity 200000 \
  --target-update  2000  \
  --target-tau     0.005 \
  --grad-clip      10.0  \
  --train-freq   4       \
  --save-every   1000    \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     hard_cnn    \
  --save-video           \
  "$@"
