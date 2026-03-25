#!/bin/bash
# ============================================================
# Step 3b — Curriculum phase 2, CNN  (expected: TRIVIALISES simple env)
# ============================================================
# Loads a phase-1 simple_cnn checkpoint and continues on the same simple
# env with init_length=7. The agent consistently reaches long lengths,
# trivialising the task and demonstrating CNN mastery before the hard env.
#
# Why init_length=15 does NOT work:
#   - Most random actions immediately cause self-collision -> replay buffer
#     floods with death experiences -> agent "learns" to die quickly.
#   - Without step_reward / distance_reward_scale the reward signal is
#     entirely absent until the agent eats food, making the gradient
#     uninformative for the vast majority of episodes.
# init_length=7 is a manageable curriculum step: the agent immediately
# faces tail-avoidance risk but still has enough open space to navigate.
#
# Architecture must match simple_cnn (no double/dueling) so weights load.
# epsilon is reset to 0.3 automatically by train.py after load().
# lr is halved vs phase-1 to protect the learned food-navigation weights.
#
# Usage:
#   bash scripts/train_phase2_cnn.sh \
#     --load-checkpoint checkpoints/simple_cnn/final.pt

set -euo pipefail
cd "$(dirname "$0")/.."

uv run -m rl_snake.train \
  --agent-type   cnn     \
  --episodes     3000    \
  --max-steps    500     \
  --init-length  7       \
  --n-gold       1       \
  --gold-reward  10.0    \
  --death-reward -10.0   \
  --step-reward  -0.01   \
  --distance-reward-scale 0.1 \
  --lr           5e-4    \
  --gamma        0.99    \
  --epsilon-start  0.3   \
  --epsilon-end    0.01  \
  --epsilon-decay  0.995 \
  --batch-size   64      \
  --buffer-capacity 100000 \
  --target-update  1000  \
  --train-freq   4       \
  --save-every   500     \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     phase2_cnn  \
  --save-video           \
  "$@"
