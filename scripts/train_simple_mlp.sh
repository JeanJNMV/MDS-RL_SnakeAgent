#!/bin/bash
# ============================================================
# Step 1 — Simple env, MLP agent  (expected: SUCCEEDS)
# ============================================================
# Baseline: lightweight 15-feature hand-crafted state vector.
# The simple env has no moving obstacles, no silver/poison food.
# 3 000 episodes is plenty; epsilon reaches its floor after
# ~100 eps (decay is per learn-step, not per episode).
# distance_reward_scale provides dense shaping early in training
# so the agent learns food navigation before self-collision.
#
# Usage: bash scripts/train_simple_mlp.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run -m rl_snake.train \
  --agent-type   mlp     \
  --episodes     3000    \
  --max-steps    500     \
  --n-gold       1       \
  --gold-reward  10.0    \
  --death-reward -10.0   \
  --step-reward  -0.01   \
  --distance-reward-scale 0.1 \
  --lr           1e-3    \
  --gamma        0.99    \
  --epsilon-start  1.0   \
  --epsilon-end    0.01  \
  --epsilon-decay  0.995 \
  --batch-size   64      \
  --buffer-capacity 100000 \
  --target-update  1000  \
  --train-freq   4       \
  --save-every   500     \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     simple_mlp  \
  --save-video           \
  "$@"
