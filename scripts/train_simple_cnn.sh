#!/bin/bash
# ============================================================
# Step 2 — Simple env, CNN agent  (expected: SUCCEEDS)
# ============================================================
# Same env as simple_mlp. Proves the full-grid CNN also solves
# the easy game, enabling a fair apples-to-apples comparison
# before moving to the hard environment.
# n_frames=1 here (single frame, no temporal context needed).
#
# Usage: bash scripts/train_simple_cnn.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run -m rl_snake.train \
  --agent-type   cnn     \
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
  --run-name     simple_cnn  \
  --save-video           \
  "$@"
