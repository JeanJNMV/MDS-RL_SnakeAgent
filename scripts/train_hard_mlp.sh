#!/bin/bash
# ============================================================
# Step 4 — Hard env, MLP agent  (expected: FAILS / PLATEAUS)
# ============================================================
# The hard environment adds:
#   - 2 silver foods  (require spatial discrimination beyond direction flags)
#   - 1 poison food   (must actively avoid)
#   - 1 dynamic obstacle (size-3 wall that bounces back and forth)
#
# Why the MLP struggles here:
#   1. TEMPORAL BLINDNESS — the 15-feature vector uses the CURRENT obstacle
#      position only (no frame history). The agent cannot infer the obstacle's
#      direction or predict where it will be next step.
#   2. LOCAL FEATURES — danger is encoded as "is the cell immediately ahead /
#      left / right dangerous?" — it misses medium-range obstacle trajectories.
#   3. NO SPATIAL MEMORY — the agent can only react, never anticipate.
# Result: the agent stalls at short lengths, frequently running into the
# obstacle it couldn't predict would move into its path.
#
# Usage: bash scripts/train_hard_mlp.sh

set -euo pipefail
cd "$(dirname "$0")/.."

uv run -m rl_snake.train \
  --agent-type   mlp     \
  --episodes     5000    \
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
  --lr           1e-3    \
  --gamma        0.99    \
  --epsilon-start  1.0   \
  --epsilon-end    0.01  \
  --epsilon-decay  0.995 \
  --batch-size   64      \
  --buffer-capacity 100000 \
  --target-update  1000  \
  --train-freq   4       \
  --save-every   1000    \
  --save-dir     checkpoints \
  --wandb-project rl-snake   \
  --run-name     hard_mlp    \
  --save-video           \
  "$@"
