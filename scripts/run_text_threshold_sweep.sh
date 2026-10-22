#!/usr/bin/env bash
# Train independent Text/IMDb models for a routing-threshold sensitivity study.
# Usage: bash scripts/run_text_threshold_sweep.sh DATA_DIR [GPU] [EPOCHS] [THRESHOLDS]
set -euo pipefail

DATA_DIR="${1:?path containing the prepared IMDb cache}"
GPU="${2:-0}"
EPOCHS="${3:-100}"
THRESHOLDS="${4:-0.3,0.4,0.5,0.6,0.7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/threshold_sweep/text_train"
mkdir -p "$OUT/checkpoints" "$OUT/records" "$OUT/logs"

IFS=',' read -r -a VALUES <<< "$THRESHOLDS"
for THRESHOLD in "${VALUES[@]}"; do
  TAG="$(python - "$THRESHOLD" <<'PY'
import sys
print(f"t{float(sys.argv[1]):.2f}".replace(".", "p"))
PY
)"
  RUN_NAME="text_threshold_${TAG}_seed1919_${EPOCHS}ep"
  LOG="$OUT/logs/${RUN_NAME}.log"

  echo "[RUN] threshold=$THRESHOLD epochs=$EPOCHS log=$LOG"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python train_lra.py \
    --seed 1919 \
    --task text \
    --text_pipeline ssm_benchmark \
    --data_dir "$DATA_DIR" \
    --batch_size 32 \
    --epochs "$EPOCHS" \
    --lr 5e-4 \
    --weight_decay 1e-2 \
    --d_model 64 \
    --n_layers 6 \
    --d_state 64 \
    --d_conv 4 \
    --expand 2 \
    --pool mean \
    --head_input spike \
    --spike_mode ste \
    --target_rate 0.3 \
    --routing_threshold "$THRESHOLD" \
    --ssm_benchmark_schedule 1 \
    --warmup_steps 0 \
    --save_dir "$OUT/checkpoints" \
    --record_dir "$OUT/records" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "$LOG"
done

