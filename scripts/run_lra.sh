#!/usr/bin/env bash
# Reproduce the five canonical runs listed in configs/lra.yaml.
# Usage: bash scripts/run_lra_logged.sh <task> <data-root> [gpu] [output-root] [seed]
set -euo pipefail
TASK="${1:?task: listops|text|retrieval|image|pathfinder}"
DATA_ROOT="${2:?root containing prepared datasets}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${4:-$ROOT/outputs}"
SEED_OVERRIDE="${5:-}"

LISTOPS_DIR="$DATA_ROOT/listops-1000"
[[ -d "$LISTOPS_DIR" ]] || LISTOPS_DIR="$DATA_ROOT/lra_release/listops-1000"
AAN_DIR="$DATA_ROOT/lra_release/lra_release/tsv_data"
[[ -d "$AAN_DIR" ]] || AAN_DIR="$DATA_ROOT/LRA/lra_release/lra_release/tsv_data"
PATHFINDER_DIR="$DATA_ROOT/lra_release/lra_release"

case "$TASK" in
  listops) ARGS=(--seed 1919 --task listops --listops_pipeline ssm_benchmark --data_dir "$LISTOPS_DIR" --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 1e-2 --d_model 64 --n_layers 10 --d_state 64 --d_conv 4 --expand 3 --pool mean --head_input spike --ssm_benchmark_schedule 1 --warmup_steps 0) ;;
  text) ARGS=(--seed 1919 --task text --text_pipeline ssm_benchmark --data_dir "$DATA_ROOT/imdb" --batch_size 32 --epochs 100 --lr 5e-4 --weight_decay 1e-2 --d_model 64 --n_layers 6 --d_state 64 --d_conv 4 --expand 2 --pool mean --head_input spike --ssm_benchmark_schedule 1 --warmup_steps 0) ;;
  retrieval) ARGS=(--seed 42 --task retrieval --data_dir "$AAN_DIR" --hf_local_path "$AAN_DIR" --max_length 4096 --batch_size 16 --epochs 20 --lr 1e-3 --weight_decay 1e-2 --d_model 256 --n_layers 6 --d_state 64 --d_conv 4 --expand 2 --pool mean --head_input cont --use_scheduler 1 --warmup_ratio 0 --warmup_steps 1000) ;;
  image) ARGS=(--seed 42 --task image --data_dir "$DATA_ROOT" --s4_image_pipeline 1 --s4_image_aug crop_flip --s4_image_val_split 0.1 --s4_image_split_seed 42 --batch_size 128 --epochs 200 --lr 1e-3 --weight_decay 0 --d_model 256 --n_layers 6 --d_state 64 --d_conv 4 --expand 2 --pool mean --head_input cont --use_scheduler 1 --warmup_ratio 0.1) ;;
  pathfinder) ARGS=(--seed 1919 --task pathfinder --data_dir "$PATHFINDER_DIR" --pathfinder_resolution 32 --pathfinder_level hard --pathfinder_split_seed 42 --pathfinder_continuous 1 --pathfinder_center 1 --batch_size 32 --epochs 200 --lr 1e-4 --weight_decay 1e-2 --d_model 128 --n_layers 6 --d_state 64 --d_conv 4 --expand 4 --pool mean --head_input spike --spike_mode ste --target_rate 0.3 --use_scheduler 1 --warmup_ratio 0 --warmup_steps 5) ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac
if [[ -n "$SEED_OVERRIDE" ]]; then
  ARGS+=(--seed "$SEED_OVERRIDE")
fi
cd "$ROOT"
mkdir -p "$OUTPUT_ROOT/$TASK/checkpoints" "$OUTPUT_ROOT/$TASK/records"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python train_lra.py \
  "${ARGS[@]}" \
  --save_dir "$OUTPUT_ROOT/$TASK/checkpoints" \
  --record_dir "$OUTPUT_ROOT/$TASK/records" \
  --run_name "${TASK}_spike_mamba${SEED_OVERRIDE:+_seed${SEED_OVERRIDE}}"
