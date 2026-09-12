#!/usr/bin/env bash
set -euo pipefail

# Usage: run_training.sh TRAIN_START TRAIN_END VAL_START VAL_END [SOURCE ...]
# Val runs are used only to prepare the later ground-truth validation split.
if (( $# < 4 )); then
  echo "Usage: $0 TRAIN_START TRAIN_END VAL_START VAL_END [SOURCE ...]" >&2
  exit 2
fi
TRAIN_START=$1; TRAIN_END=$2; VAL_START=$3; VAL_END=$4
shift 4
if (( $# == 0 )); then
  SOURCES=(transfer naive_vlm_zeroshot naive_vlm_multi_shot)
else
  SOURCES=("$@")
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
for SOURCE in "${SOURCES[@]}"; do
  DATASET="$REPOSITORY_ROOT/scan_output/yolo26_seg_datasets/$SOURCE"
  conda run --no-capture-output -n vla python \
    "$REPOSITORY_ROOT/scripts/yolo26_seg/prepare_dataset.py" \
    --scan-root "$REPOSITORY_ROOT/scan_output" \
    --train-runs "$TRAIN_START-$TRAIN_END" \
    --val-runs "$VAL_START-$VAL_END" \
    --source "$SOURCE" --output "$DATASET" --overwrite
  conda run --no-capture-output -n vla python \
    "$REPOSITORY_ROOT/scripts/yolo26_seg/train.py" \
    "$DATASET/dataset.yaml" \
    --project "$REPOSITORY_ROOT/scan_output/yolo26_seg" \
    --name "$SOURCE" --exist-ok
done
