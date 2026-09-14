#!/usr/bin/env bash
set -euo pipefail

# Usage: run_training.sh TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END [SOURCE ...]
# Validation selects checkpoints during training; test is reserved for final metrics.
if (( $# < 6 )); then
  echo "Usage: $0 TRAIN_START TRAIN_END VAL_START VAL_END TEST_START TEST_END [SOURCE ...]" >&2
  exit 2
fi
TRAIN_START=$1; TRAIN_END=$2; VAL_START=$3; VAL_END=$4
TEST_START=$5; TEST_END=$6
shift 6
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
    --test-runs "$TEST_START-$TEST_END" \
    --source "$SOURCE" --output "$DATASET" --overwrite
  conda run --no-capture-output -n vla python \
    "$REPOSITORY_ROOT/scripts/yolo26_seg/train.py" \
    "$DATASET/dataset.yaml" \
    --project "$REPOSITORY_ROOT/scan_output/yolo26_seg" \
    --name "$SOURCE" --exist-ok
done
