#!/usr/bin/env bash
set -euo pipefail

# Usage: run_validation.sh [SOURCE ...]
if (( $# == 0 )); then
  SOURCES=(transfer naive_vlm_zeroshot naive_vlm_multi_shot)
else
  SOURCES=("$@")
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
for SOURCE in "${SOURCES[@]}"; do
  DATASET="$REPOSITORY_ROOT/scan_output/yolo26_seg_datasets/$SOURCE/dataset.yaml"
  EXPERIMENT="$REPOSITORY_ROOT/scan_output/yolo26_seg/$SOURCE"
  conda run --no-capture-output -n vla python \
    "$REPOSITORY_ROOT/scripts/yolo26_seg/validate.py" \
    "$DATASET" "$EXPERIMENT/weights/best.pt" \
    --project "$EXPERIMENT" \
    --output "$EXPERIMENT/ground_truth_validation.json" \
    --exist-ok
done

# Refresh one machine-readable comparison after all requested models finish.
conda run --no-capture-output -n vla python \
  "$REPOSITORY_ROOT/scripts/yolo26_seg/summarize_validation.py" \
  --validation-root "$REPOSITORY_ROOT/scan_output/yolo26_seg" \
  --output "$REPOSITORY_ROOT/scan_output/yolo26_seg_validation_summary.json"
