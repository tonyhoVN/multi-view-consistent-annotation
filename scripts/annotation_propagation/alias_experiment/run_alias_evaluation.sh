#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 START_RUN [END_RUN] [alias-runner options ...]" >&2
  echo "  A single START_RUN with no numeric END_RUN evaluates just that run." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

START_RUN="$1"
shift
if [[ ! "$START_RUN" =~ ^[0-9]+$ ]]; then
  echo "START_RUN must be a nonnegative integer: $START_RUN" >&2
  exit 2
fi

END_RUN="$START_RUN"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  END_RUN="$1"
  shift
fi

if [[ "$END_RUN" -lt "$START_RUN" ]]; then
  echo "END_RUN ($END_RUN) must be >= START_RUN ($START_RUN)" >&2
  exit 2
fi

FAILED_RUNS=()

for RUN_INDEX in $(seq "$START_RUN" "$END_RUN"); do
  MANIFEST="$REPO_ROOT/scan_output/run_${RUN_INDEX}/manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Scan manifest does not exist, skipping run $RUN_INDEX: $MANIFEST" >&2
    FAILED_RUNS+=("$RUN_INDEX")
    continue
  fi

  # Each method gets a fresh process so CUDA/model memory is released between runs.
  for METHOD in transfer; do
    echo "===== Alias annotation + evaluation: run $RUN_INDEX, $METHOD ====="
    conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/annotate_evaluate_aliases.py" "$MANIFEST" \
      --method "$METHOD" \
      --table-origin 0 0 -0.20 \
      --table-z-axis 0 0 1 \
      --table-clearance 0.003 \
      --outlier-neighbors 20 \
      --outlier-std-ratio 2.0 \
      --save-visualizations \
      --maximum-center-distance 0.1 \
      --minimum-area-ratio 0.2 \
      --maximum-area-ratio 2.5 \
      "$@"
  done

  echo "Alias results saved under: $(dirname -- "$MANIFEST")/alias_segment"
done

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Runs with missing manifests: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
