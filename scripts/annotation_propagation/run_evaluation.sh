#!/usr/bin/env bash
# Evaluates already-generated annotations for scan_output/run_{1..24} against
# Isaac ground truth masks. Does not run transfer propagation or naive VLM
# annotation — run those first (see run_all_evaluations.sh, or run_transfer.sh
# / annotate_naive_vlm.py directly) so the prediction directories below exist.
#
# For each run it evaluates:
#   run_<n>/transfer_segment
#   run_<n>/baseline_segment/naive_vlm_zeroshot
#   run_<n>/baseline_segment/naive_vlm_multi_shot
#
# Usage:
#   scripts/annotation_propagation/run_evaluation.sh [start] [end]
#
# Defaults to runs 1 through 24. Continues past a failing or missing run
# (logged and skipped) unless --stop-on-error is passed.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCAN_DIR="$REPO_ROOT/scan_output"

START=1
END=24
STOP_ON_ERROR=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --stop-on-error) STOP_ON_ERROR=1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then START="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then END="${POSITIONAL[1]}"; fi

FAILED_RUNS=()

run_step() {
  local desc="$1"
  shift
  echo "----- $desc -----"
  if ! "$@"; then
    echo "FAILED: $desc" >&2
    return 1
  fi
}

evaluate_predictions() {
  local manifest="$1"
  local predictions="$2"
  local label="$3"
  local run="$4"

  if [[ ! -d "$predictions" ]]; then
    echo "Skipping $label (run $run): $predictions not found" >&2
    return 1
  fi

  run_step "evaluate $label (run $run)" \
    conda run --no-capture-output -n vla \
    python "$SCRIPT_DIR/evaluate_segmentation_map.py" "$manifest" \
    --predictions "$predictions" \
    --output "$predictions/map_report.json"
}

for i in $(seq "$START" "$END"); do
  RUN_DIR="$SCAN_DIR/run_${i}"
  MANIFEST="$RUN_DIR/manifest.json"
  echo ""
  echo "===== Run $i ($MANIFEST) ====="

  if [[ ! -f "$MANIFEST" ]]; then
    echo "Skipping run $i: $MANIFEST not found" >&2
    FAILED_RUNS+=("$i")
    continue
  fi

  ok=1
  evaluate_predictions "$MANIFEST" "$RUN_DIR/transfer_segment" \
    "proposed method" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$RUN_DIR/baseline_segment/naive_vlm_zeroshot" \
    "naive VLM zeroshot" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$RUN_DIR/baseline_segment/naive_vlm_multi_shot" \
    "naive VLM multi-shot" "$i" || ok=0

  if [[ $ok -eq 0 ]]; then
    FAILED_RUNS+=("$i")
    if [[ $STOP_ON_ERROR -eq 1 ]]; then
      echo "Stopping on error at run $i" >&2
      break
    fi
  fi
done

echo ""
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
  echo "All runs evaluated successfully."
else
  echo "Runs with failures or missing outputs: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
