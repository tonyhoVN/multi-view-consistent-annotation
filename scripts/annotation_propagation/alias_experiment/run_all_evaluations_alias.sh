#!/usr/bin/env bash
# Evaluates already-generated alias-experiment annotations for
# scan_output/run_{start..end}/alias_segment against Isaac ground truth masks.
# Does not run transfer propagation or naive VLM annotation — generate those
# first with run_alias_evaluation.sh (or annotate_evaluate_aliases.py
# directly) so the prediction directories below exist.
#
# For each run it evaluates whichever of these prediction directories exist:
#   alias_segment/transfer_segment
#   alias_segment/naive_vlm_zeroshot
#   alias_segment/naive_vlm_multi_shot
#   alias_segment/naive_vlm_zeroshot_no_filter
#   alias_segment/naive_vlm_multi_shot_no_filter
#
# Usage:
#   scripts/annotation_propagation/alias_experiment/run_all_evaluations_alias.sh [start] [end] [--stop-on-error]
#
# Defaults to runs 1 through 24. Continues past a failing or missing run
# (logged and skipped) unless --stop-on-error is passed.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
EVALUATOR="$REPO_ROOT/scripts/annotation_propagation/evaluate_segmentation_map.py"
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

report_gpu() {
  local label="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] $label:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  fi
  sleep 2
}

evaluate_predictions() {
  local manifest="$1"
  local predictions="$2"
  local label="$3"
  local run="$4"

  if [[ ! -d "$predictions" ]]; then
    echo "Skipping $label (run $run): $predictions not found"
    return 0
  fi

  run_step "evaluate $label (run $run)" \
    conda run --no-capture-output -n vla \
    python "$EVALUATOR" "$manifest" \
    --predictions "$predictions" \
    --output "$predictions/map_report.json"
  local status=$?
  report_gpu "after evaluating $label (run $run)"
  return $status
}

for i in $(seq "$START" "$END"); do
  RUN_DIR="$SCAN_DIR/run_${i}"
  MANIFEST="$RUN_DIR/manifest.json"
  ALIAS_DIR="$RUN_DIR/alias_segment"
  echo ""
  echo "===== Run $i ($MANIFEST) ====="

  if [[ ! -f "$MANIFEST" ]]; then
    echo "Skipping run $i: $MANIFEST not found" >&2
    FAILED_RUNS+=("$i")
    continue
  fi

  ok=1
  evaluate_predictions "$MANIFEST" "$ALIAS_DIR/transfer_segment" \
    "alias transfer" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$ALIAS_DIR/naive_vlm_zeroshot" \
    "alias naive VLM zeroshot" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$ALIAS_DIR/naive_vlm_multi_shot" \
    "alias naive VLM multi-shot" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$ALIAS_DIR/naive_vlm_zeroshot_no_filter" \
    "alias naive VLM zeroshot no-filter" "$i" || ok=0
  evaluate_predictions "$MANIFEST" "$ALIAS_DIR/naive_vlm_multi_shot_no_filter" \
    "alias naive VLM multi-shot no-filter" "$i" || ok=0

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
