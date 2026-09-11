#!/usr/bin/env bash
# Automates data generation + evaluation for scan_output/run_{1..24}/manifest.json:
#   1. post-process recorded data and propagate annotations (run_transfer.sh)
#   2. annotate with naive Grounding DINO + SAM VLM baseline
#   3. evaluate both against Isaac ground truth masks
#
# Usage:
#   scripts/annotation_propagation/run_all_evaluations.sh [start] [end]
#
# Defaults to runs 1 through 24. Continues past a failing run (logged and
# skipped) unless --stop-on-error is passed.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCAN_DIR="$REPO_ROOT/scan_output"

START=1
END=24
STOP_ON_ERROR=0
for arg in "$@"; do
  case "$arg" in
    --stop-on-error) STOP_ON_ERROR=1 ;;
    *) ;;
  esac
done
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --stop-on-error) ;;
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

# Each step is its own process, so its CUDA context and host memory are
# released by the OS when it exits. This just makes that visible and gives
# the driver a moment to reclaim memory before the next process starts.
report_gpu() {
  local label="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] $label:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  fi
  sleep 2
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

  # run_step "post-process and propagate annotations (run $i)" \
  #   "$SCRIPT_DIR/run_transfer.sh" "$MANIFEST" \
  #   || ok=0
  # report_gpu "after transfer (run $i)"

  if [[ $ok -eq 1 ]]; then
    run_step "annotate with naive VLM baseline, zeroshot (run $i)" \
      conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/annotate_naive_vlm.py" "$MANIFEST" \
      --detection-mode zeroshot \
      --save-visualizations \
      --output-dir "$RUN_DIR/baseline_segment/naive_vlm_zeroshot_no_filter" \
      --no-filter-candidates \
      || ok=0
    report_gpu "after zeroshot no filter VLM (run $i)"
  fi

  if [[ $ok -eq 1 ]]; then
    run_step "annotate with naive VLM baseline, multi-shot (run $i)" \
      conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/annotate_naive_vlm.py" "$MANIFEST" \
      --detection-mode multi-shot \
      --save-visualizations \
      --output-dir "$RUN_DIR/baseline_segment/naive_vlm_multi_shot_no_filter" \
      --no-filter-candidates \
      || ok=0
    report_gpu "after multi-shot no filter VLM (run $i)"
  fi

  if [[ $ok -eq 1 ]]; then
    run_step "evaluate proposed method (run $i)" \
      conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/evaluate_segmentation_map.py" "$MANIFEST" \
      --predictions "$RUN_DIR/transfer_segment" \
      --output "$RUN_DIR/transfer_segment/map_report.json" \
      || ok=0
    report_gpu "after evaluating proposed method (run $i)"
  fi

  if [[ $ok -eq 1 ]]; then
    run_step "evaluate naive VLM baseline, zeroshot (run $i)" \
      conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/evaluate_segmentation_map.py" "$MANIFEST" \
      --predictions "$RUN_DIR/baseline_segment/naive_vlm_zeroshot_no_filter" \
      --output "$RUN_DIR/baseline_segment/naive_vlm_zeroshot_no_filter/map_report.json" \
      || ok=0
    report_gpu "after evaluating zeroshot no filter VLM (run $i)"
  fi

  if [[ $ok -eq 1 ]]; then
    run_step "evaluate naive VLM baseline, multi-shot (run $i)" \
      conda run --no-capture-output -n vla \
      python "$SCRIPT_DIR/evaluate_segmentation_map.py" "$MANIFEST" \
      --predictions "$RUN_DIR/baseline_segment/naive_vlm_multi_shot_no_filter" \
      --output "$RUN_DIR/baseline_segment/naive_vlm_multi_shot_no_filter/map_report.json" \
      || ok=0
    report_gpu "after evaluating multi-shot no filter VLM (run $i)"
  fi

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
  echo "All runs completed successfully."
else
  echo "Runs with failures or missing manifests: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
