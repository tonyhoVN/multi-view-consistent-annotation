#!/usr/bin/env bash
# Runs the mask transfer (propagation) algorithm across a range of runs.
# For each run <n>, this post-processes scan_output/run_<n>/manifest.json and
# propagates annotations via run_transfer.sh. It does not run the naive VLM
# baselines or evaluation — see run_evaluation.sh or run_all_evaluations.sh
# for those.
#
# Usage:
#   scripts/annotation_propagation/run_all_transfer.sh [start] [end] [--stop-on-error] [-- transfer options ...]
#
# Defaults to runs 1 through 24. Continues past a failing or missing run
# (logged and skipped) unless --stop-on-error is passed. Anything after a
# literal `--` is forwarded to run_transfer.sh for every run.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCAN_DIR="$REPO_ROOT/scan_output"

START=1
END=24
STOP_ON_ERROR=0
POSITIONAL=()
EXTRA_ARGS=()
FORWARD=0
for arg in "$@"; do
  if [[ $FORWARD -eq 1 ]]; then
    EXTRA_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    --stop-on-error) STOP_ON_ERROR=1 ;;
    --) FORWARD=1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then START="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then END="${POSITIONAL[1]}"; fi

FAILED_RUNS=()

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

  echo "----- transfer annotations (run $i) -----"
  if "$SCRIPT_DIR/run_transfer.sh" "$MANIFEST" "${EXTRA_ARGS[@]}"; then
    report_gpu "after transfer (run $i)"
  else
    echo "FAILED: transfer annotations (run $i)" >&2
    FAILED_RUNS+=("$i")
    if [[ $STOP_ON_ERROR -eq 1 ]]; then
      echo "Stopping on error at run $i" >&2
      break
    fi
  fi
done

echo ""
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
  echo "All runs transferred successfully."
else
  echo "Runs with failures or missing manifests: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
