#!/usr/bin/env bash
# Run the SAM 2 video baseline over an inclusive range of scan runs.
# Usage: run_all_sam2_video.sh START END [--stop-on-error] [-- SAM2_OPTIONS ...]
set -uo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 START END [--stop-on-error] [-- SAM2_OPTIONS ...]" >&2
  exit 2
fi

START=$1
END=$2
shift 2
if [[ ! $START =~ ^[0-9]+$ || ! $END =~ ^[0-9]+$ ]]; then
  echo "START and END must be nonnegative integers" >&2
  exit 2
fi
if (( END < START )); then
  echo "END must be greater than or equal to START" >&2
  exit 2
fi

STOP_ON_ERROR=0
EXTRA_ARGS=()
FORWARD=0
for argument in "$@"; do
  if (( FORWARD )); then
    EXTRA_ARGS+=("$argument")
  elif [[ $argument == "--stop-on-error" ]]; then
    STOP_ON_ERROR=1
  elif [[ $argument == "--" ]]; then
    FORWARD=1
  else
    echo "Unknown batch option: $argument (put SAM 2 options after --)" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
FAILED_RUNS=()

for (( RUN = START; RUN <= END; RUN++ )); do
  MANIFEST="$REPOSITORY_ROOT/scan_output/run_${RUN}/manifest.json"
  echo
  echo "===== SAM 2 video baseline: run $RUN ====="
  if [[ ! -f $MANIFEST ]]; then
    echo "Skipping run $RUN: manifest not found: $MANIFEST" >&2
    FAILED_RUNS+=("$RUN")
    if (( STOP_ON_ERROR )); then break; fi
    continue
  fi
  if ! "$SCRIPT_DIR/run_sam2_video.sh" "$MANIFEST" "${EXTRA_ARGS[@]}"; then
    echo "FAILED: SAM 2 video baseline for run $RUN" >&2
    FAILED_RUNS+=("$RUN")
    if (( STOP_ON_ERROR )); then break; fi
  fi
done

if (( ${#FAILED_RUNS[@]} )); then
  echo "Runs with failures or missing manifests: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
echo "All SAM 2 video baseline runs completed successfully."
