#!/usr/bin/env bash
# Run the SAM2 + transfer-reanchor baseline over an inclusive run range.
set -uo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 START END [--stop-on-error] [-- OPTIONS ...]" >&2
  exit 2
fi
START=$1
END=$2
shift 2
if [[ ! $START =~ ^[0-9]+$ || ! $END =~ ^[0-9]+$ ]] || (( END < START )); then
  echo "Require nonnegative integers satisfying START <= END" >&2
  exit 2
fi

STOP_ON_ERROR=0
EXTRA_ARGS=()
FORWARD=0
for argument in "$@"; do
  if (( FORWARD )); then
    EXTRA_ARGS+=("$argument")
  elif [[ $argument == --stop-on-error ]]; then
    STOP_ON_ERROR=1
  elif [[ $argument == -- ]]; then
    FORWARD=1
  else
    echo "Unknown batch option: $argument (put baseline options after --)" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
FAILED_RUNS=()
for (( RUN = START; RUN <= END; RUN++ )); do
  MANIFEST="$REPOSITORY_ROOT/scan_output/run_${RUN}/manifest.json"
  echo
  echo "===== SAM2 + transfer reanchor: run $RUN ====="
  if [[ ! -f $MANIFEST ]] || ! "$SCRIPT_DIR/run_sam2_transfer.sh" "$MANIFEST" "${EXTRA_ARGS[@]}"; then
    echo "FAILED: hybrid baseline for run $RUN" >&2
    FAILED_RUNS+=("$RUN")
    if (( STOP_ON_ERROR )); then break; fi
  fi
done

if (( ${#FAILED_RUNS[@]} )); then
  echo "Runs with failures: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
echo "All SAM2 + transfer-reanchor runs completed successfully."
