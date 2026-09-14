#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 scan_output/run_N/manifest.json [OPTIONS ...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
conda run --no-capture-output -n vla \
  python "$SCRIPT_DIR/annotate_sam2_transfer.py" "$@"
