#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 scan_output/manifest_<run>.json [transfer options ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda run --no-capture-output -n vla \
  python "$SCRIPT_DIR/transfer_annotations_test.py" "$@"

