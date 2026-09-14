#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 scan_output/<run>/manifest.json [transfer options ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$1"
shift

# Refine the recorded route and canonicalize its artifacts before annotation.
conda run --no-capture-output -n vla \
  python "$REPO_ROOT/scripts/collect_data/post_process_scan_data.py" \
  "$MANIFEST" --no-backup

# These defaults match the Kinova/Isaac table. Flags supplied by the caller are
# appended afterward, so argparse lets an experiment override any default.
conda run --no-capture-output -n vla \
  python "$SCRIPT_DIR/transfer_annotations_test.py" "$MANIFEST" \
  --table-origin 0 0 -0.20 \
  --table-z-axis 0 0 1 \
  --table-clearance 0.003 \
  --outlier-neighbors 20 \
  --outlier-std-ratio 2.0 \
  --save-visualizations \
  --maximum-center-distance 0.05 \
  --minimum-area-ratio 0.1 \
  --maximum-area-ratio 2.5 \
  "$@"
