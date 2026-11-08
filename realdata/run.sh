#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

case "${1:-reproduce}" in
  -h|--help)
    cat <<'HELP'
Usage: [DATA_DIR=/path/to/data] [PYTHON=python] bash realdata/run.sh [reproduce|retrain]
  reproduce  Regenerate Table 1 and Figures 2-3 using the supplied PRINT checkpoint.
  retrain    Fit PRINT once at fixed seed 1, then regenerate the table and figures.
Environment: DATA_DIR (default: realdata/data), OUT_DIR, DEVICE (auto/cpu/cuda:0), PRINT_NET.
Regression is fitted once at seed 0 in either mode. No cluster jobs are submitted.
HELP
    exit 0
    ;;
  reproduce|retrain) MODE="${1:-reproduce}" ;;
  *) echo "Unknown mode: $1 (use --help)" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Expected at most one mode argument (use --help)." >&2
  exit 2
fi
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
export OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out}"
if [[ "$MODE" == retrain ]]; then
  "$PYTHON" -u "$SCRIPT_DIR/train_print.py"
  export PRINT_NET="$OUT_DIR/print_net.pt"
fi
"$PYTHON" -u "$SCRIPT_DIR/make_table1.py"
"$PYTHON" -u "$SCRIPT_DIR/make_figs.py"
