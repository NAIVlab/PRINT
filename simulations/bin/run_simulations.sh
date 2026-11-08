#!/usr/bin/env bash
# Run all four simulation scenarios and all three methods with fixed replicates.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: [environment settings] bash bin/run_simulations.sh

Runs PRINT, direct regression, and Kallus-Zhou in all four scenarios, then
draws Figure 1. Every Monte Carlo replicate is retained.

Settings:
  PYTHON       Python executable (default: python)
  RESULTS_DIR  Fresh output directory (default: simulations/results)
  NS           Comma-separated sample sizes (default: 1000,...,1000000)
  N_RUNS       Number of replicates with seeds 0,...,N_RUNS-1 (default: 50)
  KZ_CAP       Kallus-Zhou training sample cap (default: 50000)

Quick check:
  NS=1000 N_RUNS=1 RESULTS_DIR=/tmp/print-check bash bin/run_simulations.sh
EOF
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Unexpected argument. Use --help." >&2; exit 2; }

SIM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
RESULTS_DIR=${RESULTS_DIR:-"$SIM_DIR/results"}
export NS=${NS:-1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000}
export N_RUNS=${N_RUNS:-50}
export KZ_CAP=${KZ_CAP:-50000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"
for name in plugin_results.csv kz_mild_strong.csv kz_mild_weak.csv kz_severe_strong.csv kz_severe_weak.csv run_config.json; do
  [[ ! -e "$name" ]] || { echo "Output already exists: $PWD/$name. Choose a fresh RESULTS_DIR." >&2; exit 1; }
done

export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.matplotlib"}
"$PYTHON" - <<'PY'
import json
import os
import platform
import numpy, scipy, sklearn, matplotlib
config = {
    'sample_sizes': [int(n) for n in os.environ['NS'].split(',')],
    'replicates': int(os.environ['N_RUNS']),
    'replication_seeds': list(range(int(os.environ['N_RUNS']))),
    'evaluation_seed': 32768,
    'evaluation_size': 20000,
    'kz_cap': int(os.environ['KZ_CAP']),
    'crossfit_seed': int(os.environ.get('CF_SEED', '12345')),
    'crossfit_folds': int(os.environ.get('K_FOLDS', '5')),
    'curvature_floor': float(os.environ.get('B2_FLOOR', '0.1')),
    'diagonal_weighting': int(os.environ.get('W_DIAG', '1')),
    'python': platform.python_version(),
    'versions': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn, matplotlib)},
}
with open('run_config.json', 'x') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
PY

PARAM_SET=all OUT=plugin_results.csv "$PYTHON" "$SIM_DIR/bin/run_print.py"
for cell in mild_strong mild_weak severe_strong severe_weak; do
  PARAM_SET=$cell OUT="kz_$cell.csv" "$PYTHON" "$SIM_DIR/bin/run_kz18.py"
done
"$PYTHON" "$SIM_DIR/bin/make_fig1.py" --results-dir "$PWD" --output-dir "$PWD/figs_out"
echo "Simulation outputs: $PWD"
