#!/usr/bin/env bash
# Run every analysis on the v2 results tree. Called by run_queue.py when the
# queue is done; safe to run by hand at any time (partial results are fine).
# Outputs stay inside results_v2/ (figures in results_v2/paper_figures/); the
# manuscript images are NOT touched.
set -u
cd "$(dirname "$0")/../.."
export FEDAGAIN_RESULTS_DIR=results_v2 FEDAGAIN_DATA_VERSION=v2 FEDAGAIN_CLINICAL_ONLY=1
PY="${PYTHON:-python3}"
OUT=results_v2/analysis
mkdir -p "$OUT"
status=0
for s in clinical_metrics baselines degradation_curve convergence label_flip \
         epsilon_sensitivity overhead trust_ablation rare_client_bias; do
  echo "== analyze_$s"
  if "$PY" "analysis/analyze_$s.py" > "$OUT/$s.txt" 2>&1; then
    echo "   ok -> $OUT/$s.txt"
  else
    echo "   FAILED (exit $?) -> $OUT/$s.txt"; status=1
  fi
done
exit $status
