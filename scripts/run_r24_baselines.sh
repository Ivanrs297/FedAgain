#!/usr/bin/env bash
# R2.4 recent-robust-baseline sweep for the MEDIA revision.
# Sweeps the three newly implemented strategies -- FedDyn (R1.5), RFLPA, BOBA --
# across the full corruption grid so they fold into BOTH revision tables:
#   * rho=0.3 subset  -> tab:baseline_comparison (extended-baselines table)
#   * rho in {0.1,0.3,0.4,0.5,0.6} -> tab:degradation (5-point degradation curve)
# Same protocol as the 8-strategy envelope sweeps (non-IID label skew, severity 3,
# 10 clients, 10 rounds, three seeds), so the new rows are directly comparable.
#
#   3 strategies x 4 datasets x 5 rhos x 3 seeds = 180 runs.
#
# Notes on the two reference-based baselines (honest caveats, see the paper):
#   * RFLPA/BOBA carve a clean server root set (SERVER_ROOT_PER_CLASS); FedAgain
#     needs none. BOBA needs n-f > c-1, so it degenerates to the mean on CIFAR-10
#     (10 clients, 10 classes) and is effective only on the clinical sets (c=6).
#
# Resumable: completed runs are skipped by the tracker (the 552 pre-existing runs
# are untouched). Launch detached with:
#   setsid bash scripts/run_r24_baselines.sh </dev/null >results/logs/r24.out 2>&1 &
# The runner leaves config.py modified -> this script restores it at the end.
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/r24_baselines_${STAMP}.log"

STRATEGIES="FedDyn RFLPA BOBA"
DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"
RHOS="0.1 0.3 0.4 0.5 0.6"

echo "[r24] START $(date)" | tee -a "$LOG"
for RHO in $RHOS; do
  echo "[r24] === rho=$RHO $(date) ===" | tee -a "$LOG"
  python scripts/run_baselines.py \
    --strategies $STRATEGIES --datasets $DATASETS \
    --distribution non-iid-label --corrupt-ratio "$RHO" --severity 3 \
    --num-clients 10 --num-rounds 10 --seeds $SEEDS >> "$LOG" 2>&1
done

# Restore config.py to the committed baseline (the runner rewrites it per run).
git checkout config.py >> "$LOG" 2>&1 && echo "[r24] config.py restored" | tee -a "$LOG"
echo "[r24] DONE $(date)" | tee -a "$LOG"
echo "[r24] results in results/experiment_ledger.csv (expect +180 -> 732 completed)" | tee -a "$LOG"
