#!/usr/bin/env bash
# HIGH-corruption envelope extension for the MEDIA revision (R1.5/R2.4).
# Extends the degradation curve past the 50% Byzantine breakdown point:
# rho in {0.5, 0.6}. With 10 clients this is 5 and 6 corrupted clients, so at
# rho=0.6 the corrupted majority breaks FedMedian's coordinate-wise-median
# guarantee -- the regime where FedAgain's soft, prior-free re-weighting should
# degrade more gracefully. Same grid as run_envelope_sweep.sh otherwise.
#
#   4 datasets x 8 strategies x 2 rhos x 3 seeds = 192 runs.
#
# Resumable: completed runs are skipped by the tracker. Launch detached with:
#   setsid bash scripts/run_envelope_high.sh </dev/null >results/logs/envelope_high.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/envelope_high_${STAMP}.log"

STRATEGIES="FedAvg FedProx FedMedian Bulyan FedBN Scaffold MOON FedAgain"
DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"
RHOS="0.5 0.6"

echo "[envelope-high] START $(date)" | tee -a "$LOG"
for RHO in $RHOS; do
  echo "[envelope-high] === rho=$RHO $(date) ===" | tee -a "$LOG"
  python scripts/run_baselines.py \
    --strategies $STRATEGIES --datasets $DATASETS \
    --distribution non-iid-label --corrupt-ratio "$RHO" --severity 3 \
    --num-clients 10 --num-rounds 10 --seeds $SEEDS >> "$LOG" 2>&1
done
echo "[envelope-high] DONE $(date)" | tee -a "$LOG"
echo "[envelope-high] results in results/experiment_ledger.csv" | tee -a "$LOG"
