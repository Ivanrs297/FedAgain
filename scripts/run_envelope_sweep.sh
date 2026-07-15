#!/usr/bin/env bash
# Corruption-robustness ENVELOPE sweep for the MEDIA revision (R1.5/R2.4, honest
# robustness curve): FedAgain vs FedAvg/FedProx/FedMedian/Bulyan/FedBN/Scaffold/MOON,
# under non-IID label skew, severity 3, at corruption ratios rho in {0.1, 0.4}.
# (rho=0.3 was already swept -> completed runs are skipped by the tracker, so
#  together the three points give a 3-point degradation curve per method.)
#
#   4 datasets x 8 strategies x 2 rhos x 3 seeds = 192 runs (~13h).
#
# Resumable: completed runs are skipped by the tracker. Launch detached with:
#   setsid bash scripts/run_envelope_sweep.sh </dev/null >results/logs/envelope.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/envelope_sweep_${STAMP}.log"

STRATEGIES="FedAvg FedProx FedMedian Bulyan FedBN Scaffold MOON FedAgain"
DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"
RHOS="0.1 0.4"

echo "[envelope] START $(date)" | tee -a "$LOG"
for RHO in $RHOS; do
  echo "[envelope] === rho=$RHO $(date) ===" | tee -a "$LOG"
  python scripts/run_baselines.py \
    --strategies $STRATEGIES --datasets $DATASETS \
    --distribution non-iid-label --corrupt-ratio "$RHO" --severity 3 \
    --num-clients 10 --num-rounds 10 --seeds $SEEDS >> "$LOG" 2>&1
done
echo "[envelope] DONE $(date)" | tee -a "$LOG"
echo "[envelope] results in results/experiment_ledger.csv" | tee -a "$LOG"
