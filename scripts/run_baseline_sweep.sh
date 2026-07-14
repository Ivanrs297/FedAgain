#!/usr/bin/env bash
# Baseline comparison sweep for the MEDIA revision (Reviewer 1.5 / 2.4):
# FedAgain vs FedAvg / FedProx / FedMedian / Bulyan / FedBN / Scaffold / MOON,
# under non-IID label skew with 30% corrupted clients (severity 3).
#
# Resumable: completed runs are skipped by the tracker. Launch detached with:
#   setsid bash scripts/run_baseline_sweep.sh </dev/null >results/logs/baselines.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/baseline_sweep_${STAMP}.log"

STRATEGIES="FedAvg FedProx FedMedian Bulyan FedBN Scaffold MOON FedAgain"
DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"

echo "[baselines] START $(date)" | tee -a "$LOG"
python scripts/run_baselines.py \
  --strategies $STRATEGIES --datasets $DATASETS \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --num-clients 10 --num-rounds 10 --seeds $SEEDS >> "$LOG" 2>&1
echo "[baselines] DONE $(date)" | tee -a "$LOG"
echo "[baselines] results in results/experiment_ledger.csv" | tee -a "$LOG"
