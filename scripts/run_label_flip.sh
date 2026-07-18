#!/usr/bin/env bash
# Label-flipping ADVERSARIAL attack sweep for the MEDIA revision (R1.3/R2.3).
# A genuine data-poisoning attack (labels flipped to a wrong class on corrupted
# clients), distinct from the on-the-fly image corruption used elsewhere. Runs
# FedAgain vs the vanilla and coordinate-wise-median baselines under non-IID
# label skew at rho in {0.1, 0.3}.
#
#   4 datasets x 3 strategies x 2 rhos x 3 seeds = 72 runs.
#
# Runs tagged with `pm-label_flip` in the run_key, so they never collide with
# the image-corruption runs in results/experiment_ledger.csv.
#
# IMPORTANT: launch only when no image-corruption sweep is running (both mutate
# config.py and share the GPU). Resumable: completed runs are skipped.
#   setsid bash scripts/run_label_flip.sh </dev/null >results/logs/label_flip.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/label_flip_${STAMP}.log"

STRATEGIES="FedAgain FedAvg FedMedian"
DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"
RHOS="0.1 0.3"

echo "[label_flip] START $(date)" | tee -a "$LOG"
python scripts/run_label_flip.py \
  --strategies $STRATEGIES --datasets $DATASETS \
  --distribution non-iid-label --corrupt-ratios $RHOS --flip-ratio 1.0 \
  --num-clients 10 --num-rounds 10 --seeds $SEEDS >> "$LOG" 2>&1
echo "[label_flip] DONE $(date)" | tee -a "$LOG"
echo "[label_flip] results in results/experiment_ledger.csv (run_key tag pm-label_flip)" | tee -a "$LOG"
