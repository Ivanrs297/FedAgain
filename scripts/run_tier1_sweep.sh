#!/usr/bin/env bash
# Tier-1 experiment sweep for the MEDIA revision: #2 trust ablation and
# #3 multiplicative-vs-additive sensitivity.
#
# Safe to re-run: completed runs are skipped by the tracker, so if this crashes
# or the machine reboots, just launch it again and it resumes where it stopped.
#
# Launch detached (survives closing the terminal / Claude session):
#   setsid bash scripts/run_tier1_sweep.sh </dev/null >results/logs/tier1.out 2>&1 &
set -u

cd "$(dirname "$0")/.."          # repo root
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/tier1_sweep_${STAMP}.log"

DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"
COMMON="--distribution non-iid-label --corrupt-ratio 0.3 --severity 3 --num-clients 10 --num-rounds 10 --seeds $SEEDS"

echo "[sweep] START $(date)" | tee -a "$LOG"

# --- #2 ablation (E-only / D-only / E*D) + #3 additive (main modes) ----------
echo "[sweep] phase 1: ablation + additive  $(date)" | tee -a "$LOG"
python scripts/run_trust_ablation.py --datasets $DATASETS $COMMON \
  --modes benchmark_only divergence_only combined_mult additive >> "$LOG" 2>&1

# --- #3 weighted_additive (alpha,beta) grid on two datasets ------------------
echo "[sweep] phase 2: weighted_additive grid  $(date)" | tee -a "$LOG"
for ab in "2 1" "1 2" "0.5 1"; do
  set -- $ab
  echo "[sweep]   weighted_additive alpha=$1 beta=$2  $(date)" | tee -a "$LOG"
  python scripts/run_trust_ablation.py --datasets DAUDON_MIX CIFAR10 $COMMON \
    --modes weighted_additive --alpha "$1" --beta "$2" >> "$LOG" 2>&1
done

echo "[sweep] DONE $(date)" | tee -a "$LOG"
echo "[sweep] results in results/experiment_ledger.csv, results/runs/, results/rounds/, results/trust_scores/" | tee -a "$LOG"
