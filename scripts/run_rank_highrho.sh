#!/usr/bin/env bash
# Rank normalization under heavy corruption (Reviewer 2.5, follow-up).
#
# At rho=0.3 the rank-based normalization matched or beat the deployed
# sum-to-one rule (74.95 vs 73.71 mean) with markedly lower variance. But rank
# weights are bounded by construction: with K=10 clients and a perfect ranking
# the attainable benign/corrupted suppression ratio is capped at 3.50 for
# rho=0.3 and 2.43 for rho=0.6, whereas sum-to-one measured 4.42 on Jonathan.
# The prediction is therefore that rank degrades once the required suppression
# exceeds its ceiling. This sweep tests it where the ceiling bites hardest.
#
# The sum-to-one reference runs at rho=0.5/0.6 already exist in the ledger
# (canonical FedAgain), so only rank needs to be run: 2 rho x 4 datasets x 3
# seeds = 24 runs.
#
# Launch detached:
#   setsid bash scripts/run_rank_highrho.sh </dev/null \
#       >results/logs/rank_highrho.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/rank_highrho_${STAMP}.log"

log() { echo "[rank-highrho] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

START=$(($(wc -l < results/experiment_ledger.csv) - 1))
log "starting; ledger has ${START} runs (expecting +24 = 912)."

for RHO in 0.5 0.6; do
  log "rank at rho=${RHO} (12 runs)"
  python scripts/run_normalization_ablation.py \
    --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
    --variants rank --seeds 42 43 44 \
    --distribution non-iid-label --corrupt-ratio "${RHO}" --severity 3 \
    --num-clients 10 --num-rounds 10 >> "$LOG" 2>&1
  log "rho=${RHO} exit=$?"
done

git checkout config.py >> "$LOG" 2>&1 && log "config.py restored"

END=$(($(wc -l < results/experiment_ledger.csv) - 1))
FAILED=$(awk -F, 'NR>1 && $4=="failed"' results/experiment_ledger.csv | wc -l)
log "DONE. ledger ${START} -> ${END}; failed=${FAILED}"
