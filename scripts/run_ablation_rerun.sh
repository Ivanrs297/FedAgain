#!/usr/bin/env bash
# Re-run the trust-score ablation under the CURRENT code state, then sweep the
# alternative normalization strategies (Reviewer 2.5).
#
# Why the re-run: the original 66-run ablation (2026-07-14) was dropped from
# results/experiment_ledger.csv when the ledger was rebuilt, and it is not
# reproducible anyway -- re-executing the same run keys with the same seeds
# yields different accuracies (up to ~10 points on Jonathan), because the
# pipeline sets the RNG seeds but does not pin cuDNN determinism. As a result
# tab:trust_ablation and tab:baseline_comparison disagree on FedAgain's own
# accuracy. Re-running every non-default trust mode against the ledger's
# canonical combined_mult runs makes both tables mutually consistent.
#
# combined_mult is deliberately NOT re-run: those 12 runs are already in the
# ledger and are the ones feeding the baseline / degradation / convergence
# tables. The tracker skips completed runs, so it is safe either way.
#
# Launch detached:
#   setsid bash scripts/run_ablation_rerun.sh </dev/null \
#       >results/logs/ablation_rerun.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/ablation_rerun_${STAMP}.log"

log() { echo "[ablation-rerun] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

START=$(($(wc -l < results/experiment_ledger.csv) - 1))
log "starting; ledger has ${START} runs."

DATASETS="DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10"
SEEDS="42 43 44"

# --- Phase 1: the unweighted trust formulations (36 runs) -------------------
log "phase 1/3: benchmark_only, divergence_only, additive (36 runs)"
python scripts/run_trust_ablation.py \
  --datasets ${DATASETS} --seeds ${SEEDS} \
  --modes benchmark_only divergence_only additive \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --num-clients 10 --num-rounds 10 >> "$LOG" 2>&1
log "phase 1 exit=$?"

# --- Phase 2: weighted_additive at the three (alpha,beta) points (36 runs) --
# The paper claims "no tested (alpha,beta) recovers the multiplicative
# behaviour" but never reports the values; these are them.
for AB in "2.0 1.0" "1.0 2.0" "0.5 1.0"; do
  set -- $AB
  log "phase 2/3: weighted_additive alpha=$1 beta=$2 (12 runs)"
  python scripts/run_trust_ablation.py \
    --datasets ${DATASETS} --seeds ${SEEDS} \
    --modes weighted_additive --alpha "$1" --beta "$2" \
    --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
    --num-clients 10 --num-rounds 10 >> "$LOG" 2>&1
  log "phase 2 (alpha=$1 beta=$2) exit=$?"
done

# --- Phase 3: alternative normalizations (36 runs) --------------------------
log "phase 3/3: softmax(tau=0.5), softmax(tau=2.0), rank (36 runs)"
python scripts/run_normalization_ablation.py \
  --datasets ${DATASETS} --seeds ${SEEDS} \
  --variants softmax:0.5 softmax:2.0 rank \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --num-clients 10 --num-rounds 10 >> "$LOG" 2>&1
log "phase 3 exit=$?"

# --- Restore config.py (the runners rewrite it) -----------------------------
git checkout config.py >> "$LOG" 2>&1 && log "config.py restored"

END=$(($(wc -l < results/experiment_ledger.csv) - 1))
FAILED=$(awk -F, 'NR>1 && $4=="failed"' results/experiment_ledger.csv | wc -l)
log "DONE. ledger ${START} -> ${END} (expected +108 = 888); failed=${FAILED}"
