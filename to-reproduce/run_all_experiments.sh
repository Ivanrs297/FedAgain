#!/usr/bin/env bash
# =====================================================================
# Reproduce the full FedAgain paper sweep (MEDIA-D-26-01276), in order.
#
# Run from the REPOSITORY ROOT:
#     bash to-reproduce/run_all_experiments.sh
#
# Resumable: every runner shares the tracker ledger, so completed runs are
# skipped. Safe to interrupt and re-launch. To survive a disconnect:
#     mkdir -p results/logs
#     setsid bash to-reproduce/run_all_experiments.sh </dev/null \
#         >results/logs/reproduce_all.out 2>&1 &
#
# Total ~780 runs. Clinical datasets (A/B/C) require the private data; the
# CIFAR-10/MNIST portions reproduce from this repo alone. Edit DATASETS below
# to drop the clinical sets if you only have the public benchmarks.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1     # -> repository root
ROOT="$(pwd)"
mkdir -p results/logs

CLINICAL="DAUDON_MIX JONATHAN_MIX MYSTONE"
PUBLIC="CIFAR10"                       # MNIST is a sanity check; add if desired
SEEDS="42 43 44"

step () { echo; echo "=============================================================="; \
          echo ">>> $1"; echo "=============================================================="; }

# --- restore config.py on exit (runners leave it modified) ---
trap 'echo "[cleanup] restoring config.py"; git checkout -- config.py 2>/dev/null || true' EXIT

# 5.1  Clean IID baseline (Fig kidney_stone_no_corruption, §6.1) -------------
step "5.1  Clean IID baseline"
python scripts/run_baselines.py \
  --strategies FedAvg FedProx FedMedian Bulyan FedAgain \
  --datasets $CLINICAL \
  --distribution iid --corrupt-ratio 0.0 --severity 0 \
  --num-clients 10 --num-rounds 10 --seeds $SEEDS

# 5.2  Extended baselines @ rho=0.3 (Table/Fig baseline_comparison, §6.3) -----
step "5.2  Extended-baseline sweep (rho=0.3)"
bash scripts/run_baseline_sweep.sh

# 5.3  Recent robust baselines: FedDyn / RFLPA / BOBA (§6.3.1) ----------------
step "5.3  Recent robust baselines (FedDyn/RFLPA/BOBA)"
bash scripts/run_r24_baselines.sh

# 5.4  Corruption-degradation envelope (Table/Fig degradation, §6.4) ---------
step "5.4a  Envelope rho in {0.1, 0.4}"
bash scripts/run_envelope_sweep.sh
step "5.4b  Envelope tail rho in {0.5, 0.6}"
bash scripts/run_envelope_high.sh

# 5.5  Label-flipping poisoning attack (Fig label_flip, §6.5) ----------------
step "5.5  Label-flipping attack"
bash scripts/run_label_flip.sh

# 5.6  Trust-score ablation + normalization (§6.6 / §6.6.1) ------------------
step "5.6a  Trust-formulation ablation (logs trust scores -> also feeds §6.8)"
python scripts/run_trust_ablation.py \
  --datasets $CLINICAL --distribution non-iid-label \
  --corrupt-ratio 0.3 --severity 3 --seeds $SEEDS \
  --modes benchmark_only divergence_only combined_mult additive weighted_additive
step "5.6b  Normalization ablation"
python scripts/run_normalization_ablation.py \
  --datasets $CLINICAL --corrupt-ratio 0.3 --severity 3 --seeds $SEEDS \
  --variants softmax:0.5 softmax:2.0 rank
bash scripts/run_rank_highrho.sh

# 5.9  Epsilon sensitivity (Fig epsilon_sensitivity, §6.7) -------------------
step "5.9  Epsilon sensitivity sweep"
python scripts/run_epsilon_sensitivity.py \
  --datasets $CLINICAL --epsilons 1e-8 1e-6 1e-4 1e-2 1 \
  --corrupt-ratio 0.3 --severity 3 --seeds $SEEDS

echo
echo "=============================================================="
echo "All sweeps submitted. Ledger: results/experiment_ledger.csv"
echo "Now generate figures/tables (see to-reproduce/README.md, section 6):"
echo "  FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_baselines.py   # etc."
echo "=============================================================="
