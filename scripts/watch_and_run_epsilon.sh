#!/usr/bin/env bash
# Watcher: wait for the R2.4 baseline sweep to finish, then auto-launch the
# epsilon sensitivity sweep (R1.1c/R1.2). They must NOT overlap -- both mutate
# config.py and share the GPU -- so this chains them safely.
#
# Signal used: the parent script run_r24_baselines.sh is gone (it exits only
# after all rho runs have been attempted and it has written its DONE marker, or
# if it was killed). Either way, once it is gone the GPU and config.py are free.
#
# Launch detached:
#   setsid bash scripts/watch_and_run_epsilon.sh </dev/null \
#       >results/logs/watch_epsilon.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
WLOG="results/logs/watch_epsilon.log"

log() { echo "[watch-eps] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WLOG"; }

log "watcher started; waiting for the R2.4 baseline sweep to finish..."

# 1) Wait until the R2.4 sweep script is no longer running. Poll every 2 min.
while pgrep -f "run_r24_baselines.sh" >/dev/null 2>&1; do
  sleep 120
done
log "run_r24_baselines.sh no longer running."

# 2) Belt-and-suspenders: also wait out any straggler runner/main.py process
#    so the GPU and config.py are truly free.
while pgrep -f "run_baselines.py" >/dev/null 2>&1; do
  sleep 60
done
log "no run_baselines.py process alive; GPU is free."

COMPLETED=$(($(wc -l < results/experiment_ledger.csv) - 1))
log "ledger now has ${COMPLETED} completed runs (expected 732 if the R2.4 sweep finished cleanly)."

# 3) Launch the epsilon sensitivity sweep (canonical setting, rho=0.3 sev=3).
#    eps=1e-6 points reuse the already-completed canonical FedAgain runs.
#    4 datasets x 4 new eps x 3 seeds = 48 new runs (~2.5-3 h).
STAMP="$(date +%Y%m%d_%H%M%S)"
ELOG="results/logs/epsilon_sweep_${STAMP}.log"
log "launching epsilon sweep -> ${ELOG}"
python scripts/run_epsilon_sensitivity.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
  --epsilons 1e-8 1e-6 1e-4 1e-2 1.0 \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --num-clients 10 --num-rounds 10 --seeds 42 43 44 >> "$ELOG" 2>&1
RC=$?
log "epsilon sweep finished with exit code ${RC}."

# 4) Restore config.py to the committed baseline (the runner rewrites it).
git checkout config.py >> "$WLOG" 2>&1 && log "config.py restored"
COMPLETED=$(($(wc -l < results/experiment_ledger.csv) - 1))
log "DONE. ledger now has ${COMPLETED} completed runs (expected 780 = 732 + 48)."
