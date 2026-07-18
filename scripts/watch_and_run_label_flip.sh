#!/usr/bin/env bash
# Watcher: wait for the rho=0.5/0.6 envelope sweep to finish, then auto-launch
# the label-flipping attack sweep (R1.3/R2.3). They must NOT overlap -- both
# mutate config.py and share the GPU -- so this chains them safely.
#
# Signal used: the parent script run_envelope_high.sh is gone (it exits only
# after all rho runs have been attempted and it has written its DONE marker, or
# if it was killed). Either way, once it is gone the GPU and config.py are free.
#
# Launch detached:
#   setsid bash scripts/watch_and_run_label_flip.sh </dev/null \
#       >results/logs/watch_label_flip.out 2>&1 &
set -u

cd "$(dirname "$0")/.."
mkdir -p results/logs
WLOG="results/logs/watch_label_flip.log"

log() { echo "[watch] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WLOG"; }

log "watcher started; waiting for the rho envelope sweep to finish..."

# 1) Wait until the envelope-high sweep script is no longer running.
#    Poll every 2 minutes.
while pgrep -f "run_envelope_high.sh" >/dev/null 2>&1; do
  sleep 120
done
log "run_envelope_high.sh no longer running."

# 2) Belt-and-suspenders: also wait out any straggler run_baselines.py process
#    (the last main.py subprocess of the sweep) so the GPU is truly free.
while pgrep -f "run_baselines.py" >/dev/null 2>&1; do
  sleep 60
done
log "no run_baselines.py / main.py sweep process alive; GPU is free."

# 3) Report the rho-sweep tally for the record.
COMPLETED=$(($(wc -l < results/experiment_ledger.csv) - 1))
log "ledger now has ${COMPLETED} completed runs (expected 480 if rho sweep finished cleanly)."

# 4) Restore config.py to the committed baseline before starting the new sweep.
if git checkout config.py 2>>"$WLOG"; then
  log "config.py restored from git."
else
  log "WARNING: git checkout config.py failed; proceeding with current config.py."
fi

# 5) Launch the label-flip sweep (detached from this watcher).
log "launching label-flip sweep..."
setsid bash scripts/run_label_flip.sh </dev/null \
    >results/logs/label_flip.out 2>&1 &
log "label-flip sweep launched (see results/logs/label_flip_*.log). Watcher done."
