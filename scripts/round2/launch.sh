#!/usr/bin/env bash
# Start (or keep alive) the round-2 queue runner, detached from any terminal.
# Idempotent: a flock guarantees a single runner, so cron can call this every
# few minutes as a watchdog (and @reboot). Does nothing once the queue is
# complete or while results_v2/PAUSE exists.
#
#   crontab:  */15 * * * * /path/to/repo/scripts/round2/launch.sh
#             @reboot      /path/to/repo/scripts/round2/launch.sh
# PYTHON selects the interpreter (cron has a minimal PATH, so set it there).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
V2="$ROOT/results_v2"
mkdir -p "$V2/logs"
[ -f "$V2/QUEUE_COMPLETE" ] && exit 0
[ -f "$V2/PAUSE" ] && exit 0
cd "$ROOT" || exit 1
[ -f data/splits/MYSTONE_v2.csv ] || "$PY" scripts/build_specimen_splits.py >> "$V2/logs/queue_runner.log" 2>&1
[ -f "$V2/queue.json" ] || "$PY" scripts/round2/build_queue.py >> "$V2/logs/queue_runner.log" 2>&1
setsid nohup flock -n "$V2/.queue.lock" "$PY" scripts/round2/run_queue.py \
  >> "$V2/logs/queue_runner.log" 2>&1 < /dev/null &
exit 0
