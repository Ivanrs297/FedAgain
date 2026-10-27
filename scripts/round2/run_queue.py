"""
Unattended runner for the second-revision (data v2) experiment queue.

Runs every job of results_v2/queue.json through main.py, one at a time, with
the config passed as FEDAGAIN_CONFIG_OVERRIDES (config.py is never rewritten),
then runs the v2 analyses once everything is done.

Resilience:
  * a job counts as done only when results_v2/runs/<key>.json says
    "completed" (written by the tracker), so a crash or reboot mid-run simply
    reruns that job;
  * a failed job is retried up to MAX_ATTEMPTS times, after the rest of the
    queue, so one bad job never blocks the others;
  * per-job timeout; per-job log in results_v2/logs/;
  * `touch results_v2/PAUSE` stops cleanly after the current job (and keeps
    the cron watchdog from relaunching); delete the file to resume.
Progress: results_v2/QUEUE_STATUS.md (human-readable), queue_state.json.

Normally started by scripts/round2/launch.sh (cron watchdog), never by hand
while another instance runs (launch.sh holds a flock).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "results_v2")
QUEUE = os.path.join(V2, "queue.json")
STATE = os.path.join(V2, "queue_state.json")
STATUS_MD = os.path.join(V2, "QUEUE_STATUS.md")
PAUSE = os.path.join(V2, "PAUSE")
COMPLETE = os.path.join(V2, "QUEUE_COMPLETE")
LOGS = os.path.join(V2, "logs")
MAX_ATTEMPTS = 3
JOB_TIMEOUT_S = 3 * 3600
PHASE_NAMES = {
    "P1": "canonical table (rho=0.3)", "P2": "multi-site canonical (rho=0.3)",
    "P3": "degradation envelope", "P9": "trust/data-size mixing (lambda)",
    "P4": "label-flip attack",
    "P5": "multi-site envelope (rho 0 / 0.6)", "P6": "trust ablation",
    "P7": "normalization", "P8": "epsilon sweep",
}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def is_completed(key):
    m = load_json(os.path.join(V2, "runs", f"{key}.json"), {})
    return m.get("status") == "completed"


def write_status(queue, state, current=None, note=""):
    done = [j for j in queue if is_completed(j["key"])]
    failed = [j for j in queue if not is_completed(j["key"])
              and state["attempts"].get(j["key"], 0) >= MAX_ATTEMPTS]
    durs = state.get("durations", [])
    remaining = len(queue) - len(done) - len(failed)
    eta = ""
    if durs:
        recent = durs[-30:]
        eta = (datetime.now() + timedelta(seconds=remaining * sum(recent) / len(recent))
               ).strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Round-2 (data v2) experiment queue",
        "",
        f"- Updated: {now()}  {note}",
        f"- Done: **{len(done)} / {len(queue)}**   failed (gave up): {len(failed)}   remaining: {remaining}",
        f"- Running: `{current}`" if current else "- Running: (none)",
        f"- ETA (from the last {min(len(durs), 30)} job durations): {eta or 'n/a'}",
        "",
        "| phase | what | done | total |",
        "|---|---|---|---|",
    ]
    for p, name in PHASE_NAMES.items():
        tot = [j for j in queue if j["phase"] == p]
        lines.append(f"| {p} | {name} | {sum(is_completed(j['key']) for j in tot)} | {len(tot)} |")
    if failed:
        lines += ["", "## Failed after retries (see results_v2/logs/<key>.log)", ""]
        lines += [f"- `{j['key']}`" for j in failed]
    lines += ["", "Pause: `touch results_v2/PAUSE`.  Resume: `rm results_v2/PAUSE`",
              "(the cron watchdog relaunches within 15 min)."]
    with open(STATUS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_job(job):
    env = dict(os.environ)
    env["FEDAGAIN_DATA_VERSION"] = "v2"
    env["FEDAGAIN_CONFIG_OVERRIDES"] = json.dumps(job["overrides"])
    env.pop("FEDAGAIN_RESULTS_DIR", None)
    env.pop("EXPERIMENT_FORCE", None)
    log = os.path.join(LOGS, f"{job['key']}.log")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n===== {now()} start {job['key']}\n")
        f.flush()
        try:
            rc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")], cwd=ROOT,
                                env=env, stdout=f, stderr=subprocess.STDOUT,
                                timeout=JOB_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            f.write(f"\n===== TIMEOUT after {JOB_TIMEOUT_S}s\n")
    return rc, time.time() - t0


def run_analysis():
    """Run the v2 analyses once; outputs stay inside results_v2/."""
    script = os.path.join(ROOT, "scripts", "round2", "run_analysis.sh")
    with open(os.path.join(LOGS, "analysis.log"), "a") as f:
        f.write(f"\n===== {now()} analysis\n")
        f.flush()
        return subprocess.run(["bash", script], cwd=ROOT, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def main():
    os.makedirs(LOGS, exist_ok=True)
    queue = load_json(QUEUE, None)
    if queue is None:
        sys.exit("results_v2/queue.json missing: run scripts/round2/build_queue.py")
    state = load_json(STATE, {"attempts": {}, "durations": []})
    print(f"[{now()}] queue runner started, {len(queue)} jobs", flush=True)

    while True:
        if os.path.exists(PAUSE):
            write_status(queue, state, note="(PAUSED)")
            print(f"[{now()}] PAUSE file present, stopping", flush=True)
            return
        # Re-read the queue every iteration so jobs appended by build_queue.py
        # are picked up without restarting the runner.
        queue = load_json(QUEUE, queue)
        # Next job: first not-completed job with the fewest attempts, so failed
        # jobs are retried only after every fresh job has had its turn.
        todo = [j for j in queue if not is_completed(j["key"])
                and state["attempts"].get(j["key"], 0) < MAX_ATTEMPTS]
        if not todo:
            break
        least = min(state["attempts"].get(j["key"], 0) for j in todo)
        job = next(j for j in todo if state["attempts"].get(j["key"], 0) == least)

        state["attempts"][job["key"]] = state["attempts"].get(job["key"], 0) + 1
        save_json(STATE, state)
        write_status(queue, state, current=job["key"])
        print(f"[{now()}] {job['phase']} {job['key']} (attempt "
              f"{state['attempts'][job['key']]})", flush=True)
        rc, secs = run_job(job)
        ok = rc == 0 and is_completed(job["key"])
        if ok:
            state["durations"].append(round(secs))
        print(f"[{now()}]   -> {'ok' if ok else f'FAILED rc={rc}'} in {secs/60:.1f} min",
              flush=True)
        save_json(STATE, state)

    write_status(queue, state, note="(all jobs processed; running analysis)")
    rc = 0
    if not os.path.exists(os.path.join(V2, "ANALYSIS_DONE")):
        rc = run_analysis()
        if rc == 0:
            open(os.path.join(V2, "ANALYSIS_DONE"), "w").write(now() + "\n")
    write_status(queue, state, note="(FINISHED)" if rc == 0 else
                 f"(FINISHED, but analysis exited {rc}: see results_v2/logs/analysis.log)")
    open(COMPLETE, "w").write(now() + "\n")
    print(f"[{now()}] queue complete", flush=True)


if __name__ == "__main__":
    main()
