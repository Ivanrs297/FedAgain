"""Run the 2 missing MYSTONE FedMedian experiments (iid + non-iid-dirichlet)."""

import shutil
import subprocess
import sys
import time
from datetime import datetime


def update_config(updates: dict):
    config_path = 'config.py'
    with open(config_path, 'r') as f:
        lines = f.readlines()
    with open(config_path, 'w') as f:
        for line in lines:
            updated = False
            for key, value in updates.items():
                if line.strip().startswith(f"{key} ="):
                    if isinstance(value, str) and not value.startswith(('"', "'")):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f"{key} = {value}\n")
                    updated = True
                    break
            if not updated:
                f.write(line)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def main():
    experiments = [
        ("iid", 0.3, 3),
        ("non-iid-dirichlet", 0.3, 3),
    ]

    with open('config.py', 'r') as f:
        original_config = f.read()

    print("=" * 80)
    print("  MISSING MYSTONE FedMedian EXPERIMENTS (2 remaining)")
    print("=" * 80)

    try:
        for i, (dist, ratio, severity) in enumerate(experiments, 1):
            print(f"\n  [{i}/2] MYSTONE | {dist} | ratio={ratio} | severity={severity}")
            print("=" * 80)

            update_config({
                "DATASET": "MYSTONE",
                "STRATEGY": "FedMedian",
                "DATA_DISTRIBUTION": dist,
                "NUM_CLIENTS": 20,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": ratio,
                "CORRUPTION_SEVERITY": severity,
            })

            t0 = time.time()
            proc = subprocess.Popen(
                [sys.executable, "-u", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if "--- Round " in line:
                    elapsed = time.time() - t0
                    print(f"  {line.strip()} | elapsed: {format_duration(elapsed)}")
                elif "Global Model" in line:
                    print(f"    {line.strip()}")
                elif "Accuracy:" in line and "Global" not in line:
                    print(f"    {line.strip()}")
                elif "Precision:" in line:
                    print(f"    {line.strip()}")
                elif "F1 Score:" in line:
                    print(f"    {line.strip()}")
                elif "Results saved" in line:
                    print(f"    {line.strip()}")

            proc.wait()
            elapsed = time.time() - t0

            if proc.returncode == 0:
                print(f"  DONE in {format_duration(elapsed)}")
            else:
                print(f"  FAILED after {format_duration(elapsed)}")

            time.sleep(1)

    finally:
        with open('config.py', 'w') as f:
            f.write(original_config)
        print("\n  Config restored.")

    print("\n" + "=" * 80)
    print("  ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
