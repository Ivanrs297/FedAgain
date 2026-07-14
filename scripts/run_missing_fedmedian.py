"""
Run missing FedMedian experiments for IID and Non-IID (Dirichlet) distributions.

Currently FedMedian only has corruption data for non-iid-label.
This script fills in the missing experiments for fair comparison across distributions.

Missing experiments:
  - FedMedian + iid + corrupt_ratio=0.3 + severity=3 (all 7 kidney stone datasets)
  - FedMedian + non-iid-dirichlet + corrupt_ratio=0.3 + severity=3 (all 7 kidney stone datasets)
"""

import shutil
import subprocess
import sys
import time
from datetime import datetime


KIDNEY_DATASETS = [
    "DAUDON_MIX", "DAUDON_SEC", "DAUDON_SUR",
    "JONATHAN_MIX", "JONATHAN_SEC", "JONATHAN_SUR",
    "MYSTONE"
]

MISSING_EXPERIMENTS = [
    # (distribution, corrupt_ratio, severity)
    ("iid", 0.3, 3),
    ("non-iid-dirichlet", 0.3, 3),
]


def update_config(updates: dict):
    """Update multiple key-value pairs in config.py."""
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
    """Format seconds into a human-readable string."""
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
    total = len(KIDNEY_DATASETS) * len(MISSING_EXPERIMENTS)
    exp_num = 0
    successful = 0
    failed = 0
    exp_times = []

    # Save original config to restore later
    with open('config.py', 'r') as f:
        original_config = f.read()

    # Backup results CSV before starting
    results_csv = 'results/experiments.csv'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'results/experiments_backup_{timestamp}.csv'
    try:
        shutil.copy2(results_csv, backup_path)
        print(f"  Results backup: {backup_path}")
    except FileNotFoundError:
        print(f"  No existing results to backup")

    print("=" * 80)
    print("  MISSING FedMedian EXPERIMENTS")
    print("=" * 80)
    print(f"  Strategy:      FedMedian")
    print(f"  Datasets:      {', '.join(KIDNEY_DATASETS)}")
    print(f"  Distributions: {[e[0] for e in MISSING_EXPERIMENTS]}")
    print(f"  Corruption:    ratio=0.3, severity=3")
    print(f"  Config:        20 clients, 10 rounds, 1 local epoch, ResNet18")
    print(f"  Total runs:    {total}")
    print(f"  Results file:  {results_csv}")
    print("=" * 80)
    print()

    overall_start = datetime.now()

    try:
        for dist, ratio, severity in MISSING_EXPERIMENTS:
            for dataset in KIDNEY_DATASETS:
                exp_num += 1
                pct = (exp_num - 1) / total * 100

                # ETA calculation
                if exp_times:
                    avg_time = sum(exp_times) / len(exp_times)
                    remaining = (total - exp_num + 1) * avg_time
                    eta_str = f"ETA: {format_duration(remaining)}"
                else:
                    eta_str = "ETA: estimating..."

                print()
                print("=" * 80)
                print(f"  [{exp_num}/{total}] {pct:.0f}% complete | {eta_str}")
                print(f"  Dataset: {dataset} | Dist: {dist} | Ratio: {ratio} | Severity: {severity}")
                print("=" * 80)

                # Update config — ensure NUM_CLIENTS=20, NUM_ROUNDS=10
                print(f"  [1/4] Updating config.py...")
                update_config({
                    "DATASET": dataset,
                    "STRATEGY": "FedMedian",
                    "DATA_DISTRIBUTION": dist,
                    "NUM_CLIENTS": 20,
                    "NUM_ROUNDS": 10,
                    "LOCAL_EPOCHS": 1,
                    "CORRUPT_CLIENTS": True,
                    "CORRUPT_CLIENT_RATIO": ratio,
                    "CORRUPTION_SEVERITY": severity,
                })
                print(f"         Strategy=FedMedian, Dataset={dataset}, Dist={dist}")
                print(f"         Clients=20, Rounds=10, Epochs=1")

                # Run experiment with live output
                print(f"  [2/4] Loading dataset and model...")
                t0 = time.time()

                proc = subprocess.Popen(
                    [sys.executable, "-u", "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                current_round = 0
                for line in proc.stdout:
                    line = line.rstrip()
                    if not line:
                        continue

                    # Track round progress
                    if "--- Round " in line:
                        current_round += 1
                        elapsed = time.time() - t0
                        round_pct = current_round / 10 * 100
                        print(f"  [3/4] Training round {current_round}/10 ({round_pct:.0f}%) | elapsed: {format_duration(elapsed)}")
                    elif "Global Model" in line and "Accuracy" in line:
                        print(f"         {line.strip()}")
                    elif "Computing final metrics" in line:
                        print(f"  [4/4] Computing final metrics...")
                    elif "Accuracy:" in line and "Precision" not in line and "Global" not in line:
                        print(f"         {line.strip()}")
                    elif "Precision:" in line:
                        print(f"         {line.strip()}")
                    elif "F1 Score:" in line:
                        print(f"         {line.strip()}")
                    elif "Results saved" in line:
                        print(f"         {line.strip()}")

                proc.wait()
                elapsed = time.time() - t0

                if proc.returncode == 0:
                    exp_times.append(elapsed)
                    successful += 1
                    print(f"  DONE in {format_duration(elapsed)}")
                else:
                    failed += 1
                    print(f"  FAILED after {format_duration(elapsed)}")

                # Summary so far
                total_elapsed = (datetime.now() - overall_start).total_seconds()
                print(f"  Progress: {exp_num}/{total} | Passed: {successful} | Failed: {failed} | Total time: {format_duration(total_elapsed)}")

                time.sleep(1)

    finally:
        # Always restore original config, even if interrupted
        with open('config.py', 'w') as f:
            f.write(original_config)
        print("\n  Config restored to original values.")

    # Final summary
    total_elapsed = (datetime.now() - overall_start).total_seconds()
    print()
    print("=" * 80)
    print(f"  ALL DONE")
    print(f"  {successful}/{total} succeeded, {failed} failed")
    print(f"  Total time: {format_duration(total_elapsed)}")
    print(f"  Results saved to: {results_csv}")
    print(f"  Backup at: {backup_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
