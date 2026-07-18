"""
Baseline comparison runner: sweep FedAgain against recent FL baselines under a
fixed corrupted non-IID setting (Reviewer 1.5 / 2.4).

For each (strategy, dataset, seed) it rewrites config.py and launches main.py.
Uses the resilient tracker, so completed runs are skipped on resume.

Run from the repository root, e.g.::

    python scripts/run_baselines.py \
        --strategies FedAvg FedProx FedMedian Bulyan FedBN Scaffold MOON FedAgain \
        --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
        --corrupt-ratio 0.3 --severity 3 --num-clients 10 --num-rounds 10 \
        --seeds 42 43 44
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
MAIN_PATH = ROOT / "main.py"

ALL_STRATEGIES = ["FedAvg", "FedProx", "FedMedian", "Bulyan",
                  "FedBN", "Scaffold", "MOON", "FedAgain",
                  # R2.4 recent robust baselines (FedDyn = R1.5). RFLPA/BOBA need a
                  # server root set; BOBA needs num_clients-f > num_classes-1
                  # (degenerate on CIFAR-10/MNIST with 10 clients, 10 classes).
                  "FedDyn", "RFLPA", "BOBA"]


def update_config(values: dict) -> None:
    lines = CONFIG_PATH.read_text().splitlines(keepends=True)
    remaining = dict(values)
    out = []
    for line in lines:
        replaced = False
        for key, val in values.items():
            if line.startswith(f"{key} = ") or line.startswith(f"{key}="):
                literal = f'"{val}"' if isinstance(val, str) else val
                out.append(f"{key} = {literal}\n")
                remaining.pop(key, None)
                replaced = True
                break
        if not replaced:
            out.append(line)
    if remaining:
        raise KeyError(f"config.py keys not found: {list(remaining)}")
    CONFIG_PATH.write_text("".join(out))


def run_once(strategy, dataset, distribution, corrupt_ratio, severity,
             num_clients, num_rounds, seed):
    update_config({
        "STRATEGY": strategy,
        "DATASET": dataset,
        "DATA_DISTRIBUTION": distribution,
        "NUM_CLIENTS": num_clients,
        "NUM_ROUNDS": num_rounds,
        "CORRUPT_CLIENTS": corrupt_ratio > 0,
        "CORRUPT_CLIENT_RATIO": corrupt_ratio,
        "CORRUPTION_SEVERITY": severity,
        "SEED": seed,
    })
    print("\n" + "=" * 80)
    print(f"[baselines] {strategy} | {dataset} | {distribution} | "
          f"rho={corrupt_ratio} sev={severity} seed={seed}")
    print("=" * 80)
    return subprocess.run([sys.executable, str(MAIN_PATH)],
                          cwd=str(ROOT)).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES)
    p.add_argument("--datasets", nargs="+",
                   default=["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE", "CIFAR10"])
    p.add_argument("--distribution", default="non-iid-label")
    p.add_argument("--corrupt-ratio", type=float, default=0.3)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()

    summary = {}
    for dataset in args.datasets:
        for strat in args.strategies:
            for seed in args.seeds:
                ok = run_once(strat, dataset, args.distribution,
                              args.corrupt_ratio, args.severity,
                              args.num_clients, args.num_rounds, seed)
                summary[(dataset, strat, seed)] = "OK" if ok else "FAILED"

    print("\n" + "=" * 80)
    print("BASELINE SWEEP SUMMARY")
    print("=" * 80)
    for (dataset, strat, seed), status in summary.items():
        print(f"  {dataset:14} {strat:10} seed={seed} {status}")
    print("\nResults -> results/experiment_ledger.csv, results/runs/, results/rounds/")


if __name__ == "__main__":
    main()
