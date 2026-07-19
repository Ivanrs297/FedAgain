"""
Epsilon sensitivity runner (R1.1c / R1.2): sweep FEDAGAIN_EPSILON.

The trust score T_k = 1/(E_k * D_k + eps) uses a stability constant eps.
Reviewer 1 asked for a sensitivity discussion; the formal argument (the
relative perturbation of any raw trust score is bounded by eps/(E_k D_k))
is in the paper -- this runner provides the matching empirical evidence:
final accuracy should be flat across many orders of magnitude of eps.

For each requested eps this rewrites ``config.py`` and launches ``main.py``
with STRATEGY = FedAgain (combined_mult). The default eps=1e-6 point reuses
the already-completed canonical runs (the tracker skips them: a non-default
eps is tagged ``eps<value>`` in the run key, the default is untagged).

Run from the repository root, e.g.::

    python scripts/run_epsilon_sensitivity.py \
        --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
        --epsilons 1e-8 1e-6 1e-4 1e-2 1.0 --seeds 42 43 44
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
MAIN_PATH = ROOT / "main.py"

DEFAULT_EPSILONS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]


def update_config(values: dict) -> None:
    """Update top-level ``KEY = ...`` assignments in config.py in place."""
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


def run_once(eps, dataset, distribution, corrupt_ratio, severity,
             num_clients, num_rounds, seed):
    update_config({
        "STRATEGY": "FedAgain",
        "DATASET": dataset,
        "DATA_DISTRIBUTION": distribution,
        "NUM_CLIENTS": num_clients,
        "NUM_ROUNDS": num_rounds,
        "CORRUPT_CLIENTS": corrupt_ratio > 0,
        "CORRUPT_CLIENT_RATIO": corrupt_ratio,
        "CORRUPTION_SEVERITY": severity,
        "SEED": seed,
        "FEDAGAIN_TRUST_MODE": "combined_mult",
        "FEDAGAIN_EPSILON": eps,
        "FEDAGAIN_LOG_TRUST": False,
    })
    print("\n" + "=" * 80)
    print(f"[epsilon-sweep] eps={eps} dataset={dataset} "
          f"dist={distribution} rho={corrupt_ratio} sev={severity} seed={seed}")
    print("=" * 80)
    result = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=str(ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE", "CIFAR10"])
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=DEFAULT_EPSILONS)
    parser.add_argument("--distribution", default="non-iid-label")
    parser.add_argument("--corrupt-ratio", type=float, default=0.3)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    summary = {}
    for dataset in args.datasets:
        for eps in args.epsilons:
            for seed in args.seeds:
                ok = run_once(eps, dataset, args.distribution,
                              args.corrupt_ratio, args.severity,
                              args.num_clients, args.num_rounds, seed)
                summary[(dataset, eps, seed)] = "OK" if ok else "FAILED"

    print("\n" + "=" * 80)
    print("EPSILON SENSITIVITY SUMMARY")
    print("=" * 80)
    for (dataset, eps, seed), status in summary.items():
        print(f"  {dataset:15} eps={eps:<8g} seed={seed} {status}")
    print("\nFinal metrics -> results/experiment_ledger.csv (keys tagged eps<value>)")


if __name__ == "__main__":
    main()
