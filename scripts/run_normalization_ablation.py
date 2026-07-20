"""
Trust-normalization ablation runner (Reviewer 2.5).

R2.5 asked us to *test* alternative strategies for mapping the raw trust
scores ``T_k`` to aggregation weights, not merely to argue about them. This
runner sweeps ``FEDAGAIN_NORM_MODE`` while holding the trust formulation fixed
at the deployed multiplicative rule:

    sum_to_one : w = T / sum(T)                (default; already in the ledger)
    softmax    : w = softmax(log T / tau)      (tau != 1; tau = 1 is identical
                 to sum_to_one, which is precisely the point the paper makes)
    rank       : w proportional to the rank of T (ordering kept, magnitude lost)

Only non-default normalizations are tagged in the run key (``nm-<mode>`` plus
``tau<value>`` for softmax), so every pre-existing key stays valid and the
``sum_to_one`` reference runs are reused rather than recomputed.

Run from the repository root, e.g.::

    python scripts/run_normalization_ablation.py \
        --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
        --variants softmax:0.5 softmax:2.0 rank --seeds 42 43 44
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
MAIN_PATH = ROOT / "main.py"

DEFAULT_VARIANTS = ["softmax:0.5", "softmax:2.0", "rank"]


def parse_variant(spec):
    """``"softmax:0.5"`` -> ``("softmax", 0.5)``; ``"rank"`` -> ``("rank", 1.0)``."""
    if ":" in spec:
        mode, tau = spec.split(":", 1)
        return mode, float(tau)
    return spec, 1.0


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


def run_once(norm_mode, tau, dataset, distribution, corrupt_ratio, severity,
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
        "FEDAGAIN_NORM_MODE": norm_mode,
        "FEDAGAIN_NORM_TEMPERATURE": tau,
        "FEDAGAIN_LOG_TRUST": True,
    })
    print("\n" + "=" * 80)
    print(f"[norm-ablation] norm={norm_mode} tau={tau} dataset={dataset} "
          f"dist={distribution} rho={corrupt_ratio} sev={severity} seed={seed}")
    print("=" * 80)
    result = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=str(ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE", "CIFAR10"])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="normalization variants as MODE or MODE:TAU")
    parser.add_argument("--distribution", default="non-iid-label")
    parser.add_argument("--corrupt-ratio", type=float, default=0.3)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    summary = {}
    for dataset in args.datasets:
        for spec in args.variants:
            norm_mode, tau = parse_variant(spec)
            for seed in args.seeds:
                ok = run_once(norm_mode, tau, dataset, args.distribution,
                              args.corrupt_ratio, args.severity,
                              args.num_clients, args.num_rounds, seed)
                summary[(dataset, spec, seed)] = "OK" if ok else "FAILED"

    print("\n" + "=" * 80)
    print("NORMALIZATION ABLATION SUMMARY")
    print("=" * 80)
    for (dataset, spec, seed), status in summary.items():
        print(f"  {dataset:15} {spec:12} seed={seed} {status}")
    print("\nFinal metrics -> results/experiment_ledger.csv (keys tagged nm-<mode>)")


if __name__ == "__main__":
    main()
