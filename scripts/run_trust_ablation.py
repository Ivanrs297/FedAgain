"""
Trust-score ablation and sensitivity runner (paper experiments #2 and #3).

For each requested trust ``mode`` this rewrites ``config.py`` and launches
``main.py`` with ``STRATEGY = FedAgain`` and per-round trust logging enabled.
Results land in ``results/experiments.csv`` (final metrics) and
``results/trust_scores/`` (per-round E_k, D_k, T_k distributions).

Covers:
    * #2 ablation  : benchmark_only, divergence_only, combined_mult
    * #3 sensitivity: combined_mult, additive, weighted_additive

Run from the repository root, e.g.::

    python scripts/run_trust_ablation.py --datasets DAUDON_MIX --distribution non-iid-label \
        --corrupt-ratio 0.3 --severity 3 --modes benchmark_only divergence_only combined_mult
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
MAIN_PATH = ROOT / "main.py"

ABLATION_MODES = ["benchmark_only", "divergence_only", "combined_mult"]  # #2
SENSITIVITY_MODES = ["combined_mult", "additive", "weighted_additive"]   # #3


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


def run_once(mode, dataset, distribution, corrupt_ratio, severity,
             num_clients, num_rounds, alpha, beta):
    update_config({
        "STRATEGY": "FedAgain",
        "DATASET": dataset,
        "DATA_DISTRIBUTION": distribution,
        "NUM_CLIENTS": num_clients,
        "NUM_ROUNDS": num_rounds,
        "CORRUPT_CLIENTS": corrupt_ratio > 0,
        "CORRUPT_CLIENT_RATIO": corrupt_ratio,
        "CORRUPTION_SEVERITY": severity,
        "FEDAGAIN_TRUST_MODE": mode,
        "FEDAGAIN_ALPHA": alpha,
        "FEDAGAIN_BETA": beta,
        "FEDAGAIN_LOG_TRUST": True,
    })
    print("\n" + "=" * 80)
    print(f"[trust-ablation] mode={mode} dataset={dataset} "
          f"dist={distribution} rho={corrupt_ratio} sev={severity}")
    print("=" * 80)
    result = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=str(ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DAUDON_MIX"])
    parser.add_argument("--distribution", default="non-iid-label")
    parser.add_argument("--corrupt-ratio", type=float, default=0.3)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="weight on E_k for weighted_additive")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="weight on D_k for weighted_additive")
    parser.add_argument(
        "--modes", nargs="+",
        default=sorted(set(ABLATION_MODES + SENSITIVITY_MODES)),
        help="trust modes to sweep (default: all ablation + sensitivity modes)",
    )
    args = parser.parse_args()

    summary = {}
    for dataset in args.datasets:
        for mode in args.modes:
            ok = run_once(
                mode, dataset, args.distribution, args.corrupt_ratio,
                args.severity, args.num_clients, args.num_rounds,
                args.alpha, args.beta,
            )
            summary[(dataset, mode)] = "OK" if ok else "FAILED"

    print("\n" + "=" * 80)
    print("TRUST ABLATION SUMMARY")
    print("=" * 80)
    for (dataset, mode), status in summary.items():
        print(f"  {dataset:15} {mode:20} {status}")
    print("\nFinal metrics -> results/experiments.csv")
    print("Per-round trust scores -> results/trust_scores/")


if __name__ == "__main__":
    main()
