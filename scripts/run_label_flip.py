"""
Label-flipping adversarial-attack runner (Reviewer 1.3 / 2.3).

Unlike the image-corruption sweeps, this evaluates robustness to a genuine
*adversarial* data-poisoning attack: corrupted clients have a fraction of their
labels flipped to a wrong class (POISON_MODE="label_flip"). It sweeps FedAgain
against the key robust/vanilla baselines under non-IID label skew at a couple of
corruption fractions, to re-strengthen the (previously softened) robustness
claims with a real attack.

Self-contained and ISOLATED from run_baselines.py (which may be running a
concurrent image-corruption sweep): it has its own update_config and, crucially,
sets POISON_MODE="label_flip" so the tracker's run_key gets a distinct
`pm-label_flip` tag and these runs never collide with the image-corruption runs.

Run from the repository root, e.g.::

    python scripts/run_label_flip.py \
        --strategies FedAgain FedAvg FedMedian \
        --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
        --corrupt-ratios 0.1 0.3 --flip-ratio 1.0 --seeds 42 43 44
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.py"
MAIN_PATH = ROOT / "main.py"

DEFAULT_STRATEGIES = ["FedAgain", "FedAvg", "FedMedian"]


def ensure_keys_exist() -> None:
    """Append POISON_MODE / LABEL_FLIP_RATIO to config.py if they are missing.

    Kept idempotent and additive so it is safe to run once the image-corruption
    sweep has finished (update_config only rewrites existing keys, so the new
    keys must exist before we can set them)."""
    text = CONFIG_PATH.read_text()
    additions = []
    if "POISON_MODE" not in text:
        additions.append('POISON_MODE = "image"  # "image" | "label_flip"')
    if "LABEL_FLIP_RATIO" not in text:
        additions.append("LABEL_FLIP_RATIO = 1.0  # fraction of a poisoned client's labels to flip")
    if additions:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n# --- Label-flipping poisoning (Reviewer 1.3/2.3) ---\n" + "\n".join(additions) + "\n"
        CONFIG_PATH.write_text(text)
        print(f"[label_flip] added config keys: {additions}")


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


def run_once(strategy, dataset, distribution, corrupt_ratio, flip_ratio,
             num_clients, num_rounds, seed):
    update_config({
        "STRATEGY": strategy,
        "DATASET": dataset,
        "DATA_DISTRIBUTION": distribution,
        "NUM_CLIENTS": num_clients,
        "NUM_ROUNDS": num_rounds,
        "CORRUPT_CLIENTS": corrupt_ratio > 0,
        "CORRUPT_CLIENT_RATIO": corrupt_ratio,
        "POISON_MODE": "label_flip",
        "LABEL_FLIP_RATIO": flip_ratio,
        "SEED": seed,
    })
    print("\n" + "=" * 80)
    print(f"[label_flip] {strategy} | {dataset} | {distribution} | "
          f"rho={corrupt_ratio} flip={flip_ratio} seed={seed}")
    print("=" * 80)
    return subprocess.run([sys.executable, str(MAIN_PATH)],
                          cwd=str(ROOT)).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    p.add_argument("--datasets", nargs="+",
                   default=["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE", "CIFAR10"])
    p.add_argument("--distribution", default="non-iid-label")
    p.add_argument("--corrupt-ratios", nargs="+", type=float, default=[0.1, 0.3])
    p.add_argument("--flip-ratio", type=float, default=1.0)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()

    ensure_keys_exist()

    summary = {}
    for rho in args.corrupt_ratios:
        for dataset in args.datasets:
            for strat in args.strategies:
                for seed in args.seeds:
                    ok = run_once(strat, dataset, args.distribution, rho,
                                  args.flip_ratio, args.num_clients,
                                  args.num_rounds, seed)
                    summary[(rho, dataset, strat, seed)] = "OK" if ok else "FAILED"

    print("\n" + "=" * 80)
    print("LABEL-FLIP SWEEP SUMMARY")
    print("=" * 80)
    for (rho, dataset, strat, seed), status in summary.items():
        print(f"  rho={rho} {dataset:14} {strat:10} seed={seed} {status}")
    print("\nResults -> results/experiment_ledger.csv (run_key tagged 'pm-label_flip')")


if __name__ == "__main__":
    main()
