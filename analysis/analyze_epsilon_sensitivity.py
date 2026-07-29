#!/usr/bin/env python3
"""
Epsilon sensitivity analysis (Reviewer 1.1c / 1.2).

Reads the experiment ledger and summarizes FedAgain's final accuracy as a
function of the trust stability constant eps in T_k = 1/(E_k*D_k + eps),
under the canonical setting (non-IID label skew, rho=0.3, severity 3,
10 clients, 10 rounds, seeds 42/43/44). Non-default eps runs carry an
``eps<value>`` tag in the run key; the default 1e-6 is untagged.

Outputs:
  results/epsilon_sensitivity/epsilon_summary.csv
  results/epsilon_sensitivity/epsilon_sensitivity.png (+ copy to src-paper/)
"""
import csv
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "experiment_ledger.csv")
OUTDIR = os.path.join(ROOT, "results", "epsilon_sensitivity")
SRCPAPER = os.path.join(ROOT, "jcr-fedagain", "src-paper", "images", "results")
os.makedirs(OUTDIR, exist_ok=True)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _datasets import datasets as _ds_sel, fig_name, srcpaper_dir, cifar_only  # R1.7 + CIFAR grid
DATASETS = _ds_sel()
DS_LABEL = {"DAUDON_MIX": "A (Daudon)", "JONATHAN_MIX": "B (Jonathan)",
            "MYSTONE": "C (MyStone)", "CIFAR10": "CIFAR-10"}
DS_COLOR = {"DAUDON_MIX": "#d62728", "JONATHAN_MIX": "#1f77b4",
            "MYSTONE": "#2ca02c", "CIFAR10": "#7f7f7f"}
EPS_RE = re.compile(r"_eps([0-9.e+-]+)_s\d+$")


def load():
    """Return {(dataset, eps): [final_acc_pct, ... over seeds]}."""
    acc = defaultdict(list)
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            if (r["status"] != "completed" or r["strategy"] != "FedAgain"
                    or r["dataset"] not in DATASETS
                    or r["data_distribution"] != "non-iid-label"
                    or r["num_clients"] != "10" or r["num_rounds"] != "10"
                    or r["corrupt_ratio"] != "0.3"
                    or r["corruption_severity"] != "3"
                    or r["trust_mode"] != "combined_mult"
                    or "pm-" in r["run_key"]
                    or "nm-" in r["run_key"]):
                continue
            m = EPS_RE.search(r["run_key"])
            eps = float(m.group(1)) if m else 1e-6
            acc[(r["dataset"], eps)].append(float(r["final_accuracy"]) * 100)
    return acc


def main():
    acc = load()
    epsilons = sorted({e for (_, e) in acc})
    print("Epsilons found:", epsilons)

    rows = []
    for ds in DATASETS:
        for eps in epsilons:
            vals = acc.get((ds, eps), [])
            if vals:
                rows.append((ds, eps, st.mean(vals),
                             st.stdev(vals) if len(vals) > 1 else 0.0, len(vals)))

    with open(os.path.join(OUTDIR, "epsilon_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "epsilon", "acc_mean", "acc_std", "n_seeds"])
        for ds, eps, m, s, n in rows:
            w.writerow([ds, f"{eps:g}", f"{m:.2f}", f"{s:.2f}", n])

    print(f"\n{'dataset':<14} {'eps':>8} {'acc mean':>9} {'std':>6} {'n':>3}")
    for ds, eps, m, s, n in rows:
        print(f"{ds:<14} {eps:>8g} {m:>9.2f} {s:>6.2f} {n:>3}")

    # Max spread across eps per dataset (the headline number for the paper).
    print("\nMax accuracy spread across epsilons (per dataset):")
    for ds in DATASETS:
        means = [m for (d, e, m, s, n) in rows if d == ds]
        if means:
            print(f"  {ds:<14} {max(means) - min(means):.2f} pts")

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(5.6, 4.4) if cifar_only() else (7.5, 4.6))
    for ds in DATASETS:
        pts = sorted((e, m, s) for (d, e, m, s, n) in rows if d == ds)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                    color=DS_COLOR[ds], label=DS_LABEL[ds])
    ax.set_xscale("log")
    ax.set_xlabel(r"Stability constant $\varepsilon$")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=11,
              loc="lower right" if cifar_only() else "best")
    fig.tight_layout()
    out = os.path.join(OUTDIR, fig_name("epsilon_sensitivity") + ".png")
    fig.savefig(out, dpi=200, bbox_inches=None if cifar_only() else "tight")
    _dst = srcpaper_dir()
    if os.path.isdir(_dst):
        import shutil
        shutil.copy(out, os.path.join(_dst, fig_name("epsilon_sensitivity") + ".png"))
        print(f"\nFigure copied to {_dst}")
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
