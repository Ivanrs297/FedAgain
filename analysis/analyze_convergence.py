#!/usr/bin/env python3
"""
Empirical convergence curves (Reviewer 1.1c / 1.2, no new compute).

Reviewer 1 asked for a convergence / sensitivity discussion. This script reuses
the per-round test-accuracy logs already written for every run
(results/rounds/<key>.csv: columns round,timestamp,accuracy,loss) and plots how
the global model converges round-by-round, so the trust-weighted aggregation of
FedAgain can be shown to converge as stably as the standard and robust baselines
under non-IID label skew with corrupted clients.

Canonical setting plotted: non-IID label skew, 30% corrupted clients, severity 3,
10 clients, 10 rounds, averaged over seeds 42/43/44. FedAgain uses the default
multiplicative trust rule (tm-combined_mult).

Outputs:
  results/baselines/convergence.png (+ copy to src-paper/)
  results/baselines/convergence_summary.csv
Prints, per strategy, the mean final accuracy and the rounds-to-90%-of-final
(a simple convergence-speed proxy).
"""
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS_DIR = os.path.join(ROOT, "results", "rounds")
OUTDIR = os.path.join(ROOT, "results", "baselines")
SRCPAPER = os.path.join(ROOT, "jcr-fedagain", "src-paper", "images", "results")
os.makedirs(OUTDIR, exist_ok=True)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _datasets import datasets as _ds_sel, is_canonical_key, fig_name, srcpaper_dir, cifar_only  # R1.7 + canonical filter + CIFAR grid
DATASETS = _ds_sel()
DS_LABEL = {"DAUDON_MIX": "A (Daudon)", "JONATHAN_MIX": "B (Jonathan)",
            "MYSTONE": "C (MyStone)", "CIFAR10": "CIFAR-10"}
# Strategies to draw (only those with per-round data are plotted). FedAgain and
# FedMedian emphasized; the rest muted -- matches the degradation figure styling.
STYLE = {
    "FedAvg":    dict(color="#b0b0b0", lw=1.3, ls="-",  z=1),
    "FedProx":   dict(color="#b7c4d4", lw=1.3, ls="-",  z=1),
    "Bulyan":    dict(color="#a9d3a9", lw=1.3, ls="-",  z=1),
    "FedBN":     dict(color="#c9b8dd", lw=1.3, ls="-",  z=1),
    "Scaffold":  dict(color="#f0c38a", lw=1.3, ls="-",  z=1),
    "MOON":      dict(color="#c9a99b", lw=1.3, ls="-",  z=1),
    "FedDyn":    dict(color="#8fbf9f", lw=1.3, ls="-",  z=1),
    "RFLPA":     dict(color="#e2a0b0", lw=1.3, ls="-",  z=1),
    "BOBA":      dict(color="#b0a0d0", lw=1.3, ls="-",  z=1),
    "FedMedian": dict(color="#1f77b4", lw=2.6, ls="--", z=4),
    "FedAgain":  dict(color="#d62728", lw=3.0, ls="-",  z=5),
}
# Canonical config filter (matches the extended-baselines / degradation setting).
CFG = "non-iid-label_k10_r10_e1_cr0.3_sev3"


def key_strategy_dataset(fname):
    """Parse the strategy and dataset out of a rounds-CSV filename, keeping only
    the canonical cr0.3 config and (for FedAgain) the combined_mult trust rule."""
    base = os.path.basename(fname)[:-4]
    if CFG not in base:
        return None
    if not is_canonical_key(base):
        return None  # trust-ablation / label-flip / epsilon / normalization
    for ds in DATASETS:
        m = re.match(rf"([A-Za-z]+)_{ds}_", base)
        if m:
            return m.group(1), ds
    return None


def load():
    """Return {(strategy, dataset): {round: [acc,...over seeds]}}."""
    agg = defaultdict(lambda: defaultdict(list))
    for fname in glob.glob(os.path.join(ROUNDS_DIR, "*.csv")):
        sd = key_strategy_dataset(fname)
        if sd is None:
            continue
        strat, ds = sd
        if strat not in STYLE:
            continue
        with open(fname) as f:
            for r in csv.DictReader(f):
                agg[(strat, ds)][int(r["round"])].append(float(r["accuracy"]) * 100)
    return agg


def main():
    agg = load()
    strategies_present = sorted({s for (s, _) in agg}, key=lambda s: list(STYLE).index(s))
    print("Strategies with per-round data:", strategies_present)

    # ---- summary: final accuracy + rounds-to-90%-of-final ----
    rows = []
    for s in strategies_present:
        finals, speeds = [], []
        for ds in DATASETS:
            rounds = agg.get((s, ds))
            if not rounds:
                continue
            xs = sorted(rounds)
            ys = [st.mean(rounds[r]) for r in xs]
            final = ys[-1]
            target = 0.9 * final
            hit = next((xs[i] for i, y in enumerate(ys) if y >= target), xs[-1])
            finals.append(final)
            speeds.append(hit)
        if finals:
            rows.append((s, st.mean(finals), st.mean(speeds)))

    with open(os.path.join(OUTDIR, "convergence_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "final_acc_mean", "rounds_to_90pct_final_mean"])
        for s, fa, sp in rows:
            w.writerow([s, f"{fa:.2f}", f"{sp:.2f}"])

    print(f"\n{'strategy':<11} {'final acc':>10} {'rounds->90% final':>18}")
    for s, fa, sp in sorted(rows, key=lambda x: -x[1]):
        mark = "  <-- FedAgain" if s == "FedAgain" else ""
        print(f"{s:<11} {fa:>10.1f} {sp:>18.1f}{mark}")

    # ---- 2x2 convergence curves ----
    plt.rcParams.update({"font.size": 13 if cifar_only() else 14})
    _n = len(DATASETS)
    _nc = 1 if _n == 1 else (3 if _n == 3 else 2)
    _nr = (_n + _nc - 1) // _nc
    fig, axes = plt.subplots(_nr, _nc,
                             figsize=(5.6, 4.4) if cifar_only() else (6.5 * _nc, 4.3 * _nr),
                             squeeze=False)
    for _k in range(_n, _nr * _nc):
        axes.ravel()[_k].set_visible(False)
    for ax, ds in zip(axes.ravel(), DATASETS):
        for s in strategies_present:
            rounds = agg.get((s, ds))
            if not rounds:
                continue
            xs = sorted(rounds)
            ys = [st.mean(rounds[r]) for r in xs]
            stl = STYLE[s]
            ax.plot(xs, ys, label=s, color=stl["color"], lw=stl["lw"],
                    ls=stl["ls"], zorder=stl["z"],
                    alpha=0.95 if s in ("FedAgain", "FedMedian") else 0.7)
        ax.set_title(DS_LABEL[ds], fontsize=15)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(ls=":", alpha=0.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] not in ("FedAgain", "FedMedian"),
                                  list(STYLE).index(labels[i])))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=4 if cifar_only() else len(labels),
               loc="upper center", bbox_to_anchor=(0.5, 0.995 if cifar_only() else 1.02),
               frameon=False, fontsize=9 if cifar_only() else 11)
    fig.tight_layout(rect=[0, 0, 1, 0.74] if cifar_only() else [0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, fig_name("convergence") + ".png")
    fig.savefig(out, dpi=200, bbox_inches=None if cifar_only() else "tight")
    _dst = srcpaper_dir()
    if os.path.isdir(_dst):
        import shutil
        shutil.copy(out, os.path.join(_dst, fig_name("convergence") + ".png"))
        print(f"\nFigure copied to {_dst}")
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
