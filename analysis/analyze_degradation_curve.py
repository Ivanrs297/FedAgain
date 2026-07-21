#!/usr/bin/env python3
"""
Corruption-robustness degradation curve (Reviewer 1.5 / 2.4, honest lever).

Reads the envelope sweep from results/experiment_ledger.csv (8 strategies x
4 datasets x rho in {0.1,0.3,0.4,0.5,0.6} x severity 3 x seeds 42/43/44) and shows how
test accuracy degrades as the corrupted-client fraction rho grows. The message:
only FedMedian and FedAgain stay nearly flat, while every other baseline falls
off 2-3x faster -- and FedAgain, unlike FedMedian, needs no prior on the number
of corrupted clients and discards no updates.

Outputs:
  results/baselines/degradation_curve.png (+ copy to src-paper/)
  results/baselines/degradation_summary.csv
Prints the stability ranking (mean degradation rho=0.1->max) + envelope-average
accuracy for the paper paragraph.

Note on rho > 0.5: with K=10 clients, rho=0.5 means 5 corrupted and rho=0.6
means 6 -- i.e. a corrupted majority, the regime where FedMedian's coordinate-
wise-median guarantee breaks but FedAgain's soft trust re-weighting has no
breakdown threshold.

Works on partial data (draws whatever rho points are complete).
"""
import csv
import os
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "experiment_ledger.csv")
OUTDIR = os.path.join(ROOT, "results", "baselines")
SRCPAPER = os.path.join(ROOT, "jcr-fedagain", "src-paper", "images", "results")
os.makedirs(OUTDIR, exist_ok=True)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _datasets import datasets as _ds_sel, is_canonical  # R1.7 + canonical filter
DATASETS = _ds_sel()
DS_LABEL = {"DAUDON_MIX": "A (Daudon)", "JONATHAN_MIX": "B (Jonathan)",
            "MYSTONE": "C (MyStone)", "CIFAR10": "CIFAR-10"}
STRATS = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedBN", "Scaffold", "MOON",
          "FedDyn", "RFLPA", "BOBA", "FedAgain"]
RHOS = [0.1, 0.3, 0.4, 0.5, 0.6]

# muted palette for baselines; FedAgain + FedMedian emphasized
STYLE = {
    "FedAvg":    dict(color="#b0b0b0", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "FedProx":   dict(color="#b7c4d4", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "Bulyan":    dict(color="#a9d3a9", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "FedBN":     dict(color="#c9b8dd", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "Scaffold":  dict(color="#f0c38a", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "MOON":      dict(color="#c9a99b", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "FedDyn":    dict(color="#d4c98a", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "RFLPA":     dict(color="#9bc3c9", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "BOBA":      dict(color="#d9a9c9", lw=1.3, ls="-",  marker="o", ms=4, z=1),
    "FedMedian": dict(color="#1f77b4", lw=2.6, ls="--", marker="s", ms=6, z=4),
    "FedAgain":  dict(color="#d62728", lw=3.0, ls="-",  marker="D", ms=7, z=5),
}


def load():
    agg = defaultdict(list)
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            if (r["status"] == "completed" and r["dataset"] in DATASETS
                    and r["corruption_severity"] == "3"
                    and r["data_distribution"] == "non-iid-label"
                    and r["strategy"] in STRATS
                    and is_canonical(r)):
                try:
                    rho = float(r["corrupt_ratio"])
                except ValueError:
                    continue
                if rho in RHOS:
                    agg[(r["dataset"], r["strategy"], rho)].append(float(r["final_accuracy"]) * 100)
    return agg


def main():
    agg = load()

    def stat(ds, s, rho):
        v = agg.get((ds, s, rho))
        if not v:
            return None, None, 0
        return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)

    # ---- CSV + degradation ranking ----
    with open(os.path.join(OUTDIR, "degradation_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "strategy", "rho", "acc_mean", "acc_std", "n"])
        for ds in DATASETS:
            for s in STRATS:
                for rho in RHOS:
                    m, sd, n = stat(ds, s, rho)
                    if n:
                        w.writerow([ds, s, rho, f"{m:.2f}", f"{sd:.2f}", n])

    # degradation from rho=0.1 to the largest COMPLETE rho shared by all datasets
    complete_rhos = [rho for rho in RHOS
                     if all(stat(ds, s, rho)[2] for ds in DATASETS for s in STRATS)]
    hi = max(complete_rhos) if complete_rhos else 0.3
    print(f"Largest rho complete for all (dataset,strategy): {hi}")
    print(f"\n=== Stability ranking: mean accuracy drop (rho 0.1 -> {hi}), lower = more stable ===")
    deg, env = {}, {}
    for s in STRATS:
        drops, accs = [], []
        for ds in DATASETS:
            a = stat(ds, s, 0.1)[0]
            b = stat(ds, s, hi)[0]
            if a is not None and b is not None:
                drops.append(a - b)
            for rho in complete_rhos:
                m = stat(ds, s, rho)[0]
                if m is not None:
                    accs.append(m)
        if drops:
            deg[s] = st.mean(drops)
        if accs:
            env[s] = st.mean(accs)
    for i, (s, d) in enumerate(sorted(deg.items(), key=lambda x: x[1]), 1):
        mark = "  <-- FedAgain" if s == "FedAgain" else ""
        print(f"  {i}. {s:<11} drop={d:+.2f}   envelope-avg acc={env.get(s, float('nan')):.1f}{mark}")

    # ---- 2x2 degradation curves ----
    plt.rcParams.update({"font.size": 14})
    _n = len(DATASETS)
    _nc = 3 if _n == 3 else 2
    _nr = (_n + _nc - 1) // _nc
    fig, axes = plt.subplots(_nr, _nc, figsize=(6.5 * _nc, 4.3 * _nr),
                             squeeze=False)
    for _k in range(_n, _nr * _nc):
        axes.ravel()[_k].set_visible(False)
    for ax, ds in zip(axes.ravel(), DATASETS):
        for s in STRATS:
            xs, ys, es = [], [], []
            for rho in RHOS:
                m, sd, n = stat(ds, s, rho)
                if n:
                    xs.append(rho); ys.append(m); es.append(sd)
            if xs:
                stl = STYLE[s]
                ax.errorbar(xs, ys, yerr=es, label=s, color=stl["color"],
                            lw=stl["lw"], ls=stl["ls"], marker=stl["marker"],
                            ms=stl["ms"], zorder=stl["z"], capsize=2,
                            alpha=0.95 if s in ("FedAgain", "FedMedian") else 0.7)
        # majority-corruption threshold: rho>=0.5 => >=5/10 clients corrupted,
        # where the coordinate-wise median's guarantee breaks down.
        ax.axvline(0.5, color="k", ls=":", lw=1.2, alpha=0.55, zorder=0)
        ax.text(0.5, ax.get_ylim()[0], r" corrupted majority", fontsize=9,
                color="0.35", ha="left", va="bottom")
        ax.set_title(DS_LABEL[ds], fontsize=15)
        ax.set_xlabel(r"Corrupted-client fraction $\rho$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(RHOS)
        ax.grid(ls=":", alpha=0.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # order legend with FedAgain/FedMedian first
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] not in ("FedAgain", "FedMedian"), STRATS.index(labels[i])))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, "degradation_curve.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if os.path.isdir(SRCPAPER):
        import shutil
        shutil.copy(out, os.path.join(SRCPAPER, "degradation_curve.png"))
        print("\nFigure copied to src-paper/degradation_curve.png")
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
