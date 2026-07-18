#!/usr/bin/env python3
"""
Label-flipping poisoning attack analysis (Reviewer 1.3 / 2.3).

The reviewers noted that the original manuscript described the threat model as
"adversarial" while the experiments only injected *image-level* corruption. This
script analyses a genuine data-poisoning attack -- label flipping, where a
fraction of the corrupted clients train on deliberately mislabeled data (images
untouched, labels reassigned to a wrong class) -- to show that FedAgain's trust
re-weighting defends against a real adversary, not merely against noisy inputs.

Data: the 72-run label-flip sweep (pm-label_flip in the run key) --
FedAgain vs FedAvg (undefended) vs FedMedian, on four datasets, at corrupted
fractions rho in {0.1, 0.3}, three seeds each. As a reference point, the same
(strategy, dataset, rho) cells under *image* corruption (severity 3) are loaded
from the envelope sweep so the two threat models can be contrasted.

Outputs:
  results/baselines/label_flip_summary.csv
  results/baselines/label_flip.png  (+ copy to src-paper/)
Prints the headline numbers for the paper paragraph.
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
SRCPAPER = os.path.join(ROOT, "jcr-fedagain", "src-paper")
os.makedirs(OUTDIR, exist_ok=True)

DATASETS = ["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE", "CIFAR10"]
DS_LABEL = {"DAUDON_MIX": "A (Daudon)", "JONATHAN_MIX": "B (Jonathan)",
            "MYSTONE": "C (MyStone)", "CIFAR10": "CIFAR-10"}
STRATS = ["FedAvg", "FedMedian", "FedAgain"]
RHOS = [0.1, 0.3]

COLOR = {"FedAvg": "#b0b0b0", "FedMedian": "#1f77b4", "FedAgain": "#d62728"}


def load():
    """Return {(mode, dataset, strategy, rho): [acc, ...]} with mode in
    {'label_flip', 'image'}."""
    agg = defaultdict(list)
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            if (r["status"] != "completed" or r["dataset"] not in DATASETS
                    or r["strategy"] not in STRATS
                    or r["data_distribution"] != "non-iid-label"
                    or r["corrupt_clients"] != "True"
                    or r["corruption_severity"] != "3"):
                continue
            try:
                rho = float(r["corrupt_ratio"])
            except ValueError:
                continue
            if rho not in RHOS:
                continue
            mode = "label_flip" if "pm-label_flip" in r["run_key"] else "image"
            agg[(mode, r["dataset"], r["strategy"], rho)].append(
                float(r["final_accuracy"]) * 100)
    return agg


def main():
    agg = load()

    def stat(mode, ds, s, rho):
        v = agg.get((mode, ds, s, rho))
        if not v:
            return None, None, 0
        return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)

    # ---- CSV summary ----
    with open(os.path.join(OUTDIR, "label_flip_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threat", "dataset", "strategy", "rho", "acc_mean", "acc_std", "n"])
        for mode in ("label_flip", "image"):
            for ds in DATASETS:
                for s in STRATS:
                    for rho in RHOS:
                        m, sd, n = stat(mode, ds, s, rho)
                        if n:
                            w.writerow([mode, ds, s, rho, f"{m:.2f}", f"{sd:.2f}", n])

    # ---- Headline: FedAgain margin over undefended FedAvg under label-flip ----
    print("=== Label-flipping: accuracy by strategy (mean over 3 seeds) ===")
    margins_fa_over_avg = []
    margins_fa_over_med = []
    for rho in RHOS:
        print(f"\n-- rho = {rho} --")
        print(f"{'dataset':<13} {'FedAvg':>8} {'FedMedian':>10} {'FedAgain':>9}"
              f"  {'FA-Avg':>7} {'FA-Med':>7}")
        for ds in DATASETS:
            avg = stat("label_flip", ds, "FedAvg", rho)[0]
            med = stat("label_flip", ds, "FedMedian", rho)[0]
            fa = stat("label_flip", ds, "FedAgain", rho)[0]
            if None in (avg, med, fa):
                continue
            margins_fa_over_avg.append(fa - avg)
            margins_fa_over_med.append(fa - med)
            print(f"{ds:<13} {avg:>8.1f} {med:>10.1f} {fa:>9.1f}"
                  f"  {fa-avg:>+7.1f} {fa-med:>+7.1f}")

    print(f"\nFedAgain vs FedAvg (undefended), label-flip: "
          f"mean margin {st.mean(margins_fa_over_avg):+.1f} pts "
          f"(min {min(margins_fa_over_avg):+.1f}, max {max(margins_fa_over_avg):+.1f})")
    print(f"FedAgain vs FedMedian, label-flip:            "
          f"mean margin {st.mean(margins_fa_over_med):+.1f} pts "
          f"(min {min(margins_fa_over_med):+.1f}, max {max(margins_fa_over_med):+.1f})")

    # ---- Contrast label-flip vs image corruption at same rho ----
    print("\n=== FedAgain: label-flip vs image corruption (same rho) ===")
    for rho in RHOS:
        lf = [stat("label_flip", ds, "FedAgain", rho)[0] for ds in DATASETS]
        im = [stat("image", ds, "FedAgain", rho)[0] for ds in DATASETS]
        lf = [x for x in lf if x is not None]
        im = [x for x in im if x is not None]
        print(f"  rho={rho}: label-flip avg {st.mean(lf):.1f}%  "
              f"image avg {st.mean(im):.1f}%  (diff {st.mean(lf)-st.mean(im):+.1f})")

    # ---- Figure: grouped bars, one panel per dataset, rho=0.3 (stronger attack) ----
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
    rho_plot = 0.3
    for ax, ds in zip(axes, DATASETS):
        xs = np.arange(len(STRATS))
        for i, s in enumerate(STRATS):
            m, sd, n = stat("label_flip", ds, s, rho_plot)
            if n:
                ax.bar(i, m, yerr=sd, color=COLOR[s], capsize=3,
                       edgecolor="black", linewidth=0.6, width=0.7)
                ax.text(i, m + (sd or 0) + 1.0, f"{m:.0f}", ha="center",
                        va="bottom", fontsize=11)
        ax.set_title(DS_LABEL[ds], fontsize=13)
        ax.set_xticks(xs)
        ax.set_xticklabels(STRATS, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", ls=":", alpha=0.5)
        if ds == DATASETS[0]:
            ax.set_ylabel("Test accuracy (%)")
    fig.suptitle(r"Robustness to a label-flipping poisoning attack "
                 r"($\rho=0.3$, three seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTDIR, "label_flip.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if os.path.isdir(SRCPAPER):
        import shutil
        shutil.copy(out, os.path.join(SRCPAPER, "label_flip.png"))
        print("\nFigure copied to src-paper/label_flip.png")
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
