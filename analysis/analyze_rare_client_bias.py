#!/usr/bin/env python3
"""
Rare / shifted-distribution client bias (Reviewer 2.2).

Concern: does FedAgain's trust weighting unfairly suppress *legitimate* clients
whose local distribution is intrinsically harder/rarer (high benchmark error),
treating them like corrupted clients?

This script reconstructs, deterministically from the run seed (44) and the same
partition/corruption RNG used in the sweep, each client's:
  - corrupted vs. benign status (rho=0.3),
  - class composition and an intrinsic-difficulty proxy (benchmark error E_k),
and joins it with the logged per-round trust weights for the multiplicative
FedAgain rule vs. the benchmark-only and divergence-only ablations.

Key question answered: among BENIGN clients, are the hard/rare ones down-weighted
as much as CORRUPTED ones? If the multiplicative rule keeps benign-hard clients
clearly above corrupted (thanks to the divergence signal D_k, which is what
actually separates corruption), then FedAgain suppresses corruption, not rarity.

Reads only existing logs + reproduces the CPU-only partition; does NOT run
training and does NOT modify config.py (safe to run alongside a live sweep).

Outputs: results/baselines/rare_client_bias.{csv,png} (+ figure copy to src-paper/).
"""
import csv
import os
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.dataset import load_dataset, partition_data_non_iid_label  # noqa: E402

OUTDIR = os.path.join(ROOT, "results", "baselines")
TRUSTDIR = os.path.join(ROOT, "results", "trust_scores")
SRCPAPER = os.path.join(ROOT, "jcr-fedagain", "src-paper", "images", "results")
os.makedirs(OUTDIR, exist_ok=True)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _datasets import datasets as _ds_sel, fig_name, srcpaper_dir, cifar_only  # R1.7 + CIFAR grid
DATASETS = _ds_sel()
DS_LABEL = {"DAUDON_MIX": "A (Daudon)", "JONATHAN_MIX": "B (Jonathan)",
            "MYSTONE": "C (MyStone)", "CIFAR10": "CIFAR-10"}
SEED = 44
NUM_CLIENTS = 10
CLASSES_PER_CLIENT = 2
CORRUPT_RATIO = 0.3


def labels_of(dataset):
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def corrupted_ids():
    """Reproduce apply_corruptions_to_clients' client selection exactly."""
    num_corrupt = int(NUM_CLIENTS * CORRUPT_RATIO)
    np.random.seed(SEED)
    return set(int(x) for x in np.random.choice(NUM_CLIENTS, num_corrupt, replace=False))


def client_class_composition(dataset):
    """Return, per client, the label -> count dict (reproduces the sweep partition)."""
    labels = labels_of(dataset)
    parts = partition_data_non_iid_label(dataset, NUM_CLIENTS, CLASSES_PER_CLIENT, seed=SEED)
    comps = []
    for sub in parts:
        idx = np.asarray(sub.indices)
        cl = labels[idx]
        vals, cnts = np.unique(cl, return_counts=True)
        comps.append(dict(zip(vals.tolist(), cnts.tolist())))
    return comps, labels


def load_trust(dataset, mode):
    """mean over rounds of trust_normalized, benchmark_error, divergence per client."""
    path = os.path.join(TRUSTDIR, f"trust_{dataset}_non-iid-label_{mode}_c10_r10.csv")
    if not os.path.exists(path):
        return None
    acc = defaultdict(lambda: {"t": [], "e": [], "d": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            c = int(r["client"])
            acc[c]["t"].append(float(r["trust_normalized"]))
            acc[c]["e"].append(float(r["benchmark_error"]))
            acc[c]["d"].append(float(r["divergence"]))
    return {c: (st.mean(v["t"]), st.mean(v["e"]), st.mean(v["d"])) for c, v in acc.items()}


def main():
    corrupted = corrupted_ids()
    print(f"Corrupted client ids (seed {SEED}, rho {CORRUPT_RATIO}): {sorted(corrupted)}")

    rows = []  # dataset, client, corrupted, size, n_primary_classes, E, D, trust_mult, trust_bench, trust_div
    group_trust = defaultdict(lambda: defaultdict(list))  # dataset -> group -> [trust_mult]

    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        train, _, info = load_dataset(ds)
        comps, _ = client_class_composition(train)
        tm = load_trust(ds, "combined_mult")
        tb = load_trust(ds, "benchmark_only")
        td = load_trust(ds, "divergence_only")
        if tm is None:
            print("  (no trust log; skipping)")
            continue
        # benign difficulty split by benchmark error E (intrinsic hardness)
        benign = [c for c in range(NUM_CLIENTS) if c not in corrupted]
        benE = {c: tm[c][1] for c in benign if c in tm}
        med = st.median(benE.values()) if benE else 0.0
        for c in range(NUM_CLIENTS):
            if c not in tm:
                continue
            t_mult, E, D = tm[c]
            size = sum(comps[c].values())
            is_corr = c in corrupted
            if is_corr:
                group = "corrupted"
            else:
                group = "benign-hard" if E > med else "benign-easy"
            group_trust[ds][group].append(t_mult)
            rows.append([ds, c, int(is_corr), size, len(comps[c]),
                         f"{E:.4f}", f"{D:.4f}", f"{t_mult:.5f}",
                         f"{tb[c][0]:.5f}" if tb and c in tb else "",
                         f"{td[c][0]:.5f}" if td and c in td else "",
                         group])
        # per-dataset summary
        for g in ["benign-easy", "benign-hard", "corrupted"]:
            v = group_trust[ds].get(g, [])
            if v:
                print(f"  {g:<12} mean trust = {st.mean(v):.4f}  (n={len(v)})")

    # write CSV
    with open(os.path.join(OUTDIR, "rare_client_bias.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "client", "corrupted", "size", "n_classes",
                    "benchmark_error", "divergence", "trust_mult",
                    "trust_bench_only", "trust_div_only", "group"])
        w.writerows(rows)

    # ---- figure: grouped mean trust per dataset for the 3 groups ----
    groups = ["benign-easy", "benign-hard", "corrupted"]
    colors = {"benign-easy": "#1f77b4", "benign-hard": "#ff9e4a", "corrupted": "#d62728"}
    plt.rcParams.update({"font.size": 13 if cifar_only() else 14})
    fig, ax = plt.subplots(figsize=(5.6, 4.4) if cifar_only() else (11, 4.6))
    x = np.arange(len(DATASETS))
    w = 0.26
    # uniform-weight reference = 1/num_clients (equal trust to all)
    for i, g in enumerate(groups):
        means = [st.mean(group_trust[ds][g]) if group_trust[ds].get(g) else 0 for ds in DATASETS]
        off = (i - 1) * w
        ax.bar(x + off, means, w, label=g, color=colors[g],
               edgecolor="black", linewidth=0.6)
    ax.axhline(1.0 / NUM_CLIENTS, ls="--", color="gray", lw=1.2,
               label="uniform (1/K)")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Mean normalized trust")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(ncol=2 if cifar_only() else 4, loc="upper center",
              bbox_to_anchor=(0.5, 0.995 if cifar_only() else 1.15),
              frameon=False, fontsize=10 if cifar_only() else 12)
    fig.tight_layout(rect=[0, 0, 1, 0.80] if cifar_only() else None)
    out = os.path.join(OUTDIR, fig_name("rare_client_bias") + ".png")
    fig.savefig(out, dpi=200, bbox_inches=None if cifar_only() else "tight")
    _dst = srcpaper_dir()
    if os.path.isdir(_dst):
        import shutil
        shutil.copy(out, os.path.join(_dst, fig_name("rare_client_bias") + ".png"))
        print(f"\nFigure copied to {_dst}")
    print(f"Figure -> {out}")
    print(f"CSV    -> {os.path.join(OUTDIR, 'rare_client_bias.csv')}")

    # ---- overall honest verdict numbers ----
    print("\n=== Overall (mean over datasets) ===")
    for g in groups:
        allv = [st.mean(group_trust[ds][g]) for ds in DATASETS if group_trust[ds].get(g)]
        if allv:
            print(f"  {g:<12} {st.mean(allv):.4f}")


if __name__ == "__main__":
    main()
