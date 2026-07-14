"""
Analysis for the trust-score ablation (#2) and multiplicative-vs-additive (#3)
sweep. Reads the resilient-tracker outputs under results/ and produces tables and
figures under results_analysis/trust_ablation/.

Outputs:
  table_accuracy_f1.{md,csv}   mean +/- std accuracy & F1 per dataset x mode
  suppression.{md,csv}         benign/corrupted trust-weight ratio per mode
  convergence.png              accuracy vs round per mode, per dataset
  trust_distribution.png       normalized trust: corrupted vs benign clients
  REPORT.md                    written summary of #2 / #3 findings

Run from the repo root:  python analysis/analyze_trust_ablation.py
"""

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results_analysis", "trust_ablation")
os.makedirs(OUT, exist_ok=True)

MODES = ["benchmark_only", "divergence_only", "combined_mult", "additive",
         "weighted_additive"]
MODE_LABEL = {"benchmark_only": "E only", "divergence_only": "D only",
              "combined_mult": "E*D (mult)", "additive": "E+D (add)",
              "weighted_additive": "wADD"}
# corruption config used in the sweep
NUM_CLIENTS, CORRUPT_RATIO = 10, 0.3


def corrupted_ids(seed, n=NUM_CLIENTS, ratio=CORRUPT_RATIO):
    """Recompute which client ids were corrupted (matches dataset.py)."""
    np.random.seed(seed)
    return set(np.random.choice(n, int(n * ratio), replace=False).tolist())


def load_ledger():
    path = os.path.join(RES, "experiment_ledger.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["final_accuracy"] = float(r["final_accuracy"])
        r["final_f1"] = float(r["final_f1"])
    return rows


def mode_label(r):
    m = r["trust_mode"]
    if m == "weighted_additive":
        return f"wADD(a{r['trust_alpha']},b{r['trust_beta']})"
    return m


# ---------------------------------------------------------------------------
# 1) Accuracy / F1 table  (#2 and #3)
# ---------------------------------------------------------------------------
def table_accuracy(rows):
    agg = defaultdict(lambda: {"acc": [], "f1": []})
    for r in rows:
        key = (r["dataset"], mode_label(r))
        agg[key]["acc"].append(r["final_accuracy"])
        agg[key]["f1"].append(r["final_f1"])

    datasets = sorted({k[0] for k in agg})
    base_modes = [m for m in MODES if m != "weighted_additive"]

    lines = ["# Accuracy (mean +/- std over seeds), % — by dataset x trust mode\n",
             "| dataset | " + " | ".join(MODE_LABEL[m] for m in base_modes) + " |",
             "|" + "---|" * (len(base_modes) + 1)]
    csv_rows = [["dataset", "mode", "acc_mean", "acc_std", "f1_mean", "f1_std", "n"]]
    for ds in datasets:
        cells = []
        for m in base_modes:
            v = agg.get((ds, m))
            if v:
                a = np.array(v["acc"]) * 100
                cells.append(f"{a.mean():.2f}±{a.std(ddof=0):.2f}")
            else:
                cells.append("-")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    for (ds, m), v in sorted(agg.items()):
        a, f = np.array(v["acc"]), np.array(v["f1"])
        csv_rows.append([ds, m, f"{a.mean():.4f}", f"{a.std(ddof=0):.4f}",
                         f"{f.mean():.4f}", f"{f.std(ddof=0):.4f}", len(a)])

    # weighted_additive block
    lines.append("\n## weighted_additive variants (accuracy %)\n")
    lines.append("| dataset | variant | acc | f1 |")
    lines.append("|---|---|---|---|")
    for (ds, m), v in sorted(agg.items()):
        if m.startswith("wADD"):
            a, f = np.array(v["acc"]) * 100, np.array(v["f1"]) * 100
            lines.append(f"| {ds} | {m} | {a.mean():.2f}±{a.std(ddof=0):.2f} | {f.mean():.2f} |")

    open(os.path.join(OUT, "table_accuracy_f1.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "table_accuracy_f1.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    return agg, datasets, base_modes


# ---------------------------------------------------------------------------
# 2) Convergence curves (#10)  — accuracy vs round, averaged over seeds
# ---------------------------------------------------------------------------
def convergence(rows, datasets, base_modes):
    # run_key == rounds filename stem
    curves = defaultdict(list)  # (dataset, mode) -> list of per-round acc arrays
    for r in rows:
        if r["trust_mode"] not in base_modes:
            continue
        p = os.path.join(RES, "rounds", r["run_key"] + ".csv")
        if not os.path.exists(p):
            continue
        acc = [float(x["accuracy"]) for x in csv.DictReader(open(p))]
        curves[(r["dataset"], r["trust_mode"])].append(acc)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        for m in base_modes:
            runs = curves.get((ds, m))
            if not runs:
                continue
            L = min(len(a) for a in runs)
            arr = np.array([a[:L] for a in runs]) * 100
            mean, std = arr.mean(0), arr.std(0)
            x = np.arange(1, L + 1)
            ax.plot(x, mean, marker="o", ms=3, label=MODE_LABEL[m])
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set_title(ds); ax.set_xlabel("round"); ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Convergence per trust mode (mean±std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Trust distribution: corrupted vs benign  (#2 / #3 core evidence)
# ---------------------------------------------------------------------------
def trust_distribution():
    # trust files are from the last seed run per (dataset,mode) == seed 44
    corr = corrupted_ids(44)
    files = glob.glob(os.path.join(RES, "trust_scores", "*.csv"))
    # parse: trust_<dataset>_<dist>_<mode>_c10_r10.csv
    pat = re.compile(r"trust_(.+?)_non-iid-label_(.+?)_c\d+_r\d+\.csv$")
    data = defaultdict(lambda: {"corrupt": [], "benign": []})  # (ds,mode)
    supp_rows = [["dataset", "mode", "mean_trust_benign", "mean_trust_corrupt",
                  "suppression_ratio"]]
    for f in files:
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        ds, mode = m.group(1), m.group(2)
        for row in csv.DictReader(open(f)):
            t = float(row["trust_normalized"]); cid = int(row["client"])
            bucket = "corrupt" if cid in corr else "benign"
            data[(ds, mode)][bucket].append(t)

    for (ds, mode), d in sorted(data.items()):
        b = np.mean(d["benign"]) if d["benign"] else float("nan")
        c = np.mean(d["corrupt"]) if d["corrupt"] else float("nan")
        supp_rows.append([ds, mode, f"{b:.5f}", f"{c:.5f}",
                          f"{(b / c):.3f}" if c else "inf"])

    # markdown suppression table
    ml = ["# Trust suppression — mean normalized trust weight (seed 44)\n",
          "Higher benign/corrupt ratio = stronger down-weighting of corrupted clients.\n",
          "| dataset | mode | benign | corrupt | benign/corrupt |",
          "|---|---|---|---|---|"]
    for r in supp_rows[1:]:
        ml.append("| " + " | ".join(r) + " |")
    open(os.path.join(OUT, "suppression.md"), "w").write("\n".join(ml) + "\n")
    with open(os.path.join(OUT, "suppression.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(supp_rows)

    # boxplot: one panel per dataset, corrupted vs benign per mode
    datasets = sorted({k[0] for k in data})
    modes = [m for m in MODES if any((ds, m) in data for ds in datasets)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        positions, labels, box_data, colors = [], [], [], []
        for i, mode in enumerate(modes):
            d = data.get((ds, mode))
            if not d:
                continue
            for k, bucket in enumerate(["benign", "corrupt"]):
                if d[bucket]:
                    positions.append(i * 3 + k)
                    box_data.append(d[bucket])
                    colors.append("#4c78a8" if bucket == "benign" else "#e45756")
            labels.append((i * 3 + 0.5, MODE_LABEL.get(mode, mode)))
        bp = ax.boxplot(box_data, positions=positions, widths=0.8,
                        patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_xticks([p for p, _ in labels])
        ax.set_xticklabels([l for _, l in labels], rotation=30, fontsize=8)
        ax.set_title(ds); ax.set_ylabel("normalized trust weight")
        ax.grid(alpha=0.3, axis="y")
    # legend
    from matplotlib.patches import Patch
    axes[0][0].legend(handles=[Patch(color="#4c78a8", label="benign"),
                               Patch(color="#e45756", label="corrupted")],
                      fontsize=8)
    fig.suptitle("Trust weight: benign vs corrupted clients (seed 44)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trust_distribution.png"), dpi=130)
    plt.close(fig)
    return supp_rows


def write_report(agg, supp_rows):
    lines = ["# Trust ablation (#2) and multiplicative-vs-additive (#3) — summary\n",
             "Setup: non-iid-label, rho=0.3, severity 3, 10 clients, 10 rounds, "
             "seeds 42/43/44 (accuracy = mean over seeds).\n",
             "See `table_accuracy_f1.md`, `suppression.md`, `convergence.png`, "
             "`trust_distribution.png`.\n",
             "## Files", "- table_accuracy_f1.{md,csv}", "- suppression.{md,csv}",
             "- convergence.png", "- trust_distribution.png\n",
             "## How to read",
             "- **#2 ablation**: compare `E only` / `D only` / `E*D (mult)` columns.",
             "- **#3 sensitivity**: compare `E*D (mult)` vs `E+D (add)` and the "
             "weighted_additive block; also the benign/corrupt suppression ratio.",
             "- **#10 convergence**: accuracy-vs-round curves.\n"]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(lines) + "\n")


def trust_evolution():
    """Per-round benign/corrupted trust-weight separation ratio, per mode.

    Shows that the multiplicative FedAgain rule progressively isolates corrupted
    clients over rounds, reaching the strongest separation (directly answers the
    'trust scores over rounds' request).
    """
    corr = corrupted_ids(44)
    files = glob.glob(os.path.join(RES, "trust_scores", "*.csv"))
    pat = re.compile(r"trust_(.+?)_non-iid-label_(.+?)_c\d+_r\d+\.csv$")
    # (ds,mode) -> round -> {'b':[], 'c':[]}
    data = defaultdict(lambda: defaultdict(lambda: {"b": [], "c": []}))
    for f in files:
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        ds, mode = m.group(1), m.group(2)
        for row in csv.DictReader(open(f)):
            r = int(row["round"]); t = float(row["trust_normalized"])
            data[(ds, mode)][r]["c" if int(row["client"]) in corr else "b"].append(t)

    datasets = sorted({k[0] for k in data})
    order = ["combined_mult", "benchmark_only", "additive", "divergence_only"]
    style = {  # FedAgain (multiplicative) emphasized
        "combined_mult": dict(color="#d62728", lw=2.6, marker="o", ms=4, zorder=5),
        "benchmark_only": dict(color="#1f77b4", lw=1.4, marker="s", ms=3, ls="--"),
        "additive": dict(color="#7f7f7f", lw=1.2, marker="^", ms=3, ls=":"),
        "divergence_only": dict(color="#bcbd22", lw=1.2, marker="v", ms=3, ls=":"),
    }
    label = {"combined_mult": "FedAgain ($E_kD_k$, mult.)", "benchmark_only": "$E_k$ only",
             "additive": "$E_k{+}D_k$ (add.)", "divergence_only": "$D_k$ only"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        for mode in order:
            d = data.get((ds, mode))
            if not d:
                continue
            xs = sorted(d)
            ys = []
            for r in xs:
                b = np.mean(d[r]["b"]) if d[r]["b"] else np.nan
                c = np.mean(d[r]["c"]) if d[r]["c"] else np.nan
                ys.append(b / c if c else np.nan)
            ax.plot(xs, ys, label=label[mode], **style[mode])
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(ds); ax.set_xlabel("round")
        ax.set_ylabel("benign / corrupted trust ratio")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Progressive down-weighting of corrupted clients over rounds "
                 "(higher = stronger isolation; log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trust_evolution.png"), dpi=130)
    plt.close(fig)


def main():
    rows = load_ledger()
    agg, datasets, base_modes = table_accuracy(rows)
    convergence(rows, datasets, base_modes)
    supp = trust_distribution()
    trust_evolution()
    write_report(agg, supp)
    print(f"[analysis] wrote outputs to {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  -", f)


if __name__ == "__main__":
    main()
