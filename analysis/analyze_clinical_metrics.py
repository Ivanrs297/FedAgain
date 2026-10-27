"""
Clinical metrics for the second revision (Reviewer #3, comment 1).

Reads the per-run manifests of a results tree (default results_v2/, written by
main.py with per-class metrics since 2026-10) and reports, mean +- sample SD
(ddof=1) over seeds:
  1. canonical setting (rho=0.3, image corruption), every strategy x clinical
     dataset: accuracy, balanced accuracy (= mean per-subtype sensitivity),
     macro-F1;
  2. per-subtype sensitivity and specificity for every strategy and dataset;
  3. KIDNEY_MULTISITE (natural three-site federation): accuracy, balanced
     accuracy and accuracy per acquisition site, for each corruption level;
  4. FedAgain with trust/data-size mixing (lambda > 0, run-key tag _mix<l>)
     next to plain FedAgain (lambda = 0) and FedAvg / FedMedian, on the
     multi-site federation and on the clinical datasets.

Outputs (in <results>/clinical/): clinical_summary.md and four CSV files.

    FEDAGAIN_RESULTS_DIR=results_v2 python analysis/analyze_clinical_metrics.py
"""

import csv
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _datasets import RESULTS, CLINICAL, DS_LABEL, is_canonical_key  # noqa: E402

OUT = os.path.join(RESULTS, "clinical")
STRATS = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedBN", "Scaffold", "MOON",
          "FedDyn", "RFLPA", "BOBA", "FedAgain"]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


def fmt(xs, scale=100.0):
    m, s, n = ms(xs)
    return "--" if n == 0 else f"{m*scale:.2f} ± {s*scale:.2f}"


def load_runs():
    runs = []
    for f in glob.glob(os.path.join(RESULTS, "runs", "*.json")):
        m = json.load(open(f))
        if m.get("status") == "completed" and "final_metrics" in m:
            runs.append(m)
    return runs


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()
    canon = defaultdict(list)       # (dataset, strategy) -> [final_metrics]
    multi = defaultdict(list)       # (rho, strategy) -> [final_metrics]
    mix = defaultdict(list)         # (dataset, rho, lambda) -> [final_metrics]
    for m in runs:
        c, fm, key = m["config"], m["final_metrics"], m["run_key"]
        lam = re.search(r"_mix([0-9.]+)_s\d+$", key)
        if (lam and c["strategy"] == "FedAgain" and "_eps" not in key and "nm-" not in key
                and c.get("poison_mode", "image") == "image"
                and c.get("trust_mode") == "combined_mult"):
            rho = float(c["corrupt_ratio"]) if c.get("corrupt_clients") else 0.0
            mix[(c["dataset"], rho, float(lam.group(1)))].append(fm)
            continue
        if not is_canonical_key(key) or c.get("poison_mode", "image") != "image":
            continue
        rho = float(c["corrupt_ratio"]) if c.get("corrupt_clients") else 0.0
        if c["dataset"] == "KIDNEY_MULTISITE":
            multi[(rho, c["strategy"])].append(fm)
        elif c["dataset"] in CLINICAL and rho == 0.3:
            canon[(c["dataset"], c["strategy"])].append(fm)

    md = ["# Clinical metrics (results tree: " + os.path.basename(RESULTS) + ")", "",
          "Mean ± sample SD (ddof=1) over seeds, in %.", ""]

    # 1. canonical table
    rows = [["dataset", "strategy", "n", "accuracy", "balanced_accuracy", "macro_f1"]]
    md += ["## 1. Canonical setting (non-IID label skew, rho=0.3, severity 3)", ""]
    for ds in CLINICAL:
        md += [f"### {DS_LABEL[ds]}", "", "| strategy | n | accuracy | balanced acc. | macro-F1 |",
               "|---|---|---|---|---|"]
        for s in STRATS:
            fms = canon.get((ds, s), [])
            if not fms:
                continue
            acc = fmt([x["accuracy"] for x in fms])
            bal = fmt([x.get("balanced_accuracy") for x in fms])
            f1 = fmt([x["f1"] for x in fms])
            md.append(f"| {s} | {len(fms)} | {acc} | {bal} | {f1} |")
            rows.append([ds, s, len(fms), acc, bal, f1])
        md.append("")
    with open(os.path.join(OUT, "canonical_clinical_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # 2. per-subtype sensitivity / specificity
    rows = [["dataset", "strategy", "subtype", "n", "sensitivity", "specificity"]]
    md += ["## 2. Per-subtype sensitivity (rho=0.3)", ""]
    for ds in CLINICAL:
        names = None
        for s in STRATS:
            fms = canon.get((ds, s), [])
            if fms and fms[0].get("class_names"):
                names = fms[0]["class_names"]
                break
        if names is None:
            continue
        short = [n.split("_")[-1] for n in names]
        md += [f"### {DS_LABEL[ds]}", "", "| strategy | " + " | ".join(short) + " |",
               "|---|" + "---|" * len(short)]
        for s in STRATS:
            fms = canon.get((ds, s), [])
            if not fms or "per_class_sensitivity" not in fms[0]:
                continue
            cells = []
            for i, sub in enumerate(short):
                se = [x["per_class_sensitivity"][i] for x in fms]
                sp = [x["per_class_specificity"][i] for x in fms]
                cells.append(fmt(se))
                rows.append([ds, s, sub, len(fms), fmt(se), fmt(sp)])
            md.append(f"| {s} | " + " | ".join(cells) + " |")
        md.append("")
    with open(os.path.join(OUT, "per_subtype_sensitivity_specificity.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # 3. multi-site federation
    rows = [["rho", "strategy", "n", "accuracy", "balanced_accuracy", "site_A", "site_B", "site_C"]]
    md += ["## 3. Natural three-site federation (KIDNEY_MULTISITE)", "",
           "Sites: A = Daudon (CCD, ex vivo), B = Jonathan El-Beze (endoscope), C = MyStone.", ""]
    for rho in sorted({k[0] for k in multi}):
        md += [f"### rho = {rho}", "", "| strategy | n | accuracy | balanced acc. | site A | site B | site C |",
               "|---|---|---|---|---|---|---|"]
        for s in STRATS:
            fms = multi.get((rho, s), [])
            if not fms:
                continue
            site = {k: fmt([x.get("per_site_accuracy", {}).get(k) for x in fms]) for k in "ABC"}
            acc = fmt([x["accuracy"] for x in fms])
            bal = fmt([x.get("balanced_accuracy") for x in fms])
            md.append(f"| {s} | {len(fms)} | {acc} | {bal} | {site['A']} | {site['B']} | {site['C']} |")
            rows.append([rho, s, len(fms), acc, bal, site["A"], site["B"], site["C"]])
        md.append("")
    with open(os.path.join(OUT, "multisite_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # 4. trust/data-size mixing: w = (1 - lambda) * trust + lambda * n_k / n
    rows = [["dataset", "rho", "method", "n", "accuracy", "balanced_accuracy",
             "site_A", "site_B", "site_C"]]
    md += ["## 4. FedAgain with trust/data-size mixing (lambda)", "",
           "w_k = (1 - lambda) * trust_k + lambda * n_k / n; lambda = 0 is plain FedAgain.", ""]
    for ds, rho in sorted({(k[0], k[1]) for k in mix}):
        if ds == "KIDNEY_MULTISITE":
            ref = {s: multi.get((rho, s), []) for s in ("FedAvg", "FedMedian", "FedAgain")}
        elif rho == 0.3:
            ref = {s: canon.get((ds, s), []) for s in ("FedAvg", "FedMedian", "FedAgain")}
        else:
            ref = {}
        cells = [(s if s != "FedAgain" else "FedAgain (lambda=0)", fms) for s, fms in ref.items()]
        cells += [(f"FedAgain lambda={lam}", mix[(ds, rho, lam)])
                  for lam in sorted(l for d, r, l in mix if (d, r) == (ds, rho))]
        md += [f"### {DS_LABEL.get(ds, ds)}, rho = {rho}", "",
               "| method | n | accuracy | balanced acc. | site A | site B | site C |",
               "|---|---|---|---|---|---|---|"]
        for name, fms in cells:
            if not fms:
                continue
            site = {k: fmt([x.get("per_site_accuracy", {}).get(k) for x in fms]) for k in "ABC"}
            acc = fmt([x["accuracy"] for x in fms])
            bal = fmt([x.get("balanced_accuracy") for x in fms])
            md.append(f"| {name} | {len(fms)} | {acc} | {bal} | {site['A']} | {site['B']} | {site['C']} |")
            rows.append([ds, rho, name, len(fms), acc, bal, site["A"], site["B"], site["C"]])
        md.append("")
    with open(os.path.join(OUT, "mixing_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    with open(os.path.join(OUT, "clinical_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
