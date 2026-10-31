#!/usr/bin/env python3
"""
Can the FedAgain trust weights tell the coordinating server WHICH clients are
corrupted? (Round 2: clinical value of the per-client trust signal.)

For every FedAgain run with the deployed rule (combined_mult, sum-to-one,
default eps, no mixing) that logged per-round weights in
<results>/trust_scores/by_run/, the corrupted clients are reconstructed from
the run seed exactly as in src/dataset.apply_corruptions_to_clients
(np.random.seed(seed); np.random.choice(K, int(K*rho), replace=False)), and
the aggregation weight is scored as a detector of corruption (lower weight =
more suspicious):
  * AUROC of -weight at the final round and averaged over rounds 2..R
    (round 1 carries no evidence yet: every client sits at 1/K);
  * fraction of the k corrupted clients among the k lowest-weight clients at
    the final round (k = number of corrupted clients).

Reads logs only. Outputs <results>/analysis/corruption_detection.{csv,md}.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _datasets import RESULTS, CLINICAL  # noqa: E402

NUM_CLIENTS = 10
RUN_RE = re.compile(r"^FedAgain_(?P<ds>[A-Z_]+?)_non-iid-label_k10_r10_e1_cr(?P<rho>[0-9.]+)_sev3"
                    r"(?P<lf>_pm-label_flip)?_tm-combined_mult_s(?P<seed>\d+)\.csv$")


def corrupted_ids(seed, rho):
    np.random.seed(seed)
    return set(np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * rho), replace=False).tolist())


def auroc(suspicion, is_corrupt):
    pos = [s for s, c in zip(suspicion, is_corrupt) if c]
    neg = [s for s, c in zip(suspicion, is_corrupt) if not c]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "trust_scores", "by_run", "*.csv"))):
        m = RUN_RE.match(os.path.basename(path))
        if not m or m["ds"] not in CLINICAL:
            continue
        rho, seed = float(m["rho"]), int(m["seed"])
        corr = corrupted_ids(seed, rho)
        weights = defaultdict(dict)
        with open(path) as f:
            for r in csv.DictReader(f):
                weights[int(r["round"])][int(r["client"])] = float(r.get("weight") or r["trust_normalized"])
        last = max(weights)
        lab = [c in corr for c in range(NUM_CLIENTS)]
        final = weights[last]
        lowest = set(sorted(range(NUM_CLIENTS), key=lambda c: final[c])[:len(corr)])
        rows.append({
            "attack": "label_flip" if m["lf"] else "image",
            "dataset": m["ds"], "rho": rho, "seed": seed,
            "auroc_final": auroc([-final[c] for c in range(NUM_CLIENTS)], lab),
            "auroc_mean": st.mean(auroc([-weights[t][c] for c in range(NUM_CLIENTS)], lab)
                                  for t in weights if t >= 2),
            "hit_rate": len(lowest & corr) / len(corr),
        })
    if not rows:
        sys.exit(f"no FedAgain per-run trust logs under {RESULTS}/trust_scores/by_run")

    out = os.path.join(RESULTS, "analysis")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "corruption_detection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["attack"], r["dataset"], r["rho"])].append(r)
    lines = ["# Corrupted-client detection by the FedAgain trust weights", "",
             "| attack | dataset | rho | runs | AUROC final | AUROC mean r2-10 | corrupted among k lowest | perfect runs |",
             "|---|---|---|---|---|---|---|---|"]
    for (atk, ds, rho), g in sorted(groups.items()):
        lines.append(f"| {atk} | {ds} | {rho} | {len(g)} | {st.mean(r['auroc_final'] for r in g):.3f} | "
                     f"{st.mean(r['auroc_mean'] for r in g):.3f} | {st.mean(r['hit_rate'] for r in g):.2f} | "
                     f"{sum(r['auroc_final'] == 1.0 for r in g)}/{len(g)} |")
    for atk in ("image", "label_flip"):
        g = [r for r in rows if r["attack"] == atk]
        if g:
            lines.append(f"\n{atk}: {sum(r['auroc_final'] == 1.0 for r in g)}/{len(g)} runs with a perfect "
                         f"final-round separation; mean final AUROC {st.mean(r['auroc_final'] for r in g):.3f}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "corruption_detection.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
