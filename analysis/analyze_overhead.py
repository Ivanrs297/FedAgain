#!/usr/bin/env python3
"""
Computational overhead of the trust weighting (Reviewer 1.8).

Extracts per-run wall-clock time from results/experiment_ledger.csv under
matched conditions (non-IID label skew, rho=0.3, severity 3, ResNet-18, 10
clients, 10 rounds) and reports the overhead of each strategy relative to
FedAvg. FedAgain's cost is dominated by the per-client benchmark inference pass;
the l2 divergence is negligible. No communication overhead is added.
"""
import csv
import statistics as st
from datetime import datetime
from collections import defaultdict
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _datasets import RESULTS, datasets as _ds_sel, is_canonical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(RESULTS, "experiment_ledger.csv")
DATASETS = set(_ds_sel())
STRATS = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedBN", "Scaffold", "MOON", "FedAgain"]


def dur(r):
    try:
        return (datetime.fromisoformat(r["timestamp_end"])
                - datetime.fromisoformat(r["timestamp_start"])).total_seconds()
    except Exception:
        return None


def main():
    d = defaultdict(list)
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            # Canonical runs only: the label-flip, epsilon, normalization and
            # trust-ablation sweeps share rho=0.3 and would otherwise inflate n.
            if (is_canonical(r) and r["dataset"] in DATASETS
                    and r["corrupt_ratio"] == "0.3" and r["corruption_severity"] == "3"
                    and r["data_distribution"] == "non-iid-label"
                    and r["rounds_completed"] == "10"):
                s = dur(r)
                if s:
                    d[r["strategy"]].append(s)
    if "FedAvg" not in d:
        print("No FedAvg runs found.")
        return
    base = st.mean(d["FedAvg"])
    print("Wall-clock per run (s), rho=0.3 sev=3, 10 rounds, ResNet-18, 10 clients:")
    print(f"{'strategy':<12}{'mean(s)':>9}{'n':>4}{'vs FedAvg':>12}{'per-round':>11}")
    for s in STRATS:
        if d.get(s):
            m = st.mean(d[s])
            print(f"{s:<12}{m:>9.1f}{len(d[s]):>4}{(m/base-1)*100:>+11.1f}%{(m-base)/10:>+10.2f}s")


if __name__ == "__main__":
    main()
