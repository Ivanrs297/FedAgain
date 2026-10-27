"""
Build the job list for the second-revision reruns (data version v2).

Phase order (most important first, so a partial run is already useful):
  P1  canonical table      every strategy, rho=0.3, image corruption   (99)
  P2  multisite canonical  KIDNEY_MULTISITE, natural sites, rho=0.3     (33)
  P3  degradation envelope rho in {0.1, 0.4, 0.5, 0.6}                  (396)
  P4  label-flip attack                                                 (54)
  P5  multisite envelope   KIDNEY_MULTISITE, rho in {0.0, 0.6}          (66)
  P6  trust ablation       FedAgain trust modes
  P7  normalization        FedAgain sum_to_one / softmax / rank
  P8  epsilon sweep

P1, P3, P4, P6-P8 replicate EXACTLY the 684 clinical v1 runs: each job is
rebuilt from its v1 manifest (results/runs/*.json) and its run key is checked
against the v1 key, so the v2 tables are comparable one-to-one with v1.

Writes results_v2/queue.json. Deterministic; safe to re-run.
"""

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.tracking import make_run_key  # noqa: E402

V1_RUNS = os.path.join(ROOT, "results", "runs")
OUT_DIR = os.path.join(ROOT, "results_v2")
CLINICAL = ("DAUDON_MIX", "JONATHAN_MIX", "MYSTONE")
STRATEGIES = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedBN", "Scaffold",
              "MOON", "FedDyn", "RFLPA", "BOBA", "FedAgain"]
SEEDS = (42, 43, 44)
MULTISITE_P2 = (0.3,)
MULTISITE_P5 = (0.0, 0.6)


def tracker_config(o):
    """Mirror of main.py's tracker_config for a set of config overrides."""
    fa = o["STRATEGY"] == "FedAgain"
    return {
        "strategy": o["STRATEGY"], "dataset": o["DATASET"],
        "data_distribution": o["DATA_DISTRIBUTION"], "num_clients": o["NUM_CLIENTS"],
        "num_rounds": o["NUM_ROUNDS"], "local_epochs": o["LOCAL_EPOCHS"],
        "corrupt_clients": o["CORRUPT_CLIENTS"], "corrupt_ratio": o["CORRUPT_CLIENT_RATIO"],
        "corruption_severity": o["CORRUPTION_SEVERITY"], "poison_mode": o["POISON_MODE"],
        "trust_mode": o["FEDAGAIN_TRUST_MODE"] if fa else "",
        "trust_alpha": o["FEDAGAIN_ALPHA"] if fa else "",
        "trust_beta": o["FEDAGAIN_BETA"] if fa else "",
        "epsilon": o["FEDAGAIN_EPSILON"] if fa else "",
        "norm_mode": o["FEDAGAIN_NORM_MODE"] if fa else "",
        "norm_temperature": o["FEDAGAIN_NORM_TEMPERATURE"] if fa else "",
        "seed": o["SEED"],
    }


def overrides(strategy, dataset, distribution, rho, seed, poison="image",
              trust_mode="combined_mult", alpha=1.0, beta=1.0, eps=1e-6,
              norm="sum_to_one", tau=1.0, clients=10, rounds=10, epochs=1, severity=3):
    """Complete config override set: every key a run depends on is explicit, so
    nothing leaks from whatever state config.py is in."""
    return {
        "STRATEGY": strategy, "DATASET": dataset, "DATA_DISTRIBUTION": distribution,
        "NUM_CLIENTS": clients, "NUM_ROUNDS": rounds, "LOCAL_EPOCHS": epochs,
        "CORRUPT_CLIENTS": rho > 0, "CORRUPT_CLIENT_RATIO": rho,
        "CORRUPTION_SEVERITY": severity, "POISON_MODE": poison, "LABEL_FLIP_RATIO": 1.0,
        "FEDAGAIN_TRUST_MODE": trust_mode, "FEDAGAIN_ALPHA": alpha, "FEDAGAIN_BETA": beta,
        "FEDAGAIN_EPSILON": eps, "FEDAGAIN_NORM_MODE": norm,
        "FEDAGAIN_NORM_TEMPERATURE": tau,
        "FEDAGAIN_LOG_TRUST": strategy == "FedAgain",
        "SEED": seed,
    }


def from_v1_manifest(m):
    c, key = m["config"], m["run_key"]
    eps = re.search(r"_eps([0-9.e+-]+?)_(?:nm-|s\d+$)", key)
    nm = re.search(r"_nm-([a-z_]+?)(?:_tau|_s\d+$)", key)
    tau = re.search(r"_tau([0-9.]+)_", key)
    pm = re.search(r"_pm-([a-z_]+?)_(?:tm-|s\d+$)", key)
    fa = c["strategy"] == "FedAgain"
    o = overrides(
        c["strategy"], c["dataset"], c["data_distribution"],
        float(c["corrupt_ratio"]) if c.get("corrupt_clients") else 0.0, int(c["seed"]),
        poison=pm.group(1) if pm else c.get("poison_mode", "image") or "image",
        trust_mode=(c.get("trust_mode") or "combined_mult") if fa else "combined_mult",
        alpha=float(c.get("trust_alpha") or 1.0) if fa else 1.0,
        beta=float(c.get("trust_beta") or 1.0) if fa else 1.0,
        eps=float(eps.group(1)) if eps else 1e-6,
        norm=nm.group(1) if nm else "sum_to_one",
        tau=float(tau.group(1)) if tau else 1.0,
        clients=int(c["num_clients"]), rounds=int(c["num_rounds"]),
        epochs=int(c["local_epochs"]), severity=int(c["corruption_severity"]))
    return o


def phase_of(o):
    if o["DATASET"] == "KIDNEY_MULTISITE":
        return "P2" if o["CORRUPT_CLIENT_RATIO"] in MULTISITE_P2 else "P5"
    if o["POISON_MODE"] != "image":
        return "P4"
    if o["FEDAGAIN_EPSILON"] != 1e-6:
        return "P8"
    if o["FEDAGAIN_NORM_MODE"] != "sum_to_one":
        return "P7"
    if o["STRATEGY"] == "FedAgain" and o["FEDAGAIN_TRUST_MODE"] != "combined_mult":
        return "P6"
    return "P1" if o["CORRUPT_CLIENT_RATIO"] == 0.3 else "P3"


def main():
    jobs, mismatches = {}, []
    for f in sorted(glob.glob(os.path.join(V1_RUNS, "*.json"))):
        m = json.load(open(f))
        if m.get("status") != "completed" or m["config"]["dataset"] not in CLINICAL:
            continue
        o = from_v1_manifest(m)
        key = make_run_key(tracker_config(o))
        if key != m["run_key"]:
            mismatches.append((m["run_key"], key))
            continue
        jobs[key] = o
    if mismatches:
        for a, b in mismatches[:10]:
            print(f"KEY MISMATCH v1={a}\n             v2={b}")
        sys.exit(f"{len(mismatches)} v1 runs could not be reconstructed exactly; aborting")

    for rho in MULTISITE_P2 + MULTISITE_P5:
        for seed in SEEDS:
            for s in STRATEGIES:
                o = overrides(s, "KIDNEY_MULTISITE", "natural", rho, seed)
                jobs[make_run_key(tracker_config(o))] = o

    order = {p: i for i, p in enumerate(["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"])}
    queue = [{"key": k, "phase": phase_of(o), "overrides": o} for k, o in jobs.items()]
    # Seed-major inside a phase: a partial phase still covers every
    # (dataset, strategy) cell with at least one seed.
    queue.sort(key=lambda j: (order[j["phase"]], j["overrides"]["SEED"],
                              j["overrides"]["CORRUPT_CLIENT_RATIO"],
                              j["overrides"]["DATASET"], j["overrides"]["STRATEGY"], j["key"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "queue.json"), "w") as f:
        json.dump(queue, f, indent=1)
    counts = {}
    for j in queue:
        counts[j["phase"]] = counts.get(j["phase"], 0) + 1
    print(f"{len(queue)} jobs -> results_v2/queue.json  {counts}")


if __name__ == "__main__":
    main()
