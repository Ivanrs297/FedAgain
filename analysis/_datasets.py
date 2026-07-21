"""
Shared dataset selection for the analysis scripts (Reviewer 1.7).

R1.7 asked to shift the emphasis away from the generic benchmarks and towards
the clinical evaluation. The main text therefore reports the three kidney-stone
datasets only, and CIFAR-10/MNIST move to the supplementary material.

Set ``FEDAGAIN_CLINICAL_ONLY=1`` to produce the main-text (clinical) outputs;
leave it unset to produce the full four-dataset version for the supplement.
Both write to the same paths, so the caller is responsible for renaming, e.g.::

    python analysis/analyze_baselines.py                 # 4 datasets  -> *_supp
    FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_baselines.py   # main text
"""

import os

CLINICAL = ["DAUDON_MIX", "JONATHAN_MIX", "MYSTONE"]
BENCHMARK = ["CIFAR10"]

DS_LABEL = {
    "DAUDON_MIX": "A (Daudon)",
    "JONATHAN_MIX": "B (Jonathan)",
    "MYSTONE": "C (MyStone)",
    "CIFAR10": "CIFAR-10",
}


def clinical_only():
    return os.environ.get("FEDAGAIN_CLINICAL_ONLY", "") == "1"


def datasets():
    """Datasets to analyse, in the paper's canonical order."""
    return list(CLINICAL) if clinical_only() else CLINICAL + BENCHMARK


# --- canonical-run filter -------------------------------------------------
# The ledger accumulates several sweeps that all share the canonical
# configuration (non-IID label skew, rho=0.3, severity 3, 10 clients) and
# therefore leak into any selector that only matches on those fields:
#   pm-<mode>  label-flip poisoning sweep
#   _eps<v>    epsilon-sensitivity sweep
#   nm-<mode>  trust-normalization sweep   (keeps trust_mode=combined_mult!)
#   tm-<mode>  trust-formulation ablation  (non-multiplicative variants)
# Every analysis that reports "FedAgain" must exclude all four.
VARIANT_TAGS = ("pm-", "_eps", "nm-")


def is_canonical(row):
    """True if a ledger row belongs to the canonical image-corruption sweep."""
    if row.get("status") != "completed":
        return False
    if any(t in row["run_key"] for t in VARIANT_TAGS):
        return False
    if row.get("strategy") == "FedAgain" and row.get("trust_mode") not in ("combined_mult", "", None):
        return False
    return True


def is_canonical_key(run_key, strategy=None):
    """Same test for filename-based selectors (results/rounds/<key>.csv)."""
    if any(t in run_key for t in VARIANT_TAGS):
        return False
    if run_key.startswith("FedAgain") and "tm-combined_mult" not in run_key:
        return False
    return True
