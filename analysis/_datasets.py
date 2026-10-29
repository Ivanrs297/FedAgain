"""
Shared dataset selection for the analysis scripts (Reviewer 1.7).

R1.7 asked to shift the emphasis away from the generic benchmarks and towards
the clinical evaluation. The main text therefore reports the three kidney-stone
datasets only, and CIFAR-10/MNIST move to the supplementary material.

Set ``FEDAGAIN_CLINICAL_ONLY=1`` to produce the main-text (clinical) outputs;
leave it unset to produce the full four-dataset version for the supplement.
The four-dataset figures get a ``_supp`` suffix and go to
``src-paper/images/supplementary/``, so they never overwrite the main-text
figures of the same name, e.g.::

    python analysis/analyze_baselines.py                 # -> baseline_comparison_supp.png
    FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_baselines.py   # main text

Set ``FEDAGAIN_CIFAR_ONLY=1`` to produce the CIFAR-10-only panels for the
benchmark grid in Appendix~C (single dataset per figure). These are written with
a ``_cifar`` suffix straight into ``src-paper/images/supplementary/`` so they do
not clobber the clinical main-text figures, e.g.::

    FEDAGAIN_CIFAR_ONLY=1 python analysis/analyze_baselines.py  # -> baseline_comparison_cifar.png
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRCPAPER_RESULTS = os.path.join(_ROOT, "jcr-fedagain", "src-paper", "images", "results")
SRCPAPER_SUPP = os.path.join(_ROOT, "jcr-fedagain", "src-paper", "images", "supplementary")

# Results tree to analyse. ``results`` holds the v1 runs; the specimen-disjoint
# v2 reruns live in ``results_v2`` (FEDAGAIN_RESULTS_DIR=results_v2). For any
# tree other than ``results`` the figures stay inside it (paper_figures/) and
# never overwrite the manuscript's images.
RESULTS_NAME = os.environ.get("FEDAGAIN_RESULTS_DIR", "results")
RESULTS = os.path.join(_ROOT, RESULTS_NAME)

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


def cifar_only():
    return os.environ.get("FEDAGAIN_CIFAR_ONLY", "") == "1"


def datasets():
    """Datasets to analyse, in the paper's canonical order."""
    if cifar_only():
        return list(BENCHMARK)
    return list(CLINICAL) if clinical_only() else CLINICAL + BENCHMARK


def fig_name(base):
    """Output basename for a figure: ``_cifar`` in CIFAR-only mode, ``_supp``
    in four-dataset mode, unchanged for the clinical main-text figures."""
    if cifar_only():
        return base + "_cifar"
    return base if clinical_only() else base + "_supp"


def srcpaper_dir():
    """Where the src-paper copy of the figure goes: the results folder for the
    clinical main-text figures, the supplementary folder otherwise."""
    if RESULTS_NAME != "results":
        d = os.path.join(RESULTS, "paper_figures")
        os.makedirs(d, exist_ok=True)
        return d
    return SRCPAPER_RESULTS if clinical_only() and not cifar_only() else SRCPAPER_SUPP


# --- canonical-run filter -------------------------------------------------
# The ledger accumulates several sweeps that all share the canonical
# configuration (non-IID label skew, rho=0.3, severity 3, 10 clients) and
# therefore leak into any selector that only matches on those fields:
#   pm-<mode>  label-flip poisoning sweep
#   _eps<v>    epsilon-sensitivity sweep
#   nm-<mode>  trust-normalization sweep   (keeps trust_mode=combined_mult!)
#   tm-<mode>  trust-formulation ablation  (non-multiplicative variants)
#   _mix<l>    trust/data-size mixing (round 2; keeps the default trust mode!)
#   _root*     RFLPA/BOBA server root-set variants (round 2: size, corrupted, one site)
# Every analysis must exclude all six.
VARIANT_TAGS = ("pm-", "_eps", "nm-", "_mix", "_root")


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
