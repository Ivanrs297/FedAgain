# Reproducing the FedAgain experiments (MEDIA-D-26-01276)

This folder is a **self-contained reproduction guide** for every experiment,
table and figure in the manuscript

> *FedAgain: Trust-Aware Federated Learning for Multi-Center Distribution Shift
> and Image Corruption in Endoscopic Kidney Stone Recognition.*

All commands are run **from the repository root** (the parent of this folder),
not from inside `to-reproduce/`. The code itself lives in the repository
(`main.py`, `config.py`, `src/`, `scripts/`, `analysis/`); this guide tells you
exactly which runner produces each result and which analysis script turns the
logged runs into each paper table/figure.

Nothing here needs a GPU cluster: every experiment is a short federated run
(ResNet-18, 10 clients, 10 rounds). The full paper sweep is ~780 runs and takes
on the order of a few days on a single modern GPU; each individual result can be
reproduced on its own in minutes to a couple of hours.

---

## 1. Environment

```bash
# from the repository root
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r to-reproduce/requirements.txt
```

The exact versions the paper was produced with are pinned in
`to-reproduce/requirements.txt`. A CUDA-capable GPU is used automatically if
present (`torch.cuda.is_available()`); otherwise everything runs on CPU, only
slower.

Sanity-check the installation with the unit tests (no data needed):

```bash
python tests/test_trust.py        # 6 tests: trust formulations & normalization
python tests/test_tracking.py     # 4 tests: resilient experiment tracker
```

---

## 2. Data availability

| Dataset | Role in the paper | Availability |
|---------|-------------------|--------------|
| **CIFAR-10 / MNIST** | Sanity-check benchmarks (Appendix C) | Downloaded automatically by `torchvision` into `data/` on first use. |
| **A — Michel Daudon** | Clinical (main text) | **Private** ex-vivo endoscopic data. Not redistributable; available from the dataset authors on reasonable request (see the paper's data-availability statement). |
| **B — Jonathan El-Beze** | Clinical (main text) | **Private**, same as above. |
| **C — MyStone** | Clinical (main text) | Public benchmark; obtain from its original publication. |

> **Reviewers:** the two private clinical datasets (A, B) cannot be shared in a
> public repository for patient-data reasons. **Every result on the public
> CIFAR-10/MNIST benchmarks reproduces end-to-end from this repo alone.** The
> clinical pipeline is byte-for-byte the same code — only the image folder
> differs — so the method is fully inspectable even without the private data.
> The complete per-run ledger of all ~780 runs we executed
> (`results/experiment_ledger.csv`) is included, so every number in the paper
> can be traced to its run even where the raw images are private.

Expected layout under `data/` (clinical folders only if you have the data):

```
data/
├── cifar-10-batches-py/                 # auto-downloaded
├── MNIST/                               # auto-downloaded
├── Michel Daudon (w256 1k v1)/          # private (Dataset A)
├── Jonathan El-Beze (w256 1k v1)/       # private (Dataset B)
└── MyStone/                             # public (Dataset C)
```

Dataset names used on the command line: `CIFAR10`, `MNIST`, `DAUDON_MIX`,
`JONATHAN_MIX`, `MYSTONE`.

---

## 3. How a single experiment works

One federated run = edit `config.py`, then `python main.py`. The runners in
`scripts/` automate this: each rewrites `config.py` per configuration and
subprocesses `main.py`. Results are written incrementally by the resilient
tracker (`src/tracking.py`) to:

```
results/
├── experiment_ledger.csv     # one row per finished run (config + metrics + status)
├── runs/<key>.json           # per-run manifest, updated every round
├── rounds/<key>.csv          # per-round accuracy/loss, written live
└── trust_scores/<key>.csv    # per-round E_k, D_k, T_k (FedAgain, when logging on)
```

Key properties:

- **Resumable.** Completed runs are skipped on re-launch; set `EXPERIMENT_FORCE=1`
  to force a rerun. A crash is marked `failed` and retried, not skipped.
- **Seeded.** `config.SEED` controls data partitioning, corrupted-client
  selection, per-client corruption and training RNG. Sweep seeds with
  `--seeds 42 43 44`; the seed is part of the run key.
- **Config hygiene.** The runners leave `config.py` modified. After a manual
  sweep, restore it with `git checkout config.py`. (`run_all_experiments.sh`
  does this for you at the end.)

> **Note on reproducibility.** Seeds are fixed, but cuDNN determinism is not
> pinned, so a single run can differ by up to a few accuracy points across
> machines/driver versions. Every headline number in the paper is therefore
> **averaged over three seeds (42/43/44)**; reproduce with all three, not one.

---

## 4. One command for everything

```bash
bash to-reproduce/run_all_experiments.sh
```

runs the full paper sweep in order (resumable — safe to re-run after an
interruption). To keep it alive across an SSH/terminal disconnect:

```bash
mkdir -p results/logs
setsid bash to-reproduce/run_all_experiments.sh </dev/null \
    >results/logs/reproduce_all.out 2>&1 &
```

Then generate all figures/tables (Section 6 below). If you only want a specific
result, run just its block from Section 5 instead.

---

## 5. Experiment-by-experiment (mapped to the paper)

Each block lists: **what it is → the run command → the analysis command → the
paper element it produces**. All runners are resumable and share the ledger, so
points swept once (e.g. ρ=0.3) are reused by later blocks.

### 5.1 Clean IID baseline · Fig. `kidney_stone_no_corruption` (§6.1)

```bash
python scripts/run_baselines.py \
  --strategies FedAvg FedProx FedMedian Bulyan FedAgain \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE \
  --distribution iid --corrupt-ratio 0.0 --severity 0 \
  --num-clients 10 --num-rounds 10 --seeds 42 43 44
```

### 5.2 Extended-baseline comparison at ρ=0.3 · Table `baseline_comparison`, Fig. `baseline_comparison` (§6.3) — **R1.5**

FedAgain vs. FedAvg/FedProx/FedMedian/Bulyan/**FedBN/SCAFFOLD/MOON**, non-IID
label skew, 30 % corrupted, severity 3.

```bash
bash scripts/run_baseline_sweep.sh          # 8 strat × 4 ds × 3 seeds = 96 runs
```

This same sweep also provides the per-round curves for convergence (§5.7) and the
wall-clock times for the overhead analysis (§5.9).

### 5.3 Recent robust baselines — FedDyn, RFLPA, BOBA · adds rows to Tables `baseline_comparison` + `degradation` (§6.3.1) — **R2.4**

```bash
bash scripts/run_r24_baselines.sh           # 3 strat × 4 ds × 5 ρ × 3 seeds = 180 runs
```

RFLPA/BOBA carve a clean server root set (`SERVER_ROOT_PER_CLASS`); FedAgain
needs none. BOBA requires `n−f > c−1`, so it degenerates to the mean on CIFAR-10
(10 clients / 10 classes) and is effective only on the clinical sets (c=6). RAHFL
is intentionally **not** run (paradigm mismatch — see the paper).

### 5.4 Corruption-degradation envelope · Table `degradation`, Fig. `degradation` (§6.4) — **R1.5 / R2.4**

Sweeps the corrupted-client fraction ρ ∈ {0.1, 0.3, 0.4, 0.5, 0.6}. ρ=0.3 is
already done by §5.2; add the rest:

```bash
bash scripts/run_envelope_sweep.sh          # ρ ∈ {0.1, 0.4}  → 192 runs
bash scripts/run_envelope_high.sh           # ρ ∈ {0.5, 0.6}  → 192 runs  (majority-corruption tail)
```

### 5.5 Label-flipping poisoning attack · Fig. `label_flip` (§6.5) — **R1.3 / R2.3**

A genuine adversarial data-poisoning attack (labels flipped, images untouched):

```bash
bash scripts/run_label_flip.sh              # FedAgain/FedMedian/FedAvg × 3 ds × ρ{0.1,0.3} × 3 seeds = 72 runs
```

Tagged `pm-label_flip` in the run key so it never collides with image-corruption
runs.

### 5.6 Trust-score ablation & formulation sensitivity · Tables `trust_ablation`, `trust_suppression`, Figs. `trust_distribution`, `trust_evolution`, `trust_distribution_rounds` (§6.6) — **R1.1b / R2.5**

Benchmark-error-only, divergence-only, multiplicative, additive, weighted-additive.
Per-round trust logging is enabled here (also feeds §5.8, rare-client fairness):

```bash
python scripts/run_trust_ablation.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE --distribution non-iid-label \
  --corrupt-ratio 0.3 --severity 3 --seeds 42 43 44 \
  --modes benchmark_only divergence_only combined_mult additive weighted_additive
```

**Alternative normalization strategies** · Table `normalization` (§6.6.1) — **R2.5**:

```bash
python scripts/run_normalization_ablation.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE \
  --corrupt-ratio 0.3 --severity 3 --seeds 42 43 44 \
  --variants softmax:0.5 softmax:2.0 rank   # sum_to_one is the default reference (reused)
bash scripts/run_rank_highrho.sh            # rank vs sum_to_one at ρ ∈ {0.5, 0.6}
```

### 5.7 Empirical convergence · Fig. `convergence` (§6.7) — **R1.1c / R1.2**

No new runs — reuses the per-round logs from §5.2/§5.3/§5.4. See analysis in §6.

### 5.8 Fairness to rare / hard clients · Fig. `rare_client_bias` (§6.8) — **R2.2**

Derived from the per-round trust logs written by the trust-ablation runs of §5.6
(`combined_mult`, `results/trust_scores/`). No separate sweep.

### 5.9 ε-sensitivity · Fig. `epsilon_sensitivity` (§6.7) — **R1.1c / R1.2**

```bash
python scripts/run_epsilon_sensitivity.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE \
  --epsilons 1e-8 1e-6 1e-4 1e-2 1 \
  --corrupt-ratio 0.3 --severity 3 --seeds 42 43 44   # 48 new runs (1e-6 reuses canonical)
```

### 5.10 Computational overhead · §6.10 — **R1.8**

No new runs — extracts wall-clock per strategy from the ledger (§5.2). See §6.

---

## 6. Generating the figures and tables

After the runs above are in the ledger, regenerate every main-text (clinical)
figure and table. The `FEDAGAIN_CLINICAL_ONLY=1` switch restricts the outputs to
the three clinical datasets (main text); unset it to get the four-dataset
supplementary versions (Appendix C).

```bash
# main text (clinical A/B/C)
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_baselines.py          # Table + Fig baseline_comparison
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_degradation_curve.py  # Table + Fig degradation
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_convergence.py        # Fig convergence
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_label_flip.py         # Fig label_flip
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_epsilon_sensitivity.py# Fig epsilon_sensitivity
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_rare_client_bias.py   # Fig rare_client_bias
python analysis/analyze_trust_ablation.py                              # trust tables + 3 trust figures
python analysis/analyze_overhead.py                                    # overhead numbers (§6.10)

# supplementary (four-dataset, incl. CIFAR-10): rerun the six above WITHOUT the
# env var, then save as *_supp.png (see jcr-fedagain/README.md).
```

Figures are written to `results/baselines/` and, when the manuscript tree is
present, copied into `jcr-fedagain/src-paper/images/`.

---

## 7. Paper element → command index

| Paper element | Section | Runner (§5) | Analysis (§6) |
|---------------|---------|-------------|----------------|
| Fig `kidney_stone_no_corruption` | 6.1 | 5.1 | — |
| Table + Fig `baseline_comparison` | 6.3 | 5.2 (+5.3) | `analyze_baselines.py` |
| Recent-baseline rows (FedDyn/RFLPA/BOBA) | 6.3.1 | 5.3 | `analyze_baselines.py`, `analyze_degradation_curve.py` |
| Table + Fig `degradation` | 6.4 | 5.2, 5.3, 5.4 | `analyze_degradation_curve.py` |
| Fig `label_flip` | 6.5 | 5.5 | `analyze_label_flip.py` |
| Table `trust_ablation` / `trust_suppression` | 6.6 | 5.6 | `analyze_trust_ablation.py` |
| Figs `trust_distribution` / `_rounds` / `trust_evolution` | 6.6 | 5.6 | `analyze_trust_ablation.py` |
| Table `normalization` | 6.6.1 | 5.6 | `analyze_trust_ablation.py` |
| Fig `convergence` | 6.7 | 5.2/5.3/5.4 (reused) | `analyze_convergence.py` |
| Fig `epsilon_sensitivity` | 6.7 | 5.9 | `analyze_epsilon_sensitivity.py` |
| Fig `rare_client_bias` | 6.8 | 5.6 (trust logs) | `analyze_rare_client_bias.py` |
| Overhead numbers | 6.10 | 5.2 (reused) | `analyze_overhead.py` |
| Proposition 1 (bounded weights) | 3.4.4 | — (analytic) | `tests/test_trust.py` checks the weighting |

---

## 8. Method entry points (for code review)

| Component | File |
|-----------|------|
| FedAgain strategy (trust-weighted aggregation) | `src/strategies/fedagain.py` |
| Pluggable trust formulations + normalizations | `src/strategies/trust.py` |
| Trust score `T_k = 1/(E_k·D_k+ε)`, normalization, aggregation | `src/strategies/trust.py`, `main.py` |
| Corrupted-client simulation (19 ImageNet-C corruptions, label-flip) | `src/dataset.py` |
| Server root carving (for RFLPA/BOBA only) | `src/dataset.py` (`carve_server_root`) |
| Baselines (FedProx/FedMedian/Bulyan/FedBN/Scaffold/MOON/FedDyn/RFLPA/BOBA) | `main.py`, `src/client.py` |
| Resilient experiment tracker | `src/tracking.py` |

Relevant environment variables (all optional; defaults reproduce the paper):
`FEDAGAIN_EPSILON`, `FEDAGAIN_TRUST_MODE`, `FEDAGAIN_NORM_MODE`,
`FEDAGAIN_NORM_TEMPERATURE`, `FEDAGAIN_ALPHA`, `FEDAGAIN_BETA`,
`FEDAGAIN_LOG_TRUST`, `SERVER_ROOT_PER_CLASS`, `FEDDYN_ALPHA`, `BOBA_NUM_BYZ`,
`RFLPA_SERVER_EPOCHS`, `EXPERIMENT_FORCE`.
