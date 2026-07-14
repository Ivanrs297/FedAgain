# FedAgain — Experiment Plan (MEDIA-D-26-01276 revision)

Maps each reviewer comment to a concrete, runnable experiment. All runs use the
resilient tracker (`src/tracking.py`): results are written **per round** to
`results/rounds/<run_key>.csv`, a manifest to `results/runs/<run_key>.json`, one
row per finished run to `results/experiment_ledger.csv`, and an event stream to
`results/ledger.jsonl`. **Completed runs are skipped automatically** (set
`EXPERIMENT_FORCE=1` to rerun), so a crashed sweep can just be relaunched.

## Base protocol (unless a point overrides it)
- Model: ResNet-18, `USE_PRETRAINED=True`; batch 32; lr 0.01; 1 local epoch.
- Clients: 10; Rounds: 10.
- Datasets (7 kidney sub-datasets + 2 benchmarks):
  `DAUDON_MIX/SEC/SUR`, `JONATHAN_MIX/SEC/SUR`, `MYSTONE`, `CIFAR10`, `MNIST`.
- Distributions: `iid`, `non-iid-label`, `non-iid-dirichlet` (α=0.5).
- Corruption ratio ρ ∈ {0, 0.1, 0.3, 0.4}; severity s ∈ {1..5}.
  Scenarios: **Light** (ρ=0.1, s=3), **Moderate** (ρ=0.3, s=4), **Heavy** (ρ=0.4, s=5); default = ρ=0.3, s=3.
- **Seeds: run each cell with ≥3 seeds** (see "Seeds & variance" below) — required
  for the variance/stability claims in the paper.

---

## 🔴 Tier 1 — required

### #2 Trust-score ablation  *(no new code — ready now)*
**Goal:** show the gain is not from either signal alone. Compare `benchmark_only`
(E only), `divergence_only` (D only), `combined_mult` (E·D). Also report the
per-round trust-score distribution (auto-logged to `results/trust_scores/`).

```bash
python scripts/run_trust_ablation.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --num-clients 10 --num-rounds 10 \
  --modes benchmark_only divergence_only combined_mult
```
**Outputs → analysis:** table of {acc, F1, variance} per mode per dataset; box/violin
of trust weights per round per mode. **Cells:** 3 modes × 4 datasets × 3 seeds = 36.

### #3 Multiplicative vs. additive  *(no new code — ready now)*
**Goal:** show multiplicative suppresses noisy clients more and has lower variance.
Compare `combined_mult`, `additive`, `weighted_additive` (α,β sweep).

```bash
# multiplicative vs additive
python scripts/run_trust_ablation.py \
  --datasets DAUDON_MIX JONATHAN_MIX MYSTONE CIFAR10 \
  --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
  --modes combined_mult additive
# weighted additive sweep (run per (alpha,beta))
for ab in "2 1" "1 2" "0.5 1"; do set -- $ab; \
  python scripts/run_trust_ablation.py --datasets DAUDON_MIX CIFAR10 \
    --distribution non-iid-label --corrupt-ratio 0.3 --severity 3 \
    --modes weighted_additive --alpha $1 --beta $2; done
```
**Outputs → analysis:** for corrupted vs benign clients, mean trust-weight gap per
formulation (multiplicative should give the largest gap / lowest benign variance).
**Cells:** ~ (2 + 3 α/β) × 4 datasets × 3 seeds ≈ 60.

---

## 🟡 Tier 2 — strongly recommended

### #6 Baselines: FedBN, Scaffold, MOON  *(needs implementation first)*
**Code:** add `src/strategies/{fedbn,scaffold,moon}.py` and register them in
`main.py`'s aggregation dispatch (alongside FedAvg/FedProx/FedMedian/Bulyan).
Priority order: FedBN → Scaffold → MOON (then RFLPA/BOBA/RAHFL if time).
**Run (per new strategy):**
```bash
python scripts/run_experiments.py --strategies FedBN Scaffold MOON \
  --dataset CIFAR10   # repeat per dataset via config, or extend the runner
```
**Matrix:** {FedAvg, FedProx, FedMedian, Bulyan, FedAgain, +new} × 7 kidney +
CIFAR10 × {non-iid-label, dirichlet} × Moderate corruption × 3 seeds.

### #7 Bias against rare-distribution clients  *(mostly analysis + small knob)*
**Goal:** check FedAgain does not unfairly down-weight a *legitimate* client that
holds a rare/minority class (high E_k for benign reasons).
**Setup:** `non-iid-label` where one designated client holds a rare class only;
enable `FEDAGAIN_LOG_TRUST`. Compare that client's trust trajectory vs. a truly
corrupted client and vs. benign-majority clients (read `results/trust_scores/`).
**Deliverable:** trust-weight-over-rounds plot for {rare-legit, corrupted, typical}
clients; fairness discussion. No full retrain needed beyond the tagged runs.

---

## 🟢 Tier 3 — differentiators

### #9 One real adversarial attack — label flipping  *(needs implementation)*
**Code:** add a `label_flip` client-poisoning mode in `src/dataset.py`
(flip a fraction of labels for corrupted clients) selectable via config, distinct
from image corruption. **Run:** FedAgain vs FedAvg/FedMedian under label-flip at
ρ∈{0.1,0.3}. Lets the robustness claims be (partially) re-strengthened.

### #10 Convergence / sensitivity analysis  *(analysis — per-round logs now exist)*
- **Convergence curves:** accuracy-vs-round for FedAgain vs baselines under
  non-IID + Moderate corruption, straight from `results/rounds/*.csv`.
- **Sensitivity:** sweep ε ∈ {1e-3,1e-6,1e-9} and the (α,β) grid from #3; plot
  final accuracy / variance surface.
```bash
# example epsilon sensitivity (requires exposing FEDAGAIN_EPSILON in the runner
# or setting it in config before each run)
```

---

## Seeds & variance (do this before the sweeps)
The paper reports variance/stability claims, but corruption and partitioning
currently use a **hardcoded seed 42** (`src/dataset.py`). To support ≥3 seeds:
add `SEED` to `config.py`, thread it into the `np.random.seed(...)` calls in
`apply_corruptions_to_clients` and the partitioners, and include it in
`tracker_config["seed"]` (already a key field). Then run each cell with
`SEED ∈ {42, 43, 44}`.

## Suggested execution order
1. **Seeds knob** (small code change) — unblocks all variance claims.
2. **#2 + #3** (Tier 1, ready now) — highest priority, no new strategy code.
3. **#6 baselines** (implement FedBN → Scaffold → MOON, then sweep).
4. **#7** analysis on the tagged runs.
5. **#10** convergence/sensitivity from the per-round logs.
6. **#9** label-flip attack if time permits.

## Rough scale
Tier 1 alone ≈ 100 runs (~min each on GPU). Full plan (with baselines × datasets ×
distributions × seeds) is in the low thousands of runs — use the skip-on-complete
resume and commit `results/experiment_ledger.csv` + `results/runs/` periodically.
