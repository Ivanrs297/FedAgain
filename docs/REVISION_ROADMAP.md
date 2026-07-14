# FedAgain — Full Revision Roadmap (MEDIA-D-26-01276, deadline 13 Sep 2026)

Ordered plan for the remaining reviewer items. Status legend: ✅ done · 🔄 in
progress · ⬜ pending. Colours = which reviewer (green R1 / violet R2 / blue both)
as used in `jcr-fedagain/src-paper/FedAgain_classification.tex`.

Already done: R2.1 novelty (🟣), R1.4 clinical split (🟢), R1.6 condensation (🟢),
R1.3/R2.3 softened claims (🔵), R1.1b/R2.5 trust ablation subsection (🔵),
framing + general fixes (🔵). Baselines FedBN/Scaffold/MOON implemented.

---

## Phase 1 — Finish the baseline comparison (this week)

### 1.1 🔄 Baseline table  [R1.5] — *sweep running (~8h)*
- **When sweep done:** `git checkout config.py`; verify 96 runs in
  `results/experiment_ledger.csv`.
- **New:** `analysis/analyze_baselines.py` → per-dataset table of {acc, F1, std}
  for FedAgain vs FedAvg/FedProx/FedMedian/Bulyan/FedBN/Scaffold/MOON; bold best
  per dataset; a grouped-bar figure. Reuse patterns from `analyze_trust_ablation.py`.
- **Paper:** add a `\begin{table*}` (blue) in the results section + 1–2 paragraphs;
  copy any figure into `src-paper/`. Honest framing: FedAgain best/competitive on
  the clinical datasets, especially under corruption + heterogeneity.
- Effort: ~half day (mostly analysis + writing).

### 1.2 ⬜ Rare/hard-client bias  [R2.2] 🟣 — *no new runs needed*
- Reuse `results/trust_scores/`: tag a client that holds a rare/minority class,
  track its trust weight over rounds vs. a corrupted client and typical clients.
- May need one tagged config (one client = rare class) with `FEDAGAIN_LOG_TRUST`.
- **Paper:** short subsection + figure showing FedAgain does not unfairly suppress
  legitimate rare-distribution clients (or, if it does, discuss the fairness
  trade-off honestly).
- Effort: ~half day.

---

## Phase 2 — Quick writing items (parallel, low risk)

### 2.1 ⬜ Novelty vs. other trust-aware/reputation FL  [R1.1a] 🟢
- Add a short related-work paragraph/table contrasting FedAgain with existing
  trust/reputation FL (beyond the previous FedAgain, which is R2.1).

### 2.2 ⬜ Code/data release + overhead  [R1.8] 🟢
- State the code will be released (repo already exists). Add the trust-weighting
  **overhead**: only a validation loss + one ℓ2 norm per client per round —
  measure wall-clock delta FedAgain vs FedAvg from the ledger and report it.

### 2.3 ⬜ Reduce MNIST/CIFAR emphasis  [R1.7] 🟢
- Reframe MNIST/CIFAR as sanity checks; move clinical results forward; trim.

*Effort for 2.1–2.3: ~1 day total (writing).*

---

## Phase 3 — Recent robust baselines  [R2.4 / R1.5] 🟣🟢 — *biggest chunk*

### 3.1 ⬜ RFLPA, BOBA, RAHFL (+ optional FedDyn)
- Implement each as a strategy (pattern: `src/strategies/` + a `train_*` in
  `src/client.py` if it changes local training, + dispatch/state in `main.py`,
  exactly like Scaffold/MOON). Verify each on MNIST (no divergence) before sweeping.
- **Gotcha (learned):** control-variate / drift methods need plain SGD; verify
  convergence on a tiny run first.
- Re-run `scripts/run_baselines.py --strategies RFLPA BOBA RAHFL ...` and fold
  into the Phase-1 table.
- Effort: ~1–2 days each (implementation + verification + runs). Highest risk —
  budget time; a wrong baseline is worse than none.

---

## Phase 4 — Adversarial attack  [R1.3 / R2.3] 🔵

### 4.1 ⬜ Label-flipping (easiest) — re-strengthen robustness claims
- Add a `label_flip` poisoning mode in `src/dataset.py` (flip a fraction of labels
  for corrupted clients), selectable via config; distinct from image corruption.
- Run FedAgain vs FedAvg/FedMedian under label-flip at ρ∈{0.1,0.3}.
- **Paper:** re-tighten the softened claims (partially) with a real attack result.
- Effort: ~1 day.

---

## Phase 5 — Theory  [R1.1c / R1.2] 🟢 (Tier 3)

### 5.1 ⬜ Convergence / sensitivity analysis
- No full theorem required: a convergence-under-non-IID argument or a formal
  sensitivity analysis of the trust weight (ε, α/β). Convergence curves already
  exist (`analysis/.../convergence.png`); add an ε/(α,β) sensitivity sweep +
  discussion.
- Effort: ~1–2 days.

---

## Phase 6 — Wrap-up (before submission)

- ⬜ **Response-to-reviewers letter** (point-by-point, referencing the coloured
  changes) — essential deliverable for a journal revision.
- ⬜ Full Overleaf compile + proofread; verify all figures synced
  (`trust_distribution.png`, `trust_evolution.png`, baseline figure, +new).
- ⬜ Decide whether to keep the reviewer colours or produce a clean final.
- ⬜ Optional: decide on committing the paper to git (currently untracked;
  under-review manuscript).

---

## Suggested sequencing

1. **Now → tomorrow:** finish sweep → 1.1 baseline table → 1.2 rare-class bias.
2. **Then:** Phase 2 writing items (quick wins) while deciding Phase-3 scope.
3. **Main effort:** Phase 3 (R2.4 baselines) — start with one method, verify, sweep.
4. **Then:** Phase 4 attack → Phase 5 theory.
5. **Last:** Phase 6 response letter + final compile.

Deadline 13 Sep 2026 leaves comfortable time; Phase 3 (baselines) is the long pole.
