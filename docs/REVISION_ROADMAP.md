# FedAgain — Full Revision Roadmap (MEDIA-D-26-01276, deadline 13 Sep 2026)

Ordered plan for the remaining reviewer items. Status legend: ✅ done · 🔄 in
progress · ⬜ pending. Colours = which reviewer (green R1 / violet R2 / blue both)
as used in `jcr-fedagain/src-paper/FedAgain_classification.tex`.

Already done: R2.1 novelty (🟣), R1.4 clinical split (🟢), R1.6 condensation (🟢),
R1.3/R2.3 softened claims (🔵), R1.1b/R2.5 trust ablation subsection (🔵),
framing + general fixes (🔵). Baselines FedBN/Scaffold/MOON implemented.

---

## Phase 1 — Finish the baseline comparison (this week)

### 1.1 ✅ Baseline table  [R1.5] — *DONE (sweep 96/96, committed b3114b2)*
- Sweep finished 2026-07-14 20:39 (96/96). `analysis/analyze_baselines.py` →
  `results/baselines/{baseline_table.tex, baseline_summary.csv, baseline_comparison.png}`.
- **Paper:** new subsection `subsec:extended_baselines` (blue) with
  `tab:baseline_comparison` + `fig:baseline_comparison`; FedBN/SCAFFOLD/MOON added
  to the compared-strategies list; 3 refs added to `biblio.bib`.
- **Result (honest):** FedMedian best in mean (74.9%, rank 1.25); FedAgain 2nd
  (73.7%, rank 2.0) — top-2 on every dataset, best on CIFAR-10, beats
  FedBN/SCAFFOLD/MOON on all clinical sets. Framing: no need to know #corrupted,
  retains all clients, interpretable trust weights.
- **Overleaf to sync:** `baseline_comparison.png`, `biblio.bib`, the `.tex`.

### 1.2 ✅ Rare/hard-client bias  [R2.2] 🟣 — *DONE (committed a3f435d)*
- `analysis/analyze_rare_client_bias.py` reconstructs corrupted/benign status +
  intrinsic difficulty (benchmark error) deterministically (seed 44, ρ=0.3) and
  joins with `results/trust_scores/` logs. No new runs (CPU-only reconstruction).
- **Finding (honest):** corrupted → trust $0.045$ (<½ of uniform $1/K$); legitimate
  hard/rare clients → $0.105$ (≈uniform, $2.3\times$ above corrupted) on every
  dataset. Mild difficulty gradient (hard $0.105$ vs easy $0.137$) disclosed; not
  worsened by the multiplicative rule. FedAgain suppresses corruption, not rarity.
- **Paper:** subsection `subsec:rare_client_bias` (violet R2) + `fig:rare_client_bias`.
- **Overleaf to sync:** `rare_client_bias.png`.

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
