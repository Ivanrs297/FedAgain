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

### 1.3 ✅ Corruption-degradation curve, 5-point ρ 0.1→0.6  [R1.5 / R2.4] 🔵 — *DONE (envelope sweep 480/480)*
- `scripts/run_envelope_sweep.sh` (ρ∈{0.1,0.3,0.4}) + `scripts/run_envelope_high.sh`
  (ρ∈{0.5,0.6}) → 8 strategies × 4 datasets × 5 ρ × 3 seeds = 480 runs.
  `analysis/analyze_degradation_curve.py` (RHOS=[0.1,0.3,0.4,0.5,0.6]) →
  `results/baselines/{degradation_curve.png, degradation_summary.csv}`.
- **Result (honest, two-regime story):** With K=10, ρ≥0.5 is a *corrupted majority*
  — the coordinate-wise-median breakdown point.
  - *Minority (ρ≤0.4):* FedMedian (drop +3.3) and FedAgain (+4.7) are the two most
    stable and statistically tied (within 1σ on 11/12 dataset×ρ points).
  - *Majority (ρ≥0.5):* FedMedian **collapses to ~20–28% (≈chance)** on all 4
    datasets at ρ=0.6; FedAgain holds 50–80% (+30 to +60 pt margin).
  - *Full range:* FedAgain is #1 on both drop (+9.0 vs FedMedian +53.2) and
    envelope-avg acc (71.8 vs 61.4). No #corrupt prior; discards no updates.
- **Paper:** `subsec:degradation` (blue) — intro, `tab:degradation` (now 4-col
  `table*`, two drop columns + envelope), `fig:degradation` (vertical "corrupted
  majority" guide at ρ=0.5), findings paragraph — all rewritten for 5 points.
- **Overleaf to sync:** `degradation_curve.png`, the `.tex`.

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

### 2.1 ✅ Novelty vs. other trust-aware/reputation FL  [R1.1a] 🟢 — *DONE*
- Paper: subsection `subsec:trust_positioning` (green R1) + comparison table
  `tab:trust_positioning` contrasting FedAgain vs Krum/Bulyan, FedMedian, FLTrust,
  FoolsGold, and reputation/blockchain FL (Kang) across 4 axes: dual signal,
  no clean root data, no #corrupt prior / no extra infra, different threat model
  (data corruption vs Byzantine/sybil poisoning). 4 refs added to `biblio.bib`
  (blanchard2017machine, cao2021fltrust, fung2020limitations, kang2019reliable;
  verified via web search). Filled the pre-existing inline TODO at old line ~334.
- **Overleaf to sync:** `biblio.bib`, the `.tex` (no new figures).

### 2.2 ✅ Code/data release + overhead  [R1.8] 🟢 — *DONE (committed 84d528b)*
- `analysis/analyze_overhead.py` from the ledger: FedAgain 281s/run vs FedAvg
  204s = **+38%** (~+7.7s/round), from the per-client benchmark inference pass
  (ℓ2 negligible; no communication overhead). Comparable to MOON (+42%), above
  FedMedian (+7%)/Bulyan (+9%). **Honest**: not "negligible" as first assumed.
- **Paper:** subsection `subsec:overhead` (green R1) — measured overhead + code
  release (github.com/Ivanrs297/FedAgain) + data-availability statement (private
  kidney datasets not redistributable). Cross-linked from clinical-deployment §.
- No Overleaf figures to sync (text only); just the `.tex`.

### 2.3 ✅ Reduce MNIST/CIFAR emphasis  [R1.7] 🟢 — *DONE*
- Reframed (green R1) in 3 high-visibility spots: abstract (kidney-stone datasets
  foregrounded, MNIST/CIFAR "used only as sanity checks"); Benchmark-Validation
  intro ("secondary check that complements—not replaces—the clinical evaluation");
  closing of that subsection ("primary evaluation target ... medical results given
  precedence"). No content deleted, no restructuring. Text-only (no Overleaf figs).

*Effort for 2.1–2.3: DONE (writing).*

---

## Phase 3 — Recent robust baselines  [R2.4 / R1.5] 🟣🟢 — *biggest chunk*

### 3.1 🔄 FedDyn, RFLPA, BOBA (RAHFL dropped) — *CODE DONE + verified; sweep pending*
- ✅ **FedDyn** (Acar et al., ICLR 2021; R1.5): `train_feddyn` + server drift `h`;
  `θ=mean(θ_k)-(1/α)h`. No root data, momentum OK. Verified MNIST 0.97→0.99.
- ✅ **RFLPA** (Mai et al., NeurIPS 2024) = FLTrust core, crypto omitted: ReLU'd
  cosine to a **step-matched** server reference on a clean root set, norm-clipped +
  trust-weighted. Verified MNIST 0.90→0.99. (Step-matching was essential — a small
  root set otherwise yields a tiny reference update and the model freezes.)
- ✅ **BOBA** (Bao et al., AISTATS 2024): robust affine-subspace fit + barycentric
  on-simplex filter (`boba_aggregate`). Feasibility clamp `n-f>c-1` → degenerates
  to mean on CIFAR/MNIST (10 clients, 10 classes), effective on clinical (c=6).
  Verified DAUDON 0.66→0.74 (kept 8/10). Note the CIFAR limitation honestly.
- ✅ Shared infra `carve_server_root` (balanced clean hold-out before partitioning,
  no leakage) → `dataset_info["server_root"]`; only for RFLPA/BOBA (backward
  compatible, regression-checked). Commit a961b5e.
- ❌ **RAHFL dropped** (decision): model-heterogeneous public-dataset distillation,
  no param aggregation → paradigm mismatch. Address as a justification paragraph
  (cite AugHFL, ICCV 2023, its homogeneous-FL predecessor).
- ⬜ **NEXT:** sweep FedDyn/RFLPA/BOBA (4 datasets × ρ × 3 seeds), fold into the
  baseline + degradation tables, write the R2.4 paragraph + RAHFL justification.

---

## Phase 4 — Adversarial attack  [R1.3 / R2.3] 🔵

### 4.1 ✅ Label-flipping poisoning attack — re-strengthened robustness claims — *DONE*
- ✅ **Code:** `LabelFlipDataset` + `poison_mode` branch in `src/dataset.py`
  (flips a fraction of a corrupted client's labels to a wrong class; images
  untouched; deterministic per client seed). Config flags `POISON_MODE`/
  `LABEL_FLIP_RATIO`; `make_run_key` `pm-label_flip` tag (no ledger collision).
  Runner `scripts/run_label_flip.{py,sh}` (FedAgain vs FedAvg/FedMedian,
  ρ∈{0.1,0.3}, 4 datasets, 3 seeds = 72 runs). Unit + integration tests PASS.
- ✅ **Sweep DONE** (72/72, ledger 480→552, 0 failed). `analyze_label_flip.py`
  → `results/baselines/{label_flip_summary.csv, label_flip.png}`.
- ✅ **Result (honest):** FedAgain's label-flip accuracy ≈ its image-corruption
  accuracy at the same ρ (74.1 vs 73.7% at ρ=0.3) — the benchmark-error term
  suppresses poisoned clients. At ρ=0.3 FedAgain beats undefended FedAvg by
  +16.7/+29.7/+6.6/+14.3 pts; ties FedMedian on the 3 clinical sets, +12.3 on
  CIFAR. Mechanism: flipped-label model → high E_k → trust weight collapses.
- ✅ **Paper:** `subsec:label_flip` (blue) + `fig:label_flip`; softened R1.3/R2.3
  claim (line ~1198) re-tightened to "robust to data-level poisoning"; limitation
  (line ~1488) narrowed to exclude only gradient-space/Byzantine attacks.
- **Overleaf to sync:** `label_flip.png`, the `.tex`.

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
