# FedAgain — manuscript (MEDIA-D-26-01276)

Manuscript under revision for *Medical Image Analysis*. **Not tracked in git**
(under-review paper); keep a backup before any structural edit.

## Layout

```
jcr-fedagain/
└── src-paper/                        <- upload this folder to Overleaf
    ├── FedAgain_classification.tex   <- the manuscript (Elsevier cas-dc, two-column)
    ├── biblio.bib                    <- bibliography (129 entries)
    └── images/
        ├── results/                  <- GENERATED — do not edit by hand
        ├── supplementary/            <- GENERATED — four-dataset (*_supp) variants
        ├── datasets/                 <- dataset sample images and corruption examples
        └── diagrams/                 <- conceptual schematics and legacy plots
```

`\graphicspath` in the preamble lists all four sub-folders, so every
`\includegraphics` stays path-free: figures can be re-categorized without
touching the body of the paper.

## Regenerating the figures

`images/results/` and `images/supplementary/` are written by the analysis
scripts — never edit them by hand, they are overwritten. From the repo root:

```bash
# main text: the three clinical datasets (A, B, C)
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_baselines.py
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_degradation_curve.py
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_convergence.py
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_label_flip.py
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_epsilon_sensitivity.py
FEDAGAIN_CLINICAL_ONLY=1 python analysis/analyze_rare_client_bias.py
python analysis/analyze_trust_ablation.py          # trust figures (clinical by construction)

# supplementary: same six figures with CIFAR-10 included, saved as *_supp
# (run the same commands with FEDAGAIN_CLINICAL_ONLY unset, then rename to
#  *_supp.png and move to images/supplementary/)
```

Rationale for the split: Reviewer 1.7 asked that MNIST/CIFAR-10 be
de-emphasized, so the main text reports the clinical datasets only and the
benchmarks live in the appendix.

## Conventions

- **Revision changes are colour-coded by reviewer**: `\textcolor{revIcol}{}`
  green = Reviewer 1, `\textcolor{revIIcol}{}` violet = Reviewer 2,
  `\textcolor{blue}{}` = common. Multi-paragraph blocks use `{\color{...} ... }`
  (a group, so `\par` is allowed); inline uses `\textcolor{...}{...}`.
- `\revtag{Rx.y}` next to a heading names the reviewer comment it resolves.
- The paper compiles in **Overleaf only** (no local `pdflatex`/`latexdiff`).
- Verify before every upload: balanced braces/environments, no broken
  `\ref`, no `\cite` without a bib entry, and every `\includegraphics`
  resolvable through `\graphicspath`.
