# Trust ablation (#2) and multiplicative-vs-additive (#3) — summary

Setup: non-iid-label, rho=0.3, severity 3, 10 clients, 10 rounds, seeds 42/43/44 (accuracy = mean over seeds).

See `table_accuracy_f1.md`, `suppression.md`, `convergence.png`, `trust_distribution.png`.

## Files
- table_accuracy_f1.{md,csv}
- suppression.{md,csv}
- convergence.png
- trust_distribution.png

## How to read
- **#2 ablation**: compare `E only` / `D only` / `E*D (mult)` columns.
- **#3 sensitivity**: compare `E*D (mult)` vs `E+D (add)` and the weighted_additive block; also the benign/corrupt suppression ratio.
- **#10 convergence**: accuracy-vs-round curves.

