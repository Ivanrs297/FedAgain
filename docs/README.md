# Documentation

Project guides and notes (moved out of the repo root during the src/ restructure).

## Revision (Medical Image Analysis)
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — experiments mapped to each reviewer comment (start here for the revision).
- [Plan_Revision_FedAgain_jul2026.md](Plan_Revision_FedAgain_jul2026.md) — the full reviewer-response plan (Tier 1–3).

## Setup & usage
- [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
- [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)
- [ANALYSIS_TOOLS_README.md](ANALYSIS_TOOLS_README.md)
- [requirements](../requirements.txt) (repo root)

## Datasets & corruption
- [KIDNEY_STONE_DATASETS.md](KIDNEY_STONE_DATASETS.md)
- [KIDNEY_STONE_TESTING.md](KIDNEY_STONE_TESTING.md)
- [CIFAR10C_README.md](CIFAR10C_README.md)
- [CORRUPTED_CLIENTS_INFO.txt](CORRUPTED_CLIENTS_INFO.txt)
- [DATASET_ANALYSIS_GUIDE.md](DATASET_ANALYSIS_GUIDE.md)

## Analysis / results
- [ALL_STRATEGIES_SUMMARY.md](ALL_STRATEGIES_SUMMARY.md)
- [CROSS_DATASET_ANALYSIS_GUIDE.md](CROSS_DATASET_ANALYSIS_GUIDE.md)
- [FEDAGAIN_CROSS_ANALYSIS_SUMMARY.md](FEDAGAIN_CROSS_ANALYSIS_SUMMARY.md)
- [FEDAGAIN_HIGHLIGHTS_GUIDE.md](FEDAGAIN_HIGHLIGHTS_GUIDE.md)
- [FEDAGAIN_TESTING_GUIDE.md](FEDAGAIN_TESTING_GUIDE.md)

## Experiment records (generated at runtime)
Written by `src/tracking.py` under `results/`:
- `experiment_ledger.csv` — one row per finished run (config + final metrics + status).
- `runs/<run_key>.json` — per-run manifest (config, status, every round).
- `rounds/<run_key>.csv` — per-round metrics, written live.
- `ledger.jsonl` — append-only start/finish/fail event stream.
