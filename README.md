# Federated Learning with PyTorch

A simple and extensible federated learning project using PyTorch with CUDA support for image classification.

## MedIA Revision Status (MEDIA-D-26-01276)

This repo backs a *Medical Image Analysis* revision responding to two reviewers.
The paper marks each change with a `\revtag{Rx.y}` tag naming the comment it
resolves, and the full point-by-point mapping is in
[`response_to_reviewers.md`](response_to_reviewers.md).

**All 15 reviewer sub-comments addressed** (~780 tracked runs). Fully resolved:
R1.1a/b/c (novelty vs trust FL, mult-vs-add ablation, convergence Proposition 1) ·
R1.2 theoretical analysis · R1.4 clinical split/leakage · R1.5 extended baselines
(FedBN/SCAFFOLD/MOON) + degradation curve · R1.7 MNIST/CIFAR de-emphasis (→ appendix) ·
R1.8 code release + overhead · R2.1 novelty vs prior FedAgain · R2.2 rare/hard-client
fairness · R2.4 recent baselines (FedDyn/RFLPA/BOBA; RAHFL paradigm-mismatch) ·
R2.5 full trust + normalization ablation.
Addressed **by scoping** (position stated explicitly in the paper): R1.3/R2.3
adversarial robustness — a label-flipping data-poisoning attack was added and the
threat model was tightened to exclude gradient-space Byzantine attacks; R1.6 length —
main body condensed −19 % with four appendices (net length conceded honestly).

### Reproducing the paper

Every experiment, table and figure is reproducible via the guided package in
**[`to-reproduce/`](to-reproduce/README.md)** — environment, data availability,
one command per result, and a paper-element → command index. Public
CIFAR-10/MNIST results reproduce from this repo alone; the two private clinical
datasets are available on request (see the paper's data-availability statement).

## Features

- **Framework**: Pure PyTorch (no Flower/Ray/TensorFlow dependencies)
- **Deep Learning**: PyTorch with CUDA support
- **Model**: ResNet18 with ImageNet pretrained weights
- **Datasets**: EMNIST, MNIST, CIFAR10 (easily extensible)
- **Strategies**: FedAvg, FedProx, FedYogi, FedMedian, Bulyan, FedBN, Scaffold, MOON, FedDyn, RFLPA, BOBA, FedAgain (custom)
- **Simple**: Clean manual implementation, easy to understand and modify

## Project Structure

```
├── main.py                 # Entry point: federated loop (all strategies)
├── config.py               # Configuration (dataset, strategy, hyperparameters)
├── src/                    # Library package
│   ├── model.py            # ResNet18 model
│   ├── dataset.py          # Dataset loading, partitioning, corruption (seeded)
│   ├── client.py           # Local training (incl. train_moon, train_scaffold)
│   ├── logger.py           # Legacy CSV logger (results/experiments.csv)
│   ├── tracking.py         # Resilient experiment tracker (incremental, resumable)
│   └── strategies/
│       ├── fedagain.py     # FedAgain trust-based strategy
│       └── trust.py        # Pluggable trust formulations (#2/#3 ablation)
├── scripts/                # Experiment runners (run from repo root)
│   ├── run_trust_ablation.py   # #2/#3 trust ablation sweep
│   ├── run_baselines.py        # FedAgain vs baselines sweep
│   ├── run_baseline_sweep.sh   # detached baseline comparison (ρ=0.3)
│   ├── run_envelope_sweep.sh   # detached corruption-envelope sweep (ρ=0.1/0.4)
│   ├── run_envelope_high.sh    # high-corruption tail sweep (ρ=0.5/0.6, R1.5/R2.4)
│   ├── run_label_flip.{py,sh}  # label-flipping adversarial attack sweep (R1.3/R2.3)
│   ├── watch_and_run_label_flip.sh # chains label-flip after the ρ sweep finishes
│   └── run_*.py                # other batch runners
├── analysis/               # Analysis & plotting (analyze_*, visualize_*)
│   ├── analyze_baselines.py         # extended-baseline table + figure (R1.5)
│   ├── analyze_degradation_curve.py # corruption-degradation curves (R1.5/R2.4)
│   ├── analyze_rare_client_bias.py  # rare/hard-client trust fairness (R2.2)
│   ├── analyze_overhead.py          # trust-weighting wall-clock overhead (R1.8)
│   └── analyze_trust_ablation.py    # trust-formulation ablation (#2/#3)
├── tests/                  # Unit tests (test_trust, test_tracking, ...)
├── docs/                   # Guides + EXPERIMENT_PLAN.md (reviewer mapping)
├── results/                # Experiment outputs (see below)
├── jcr-fedagain/src-paper/ # LaTeX manuscript (MedIA revision)
└── requirements.txt

# results/ layout (written by src/tracking.py, resumable):
#   experiment_ledger.csv   one row per finished run (config + metrics + status)
#   runs/<key>.json         per-run manifest (updated every round)
#   rounds/<key>.csv        per-round metrics, written live
#   trust_scores/           per-round E_k/D_k/T_k logs (FedAgain, FEDAGAIN_LOG_TRUST)
#   experiments.csv         legacy logger output (unchanged)
```

> **See [`CLAUDE.md`](CLAUDE.md)** for the full project context and the Medical
> Image Analysis revision status, and [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md)
> for the experiments mapped to each reviewer comment.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run federated learning:
```bash
python main.py
```

### Changing Dataset

Edit `config.py`:
```python
DATASET = "CIFAR10"  # Options: "EMNIST", "MNIST", "CIFAR10"
```

### Changing Strategy

Edit `config.py`:
```python
STRATEGY = "FedProx"  # Options: "FedAvg", "FedProx", "FedAdam", "FedMedian", "Bulyan", "FedAgain"
```

Then run:
```bash
python main.py
```

### Running Multiple Experiments

Test all strategies automatically:
```bash
# Test all 6 strategies with current dataset
python run_experiments.py

# Test specific strategies
python run_experiments.py --strategies FedAvg FedProx FedAgain

# Test all strategies on MNIST
python run_experiments.py --dataset MNIST

# Quick mode: test 3 main strategies
python run_experiments.py --quick

# Combine options
python run_experiments.py --dataset CIFAR10 --strategies FedAvg FedAgain
```

## Experiment Results

All experiment results are automatically logged to `results/experiments.csv` with:
- **Configuration**: Model, strategy, dataset, distribution, hyperparameters
- **Metrics**: Accuracy, Precision, Recall, F1 Score (macro-averaged)
- **Timestamp**: Date and time of experiment

Example CSV output:
```csv
timestamp,date,model,strategy,dataset,data_distribution,num_clients,num_rounds,local_epochs,batch_size,learning_rate,final_accuracy,final_precision,final_recall,final_f1
2025-01-15 10:30:45,2025-01-15,ResNet18,FedAvg,MNIST,iid,5,3,1,32,0.01,0.9950,0.9952,0.9949,0.9950
2025-01-15 11:15:20,2025-01-15,ResNet18,FedAgain,MNIST,non-iid-label,5,3,1,32,0.01,0.9845,0.9850,0.9842,0.9846
```

### Configuration

All configuration is in `config.py`:

```python
# Dataset Configuration
DATASET = "EMNIST"  # Options: "EMNIST", "MNIST", "CIFAR10"

# Federated Learning Configuration
STRATEGY = "FedAvg"  # Options: "FedAvg", "FedProx", "FedOpt", "FedMedian"
NUM_CLIENTS = 5      # Number of federated clients
NUM_ROUNDS = 3       # Number of federated learning rounds
LOCAL_EPOCHS = 1     # Local training epochs per round

# Data Distribution (Realistic Federated Scenarios)
DATA_DISTRIBUTION = "non-iid-label"  # Options: "iid", "non-iid-label", "non-iid-dirichlet"
NON_IID_CLASSES_PER_CLIENT = 2       # For label skew: classes per client
DIRICHLET_ALPHA = 0.5                 # For Dirichlet: lower = more non-IID

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Model Configuration
USE_PRETRAINED = True  # Use ImageNet pretrained weights

# Strategy-specific parameters
FEDPROX_PROXIMAL_MU = 0.1       # Proximal term for FedProx
FEDADAM_SERVER_LR = 0.01         # Server learning rate for FedAdam
FEDADAM_BETA1 = 0.9              # FedAdam momentum
FEDADAM_BETA2 = 0.99             # FedAdam second moment
FEDADAM_TAU = 1e-3               # FedAdam epsilon/tau
BULYAN_NUM_MALICIOUS = 1         # Bulyan: number of Byzantine clients to tolerate
BULYAN_TO_KEEP = None            # Bulyan: clients to keep (None = auto)
FEDAGAIN_EPSILON = 1e-6          # FedAgain: numerical stability constant

# Corrupted-client simulation
CORRUPT_CLIENTS = True           # enable corrupted clients
CORRUPT_CLIENT_RATIO = 0.3       # fraction of clients corrupted (ρ)
CORRUPTION_SEVERITY = 3          # image-corruption severity (1-5)
POISON_MODE = "image"            # "image" (on-the-fly corruption) | "label_flip" (attack)
LABEL_FLIP_RATIO = 1.0           # label_flip: fraction of a poisoned client's labels flipped
```

**Poisoning modes** (`POISON_MODE`): `"image"` applies on-the-fly image
corruptions (blur/noise/etc.) to corrupted clients, simulating data-quality
issues; `"label_flip"` is a genuine adversarial data-poisoning attack that flips
a fraction of a corrupted client's labels to a wrong class (images untouched).
Both share the same seeded corrupted-client selection; label-flip runs are tagged
`pm-label_flip` in the run key so they never collide with image-corruption runs.

## Supported Datasets

| Dataset | Classes | Channels | Description |
|---------|---------|----------|-------------|
| EMNIST  | 62      | 1 (grayscale) | Digits + uppercase + lowercase letters |
| MNIST   | 10      | 1 (grayscale) | Handwritten digits |
| CIFAR10 | 10      | 3 (RGB)       | Natural images (airplane, car, bird, etc.) |

## Data Distribution Options

Realistic federated scenarios with non-IID data:

| Distribution | Description | Use Case |
|--------------|-------------|----------|
| **iid** | Random uniform split | Baseline (unrealistic) |
| **non-iid-label** | Label skew - each client has only a few classes | Different clients have different types of data (e.g., different handwriting styles) |
| **non-iid-dirichlet** | Dirichlet distribution (α parameter) | Realistic quantity skew - clients have different amounts of each class |

**Example**: With `non-iid-label` and `classes_per_client=2` on MNIST, Client 1 might have only digits 0 and 1, Client 2 has 2 and 3, etc. This simulates real-world scenarios where different organizations have different data types.

## Supported Strategies

Manual implementations (no external framework dependencies):

| Strategy | Description | Key Feature |
|----------|-------------|-------------|
| **FedAvg** | Federated Averaging (McMahan et al., 2017) | Weighted average of client models |
| **FedProx** | Federated Proximal (Li et al., 2020) | Adds proximal term to handle heterogeneity |
| **FedMedian** | Coordinate-wise Median | Robust to Byzantine attacks and outliers |
| **Bulyan** | Byzantine-robust (Mhamdi et al., 2018) | Removes outliers before aggregation |
| **FedBN** | Local BatchNorm (Li et al., 2021) | Keeps each client's BN params local |
| **Scaffold** | Control variates (Karimireddy et al., 2020) | Corrects client drift under non-IID (plain SGD) |
| **MOON** | Model-contrastive (Li et al., 2021) | Contrastive term: local rep ↔ global vs. previous |
| **FedAgain** | Trust-based aggregation (Ours) | Weights clients by benchmark error × divergence |

### Strategy Details

All strategies are **fully implemented** manually (no Flower/Ray dependencies):

- **FedAvg**: Classic federated learning - weighted averaging of client models based on dataset sizes
- **FedProx**: Adds proximal term `(μ/2)||w - w_global||²` to client loss function to handle data heterogeneity - aggregation is weighted averaging
- **FedYogi**: Full server-side Yogi optimization with sign-based velocity update `v = v - (1-β₂)Δ²sign(v-Δ²)` - uses very conservative learning rate (0.001) and gradient clipping for deep network stability
- **FedMedian**: Coordinate-wise median aggregation - robust to outliers and malicious clients
- **Bulyan**: Byzantine-robust - removes `f` furthest outliers from median, then averages remaining clients
- **FedBN**: Keeps each client's BatchNorm parameters local (state in `client_bn_state`); only non-BN weights are aggregated
- **Scaffold**: Server/client control variates correct client drift; **uses plain SGD** (momentum breaks the control-variate estimate → divergence)
- **MOON**: Adds a model-contrastive loss aligning each client's representation with the global model vs. its previous local model (`train_moon`)
- **FedAgain**: Trust-based weighting - clients weighted by `T_k = 1/(E_k×D_k+ε)` where `E_k` is benchmark error and `D_k` is parameter divergence

## Adding New Datasets

To add a new dataset, modify `dataset.py`:

1. Add dataset configuration to `get_dataset_info()`:
```python
"YOUR_DATASET": {
    "num_classes": 10,
    "num_channels": 3,
    "input_size": 224,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5]
}
```

2. Add dataset loading logic to `load_dataset()`:
```python
elif dataset_name == "YOUR_DATASET":
    train_dataset = datasets.YourDataset(...)
    test_dataset = datasets.YourDataset(...)
```

3. Update `config.py` to use your dataset:
```python
DATASET = "YOUR_DATASET"
```

## Implementation Notes

- **Pure PyTorch**: No Flower, Ray, or TensorFlow dependencies
- **Manual Aggregation**: All strategies implemented from scratch for transparency
- **Automatic Adaptation**: ResNet18 adapts to grayscale/RGB and number of classes
- **Non-IID Support**: Realistic data distributions (label skew, Dirichlet)
- **Reproducible**: Fixed seeds for data partitioning
- **Validation Split**: Each client uses 90% train / 10% validation
- **CUDA Support**: Automatic GPU usage if available

## Troubleshooting

### FedAdam shows `nan` or exploding loss

**Symptoms**: Loss increases (e.g., 1.2 → 9.1 → 26.2), test loss shows `nan`

**Solutions**:
1. Reduce `FEDADAM_SERVER_LR` (try 0.01 or 0.001)
2. Increase `FEDADAM_TAU` (try 1e-3 instead of 1e-9)
3. Reduce `FEDADAM_BETA2` (try 0.99 instead of 0.999)
4. Check that pseudo-gradient direction is correct: `Δ = w_avg - w_t` (not `w_t - w_avg`)

See `FEDADAM_FIX.md` for detailed explanation.

### Low accuracy with non-IID data

**Solutions**:
1. Increase `NUM_ROUNDS` (try 10-20 instead of 3)
2. Increase `NON_IID_CLASSES_PER_CLIENT` (try 3-4 instead of 2)
3. Use `DATA_DISTRIBUTION = "iid"` to verify the model works
4. For non-IID, try **FedProx** with higher `FEDPROX_PROXIMAL_MU` (0.1-0.5)

### Out of memory errors

**Solutions**:
1. Reduce `BATCH_SIZE` (try 16 or 8 instead of 32)
2. Reduce `NUM_CLIENTS` (try 3 instead of 5)
3. Use `USE_PRETRAINED = False` to train from scratch (smaller initial model)

---

## 📊 Analysis Tools

This project includes powerful analysis tools for organizing and analyzing experimental results by dataset.

### Quick Start

```bash
# 1. Run experiments on a dataset
python test_fedagain_superiority.py --mystone

# 2. Analyze and organize results by dataset
python analyze_by_dataset.py

# 3. Compare performance across multiple datasets
python compare_datasets.py

# 4. FedAgain-focused cross-dataset analysis
python analyze_fedagain_cross_dataset.py

# 5. Check current status
python show_summary.py
```

### Features

- **Automatic Dataset Detection** - Detects all datasets in experiments.csv
- **Organized Structure** - Each dataset gets its own folder with analysis
- **5 Visualization Plots** - Publication-ready figures per dataset
- **Comprehensive Reports** - Detailed ANALYSIS_SUMMARY.md for each dataset
- **Cross-Dataset Comparison** - Compare FedAgain across datasets
- **Paper-Ready Outputs** - LaTeX-compatible figures and tables

### Output Structure

```
results/
├── experiments.csv              # All experiments
├── dataset_comparison.csv       # Cross-dataset comparison
│
├── MYSTONE/                     # Dataset-specific analysis
│   ├── experiments.csv
│   ├── ANALYSIS_SUMMARY.md
│   ├── plot1_corruption_comparison.png
│   ├── plot2_strategy_bar_30pct.png
│   ├── plot3_severity_impact.png
│   ├── plot4_improvement_percentage.png
│   └── plot5_f1_comparison.png
│
└── [other datasets...]/
```

### Documentation

- **SETUP_COMPLETE.md** - Overview of what's been set up
- **ANALYSIS_TOOLS_README.md** - Comprehensive tool reference
- **DATASET_ANALYSIS_GUIDE.md** - Detailed analysis guide

### Example Workflow

```bash
# Run experiments on multiple datasets
python test_fedagain_superiority.py --mystone
python test_fedagain_superiority.py --cifar10
python test_fedagain_superiority.py --jonathan-sur

# Analyze all datasets at once
python analyze_by_dataset.py

# Compare results across datasets
python compare_datasets.py

# View organized results
cd results/MYSTONE
# Windows: type ANALYSIS_SUMMARY.md
# Linux/Mac: cat ANALYSIS_SUMMARY.md
```

---

