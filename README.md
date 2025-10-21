# Federated Learning with PyTorch

A simple and extensible federated learning project using PyTorch with CUDA support for image classification.

## Features

- **Framework**: Pure PyTorch (no Flower/Ray/TensorFlow dependencies)
- **Deep Learning**: PyTorch with CUDA support
- **Model**: ResNet18 with ImageNet pretrained weights
- **Datasets**: EMNIST, MNIST, CIFAR10 (easily extensible)
- **Strategies**: FedAvg, FedProx, FedAdam, FedMedian, Bulyan, FedAgain (custom)
- **Simple**: Clean manual implementation, easy to understand and modify

## Project Structure

```
├── main.py             # Main federated learning loop (all strategies)
├── run_experiments.py  # Batch experiment runner
├── config.py           # Configuration (dataset, strategy, hyperparameters)
├── client.py           # Client training and evaluation functions
├── model.py            # ResNet18 model definition
├── dataset.py          # Dataset loading and partitioning
├── fedagain.py         # FedAgain custom strategy
├── logger.py           # CSV experiment logger
├── results/            # Experiment results (CSV files)
└── requirements.txt    # Dependencies
```

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
```

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
| **FedAgain** | Trust-based aggregation (Custom) | Weights clients by benchmark error × divergence |

### Strategy Details

All strategies are **fully implemented** manually (no Flower/Ray dependencies):

- **FedAvg**: Classic federated learning - weighted averaging of client models based on dataset sizes
- **FedProx**: Adds proximal term `(μ/2)||w - w_global||²` to client loss function to handle data heterogeneity - aggregation is weighted averaging
- **FedYogi**: Full server-side Yogi optimization with sign-based velocity update `v = v - (1-β₂)Δ²sign(v-Δ²)` - uses very conservative learning rate (0.001) and gradient clipping for deep network stability
- **FedMedian**: Coordinate-wise median aggregation - robust to outliers and malicious clients
- **Bulyan**: Byzantine-robust - removes `f` furthest outliers from median, then averages remaining clients
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

