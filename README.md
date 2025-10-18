# Federated Learning with PyTorch

A simple and extensible federated learning project using PyTorch with CUDA support for image classification.

## Features

- **Framework**: PyTorch + Flower (simple, no TensorFlow)
- **Deep Learning**: PyTorch with CUDA support
- **Model**: ResNet18 with ImageNet pretrained weights
- **Datasets**: EMNIST, MNIST, CIFAR10 (easily extensible)
- **Strategies**: FedAvg, FedProx, FedOpt (FedAdam), FedMedian
- **Simple**: Single main file, strategies in server.py

## Project Structure

```
├── main.py          # Main federated learning with Flower
├── config.py        # Configuration (dataset, strategy, hyperparameters)
├── server.py        # Strategies (FedAvg, FedProx, FedOpt, FedMedian)
├── client.py        # Client training and evaluation
├── model.py         # ResNet18 model definition
├── dataset.py       # Dataset loading and partitioning
└── requirements.txt # Dependencies
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
STRATEGY = "FedProx"  # Options: "FedAvg", "FedProx", "FedOpt", "FedMedian"
```

Then run:
```bash
python main.py
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
FEDOPT_SERVER_LR = 0.01          # Server learning rate for FedOpt
FEDOPT_BETA1 = 0.9               # FedOpt momentum
FEDOPT_BETA2 = 0.99              # FedOpt second moment
FEDOPT_TAU = 1e-3                # FedOpt epsilon/tau
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

Using Flower's built-in implementations:

| Strategy | Description | Key Feature |
|----------|-------------|-------------|
| **FedAvg** | Federated Averaging (McMahan et al., 2017) | Weighted average of client models |
| **FedProx** | Federated Proximal (Li et al., 2020) | Adds proximal term to handle heterogeneity |
| **FedOpt** | FedAdam (Reddi et al., 2020) | Server-side adaptive optimization (Adam) |
| **FedMedian** | Coordinate-wise Median | Robust to Byzantine attacks and outliers |
| **Bulyan** | Byzantine-robust (Mhamdi et al., 2018) | Removes outliers before aggregation |
| **FedAgain** | Trust-based aggregation (Custom) | Weights clients by benchmark error × divergence |

### Strategy Details

- **FedAvg**: Classic federated learning - aggregates client models using weighted averaging
- **FedProx**: Adds a proximal term to limit client drift (good for non-IID data)
- **FedOpt (FedAdam)**: Uses server-side Adam optimizer for potentially faster convergence
- **FedMedian**: Uses coordinate-wise median - more robust to outliers/malicious clients
- **Bulyan**: Byzantine-robust aggregation - removes outliers before computing mean (strongest defense against malicious clients)
- **FedAgain**: Trust-based weighting - clients with lower benchmark error and smaller divergence get higher weight (T_k = 1/(E_k×D_k+ε))

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

## Notes

- The ResNet18 model automatically adapts to the number of channels (grayscale or RGB)
- The final layer is adapted for the dataset's number of classes
- Data is randomly partitioned across clients using a fixed seed for reproducibility
- Each client uses 90% of its data for training and 10% for validation
- No Ray or TensorFlow dependencies - pure PyTorch!

