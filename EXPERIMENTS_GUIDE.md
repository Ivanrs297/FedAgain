# Comprehensive Experiments Guide

This guide explains how to run experiments across all combinations of datasets, strategies, and models.

## 📊 Available Configurations

### Datasets (4)
- **MNIST**: Handwritten digits (10 classes, grayscale)
- **EMNIST**: Extended MNIST with letters (47 classes, grayscale)
- **CIFAR10**: Natural images (10 classes, RGB)
- **CIFAR100**: Natural images (100 classes, RGB)

### Strategies (6)
- **FedAvg**: Federated Averaging (baseline)
- **FedProx**: Proximal term for heterogeneous data
- **FedYogi**: Adaptive server-side optimization
- **FedMedian**: Coordinate-wise median (Byzantine-robust)
- **Bulyan**: Outlier removal (Byzantine-robust)
- **FedAgain**: Trust-based aggregation (custom)

### Models (2)
- **ResNet18**: 11M parameters, faster training
- **ResNet50**: 25M parameters, better accuracy

### Total Combinations
**4 datasets × 6 strategies × 2 models = 48 experiments**

---

## 🚀 Running Experiments

### Option 1: Single Experiment

Modify `config.py` and run:

```python
# In config.py
DATASET = "MNIST"
STRATEGY = "FedAvg"
MODEL_NAME = "resnet18"
NUM_ROUNDS = 3
```

```bash
python main.py
```

---

### Option 2: Batch Experiments (All Combinations)

Run all 48 experiments automatically:

```bash
python run_full_experiments.py
```

**Estimated time**: 4-12 hours depending on your hardware

**What it does:**
- Tests every combination of dataset, strategy, and model
- Saves all results to `results/experiments.csv`
- Handles failures gracefully
- Shows progress and timing

---

### Option 3: Quick Test (6 experiments)

Test a subset quickly:

```bash
python run_full_experiments.py --quick
```

**Runs:**
- Datasets: MNIST, CIFAR10
- Strategies: FedAvg, FedProx, FedAgain
- Model: ResNet18 only

**Estimated time**: 30-60 minutes

---

### Option 4: Custom Subset

Select specific configurations:

```bash
# Test specific datasets
python run_full_experiments.py --datasets MNIST CIFAR10

# Test specific strategies
python run_full_experiments.py --strategies FedAvg FedProx FedYogi

# Test specific models
python run_full_experiments.py --models resnet18

# Combine filters
python run_full_experiments.py --datasets MNIST --strategies FedAvg FedProx --models resnet18
```

---

## 📋 Results

All results are saved to `results/experiments.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | When the experiment was run |
| `date` | Date of experiment |
| `model` | ResNet18 or ResNet50 |
| `strategy` | Federated learning strategy |
| `dataset` | Dataset used |
| `data_distribution` | IID or non-IID |
| `num_clients` | Number of federated clients |
| `num_rounds` | Training rounds |
| `local_epochs` | Epochs per client per round |
| `batch_size` | Batch size for training |
| `learning_rate` | Learning rate |
| `final_accuracy` | Test accuracy (0-1) |
| `final_precision` | Macro precision |
| `final_recall` | Macro recall |
| `final_f1` | Macro F1 score |
| `use_pretrained` | Whether ImageNet weights were used |

---

## ⚙️ Configuration Tips

### For Faster Experiments

```python
# In config.py
NUM_ROUNDS = 3  # Reduce rounds
NUM_CLIENTS = 3  # Fewer clients
BATCH_SIZE = 64  # Larger batches
MODEL_NAME = "resnet18"  # Smaller model
```

### For Better Accuracy

```python
# In config.py
NUM_ROUNDS = 10  # More rounds
NUM_CLIENTS = 10  # More clients
BATCH_SIZE = 16  # Smaller batches (better generalization)
MODEL_NAME = "resnet50"  # Larger model
DATA_DISTRIBUTION = "iid"  # Easier data distribution
```

### For Non-IID Experiments

```python
# In config.py
DATA_DISTRIBUTION = "non-iid-label"  # Label skew
NON_IID_CLASSES_PER_CLIENT = 2  # Each client has 2 primary classes

# Or
DATA_DISTRIBUTION = "non-iid-dirichlet"  # Quantity skew
DIRICHLET_ALPHA = 0.5  # Lower = more non-IID
```

---

## 🔍 Monitoring Progress

The experiment runner shows:
- Current experiment (X/48)
- Configuration being tested
- Success/failure status
- Time per experiment
- Final summary table

Example output:
```
================================================================================
[12/48] Running: RESNET18 + FedProx on MNIST
================================================================================
✓ Completed in 312.5s

FINAL SUMMARY
================================================================================
Total experiments: 48
Successful: 45 (93.8%)
Failed: 3 (6.2%)
Total time: 4:23:15
Average time per experiment: 327.2s
================================================================================
```

---

## 🎯 Expected Performance

### MNIST (Easy)
- **FedAvg**: 95-98%
- **FedProx**: 95-98%
- **FedYogi**: 90-95% (needs more rounds)
- **FedMedian**: 92-96%
- **Bulyan**: 90-94%
- **FedAgain**: 94-97%

### CIFAR10 (Medium)
- **FedAvg**: 60-70%
- **FedProx**: 60-70%
- **FedYogi**: 50-65% (needs many rounds)
- **FedMedian**: 55-65%
- **Bulyan**: 50-60%
- **FedAgain**: 58-68%

### CIFAR100 (Hard)
- **FedAvg**: 30-40%
- **FedProx**: 30-40%
- **FedYogi**: 25-35%
- **FedMedian**: 25-35%
- **Bulyan**: 20-30%
- **FedAgain**: 28-38%

### EMNIST (Medium)
- **FedAvg**: 75-85%
- **FedProx**: 75-85%
- **FedYogi**: 70-80%
- **FedMedian**: 70-80%
- **Bulyan**: 65-75%
- **FedAgain**: 73-83%

**Note**: ResNet50 typically achieves 2-5% higher accuracy than ResNet18 but takes 2-3x longer to train.

---

## 💾 Managing Results

### View Results

```bash
# View in Excel/LibreOffice
open results/experiments.csv

# Or use pandas in Python
python
>>> import pandas as pd
>>> df = pd.read_csv('results/experiments.csv')
>>> print(df.groupby(['dataset', 'strategy'])['final_accuracy'].mean())
```

### Backup Results

```bash
cp results/experiments.csv results/experiments_backup_$(date +%Y%m%d).csv
```

### Clear Results

```bash
rm results/experiments.csv
# Will be recreated on next run
```

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Use fewer clients
NUM_CLIENTS = 3

# Use ResNet18 instead of ResNet50
MODEL_NAME = "resnet18"
```

### Slow Training

```python
# Check if CUDA is being used
# Should see: Using device: cuda:0

# Increase batch size (if memory allows)
BATCH_SIZE = 64

# Reduce image resolution (advanced - edit dataset.py)
transforms.Resize(112)  # instead of 224
```

### FedYogi Low Accuracy

```python
# FedYogi needs more rounds
NUM_ROUNDS = 10  # instead of 3

# Or reduce server LR even more
FEDYOGI_SERVER_LR = 0.0001  # instead of 0.001
```

---

## 📚 References

- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
- **FedYogi**: Reddi et al., "Adaptive Federated Optimization" (2020)
- **FedMedian**: Yin et al., "Byzantine-Robust Distributed Learning" (2018)
- **Bulyan**: Mhamdi et al., "The Hidden Vulnerability of Distributed Learning" (2018)

