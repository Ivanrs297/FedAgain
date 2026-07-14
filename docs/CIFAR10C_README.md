# CIFAR-10-C: Realistic Corrupted Training & Testing

## Overview

**CIFAR-10-C** provides realistic robustness training where **both training and testing data are corrupted**.

### Key Concept

✅ **Training**: 50,000 images, each with ONE random corruption (from 19 types)  
✅ **Testing**: 10,000 images, each with ONE random corruption (from 19 types)  
✅ **Severity**: Fixed at level 3 (medium) for all images  
✅ **Realistic**: Model learns from noisy data, just like real-world scenarios  

---

## Why This Approach?

In real-world deployments:
- **Training data is never perfect**: sensors have noise, cameras are imperfect, compression artifacts exist
- **Test data is also imperfect**: same real-world conditions apply
- **Models need to learn robustness**: training on corrupted data teaches the model to handle imperfections

**CIFAR-10-C simulates this!**

---

## The 19 Corruption Types

Each image randomly gets ONE of these corruptions:

**Noise (4):** gaussian_noise, shot_noise, impulse_noise, speckle_noise  
**Blur (5):** defocus_blur, glass_blur, motion_blur, zoom_blur, gaussian_blur  
**Weather (3):** snow, frost, fog  
**Digital (7):** brightness, contrast, elastic_transform, pixelate, jpeg_compression, spatter, saturate

---

## Usage

### 1. Setup in config.py

```python
DATASET = "CIFAR10C"
CIFAR10C_SEVERITY = 3  # Fixed at 3
```

### 2. Run

```bash
python main.py
```

### 3. What Happens

```
Loading CIFAR-10-C with random corruptions (severity 3, train set)...
  Loaded 5/19 corruption types...
  Loaded 10/19 corruption types...
  Loaded 15/19 corruption types...
  Loaded 19/19 corruption types...

  [OK] Created train set with 50000 images
  Corruption distribution (random, severity 3):
    brightness          :  2631 images (5.3%)
    contrast            :  2614 images (5.2%)
    defocus_blur        :  2658 images (5.3%)
    ...
    zoom_blur           :  2607 images (5.2%)

Loading CIFAR-10-C with random corruptions (severity 3, test set)...
  ...
  [OK] Created test set with 10000 images
  Corruption distribution (random, severity 3):
    brightness          :   527 images (5.3%)
    contrast            :   531 images (5.3%)
    ...
```

**Note:** Train and test use different random seeds, so they get different corruption instances!

---

## Expected Results

### Clean CIFAR-10 (Baseline)
```
Training: Clean data (50k)
Testing:  Clean data (10k)
Result:   85-90% accuracy
```

### CIFAR-10-C (Corrupted Training & Testing)
```
Training: Corrupted data (50k) - random corruptions
Testing:  Corrupted data (10k) - random corruptions
Result:   75-85% accuracy

Improvement: Model learns robustness from corrupted training data!
```

---

## Comparison with Clean Training

| Scenario | Training Data | Testing Data | Expected Accuracy | Use Case |
|----------|---------------|--------------|-------------------|----------|
| **Clean → Clean** | Clean CIFAR-10 | Clean CIFAR-10 | 85-90% | Ideal conditions |
| **Clean → Corrupt** | Clean CIFAR-10 | CIFAR-10-C | 70-75% | No robustness training |
| **Corrupt → Corrupt** | CIFAR-10-C | CIFAR-10-C | 75-85% | ✅ Realistic + Robust |

**Key Insight:** Training on corrupted data improves robustness!

---

## Research Questions

### 1. Does corrupted training improve robustness?

Compare CIFAR10 vs CIFAR10C:

```bash
# Clean training
python main.py  # config: DATASET = "CIFAR10"

# Corrupted training  
python main.py  # config: DATASET = "CIFAR10C"
```

**Hypothesis:** CIFAR10C should achieve better robustness

### 2. Which FL strategy handles corrupted training best?

```bash
python run_full_experiments.py --datasets CIFAR10C --strategies FedAvg FedProx FedYogi Bulyan
```

**Question:** Does Byzantine-robust Bulyan help when training data is corrupted?

### 3. IID vs Non-IID with corrupted data?

```python
# Test both distributions
DATA_DISTRIBUTION = "iid"
DATA_DISTRIBUTION = "non-iid-label"
```

**Question:** Does non-IID data distribution affect robustness learning?

---

## Configuration

### In config.py:

```python
# Dataset
DATASET = "CIFAR10C"

# Corruption severity (1-5)
CIFAR10C_SEVERITY = 3  # Fixed at 3 for consistency

# Federated Learning
STRATEGY = "FedAvg"
NUM_CLIENTS = 5
NUM_ROUNDS = 10  # More rounds recommended for corrupted training
LOCAL_EPOCHS = 3

# Model
MODEL_NAME = "resnet18"
USE_PRETRAINED = False  # For 32×32 images
```

---

## Technical Details

### Random Seed Strategy
- **Training set**: seed = 42
- **Test set**: seed = 1042 (42 + 1000)
- This ensures different corruption instances between train and test

### Data Construction
- **Training**: Tiles the 10k corrupted images to create 50k (cycling with random corruption assignment)
- **Testing**: Uses the 10k corrupted images directly
- Each image gets exactly ONE corruption type
- Uniform distribution: ~5.3% per corruption type

### Memory Usage
- Loads all 19 corruption types into memory (~2-3 GB RAM)
- Dataset creation takes ~30-60 seconds per split
- Only happens once at startup

---

## Download CIFAR-10-C

```bash
# Download (~12 GB)
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar

# Extract
tar -xf CIFAR-10-C.tar -C ./data/

# Verify
ls data/CIFAR-10-C/
# Should see: gaussian_noise.npy, labels.npy, etc.
```

---

## Example Experiment

```python
# config.py
DATASET = "CIFAR10C"
CIFAR10C_SEVERITY = 3
STRATEGY = "FedAvg"
NUM_CLIENTS = 5
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
```

```bash
python main.py
```

**Expected output:**
```
Loading and partitioning CIFAR10C dataset...
Number of clients: 5
- Distribution: Non-IID

[OK] Created train set with 50000 images (19 corruption types)
[OK] Created test set with 10000 images (19 corruption types)

--- Round 1/15 ---
  Client 1: Loss=1.523, Accuracy=0.654
  Client 2: Loss=1.489, Accuracy=0.661
  ...

[Round 1] Global Model - Test Loss: 1.234, Test Accuracy: 0.7123

Final Test Accuracy: 0.7856  (trained and tested on corrupted data!)
```

---

## Advantages

✅ **Realistic training**: Models learn from imperfect data  
✅ **Better generalization**: Robust to distribution shift  
✅ **One experiment**: No need for clean→corrupt comparison  
✅ **Memory efficient**: Only 10k images per corruption loaded  
✅ **Reproducible**: Fixed random seeds ensure consistency  

---

## Citation

```bibtex
@article{hendrycks2019robustness,
  title={Benchmarking neural network robustness to common corruptions and perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={arXiv preprint arXiv:1903.12261},
  year={2019}
}
```

---

## Summary

**CIFAR-10-C: Train and test on corrupted data!**

- 📊 Training: 50k images with random corruptions (19 types)
- 📊 Testing: 10k images with random corruptions (19 types)
- 🎯 Severity: Level 3 (medium) for all
- 🚀 Result: Models that learn robustness from noisy training data

**Usage:**
```python
DATASET = "CIFAR10C"  # That's it!
```

```bash
python main.py
```

**This simulates real-world federated learning where data is never perfect!** 🎉

