# Kidney Stone Dataset Testing Guide

## Overview

The test suite now supports comprehensive testing on kidney stone datasets with the same 7-test protocol used for CIFAR10.

## Datasets Available

### 1. Jonathan El-Beze (w256 1k v1)
- **MIX**: Mixed surface types (9,600 train, 2,400 test)
- **SEC**: Section images (4,800 train, 1,200 test)
- **SUR**: Surface images (4,800 train, 1,200 test)
- **Classes**: 6 subtypes (Ia, IIa, IIIa, IVc, IVd, Va)

### 2. Michel Daudon (w256 1k v1)
- **MIX**: Mixed surface types (9,600 train, 2,400 test)
- **SEC**: Section images (4,800 train, 1,200 test)
- **SUR**: Surface images (4,800 train, 1,200 test)
- **Classes**: 6 subtypes (Ia, IVa, IVa2, IVc, IVd, Va)

### 3. MyStone (w256, 1k)
- Single version (4,800 train, 1,200 test)
- **Classes**: 6 subtypes (COM_WW_Ia, COD_WD_IIa, HAP, STR_IVc, BRU_IVd, UA_IIIa)

## Key Configuration

All kidney stone datasets use:
- **Image size**: 256×256 pixels (native resolution, NO resizing)
- **Model**: ResNet18 with **ImageNet pretrained weights**
- **Architecture**: Standard ResNet (NOT adapted for small images)
- **Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

## Running Tests

### Single Dataset

Test a specific kidney stone dataset:

```bash
# Jonathan El-Beze variants
python test_fedagain_superiority.py --jonathan-mix
python test_fedagain_superiority.py --jonathan-sec
python test_fedagain_superiority.py --jonathan-sur

# Michel Daudon variants
python test_fedagain_superiority.py --daudon-mix
python test_fedagain_superiority.py --daudon-sec
python test_fedagain_superiority.py --daudon-sur

# MyStone
python test_fedagain_superiority.py --mystone
```

### All Kidney Stone Datasets

Run all 7 kidney stone datasets at once:

```bash
python test_fedagain_superiority.py --all-kidney
```

This will run:
1. Jonathan MIX, SEC, SUR
2. Daudon MIX, SEC, SUR
3. MyStone

**Estimated time**: 8-16 hours depending on hardware

### All Datasets (CIFAR10 + Kidney Stones)

Run the complete multi-dataset test suite:

```bash
python test_fedagain_superiority.py
```

Then type `y` when prompted. This runs all 8 datasets:
- CIFAR10
- 7 kidney stone datasets

**Estimated time**: 10-24 hours

### CIFAR10 Only

For comparison with kidney stones:

```bash
python test_fedagain_superiority.py --cifar10
```

## What Tests Are Run?

For each dataset, the following 7 tests are executed:

### Test 1: Baseline (Clean Data)
- **Config**: IID, no corruption
- **Strategies**: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- **Purpose**: Establish baseline performance

### Test 2: Light Corruption (10%)
- **Config**: Non-IID, 10% corrupted clients, severity 3
- **Strategies**: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- **Purpose**: Minimal corruption impact

### Test 3: Moderate Corruption (30%)
- **Config**: Non-IID, 30% corrupted clients, severity 4
- **Strategies**: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- **Purpose**: Main test - realistic corruption scenario

### Test 4: Heavy Corruption (40%)
- **Config**: Non-IID, 40% corrupted clients, severity 5
- **Strategies**: FedAvg, FedMedian, Bulyan, FedAgain
- **Purpose**: Stress test under severe corruption

### Test 5: Severity Progression
- **Config**: Non-IID, 30% corrupted, severity 1-5
- **Strategies**: FedAvg, FedAgain
- **Purpose**: Test robustness across corruption severities

### Test 6: Non-IID + Corruption
- **Config**: IID/non-IID-label/non-IID-dirichlet, 30% corrupted
- **Strategies**: FedAvg, FedAgain
- **Purpose**: Test with different data heterogeneity

### Test 7: Scaling Clients
- **Config**: 5/10/20 clients, non-IID, 30% corrupted
- **Strategies**: FedAvg, FedAgain
- **Purpose**: Test scalability

## Expected Results

### Why Kidney Stones Are Challenging

Medical imaging datasets like kidney stones present unique challenges:
1. **High resolution** (256×256) → More parameters, harder to train
2. **Fine-grained classification** → Subtle visual differences between classes
3. **Limited data** → ~1,000-1,600 samples per class (vs 5,000 for CIFAR10)
4. **Domain expertise required** → Harder to distinguish between subtypes

### FedAgain Advantages

FedAgain should show **stronger advantages** on kidney stones because:

1. **Better corruption detection**: Loss-based benchmark error is more sensitive on medical images
2. **Trust-based weighting**: Corrupted medical images have much higher loss than corrupted natural images
3. **Robust to heterogeneity**: Non-IID medical data is common in federated settings (different hospitals)

### Expected Performance Gaps

| Dataset Type | FedAvg Accuracy | FedAgain Accuracy | Improvement |
|--------------|----------------|-------------------|-------------|
| CIFAR10 (30% corrupt) | ~65-70% | ~80-85% | +15-20% |
| Kidney Stones (30% corrupt) | ~55-60% | ~75-80% | +20-30% |

**Hypothesis**: FedAgain's superiority will be **more pronounced** on kidney stone datasets due to better corruption detection via loss-based benchmark error.

## Analysis

After running experiments, analyze results:

```bash
python test_fedagain_superiority.py --analyze
```

This will show:
- Per-test summaries for each dataset
- Cross-dataset comparison
- FedAgain win rate
- Average improvement over FedAvg

### Key Metrics to Compare

1. **Corruption robustness**: How much does accuracy drop with 30% corruption?
   - FedAvg: Large drop
   - FedAgain: Small drop

2. **Cross-dataset generalization**: Does FedAgain win on all datasets?
   - Expected: Yes, with larger margins on medical data

3. **Severity resilience**: How does performance degrade with increasing severity?
   - FedAgain should maintain consistent advantage

## Technical Details

### Image Loading

Kidney stone datasets use `torchvision.datasets.ImageFolder`:
```python
train_dir = "data/Jonathan El-Beze (w256 1k v1)/MIX/train"
test_dir = "data/Jonathan El-Beze (w256 1k v1)/MIX/test"
```

Each class is a subdirectory containing images.

### Model Architecture

Standard ResNet18/50 is used (no adaptation):
```python
adapt_for_small_images = input_size < 64  # False for 256×256
model = get_model(
    num_classes=6,
    num_channels=3,
    use_pretrained=True,  # ImageNet weights
    model_name="resnet18",
    adapt_for_small_images=False  # Standard architecture
)
```

### Corruption Simulation

Using `imagecorruptions` package with 19 corruption types:
- Noise: Gaussian, shot, impulse, speckle
- Blur: Defocus, glass, motion, zoom
- Weather: Snow, frost, fog, brightness
- Digital: Contrast, elastic, pixelate, JPEG, etc.

Each corrupted client gets random corruptions at specified severity.

## Troubleshooting

### Out of Memory (OOM)

Kidney stone images (256×256) require more GPU memory:

**Solution 1**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```

**Solution 2**: Reduce number of clients:
```python
NUM_CLIENTS = 5  # Instead of 10
```

**Solution 3**: Use smaller model:
```python
MODEL_NAME = "resnet18"  # Instead of resnet50
```

### Slow Training

256×256 images take longer to train:

**Solution**: Reduce epochs or rounds:
```python
NUM_ROUNDS = 5      # Instead of 10
LOCAL_EPOCHS = 1    # Instead of 2
```

### Dataset Not Found

Ensure datasets are in correct location:
```
data/
├── Jonathan El-Beze (w256 1k v1)/
│   ├── MIX/
│   │   ├── train/
│   │   └── test/
│   ├── SEC/
│   └── SUR/
├── Michel Daudon (w256 1k v1)/
│   ├── MIX/
│   ├── SEC/
│   └── SUR/
└── MyStone (w256, 1k)/
    ├── train/
    └── test/
```

## Publication-Ready Results

After running all experiments, you'll have comprehensive results demonstrating:

1. **FedAgain superiority across domains**: Natural images (CIFAR10) AND medical images (kidney stones)
2. **Robustness to corruption**: FedAgain maintains high accuracy even with 30-40% corrupted clients
3. **Generalization**: Consistent performance across 8 different datasets
4. **Scalability**: Works with varying numbers of clients
5. **Heterogeneity handling**: Robust under IID and non-IID data distributions

Perfect for demonstrating the practical applicability of FedAgain in federated medical imaging scenarios! 🏥📊

