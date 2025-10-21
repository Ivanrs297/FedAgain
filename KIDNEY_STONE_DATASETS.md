# Kidney Stone Datasets

This document describes the kidney stone microscopy image datasets integrated into the federated learning framework.

## 📊 Available Datasets

### 1. Jonathan El-Beze Collection
**Base Path**: `data/Jonathan El-Beze (w256 1k v1)/`

| Dataset | Classes | Train Images | Test Images | Description |
|---------|---------|--------------|-------------|-------------|
| `JONATHAN_MIX` | 6 | 9,600 (1,600/class) | 2,400 (400/class) | Mixed imaging modality |
| `JONATHAN_SEC` | 6 | 4,800 (800/class) | 1,200 (200/class) | Section imaging |
| `JONATHAN_SUR` | 6 | 4,800 (800/class) | 1,200 (200/class) | Surface imaging |

**Classes**: 
- MIX/SEC/SUR-Subtype_Ia
- MIX/SEC/SUR-Subtype_IIa
- MIX/SEC/SUR-Subtype_IIIa
- MIX/SEC/SUR-Subtype_IVc
- MIX/SEC/SUR-Subtype_IVd
- MIX/SEC/SUR-Subtype_Va

### 2. Michel Daudon Collection
**Base Path**: `data/Michel Daudon (w256 1k v1)/`

| Dataset | Classes | Train Images | Test Images | Description |
|---------|---------|--------------|-------------|-------------|
| `DAUDON_MIX` | 6 | 9,600 (1,600/class) | 2,400 (400/class) | Mixed imaging modality |
| `DAUDON_SEC` | 6 | 4,800 (800/class) | 1,200 (200/class) | Section imaging |
| `DAUDON_SUR` | 6 | 4,800 (800/class) | 1,223 (variable/class) | Surface imaging |

**Classes**:
- MIX/SEC/SUR-Subtype_Ia
- MIX/SEC/SUR-Subtype_IVa
- MIX/SEC/SUR-Subtype_IVa2
- MIX/SEC/SUR-Subtype_IVc
- MIX/SEC/SUR-Subtype_IVd
- MIX/SEC/SUR-Subtype_Va

### 3. MyStone Collection
**Base Path**: `data/MyStone (w256, 1k)/`

| Dataset | Classes | Train Images | Test Images | Description |
|---------|---------|--------------|-------------|-------------|
| `MYSTONE` | 6 | 4,800 (800/class) | 1,200 (200/class) | General kidney stone dataset |

**Classes**:
- 1_Subtype_COM_WW_Ia
- 2_Subtype_COD_WD_IIa
- 4_Subtype_HAP
- 5_Subtype_STR_IVc
- 6_Subtype_BRU_IVd
- 7_Subtype_UA_IIIa

## 🔧 Technical Specifications

### Image Properties
- **Resolution**: 256×256 pixels (native, no resizing)
- **Format**: PNG (Jonathan & Daudon), JPG (MyStone)
- **Channels**: 3 (RGB)
- **Normalization**: ImageNet standard
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Model Configuration
For these 256×256 images, the ResNet architecture uses:
- **Original conv1**: 7×7 kernel, stride 2 (kept as-is)
- **MaxPool**: stride 2 (kept as-is)
- **Pretrained weights**: Can use ImageNet pretrained (recommended!)

Unlike MNIST/CIFAR, these images are large enough to use standard ResNet without modifications.

## 🚀 Usage

### Single Experiment

```python
# In config.py
DATASET = "JONATHAN_MIX"  # or any other kidney stone dataset
MODEL_NAME = "resnet18"
USE_PRETRAINED = True  # Recommended for 256x256 images
NUM_CLIENTS = 5
NUM_ROUNDS = 10
```

```bash
python main.py
```

### Batch Experiments

Test all kidney stone datasets:
```bash
python run_full_experiments.py --datasets JONATHAN_MIX JONATHAN_SEC JONATHAN_SUR DAUDON_MIX DAUDON_SEC DAUDON_SUR MYSTONE --strategies FedAvg FedProx
```

Test specific collection:
```bash
# Jonathan El-Beze collection only
python run_full_experiments.py --datasets JONATHAN_MIX JONATHAN_SEC JONATHAN_SUR

# Daudon collection only
python run_full_experiments.py --datasets DAUDON_MIX DAUDON_SEC DAUDON_SUR

# MyStone only
python run_full_experiments.py --datasets MYSTONE
```

## 📈 Expected Performance

### Training Speed (ResNet18, 5 clients, 1 epoch)
- **Per round**: ~30-60 seconds (depends on GPU)
- **Much faster than 224×224 CIFAR** due to better GPU utilization with 256×256

### Accuracy Expectations
- **Baseline (Random)**: ~16.7% (6 classes)
- **FedAvg (10 rounds)**: 60-80% (depends on dataset and non-IID setting)
- **Best strategies**: FedProx, FedAgain typically perform best on medical images

## 🔬 Dataset Characteristics

### Modality Differences
- **MIX**: Combined section + surface views → More diverse, harder classification
- **SEC**: Section views only → More uniform, clearer internal structure
- **SUR**: Surface views only → More consistent, external morphology

### Collection Differences
- **Jonathan El-Beze**: Consistent labeling across subtypes IIa, IIIa
- **Michel Daudon**: Different subtype IVa variants (IVa, IVa2)
- **MyStone**: Different naming convention, focused on specific compounds

## 🧪 Federated Learning Scenarios

### Scenario 1: Multi-Center Study
```python
# Each hospital (client) has a different dataset
DATA_DISTRIBUTION = "iid"  # Within same dataset
DATASET = "JONATHAN_MIX"
NUM_CLIENTS = 5  # 5 hospitals
```

### Scenario 2: Multi-Modality Fusion
```python
# Some clients have SEC, others have SUR
# Would require custom data loading - future work
```

### Scenario 3: Cross-Dataset Generalization
```python
# Train on Jonathan, test on Daudon
# Would require custom evaluation - future work
```

## 📝 Notes

1. **Class Imbalance**: DAUDON_SUR test set has 1,223 images (slight imbalance)
2. **Different Subtypes**: Jonathan vs Daudon have different subtype labels
3. **File Formats**: Mixed PNG and JPG (handled automatically by torchvision)
4. **Use Pretrained**: For 256×256 images, ImageNet pretrained weights help significantly

## 🎯 Recommended Configuration

For kidney stone experiments:
```python
# config.py
DATASET = "JONATHAN_MIX"  # Start with largest dataset
MODEL_NAME = "resnet18"  # Or resnet50 for better accuracy
USE_PRETRAINED = True  # Important for medical images!
NUM_CLIENTS = 5
NUM_ROUNDS = 15  # Medical images need more rounds
LOCAL_EPOCHS = 3
BATCH_SIZE = 16  # Smaller for 256x256 images
LEARNING_RATE = 0.001  # Lower for pretrained models
DATA_DISTRIBUTION = "non-iid-label"  # Realistic federated scenario
NON_IID_CLASSES_PER_CLIENT = 2  # Each center specializes in some subtypes
```

## 📚 Citation

If you use these datasets, please cite the original sources:
- Jonathan El-Beze collection: [Add citation]
- Michel Daudon collection: [Add citation]
- MyStone collection: [Add citation]

