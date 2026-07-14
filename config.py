"""
Configuration file for federated learning experiments.
Modify these settings to change dataset, model, and training parameters.
"""

# Dataset Configuration
# Options: "MNIST", "EMNIST", "CIFAR10", "CIFAR100"
# Kidney Stones: "JONATHAN_MIX", "JONATHAN_SEC", "JONATHAN_SUR", "DAUDON_MIX", "DAUDON_SEC", "DAUDON_SUR", "MYSTONE"
DATASET = "DAUDON_MIX"

# Federated Learning Configuration
STRATEGY = "FedMedian"
NUM_CLIENTS = 20
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1

# Data Distribution Configuration
DATA_DISTRIBUTION = "iid"
NON_IID_CLASSES_PER_CLIENT = 2  # For non-iid-label: how many PRIMARY classes each client specializes in (they still get small samples of other classes)
DIRICHLET_ALPHA = 0.5  # For non-iid-dirichlet: concentration parameter (lower = more non-IID)

# Corrupted Clients Configuration (Simulate realistic data quality issues)
CORRUPT_CLIENTS = True
CORRUPT_CLIENT_RATIO = 0.3
CORRUPTION_SEVERITY = 3
# Note: Corruptions are applied only to corrupted clients during training
#       Global model is tested on clean (benign) images

# Strategy-specific parameters
FEDPROX_PROXIMAL_MU = 0.1  # Proximal term coefficient for FedProx
FEDYOGI_SERVER_LR = 0.001  # Server learning rate for FedYogi (very small for stability with deep networks)
FEDYOGI_BETA1 = 0.9  # FedYogi momentum parameter (first moment decay)
FEDYOGI_BETA2 = 0.99  # FedYogi second moment parameter (velocity decay)
FEDYOGI_TAU = 1e-2  # FedYogi numerical stability constant (larger for deep networks)
BULYAN_NUM_MALICIOUS = 1  # Bulyan: number of Byzantine/malicious clients to tolerate
BULYAN_TO_KEEP = None  # Bulyan: number of clients to keep (None = auto)
FEDAGAIN_EPSILON = 1e-6  # FedAgain: small constant for numerical stability in trust weight
# FedAgain trust formulation (experiments #2 ablation and #3 sensitivity):
#   "combined_mult"     -> T = 1/(E*D + eps)          (default)
#   "benchmark_only"    -> T = 1/(E + eps)            (ablation: E only)
#   "divergence_only"   -> T = 1/(D + eps)            (ablation: D only)
#   "additive"          -> T = 1/(E + D + eps)
#   "weighted_additive" -> T = 1/(alpha*E + beta*D + eps)
FEDAGAIN_TRUST_MODE = "combined_mult"
FEDAGAIN_ALPHA = 1.0  # weight on benchmark error E_k (weighted_additive only)
FEDAGAIN_BETA = 1.0   # weight on divergence D_k (weighted_additive only)
FEDAGAIN_LOG_TRUST = False  # dump per-round trust scores to results/trust_scores/

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Model Configuration
MODEL_NAME = "resnet18"
USE_PRETRAINED = True  # For small images (28x28, 32x32), train from scratch is better

# Device Configuration (auto-detected, but can be overridden)
# DEVICE = "cuda"  # Options: "cuda", "cpu", or leave commented for auto-detection

