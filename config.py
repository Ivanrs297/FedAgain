"""
Configuration file for federated learning experiments.
Modify these settings to change dataset, model, and training parameters.
"""

# Dataset Configuration
DATASET = "MNIST"  # Options: "EMNIST", "MNIST", "CIFAR10"

# Federated Learning Configuration
STRATEGY = "Bulyan"  # Options: "FedAvg", "FedProx", "FedAdam", "FedMedian", "Bulyan", "FedAgain"
NUM_CLIENTS = 5
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1

# Data Distribution Configuration
DATA_DISTRIBUTION = "non-iid-label"  # Options: "iid", "non-iid-label", "non-iid-dirichlet"
NON_IID_CLASSES_PER_CLIENT = 2  # For non-iid-label: how many PRIMARY classes each client specializes in (they still get small samples of other classes)
DIRICHLET_ALPHA = 0.5  # For non-iid-dirichlet: concentration parameter (lower = more non-IID)

# Strategy-specific parameters
FEDPROX_PROXIMAL_MU = 0.1  # Proximal term coefficient for FedProx
FEDADAM_SERVER_LR = 1.0  # Server learning rate for FedAdam (1.0 = full update)
FEDADAM_BETA1 = 0.9  # FedAdam momentum parameter
FEDADAM_BETA2 = 0.999  # FedAdam second moment parameter (higher = more stable)
FEDADAM_TAU = 1e-8  # FedAdam adaptive learning rate parameter
BULYAN_NUM_MALICIOUS = 1  # Bulyan: number of Byzantine/malicious clients to tolerate
BULYAN_TO_KEEP = None  # Bulyan: number of clients to keep (None = auto)
FEDAGAIN_EPSILON = 1e-6  # FedAgain: small constant for numerical stability in trust weight

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Model Configuration
USE_PRETRAINED = True

# Device Configuration (auto-detected, but can be overridden)
# DEVICE = "cuda"  # Options: "cuda", "cpu", or leave commented for auto-detection

