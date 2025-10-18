"""
Configuration file for federated learning experiments.
Modify these settings to change dataset, model, and training parameters.
"""

# Dataset Configuration
DATASET = "MNIST"  # Options: "EMNIST", "MNIST", "CIFAR10"

# Federated Learning Configuration
STRATEGY = "Bulyan"  # Options: "FedAvg", "FedProx", "FedOpt", "FedMedian", "Bulyan"
NUM_CLIENTS = 5
NUM_ROUNDS = 1
LOCAL_EPOCHS = 1

# Data Distribution Configuration
DATA_DISTRIBUTION = "non-iid-label"  # Options: "iid", "non-iid-label", "non-iid-dirichlet"
NON_IID_CLASSES_PER_CLIENT = 2  # For non-iid-label: how many PRIMARY classes each client specializes in (they still get small samples of other classes)
DIRICHLET_ALPHA = 0.5  # For non-iid-dirichlet: concentration parameter (lower = more non-IID)

# Strategy-specific parameters
FEDPROX_PROXIMAL_MU = 0.1  # Proximal term coefficient for FedProx
FEDOPT_SERVER_LR = 0.01  # Server learning rate for FedOpt (FedAdam) - much smaller than 1.0
FEDOPT_BETA1 = 0.9  # FedOpt momentum parameter
FEDOPT_BETA2 = 0.99  # FedOpt second moment parameter
FEDOPT_TAU = 1e-3  # FedOpt adaptive learning rate parameter
BULYAN_NUM_MALICIOUS = 1  # Bulyan: number of Byzantine/malicious clients to tolerate
BULYAN_TO_KEEP = None  # Bulyan: number of clients to keep (None = auto)

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Model Configuration
USE_PRETRAINED = True

# Device Configuration (auto-detected, but can be overridden)
# DEVICE = "cuda"  # Options: "cuda", "cpu", or leave commented for auto-detection

