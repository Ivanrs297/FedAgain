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
POISON_MODE = "image"  # "image" | "label_flip"
LABEL_FLIP_RATIO = 1.0  # fraction of a poisoned client's labels to flip
# Data version: "v1" = released folders (results/), "v2" = specimen-disjoint
# splits + MyStone patches only + KIDNEY_MULTISITE (results_v2/). The env var
# FEDAGAIN_DATA_VERSION overrides this. See src/dataset.py.
DATA_VERSION = "v1"
# Note: Corruptions are applied only to corrupted clients during training
#       Global model is tested on clean (benign) images

# Strategy-specific parameters
FEDPROX_PROXIMAL_MU = 0.1  # Proximal term coefficient for FedProx
FEDYOGI_SERVER_LR = 0.001  # Server learning rate for FedYogi (very small for stability with deep networks)
FEDYOGI_BETA1 = 0.9  # FedYogi momentum parameter (first moment decay)
FEDYOGI_BETA2 = 0.99  # FedYogi second moment parameter (velocity decay)
FEDYOGI_TAU = 1e-2  # FedYogi numerical stability constant (larger for deep networks)
MOON_MU = 1.0  # MOON: weight of the model-contrastive loss term
MOON_TEMPERATURE = 0.5  # MOON: temperature for the contrastive loss
BULYAN_NUM_MALICIOUS = 1  # Bulyan: number of Byzantine/malicious clients to tolerate
BULYAN_TO_KEEP = None  # Bulyan: number of clients to keep (None = auto)
FEDDYN_ALPHA = 1.0  # FedDyn dynamic-reg coeff. 1.0 (not the paper default 0.1) for stability here: 10 clients, full participation, pretrained ResNet18, 10 rounds -- small alpha amplifies the server drift h by 1/alpha and diverges (loss blowup); alpha=1.0 is within the paper's tested range and converges cleanly.
FEDDYN_CLIP = 10.0  # FedDyn: gradient-norm clip for local training stability
# Reference-based robust baselines (need a clean server root set; see load_data).
SERVER_ROOT_PER_CLASS = 20  # samples/class carved for RFLPA/BOBA server reference (0=off)
RFLPA_SERVER_EPOCHS = 1  # RFLPA/FLTrust: local epochs to train the server reference update
# BOBA (Bao et al., AISTATS 2024): Byzantine-robust under label skew.
BOBA_NUM_BYZ = 2  # f: Byzantine budget (trims f, keeps n-f); need n-f > num_classes-1
BOBA_PMIN = -0.5  # reject client if min barycentric coord < pmin (must be <= 0)
BOBA_MAX_ITER = 20  # stage-1 alternating-minimization iterations
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

# Normalization mapping raw trust scores to aggregation weights (Reviewer 2.5).
#   "sum_to_one" -> w = T / sum(T)                    (default / deployed)
#   "softmax"    -> w = softmax(log T / tau)          (tau = FEDAGAIN_NORM_TEMPERATURE;
#                   note tau=1 is mathematically identical to sum_to_one)
#   "rank"       -> w proportional to the rank of T   (magnitude discarded)
FEDAGAIN_NORM_MODE = "sum_to_one"
FEDAGAIN_NORM_TEMPERATURE = 1.0
# Trust/data-size mixing (round 2, multi-site domain shift):
#   w = (1 - lambda) * w_trust + lambda * n_k / n.  0 = plain FedAgain.
FEDAGAIN_MIX_LAMBDA = 0.0

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_WORKERS = 8  # DataLoader worker processes. Parallelizes the CPU-bound
                 # on-the-fly image corruption. Clients train sequentially, so
                 # only one loader's workers are active at a time.
SEED = 42  # Global seed: controls partitioning, corrupted-client selection,
           # per-client corruption, and training RNG. Vary (e.g. 42/43/44)
           # across repeated runs for variance/stability estimates.

# Model Configuration
MODEL_NAME = "resnet18"
USE_PRETRAINED = True  # For small images (28x28, 32x32), train from scratch is better

# Device Configuration (auto-detected, but can be overridden)
# DEVICE = "cuda"  # Options: "cuda", "cpu", or leave commented for auto-detection

