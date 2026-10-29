"""
Federated learning with multiple aggregation strategies.
Simple implementation without Ray/TensorFlow dependencies.
Supports FedAvg, FedProx, FedYogi, FedMedian, Bulyan, FedBN, Scaffold, MOON,
FedDyn, RFLPA, BOBA, and FedAgain.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
from collections import OrderedDict
from src.model import get_model
from src.dataset import load_data, data_version
from src.client import train, test, train_moon, train_scaffold, train_feddyn
from src.strategies.fedagain import FedAgainStrategy, compute_benchmark_error
from src.logger import ExperimentLogger, compute_classification_metrics
from src.tracking import ExperimentTracker
import config
import numpy as np
import json

# Per-run overrides without rewriting config.py (used by the unattended queue in
# scripts/round2/): FEDAGAIN_CONFIG_OVERRIDES='{"STRATEGY": "FedAvg", ...}'.
# src/dataset.py reads the same module object, so it sees them too.
for _k, _v in json.loads(os.environ.get("FEDAGAIN_CONFIG_OVERRIDES", "{}")).items():
    if not hasattr(config, _k):
        raise KeyError(f"FEDAGAIN_CONFIG_OVERRIDES: unknown config key {_k}")
    setattr(config, _k, _v)

# Each data version keeps its own results tree, so runs on the v2
# (specimen-disjoint) data can never be mixed with, or skipped because of, v1 runs.
DATA_VERSION = data_version()
RESULTS_DIR = os.environ.get("FEDAGAIN_RESULTS_DIR") or (
    "results" if DATA_VERSION == "v1" else f"results_{DATA_VERSION}")


# Global variables for data loaders (shared across clients)
train_loaders = None
val_loaders = None
test_loader = None
dataset_info = None

# FedBN: indices (in state_dict order) of BatchNorm parameters that are kept
# local per client, and the per-client BatchNorm state persisted across rounds.
bn_param_indices = None
client_bn_state = {}

# MOON: per-client previous-round model parameters (for the contrastive term).
client_prev_state = {}

# SCAFFOLD: global control variate, per-client control variates, and the
# per-round accumulator of client control-variate deltas.
scaffold_c_global = None
scaffold_c_local_state = {}
scaffold_round_deltas = []

# FedDyn: per-client persisted gradient state g_k {name: tensor} and the server
# drift accumulator h {name: tensor}; both init lazily to zeros (Acar et al.).
feddyn_g_local = {}
feddyn_h = None
feddyn_trainable_idx = None


def compute_bn_param_indices(model):
    """Return state_dict positions of all BatchNorm parameters/buffers."""
    keys = list(model.state_dict().keys())
    bn_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for pn, _ in module.named_parameters(recurse=False):
                bn_keys.add(f"{name}.{pn}")
            for bn, _ in module.named_buffers(recurse=False):
                bn_keys.add(f"{name}.{bn}")
    return [i for i, k in enumerate(keys) if k in bn_keys]


def compute_trainable_param_indices(model):
    """Return the set of state_dict positions that are trainable parameters
    (i.e. in named_parameters, excluding buffers like BN running stats).

    FedDyn's dynamic regularizer and drift accumulator act on trainable
    parameters only; non-parameter buffers are averaged plainly (FedAvg-style).
    """
    keys = list(model.state_dict().keys())
    param_names = {n for n, _ in model.named_parameters()}
    return {i for i, k in enumerate(keys) if k in param_names}


def compute_server_reference_update(global_params, ref_model, root_loader,
                                    device, lr, momentum, target_steps):
    """Train ``ref_model`` from ``global_params`` on the clean server root set for
    ``target_steps`` SGD steps (cycling the small root loader) and return the
    server reference update ``g0 = w_ref - w_global`` (list of arrays).

    Used by RFLPA (= FLTrust defense core): the reference direction against which
    client updates are scored by ReLU'd cosine similarity. FLTrust assumes the
    server performs comparable local computation to the clients, so the number of
    optimizer STEPS is matched to a client's (not the number of epochs) -- the
    small root set otherwise yields a tiny-magnitude update and the norm-clipped
    aggregate barely moves the global model.
    """
    import torch.nn as nn
    keys = list(ref_model.state_dict().keys())
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, global_params)})
    ref_model.load_state_dict(state, strict=True)
    ref_model.to(device)
    ref_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr, momentum=momentum)
    steps = 0
    while steps < target_steps:
        for images, labels in root_loader:
            if steps >= target_steps:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(ref_model(images), labels)
            loss.backward()
            optimizer.step()
            steps += 1
    ref_params = [p.detach().cpu().numpy() for p in ref_model.state_dict().values()]
    return [ref_params[i] - global_params[i] for i in range(len(global_params))]


def boba_aggregate(G, S, f, c, pmin=-0.5, max_iter=20):
    """BOBA server aggregation (Bao et al., AISTATS 2024).

    Byzantine-robust and unbiased under label skew. Honest client updates lie near
    the (c-1)-dim affine "honest simplex" spanned by the c per-class server
    vertices. Stage 1 robustly fits that subspace (trimmed truncated-SVD); stage 2
    filters clients whose barycentric coordinates fall off the simplex, then
    averages the projected survivors.

    Args:
        G: (n, d) client update/param vectors.
        S: (c, d) server per-class vertex vectors.
        f: Byzantine budget (upper bound); trims f, keeps n-f.
        c: number of classes.
        pmin: reject client if min_z barycentric coord < pmin (<= 0).
        max_iter: stage-1 alternating-minimization iterations.

    Returns:
        (mu, info) where mu is the (d,) aggregate and info is a small dict for logging.
    """
    from sklearn.decomposition import PCA
    n = G.shape[0]
    # Structural feasibility clamps: need >= c kept points to fit a (c-1)-dim
    # affine subspace. With few clients relative to classes (n <= c, e.g. 10
    # clients on CIFAR-10) trimming is disabled and the subspace dim is reduced.
    f = int(max(0, min(f, n - c)))
    k = n - f
    ncomp = int(min(c - 1, k - 1))
    if ncomp < 1:
        # Degenerate (too few clients/classes) -> fall back to plain mean.
        return G.mean(axis=0), {"kept": n, "ncomp": 0, "degenerate": True}

    pca = PCA(n_components=ncomp)
    # ---- Stage 1: robust affine-subspace fitting ----
    pca.fit(S)
    kept = np.arange(n)
    for _ in range(max_iter):
        Z = pca.transform(G)
        rec = pca.inverse_transform(Z)
        res = ((G - rec) ** 2).sum(axis=1)
        new = np.argsort(res)[:k]
        if set(new.tolist()) == set(kept.tolist()):
            kept = new
            break
        kept = new
        pca.fit(G[kept])

    # ---- Stage 2: barycentric filter on the honest simplex ----
    Zg = pca.transform(G)                      # (n, ncomp)
    Zs = pca.transform(S)                      # (c, ncomp)
    Zg1 = np.concatenate([Zg, np.ones((n, 1))], axis=1)   # (n, ncomp+1)
    Zs1 = np.concatenate([Zs, np.ones((c, 1))], axis=1)   # (c, ncomp+1)
    P = Zg1 @ np.linalg.pinv(Zs1)              # (n, c) barycentric coords
    low = P.min(axis=1)                        # min_z p_iz
    order = np.argsort(low)
    if low[order[-k]] > pmin:                  # enough clients clear the threshold
        sel = np.where(low >= pmin)[0]
    else:                                      # fallback: top-k by min-coord
        sel = order[-k:]

    mu_latent = Zg[sel].mean(axis=0)
    mu = pca.inverse_transform(mu_latent[None, :])[0]
    return mu, {"kept": int(len(sel)), "ncomp": ncomp, "f": f, "degenerate": False}


class FederatedClient:
    """Simple federated client for local training and evaluation."""
    
    def __init__(self, cid: str, device: torch.device):
        self.cid = cid
        self.device = device
        
        # Create model
        # Adapt for small images if input size < 64
        adapt_small = dataset_info['input_size'] < 64
        self.model = get_model(
            num_classes=dataset_info['num_classes'],
            num_channels=dataset_info['num_channels'],
            use_pretrained=config.USE_PRETRAINED,
            model_name=config.MODEL_NAME,
            adapt_for_small_images=adapt_small
        )
        self.model.to(self.device)
        
        # Get data loaders for this client
        self.trainloader = train_loaders[int(cid)]
        self.valloader = val_loaders[int(cid)]
    
    def get_parameters(self, config_dict):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config_dict):
        """Train the model on local data."""
        # FedBN: substitute this client's own BatchNorm parameters (local BN);
        # only the non-BN parameters are shared/aggregated across clients.
        if config.STRATEGY == "FedBN" and bn_param_indices:
            parameters = list(parameters)
            stored = client_bn_state.get(self.cid)
            if stored is not None:
                for idx, val in zip(bn_param_indices, stored):
                    parameters[idx] = val
        self.set_parameters(parameters)
        
        # Compute benchmark error on incoming global model (for FedAgain)
        benchmark_error = 0.0
        if config.STRATEGY == "FedAgain":
            # Use validation set as benchmark
            benchmark_error = compute_benchmark_error(self.model, self.valloader, self.device)
        
        # FedProx: Add proximal term
        proximal_mu = 0.0
        global_params = None
        if config.STRATEGY == "FedProx":
            proximal_mu = config.FEDPROX_PROXIMAL_MU
            global_params = parameters  # Store global model parameters
        
        # Train locally (strategy-specific local objective).
        if config.STRATEGY == "MOON":
            import copy
            global_net = copy.deepcopy(self.model)   # the just-set global model
            prev_net = copy.deepcopy(self.model)
            prev = client_prev_state.get(self.cid)
            if prev is not None:
                pd = zip(prev_net.state_dict().keys(), prev)
                prev_net.load_state_dict(
                    OrderedDict({k: torch.tensor(v) for k, v in pd}), strict=True)
            train_loss, train_acc = train_moon(
                self.model, self.trainloader, config.LOCAL_EPOCHS, self.device,
                config.LEARNING_RATE, config.MOMENTUM, global_net, prev_net,
                mu=getattr(config, "MOON_MU", 1.0),
                temperature=getattr(config, "MOON_TEMPERATURE", 0.5))
        elif config.STRATEGY == "Scaffold":
            c_local = scaffold_c_local_state.get(self.cid)
            if c_local is None:
                c_local = {n: torch.zeros_like(p)
                           for n, p in self.model.named_parameters()}
            c_global = scaffold_c_global if scaffold_c_global is not None else \
                {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
            train_loss, train_acc, new_c_local, delta_c = train_scaffold(
                self.model, self.trainloader, config.LOCAL_EPOCHS, self.device,
                config.LEARNING_RATE, config.MOMENTUM, c_global, c_local)
            scaffold_c_local_state[self.cid] = new_c_local
            scaffold_round_deltas.append(delta_c)
        elif config.STRATEGY == "FedDyn":
            server_params = {n: p.detach().clone()
                             for n, p in self.model.named_parameters()}
            g_local = feddyn_g_local.get(self.cid)
            if g_local is None:
                g_local = {n: torch.zeros_like(p)
                           for n, p in self.model.named_parameters()}
            train_loss, train_acc, new_g_local = train_feddyn(
                self.model, self.trainloader, config.LOCAL_EPOCHS, self.device,
                config.LEARNING_RATE, config.MOMENTUM, g_local, server_params,
                alpha=getattr(config, "FEDDYN_ALPHA", 0.1),
                clip=getattr(config, "FEDDYN_CLIP", 10.0))
            feddyn_g_local[self.cid] = new_g_local
        else:
            train_loss, train_acc = train(
                self.model,
                self.trainloader,
                epochs=config.LOCAL_EPOCHS,
                device=self.device,
                lr=config.LEARNING_RATE,
                momentum=config.MOMENTUM,
                proximal_mu=proximal_mu,
                global_params=global_params
            )
        
        print(f"  Client {int(self.cid) + 1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}, BenchErr={benchmark_error:.4f}")
        
        # FedBN: remember this client's BatchNorm parameters for the next round.
        if config.STRATEGY == "FedBN" and bn_param_indices:
            full = self.get_parameters(config_dict={})
            client_bn_state[self.cid] = [full[idx] for idx in bn_param_indices]

        # MOON: remember this client's post-training model for the next round.
        if config.STRATEGY == "MOON":
            client_prev_state[self.cid] = self.get_parameters(config_dict={})

        # Return parameters with benchmark error in metrics
        metrics = {"benchmark_error": benchmark_error} if config.STRATEGY == "FedAgain" else {}
        return self.get_parameters(config_dict={}), len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters, config_dict):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str) -> FederatedClient:
    """Create a federated client for the given client ID."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return FederatedClient(cid, device)


def evaluate_global_model(server_round: int, parameters, config_dict, 
                         return_predictions: bool = False):
    """Evaluate the global model on the centralized test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model and set parameters
    # Adapt for small images if input size < 64
    adapt_small = dataset_info['input_size'] < 64
    model = get_model(
        num_classes=dataset_info['num_classes'],
        num_channels=dataset_info['num_channels'],
        use_pretrained=config.USE_PRETRAINED,
        model_name=config.MODEL_NAME,
        adapt_for_small_images=adapt_small
    )
    model.to(device)
    
    # Set parameters with proper dtype handling
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    # Evaluate
    if return_predictions:
        # Get predictions for metrics computation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_labels), np.array(all_preds)
    else:
        loss, accuracy = test(model, test_loader, device=device)
        print(f"\n[Round {server_round}] Global Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, {"accuracy": accuracy}


if __name__ == "__main__":
    # Initialize experiment logger
    logger = ExperimentLogger(results_dir=RESULTS_DIR)

    # Build the experiment identity and resilient tracker. It writes results
    # incrementally (per round) and lets already-completed runs be skipped on
    # resume, which matters for long, failure-prone experiment sweeps.
    is_fedagain = config.STRATEGY == "FedAgain"
    uses_root = config.STRATEGY in ("RFLPA", "BOBA")
    tracker_config = {
        "model": config.MODEL_NAME.upper(),
        "strategy": config.STRATEGY,
        "dataset": config.DATASET,
        "data_distribution": config.DATA_DISTRIBUTION,
        "num_clients": config.NUM_CLIENTS,
        "num_rounds": config.NUM_ROUNDS,
        "local_epochs": config.LOCAL_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "use_pretrained": config.USE_PRETRAINED,
        "corrupt_clients": getattr(config, "CORRUPT_CLIENTS", False),
        "corrupt_ratio": getattr(config, "CORRUPT_CLIENT_RATIO", 0.0),
        "corruption_severity": getattr(config, "CORRUPTION_SEVERITY", 0),
        "poison_mode": getattr(config, "POISON_MODE", "image"),
        "trust_mode": getattr(config, "FEDAGAIN_TRUST_MODE", "combined_mult") if is_fedagain else "",
        "trust_alpha": getattr(config, "FEDAGAIN_ALPHA", 1.0) if is_fedagain else "",
        "trust_beta": getattr(config, "FEDAGAIN_BETA", 1.0) if is_fedagain else "",
        "epsilon": getattr(config, "FEDAGAIN_EPSILON", 1e-6) if is_fedagain else "",
        "norm_mode": getattr(config, "FEDAGAIN_NORM_MODE", "sum_to_one") if is_fedagain else "",
        "norm_temperature": getattr(config, "FEDAGAIN_NORM_TEMPERATURE", 1.0) if is_fedagain else "",
        "mix_lambda": getattr(config, "FEDAGAIN_MIX_LAMBDA", 0.0) if is_fedagain else "",
        "root_per_class": getattr(config, "SERVER_ROOT_PER_CLASS", 20) if uses_root else "",
        "root_mode": getattr(config, "SERVER_ROOT_MODE", "clean") if uses_root else "",
        "root_site": getattr(config, "SERVER_ROOT_SITE", "") if uses_root else "",
        "seed": getattr(config, "SEED", 42),
        "data_version": DATA_VERSION,
    }
    tracker = ExperimentTracker(tracker_config, results_dir=RESULTS_DIR)
    if tracker.is_completed() and os.environ.get("EXPERIMENT_FORCE") != "1":
        print(f"[tracker] run '{tracker.run_key}' already completed -> "
              f"skipping (set EXPERIMENT_FORCE=1 to force a rerun).")
        sys.exit(0)

    # Seed the global RNGs for reproducible training (partitioning and
    # corruption are additionally seeded inside load_data).
    seed = getattr(config, "SEED", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {config.DATASET}")
    print(f"Model: {config.MODEL_NAME.upper()}")
    print(f"Strategy: {config.STRATEGY}")
    print(f"Num Clients: {config.NUM_CLIENTS}")
    print(f"Num Rounds: {config.NUM_ROUNDS}")
    print(f"Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading and partitioning {config.DATASET} dataset...")
    print(f"Number of clients: {config.NUM_CLIENTS}")
    
    # Load and partition data
    train_loaders, val_loaders, test_loader, dataset_info = load_data(
        dataset_name=config.DATASET,
        num_clients=config.NUM_CLIENTS,
        batch_size=config.BATCH_SIZE,
        distribution=config.DATA_DISTRIBUTION,
        classes_per_client=config.NON_IID_CLASSES_PER_CLIENT,
        dirichlet_alpha=config.DIRICHLET_ALPHA,
        seed=seed,
        num_workers=getattr(config, "NUM_WORKERS", 0),
        server_root_per_class=(getattr(config, "SERVER_ROOT_PER_CLASS", 20)
                               if config.STRATEGY in ("RFLPA", "BOBA") else 0),
        server_root_mode=getattr(config, "SERVER_ROOT_MODE", "clean"),
        server_root_site=getattr(config, "SERVER_ROOT_SITE", ""),
        server_root_severity=getattr(config, "CORRUPTION_SEVERITY", 3),
    )

    print(f"\nDataset Information:")
    print(f"  - Classes: {dataset_info['num_classes']}")
    print(f"  - Channels: {dataset_info['num_channels']}")
    print(f"  - Input size: {dataset_info['input_size']}x{dataset_info['input_size']}")
    print(f"  - Training samples per client: ~{len(train_loaders[0].dataset)}")
    print(f"  - Total test samples: {len(test_loader.dataset)}")
    
    print(f"\nTraining Configuration:")
    print(f"  - Pretrained: {config.USE_PRETRAINED}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Momentum: {config.MOMENTUM}")
    
    # Strategy-specific parameters
    if config.STRATEGY == "FedProx":
        print(f"  - FedProx mu: {config.FEDPROX_PROXIMAL_MU}")
    elif config.STRATEGY == "FedYogi":
        print(f"  - Server LR: {config.FEDYOGI_SERVER_LR}")
        print(f"  - Beta1: {config.FEDYOGI_BETA1}, Beta2: {config.FEDYOGI_BETA2}")
    elif config.STRATEGY == "Bulyan":
        print(f"  - Byzantine clients to tolerate: {config.BULYAN_NUM_MALICIOUS}")
    
    # Initialize strategy-specific components
    fedyogi_m = None
    fedyogi_v = None
    fedyogi_t = 0
    fedagain_strategy = None
    
    if config.STRATEGY == "FedYogi":
        # Will be initialized after first round
        pass
    elif config.STRATEGY == "FedAgain":
        # Initialize FedAgain strategy (trust formulation configurable for the
        # ablation (#2) and multiplicative-vs-additive (#3) experiments).
        epsilon = getattr(config, 'FEDAGAIN_EPSILON', 1e-6)
        trust_mode = getattr(config, 'FEDAGAIN_TRUST_MODE', 'combined_mult')
        alpha = getattr(config, 'FEDAGAIN_ALPHA', 1.0)
        beta = getattr(config, 'FEDAGAIN_BETA', 1.0)
        log_trust = getattr(config, 'FEDAGAIN_LOG_TRUST', False)
        norm_mode = getattr(config, 'FEDAGAIN_NORM_MODE', 'sum_to_one')
        norm_temp = getattr(config, 'FEDAGAIN_NORM_TEMPERATURE', 1.0)
        mix_lambda = getattr(config, 'FEDAGAIN_MIX_LAMBDA', 0.0)
        fedagain_strategy = FedAgainStrategy(
            epsilon=epsilon, mode=trust_mode, alpha=alpha, beta=beta,
            log_trust=log_trust, norm_mode=norm_mode,
            norm_temperature=norm_temp, mix_lambda=mix_lambda,
        )
        if mix_lambda > 0:
            print(f"  - FedAgain trust/data-size mixing: lambda={mix_lambda}")
        print(f"  - FedAgain epsilon: {epsilon}")
        print(f"  - FedAgain normalization: {norm_mode}"
              + (f" (tau={norm_temp})" if norm_mode == "softmax" else ""))
        print(f"  - FedAgain trust mode: {trust_mode}"
              + (f" (alpha={alpha}, beta={beta})"
                 if trust_mode == "weighted_additive" else ""))
    elif config.STRATEGY == "FedBN":
        adapt_small = dataset_info['input_size'] < 64
        _tmp = get_model(num_classes=dataset_info['num_classes'],
                         num_channels=dataset_info['num_channels'],
                         use_pretrained=config.USE_PRETRAINED,
                         model_name=config.MODEL_NAME,
                         adapt_for_small_images=adapt_small)
        bn_param_indices = compute_bn_param_indices(_tmp)
        del _tmp
        print(f"  - FedBN: {len(bn_param_indices)} BatchNorm tensors kept local per client")
    elif config.STRATEGY == "FedDyn":
        adapt_small = dataset_info['input_size'] < 64
        _tmp = get_model(num_classes=dataset_info['num_classes'],
                         num_channels=dataset_info['num_channels'],
                         use_pretrained=config.USE_PRETRAINED,
                         model_name=config.MODEL_NAME,
                         adapt_for_small_images=adapt_small)
        feddyn_trainable_idx = compute_trainable_param_indices(_tmp)
        del _tmp
        print(f"  - FedDyn: alpha={getattr(config, 'FEDDYN_ALPHA', 0.1)}, "
              f"{len(feddyn_trainable_idx)} trainable tensors (BN buffers averaged)")
    elif config.STRATEGY in ("RFLPA", "BOBA"):
        adapt_small = dataset_info['input_size'] < 64
        server_ref_model = get_model(num_classes=dataset_info['num_classes'],
                                     num_channels=dataset_info['num_channels'],
                                     use_pretrained=config.USE_PRETRAINED,
                                     model_name=config.MODEL_NAME,
                                     adapt_for_small_images=adapt_small)
        if dataset_info.get("server_root") is None:
            raise RuntimeError(f"{config.STRATEGY} requires a server root set "
                               f"(SERVER_ROOT_PER_CLASS>0).")
        print(f"  - {config.STRATEGY}: server root "
              f"{len(dataset_info['server_root']['dataset'])} samples")
        if config.STRATEGY == "BOBA":
            # Per-class root loaders for building the c simplex vertices each round.
            from torch.utils.data import Subset, DataLoader as _DL
            _root = dataset_info["server_root"]["dataset"]
            _rlabels = dataset_info["server_root"]["labels"]
            # A single-site root set (SERVER_ROOT_SITE) can lack classes: the
            # server can only build vertices for the classes it actually holds.
            boba_class_loaders = []
            for _z in sorted(set(_rlabels.tolist())):
                _idx = np.where(_rlabels == _z)[0].tolist()
                boba_class_loaders.append(_DL(Subset(_root, _idx),
                                              batch_size=config.BATCH_SIZE, shuffle=True))
            _c = len(boba_class_loaders)
            print(f"  - BOBA: f={getattr(config, 'BOBA_NUM_BYZ', 2)}, "
                  f"{_c}/{dataset_info['num_classes']} class vertices, "
                  f"need n-f > c-1 ({config.NUM_CLIENTS}-f > {_c-1})")

    print("\n" + "=" * 70)
    print(f"Starting Federated Learning with {config.STRATEGY}...")
    print("=" * 70)
    
    # Run federated learning rounds manually (no simulation/Ray)
    final_acc = 0.0
    tracker.start()

    for round_num in range(config.NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{config.NUM_ROUNDS} ---")
        
        # Train all clients
        clients = [client_fn(str(i)) for i in range(config.NUM_CLIENTS)]
        
        # Get initial parameters from strategy or first client
        if round_num == 0:
            parameters = clients[0].get_parameters(config_dict={})
        
        # Fit clients
        scaffold_round_deltas.clear()  # SCAFFOLD: reset per-round delta accumulator
        results = []
        for i, client in enumerate(clients):
            fit_result = client.fit(parameters, config_dict={})
            results.append(fit_result)
        
        # Aggregate parameters manually based on strategy
        import numpy as np
        
        if config.STRATEGY == "FedAgain":
            # FedAgain: Trust-based aggregation
            parameters = fedagain_strategy.aggregate(results, parameters)
        
        elif config.STRATEGY == "Bulyan":
            # Bulyan: Byzantine-robust aggregation
            # Remove outliers, then use coordinate-wise median
            num_params = len(results[0][0])
            aggregated = []
            
            num_malicious = config.BULYAN_NUM_MALICIOUS if hasattr(config, 'BULYAN_NUM_MALICIOUS') else 1
            
            for i in range(num_params):
                param_stack = np.stack([result[0][i] for result in results], axis=0)
                
                # Bulyan removes top and bottom outliers
                # Remove num_malicious from each end
                if param_stack.shape[0] > 2 * num_malicious:
                    # Compute distances from median for each client
                    median_param = np.median(param_stack, axis=0)
                    distances = np.sum((param_stack - median_param) ** 2, axis=tuple(range(1, param_stack.ndim)))
                    
                    # Sort by distance and remove outliers
                    sorted_indices = np.argsort(distances)
                    kept_indices = sorted_indices[num_malicious:-num_malicious] if num_malicious > 0 else sorted_indices
                    
                    # Aggregate remaining clients
                    filtered_params = param_stack[kept_indices]
                    aggregated_param = np.mean(filtered_params, axis=0)
                else:
                    # Not enough clients, just use median
                    aggregated_param = np.median(param_stack, axis=0)
                
                aggregated.append(aggregated_param)
            
            parameters = aggregated
        
        elif config.STRATEGY == "FedMedian":
            # FedMedian: coordinate-wise median
            num_params = len(results[0][0])
            aggregated = []
            for i in range(num_params):
                param_stack = np.stack([result[0][i] for result in results], axis=0)
                median_param = np.median(param_stack, axis=0)
                aggregated.append(median_param)
            parameters = aggregated
            
        elif config.STRATEGY == "FedYogi":
            # FedYogi: Full implementation with sign-based adaptive velocity update
            # Using very conservative hyperparameters for deep network stability
            total_examples = sum([result[1] for result in results])
            num_params = len(results[0][0])
            
            # Weighted average of client updates
            aggregated = []
            for i in range(num_params):
                weighted_sum = None
                for result in results:
                    param = result[0][i].astype(np.float32)  # Ensure float32
                    num_examples = result[1]
                    weight = num_examples / total_examples
                    
                    if weighted_sum is None:
                        weighted_sum = param * weight
                    else:
                        weighted_sum += param * weight
                
                aggregated.append(weighted_sum)
            
            # Initialize momentum and velocity on first round
            if fedyogi_m is None:
                fedyogi_m = [np.zeros_like(p, dtype=np.float32) for p in parameters]
                fedyogi_v = [np.zeros_like(p, dtype=np.float32) for p in parameters]
            
            # Convert parameters to float32 for computation
            parameters_f32 = [p.astype(np.float32) for p in parameters]
            
            # Compute pseudo-gradient: Delta = w_avg - w_t (direction towards aggregated model)
            pseudo_grads = [aggregated[i] - parameters_f32[i] for i in range(num_params)]
            
            # Update timestep
            fedyogi_t += 1
            
            # FedYogi update with sign-based velocity
            new_params = []
            for i in range(num_params):
                # Update biased first moment estimate (momentum) - same as Adam
                fedyogi_m[i] = config.FEDYOGI_BETA1 * fedyogi_m[i] + (1 - config.FEDYOGI_BETA1) * pseudo_grads[i]
                
                # Update biased second moment estimate (velocity) - YOGI-STYLE
                # Key innovation: v = v - (1-beta2)*Delta^2*sign(v-Delta^2)
                # This prevents unbounded growth and provides self-correction
                grad_squared = pseudo_grads[i] ** 2
                fedyogi_v[i] = fedyogi_v[i] - (1 - config.FEDYOGI_BETA2) * grad_squared * np.sign(fedyogi_v[i] - grad_squared)
                
                # Bias correction
                m_hat = fedyogi_m[i] / (1 - config.FEDYOGI_BETA1 ** fedyogi_t)
                v_hat = fedyogi_v[i] / (1 - config.FEDYOGI_BETA2 ** fedyogi_t)
                
                # Update parameters with adaptive learning rate
                # w_{t+1} = w_t + eta * m_hat / (sqrt(|v_hat|) + tau)
                # Using |v_hat| because Yogi can make v negative
                # Using very small eta (0.001) to prevent overshooting
                denominator = np.sqrt(np.abs(v_hat)) + config.FEDYOGI_TAU
                update_step = config.FEDYOGI_SERVER_LR * m_hat / denominator
                
                # Clip update to prevent extreme changes
                # This is critical for deep networks
                max_update = 0.1  # Don't change any parameter by more than 10% per round
                update_norm = np.abs(update_step)
                param_norm = np.abs(parameters_f32[i]) + 1e-8
                scale = np.minimum(1.0, max_update * param_norm / (update_norm + 1e-8))
                clipped_update = update_step * scale
                
                new_param = parameters_f32[i] + clipped_update
                
                # Final safety check for nan/inf
                if np.any(np.isnan(new_param)) or np.any(np.isinf(new_param)):
                    print(f"WARNING: NaN/Inf detected in parameter {i} after FedYogi update!")
                    print(f"  Falling back to aggregated parameters")
                    new_param = aggregated[i]
                
                # Convert back to original dtype
                original_dtype = parameters[i].dtype
                new_params.append(new_param.astype(original_dtype))
            
            parameters = new_params

        elif config.STRATEGY == "FedDyn":
            # FedDyn: theta = mean_active(theta_k) - (1/alpha)*h, with the server
            # drift accumulator  h <- h - (alpha/m)*sum_active(theta_k - theta_srv).
            # m = total clients (= |P_t| here, full participation). Applied to
            # trainable params only; buffers (BN stats) are averaged plainly.
            num_params = len(results[0][0])
            m_total = config.NUM_CLIENTS
            alpha = getattr(config, "FEDDYN_ALPHA", 0.1)
            if feddyn_h is None:
                feddyn_h = [np.zeros_like(p, dtype=np.float64) for p in parameters]

            new_params = []
            for i in range(num_params):
                stack = np.stack([r[0][i] for r in results], axis=0)
                mean_i = np.mean(stack, axis=0)
                if i in feddyn_trainable_idx:
                    drift = np.sum(stack - parameters[i], axis=0)  # sum_active(theta_k - theta_srv)
                    feddyn_h[i] = feddyn_h[i] - (alpha / m_total) * drift
                    new_i = mean_i - (1.0 / alpha) * feddyn_h[i]
                    new_params.append(new_i.astype(parameters[i].dtype))
                else:
                    new_params.append(mean_i.astype(parameters[i].dtype))
            parameters = new_params

        elif config.STRATEGY == "RFLPA":
            # RFLPA (= FLTrust defense core; secret-sharing omitted, accuracy-
            # equivalent in exact arithmetic). Score client updates by ReLU'd
            # cosine similarity to a server reference update g0 trained on the
            # clean root set, norm-clip to ||g0||, then trust-weighted average.
            eps = 1e-10
            # Match the server's optimizer steps to an average client's so the
            # reference-update magnitude is comparable (FLTrust assumption).
            target_steps = max(1, int(np.mean([len(tl) for tl in train_loaders]))
                               * config.LOCAL_EPOCHS)
            g0 = compute_server_reference_update(
                parameters, server_ref_model, dataset_info["server_root"]["loader"],
                device, config.LEARNING_RATE, config.MOMENTUM, target_steps)
            g0_flat = np.concatenate([a.ravel() for a in g0]).astype(np.float64)
            g0_norm = float(np.linalg.norm(g0_flat)) + eps

            num_params = len(parameters)
            accum = [np.zeros_like(p, dtype=np.float64) for p in parameters]
            ts_sum = 0.0
            for r in results:
                gi = [r[0][i] - parameters[i] for i in range(num_params)]
                gi_flat = np.concatenate([a.ravel() for a in gi]).astype(np.float64)
                gi_norm = float(np.linalg.norm(gi_flat)) + eps
                cos = float(np.dot(gi_flat, g0_flat) / (gi_norm * g0_norm))
                ts = max(0.0, cos)
                if ts <= 0.0:
                    continue
                scale = ts * (g0_norm / gi_norm)  # trust * norm-clip factor
                for i in range(num_params):
                    accum[i] += scale * gi[i]
                ts_sum += ts
            if ts_sum < eps:
                print("  [RFLPA] all clients suppressed (sum trust ~ 0); keeping global model")
            else:
                parameters = [(parameters[i] + accum[i] / ts_sum).astype(parameters[i].dtype)
                              for i in range(num_params)]

        elif config.STRATEGY == "BOBA":
            # BOBA (Bao et al., AISTATS 2024): robust affine-subspace aggregation
            # under label skew. Build c per-class server vertices, then fit the
            # honest simplex and average the on-simplex client updates.
            shapes = [p.shape for p in parameters]
            dtypes = [p.dtype for p in parameters]

            def _flat(lst):
                return np.concatenate([a.ravel() for a in lst]).astype(np.float64)

            def _unflat(vec):
                out, off = [], 0
                for sh, dt in zip(shapes, dtypes):
                    size = int(np.prod(sh)) if sh else 1
                    out.append(vec[off:off + size].reshape(sh).astype(dt))
                    off += size
                return out

            c = len(boba_class_loaders)  # = num_classes unless the root set lacks classes
            target_steps = max(1, int(np.mean([len(tl) for tl in train_loaders]))
                               * config.LOCAL_EPOCHS)
            # c simplex vertices: global model trained on each single-class root subset.
            S = np.stack([_flat(compute_server_reference_update(
                              parameters, server_ref_model, boba_class_loaders[z],
                              device, config.LEARNING_RATE, config.MOMENTUM, target_steps))
                          + _flat(parameters)  # vertex = w_global + class-z update
                          for z in range(c)], axis=0)
            G = np.stack([_flat(r[0]) for r in results], axis=0)
            mu, info = boba_aggregate(
                G, S, f=getattr(config, "BOBA_NUM_BYZ", 2), c=c,
                pmin=getattr(config, "BOBA_PMIN", -0.5),
                max_iter=getattr(config, "BOBA_MAX_ITER", 20))
            print(f"  [BOBA] kept {info['kept']}/{len(results)} clients, "
                  f"subspace dim {info['ncomp']}"
                  + ("  (DEGENERATE -> mean)" if info.get("degenerate") else ""))
            parameters = _unflat(mu)

        else:
            # FedAvg: weighted average (FedProx uses same aggregation, proximal term is in client training)
            total_examples = sum([result[1] for result in results])
            num_params = len(results[0][0])
            aggregated = []
            
            for i in range(num_params):
                weighted_sum = None
                for result in results:
                    param = result[0][i]
                    num_examples = result[1]
                    weight = num_examples / total_examples
                    
                    if weighted_sum is None:
                        weighted_sum = param * weight
                    else:
                        weighted_sum += param * weight
                
                aggregated.append(weighted_sum)
            
            parameters = aggregated

        # SCAFFOLD: update the global control variate  c <- c + mean(delta c_i).
        if config.STRATEGY == "Scaffold" and scaffold_round_deltas:
            if scaffold_c_global is None:
                scaffold_c_global = {n: torch.zeros_like(d)
                                     for n, d in scaffold_round_deltas[0].items()}
            n_cli = len(scaffold_round_deltas)
            for n in scaffold_c_global:
                scaffold_c_global[n] = scaffold_c_global[n] + \
                    sum(d[n] for d in scaffold_round_deltas) / n_cli

        # Evaluate global model
        loss, metrics = evaluate_global_model(round_num + 1, parameters, {})
        final_acc = metrics["accuracy"]

        # Persist this round's result immediately (crash-resilient).
        tracker.log_round(round_num + 1,
                          {"accuracy": metrics["accuracy"], "loss": loss})

        print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Federated learning completed!")
    print(f"Strategy: {config.STRATEGY}")
    print(f"Dataset: {config.DATASET}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    
    # Compute detailed metrics on final model
    print("\nComputing final metrics...")
    y_true, y_pred = evaluate_global_model(
        server_round=config.NUM_ROUNDS,
        parameters=parameters,
        config_dict={},
        return_predictions=True
    )
    
    final_metrics = compute_classification_metrics(
        y_true, y_pred, dataset_info['num_classes']
    )
    
    print(f"  - Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  - Precision: {final_metrics['precision']:.4f}")
    print(f"  - Recall:    {final_metrics['recall']:.4f}")
    print(f"  - F1 Score:  {final_metrics['f1']:.4f}")

    # Clinical view of the same predictions (manifest only; the ledger CSV keeps
    # its fixed columns): per-subtype sensitivity/specificity, balanced
    # accuracy, confusion matrix and, for the multi-site federation, accuracy
    # per acquisition site.
    n_cls = dataset_info['num_classes']
    cm = np.bincount(n_cls * y_true.astype(int) + y_pred.astype(int),
                     minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    tn = cm.sum() - support - predicted + tp
    sens = np.divide(tp, support, out=np.full(n_cls, np.nan), where=support > 0)
    spec = np.divide(tn, tn + predicted - tp, out=np.full(n_cls, np.nan),
                     where=(tn + predicted - tp) > 0)
    final_metrics.update({
        "balanced_accuracy": float(np.nanmean(sens)),
        "class_names": dataset_info.get("class_names") or [str(i) for i in range(n_cls)],
        "per_class_sensitivity": [None if np.isnan(v) else float(v) for v in sens],
        "per_class_specificity": [None if np.isnan(v) else float(v) for v in spec],
        "per_class_support": support.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    })
    if dataset_info.get("test_sites") is not None:
        sites = np.array(dataset_info["test_sites"])
        final_metrics["per_site_accuracy"] = {
            s: float(np.mean(y_pred[sites == s] == y_true[sites == s]))
            for s in sorted(set(sites.tolist()))}
        print(f"  - Per-site accuracy: {final_metrics['per_site_accuracy']}")
    print(f"  - Balanced acc: {final_metrics['balanced_accuracy']:.4f}")
    
    # Log experiment results
    experiment_config = {
        "model": config.MODEL_NAME.upper(),
        "strategy": config.STRATEGY,
        "dataset": config.DATASET,
        "data_distribution": config.DATA_DISTRIBUTION,
        "num_clients": config.NUM_CLIENTS,
        "num_rounds": config.NUM_ROUNDS,
        "local_epochs": config.LOCAL_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "use_pretrained": config.USE_PRETRAINED,
        "corrupt_clients": getattr(config, 'CORRUPT_CLIENTS', False),
        "corrupt_ratio": getattr(config, 'CORRUPT_CLIENT_RATIO', 0.0),
        "corruption_severity": getattr(config, 'CORRUPTION_SEVERITY', 0)
    }
    
    logger.log_experiment(experiment_config, final_metrics)

    # Save per-round trust scores for the FedAgain ablation (#2) / sensitivity
    # (#3) studies.
    if (config.STRATEGY == "FedAgain"
            and getattr(config, "FEDAGAIN_LOG_TRUST", False)
            and fedagain_strategy is not None):
        trust_dir = os.path.join(RESULTS_DIR, "trust_scores")
        os.makedirs(trust_dir, exist_ok=True)
        mode = getattr(config, "FEDAGAIN_TRUST_MODE", "combined_mult")
        # The variant tags matter: without them a weighted_additive sweep over
        # (alpha,beta), or a normalization sweep (which keeps trust_mode at
        # combined_mult), silently overwrites the reference log of the mode it
        # shares a name with.
        tag = ""
        if mode == "weighted_additive":
            tag += f"_a{getattr(config, 'FEDAGAIN_ALPHA', 1.0)}"
            tag += f"_b{getattr(config, 'FEDAGAIN_BETA', 1.0)}"
        norm = getattr(config, "FEDAGAIN_NORM_MODE", "sum_to_one")
        if norm != "sum_to_one":
            tag += f"_nm-{norm}"
            if norm == "softmax":
                tag += f"_tau{getattr(config, 'FEDAGAIN_NORM_TEMPERATURE', 1.0)}"
        if getattr(config, "FEDAGAIN_MIX_LAMBDA", 0.0) > 0:
            tag += f"_mix{config.FEDAGAIN_MIX_LAMBDA}"
        trust_path = os.path.join(
            trust_dir,
            f"trust_{config.DATASET}_{config.DATA_DISTRIBUTION}_{mode}{tag}"
            f"_c{config.NUM_CLIENTS}_r{config.NUM_ROUNDS}.csv"
        )
        if DATA_VERSION == "v1":
            fedagain_strategy.save_trust_log(trust_path)
            print(f"  - Trust scores saved to: {trust_path}")
        else:
            # v2: one complete log per run (seed and rho included via the run
            # key), plus the legacy name the analysis scripts read, written only
            # by the reference configuration (rho=0.3, image corruption, default
            # epsilon, seed 44) so later runs cannot overwrite it.
            by_run = os.path.join(trust_dir, "by_run")
            os.makedirs(by_run, exist_ok=True)
            fedagain_strategy.save_trust_log(os.path.join(by_run, f"{tracker.run_key}.csv"))
            if (getattr(config, "CORRUPT_CLIENT_RATIO", None) == 0.3
                    and getattr(config, "POISON_MODE", "image") == "image"
                    and getattr(config, "FEDAGAIN_EPSILON", 1e-6) == 1e-6
                    and getattr(config, "SEED", 42) == 44):
                fedagain_strategy.save_trust_log(trust_path)
            print(f"  - Trust scores saved to: {by_run}/{tracker.run_key}.csv")

    # Mark the run completed in the resilient tracker.
    tracker.finish(final_metrics)

    print("=" * 70)
