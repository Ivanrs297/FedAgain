"""
Federated learning with multiple aggregation strategies.
Simple implementation without Ray/TensorFlow dependencies.
Supports FedAvg, FedProx, FedAdam, FedMedian, Bulyan, and FedAgain.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
from collections import OrderedDict
from src.model import get_model
from src.dataset import load_data
from src.client import train, test
from src.strategies.fedagain import FedAgainStrategy, compute_benchmark_error
from src.logger import ExperimentLogger, compute_classification_metrics
from src.tracking import ExperimentTracker
import config
import numpy as np


# Global variables for data loaders (shared across clients)
train_loaders = None
val_loaders = None
test_loader = None
dataset_info = None


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
        
        # Train locally
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
    logger = ExperimentLogger(results_dir="results")

    # Build the experiment identity and resilient tracker. It writes results
    # incrementally (per round) and lets already-completed runs be skipped on
    # resume, which matters for long, failure-prone experiment sweeps.
    is_fedagain = config.STRATEGY == "FedAgain"
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
        "trust_mode": getattr(config, "FEDAGAIN_TRUST_MODE", "combined_mult") if is_fedagain else "",
        "trust_alpha": getattr(config, "FEDAGAIN_ALPHA", 1.0) if is_fedagain else "",
        "trust_beta": getattr(config, "FEDAGAIN_BETA", 1.0) if is_fedagain else "",
        "seed": 42,
    }
    tracker = ExperimentTracker(tracker_config, results_dir="results")
    if tracker.is_completed() and os.environ.get("EXPERIMENT_FORCE") != "1":
        print(f"[tracker] run '{tracker.run_key}' already completed -> "
              f"skipping (set EXPERIMENT_FORCE=1 to force a rerun).")
        sys.exit(0)

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
        dirichlet_alpha=config.DIRICHLET_ALPHA
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
        fedagain_strategy = FedAgainStrategy(
            epsilon=epsilon, mode=trust_mode, alpha=alpha, beta=beta,
            log_trust=log_trust,
        )
        print(f"  - FedAgain epsilon: {epsilon}")
        print(f"  - FedAgain trust mode: {trust_mode}"
              + (f" (alpha={alpha}, beta={beta})"
                 if trust_mode == "weighted_additive" else ""))
    
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
        trust_dir = os.path.join("results", "trust_scores")
        os.makedirs(trust_dir, exist_ok=True)
        mode = getattr(config, "FEDAGAIN_TRUST_MODE", "combined_mult")
        trust_path = os.path.join(
            trust_dir,
            f"trust_{config.DATASET}_{config.DATA_DISTRIBUTION}_{mode}"
            f"_c{config.NUM_CLIENTS}_r{config.NUM_ROUNDS}.csv"
        )
        fedagain_strategy.save_trust_log(trust_path)
        print(f"  - Trust scores saved to: {trust_path}")

    # Mark the run completed in the resilient tracker.
    tracker.finish(final_metrics)

    print("=" * 70)
