"""
Federated learning using Flower framework with multiple strategies.
Supports FedAvg, FedProx, FedOpt (FedAdam), and FedMedian.
"""

import torch
import flwr as fl
from collections import OrderedDict
from model import get_model
from dataset import load_data
from client import train, test
from server import get_strategy
import config


# Global variables for data loaders (shared across clients)
train_loaders = None
val_loaders = None
test_loader = None
dataset_info = None


class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing local training and evaluation."""
    
    def __init__(self, cid: str, device: torch.device):
        self.cid = cid
        self.device = device
        
        # Create model
        self.model = get_model(
            num_classes=dataset_info['num_classes'],
            num_channels=dataset_info['num_channels'],
            use_pretrained=config.USE_PRETRAINED
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
        
        train_loss, train_acc = train(
            self.model,
            self.trainloader,
            epochs=config.LOCAL_EPOCHS,
            device=self.device,
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM
        )
        
        print(f"  Client {int(self.cid) + 1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
        
        return self.get_parameters(config_dict={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config_dict):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client for the given client ID."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return FlowerClient(cid, device)


def evaluate_global_model(server_round: int, parameters, config_dict):
    """Evaluate the global model on the centralized test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model and set parameters
    model = get_model(
        num_classes=dataset_info['num_classes'],
        num_channels=dataset_info['num_channels'],
        use_pretrained=config.USE_PRETRAINED
    )
    model.to(device)
    
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    # Evaluate
    loss, accuracy = test(model, test_loader, device=device)
    
    print(f"\n[Round {server_round}] Global Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return loss, {"accuracy": accuracy}


if __name__ == "__main__":
    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Federated Learning with Flower Framework")
    print("=" * 70)
    print(f"Dataset: {config.DATASET}")
    print(f"Strategy: {config.STRATEGY}")
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
    
    print(f"\nModel: ResNet18 (Pretrained: {config.USE_PRETRAINED})")
    print(f"Training Configuration:")
    print(f"  - Strategy: {config.STRATEGY}")
    print(f"  - Rounds: {config.NUM_ROUNDS}")
    print(f"  - Local epochs: {config.LOCAL_EPOCHS}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Momentum: {config.MOMENTUM}")
    
    # Strategy-specific parameters
    if config.STRATEGY == "FedProx":
        print(f"  - FedProx μ: {config.FEDPROX_PROXIMAL_MU}")
    elif config.STRATEGY == "FedOpt":
        print(f"  - Server LR: {config.FEDOPT_SERVER_LR}")
        print(f"  - β1: {config.FEDOPT_BETA1}, β2: {config.FEDOPT_BETA2}")
    elif config.STRATEGY == "Bulyan":
        print(f"  - Byzantine clients to tolerate: {config.BULYAN_NUM_MALICIOUS}")
    
    # Initialize FedOpt optimizer if needed
    fedopt_m = None
    fedopt_v = None
    fedopt_t = 0
    if config.STRATEGY == "FedOpt":
        # Will be initialized after first round
        pass
    
    print("\n" + "=" * 70)
    print(f"Starting Federated Learning with {config.STRATEGY}...")
    print("=" * 70)
    
    # Run federated learning rounds manually (no simulation/Ray)
    final_acc = 0.0
    
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
        
        if config.STRATEGY == "Bulyan":
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
            
        elif config.STRATEGY == "FedOpt":
            # FedOpt (FedAdam): Server-side Adam optimization
            # First do weighted average of client updates
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
            
            # Initialize momentum and velocity on first round
            if fedopt_m is None:
                fedopt_m = [np.zeros_like(param) for param in parameters]
                fedopt_v = [np.zeros_like(param) for param in parameters]
            
            # Compute pseudo-gradients and apply Adam
            fedopt_t += 1
            new_parameters = []
            
            for i in range(num_params):
                # Delta (pseudo-gradient): aggregated - server (direction to move)
                delta = aggregated[i] - parameters[i]
                
                # Update biased first moment estimate
                fedopt_m[i] = config.FEDOPT_BETA1 * fedopt_m[i] + (1 - config.FEDOPT_BETA1) * delta
                
                # Update biased second moment estimate
                fedopt_v[i] = config.FEDOPT_BETA2 * fedopt_v[i] + (1 - config.FEDOPT_BETA2) * (delta ** 2)
                
                # Compute bias-corrected moments
                m_hat = fedopt_m[i] / (1 - config.FEDOPT_BETA1 ** fedopt_t)
                v_hat = fedopt_v[i] / (1 - config.FEDOPT_BETA2 ** fedopt_t)
                
                # Update parameters (Adam step): move in direction of delta
                new_param = parameters[i] + config.FEDOPT_SERVER_LR * m_hat / (np.sqrt(v_hat) + config.FEDOPT_TAU)
                new_parameters.append(new_param.astype(parameters[i].dtype))
            
            parameters = new_parameters
            
        else:
            # FedAvg, FedProx: weighted average
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
        
        print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Federated learning completed!")
    print(f"Strategy: {config.STRATEGY}")
    print(f"Dataset: {config.DATASET}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print("=" * 70)
