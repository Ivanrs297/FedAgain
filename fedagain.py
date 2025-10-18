"""
FedAgain: Trust-based Federated Learning Strategy
Weights clients based on benchmark error and model divergence.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from collections import OrderedDict


class FedAgainStrategy:
    """
    FedAgain aggregation strategy.
    
    Trust weight: T_k = 1 / (E_k * D_k + epsilon)
    where:
        E_k = benchmark error of client k
        D_k = divergence (L2 norm of parameter difference)
        epsilon = small constant for numerical stability
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize FedAgain strategy.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon
        self.global_params = None
    
    def aggregate(self, results: List[Tuple], global_params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Aggregate client updates using trust-based weighting.
        
        Args:
            results: List of (client_params, num_examples, metrics) from clients
            global_params: Current global model parameters
        
        Returns:
            Aggregated parameters
        """
        if len(results) == 0:
            return global_params
        
        # Extract data from results
        client_params_list = [result[0] for result in results]
        client_sizes = [result[1] for result in results]
        client_metrics = [result[2] for result in results]
        
        # Get benchmark errors (E_k) from client metrics
        benchmark_errors = []
        for metrics in client_metrics:
            if 'benchmark_error' in metrics:
                benchmark_errors.append(metrics['benchmark_error'])
            else:
                # If no benchmark error provided, use 1.0 (neutral weight)
                benchmark_errors.append(1.0)
        
        # Compute divergences (D_k) for each client
        divergences = []
        for client_params in client_params_list:
            divergence = self._compute_divergence(client_params, global_params)
            divergences.append(divergence)
        
        # Compute trust weights (T_k)
        trust_weights = []
        for E_k, D_k in zip(benchmark_errors, divergences):
            T_k = 1.0 / (E_k * D_k + self.epsilon)
            trust_weights.append(T_k)
        
        # Normalize trust weights
        total_trust = sum(trust_weights)
        normalized_weights = [w / total_trust for w in trust_weights]
        
        # Aggregate with trust-weighted average
        num_params = len(client_params_list[0])
        aggregated = []
        
        for i in range(num_params):
            weighted_sum = None
            for client_params, weight in zip(client_params_list, normalized_weights):
                param = client_params[i]
                if weighted_sum is None:
                    weighted_sum = param * weight
                else:
                    weighted_sum += param * weight
            aggregated.append(weighted_sum)
        
        return aggregated
    
    def _compute_divergence(self, client_params: List[np.ndarray], 
                           global_params: List[np.ndarray]) -> float:
        """
        Compute L2 divergence between client and global parameters.
        
        D_k = ||w^{(k)}_{t+1} - w_t||_2
        
        Args:
            client_params: Client model parameters
            global_params: Global model parameters
        
        Returns:
            L2 divergence
        """
        divergence = 0.0
        for client_param, global_param in zip(client_params, global_params):
            diff = client_param - global_param
            divergence += np.sum(diff ** 2)
        
        return np.sqrt(divergence)


def compute_benchmark_error(model, benchmark_loader, device):
    """
    Compute benchmark error on a small validation/benchmark set.
    
    This is the error E_k that each client computes on the incoming global model
    before local training.
    
    Args:
        model: PyTorch model
        benchmark_loader: DataLoader for benchmark data
        device: Device to run on
    
    Returns:
        Error rate (1 - accuracy) on benchmark set
    """
    import torch.nn as nn
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in benchmark_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    error = 1.0 - accuracy
    
    return error

