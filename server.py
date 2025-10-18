"""
Server-side strategies for federated learning using Flower framework.
Contains implementations of FedAvg, FedProx, FedOpt (FedAdam), and FedMedian.
"""

import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted averaging of metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_strategy(strategy_name: str = "FedAvg", **kwargs):
    """
    Get the specified federated learning strategy from Flower.
    
    Args:
        strategy_name: Name of strategy ("FedAvg", "FedProx", "FedOpt", "FedMedian")
        **kwargs: Strategy-specific parameters
            - proximal_mu: Proximal term for FedProx (default: 0.1)
            - server_lr: Server learning rate for FedOpt (default: 1.0)
            - beta1, beta2, tau: Parameters for FedOpt
    
    Returns:
        Flower strategy instance
    """
    # Common parameters for all strategies
    common_params = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }
    
    if strategy_name == "FedAvg":
        return fl.server.strategy.FedAvg(**common_params)
    
    elif strategy_name == "FedProx":
        proximal_mu = kwargs.get("proximal_mu", 0.1)
        return fl.server.strategy.FedProx(
            **common_params,
            proximal_mu=proximal_mu
        )
    
    elif strategy_name == "FedOpt":
        server_lr = kwargs.get("server_lr", 1.0)
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.99)
        tau = kwargs.get("tau", 1e-3)
        return fl.server.strategy.FedAdam(
            **common_params,
            eta=server_lr,
            eta_l=0.1,
            beta_1=beta1,
            beta_2=beta2,
            tau=tau
        )
    
    elif strategy_name == "FedMedian":
        return fl.server.strategy.FedMedian(**common_params)
    
    elif strategy_name == "Bulyan":
        # Bulyan is a Byzantine-robust aggregation strategy
        # It requires specifying the number of Byzantine clients to tolerate
        num_malicious = kwargs.get("num_malicious", 0)
        to_keep = kwargs.get("to_keep", None)
        return fl.server.strategy.Bulyan(
            **common_params,
            num_malicious_clients=num_malicious,
            to_keep=to_keep
        )
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: FedAvg, FedProx, FedOpt, FedMedian, Bulyan"
        )

