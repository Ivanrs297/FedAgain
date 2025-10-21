"""
Simple experiment runner for testing multiple federated learning strategies.
Automatically runs experiments with different configurations and logs results.
"""

import subprocess
import sys
from pathlib import Path


def update_config(strategy: str, dataset: str = None, num_rounds: int = None):
    """
    Update config.py with new strategy and parameters.
    
    Args:
        strategy: Strategy name to use
        dataset: Dataset name (optional)
        num_rounds: Number of rounds (optional)
    """
    config_path = Path("config.py")
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    with open(config_path, 'w') as f:
        for line in lines:
            # Update strategy
            if line.startswith('STRATEGY = '):
                f.write(f'STRATEGY = "{strategy}"  # Options: "FedAvg", "FedProx", "FedAdam", "FedMedian", "Bulyan", "FedAgain"\n')
            # Update dataset if provided
            elif dataset and line.startswith('DATASET = '):
                f.write(f'DATASET = "{dataset}"  # Options: "EMNIST", "MNIST", "CIFAR10"\n')
            # Update rounds if provided
            elif num_rounds and line.startswith('NUM_ROUNDS = '):
                f.write(f'NUM_ROUNDS = {num_rounds}\n')
            else:
                f.write(line)


def run_experiment(strategy: str, dataset: str = None):
    """
    Run a single experiment with the given strategy.
    
    Args:
        strategy: Strategy to test
        dataset: Dataset to use (optional, keeps current if None)
    """
    print("\n" + "="*80)
    print(f"Running experiment: Strategy={strategy}, Dataset={dataset or 'current'}")
    print("="*80 + "\n")
    
    # Update config
    update_config(strategy, dataset)
    
    # Run main.py
    result = subprocess.run([sys.executable, "main.py"], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[WARNING] Experiment failed for {strategy}")
        return False
    
    print(f"\n[OK] Experiment completed for {strategy}")
    return True


def run_all_strategies(dataset: str = None, strategies: list = None):
    """
    Run experiments for all strategies.
    
    Args:
        dataset: Dataset to use for all experiments (None = use current)
        strategies: List of strategies to test (None = all)
    """
    if strategies is None:
        strategies = ["FedAvg", "FedProx", "FedYogi", "FedMedian", "Bulyan", "FedAgain"]
    
    print("\n" + "="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print(f"Dataset: {dataset or 'Current config'}")
    print(f"Strategies to test: {', '.join(strategies)}")
    print(f"Total experiments: {len(strategies)}")
    print("="*80)
    
    results = {}
    for i, strategy in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] Testing {strategy}...")
        success = run_experiment(strategy, dataset)
        results[strategy] = "[OK] Success" if success else "[X] Failed"
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for strategy, status in results.items():
        print(f"{strategy:15} {status}")
    print("="*80)
    print(f"\n[OK] All results saved to: results/experiments.csv")


def main():
    """Main entry point with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments with multiple strategies"
    )
    
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["FedAvg", "FedProx", "FedYogi", "FedMedian", "Bulyan", "FedAgain"],
        help="Strategies to test (default: all)"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "EMNIST", "CIFAR10"],
        help="Dataset to use for experiments"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test only FedAvg, FedProx, and FedAgain"
    )
    
    args = parser.parse_args()
    
    # Determine which strategies to run
    if args.quick:
        strategies = ["FedAvg", "FedProx", "FedYogi", "FedAgain"]
        print("\n🚀 Quick mode: Testing 4 strategies")
    else:
        strategies = args.strategies
    
    # Run experiments
    run_all_strategies(dataset=args.dataset, strategies=strategies)


if __name__ == "__main__":
    main()

