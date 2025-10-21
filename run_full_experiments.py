"""
Comprehensive experiment runner for federated learning.
Tests all combinations of datasets, strategies, and models.
"""

import subprocess
import os
import sys
import time
from datetime import datetime

# Configuration
# DATASETS = ["MNIST", "CIFAR10",  # "CIFAR10C" requires download
#             "JONATHAN_MIX", "JONATHAN_SEC", "JONATHAN_SUR",
#             "DAUDON_MIX", "DAUDON_SEC", "DAUDON_SUR", 
#             "MYSTONE"]
DATASETS = ["CIFAR10"]
STRATEGIES = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedAgain"]
MODELS = ["resnet18"]


def update_config(updates: dict):
    """Update multiple key-value pairs in config.py."""
    config_path = 'config.py'
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    with open(config_path, 'w') as f:
        for line in lines:
            updated = False
            for key, value in updates.items():
                if line.strip().startswith(f"{key} ="):
                    # Handle string values
                    if isinstance(value, str) and not value.startswith(('"', "'")):
                        f.write(f"{key} = \"{value}\"\n")
                    else:
                        f.write(f"{key} = {value}\n")
                    updated = True
                    break
            if not updated:
                f.write(line)


def run_single_experiment(dataset, strategy, model, exp_num, total_exps):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"[{exp_num}/{total_exps}] Running: {model.upper()} + {strategy} on {dataset}")
    print(f"{'='*80}")
    
    # Update config
    update_config({
        "DATASET": dataset,
        "STRATEGY": strategy,
        "MODEL_NAME": model
    })
    
    # Run experiment
    start_time = time.time()
    process = subprocess.run(
        ["python", "main.py"],
        capture_output=True,
        text=True
    )
    elapsed_time = time.time() - start_time
    
    if process.returncode == 0:
        print(f"[OK] Completed in {elapsed_time:.1f}s")
        return True, elapsed_time
    else:
        print(f"[FAILED] after {elapsed_time:.1f}s")
        print("Error output:")
        print(process.stderr[-500:])  # Last 500 chars of error
        return False, elapsed_time


def run_all_experiments(datasets=None, strategies=None, models=None, skip_existing=True):
    """
    Run all experiment combinations.
    
    Args:
        datasets: List of datasets to test (default: all)
        strategies: List of strategies to test (default: all)
        models: List of models to test (default: all)
        skip_existing: Skip if results already exist in CSV
    """
    datasets = datasets or DATASETS
    strategies = strategies or STRATEGIES
    models = models or MODELS
    
    # Calculate total experiments
    total_exps = len(datasets) * len(strategies) * len(models)
    
    # Read current config for num_clients, num_rounds, local_epochs
    import config
    
    print("\n" + "="*80)
    print("COMPREHENSIVE FEDERATED LEARNING EXPERIMENTS")
    print("="*80)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Models: {', '.join(models)}")
    print(f"Num Clients: {config.NUM_CLIENTS}")
    print(f"Num Rounds: {config.NUM_ROUNDS}")
    print(f"Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"Total experiments: {total_exps}")
    print(f"Estimated time: {total_exps * 5:.0f}-{total_exps * 15:.0f} minutes")
    print("="*80)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    total_time = 0
    exp_num = 0
    
    start_time = datetime.now()
    
    # Run all combinations
    for dataset in datasets:
        for model in models:
            for strategy in strategies:
                exp_num += 1
                success, elapsed = run_single_experiment(
                    dataset, strategy, model, exp_num, total_exps
                )
                
                results.append({
                    "dataset": dataset,
                    "model": model,
                    "strategy": strategy,
                    "success": success,
                    "time": elapsed
                })
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                total_time += elapsed
                
                # Small delay between experiments
                time.sleep(1)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_exps}")
    print(f"Successful: {successful} ({successful/total_exps*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_exps*100:.1f}%)")
    print(f"Total time: {duration}")
    print(f"Average time per experiment: {total_time/total_exps:.1f}s")
    print("="*80)
    
    # Detailed results table
    print("\nDETAILED RESULTS:")
    print(f"{'Dataset':<12} {'Model':<12} {'Strategy':<12} {'Status':<12} {'Time (s)':<10}")
    print("-"*80)
    for result in results:
        status = "[OK] Success" if result["success"] else "[X] Failed"
        print(f"{result['dataset']:<12} {result['model']:<12} {result['strategy']:<12} {status:<12} {result['time']:<10.1f}")
    
    print("\n" + "="*80)
    print(f"Results saved to: results/experiments.csv")
    print("="*80)


def main():
    """Main entry point with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive federated learning experiments"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        help="Datasets to test (default: all)"
    )
    
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        help="Strategies to test (default: all)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help="Models to test (default: all)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: MNIST + CIFAR10, FedAvg + FedProx, ResNet18 only"
    )
    
    args = parser.parse_args()
    
    # Determine configurations
    if args.quick:
        datasets = ["MNIST", "CIFAR10"]
        strategies = ["FedAvg", "FedProx", "FedAgain"]
        models = ["resnet18"]
        print("\n[QUICK MODE] 6 experiments")
    else:
        datasets = args.datasets
        strategies = args.strategies
        models = args.models
    
    # Run experiments
    run_all_experiments(
        datasets=datasets,
        strategies=strategies,
        models=models
    )


if __name__ == "__main__":
    main()

