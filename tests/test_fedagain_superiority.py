"""
Comprehensive testing protocol to demonstrate FedAgain superiority.
Tests multiple scenarios comparing FedAgain against other FL strategies.
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def update_config(updates):
    """Update config.py with new settings."""
    config_path = Path('config.py')
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    with open(config_path, 'w') as f:
        for line in lines:
            updated = False
            for key, value in updates.items():
                if line.strip().startswith(f"{key} ="):
                    if isinstance(value, str):
                        f.write(f"{key} = \"{value}\"\n")
                    elif isinstance(value, bool):
                        f.write(f"{key} = {value}\n")
                    else:
                        f.write(f"{key} = {value}\n")
                    updated = True
                    break
            if not updated:
                f.write(line)


def run_experiment(name, config_updates):
    """Run a single experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}")
    
    update_config(config_updates)
    
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[OK] {name} completed successfully")
        return True
    else:
        print(f"[FAILED] {name}")
        print(result.stderr[-500:])
        return False


def test_1_baseline_clean_data():
    """
    Test 1: Baseline Comparison (Clean Data)
    Compare all strategies without corrupted clients.
    Expected: All strategies should perform similarly.
    """
    print("\n" + "="*80)
    print("TEST 1: BASELINE COMPARISON (CLEAN DATA)")
    print("="*80)
    print("Hypothesis: All strategies perform similarly on clean data")
    print("Dataset: CIFAR10, 10 clients, IID distribution, NO corruption")
    
    strategies = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedAgain"]
    
    base_config = {
        "DATASET": "CIFAR10",
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "iid",
        "CORRUPT_CLIENTS": False  # No corruption
    }
    
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"Test1-Clean-{strategy}", config)
    
    print("\n[TEST 1 COMPLETE] Check results/experiments.csv")
    print("Expected: Similar accuracy across all strategies (~85-90%)")


def test_2_corrupted_clients_light():
    """
    Test 2: Light Corruption (10% corrupted clients)
    Compare strategies with minimal corruption.
    Expected: FedAgain should maintain high accuracy.
    """
    print("\n" + "="*80)
    print("TEST 2: LIGHT CORRUPTION (10% CORRUPTED CLIENTS)")
    print("="*80)
    print("Hypothesis: FedAgain maintains better accuracy than FedAvg")
    print("Dataset: CIFAR10, 10 clients, 10% corrupted (1 client)")
    
    strategies = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedAgain"]
    
    base_config = {
        "DATASET": "CIFAR10",
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.1,  # 10% corrupted
        "CORRUPTION_SEVERITY": 3
    }
    
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"Test2-Light10%-{strategy}", config)
    
    print("\n[TEST 2 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAgain > Byzantine-robust > FedAvg")


def test_3_corrupted_clients_moderate():
    """
    Test 3: Moderate Corruption (30% corrupted clients)
    The main test - 30% corrupted clients.
    Expected: FedAgain significantly outperforms FedAvg.
    """
    print("\n" + "="*80)
    print("TEST 3: MODERATE CORRUPTION (30% CORRUPTED CLIENTS)")
    print("="*80)
    print("Hypothesis: FedAgain >> Byzantine-robust >> FedAvg")
    print("Dataset: CIFAR10, 10 clients, 30% corrupted (3 clients)")
    
    strategies = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedAgain"]
    
    base_config = {
        "DATASET": "CIFAR10",
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.3,  # 30% corrupted
        "CORRUPTION_SEVERITY": 4
    }
    
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"Test3-Moderate30%-{strategy}", config)
    
    print("\n[TEST 3 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAgain shows clear superiority")


def test_4_corrupted_clients_heavy():
    """
    Test 4: Heavy Corruption (50% corrupted clients)
    Stress test with majority corrupted.
    Expected: Only Byzantine-robust strategies survive, FedAgain best.
    """
    print("\n" + "="*80)
    print("TEST 4: HEAVY CORRUPTION (50% CORRUPTED CLIENTS)")
    print("="*80)
    print("Hypothesis: Only robust strategies maintain performance")
    print("Dataset: CIFAR10, 10 clients, 50% corrupted (5 clients)")
    
    strategies = ["FedAvg", "FedMedian", "Bulyan", "FedAgain"]
    
    base_config = {
        "DATASET": "CIFAR10",
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.4,  # 50% corrupted
        "CORRUPTION_SEVERITY": 5
    }
    
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"Test4-Heavy50%-{strategy}", config)
    
    print("\n[TEST 4 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAvg collapses, FedAgain remains strong")


def test_5_severity_progression():
    """
    Test 5: Corruption Severity Analysis
    Test FedAgain vs FedAvg at different corruption severities.
    Expected: FedAgain maintains consistent performance.
    """
    print("\n" + "="*80)
    print("TEST 5: CORRUPTION SEVERITY PROGRESSION")
    print("="*80)
    print("Hypothesis: FedAgain handles all severity levels better")
    print("Dataset: CIFAR10, 30% corrupted, varying severity")
    
    severities = [1, 2, 3, 4, 5]
    strategies = ["FedAvg", "FedAgain"]
    
    for severity in severities:
        for strategy in strategies:
            config = {
                "DATASET": "CIFAR10",
                "STRATEGY": strategy,
                "NUM_CLIENTS": 10,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": "non-iid-label",
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": severity
            }
            run_experiment(f"Test5-Severity{severity}-{strategy}", config)
    
    print("\n[TEST 5 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAgain performance gap increases with severity")


def test_6_non_iid_robustness():
    """
    Test 6: Non-IID + Corruption
    Test with non-IID data distribution (more realistic).
    Expected: FedAgain handles heterogeneity + corruption best.
    """
    print("\n" + "="*80)
    print("TEST 6: NON-IID + CORRUPTION")
    print("="*80)
    print("Hypothesis: FedAgain excels in realistic non-IID scenario")
    print("Dataset: CIFAR10, non-IID, 30% corrupted")
    
    distributions = ["iid", "non-iid-label", "non-iid-dirichlet"]
    strategies = ["FedAvg", "FedAgain"]
    
    for distribution in distributions:
        for strategy in strategies:
            config = {
                "DATASET": "CIFAR10",
                "STRATEGY": strategy,
                "NUM_CLIENTS": 10,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": distribution,
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": 3
            }
            run_experiment(f"Test6-{distribution}-{strategy}", config)
    
    print("\n[TEST 6 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAgain advantage increases with non-IID data")


def test_7_scaling_clients():
    """
    Test 7: Scaling Number of Clients
    Test with different numbers of clients (5, 10, 20).
    Expected: FedAgain scales better.
    """
    print("\n" + "="*80)
    print("TEST 7: SCALING NUMBER OF CLIENTS")
    print("="*80)
    print("Hypothesis: FedAgain maintains advantage with more clients")
    print("Dataset: CIFAR10, varying client counts, 30% corrupted")
    
    client_counts = [5, 10, 20]
    strategies = ["FedAvg", "FedAgain"]
    
    for num_clients in client_counts:
        for strategy in strategies:
            config = {
                "DATASET": "CIFAR10",
                "STRATEGY": strategy,
                "NUM_CLIENTS": num_clients,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": "non-iid-label",
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": 3
            }
            run_experiment(f"Test7-Clients{num_clients}-{strategy}", config)
    
    print("\n[TEST 7 COMPLETE] Check results/experiments.csv")
    print("Expected: FedAgain advantage consistent across scales")


def analyze_results():
    """
    Analyze all experimental results and generate summary.
    """
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    csv_path = Path("results/experiments.csv")
    if not csv_path.exists():
        print("[ERROR] No results file found!")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n=== SUMMARY BY TEST ===\n")
    
    # Test 1: Baseline (Clean)
    test1 = df[df['corrupt_clients'] == False]
    if len(test1) > 0:
        print("TEST 1 - Baseline (Clean Data):")
        print(test1.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
        if 'FedAgain' in test1['strategy'].values:
            fedagain_rank = test1.groupby('strategy')['final_accuracy'].mean().rank(ascending=False)['FedAgain']
            print(f"FedAgain rank: {fedagain_rank:.0f}")
    
    # Test 2: 10% Corruption
    test2 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.1)]
    if len(test2) > 0:
        print("\nTEST 2 - Light Corruption (10%):")
        print(test2.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
        if 'FedAgain' in test2['strategy'].values and 'FedAvg' in test2['strategy'].values:
            fedagain_acc = test2[test2['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fedavg_acc = test2[test2['strategy'] == 'FedAvg']['final_accuracy'].mean()
            print(f"FedAgain improvement over FedAvg: {((fedagain_acc - fedavg_acc) / fedavg_acc * 100):.2f}%")
    
    # Test 3: 30% Corruption
    test3 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    if len(test3) > 0:
        print("\nTEST 3 - Moderate Corruption (30%):")
        print(test3.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
        if 'FedAgain' in test3['strategy'].values and 'FedAvg' in test3['strategy'].values:
            fedagain_acc = test3[test3['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fedavg_acc = test3[test3['strategy'] == 'FedAvg']['final_accuracy'].mean()
            print(f"FedAgain improvement over FedAvg: {((fedagain_acc - fedavg_acc) / fedavg_acc * 100):.2f}%")
    
    # Test 4: 40% Corruption
    test4 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.4)]
    if len(test4) > 0:
        print("\nTEST 4 - Heavy Corruption (40%):")
        print(test4.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
        if 'FedAgain' in test4['strategy'].values and 'FedAvg' in test4['strategy'].values:
            fedagain_acc = test4[test4['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fedavg_acc = test4[test4['strategy'] == 'FedAvg']['final_accuracy'].mean()
            print(f"FedAgain improvement over FedAvg: {((fedagain_acc - fedavg_acc) / fedavg_acc * 100):.2f}%")
    
    # Test 5: Severity Progression (analyze by severity levels)
    test5_data = df[(df['corrupt_clients'] == True) & 
                    (df['corrupt_ratio'] == 0.3) & 
                    (df['corruption_severity'].isin([1, 2, 3, 4, 5]))]
    if len(test5_data) > 0:
        print("\nTEST 5 - Severity Progression (30% corrupted):")
        for severity in sorted(test5_data['corruption_severity'].unique()):
            subset = test5_data[test5_data['corruption_severity'] == severity]
            print(f"\n  Severity {severity}:")
            print(subset.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
            if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                print(f"    FedAgain vs FedAvg: {fedagain_acc:.4f} vs {fedavg_acc:.4f} " +
                      f"({((fedagain_acc - fedavg_acc) / fedavg_acc * 100):+.2f}%)")
    
    # Test 6: Non-IID Distribution Analysis
    test6_data = df[(df['corrupt_clients'] == True) & 
                    (df['corrupt_ratio'] == 0.3) &
                    (df['corruption_severity'] == 3)]
    if len(test6_data) > 0:
        distributions = test6_data['data_distribution'].unique()
        if len(distributions) > 1:
            print("\nTEST 6 - Non-IID + Corruption:")
            for dist in ['iid', 'non-iid-label', 'non-iid-dirichlet']:
                subset = test6_data[test6_data['data_distribution'] == dist]
                if len(subset) > 0:
                    print(f"\n  Distribution: {dist}")
                    print(subset.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
                    if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                        fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                        fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                        print(f"    FedAgain improvement: {((fedagain_acc - fedavg_acc) / fedavg_acc * 100):+.2f}%")
    
    # Test 7: Scaling Clients
    test7_data = df[(df['corrupt_clients'] == True) & 
                    (df['corrupt_ratio'] == 0.3) &
                    (df['corruption_severity'] == 3) &
                    (df['data_distribution'] == 'non-iid-label')]
    if len(test7_data) > 0:
        client_counts = test7_data['num_clients'].unique()
        if len(client_counts) > 1:
            print("\nTEST 7 - Scaling Number of Clients:")
            for num_clients in sorted(client_counts):
                subset = test7_data[test7_data['num_clients'] == num_clients]
                if len(subset) > 0:
                    print(f"\n  {num_clients} clients:")
                    print(subset.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False))
                    if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                        fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                        fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                        print(f"    FedAgain improvement: {((fedagain_acc - fedavg_acc) / fedavg_acc * 100):+.2f}%")
    
    # Overall statistics
    print("\n" + "="*80)
    print("=== OVERALL STATISTICS ===")
    print("="*80)
    print(f"\nTotal experiments: {len(df)}")
    if 'FedAgain' in df['strategy'].values:
        print(f"FedAgain experiments: {len(df[df['strategy'] == 'FedAgain'])}")
        print(f"FedAgain average accuracy: {df[df['strategy'] == 'FedAgain']['final_accuracy'].mean():.4f}")
    if 'FedAvg' in df['strategy'].values:
        print(f"FedAvg average accuracy: {df[df['strategy'] == 'FedAvg']['final_accuracy'].mean():.4f}")
    
    # Best strategy per corruption level
    print("\n=== BEST STRATEGY BY CORRUPTION LEVEL ===")
    for ratio in sorted(df['corrupt_ratio'].unique()):
        subset = df[df['corrupt_ratio'] == ratio]
        if len(subset) > 0:
            best = subset.groupby('strategy')['final_accuracy'].mean().idxmax()
            acc = subset.groupby('strategy')['final_accuracy'].mean().max()
            print(f"Corruption {ratio*100:.0f}%: {best} ({acc:.4f})")
    
    # FedAgain vs All Strategies (Win Rate)
    if 'FedAgain' in df['strategy'].values:
        print("\n=== FEDAGAIN WIN RATE ===")
        # Group by experiment conditions and find winner
        conditions = ['dataset', 'data_distribution', 'num_clients', 'corrupt_clients', 'corrupt_ratio', 'corruption_severity']
        available_conditions = [c for c in conditions if c in df.columns]
        
        if len(available_conditions) > 0:
            grouped = df.groupby(available_conditions)
            wins = 0
            total = 0
            for name, group in grouped:
                if len(group) > 1:  # Only count if multiple strategies tested
                    best_strategy = group.loc[group['final_accuracy'].idxmax(), 'strategy']
                    total += 1
                    if best_strategy == 'FedAgain':
                        wins += 1
            
            if total > 0:
                win_rate = (wins / total) * 100
                print(f"FedAgain wins in {wins}/{total} scenarios ({win_rate:.1f}%)")
        
        # Average improvement over FedAvg
        if 'FedAvg' in df['strategy'].values:
            fedagain_avg = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fedavg_avg = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            improvement = ((fedagain_avg - fedavg_avg) / fedavg_avg) * 100
            print(f"Average improvement over FedAvg: {improvement:+.2f}%")


def run_all_tests_for_dataset(dataset_name, dataset_display_name):
    """Run all 7 tests for a specific dataset."""
    print("\n" + "="*80)
    print(f"RUNNING ALL TESTS FOR: {dataset_display_name}")
    print("="*80)
    
    # Temporarily override test functions to use this dataset
    original_cifar10 = "CIFAR10"
    
    # We'll modify each test's dataset on the fly
    # Test 1
    # print(f"\n[{dataset_display_name}] Running Test 1: Baseline (Clean Data)")
    strategies = ["FedAvg", "FedProx", "FedMedian", "Bulyan", "FedAgain"]
    # base_config = {
    #     "DATASET": dataset_name,
    #     "NUM_CLIENTS": 10,
    #     "NUM_ROUNDS": 10,
    #     "LOCAL_EPOCHS": 1,
    #     "DATA_DISTRIBUTION": "iid",
    #     "CORRUPT_CLIENTS": False
    # }
    # for strategy in strategies:
    #     config = base_config.copy()
    #     config["STRATEGY"] = strategy
    #     run_experiment(f"{dataset_name}-Test1-Clean-{strategy}", config)
    
    # Test 2
    print(f"\n[{dataset_display_name}] Running Test 2: Light Corruption (10%)")
    base_config = {
        "DATASET": dataset_name,
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.1,
        "CORRUPTION_SEVERITY": 3
    }
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"{dataset_name}-Test2-Light10%-{strategy}", config)
    
    # Test 3
    print(f"\n[{dataset_display_name}] Running Test 3: Moderate Corruption (30%)")
    base_config = {
        "DATASET": dataset_name,
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.3,
        "CORRUPTION_SEVERITY": 4
    }
    for strategy in strategies:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"{dataset_name}-Test3-Moderate30%-{strategy}", config)
    
    # Test 4
    print(f"\n[{dataset_display_name}] Running Test 4: Heavy Corruption (40%)")
    strategies_t4 = ["FedAvg", "FedMedian", "Bulyan", "FedAgain"]
    base_config = {
        "DATASET": dataset_name,
        "NUM_CLIENTS": 10,
        "NUM_ROUNDS": 10,
        "LOCAL_EPOCHS": 1,
        "DATA_DISTRIBUTION": "non-iid-label",
        "CORRUPT_CLIENTS": True,
        "CORRUPT_CLIENT_RATIO": 0.4,
        "CORRUPTION_SEVERITY": 5
    }
    for strategy in strategies_t4:
        config = base_config.copy()
        config["STRATEGY"] = strategy
        run_experiment(f"{dataset_name}-Test4-Heavy40%-{strategy}", config)
    
    # Test 5
    print(f"\n[{dataset_display_name}] Running Test 5: Severity Progression")
    severities = [1, 2, 3, 4, 5]
    strategies_t5 = ["FedAvg", "FedAgain"]
    for severity in severities:
        for strategy in strategies_t5:
            config = {
                "DATASET": dataset_name,
                "STRATEGY": strategy,
                "NUM_CLIENTS": 10,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": "non-iid-label",
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": severity
            }
            run_experiment(f"{dataset_name}-Test5-Severity{severity}-{strategy}", config)
    
    # Test 6
    print(f"\n[{dataset_display_name}] Running Test 6: Non-IID + Corruption")
    distributions = ["iid", "non-iid-label", "non-iid-dirichlet"]
    for distribution in distributions:
        for strategy in strategies_t5:
            config = {
                "DATASET": dataset_name,
                "STRATEGY": strategy,
                "NUM_CLIENTS": 10,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": distribution,
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": 3
            }
            run_experiment(f"{dataset_name}-Test6-{distribution}-{strategy}", config)
    
    # Test 7
    print(f"\n[{dataset_display_name}] Running Test 7: Scaling Clients")
    client_counts = [5, 10, 20]
    for num_clients in client_counts:
        for strategy in strategies_t5:
            config = {
                "DATASET": dataset_name,
                "STRATEGY": strategy,
                "NUM_CLIENTS": num_clients,
                "NUM_ROUNDS": 10,
                "LOCAL_EPOCHS": 1,
                "DATA_DISTRIBUTION": "non-iid-label",
                "CORRUPT_CLIENTS": True,
                "CORRUPT_CLIENT_RATIO": 0.3,
                "CORRUPTION_SEVERITY": 3
            }
            run_experiment(f"{dataset_name}-Test7-Clients{num_clients}-{strategy}", config)
    
    print(f"\n[{dataset_display_name}] ALL TESTS COMPLETE!")


def main():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("FEDAGAIN SUPERIORITY TEST SUITE - MULTI-DATASET")
    print("="*80)
    print("\nThis will run comprehensive tests to demonstrate FedAgain's superiority")
    print("over other federated learning strategies, especially under corruption.")
    print("\nTests include:")
    print("  1. Baseline (Clean Data)")
    print("  2. Light Corruption (10%)")
    print("  3. Moderate Corruption (30%)")
    print("  4. Heavy Corruption (40%)")
    print("  5. Severity Progression")
    print("  6. Non-IID + Corruption")
    print("  7. Scaling Clients")
    print("\nDatasets:")
    print("  - CIFAR10")
    print("  - MNIST")
    print("  - Jonathan El-Beze: MIX, SEC, SUR")
    print("  - Michel Daudon: MIX, SEC, SUR")
    print("  - MyStone")
    print("\nEstimated time: 10-24 hours depending on hardware")
    
    response = input("\nRun complete test suite on all datasets? (y/n): ")
    if response.lower() != 'y':
        print("\nYou can run individual tests or datasets:")
        print("  python test_fedagain_superiority.py --cifar10       # CIFAR10 only")
        print("  python test_fedagain_superiority.py --jonathan-mix  # Jonathan MIX")
        print("  python test_fedagain_superiority.py --all-kidney    # All kidney datasets")
        print("  python test_fedagain_superiority.py --analyze       # Analyze results")
        return
    
    # Define all datasets to test
    datasets = [
        ("CIFAR10", "CIFAR10"),
        ("MNIST", "MNIST"),
        # ("JONATHAN_MIX", "Jonathan El-Beze (MIX)"),
        # ("JONATHAN_SEC", "Jonathan El-Beze (SEC)"),
        ("JONATHAN_SUR", "Jonathan El-Beze (SUR)"),
        ("DAUDON_MIX", "Michel Daudon (MIX)"),
        ("DAUDON_SEC", "Michel Daudon (SEC)"),
        ("DAUDON_SUR", "Michel Daudon (SUR)"),
        ("MYSTONE", "MyStone")
    ]
    
    # Run all tests for each dataset
    for dataset_name, dataset_display in datasets:
        run_all_tests_for_dataset(dataset_name, dataset_display)
    
    # Analyze results
    # analyze_results()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE!")
    print("="*80)
    print("\nResults saved to: results/experiments.csv")
    print("Run: python test_fedagain_superiority.py --analyze")
    print("     to re-analyze results")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Individual tests (CIFAR10 only for backward compatibility)
        if arg == "--test1":
            test_1_baseline_clean_data()
        elif arg == "--test2":
            test_2_corrupted_clients_light()
        elif arg == "--test3":
            test_3_corrupted_clients_moderate()
        elif arg == "--test4":
            test_4_corrupted_clients_heavy()
        elif arg == "--test5":
            test_5_severity_progression()
        elif arg == "--test6":
            test_6_non_iid_robustness()
        elif arg == "--test7":
            test_7_scaling_clients()
        
        # Dataset-specific test suites
        elif arg == "--cifar10":
            run_all_tests_for_dataset("CIFAR10", "CIFAR10")
        elif arg == "--mnist":
            run_all_tests_for_dataset("MNIST", "MNIST")
        elif arg == "--jonathan-mix":
            run_all_tests_for_dataset("JONATHAN_MIX", "Jonathan El-Beze (MIX)")
        elif arg == "--jonathan-sec":
            run_all_tests_for_dataset("JONATHAN_SEC", "Jonathan El-Beze (SEC)")
        elif arg == "--jonathan-sur":
            run_all_tests_for_dataset("JONATHAN_SUR", "Jonathan El-Beze (SUR)")
        elif arg == "--daudon-mix":
            run_all_tests_for_dataset("DAUDON_MIX", "Michel Daudon (MIX)")
        elif arg == "--daudon-sec":
            run_all_tests_for_dataset("DAUDON_SEC", "Michel Daudon (SEC)")
        elif arg == "--daudon-sur":
            run_all_tests_for_dataset("DAUDON_SUR", "Michel Daudon (SUR)")
        elif arg == "--mystone":
            run_all_tests_for_dataset("MYSTONE", "MyStone")
        
        # Group options
        elif arg == "--all-kidney":
            print("Running all kidney stone datasets...")
            kidney_datasets = [
                # ("JONATHAN_MIX", "Jonathan El-Beze (MIX)"),
                ("JONATHAN_SEC", "Jonathan El-Beze (SEC)"),
                # ("JONATHAN_SUR", "Jonathan El-Beze (SUR)"),
                # ("DAUDON_MIX", "Michel Daudon (MIX)"),
                # ("DAUDON_SEC", "Michel Daudon (SEC)"),
                # ("DAUDON_SUR", "Michel Daudon (SUR)"),
                # ("MYSTONE", "MyStone")
            ]
            for dataset_name, dataset_display in kidney_datasets:
                run_all_tests_for_dataset(dataset_name, dataset_display)
            # analyze_results()
        
        # Analysis
        elif arg == "--analyze":
            analyze_results()
        
        else:
            print(f"Unknown argument: {arg}")
            print("\nAvailable options:")
            print("  Individual tests (CIFAR10):")
            print("    --test1, --test2, --test3, --test4, --test5, --test6, --test7")
            print("\n  Dataset-specific:")
            print("    --cifar10, --mnist, --jonathan-mix, --jonathan-sec, --jonathan-sur")
            print("    --daudon-mix, --daudon-sec, --daudon-sur, --mystone")
            print("\n  Groups:")
            print("    --all-kidney    # All kidney stone datasets")
            print("\n  Analysis:")
            print("    --analyze       # Analyze existing results")
    else:
        main()

