"""
Visualization script to create plots demonstrating FedAgain superiority.
Generates comparison charts from experimental results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_corruption_comparison():
    """
    Plot 1: Accuracy vs Corruption Ratio
    Shows how different strategies degrade with increasing corruption.
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Filter relevant experiments
    strategies = ['FedAvg', 'FedProx', 'FedMedian', 'Bulyan', 'FedAgain']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        if len(strategy_data) == 0:
            continue
        
        # Group by corruption ratio
        grouped = strategy_data.groupby('corrupt_ratio')['final_accuracy'].mean()
        
        ax.plot(grouped.index * 100, grouped.values, 
                marker='o', linewidth=2, label=strategy, markersize=8)
    
    ax.set_xlabel('Corrupted Clients (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('FL Strategy Robustness: Accuracy vs Corruption Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/plot1_corruption_comparison.png', dpi=300)
    print("[OK] Saved: results/plot1_corruption_comparison.png")
    plt.close()


def plot_strategy_comparison_bar():
    """
    Plot 2: Bar Chart Comparison at 30% Corruption
    Direct comparison of all strategies at the key test point.
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Filter 30% corruption experiments
    test_data = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    
    if len(test_data) == 0:
        print("[SKIP] No 30% corruption data available")
        return
    
    # Get mean accuracy per strategy
    strategy_acc = test_data.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728' if s == 'FedAvg' else '#2ca02c' if s == 'FedAgain' else '#1f77b4' 
              for s in strategy_acc.index]
    
    bars = ax.bar(range(len(strategy_acc)), strategy_acc.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(strategy_acc)))
    ax.set_xticklabels(strategy_acc.index, fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Strategy Comparison: 30% Corrupted Clients', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/plot2_strategy_bar_30pct.png', dpi=300)
    print("[OK] Saved: results/plot2_strategy_bar_30pct.png")
    plt.close()


def plot_severity_impact():
    """
    Plot 3: Impact of Corruption Severity
    Shows FedAgain vs FedAvg across severity levels.
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Filter severity progression experiments
    severity_data = df[df['corrupt_ratio'] == 0.3]  # 30% corruption
    
    if len(severity_data) == 0:
        print("[SKIP] No severity data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in ['FedAvg', 'FedAgain']:
        strategy_data = severity_data[severity_data['strategy'] == strategy]
        if len(strategy_data) == 0:
            continue
        
        grouped = strategy_data.groupby('corruption_severity')['final_accuracy'].mean()
        
        ax.plot(grouped.index, grouped.values,
                marker='o', linewidth=2.5, label=strategy, markersize=10)
    
    ax.set_xlabel('Corruption Severity', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Robustness to Corruption Severity (30% Corrupted Clients)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/plot3_severity_impact.png', dpi=300)
    print("[OK] Saved: results/plot3_severity_impact.png")
    plt.close()


def plot_improvement_percentage():
    """
    Plot 4: FedAgain Improvement Percentage
    Shows percentage improvement of FedAgain over FedAvg.
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Calculate improvement at each corruption level
    corruption_levels = df['corrupt_ratio'].unique()
    improvements = []
    
    for ratio in sorted(corruption_levels):
        fedagain_acc = df[(df['strategy'] == 'FedAgain') & 
                         (df['corrupt_ratio'] == ratio)]['final_accuracy'].mean()
        fedavg_acc = df[(df['strategy'] == 'FedAvg') & 
                       (df['corrupt_ratio'] == ratio)]['final_accuracy'].mean()
        
        if not np.isnan(fedagain_acc) and not np.isnan(fedavg_acc) and fedavg_acc > 0:
            improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
            improvements.append((ratio * 100, improvement))
    
    if len(improvements) == 0:
        print("[SKIP] No improvement data available")
        return
    
    improvements = sorted(improvements)
    ratios, impr_values = zip(*improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(ratios, impr_values, color='#2ca02c', alpha=0.8, width=5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Corrupted Clients (%)', fontsize=12)
    ax.set_ylabel('Improvement over FedAvg (%)', fontsize=12)
    ax.set_title('FedAgain Performance Improvement', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/plot4_improvement_percentage.png', dpi=300)
    print("[OK] Saved: results/plot4_improvement_percentage.png")
    plt.close()


def plot_f1_comparison():
    """
    Plot 5: F1 Score Comparison
    Compare F1 scores (more robust metric than accuracy).
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Filter 30% corruption
    test_data = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    
    if len(test_data) == 0 or 'final_f1' not in df.columns:
        print("[SKIP] No F1 score data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy
    strategy_acc = test_data.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
    ax1.barh(range(len(strategy_acc)), strategy_acc.values, 
             color=['#2ca02c' if s == 'FedAgain' else '#1f77b4' for s in strategy_acc.index])
    ax1.set_yticks(range(len(strategy_acc)))
    ax1.set_yticklabels(strategy_acc.index)
    ax1.set_xlabel('Accuracy', fontsize=11)
    ax1.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 1.0])
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot 2: F1 Score
    strategy_f1 = test_data.groupby('strategy')['final_f1'].mean().sort_values(ascending=False)
    ax2.barh(range(len(strategy_f1)), strategy_f1.values,
             color=['#2ca02c' if s == 'FedAgain' else '#1f77b4' for s in strategy_f1.index])
    ax2.set_yticks(range(len(strategy_f1)))
    ax2.set_yticklabels(strategy_f1.index)
    ax2.set_xlabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1.0])
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle('Strategy Comparison: Multiple Metrics (30% Corruption)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/plot5_f1_comparison.png', dpi=300)
    print("[OK] Saved: results/plot5_f1_comparison.png")
    plt.close()


def generate_summary_table():
    """
    Generate LaTeX table for paper.
    """
    df = pd.read_csv("results/experiments.csv")
    
    # Summary table: corruption ratio vs strategies
    corruption_levels = [0.0, 0.1, 0.3, 0.5]
    strategies = ['FedAvg', 'FedProx', 'FedMedian', 'Bulyan', 'FedAgain']
    
    print("\n" + "="*80)
    print("LATEX TABLE (copy to paper)")
    print("="*80)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Test Accuracy Comparison Under Different Corruption Levels}")
    print("\\begin{tabular}{l" + "c" * len(corruption_levels) + "}")
    print("\\hline")
    print("Strategy & " + " & ".join([f"{int(r*100)}\\%" for r in corruption_levels]) + " \\\\")
    print("\\hline")
    
    for strategy in strategies:
        row = [strategy]
        for ratio in corruption_levels:
            acc = df[(df['strategy'] == strategy) & 
                    (df['corrupt_ratio'] == ratio)]['final_accuracy'].mean()
            if not np.isnan(acc):
                row.append(f"{acc:.3f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\n")


def main():
    """Generate all visualizations."""
    print("\n" + "="*80)
    print("FEDAGAIN RESULTS VISUALIZATION")
    print("="*80)
    
    csv_path = Path("results/experiments.csv")
    if not csv_path.exists():
        print("[ERROR] No results file found!")
        print("Run experiments first: python test_fedagain_superiority.py")
        return
    
    print("\nGenerating plots...")
    
    try:
        plot_corruption_comparison()
        plot_strategy_comparison_bar()
        plot_severity_impact()
        plot_improvement_percentage()
        plot_f1_comparison()
        generate_summary_table()
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print("\nGenerated plots:")
        print("  1. results/plot1_corruption_comparison.png")
        print("  2. results/plot2_strategy_bar_30pct.png")
        print("  3. results/plot3_severity_impact.png")
        print("  4. results/plot4_improvement_percentage.png")
        print("  5. results/plot5_f1_comparison.png")
        print("\nLaTeX table printed above for paper inclusion")
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")


if __name__ == "__main__":
    main()

