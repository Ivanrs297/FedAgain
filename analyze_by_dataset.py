"""
Multi-Dataset Analysis Script
Analyzes experimental results and generates visualizations organized by dataset.
Each dataset gets its own folder with analysis summary and plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def create_dataset_folder(dataset_name):
    """Create a folder for dataset-specific results."""
    folder_path = Path(f"results/{dataset_name}")
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def plot_corruption_comparison(df, output_dir, dataset_name):
    """Plot 1: Accuracy vs Corruption Ratio"""
    strategies = df['strategy'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        if len(strategy_data) == 0:
            continue
        
        # Group by corruption ratio
        grouped = strategy_data.groupby('corrupt_ratio')['final_accuracy'].mean()
        
        if len(grouped) > 0:
            ax.plot(grouped.index * 100, grouped.values, 
                    marker='o', linewidth=2, label=strategy, markersize=8)
    
    ax.set_xlabel('Corrupted Clients (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Accuracy vs Corruption Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / 'plot1_corruption_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_strategy_comparison_bar(df, output_dir, dataset_name):
    """Plot 2: Bar Chart Comparison at 30% Corruption"""
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
    ax.set_title(f'{dataset_name}: Strategy Comparison (30% Corruption)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'plot2_strategy_bar_30pct.png'
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_severity_impact(df, output_dir, dataset_name):
    """Plot 3: Impact of Corruption Severity"""
    severity_data = df[df['corrupt_ratio'] == 0.3]
    
    if len(severity_data) == 0:
        print("[SKIP] No severity data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in ['FedAvg', 'FedAgain']:
        strategy_data = severity_data[severity_data['strategy'] == strategy]
        if len(strategy_data) == 0:
            continue
        
        grouped = strategy_data.groupby('corruption_severity')['final_accuracy'].mean()
        
        if len(grouped) > 0:
            ax.plot(grouped.index, grouped.values,
                    marker='o', linewidth=2.5, label=strategy, markersize=10)
    
    ax.set_xlabel('Corruption Severity', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Robustness to Corruption Severity (30% Corrupted)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / 'plot3_severity_impact.png'
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_improvement_percentage(df, output_dir, dataset_name):
    """Plot 4: FedAgain Improvement Percentage"""
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
    ax.set_title(f'{dataset_name}: FedAgain Performance Improvement', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'plot4_improvement_percentage.png'
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_f1_comparison(df, output_dir, dataset_name):
    """Plot 5: F1 Score Comparison"""
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
    
    plt.suptitle(f'{dataset_name}: Multiple Metrics (30% Corruption)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'plot5_f1_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Saved: {output_path}")
    plt.close()


def generate_analysis_text(df, dataset_name):
    """Generate comprehensive analysis text."""
    analysis = []
    analysis.append(f"# FedAgain Experimental Analysis: {dataset_name}")
    analysis.append(f"**Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    analysis.append(f"**Total Experiments: {len(df)}**\n")
    analysis.append("---\n")
    
    # Executive Summary
    analysis.append("## Executive Summary\n")
    
    total_exp = len(df)
    strategies = df['strategy'].unique()
    analysis.append(f"This analysis evaluates {len(strategies)} federated learning strategies on the {dataset_name} dataset.\n")
    
    if 'FedAgain' in strategies and 'FedAvg' in strategies:
        fedagain_avg = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        fedavg_avg = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
        improvement = ((fedagain_avg - fedavg_avg) / fedavg_avg) * 100
        
        analysis.append(f"**Overall Performance:**")
        analysis.append(f"- FedAgain Average Accuracy: {fedagain_avg:.4f}")
        analysis.append(f"- FedAvg Average Accuracy: {fedavg_avg:.4f}")
        analysis.append(f"- Overall Improvement: {improvement:+.2f}%\n")
    
    analysis.append("---\n")
    
    # Strategy Performance Summary
    analysis.append("## Strategy Performance Summary\n")
    strategy_stats = df.groupby('strategy')['final_accuracy'].agg(['mean', 'std', 'min', 'max', 'count'])
    analysis.append("| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |")
    analysis.append("|----------|----------|---------|-----|-----|-------------|")
    for strategy in strategy_stats.sort_values('mean', ascending=False).index:
        stats = strategy_stats.loc[strategy]
        analysis.append(f"| {strategy} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {int(stats['count'])} |")
    analysis.append("\n---\n")
    
    # Test 1: Baseline (Clean Data)
    analysis.append("## Test 1: Baseline (Clean Data)\n")
    test1 = df[df['corrupt_clients'] == False]
    if len(test1) > 0:
        analysis.append("**Setup:** No corrupted clients\n")
        test1_results = test1.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        analysis.append("**Results:**")
        analysis.append("| Strategy | Accuracy |")
        analysis.append("|----------|----------|")
        for strategy, acc in test1_results.items():
            analysis.append(f"| {strategy} | {acc:.4f} |")
        
        if 'FedAgain' in test1_results.index:
            rank = list(test1_results.index).index('FedAgain') + 1
            analysis.append(f"\n**FedAgain Rank:** {rank} of {len(test1_results)}")
        analysis.append("\n")
    else:
        analysis.append("No clean data experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 2: Light Corruption (10%)
    analysis.append("## Test 2: Light Corruption (10% Corrupted Clients)\n")
    test2 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.1)]
    if len(test2) > 0:
        analysis.append("**Setup:** 10% corrupted clients, non-IID label distribution\n")
        test2_results = test2.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        analysis.append("**Results:**")
        analysis.append("| Strategy | Accuracy |")
        analysis.append("|----------|----------|")
        for strategy, acc in test2_results.items():
            analysis.append(f"| {strategy} | {acc:.4f} |")
        
        if 'FedAgain' in test2_results.index and 'FedAvg' in test2_results.index:
            fedagain_acc = test2_results['FedAgain']
            fedavg_acc = test2_results['FedAvg']
            improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
            analysis.append(f"\n**FedAgain vs FedAvg:** {improvement:+.2f}% improvement")
        analysis.append("\n")
    else:
        analysis.append("No 10% corruption experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 3: Moderate Corruption (30%)
    analysis.append("## Test 3: Moderate Corruption (30% Corrupted Clients)\n")
    test3 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    if len(test3) > 0:
        analysis.append("**Setup:** 30% corrupted clients, non-IID label distribution\n")
        test3_results = test3.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        analysis.append("**Results:**")
        analysis.append("| Strategy | Accuracy |")
        analysis.append("|----------|----------|")
        for strategy, acc in test3_results.items():
            analysis.append(f"| {strategy} | {acc:.4f} |")
        
        if 'FedAgain' in test3_results.index and 'FedAvg' in test3_results.index:
            fedagain_acc = test3_results['FedAgain']
            fedavg_acc = test3_results['FedAvg']
            improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
            analysis.append(f"\n**FedAgain vs FedAvg:** {improvement:+.2f}% improvement")
        analysis.append("\n")
    else:
        analysis.append("No 30% corruption experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 4: Heavy Corruption (40%)
    analysis.append("## Test 4: Heavy Corruption (40% Corrupted Clients)\n")
    test4 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.4)]
    if len(test4) > 0:
        analysis.append("**Setup:** 40% corrupted clients, non-IID label distribution\n")
        test4_results = test4.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        analysis.append("**Results:**")
        analysis.append("| Strategy | Accuracy |")
        analysis.append("|----------|----------|")
        for strategy, acc in test4_results.items():
            analysis.append(f"| {strategy} | {acc:.4f} |")
        
        if 'FedAgain' in test4_results.index and 'FedAvg' in test4_results.index:
            fedagain_acc = test4_results['FedAgain']
            fedavg_acc = test4_results['FedAvg']
            improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
            analysis.append(f"\n**FedAgain vs FedAvg:** {improvement:+.2f}% improvement")
        analysis.append("\n")
    else:
        analysis.append("No 40% corruption experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 5: Severity Progression
    analysis.append("## Test 5: Corruption Severity Progression\n")
    test5_data = df[(df['corrupt_clients'] == True) & 
                    (df['corrupt_ratio'] == 0.3)]
    if len(test5_data) > 0 and 'corruption_severity' in df.columns:
        severities = sorted(test5_data['corruption_severity'].unique())
        if len(severities) > 1:
            analysis.append("**Setup:** 30% corrupted clients, varying severity levels\n")
            analysis.append("| Severity | FedAgain | FedAvg | Improvement |")
            analysis.append("|----------|----------|--------|-------------|")
            
            for severity in severities:
                subset = test5_data[test5_data['corruption_severity'] == severity]
                if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                    fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
                    analysis.append(f"| {severity} | {fedagain_acc:.4f} | {fedavg_acc:.4f} | {improvement:+.2f}% |")
            analysis.append("\n")
    else:
        analysis.append("No severity progression experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 6: Non-IID Distribution Analysis
    analysis.append("## Test 6: Non-IID + Corruption Analysis\n")
    test6_data = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    if len(test6_data) > 0 and 'data_distribution' in df.columns:
        distributions = test6_data['data_distribution'].unique()
        if len(distributions) > 1:
            analysis.append("**Setup:** 30% corrupted clients, various data distributions\n")
            analysis.append("| Distribution | FedAgain | FedAvg | Improvement |")
            analysis.append("|--------------|----------|--------|-------------|")
            
            for dist in ['iid', 'non-iid-label', 'non-iid-dirichlet']:
                subset = test6_data[test6_data['data_distribution'] == dist]
                if len(subset) > 0 and 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                    fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
                    analysis.append(f"| {dist} | {fedagain_acc:.4f} | {fedavg_acc:.4f} | {improvement:+.2f}% |")
            analysis.append("\n")
    else:
        analysis.append("No distribution variation experiments found.\n")
    
    analysis.append("---\n")
    
    # Test 7: Scaling Clients
    analysis.append("## Test 7: Scaling Number of Clients\n")
    test7_data = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
    if len(test7_data) > 0 and 'num_clients' in df.columns:
        client_counts = sorted(test7_data['num_clients'].unique())
        if len(client_counts) > 1:
            analysis.append("**Setup:** 30% corrupted clients, varying client counts\n")
            analysis.append("| Clients | FedAgain | FedAvg | Improvement |")
            analysis.append("|---------|----------|--------|-------------|")
            
            for num_clients in client_counts:
                subset = test7_data[test7_data['num_clients'] == num_clients]
                if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                    fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
                    analysis.append(f"| {num_clients} | {fedagain_acc:.4f} | {fedavg_acc:.4f} | {improvement:+.2f}% |")
            analysis.append("\n")
    else:
        analysis.append("No scaling experiments found.\n")
    
    analysis.append("---\n")
    
    # Key Findings
    analysis.append("## Key Findings\n")
    
    if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
        # Find best and worst scenarios
        test_scenarios = []
        
        # 10% corruption
        test2 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.1)]
        if len(test2) > 0:
            if 'FedAgain' in test2['strategy'].values and 'FedAvg' in test2['strategy'].values:
                fa = test2[test2['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fv = test2[test2['strategy'] == 'FedAvg']['final_accuracy'].mean()
                test_scenarios.append(('10% Corruption', (fa - fv) / fv * 100))
        
        # 30% corruption
        test3 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
        if len(test3) > 0:
            if 'FedAgain' in test3['strategy'].values and 'FedAvg' in test3['strategy'].values:
                fa = test3[test3['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fv = test3[test3['strategy'] == 'FedAvg']['final_accuracy'].mean()
                test_scenarios.append(('30% Corruption', (fa - fv) / fv * 100))
        
        # 40% corruption
        test4 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.4)]
        if len(test4) > 0:
            if 'FedAgain' in test4['strategy'].values and 'FedAvg' in test4['strategy'].values:
                fa = test4[test4['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fv = test4[test4['strategy'] == 'FedAvg']['final_accuracy'].mean()
                test_scenarios.append(('40% Corruption', (fa - fv) / fv * 100))
        
        if test_scenarios:
            test_scenarios.sort(key=lambda x: x[1], reverse=True)
            analysis.append("### Best Scenarios for FedAgain:")
            for scenario, improvement in test_scenarios[:3]:
                if improvement > 0:
                    analysis.append(f"- **{scenario}:** {improvement:+.2f}% improvement over FedAvg")
            
            analysis.append("\n### Challenging Scenarios:")
            for scenario, improvement in reversed(test_scenarios[-3:]):
                if improvement < 0:
                    analysis.append(f"- **{scenario}:** {improvement:.2f}% vs FedAvg")
    
    analysis.append("\n---\n")
    
    # Recommendations
    analysis.append("## Recommendations\n")
    analysis.append(f"Based on the analysis of {dataset_name} dataset:\n")
    
    if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
        fedagain_avg = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        fedavg_avg = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
        
        if fedagain_avg > fedavg_avg:
            analysis.append(f"✅ **FedAgain shows overall improvement** ({((fedagain_avg - fedavg_avg) / fedavg_avg * 100):+.2f}%) over FedAvg")
        else:
            analysis.append(f"⚠️ **FedAgain underperforms** ({((fedagain_avg - fedavg_avg) / fedavg_avg * 100):.2f}%) compared to FedAvg")
    
    analysis.append("\n### Suggested Use Cases:")
    analysis.append("- Consider dataset-specific characteristics when choosing FL strategy")
    analysis.append("- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption")
    analysis.append("- Test with multiple data distributions to ensure robustness\n")
    
    analysis.append("---\n")
    analysis.append(f"\n*Analysis generated for {dataset_name} dataset*\n")
    analysis.append(f"*Timestamp: {pd.Timestamp.now()}*")
    
    return "\n".join(analysis)


def analyze_dataset(csv_path, dataset_name=None):
    """Analyze a specific dataset's results."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # If no dataset name provided, use the one from CSV or default
    if dataset_name is None:
        if 'dataset' in df.columns and len(df['dataset'].unique()) == 1:
            dataset_name = df['dataset'].unique()[0]
        else:
            dataset_name = "UNKNOWN"
    
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total experiments: {len(df)}")
    print(f"Strategies: {', '.join(df['strategy'].unique())}")
    
    # Create output directory
    output_dir = create_dataset_folder(dataset_name)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_corruption_comparison(df, output_dir, dataset_name)
    plot_strategy_comparison_bar(df, output_dir, dataset_name)
    plot_severity_impact(df, output_dir, dataset_name)
    plot_improvement_percentage(df, output_dir, dataset_name)
    plot_f1_comparison(df, output_dir, dataset_name)
    
    # Generate analysis summary
    print("\nGenerating analysis summary...")
    analysis_text = generate_analysis_text(df, dataset_name)
    summary_path = output_dir / 'ANALYSIS_SUMMARY.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    print(f"[OK] Saved: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE FOR {dataset_name}")
    print(f"{'='*80}")
    print(f"\nGenerated files in: {output_dir}/")
    print("  - ANALYSIS_SUMMARY.md")
    print("  - plot1_corruption_comparison.png")
    print("  - plot2_strategy_bar_30pct.png")
    print("  - plot3_severity_impact.png")
    print("  - plot4_improvement_percentage.png")
    print("  - plot5_f1_comparison.png")
    
    return output_dir


def analyze_all_datasets_from_csv(csv_path):
    """Analyze all datasets found in a CSV file."""
    df = pd.read_csv(csv_path)
    
    if 'dataset' not in df.columns:
        print("[ERROR] CSV file must have a 'dataset' column")
        return
    
    datasets = df['dataset'].unique()
    print(f"\n{'='*80}")
    print(f"FOUND {len(datasets)} DATASET(S) IN CSV")
    print(f"{'='*80}")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    # Analyze each dataset
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        # Save dataset-specific CSV
        dataset_csv_path = Path("results") / dataset / "experiments.csv"
        dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(dataset_csv_path, index=False)
        print(f"\n[OK] Saved dataset CSV: {dataset_csv_path}")
        
        # Analyze
        analyze_dataset(dataset_csv_path, dataset)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "results/experiments.csv"
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        print("\nUsage:")
        print("  python analyze_by_dataset.py [csv_path]")
        print("\nExample:")
        print("  python analyze_by_dataset.py results/experiments.csv")
        return
    
    print("\n" + "="*80)
    print("MULTI-DATASET ANALYSIS TOOL")
    print("="*80)
    print(f"Input CSV: {csv_path}")
    
    # Analyze all datasets in the CSV
    analyze_all_datasets_from_csv(csv_path)
    
    print("\n" + "="*80)
    print("ALL DATASETS ANALYZED!")
    print("="*80)
    print("\nCheck results/ directory for dataset-specific folders")


if __name__ == "__main__":
    main()

