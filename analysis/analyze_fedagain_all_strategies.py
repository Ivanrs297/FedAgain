"""
Comprehensive FedAgain Analysis vs All Strategies
Compares FedAgain against FedAvg, FedProx, FedMedian, and Bulyan
across multiple datasets with detailed tables and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def load_all_datasets():
    """Load experiments from consolidated experiments.csv and split by dataset."""
    csv_path = Path("results/experiments.csv")
    datasets = {}
    
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found!")
        return datasets
    
    # Read consolidated CSV
    df = pd.read_csv(csv_path)
    
    # Split by dataset
    if 'dataset' in df.columns:
        for dataset_name in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset_name].copy()
            datasets[dataset_name] = dataset_df
            print(f"  [OK] Loaded {dataset_name}: {len(dataset_df)} experiments")
    else:
        print("[ERROR] 'dataset' column not found in experiments.csv")
    
    return datasets


def generate_comprehensive_tables(datasets):
    """Generate comprehensive comparison tables for all strategies."""
    tables = []
    
    # Get all unique strategies
    all_strategies = set()
    for df in datasets.values():
        all_strategies.update(df['strategy'].unique())
    all_strategies = sorted(all_strategies)
    
    print(f"  Strategies found: {', '.join(all_strategies)}")
    
    # Table 1: Overall Performance - All Strategies by Dataset
    tables.append("\n## Table 1: Overall Performance - All Strategies by Dataset\n")
    tables.append("Mean accuracy ± standard deviation\n")
    
    overall_data = []
    for dataset_name in sorted(datasets.keys()):
        df = datasets[dataset_name]
        row = {'Dataset': dataset_name}
        
        for strategy in all_strategies:
            if strategy in df['strategy'].values:
                acc_mean = df[df['strategy'] == strategy]['final_accuracy'].mean()
                acc_std = df[df['strategy'] == strategy]['final_accuracy'].std()
                row[strategy] = f"{acc_mean:.4f}±{acc_std:.4f}"
            else:
                row[strategy] = '--'
        
        # Find best strategy
        strategy_means = df.groupby('strategy')['final_accuracy'].mean()
        best_strategy = strategy_means.idxmax()
        row['Best'] = best_strategy
        
        overall_data.append(row)
    
    overall_df = pd.DataFrame(overall_data)
    tables.append("| " + " | ".join(overall_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in overall_df.columns]) + "|")
    for _, row in overall_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 2: FedAgain vs Each Strategy - Head-to-Head Comparison
    tables.append("\n\n## Table 2: FedAgain Head-to-Head Comparison\n")
    tables.append("Improvement of FedAgain over each strategy (percentage)\n")
    
    comparison_strategies = [s for s in all_strategies if s != 'FedAgain']
    comparison_data = []
    
    for dataset_name in sorted(datasets.keys()):
        df = datasets[dataset_name]
        if 'FedAgain' not in df['strategy'].values:
            continue
            
        fedagain_acc = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        row = {'Dataset': dataset_name, 'FedAgain Acc': f"{fedagain_acc:.4f}"}
        
        for strategy in comparison_strategies:
            if strategy in df['strategy'].values:
                other_acc = df[df['strategy'] == strategy]['final_accuracy'].mean()
                improvement = ((fedagain_acc - other_acc) / other_acc) * 100
                row[f'vs {strategy}'] = f"{improvement:+.2f}%"
            else:
                row[f'vs {strategy}'] = '--'
        
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        tables.append("| " + " | ".join(comparison_df.columns) + " |")
        tables.append("|" + "|".join(['---' for _ in comparison_df.columns]) + "|")
        for _, row in comparison_df.iterrows():
            tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 3: Strategy Rankings by Dataset
    tables.append("\n\n## Table 3: Strategy Rankings by Dataset\n")
    tables.append("Rank based on mean accuracy (1=best)\n")
    
    ranking_data = []
    for dataset_name in sorted(datasets.keys()):
        df = datasets[dataset_name]
        strategy_means = df.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        
        row = {'Dataset': dataset_name}
        for rank, (strategy, acc) in enumerate(strategy_means.items(), 1):
            row[strategy] = f"{rank}"
        
        ranking_data.append(row)
    
    ranking_df = pd.DataFrame(ranking_data)
    tables.append("| " + " | ".join(ranking_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in ranking_df.columns]) + "|")
    for _, row in ranking_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 4: Performance by Corruption Level - All Strategies
    tables.append("\n\n## Table 4: Performance by Corruption Level - All Strategies\n")
    
    for ratio in [0.1, 0.3, 0.4]:
        tables.append(f"\n### {int(ratio*100)}% Corruption\n")
        
        corruption_data = []
        for dataset_name in sorted(datasets.keys()):
            df = datasets[dataset_name]
            subset = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == ratio)]
            
            if len(subset) == 0:
                continue
            
            row = {'Dataset': dataset_name}
            for strategy in all_strategies:
                if strategy in subset['strategy'].values:
                    acc = subset[subset['strategy'] == strategy]['final_accuracy'].mean()
                    row[strategy] = f"{acc:.4f}"
                else:
                    row[strategy] = '--'
            
            # Best strategy for this corruption level
            strategy_accs = subset.groupby('strategy')['final_accuracy'].mean()
            best = strategy_accs.idxmax()
            row['Best'] = best
            
            corruption_data.append(row)
        
        if corruption_data:
            corruption_df = pd.DataFrame(corruption_data)
            tables.append("| " + " | ".join(corruption_df.columns) + " |")
            tables.append("|" + "|".join(['---' for _ in corruption_df.columns]) + "|")
            for _, row in corruption_df.iterrows():
                tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 5: Win Rate by Strategy
    tables.append("\n\n## Table 5: Win Rate Analysis - All Strategies\n")
    tables.append("How often each strategy achieves best performance\n")
    
    win_rate_data = []
    for dataset_name in sorted(datasets.keys()):
        df = datasets[dataset_name]
        
        conditions = ['corrupt_ratio', 'corruption_severity', 'data_distribution']
        available_conditions = [c for c in conditions if c in df.columns]
        
        strategy_wins = {s: 0 for s in all_strategies if s in df['strategy'].values}
        total = 0
        
        if len(available_conditions) > 0:
            grouped = df.groupby(available_conditions)
            for name, group in grouped:
                if len(group['strategy'].unique()) > 1:
                    best = group.loc[group['final_accuracy'].idxmax(), 'strategy']
                    total += 1
                    if best in strategy_wins:
                        strategy_wins[best] += 1
        
        row = {'Dataset': dataset_name, 'Total Scenarios': total}
        for strategy in all_strategies:
            if strategy in strategy_wins:
                wins = strategy_wins[strategy]
                win_rate = (wins / total * 100) if total > 0 else 0
                row[f'{strategy}'] = f"{wins} ({win_rate:.1f}%)"
            else:
                row[f'{strategy}'] = '--'
        
        win_rate_data.append(row)
    
    win_rate_df = pd.DataFrame(win_rate_data)
    tables.append("| " + " | ".join(win_rate_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in win_rate_df.columns]) + "|")
    for _, row in win_rate_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 6: Statistical Summary - All Strategies
    tables.append("\n\n## Table 6: Cross-Dataset Statistical Summary\n")
    
    stats_data = []
    for strategy in all_strategies:
        strategy_accs = []
        for df in datasets.values():
            if strategy in df['strategy'].values:
                strategy_accs.append(df[df['strategy'] == strategy]['final_accuracy'].mean())
        
        if strategy_accs:
            stats_data.append({
                'Strategy': strategy,
                'Mean Acc': f"{np.mean(strategy_accs):.4f}",
                'Std Dev': f"{np.std(strategy_accs):.4f}",
                'Min': f"{np.min(strategy_accs):.4f}",
                'Max': f"{np.max(strategy_accs):.4f}",
                'Datasets': len(strategy_accs)
            })
    
    stats_df = pd.DataFrame(stats_data).sort_values('Mean Acc', ascending=False)
    tables.append("| " + " | ".join(stats_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in stats_df.columns]) + "|")
    for _, row in stats_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    return "\n".join(tables), all_strategies


def plot_all_strategies_comparison(datasets, all_strategies, output_dir):
    """Plot 1: All strategies comparison across datasets."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    dataset_names = sorted(datasets.keys())
    x = np.arange(len(dataset_names))
    width = 0.15
    
    colors = {
        'FedAvg': '#d62728',
        'FedProx': '#ff7f0e',
        'FedMedian': '#1f77b4',
        'Bulyan': '#9467bd',
        'FedAgain': '#2ca02c'
    }
    
    for idx, strategy in enumerate(all_strategies):
        accs = []
        for dataset_name in dataset_names:
            df = datasets[dataset_name]
            if strategy in df['strategy'].values:
                accs.append(df[df['strategy'] == strategy]['final_accuracy'].mean())
            else:
                accs.append(0)
        
        offset = (idx - len(all_strategies)/2) * width
        color = colors.get(strategy, plt.cm.tab10(idx))
        ax.bar(x + offset, accs, width, label=strategy, color=color, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Strategy Comparison Across All Datasets', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / 'all_strategies_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_fedagain_vs_all_heatmap(datasets, all_strategies, output_dir):
    """Plot 2: FedAgain improvement heatmap vs all strategies."""
    comparison_strategies = [s for s in all_strategies if s != 'FedAgain']
    dataset_names = sorted(datasets.keys())
    
    heatmap_data = []
    for dataset_name in dataset_names:
        df = datasets[dataset_name]
        if 'FedAgain' not in df['strategy'].values:
            continue
        
        fedagain_acc = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        row = []
        
        for strategy in comparison_strategies:
            if strategy in df['strategy'].values:
                other_acc = df[df['strategy'] == strategy]['final_accuracy'].mean()
                improvement = ((fedagain_acc - other_acc) / other_acc) * 100
                row.append(improvement)
            else:
                row.append(np.nan)
        
        heatmap_data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=15)
    
    ax.set_xticks(np.arange(len(comparison_strategies)))
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(comparison_strategies, rotation=45, ha='right')
    ax.set_yticklabels(dataset_names)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('FedAgain Improvement (%)', rotation=270, labelpad=20, fontsize=12)
    
    for i in range(len(dataset_names)):
        for j in range(len(comparison_strategies)):
            if not np.isnan(heatmap_array[i, j]):
                text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                             ha="center", va="center", color="black", 
                             fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Competing Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('FedAgain Improvement vs All Strategies\n(% improvement, positive=better)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'fedagain_vs_all_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_strategy_rankings(datasets, all_strategies, output_dir):
    """Plot 3: Average ranking of each strategy across datasets."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategy_ranks = {s: [] for s in all_strategies}
    
    for dataset_name, df in datasets.items():
        strategy_means = df.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
        for rank, strategy in enumerate(strategy_means.index, 1):
            if strategy in strategy_ranks:
                strategy_ranks[strategy].append(rank)
    
    # Calculate average rank
    avg_ranks = {}
    for strategy, ranks in strategy_ranks.items():
        if ranks:
            avg_ranks[strategy] = np.mean(ranks)
    
    # Sort by average rank
    sorted_strategies = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
    avg_rank_values = [avg_ranks[s] for s in sorted_strategies]
    
    colors = ['#2ca02c' if s == 'FedAgain' else '#1f77b4' for s in sorted_strategies]
    bars = ax.barh(sorted_strategies, avg_rank_values, color=colors, alpha=0.85)
    
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Average Strategy Ranking Across All Datasets', fontsize=14, fontweight='bold')
    ax.set_xlim([1, len(all_strategies)])
    ax.invert_xaxis()  # Lower ranks on the right
    ax.grid(True, axis='x', alpha=0.3)
    
    for bar, val in zip(bars, avg_rank_values):
        width = bar.get_width()
        ax.text(width - 0.1, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}', ha='right', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    output_path = output_dir / 'strategy_rankings.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_corruption_robustness_all(datasets, all_strategies, output_dir):
    """Plot 4: All strategies performance across corruption levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    corruption_levels = [0.1, 0.3, 0.4]
    titles = ['10% Corruption', '30% Corruption', '40% Corruption']
    
    for ax, ratio, title in zip(axes, corruption_levels, titles):
        dataset_names = []
        strategy_data = {s: [] for s in all_strategies}
        
        for dataset_name in sorted(datasets.keys()):
            df = datasets[dataset_name]
            subset = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == ratio)]
            
            if len(subset) > 0:
                has_data = False
                for strategy in all_strategies:
                    if strategy in subset['strategy'].values:
                        acc = subset[subset['strategy'] == strategy]['final_accuracy'].mean()
                        strategy_data[strategy].append(acc)
                        has_data = True
                    else:
                        strategy_data[strategy].append(0)
                
                if has_data:
                    dataset_names.append(dataset_name)
        
        x = np.arange(len(dataset_names))
        width = 0.15
        
        colors = {
            'FedAvg': '#d62728',
            'FedProx': '#ff7f0e',
            'FedMedian': '#1f77b4',
            'Bulyan': '#9467bd',
            'FedAgain': '#2ca02c'
        }
        
        for idx, strategy in enumerate(all_strategies):
            if any(strategy_data[strategy]):
                offset = (idx - len(all_strategies)/2) * width
                color = colors.get(strategy, plt.cm.tab10(idx))
                ax.bar(x + offset, strategy_data[strategy][:len(dataset_names)], 
                      width, label=strategy, color=color, alpha=0.85)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        
        if ax == axes[0]:
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
    
    plt.suptitle('Strategy Performance Across Corruption Levels', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'corruption_robustness_all_strategies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_win_rate_comparison(datasets, all_strategies, output_dir):
    """Plot 5: Win rate comparison for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dataset_names = sorted(datasets.keys())
    x = np.arange(len(dataset_names))
    width = 0.15
    
    colors = {
        'FedAvg': '#d62728',
        'FedProx': '#ff7f0e',
        'FedMedian': '#1f77b4',
        'Bulyan': '#9467bd',
        'FedAgain': '#2ca02c'
    }
    
    for idx, strategy in enumerate(all_strategies):
        win_rates = []
        
        for dataset_name in dataset_names:
            df = datasets[dataset_name]
            
            if strategy not in df['strategy'].values:
                win_rates.append(0)
                continue
            
            conditions = ['corrupt_ratio', 'corruption_severity', 'data_distribution']
            available_conditions = [c for c in conditions if c in df.columns]
            
            wins = 0
            total = 0
            
            if len(available_conditions) > 0:
                grouped = df.groupby(available_conditions)
                for name, group in grouped:
                    if len(group['strategy'].unique()) > 1:
                        best = group.loc[group['final_accuracy'].idxmax(), 'strategy']
                        total += 1
                        if best == strategy:
                            wins += 1
            
            win_rate = (wins / total * 100) if total > 0 else 0
            win_rates.append(win_rate)
        
        offset = (idx - len(all_strategies)/2) * width
        color = colors.get(strategy, plt.cm.tab10(idx))
        ax.bar(x + offset, win_rates, width, label=strategy, color=color, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate Comparison: All Strategies Across Datasets', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_all_strategies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_strategy_consistency(datasets, all_strategies, output_dir):
    """Plot 6: Strategy consistency (std dev) across datasets."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strategy_stats = []
    
    for strategy in all_strategies:
        accs = []
        for df in datasets.values():
            if strategy in df['strategy'].values:
                accs.append(df[df['strategy'] == strategy]['final_accuracy'].mean())
        
        if accs:
            strategy_stats.append({
                'Strategy': strategy,
                'Mean': np.mean(accs),
                'StdDev': np.std(accs)
            })
    
    stats_df = pd.DataFrame(strategy_stats).sort_values('StdDev')
    
    colors = ['#2ca02c' if s == 'FedAgain' else '#1f77b4' for s in stats_df['Strategy']]
    bars = ax.barh(stats_df['Strategy'], stats_df['StdDev'], color=colors, alpha=0.85)
    
    ax.set_xlabel('Standard Deviation (lower = more consistent)', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Consistency Across Datasets', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    for bar, val in zip(bars, stats_df['StdDev']):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'strategy_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def generate_report(datasets, tables, all_strategies, output_dir):
    """Generate comprehensive report."""
    report = []
    
    report.append("# Comprehensive Strategy Comparison Report")
    report.append(f"\n**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Datasets Analyzed:** {len(datasets)}")
    report.append(f"**Strategies Compared:** {', '.join(all_strategies)}")
    
    total_exp = sum(len(df) for df in datasets.values())
    report.append(f"**Total Experiments:** {total_exp}\n")
    
    report.append("---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    # Overall best strategy
    strategy_accs = {}
    for strategy in all_strategies:
        accs = []
        for df in datasets.values():
            if strategy in df['strategy'].values:
                accs.append(df[df['strategy'] == strategy]['final_accuracy'].mean())
        if accs:
            strategy_accs[strategy] = np.mean(accs)
    
    best_overall = max(strategy_accs, key=strategy_accs.get)
    report.append(f"**Best Overall Strategy:** {best_overall} ({strategy_accs[best_overall]:.4f} mean accuracy)")
    
    # FedAgain position
    if 'FedAgain' in strategy_accs:
        sorted_strategies = sorted(strategy_accs.items(), key=lambda x: x[1], reverse=True)
        fedagain_rank = [i for i, (s, _) in enumerate(sorted_strategies, 1) if s == 'FedAgain'][0]
        report.append(f"**FedAgain Rank:** {fedagain_rank} of {len(strategy_accs)}")
        report.append(f"**FedAgain Mean Accuracy:** {strategy_accs['FedAgain']:.4f}\n")
    
    report.append("---\n")
    
    # Add all tables
    report.append("# Detailed Comparison Tables\n")
    report.append(tables)
    
    report.append("\n---\n")
    
    # Visualizations
    report.append("## Generated Visualizations\n")
    report.append("1. **all_strategies_comparison.png** - Overall comparison across datasets")
    report.append("2. **fedagain_vs_all_heatmap.png** - FedAgain improvement heatmap")
    report.append("3. **strategy_rankings.png** - Average rankings")
    report.append("4. **corruption_robustness_all_strategies.png** - Performance at corruption levels")
    report.append("5. **win_rate_all_strategies.png** - Win rate comparison")
    report.append("6. **strategy_consistency.png** - Consistency analysis\n")
    
    report.append("---\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    
    # Win rates
    report.append("### Win Rates Across All Datasets\n")
    for strategy in all_strategies:
        total_wins = 0
        total_scenarios = 0
        
        for df in datasets.values():
            if strategy not in df['strategy'].values:
                continue
            
            conditions = ['corrupt_ratio', 'corruption_severity', 'data_distribution']
            available_conditions = [c for c in conditions if c in df.columns]
            
            if len(available_conditions) > 0:
                grouped = df.groupby(available_conditions)
                for name, group in grouped:
                    if len(group['strategy'].unique()) > 1:
                        best = group.loc[group['final_accuracy'].idxmax(), 'strategy']
                        total_scenarios += 1
                        if best == strategy:
                            total_wins += 1
        
        win_rate = (total_wins / total_scenarios * 100) if total_scenarios > 0 else 0
        report.append(f"- **{strategy}:** {total_wins}/{total_scenarios} scenarios ({win_rate:.1f}%)")
    
    report.append("\n---\n")
    report.append(f"\n*Report generated: {pd.Timestamp.now()}*")
    
    return "\n".join(report)


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STRATEGY COMPARISON ANALYSIS")
    print("="*80)
    
    output_dir = Path("results/ALL_STRATEGIES_COMPARISON")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    print("Loading datasets...")
    datasets = load_all_datasets()
    
    if len(datasets) == 0:
        print("\n[ERROR] No datasets found!")
        return
    
    print(f"\n[OK] Loaded {len(datasets)} dataset(s)\n")
    
    print("Generating comprehensive comparison tables...")
    tables, all_strategies = generate_comprehensive_tables(datasets)
    print("  [OK] Tables generated\n")
    
    print("Generating visualizations...")
    plot_all_strategies_comparison(datasets, all_strategies, output_dir)
    plot_fedagain_vs_all_heatmap(datasets, all_strategies, output_dir)
    plot_strategy_rankings(datasets, all_strategies, output_dir)
    plot_corruption_robustness_all(datasets, all_strategies, output_dir)
    plot_win_rate_comparison(datasets, all_strategies, output_dir)
    plot_strategy_consistency(datasets, all_strategies, output_dir)
    print()
    
    print("Generating comprehensive report...")
    report = generate_report(datasets, tables, all_strategies, output_dir)
    report_path = output_dir / "ALL_STRATEGIES_COMPARISON_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] {report_path}\n")
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - ALL_STRATEGIES_COMPARISON_REPORT.md")
    print("  - all_strategies_comparison.png")
    print("  - fedagain_vs_all_heatmap.png")
    print("  - strategy_rankings.png")
    print("  - corruption_robustness_all_strategies.png")
    print("  - win_rate_all_strategies.png")
    print("  - strategy_consistency.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

