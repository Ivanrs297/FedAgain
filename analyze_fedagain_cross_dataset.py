"""
Cross-Dataset FedAgain Analysis
Comprehensive comparison of FedAgain performance across multiple datasets.
Generates comparison tables and visualizations highlighting FedAgain's performance.
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


def generate_comparison_tables(datasets):
    """Generate comprehensive comparison tables."""
    tables = []
    
    # Table 1: Overall Performance Summary
    tables.append("\n## Table 1: Overall Performance by Dataset\n")
    overall_data = []
    
    for dataset_name, df in sorted(datasets.items()):
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            fedagain_mean = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fedagain_std = df[df['strategy'] == 'FedAgain']['final_accuracy'].std()
            fedavg_mean = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            fedavg_std = df[df['strategy'] == 'FedAvg']['final_accuracy'].std()
            improvement = ((fedagain_mean - fedavg_mean) / fedavg_mean) * 100
            
            # Find best strategy
            best_strategy = df.groupby('strategy')['final_accuracy'].mean().idxmax()
            best_acc = df.groupby('strategy')['final_accuracy'].mean().max()
            
            overall_data.append({
                'Dataset': dataset_name,
                'FedAgain': f"{fedagain_mean:.4f}±{fedagain_std:.4f}",
                'FedAvg': f"{fedavg_mean:.4f}±{fedavg_std:.4f}",
                'Improvement': f"{improvement:+.2f}%",
                'Best Strategy': best_strategy,
                'Best Acc': f"{best_acc:.4f}",
                'Winner': '*' if best_strategy == 'FedAgain' else ''
            })
    
    overall_df = pd.DataFrame(overall_data)
    tables.append("| " + " | ".join(overall_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in overall_df.columns]) + "|")
    for _, row in overall_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 2: Performance by Corruption Level
    tables.append("\n\n## Table 2: FedAgain Performance by Corruption Level\n")
    corruption_data = []
    
    for dataset_name, df in sorted(datasets.items()):
        for ratio in [0.1, 0.3, 0.4]:
            subset = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == ratio)]
            if len(subset) > 0 and 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                fedagain_acc = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fedavg_acc = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
                
                corruption_data.append({
                    'Dataset': dataset_name,
                    'Corruption': f"{int(ratio*100)}%",
                    'FedAgain': f"{fedagain_acc:.4f}",
                    'FedAvg': f"{fedavg_acc:.4f}",
                    'Improvement': f"{improvement:+.2f}%",
                    'Status': 'OK' if improvement > 0 else 'NO'
                })
    
    if corruption_data:
        corruption_df = pd.DataFrame(corruption_data)
        tables.append("| " + " | ".join(corruption_df.columns) + " |")
        tables.append("|" + "|".join(['---' for _ in corruption_df.columns]) + "|")
        for _, row in corruption_df.iterrows():
            tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 3: Win Rate Analysis
    tables.append("\n\n## Table 3: FedAgain Win Rate by Dataset\n")
    win_data = []
    
    for dataset_name, df in sorted(datasets.items()):
        # Count scenarios where FedAgain is best
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
                    if best == 'FedAgain':
                        wins += 1
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        win_data.append({
            'Dataset': dataset_name,
            'Wins': wins,
            'Total Scenarios': total,
            'Win Rate': f"{win_rate:.1f}%",
            'Grade': 'A' if win_rate >= 70 else 'B' if win_rate >= 50 else 'C' if win_rate >= 30 else 'D'
        })
    
    win_df = pd.DataFrame(win_data)
    tables.append("| " + " | ".join(win_df.columns) + " |")
    tables.append("|" + "|".join(['---' for _ in win_df.columns]) + "|")
    for _, row in win_df.iterrows():
        tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    # Table 4: Statistical Summary
    tables.append("\n\n## Table 4: Statistical Summary Across All Datasets\n")
    
    all_improvements = []
    for dataset_name, df in datasets.items():
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            fa = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fv = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            imp = ((fa - fv) / fv) * 100
            all_improvements.append(imp)
    
    if all_improvements:
        stats_data = [{
            'Metric': 'Average Improvement',
            'Value': f"{np.mean(all_improvements):+.2f}%"
        }, {
            'Metric': 'Best Improvement',
            'Value': f"{np.max(all_improvements):+.2f}%"
        }, {
            'Metric': 'Worst Improvement',
            'Value': f"{np.min(all_improvements):+.2f}%"
        }, {
            'Metric': 'Std Deviation',
            'Value': f"{np.std(all_improvements):.2f}%"
        }, {
            'Metric': 'Datasets Improved',
            'Value': f"{sum(1 for x in all_improvements if x > 0)}/{len(all_improvements)}"
        }]
        
        stats_df = pd.DataFrame(stats_data)
        tables.append("| " + " | ".join(stats_df.columns) + " |")
        tables.append("|" + "|".join(['---' for _ in stats_df.columns]) + "|")
        for _, row in stats_df.iterrows():
            tables.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    return "\n".join(tables)


def plot_cross_dataset_comparison(datasets, output_dir):
    """Plot 1: Overall accuracy comparison across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dataset_names = []
    fedagain_accs = []
    fedavg_accs = []
    
    for dataset_name, df in sorted(datasets.items()):
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            dataset_names.append(dataset_name)
            fedagain_accs.append(df[df['strategy'] == 'FedAgain']['final_accuracy'].mean())
            fedavg_accs.append(df[df['strategy'] == 'FedAvg']['final_accuracy'].mean())
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fedavg_accs, width, label='FedAvg', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, fedagain_accs, width, label='FedAgain', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('FedAgain vs FedAvg: Cross-Dataset Performance Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'cross_dataset_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_improvement_heatmap(datasets, output_dir):
    """Plot 2: Heatmap of FedAgain improvement across datasets and corruption levels."""
    # Prepare data
    corruption_levels = [0.1, 0.3, 0.4]
    dataset_names = sorted(datasets.keys())
    
    heatmap_data = []
    for dataset_name in dataset_names:
        df = datasets[dataset_name]
        row = []
        for ratio in corruption_levels:
            subset = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == ratio)]
            if len(subset) > 0 and 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                fa = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fv = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                improvement = ((fa - fv) / fv) * 100
                row.append(improvement)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=20)
    
    # Set ticks
    ax.set_xticks(np.arange(len(corruption_levels)))
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_xticklabels([f'{int(r*100)}%' for r in corruption_levels])
    ax.set_yticklabels(dataset_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement over FedAvg (%)', rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotations
    for i in range(len(dataset_names)):
        for j in range(len(corruption_levels)):
            if not np.isnan(heatmap_array[i, j]):
                text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Corruption Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('FedAgain Improvement Heatmap\n(% improvement over FedAvg)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_win_rate_analysis(datasets, output_dir):
    """Plot 3: Win rate by dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dataset_names = []
    win_rates = []
    colors = []
    
    for dataset_name, df in sorted(datasets.items()):
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
                    if best == 'FedAgain':
                        wins += 1
        
        if total > 0:
            win_rate = (wins / total) * 100
            dataset_names.append(dataset_name)
            win_rates.append(win_rate)
            
            # Color based on performance
            if win_rate >= 50:
                colors.append('#2ca02c')  # Green
            elif win_rate >= 30:
                colors.append('#ff7f0e')  # Orange
            else:
                colors.append('#d62728')  # Red
    
    bars = ax.bar(dataset_names, win_rates, color=colors, alpha=0.8)
    
    # Add threshold lines
    ax.axhline(y=50, color='green', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.axhline(y=30, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='30% threshold')
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('FedAgain Win Rate by Dataset\n(% of scenarios where FedAgain performs best)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_severity_comparison(datasets, output_dir):
    """Plot 4: FedAgain performance across severity levels for all datasets."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for idx, (dataset_name, df) in enumerate(sorted(datasets.items())):
        severity_data = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
        
        if len(severity_data) > 0 and 'corruption_severity' in df.columns:
            improvements = []
            severities = []
            
            for severity in sorted(severity_data['corruption_severity'].unique()):
                subset = severity_data[severity_data['corruption_severity'] == severity]
                if 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                    fa = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fv = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    improvement = ((fa - fv) / fv) * 100
                    improvements.append(improvement)
                    severities.append(severity)
            
            if len(improvements) > 0:
                ax.plot(severities, improvements, marker='o', linewidth=2.5, 
                       label=dataset_name, color=colors[idx], markersize=8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Corruption Severity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over FedAvg (%)', fontsize=12, fontweight='bold')
    ax.set_title('FedAgain Performance vs Corruption Severity\n(30% corrupted clients)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'severity_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def plot_overall_improvement_distribution(datasets, output_dir):
    """Plot 5: Distribution of improvements across all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all improvements
    all_improvements = []
    dataset_labels = []
    
    for dataset_name, df in sorted(datasets.items()):
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            fa = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fv = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            imp = ((fa - fv) / fv) * 100
            all_improvements.append(imp)
            dataset_labels.append(dataset_name)
    
    # Plot 1: Box plot
    positions = range(len(dataset_labels))
    bp = ax1.boxplot([all_improvements], positions=[0], widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='#2ca02c', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    
    # Add individual points
    ax1.scatter([0] * len(all_improvements), all_improvements, alpha=0.6, s=100, c='blue')
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Improvement over FedAvg (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Improvements', fontsize=12, fontweight='bold')
    ax1.set_xticks([0])
    ax1.set_xticklabels(['All Datasets'])
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Individual dataset improvements
    colors_improve = ['#2ca02c' if x > 0 else '#d62728' for x in all_improvements]
    bars = ax2.barh(dataset_labels, all_improvements, color=colors_improve, alpha=0.8)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Improvement over FedAvg (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Improvement by Dataset', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, all_improvements):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:+.2f}%', ha='left' if val > 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] {output_path}")
    plt.close()


def generate_report(datasets, tables, output_dir):
    """Generate comprehensive Markdown report."""
    report = []
    
    report.append("# FedAgain Cross-Dataset Analysis Report")
    report.append(f"\n**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Datasets Analyzed:** {len(datasets)}")
    
    total_exp = sum(len(df) for df in datasets.values())
    report.append(f"**Total Experiments:** {total_exp}\n")
    
    report.append("---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    all_improvements = []
    for dataset_name, df in datasets.items():
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            fa = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fv = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            imp = ((fa - fv) / fv) * 100
            all_improvements.append((dataset_name, imp))
    
    avg_improvement = np.mean([x[1] for x in all_improvements])
    best_dataset = max(all_improvements, key=lambda x: x[1])
    worst_dataset = min(all_improvements, key=lambda x: x[1])
    
    report.append(f"**Average Improvement:** {avg_improvement:+.2f}%")
    report.append(f"**Best Performance:** {best_dataset[0]} ({best_dataset[1]:+.2f}%)")
    report.append(f"**Worst Performance:** {worst_dataset[0]} ({worst_dataset[1]:+.2f}%)")
    report.append(f"**Datasets Improved:** {sum(1 for _, imp in all_improvements if imp > 0)}/{len(all_improvements)}\n")
    
    report.append("---\n")
    
    # Dataset Overview
    report.append("## Datasets Overview\n")
    for dataset_name, df in sorted(datasets.items()):
        report.append(f"### {dataset_name}")
        report.append(f"- Experiments: {len(df)}")
        report.append(f"- Strategies: {', '.join(df['strategy'].unique())}")
        if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
            fa = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
            fv = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
            imp = ((fa - fv) / fv) * 100
            report.append(f"- FedAgain Improvement: {imp:+.2f}%\n")
    
    report.append("---\n")
    
    # Add tables
    report.append("# Comparison Tables\n")
    report.append(tables)
    
    report.append("\n---\n")
    
    # Visualizations
    report.append("## Generated Visualizations\n")
    report.append("1. **cross_dataset_comparison.png** - Bar chart comparing FedAgain vs FedAvg")
    report.append("2. **improvement_heatmap.png** - Heatmap of improvements across corruption levels")
    report.append("3. **win_rate_analysis.png** - Win rate by dataset")
    report.append("4. **severity_comparison.png** - Performance across severity levels")
    report.append("5. **improvement_distribution.png** - Distribution of improvements\n")
    
    report.append("---\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    
    # Find best corruption level
    best_corruption_improvements = {}
    for ratio in [0.1, 0.3, 0.4]:
        improvements = []
        for dataset_name, df in datasets.items():
            subset = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == ratio)]
            if len(subset) > 0 and 'FedAgain' in subset['strategy'].values and 'FedAvg' in subset['strategy'].values:
                fa = subset[subset['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fv = subset[subset['strategy'] == 'FedAvg']['final_accuracy'].mean()
                improvements.append(((fa - fv) / fv) * 100)
        if improvements:
            best_corruption_improvements[ratio] = np.mean(improvements)
    
    if best_corruption_improvements:
        best_ratio = max(best_corruption_improvements, key=best_corruption_improvements.get)
        report.append(f"1. **Best Corruption Level:** {int(best_ratio*100)}% corruption (avg improvement: {best_corruption_improvements[best_ratio]:+.2f}%)")
    
    report.append(f"2. **Most Consistent Dataset:** {max(all_improvements, key=lambda x: x[1])[0]}")
    report.append(f"3. **Most Challenging Dataset:** {min(all_improvements, key=lambda x: x[1])[0]}")
    
    positive_count = sum(1 for _, imp in all_improvements if imp > 0)
    report.append(f"4. **Success Rate:** {positive_count}/{len(all_improvements)} datasets show improvement\n")
    
    report.append("---\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    
    if avg_improvement > 5:
        report.append("[STRONG] **FedAgain shows strong performance** across datasets with an average {:.2f}% improvement.".format(avg_improvement))
    elif avg_improvement > 0:
        report.append("[MODEST] **FedAgain shows modest improvement** with an average {:.2f}% gain.".format(avg_improvement))
    else:
        report.append("[WEAK] **FedAgain underperforms** with an average {:.2f}% decline.".format(avg_improvement))
    
    report.append("\n**Dataset-Specific Recommendations:**")
    for dataset_name, imp in sorted(all_improvements, key=lambda x: x[1], reverse=True):
        if imp > 5:
            report.append(f"- **{dataset_name}:** Strongly recommended (improvement: {imp:+.2f}%)")
        elif imp > 0:
            report.append(f"- **{dataset_name}:** Consider using (improvement: {imp:+.2f}%)")
        else:
            report.append(f"- **{dataset_name}:** Use alternative strategies (decline: {imp:.2f}%)")
    
    report.append("\n---\n")
    report.append(f"\n*Report generated: {pd.Timestamp.now()}*")
    
    return "\n".join(report)


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("FEDAGAIN CROSS-DATASET ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path("results/CROSS_DATASET_ANALYSIS")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_all_datasets()
    
    if len(datasets) == 0:
        print("\n[ERROR] No datasets found!")
        print("Run 'python analyze_by_dataset.py' first to organize results.")
        return
    
    print(f"\n[OK] Loaded {len(datasets)} dataset(s)\n")
    
    # Generate tables
    print("Generating comparison tables...")
    tables = generate_comparison_tables(datasets)
    print("  [OK] Tables generated\n")
    
    # Generate plots
    print("Generating visualizations...")
    plot_cross_dataset_comparison(datasets, output_dir)
    plot_improvement_heatmap(datasets, output_dir)
    plot_win_rate_analysis(datasets, output_dir)
    plot_severity_comparison(datasets, output_dir)
    plot_overall_improvement_distribution(datasets, output_dir)
    print()
    
    # Generate report
    print("Generating comprehensive report...")
    report = generate_report(datasets, tables, output_dir)
    report_path = output_dir / "FEDAGAIN_CROSS_DATASET_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] {report_path}\n")
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - FEDAGAIN_CROSS_DATASET_REPORT.md")
    print("  - cross_dataset_comparison.png")
    print("  - improvement_heatmap.png")
    print("  - win_rate_analysis.png")
    print("  - severity_comparison.png")
    print("  - improvement_distribution.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

