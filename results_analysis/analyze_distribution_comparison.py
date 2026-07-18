import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Read the experiments data
df = pd.read_csv('../results/experiments.csv')

# Ensure numeric columns
for col in ['final_accuracy', 'final_precision', 'final_recall', 'final_f1',
            'corrupt_ratio', 'corruption_severity', 'num_clients']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter for FedAvg vs FedMedian vs FedAgain comparison
# Consistent corruption settings: ratio=0.3, severity=3 across all distributions
df_comparison = df[
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) &
    (df['corruption_severity'] == 3) &
    (df['data_distribution'].isin(['iid', 'non-iid-label', 'non-iid-dirichlet'])) &
    (df['strategy'].isin(['FedAvg', 'FedAgain', 'FedMedian']))
].copy()

# For non-iid-label there are multiple num_clients; keep only the latest run per (strategy, dataset, distribution)
df_comparison = df_comparison.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'data_distribution'], keep='last'
)

print(f"Total comparison samples: {len(df_comparison)}")

# Define distributions
distributions = ['iid', 'non-iid-label', 'non-iid-dirichlet']
distribution_labels = {
    'iid': 'IID',
    'non-iid-label': 'Non-IID (Label)',
    'non-iid-dirichlet': 'Non-IID (Dirichlet)'
}

print(f"\nSamples by distribution:")
for dist in distributions:
    count = len(df_comparison[df_comparison['data_distribution'] == dist])
    print(f"  {distribution_labels[dist]}: {count} samples")

print(f"\nSamples by strategy:")
print(df_comparison['strategy'].value_counts())

# Define metrics to analyze
metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# Define colors for strategies
bar_colors = {
    'FedAvg': '#e74c3c',      # Red
    'FedMedian': '#f39c12',   # Orange
    'FedAgain': '#2ecc71'     # Green
}
compare_strategies = ['FedAvg', 'FedMedian', 'FedAgain']

# Filter for all kidney stone datasets including MyStone
all_kidney_datasets = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
                       'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR',
                       'MYSTONE']

df_kidney = df_comparison[df_comparison['dataset'].isin(all_kidney_datasets)]

print(f"\nKidney stone dataset samples: {len(df_kidney)}")

# =============================================================================
# Plot 1: Box plot comparing FedAvg vs FedMedian vs FedAgain across distributions
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    # Prepare data for box plot
    data_to_plot = []
    positions = []
    colors = []

    pos = 0
    for dist in distributions:
        for strategy in compare_strategies:
            sdata = df_kidney[
                (df_kidney['strategy'] == strategy) &
                (df_kidney['data_distribution'] == dist)
            ]
            if len(sdata) > 0:
                data_to_plot.append(sdata[metric].values)
                positions.append(pos)
                colors.append(bar_colors[strategy])
            pos += 1

        # Add spacing between distribution groups
        pos += 0.5

    if len(data_to_plot) > 0:
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions[:len(data_to_plot)], widths=0.4,
                        patch_artist=True,
                        showmeans=False,
                        medianprops=dict(color='black', linewidth=2.5),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray',
                                      markersize=5, alpha=0.5))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    # Set x-axis labels at center of each distribution group
    label_positions = []
    for i in range(len(distributions)):
        center_pos = i * 3.5 + 1
        label_positions.append(center_pos)

    ax.set_xticks(label_positions)
    ax.set_xticklabels([distribution_labels[d] for d in distributions],
                       fontsize=12, rotation=15, ha='right')

    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim([0.4, 1.0])

# Add shared legend using manual patches (always visible)
legend_patches = [Patch(facecolor=bar_colors[s], edgecolor='black', label=s, alpha=0.7)
                  for s in compare_strategies]
fig.legend(handles=legend_patches,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.02),
          ncol=3,
          fontsize=14,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.savefig('fedagain_vs_fedavg_vs_fedmedian_distributions.png', dpi=300, bbox_inches='tight')
print("\nSaved: fedagain_vs_fedavg_vs_fedmedian_distributions.png")
plt.close()

# =============================================================================
# Plot 2: Improvement percentage by distribution
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

imp_bar_colors = {'vs FedAvg': '#e74c3c', 'vs FedMedian': '#f39c12'}

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    width = 0.35
    x_pos = np.arange(len(distributions))

    for ref_strategy, offset, color, label_name in [
        ('FedAvg', -width/2, '#e74c3c', 'vs FedAvg'),
        ('FedMedian', width/2, '#f39c12', 'vs FedMedian')
    ]:
        improvements = []
        has_data = []
        for dist in distributions:
            ref_data = df_kidney[
                (df_kidney['strategy'] == ref_strategy) &
                (df_kidney['data_distribution'] == dist)
            ]
            fedagain_data = df_kidney[
                (df_kidney['strategy'] == 'FedAgain') &
                (df_kidney['data_distribution'] == dist)
            ]
            if len(ref_data) > 0 and len(fedagain_data) > 0:
                ref_mean = ref_data[metric].mean()
                fedagain_mean = fedagain_data[metric].mean()
                if ref_mean > 0:
                    improvement_pct = ((fedagain_mean - ref_mean) / ref_mean) * 100
                    improvements.append(improvement_pct)
                    has_data.append(True)
                else:
                    improvements.append(0)
                    has_data.append(False)
            else:
                improvements.append(0)
                has_data.append(False)

        bars = ax.bar(x_pos + offset, improvements, width,
                      color=color, edgecolor='black', linewidth=1.5,
                      alpha=0.8)

        # Hide bars where no data, label where it does
        for bar, imp, valid in zip(bars, improvements, has_data):
            if not valid:
                bar.set_alpha(0)
                bar.set_edgecolor('none')
                bar.set_height(0)
            else:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{imp:+.1f}%',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9, fontweight='bold')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Data Distribution', fontsize=13)
    ax.set_ylabel(f'{label} (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([distribution_labels[d] for d in distributions],
                       rotation=15, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='y', labelsize=14)

# Add shared legend using manual patches (always visible)
imp_patches = [Patch(facecolor=imp_bar_colors[s], edgecolor='black', label=s)
               for s in ['vs FedAvg', 'vs FedMedian']]
fig.legend(handles=imp_patches,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.02),
          ncol=2,
          fontsize=14,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.savefig('fedagain_improvement_by_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_improvement_by_distribution.png")
plt.close()

# =============================================================================
# Summary statistics
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: FedAgain vs FedAvg vs FedMedian Across Data Distributions")
print("="*80)

for dist in distributions:
    print(f"\n{'='*80}")
    print(f"Distribution: {distribution_labels[dist]}")
    print(f"{'='*80}")

    dist_data = df_kidney[df_kidney['data_distribution'] == dist]

    if len(dist_data) == 0:
        print("No data available")
        continue

    print(f"\nMetrics by Strategy (averaged across kidney stone datasets):")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)

    for strategy in compare_strategies:
        strategy_data = dist_data[dist_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc_mean = strategy_data['final_accuracy'].mean()
            acc_std = strategy_data['final_accuracy'].std()
            prec_mean = strategy_data['final_precision'].mean()
            prec_std = strategy_data['final_precision'].std()
            rec_mean = strategy_data['final_recall'].mean()
            rec_std = strategy_data['final_recall'].std()
            f1_mean = strategy_data['final_f1'].mean()
            f1_std = strategy_data['final_f1'].std()

            print(f"{strategy:<12} {acc_mean:.4f}±{acc_std:.3f}  {prec_mean:.4f}±{prec_std:.3f}  {rec_mean:.4f}±{rec_std:.3f}  {f1_mean:.4f}±{f1_std:.3f}")
        else:
            print(f"{strategy:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    # Calculate improvement vs both baselines
    fedagain_data = dist_data[dist_data['strategy'] == 'FedAgain']

    for ref_name in ['FedAvg', 'FedMedian']:
        ref_data = dist_data[dist_data['strategy'] == ref_name]
        if len(ref_data) > 0 and len(fedagain_data) > 0:
            acc_imp = ((fedagain_data['final_accuracy'].mean() - ref_data['final_accuracy'].mean()) /
                       ref_data['final_accuracy'].mean()) * 100
            f1_imp = ((fedagain_data['final_f1'].mean() - ref_data['final_f1'].mean()) /
                      ref_data['final_f1'].mean()) * 100
            print(f"\nFedAgain vs {ref_name}: Accuracy {acc_imp:+.2f}%, F1 Score {f1_imp:+.2f}%")

# Performance degradation analysis
print("\n" + "="*80)
print("Performance Impact of Data Heterogeneity:")
print("="*80)

for strategy in compare_strategies:
    print(f"\n{strategy}:")
    iid_data = df_kidney[(df_kidney['strategy'] == strategy) & (df_kidney['data_distribution'] == 'iid')]
    noniid_label_data = df_kidney[(df_kidney['strategy'] == strategy) & (df_kidney['data_distribution'] == 'non-iid-label')]
    noniid_dir_data = df_kidney[(df_kidney['strategy'] == strategy) & (df_kidney['data_distribution'] == 'non-iid-dirichlet')]

    if len(iid_data) > 0:
        iid_acc = iid_data['final_accuracy'].mean()
        print(f"  IID accuracy: {iid_acc:.4f}")

        if len(noniid_label_data) > 0:
            noniid_label_acc = noniid_label_data['final_accuracy'].mean()
            degradation = ((noniid_label_acc - iid_acc) / iid_acc) * 100
            print(f"  Non-IID (Label) accuracy: {noniid_label_acc:.4f} ({degradation:+.2f}% vs IID)")

        if len(noniid_dir_data) > 0:
            noniid_dir_acc = noniid_dir_data['final_accuracy'].mean()
            degradation = ((noniid_dir_acc - iid_acc) / iid_acc) * 100
            print(f"  Non-IID (Dirichlet) accuracy: {noniid_dir_acc:.4f} ({degradation:+.2f}% vs IID)")
    elif len(noniid_label_data) > 0:
        noniid_label_acc = noniid_label_data['final_accuracy'].mean()
        print(f"  Non-IID (Label) accuracy: {noniid_label_acc:.4f} (no IID data)")

print("\n" + "="*80)
print("Analysis complete! Generated 2 plots:")
print("  1. fedagain_vs_fedavg_distributions.png - Performance comparison")
print("  2. fedagain_improvement_by_distribution.png - Improvement percentages")
print("="*80)
