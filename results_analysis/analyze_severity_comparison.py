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

# Filter for FedAgain vs FedAvg vs FedMedian comparison
# Conditions: non-iid-label, matching (corrupt_ratio, severity) combinations
# where all 3 strategies have data:
#   Light:    ratio=0.1, severity=3
#   Moderate: ratio=0.3, severity=4
#   Heavy:    ratio=0.4, severity=5
corruption_scenarios = [
    (0.1, 3, 'Light (10%, Sev.3)'),
    (0.3, 4, 'Moderate (30%, Sev.4)'),
    (0.4, 5, 'Heavy (40%, Sev.5)')
]

df_comparison = pd.concat([
    df[(df['corrupt_clients'] == True) &
       (df['data_distribution'] == 'non-iid-label') &
       (df['strategy'].isin(['FedAvg', 'FedAgain', 'FedMedian'])) &
       (df['corrupt_ratio'] == ratio) &
       (df['corruption_severity'] == sev)].assign(scenario=label)
    for ratio, sev, label in corruption_scenarios
])

print(f"Total comparison samples: {len(df_comparison)}")
print(f"\nSamples by scenario:")
for _, sev, label in corruption_scenarios:
    count = len(df_comparison[df_comparison['corruption_severity'] == sev])
    print(f"  {label}: {count} samples")

print(f"\nSamples by strategy:")
print(df_comparison['strategy'].value_counts())

# Define metrics to analyze
metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# Define scenario labels for x-axis
scenario_labels = [label for _, _, label in corruption_scenarios]
severity_levels = [sev for _, sev, _ in corruption_scenarios]

# Define colors for strategies
strategy_colors = {
    'FedAvg': '#1f77b4',      # Blue
    'FedMedian': '#2ca02c',   # Green
    'FedAgain': '#9467bd'     # Purple
}

# Filter for all kidney stone datasets including MyStone
all_kidney_datasets = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
                       'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 
                       'MYSTONE']

df_kidney = df_comparison[df_comparison['dataset'].isin(all_kidney_datasets)]

print(f"\nKidney stone dataset samples: {len(df_kidney)}")

# =============================================================================
# Plot 1: Bar plot showing FedAgain vs FedAvg across severity levels
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

# Define colors for clearer comparison
bar_colors = {
    'FedAvg': '#e74c3c',      # Red
    'FedMedian': '#f39c12',   # Orange
    'FedAgain': '#2ecc71'     # Green
}

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    x_pos = np.arange(len(scenario_labels))
    width = 0.25

    for strategy, offset in [('FedAvg', -width), ('FedMedian', 0), ('FedAgain', width)]:
        means = []
        stds = []
        for severity in severity_levels:
            sdata = df_kidney[
                (df_kidney['strategy'] == strategy) &
                (df_kidney['corruption_severity'] == severity)
            ]
            if len(sdata) > 0:
                means.append(sdata[metric].mean())
                stds.append(sdata[metric].std())
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x_pos + offset, means, width,
               yerr=stds,
               color=bar_colors[strategy],
               edgecolor='black', linewidth=1.5, alpha=0.85, capsize=4)

    ax.set_xlabel('Corruption Scenario', fontsize=13)
    ax.set_title(f'{label}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim([0.5, 1.0])

# Add shared legend using manual patches (always visible)
legend_patches = [Patch(facecolor=bar_colors[s], edgecolor='black', label=s)
                  for s in ['FedAvg', 'FedMedian', 'FedAgain']]
fig.legend(handles=legend_patches,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.02),
          ncol=3,
          fontsize=14,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.savefig('fedagain_vs_fedavg_severity.png', dpi=300, bbox_inches='tight')
print("\nSaved: fedagain_vs_fedavg_severity.png")
plt.close()

# =============================================================================
# Plot 2: Bar plot showing improvement percentage
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

imp_bar_colors = {'vs FedAvg': '#e74c3c', 'vs FedMedian': '#f39c12'}

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    width = 0.35
    x_pos = np.arange(len(scenario_labels))

    # Compute improvement of FedAgain over FedAvg and FedMedian
    for ref_strategy, offset, color, label_name in [
        ('FedAvg', -width/2, '#e74c3c', 'vs FedAvg'),
        ('FedMedian', width/2, '#f39c12', 'vs FedMedian')
    ]:
        improvements = []
        for severity in severity_levels:
            ref_data = df_kidney[
                (df_kidney['strategy'] == ref_strategy) &
                (df_kidney['corruption_severity'] == severity)
            ]
            fedagain_data = df_kidney[
                (df_kidney['strategy'] == 'FedAgain') &
                (df_kidney['corruption_severity'] == severity)
            ]
            if len(ref_data) > 0 and len(fedagain_data) > 0:
                ref_mean = ref_data[metric].mean()
                fedagain_mean = fedagain_data[metric].mean()
                improvement_pct = ((fedagain_mean - ref_mean) / ref_mean) * 100
                improvements.append(improvement_pct)
            else:
                improvements.append(0)

        bars = ax.bar(x_pos + offset, improvements, width,
                      color=color, edgecolor='black', linewidth=1.5,
                      alpha=0.8)

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Corruption Scenario', fontsize=13)
    ax.set_title(f'{label}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right', fontsize=11)
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
plt.savefig('fedagain_improvement_by_severity.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_improvement_by_severity.png")
plt.close()

# =============================================================================
# Summary statistics
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: FedAgain vs FedAvg Across Corruption Severity Levels")
print("Configuration: 30% Corrupted Clients, Non-IID Label Distribution")
print("="*80)

for severity in severity_levels:
    print(f"\n--- Severity Level {severity} ---")
    
    severity_data = df_kidney[df_kidney['corruption_severity'] == severity]
    
    if len(severity_data) == 0:
        print("No data available")
        continue
    
    print(f"\nMetrics by Strategy (averaged across kidney stone datasets):")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for strategy in ['FedAvg', 'FedMedian', 'FedAgain']:
        strategy_data = severity_data[severity_data['strategy'] == strategy]
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
    
    # Calculate improvement
    fedavg_data = severity_data[severity_data['strategy'] == 'FedAvg']
    fedagain_data = severity_data[severity_data['strategy'] == 'FedAgain']
    
    if len(fedavg_data) > 0 and len(fedagain_data) > 0:
        acc_improvement = ((fedagain_data['final_accuracy'].mean() - fedavg_data['final_accuracy'].mean()) / 
                          fedavg_data['final_accuracy'].mean()) * 100
        f1_improvement = ((fedagain_data['final_f1'].mean() - fedavg_data['final_f1'].mean()) / 
                         fedavg_data['final_f1'].mean()) * 100
        print(f"\nFedAgain Improvement: Accuracy {acc_improvement:+.2f}%, F1 Score {f1_improvement:+.2f}%")

# Calculate overall consistency (variance across severities)
print("\n" + "="*80)
print("Performance Consistency Across Severity Levels:")
print("="*80)

for strategy in ['FedAvg', 'FedMedian', 'FedAgain']:
    strategy_data = df_kidney[df_kidney['strategy'] == strategy]
    
    if len(strategy_data) > 0:
        # Group by severity and calculate means
        severity_means = []
        for severity in severity_levels:
            sev_data = strategy_data[strategy_data['corruption_severity'] == severity]
            if len(sev_data) > 0:
                severity_means.append(sev_data['final_accuracy'].mean())
        
        if len(severity_means) > 1:
            consistency_std = np.std(severity_means)
            consistency_range = max(severity_means) - min(severity_means)
            print(f"\n{strategy}:")
            print(f"  Std Dev across severities: {consistency_std:.4f}")
            print(f"  Range across severities: {consistency_range:.4f}")
            print(f"  Mean accuracy: {np.mean(severity_means):.4f}")

print("\n" + "="*80)
print("Analysis complete! Generated 2 plots:")
print("  1. fedagain_vs_fedavg_severity.png - Line comparison across severity levels")
print("  2. fedagain_improvement_by_severity.png - Improvement percentage bars")
print("="*80)

