import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Read the experiments data
df = pd.read_csv('../results/experiments.csv')

# Filter for corruption scenarios
# Light: 10% clients, severity 3
# Moderate: 30% clients, severity 4
# Heavy: 40% clients, severity 5

df_light = df[(df['corrupt_clients'] == True) & 
              (df['corrupt_ratio'] == 0.1) & 
              (df['corruption_severity'] == 3)].copy()

df_moderate = df[(df['corrupt_clients'] == True) & 
                 (df['corrupt_ratio'] == 0.3) & 
                 (df['corruption_severity'] == 4)].copy()

df_heavy = df[(df['corrupt_clients'] == True) & 
              (df['corrupt_ratio'] == 0.4) & 
              (df['corruption_severity'] == 5)].copy()

print(f"Light corruption samples: {len(df_light)}")
print(f"Moderate corruption samples: {len(df_moderate)}")
print(f"Heavy corruption samples: {len(df_heavy)}")

# Define metrics to analyze
metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# Define strategies in order
strategies = ['FedAvg', 'FedProx', 'FedMedian', 'Bulyan', 'FedAgain']

# Define colors for strategies - traditional scientific colors
strategy_colors = {
    'FedAvg': '#1f77b4',      # Blue
    'FedProx': '#ff7f0e',     # Orange
    'FedMedian': '#2ca02c',   # Green
    'Bulyan': '#d62728',      # Red
    'FedAgain': '#9467bd'     # Purple
}

# Filter for all kidney stone datasets including MyStone
all_kidney_datasets = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
                       'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 
                       'MYSTONE']

# =============================================================================
# Plot: Corruption Impact on All Kidney Stone Datasets
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

metric = 'final_accuracy'

# Prepare data for plotting
x_pos = np.arange(len(strategies))
width = 0.25

# For each strategy, collect mean and std across all datasets for each corruption level
light_means = []
light_stds = []
moderate_means = []
moderate_stds = []
heavy_means = []
heavy_stds = []

for strategy in strategies:
    # Light corruption
    light_data = df_light[df_light['strategy'] == strategy]
    light_data = light_data[light_data['dataset'].isin(all_kidney_datasets)]
    if len(light_data) > 0:
        light_means.append(light_data[metric].mean())
        light_stds.append(light_data[metric].std())
    else:
        light_means.append(0)
        light_stds.append(0)

    # Moderate corruption
    moderate_data = df_moderate[df_moderate['strategy'] == strategy]
    moderate_data = moderate_data[moderate_data['dataset'].isin(all_kidney_datasets)]
    if len(moderate_data) > 0:
        moderate_means.append(moderate_data[metric].mean())
        moderate_stds.append(moderate_data[metric].std())
    else:
        moderate_means.append(0)
        moderate_stds.append(0)

    # Heavy corruption
    heavy_data = df_heavy[df_heavy['strategy'] == strategy]
    heavy_data = heavy_data[heavy_data['dataset'].isin(all_kidney_datasets)]
    if len(heavy_data) > 0:
        heavy_means.append(heavy_data[metric].mean())
        heavy_stds.append(heavy_data[metric].std())
    else:
        heavy_means.append(0)
        heavy_stds.append(0)

# Plot bars with error bars
bars1 = ax.bar(x_pos - width, light_means, width,
               yerr=light_stds,
               label='Light (10%, Sev. 3)',
               alpha=0.8, capsize=5,
               color='#2ecc71', edgecolor='black', linewidth=1)

bars2 = ax.bar(x_pos, moderate_means, width,
               yerr=moderate_stds,
               label='Moderate (30%, Sev. 4)',
               alpha=0.8, capsize=5,
               color='#f39c12', edgecolor='black', linewidth=1)

bars3 = ax.bar(x_pos + width, heavy_means, width,
               yerr=heavy_stds,
               label='Heavy (40%, Sev. 5)',
               alpha=0.8, capsize=5,
               color='#e74c3c', edgecolor='black', linewidth=1)

ax.set_ylabel('Accuracy', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([0.5, 1])

ax.legend(fontsize=13, frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('corruption_levels_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: corruption_levels_comparison.png")
plt.close()

# =============================================================================
# Plot 2: Per-dataset view with box plots
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Combine all corruption data
df_all_corruption = pd.concat([
    df_light.assign(corruption_level='Light (10%, Sev. 3)'),
    df_moderate.assign(corruption_level='Moderate (30%, Sev. 4)'),
    df_heavy.assign(corruption_level='Heavy (40%, Sev. 5)')
])

df_all_corruption = df_all_corruption[df_all_corruption['dataset'].isin(all_kidney_datasets)]

metric = 'final_accuracy'

# Prepare data for box plot
data_to_plot = []
positions = []
colors = []

pos = 0
for strategy in strategies:
    strategy_data = df_all_corruption[df_all_corruption['strategy'] == strategy]

    # Get values for this strategy across all corruption levels
    values = strategy_data[metric].values
    if len(values) > 0:
        data_to_plot.append(values)
        positions.append(pos)
        colors.append(strategy_colors[strategy])
        pos += 1

# Create box plot
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
                medianprops=dict(color='black', linewidth=2),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=14)
ax.set_xticks(positions)
ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig('corruption_boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: corruption_boxplot_comparison.png")
plt.close()

# =============================================================================
# Summary statistics
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: Performance Under Different Corruption Levels")
print("="*80)

for corruption_name, df_corruption in [
    ('Light (10%, Severity 3)', df_light),
    ('Moderate (30%, Severity 4)', df_moderate),
    ('Heavy (40%, Severity 5)', df_heavy)
]:
    print(f"\n--- {corruption_name} ---")
    df_kidney = df_corruption[df_corruption['dataset'].isin(all_kidney_datasets)]
    
    if len(df_kidney) == 0:
        print("No data available")
        continue
    
    print(f"\nMetrics by Strategy (averaged across all kidney stone datasets):")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for strategy in strategies:
        strategy_data = df_kidney[df_kidney['strategy'] == strategy]
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

print("\n" + "="*80)
print("Analysis complete! Generated 2 plots:")
print("  1. corruption_levels_comparison.png - Bar chart with error bars")
print("  2. corruption_boxplot_comparison.png - Box plots showing distribution")
print("="*80)

