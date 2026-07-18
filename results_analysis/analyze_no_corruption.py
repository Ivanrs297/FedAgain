import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Read the experiments data
df = pd.read_csv('../results/experiments.csv')

# Filter for no corruption (corrupt_clients=False)
df_no_corruption = df[df['corrupt_clients'] == False].copy()

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

# =============================================================================
# Plot 1: All Kidney Stone Datasets (Daudon=A, Jonathan=B, MyStone=C)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

# Filter for all kidney stone datasets including MyStone
all_kidney_datasets = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
                       'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 
                       'MYSTONE']
df_all_kidney = df_no_corruption[df_no_corruption['dataset'].isin(all_kidney_datasets)]

# Define colors and styles for each dataset
# A = Daudon (Michel), B = Jonathan, C = MyStone
dataset_styles = {
    'DAUDON_MIX': {'color': '#d62728', 'marker': 's', 'linestyle': '-', 'label': 'A-MIX'},
    'DAUDON_SEC': {'color': '#9467bd', 'marker': 's', 'linestyle': '-', 'label': 'A-SEC'},
    'DAUDON_SUR': {'color': '#8c564b', 'marker': 's', 'linestyle': '-', 'label': 'A-SUR'},
    'JONATHAN_MIX': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '--', 'label': 'B-MIX'},
    'JONATHAN_SEC': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '--', 'label': 'B-SEC'},
    'JONATHAN_SUR': {'color': '#2ca02c', 'marker': 'o', 'linestyle': '--', 'label': 'B-SUR'},
    'MYSTONE': {'color': '#e377c2', 'marker': 'D', 'linestyle': '-.', 'label': 'C-MIX'},
}

# Store handles and labels for shared legend
handles = []
labels = []

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    # Prepare data for plotting
    x_pos = np.arange(len(strategies))
    
    # Collect all values to set y-axis limits
    all_values = []
    # Track per-dataset spreads (max - min across strategies)
    dataset_spreads = []

    # Plot each dataset
    for dataset in all_kidney_datasets:
        dataset_data = df_all_kidney[df_all_kidney['dataset'] == dataset]

        values = []
        for strategy in strategies:
            strategy_data = dataset_data[dataset_data['strategy'] == strategy]
            if len(strategy_data) > 0:
                val = strategy_data[metric].mean()
                values.append(val)
                all_values.append(val)
            else:
                values.append(np.nan)

        # Get style for this dataset
        style = dataset_styles[dataset]

        # Compute per-dataset spread across strategies
        valid_vals = [v for v in values if not np.isnan(v)]
        if len(valid_vals) >= 2:
            ds_spread = max(valid_vals) - min(valid_vals)
            dataset_spreads.append(ds_spread)
            # Draw shaded band between min and max across strategies
            ax.fill_between(x_pos, [min(valid_vals)] * len(x_pos),
                            [max(valid_vals)] * len(x_pos),
                            color=style['color'], alpha=0.08)

        # Plot line with markers
        line, = ax.plot(x_pos, values,
                marker=style['marker'],
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2.5,
                markersize=8,
                alpha=0.85)

        # Collect handles and labels from first subplot only
        if idx == 0:
            handles.append(line)
            labels.append(style['label'])

    ax.set_title(f'{label}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=14)

    # Set y-axis limits to show variation better
    if all_values:
        valid = [v for v in all_values if not np.isnan(v)]
        y_min = min(valid) * 0.95
        y_max = 1.0 if max(valid) > 0.9 else max(valid) * 1.05
        ax.set_ylim([y_min, y_max])

# Add shared legend at the bottom center
fig.legend(handles, labels,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.02),
          ncol=4,
          fontsize=13,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout(rect=[0, 0.06, 1, 1.0])
plt.savefig('kidney_stone_no_corruption.png', dpi=300, bbox_inches='tight')
print("Saved: kidney_stone_no_corruption.png")
plt.close()

# =============================================================================
# Plot 2: MyStone dataset
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('MyStone Dataset: Strategy Performance without Corruption', 
             fontsize=14, fontweight='bold', y=0.995)

df_mystone = df_no_corruption[df_no_corruption['dataset'] == 'MYSTONE']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    values = []
    for strategy in strategies:
        strategy_data = df_mystone[df_mystone['strategy'] == strategy]
        if len(strategy_data) > 0:
            values.append(strategy_data[metric].mean())
        else:
            values.append(0)
    
    # Create bar plot
    bars = ax.bar(strategies, values, color=[strategy_colors[s] for s in strategies],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Set y-axis limits
    if len(values) > 0 and max(values) > 0:
        y_min = min([v for v in values if v > 0]) * 0.95
        y_max = 1.0 if max(values) > 0.9 else max(values) * 1.05
        ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig('mystone_no_corruption.png', dpi=300, bbox_inches='tight')
print("Saved: mystone_no_corruption.png")
plt.close()

# =============================================================================
# Plot 3: CIFAR10 dataset
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('CIFAR10 Dataset: Strategy Performance without Corruption', 
             fontsize=14, fontweight='bold', y=0.995)

df_cifar = df_no_corruption[df_no_corruption['dataset'] == 'CIFAR10']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    values = []
    for strategy in strategies:
        strategy_data = df_cifar[df_cifar['strategy'] == strategy]
        if len(strategy_data) > 0:
            values.append(strategy_data[metric].mean())
        else:
            values.append(0)
    
    # Create bar plot
    bars = ax.bar(strategies, values, color=[strategy_colors[s] for s in strategies],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Set y-axis limits
    if len(values) > 0 and max(values) > 0:
        y_min = min([v for v in values if v > 0]) * 0.95
        y_max = 1.0 if max(values) > 0.9 else max(values) * 1.05
        ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig('cifar10_no_corruption.png', dpi=300, bbox_inches='tight')
print("Saved: cifar10_no_corruption.png")
plt.close()

# =============================================================================
# Plot 4: MNIST dataset
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('MNIST Dataset: Strategy Performance without Corruption', 
             fontsize=14, fontweight='bold', y=0.995)

df_mnist = df_no_corruption[df_no_corruption['dataset'] == 'MNIST']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    values = []
    for strategy in strategies:
        strategy_data = df_mnist[df_mnist['strategy'] == strategy]
        if len(strategy_data) > 0:
            values.append(strategy_data[metric].mean())
        else:
            values.append(0)
    
    # Create bar plot
    bars = ax.bar(strategies, values, color=[strategy_colors[s] for s in strategies],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Set y-axis limits - MNIST has very high accuracy
    if len(values) > 0 and max(values) > 0:
        y_min = min([v for v in values if v > 0]) * 0.998  # Tighter range for MNIST
        y_max = 1.0
        ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig('mnist_no_corruption.png', dpi=300, bbox_inches='tight')
print("Saved: mnist_no_corruption.png")
plt.close()

# =============================================================================
# Summary statistics table
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: Strategy Performance without Corruption")
print("="*80)

# Define kidney datasets for summary
kidney_datasets = ['JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 
                   'DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR']

for dataset_group in [['MYSTONE'], ['CIFAR10'], ['MNIST'], kidney_datasets]:
    if dataset_group == kidney_datasets:
        print("\n--- KIDNEY STONE DATASETS (Jonathan + Daudon) ---")
        df_group = df_no_corruption[df_no_corruption['dataset'].isin(dataset_group)]
    else:
        dataset_name = dataset_group[0]
        print(f"\n--- {dataset_name} DATASET ---")
        df_group = df_no_corruption[df_no_corruption['dataset'] == dataset_name]
    
    if len(df_group) == 0:
        print("No data available")
        continue
    
    print(f"\nMetrics by Strategy:")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 80)
    
    for strategy in strategies:
        strategy_data = df_group[df_group['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc = strategy_data['final_accuracy'].mean()
            prec = strategy_data['final_precision'].mean()
            rec = strategy_data['final_recall'].mean()
            f1 = strategy_data['final_f1'].mean()
            print(f"{strategy:<12} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        else:
            print(f"{strategy:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("\n" + "="*80)
print("Analysis complete! Generated 4 plots:")
print("  1. kidney_stone_no_corruption.png - Jonathan & Daudon datasets (overlapped)")
print("  2. mystone_no_corruption.png - MyStone dataset")
print("  3. cifar10_no_corruption.png - CIFAR10 dataset")
print("  4. mnist_no_corruption.png - MNIST dataset")
print("="*80)

