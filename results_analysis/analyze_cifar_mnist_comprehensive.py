"""
Comprehensive Analysis: FedAgain Performance on CIFAR10 and MNIST
Four subplots: 
1. Clean baseline (no corruption)
2. Corruption levels comparison
3. Severity levels comparison
4. Client scaling comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Read the data
df = pd.read_csv('../results/experiments.csv')

print("="*80)
print("COMPREHENSIVE ANALYSIS: CIFAR10 and MNIST")
print("="*80)

# Filter for CIFAR10 and MNIST datasets
datasets = ['CIFAR10', 'MNIST']
df_filtered = df[df['dataset'].isin(datasets)]

print(f"\nTotal samples for CIFAR10 and MNIST: {len(df_filtered)}")
print(f"\nSamples by dataset:")
print(df_filtered['dataset'].value_counts())

# All strategies to compare
strategies = ['FedAvg', 'FedProx', 'FedMedian', 'Bulyan', 'FedAgain']

# Define colors for strategies
strategy_colors = {
    'FedAvg': '#e74c3c',      # Red
    'FedProx': '#3498db',     # Blue
    'FedMedian': '#2ecc71',   # Green
    'Bulyan': '#f39c12',      # Orange
    'FedAgain': '#9b59b6'     # Purple
}

# =============================================================================
# Create figure with 4 subplots
# =============================================================================

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Main title
fig.suptitle('Comprehensive Performance Analysis: CIFAR10 and MNIST Datasets',
             fontsize=18, fontweight='bold', y=0.98)

# =============================================================================
# SUBPLOT 1: Clean Baseline (No Corruption)
# =============================================================================

ax1 = fig.add_subplot(gs[0, 0])

# Filter for no corruption
df_clean = df_filtered[df_filtered['corrupt_clients'] == False]

print(f"\n{'='*80}")
print("SUBPLOT 1: Clean Baseline")
print(f"{'='*80}")
print(f"Clean samples: {len(df_clean)}")

# Plot lines for each dataset
for dataset in datasets:
    dataset_data = df_clean[df_clean['dataset'] == dataset]
    
    accuracies = []
    for strategy in strategies:
        strategy_data = dataset_data[dataset_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            accuracies.append(strategy_data['final_accuracy'].mean())
        else:
            accuracies.append(np.nan)
    
    # Plot line
    marker = 'o' if dataset == 'CIFAR10' else 's'
    linestyle = '-' if dataset == 'CIFAR10' else '--'
    ax1.plot(strategies, accuracies,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            markersize=10,
            label=dataset,
            alpha=0.85)

ax1.set_title('(a) Clean Baseline Performance', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_xticklabels(strategies, rotation=20, ha='right', fontsize=11)
ax1.set_ylim([0.5, 1.0])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(fontsize=11, loc='lower right', framealpha=0.9, edgecolor='black')

# =============================================================================
# SUBPLOT 2: Corruption Levels Comparison (Bar Plot) - Separate by Dataset
# =============================================================================

ax2 = fig.add_subplot(gs[0, 1])

print(f"\n{'='*80}")
print("SUBPLOT 2: Corruption Levels")
print(f"{'='*80}")

# Define corruption levels
corruption_configs = [
    {'ratio': 0.1, 'severity': 3, 'label': 'Light'},
    {'ratio': 0.3, 'severity': 4, 'label': 'Moderate'},
    {'ratio': 0.4, 'severity': 5, 'label': 'Heavy'}
]

# Create grouped bars by dataset and corruption level
num_groups = len(corruption_configs)
group_width = 0.8
bar_width = group_width / (len(datasets))
x_base = np.arange(len(strategies))

colors_dataset = {'CIFAR10': '#3498db', 'MNIST': '#e74c3c'}
hatches = ['///', '\\\\\\', 'xxx']

for config_idx, config in enumerate(corruption_configs):
    df_corruption = df_filtered[
        (df_filtered['corrupt_clients'] == True) &
        (df_filtered['corrupt_ratio'] == config['ratio']) &
        (df_filtered['corruption_severity'] == config['severity'])
    ]
    
    print(f"\n{config['label']}: {len(df_corruption)} samples")
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_corruption = df_corruption[df_corruption['dataset'] == dataset]
        
        # Calculate mean accuracy for each strategy
        accuracies = []
        for strategy in strategies:
            strategy_data = dataset_corruption[dataset_corruption['strategy'] == strategy]
            if len(strategy_data) > 0:
                accuracies.append(strategy_data['final_accuracy'].mean())
            else:
                accuracies.append(0)
        
        # Plot bars
        offset = (config_idx * num_groups + dataset_idx - num_groups) * bar_width
        ax2.bar(x_base + offset, accuracies, bar_width,
               label=f'{dataset} - {config["label"]}' if config_idx == 0 or dataset_idx == 0 else '',
               alpha=0.85,
               edgecolor='black',
               linewidth=1.2,
               color=colors_dataset[dataset],
               hatch=hatches[config_idx] if dataset == 'CIFAR10' else None)

ax2.set_title('(b) Performance Under Corruption Levels', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_xticks(x_base)
ax2.set_xticklabels(strategies, rotation=20, ha='right', fontsize=11)
ax2.set_ylim([0.5, 1.0])
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_dataset['CIFAR10'], edgecolor='black', label='CIFAR10', alpha=0.85),
    Patch(facecolor=colors_dataset['MNIST'], edgecolor='black', label='MNIST', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='Light', hatch='///', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='Moderate', hatch='\\\\\\', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='Heavy', hatch='xxx', alpha=0.85),
]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower left', framealpha=0.9, edgecolor='black', ncol=2)

# =============================================================================
# SUBPLOT 3: Severity Levels Comparison (FedAvg vs FedAgain) - Separate by Dataset
# =============================================================================

ax3 = fig.add_subplot(gs[1, 0])

print(f"\n{'='*80}")
print("SUBPLOT 3: Severity Levels (FedAvg vs FedAgain)")
print(f"{'='*80}")

# Filter for 30% corrupted clients, non-IID label distribution
df_severity = df_filtered[
    (df_filtered['corrupt_clients'] == True) &
    (df_filtered['corrupt_ratio'] == 0.3) &
    (df_filtered['data_distribution'] == 'non-iid-label')
]

print(f"Severity samples: {len(df_severity)}")

severity_levels = [1, 2, 3, 4, 5]
comparison_strategies = ['FedAvg', 'FedAgain']

x_pos = np.arange(len(severity_levels))
width = 0.2

colors_dataset = {'CIFAR10': '#3498db', 'MNIST': '#e74c3c'}

for dataset_idx, dataset in enumerate(datasets):
    dataset_severity = df_severity[df_severity['dataset'] == dataset]
    
    for strategy_idx, strategy in enumerate(comparison_strategies):
        accuracies = []
        
        for severity in severity_levels:
            severity_data = dataset_severity[
                (dataset_severity['strategy'] == strategy) &
                (dataset_severity['corruption_severity'] == severity)
            ]
            if len(severity_data) > 0:
                accuracies.append(severity_data['final_accuracy'].mean())
            else:
                accuracies.append(0)
        
        # Plot bars
        offset = (dataset_idx * 2 + strategy_idx - 1.5) * width
        hatch = None if strategy == 'FedAgain' else '///'
        ax3.bar(x_pos + offset, accuracies, width,
               label=f'{dataset} - {strategy}',
               alpha=0.85,
               edgecolor='black',
               linewidth=1.2,
               color=colors_dataset[dataset],
               hatch=hatch)

ax3.set_title('(c) Corruption Severity Comparison', fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel('Corruption Severity', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(severity_levels, fontsize=11)
ax3.set_ylim([0.5, 1.0])
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Create custom legend
legend_elements = [
    Patch(facecolor=colors_dataset['CIFAR10'], edgecolor='black', label='CIFAR10', alpha=0.85),
    Patch(facecolor=colors_dataset['MNIST'], edgecolor='black', label='MNIST', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='FedAvg', hatch='///', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='FedAgain', alpha=0.85),
]
ax3.legend(handles=legend_elements, fontsize=10, loc='lower left', framealpha=0.9, edgecolor='black', ncol=2)

# =============================================================================
# SUBPLOT 4: Client Scaling (FedAvg vs FedAgain) - Separate by Dataset
# =============================================================================

ax4 = fig.add_subplot(gs[1, 1])

print(f"\n{'='*80}")
print("SUBPLOT 4: Client Scaling")
print(f"{'='*80}")

# Filter for 30% corrupted clients, severity 3, non-IID label
df_scaling = df_filtered[
    (df_filtered['corrupt_clients'] == True) &
    (df_filtered['corrupt_ratio'] == 0.3) &
    (df_filtered['corruption_severity'] == 3) &
    (df_filtered['data_distribution'] == 'non-iid-label')
]

print(f"Scaling samples: {len(df_scaling)}")

client_numbers = [5, 10, 20]

x_pos = np.arange(len(client_numbers))
width = 0.2

colors_dataset = {'CIFAR10': '#3498db', 'MNIST': '#e74c3c'}

for dataset_idx, dataset in enumerate(datasets):
    dataset_scaling = df_scaling[df_scaling['dataset'] == dataset]
    
    for strategy_idx, strategy in enumerate(comparison_strategies):
        accuracies = []
        
        for num_clients in client_numbers:
            client_data = dataset_scaling[
                (dataset_scaling['strategy'] == strategy) &
                (dataset_scaling['num_clients'] == num_clients)
            ]
            if len(client_data) > 0:
                accuracies.append(client_data['final_accuracy'].mean())
            else:
                accuracies.append(0)
        
        # Plot bars
        offset = (dataset_idx * 2 + strategy_idx - 1.5) * width
        hatch = None if strategy == 'FedAgain' else '///'
        ax4.bar(x_pos + offset, accuracies, width,
               label=f'{dataset} - {strategy}',
               alpha=0.85,
               edgecolor='black',
               linewidth=1.2,
               color=colors_dataset[dataset],
               hatch=hatch)

ax4.set_title('(d) Client Scaling Performance', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(client_numbers, fontsize=11)
ax4.set_ylim([0.5, 1.0])
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Create custom legend
legend_elements = [
    Patch(facecolor=colors_dataset['CIFAR10'], edgecolor='black', label='CIFAR10', alpha=0.85),
    Patch(facecolor=colors_dataset['MNIST'], edgecolor='black', label='MNIST', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='FedAvg', hatch='///', alpha=0.85),
    Patch(facecolor='gray', edgecolor='black', label='FedAgain', alpha=0.85),
]
ax4.legend(handles=legend_elements, fontsize=10, loc='lower left', framealpha=0.9, edgecolor='black', ncol=2)

# =============================================================================
# Save figure
# =============================================================================

plt.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('cifar_mnist_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Saved: cifar_mnist_comprehensive_analysis.png")
print("="*80)
plt.close()

# =============================================================================
# Generate detailed summary statistics
# =============================================================================

print("\n" + "="*80)
print("DETAILED SUMMARY STATISTICS")
print("="*80)

# Summary for each subplot
print("\n" + "="*80)
print("1. CLEAN BASELINE PERFORMANCE")
print("="*80)

for dataset in datasets:
    print(f"\n{dataset}:")
    print("-" * 80)
    dataset_clean = df_clean[df_clean['dataset'] == dataset]
    
    for strategy in strategies:
        strategy_data = dataset_clean[dataset_clean['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc_mean = strategy_data['final_accuracy'].mean()
            acc_std = strategy_data['final_accuracy'].std()
            print(f"  {strategy:12s}: Accuracy = {acc_mean:.4f}±{acc_std:.4f}")

print("\n" + "="*80)
print("2. CORRUPTION LEVELS PERFORMANCE (Averaged across datasets)")
print("="*80)

for config in corruption_configs:
    print(f"\n{config['label']}:")
    print("-" * 80)
    
    df_corruption = df_filtered[
        (df_filtered['corrupt_clients'] == True) &
        (df_filtered['corrupt_ratio'] == config['ratio']) &
        (df_filtered['corruption_severity'] == config['severity'])
    ]
    
    for strategy in strategies:
        strategy_data = df_corruption[df_corruption['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc_mean = strategy_data['final_accuracy'].mean()
            acc_std = strategy_data['final_accuracy'].std()
            print(f"  {strategy:12s}: Accuracy = {acc_mean:.4f}±{acc_std:.4f}")

print("\n" + "="*80)
print("3. SEVERITY LEVELS (30% corrupted, non-IID label)")
print("="*80)

for severity in severity_levels:
    print(f"\nSeverity {severity}:")
    print("-" * 80)
    
    severity_data = df_severity[df_severity['corruption_severity'] == severity]
    
    for strategy in comparison_strategies:
        strategy_data = severity_data[severity_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc_mean = strategy_data['final_accuracy'].mean()
            acc_std = strategy_data['final_accuracy'].std()
            print(f"  {strategy:12s}: Accuracy = {acc_mean:.4f}±{acc_std:.4f}")

print("\n" + "="*80)
print("4. CLIENT SCALING (30% corrupted, severity 3, non-IID label)")
print("="*80)

for num_clients in client_numbers:
    print(f"\n{num_clients} Clients:")
    print("-" * 80)
    
    client_data = df_scaling[df_scaling['num_clients'] == num_clients]
    
    for strategy in comparison_strategies:
        strategy_data = client_data[client_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            acc_mean = strategy_data['final_accuracy'].mean()
            acc_std = strategy_data['final_accuracy'].std()
            print(f"  {strategy:12s}: Accuracy = {acc_mean:.4f}±{acc_std:.4f}")

# =============================================================================
# Generate summary CSV
# =============================================================================

summary_data = []

# Add clean baseline data
for dataset in datasets:
    dataset_clean = df_clean[df_clean['dataset'] == dataset]
    for strategy in strategies:
        strategy_data = dataset_clean[dataset_clean['strategy'] == strategy]
        if len(strategy_data) > 0:
            summary_data.append({
                'analysis': 'clean_baseline',
                'dataset': dataset,
                'strategy': strategy,
                'configuration': 'no_corruption',
                'accuracy_mean': strategy_data['final_accuracy'].mean(),
                'accuracy_std': strategy_data['final_accuracy'].std(),
                'num_samples': len(strategy_data)
            })

# Add corruption levels data
for config in corruption_configs:
    df_corruption = df_filtered[
        (df_filtered['corrupt_clients'] == True) &
        (df_filtered['corrupt_ratio'] == config['ratio']) &
        (df_filtered['corruption_severity'] == config['severity'])
    ]
    
    for strategy in strategies:
        strategy_data = df_corruption[df_corruption['strategy'] == strategy]
        if len(strategy_data) > 0:
            summary_data.append({
                'analysis': 'corruption_levels',
                'dataset': 'CIFAR10+MNIST',
                'strategy': strategy,
                'configuration': config['label'],
                'accuracy_mean': strategy_data['final_accuracy'].mean(),
                'accuracy_std': strategy_data['final_accuracy'].std(),
                'num_samples': len(strategy_data)
            })

# Add severity levels data
for severity in severity_levels:
    severity_data = df_severity[df_severity['corruption_severity'] == severity]
    for strategy in comparison_strategies:
        strategy_data = severity_data[severity_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            summary_data.append({
                'analysis': 'severity_levels',
                'dataset': 'CIFAR10+MNIST',
                'strategy': strategy,
                'configuration': f'severity_{severity}',
                'accuracy_mean': strategy_data['final_accuracy'].mean(),
                'accuracy_std': strategy_data['final_accuracy'].std(),
                'num_samples': len(strategy_data)
            })

# Add client scaling data
for num_clients in client_numbers:
    client_data = df_scaling[df_scaling['num_clients'] == num_clients]
    for strategy in comparison_strategies:
        strategy_data = client_data[client_data['strategy'] == strategy]
        if len(strategy_data) > 0:
            summary_data.append({
                'analysis': 'client_scaling',
                'dataset': 'CIFAR10+MNIST',
                'strategy': strategy,
                'configuration': f'{num_clients}_clients',
                'accuracy_mean': strategy_data['final_accuracy'].mean(),
                'accuracy_std': strategy_data['final_accuracy'].std(),
                'num_samples': len(strategy_data)
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('cifar_mnist_comprehensive_summary.csv', index=False)
print("\n" + "="*80)
print("Summary saved to: cifar_mnist_comprehensive_summary.csv")
print("="*80)

