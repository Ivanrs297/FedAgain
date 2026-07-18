"""
Analysis: FedAvg vs FedAgain Performance with Different Numbers of Clients
Compare performance across 5, 10, and 20 clients for MIX kidney stone datasets
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
print("ANALYSIS: Client Scaling Comparison (FedAvg vs FedAgain)")
print("="*80)

# Filter for the specific conditions
conditions = {
    'data_distribution': 'non-iid-label',
    'corrupt_clients': True,
    'corrupt_ratio': 0.3,
    'corruption_severity': 3
}

df_filtered = df[
    (df['data_distribution'] == conditions['data_distribution']) &
    (df['corrupt_clients'] == conditions['corrupt_clients']) &
    (df['corrupt_ratio'] == conditions['corrupt_ratio']) &
    (df['corruption_severity'] == conditions['corruption_severity'])
]

print(f"\nTotal samples matching conditions: {len(df_filtered)}")
print(f"\nConditions:")
print(f"  - Data Distribution: {conditions['data_distribution']}")
print(f"  - Corrupted Clients: {conditions['corrupt_clients']}")
print(f"  - Corruption Ratio: {conditions['corrupt_ratio']}")
print(f"  - Corruption Severity: {conditions['corruption_severity']}")

# Filter for MIX datasets only
mix_datasets = ['DAUDON_MIX', 'JONATHAN_MIX', 'MYSTONE']
df_mix = df_filtered[df_filtered['dataset'].isin(mix_datasets)]

print(f"\nMIX dataset samples: {len(df_mix)}")
print(f"\nSamples by dataset:")
print(df_mix['dataset'].value_counts())

# Filter for FedAvg and FedAgain only
strategies = ['FedAvg', 'FedAgain']
df_comparison = df_mix[df_mix['strategy'].isin(strategies)]

print(f"\nSamples by strategy:")
print(df_comparison['strategy'].value_counts())

print(f"\nSamples by number of clients:")
print(df_comparison['num_clients'].value_counts().sort_index())

# Dataset mapping
dataset_mapping = {
    'DAUDON_MIX': 'A-MIX',
    'JONATHAN_MIX': 'B-MIX',
    'MYSTONE': 'C-MIX'
}

# Client numbers to analyze
client_numbers = [5, 10, 20]

# Check available data
print(f"\n" + "="*80)
print("Data availability check:")
print("="*80)
for dataset in mix_datasets:
    print(f"\n{dataset_mapping[dataset]}:")
    dataset_data = df_comparison[df_comparison['dataset'] == dataset]
    for strategy in strategies:
        strategy_data = dataset_data[dataset_data['strategy'] == strategy]
        print(f"  {strategy}:")
        for num_clients in client_numbers:
            client_data = strategy_data[strategy_data['num_clients'] == num_clients]
            print(f"    {num_clients} clients: {len(client_data)} samples")

# =============================================================================
# Plot: Accuracy comparison across different numbers of clients
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
fig.suptitle('FedAvg vs FedAgain: Accuracy Scaling with Number of Clients\n' +
             '(30% Corrupted Clients, Severity 3, Non-IID Label Distribution)',
             fontsize=16, y=0.98)

# Define colors for datasets
dataset_colors = {
    'A-MIX': '#d62728',  # Red
    'B-MIX': '#1f77b4',  # Blue
    'C-MIX': 'green',  # Pink
}

# Define line styles for strategies
strategy_styles = {
    'FedAvg': {'linestyle': '--', 'marker': 'o', 'markersize': 10, 'linewidth': 2.5},
    'FedAgain': {'linestyle': '-', 'marker': 's', 'markersize': 10, 'linewidth': 2.5}
}

# Store handles and labels for legend
handles = []
labels = []

# Plot each dataset-strategy combination
for dataset in mix_datasets:
    dataset_label = dataset_mapping[dataset]
    dataset_data = df_comparison[df_comparison['dataset'] == dataset]
    
    for strategy in strategies:
        strategy_data = dataset_data[dataset_data['strategy'] == strategy]
        
        # Collect accuracy values for each number of clients
        accuracies = []
        stds = []
        
        for num_clients in client_numbers:
            client_data = strategy_data[strategy_data['num_clients'] == num_clients]
            if len(client_data) > 0:
                accuracies.append(client_data['final_accuracy'].mean())
                stds.append(client_data['final_accuracy'].std())
            else:
                accuracies.append(np.nan)
                stds.append(0)
        
        # Plot line with error bars
        line = ax.plot(client_numbers, accuracies,
                      label=f'{dataset_label} - {strategy}',
                      color=dataset_colors[dataset_label],
                      linestyle=strategy_styles[strategy]['linestyle'],
                      marker=strategy_styles[strategy]['marker'],
                      markersize=strategy_styles[strategy]['markersize'],
                      linewidth=strategy_styles[strategy]['linewidth'],
                      alpha=0.85 if strategy == 'FedAgain' else 0.7)
        
        # Add error bars
        # ax.errorbar(client_numbers, accuracies, yerr=stds,
        #            color=dataset_colors[dataset_label],
        #            linestyle='none',
        #            capsize=5,
        #            capthick=2,
        #            alpha=0.6)

# Customize plot
ax.set_xlabel('Number of Clients', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_xticks(client_numbers)
ax.set_xticklabels(client_numbers, fontsize=14)
ax.set_ylim([0.4, 1.0])
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)

# Create custom legend with two columns
from matplotlib.lines import Line2D

# Column 1: Datasets (in specific order: A-MIX, B-MIX, C-MIX)
dataset_order = ['A-MIX', 'B-MIX', 'C-MIX']
dataset_handles = []
for dataset_label in dataset_order:
    color = dataset_colors[dataset_label]
    dataset_handles.append(Line2D([0], [0], color=color, linewidth=3, 
                                  label=dataset_label, alpha=0.85))

# Column 2: Methods (FedAgain first, then FedAvg)
method_handles = [
    Line2D([0], [0], color='gray', linewidth=2.5,
           linestyle='-', marker='s', markersize=10,
           label='FedAgain', alpha=0.85),
    Line2D([0], [0], color='gray', linewidth=2.5,
           linestyle='--', marker='o', markersize=10,
           label='FedAvg', alpha=0.7)
]

# Combine handles: interleave datasets and methods to create two-column layout
# We want: A-MIX    | FedAgain
#          B-MIX    | FedAvg
#          C-MIX    | 
legend_elements = []
for i in range(max(len(dataset_handles), len(method_handles))):
    if i < len(dataset_handles):
        legend_elements.append(dataset_handles[i])
    else:
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    if i < len(method_handles):
        legend_elements.append(method_handles[i])
    else:
        legend_elements.append(Line2D([0], [0], color='none', label=''))

# Add legend with 2 columns
ax.legend(handles=legend_elements,
         loc='best',
         fontsize=14,
         ncol=3,
         frameon=True,
         framealpha=0.9,
         edgecolor='black',
         columnspacing=1.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('fedagain_vs_fedavg_client_scaling.png', dpi=300, bbox_inches='tight')
print("\nSaved: fedagain_vs_fedavg_client_scaling.png")
plt.close()

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: Performance by Number of Clients")
print("="*80)

for num_clients in client_numbers:
    print(f"\n{'='*80}")
    print(f"Number of Clients: {num_clients}")
    print(f"{'='*80}")
    
    client_data = df_comparison[df_comparison['num_clients'] == num_clients]
    
    for dataset in mix_datasets:
        dataset_label = dataset_mapping[dataset]
        print(f"\n{dataset_label}:")
        print("-" * 80)
        
        dataset_client_data = client_data[client_data['dataset'] == dataset]
        
        for strategy in strategies:
            strategy_data = dataset_client_data[dataset_client_data['strategy'] == strategy]
            
            if len(strategy_data) > 0:
                acc_mean = strategy_data['final_accuracy'].mean()
                acc_std = strategy_data['final_accuracy'].std()
                f1_mean = strategy_data['final_f1'].mean()
                f1_std = strategy_data['final_f1'].std()
                
                print(f"  {strategy:10s}: Accuracy = {acc_mean:.4f}±{acc_std:.4f}, " +
                      f"F1 Score = {f1_mean:.4f}±{f1_std:.4f}")
            else:
                print(f"  {strategy:10s}: No data available")
        
        # Calculate improvement
        fedavg_data = dataset_client_data[dataset_client_data['strategy'] == 'FedAvg']
        fedagain_data = dataset_client_data[dataset_client_data['strategy'] == 'FedAgain']
        
        if len(fedavg_data) > 0 and len(fedagain_data) > 0:
            acc_improvement = ((fedagain_data['final_accuracy'].mean() - 
                               fedavg_data['final_accuracy'].mean()) / 
                              fedavg_data['final_accuracy'].mean()) * 100
            print(f"\n  FedAgain Improvement: {acc_improvement:+.2f}%")

# =============================================================================
# Generate summary CSV
# =============================================================================

summary_data = []
for num_clients in client_numbers:
    client_data = df_comparison[df_comparison['num_clients'] == num_clients]
    
    for dataset in mix_datasets:
        dataset_label = dataset_mapping[dataset]
        dataset_client_data = client_data[client_data['dataset'] == dataset]
        
        for strategy in strategies:
            strategy_data = dataset_client_data[dataset_client_data['strategy'] == strategy]
            
            if len(strategy_data) > 0:
                summary_data.append({
                    'num_clients': num_clients,
                    'dataset': dataset_label,
                    'strategy': strategy,
                    'accuracy_mean': strategy_data['final_accuracy'].mean(),
                    'accuracy_std': strategy_data['final_accuracy'].std(),
                    'precision_mean': strategy_data['final_precision'].mean(),
                    'precision_std': strategy_data['final_precision'].std(),
                    'recall_mean': strategy_data['final_recall'].mean(),
                    'recall_std': strategy_data['final_recall'].std(),
                    'f1_mean': strategy_data['final_f1'].mean(),
                    'f1_std': strategy_data['final_f1'].std(),
                    'num_samples': len(strategy_data)
                })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('client_scaling_summary.csv', index=False)
print("\n" + "="*80)
print("Summary saved to: client_scaling_summary.csv")
print("="*80)

