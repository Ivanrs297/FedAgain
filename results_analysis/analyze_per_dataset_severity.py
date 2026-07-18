import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")

# Read the experiments data
df = pd.read_csv('../results/experiments.csv')

# Filter for FedAgain vs FedAvg comparison
# Conditions: non-iid-label, corrupt_ratio=0.3, varying severity 1-5
df_comparison = df[
    (df['corrupt_clients'] == True) & 
    (df['corrupt_ratio'] == 0.3) &
    (df['data_distribution'] == 'non-iid-label') &
    (df['strategy'].isin(['FedAvg', 'FedAgain']))
].copy()

print(f"Total comparison samples: {len(df_comparison)}")

# Define severity levels
severity_levels = [1, 2, 3, 4, 5]

# Define colors for strategies
bar_colors = {
    'FedAvg': '#e74c3c',      # Red
    'FedAgain': '#2ecc71'     # Green
}

# Filter for all kidney stone datasets including MyStone
# Map to A, B, C labels
dataset_mapping = {
    'DAUDON_MIX': 'A-MIX',
    'DAUDON_SEC': 'A-SEC',
    'DAUDON_SUR': 'A-SUR',
    'JONATHAN_MIX': 'B-MIX',
    'JONATHAN_SEC': 'B-SEC',
    'JONATHAN_SUR': 'B-SUR',
    'MYSTONE': 'C-MIX'
}

all_kidney_datasets = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
                       'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 
                       'MYSTONE']

df_kidney = df_comparison[df_comparison['dataset'].isin(all_kidney_datasets)].copy()
df_kidney['dataset_label'] = df_kidney['dataset'].map(dataset_mapping)

print(f"\nKidney stone dataset samples: {len(df_kidney)}")
print(f"\nSamples per dataset:")
for dataset in all_kidney_datasets:
    count = len(df_kidney[df_kidney['dataset'] == dataset])
    print(f"  {dataset_mapping[dataset]}: {count} samples")

# =============================================================================
# Plot: Only MIX subsets in a single row
# =============================================================================

# Filter for only MIX datasets
mix_datasets = ['DAUDON_MIX', 'JONATHAN_MIX', 'MYSTONE']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('FedAvg vs FedAgain: F1 Score Across Corruption Severity Levels (MIX Subsets)\n(30% Corrupted Clients, Non-IID Label Distribution)', 
             fontsize=16, y=1.00)

# Store handles and labels for shared legend
handles = []
labels = []

for idx, dataset in enumerate(mix_datasets):
    ax = axes[idx]
    dataset_label = dataset_mapping[dataset]
    
    # Filter data for this dataset
    dataset_data = df_kidney[df_kidney['dataset'] == dataset]
    
    # Prepare data
    x_pos = np.arange(len(severity_levels))
    width = 0.35
    
    fedavg_means = []
    fedagain_means = []
    
    for severity in severity_levels:
        # FedAvg data
        fedavg_data = dataset_data[
            (dataset_data['strategy'] == 'FedAvg') & 
            (dataset_data['corruption_severity'] == severity)
        ]
        if len(fedavg_data) > 0:
            fedavg_means.append(fedavg_data['final_f1'].mean())
        else:
            fedavg_means.append(np.nan)
        
        # FedAgain data
        fedagain_data = dataset_data[
            (dataset_data['strategy'] == 'FedAgain') & 
            (dataset_data['corruption_severity'] == severity)
        ]
        if len(fedagain_data) > 0:
            fedagain_means.append(fedagain_data['final_f1'].mean())
        else:
            fedagain_means.append(np.nan)
    
    # Plot bars without error bars
    bars1 = ax.bar(x_pos - width/2, fedavg_means, width,
                   label='FedAvg',
                   color=bar_colors['FedAvg'],
                   edgecolor='black',
                   linewidth=1.5,
                   alpha=0.85)
    
    bars2 = ax.bar(x_pos + width/2, fedagain_means, width,
                   label='FedAgain',
                   color=bar_colors['FedAgain'],
                   edgecolor='black',
                   linewidth=1.5,
                   alpha=0.85)
    
    # Collect handles for legend from first subplot
    if idx == 0:
        handles = [bars1, bars2]
        labels = ['FedAvg', 'FedAgain']
    
    ax.set_xlabel('Corruption Severity', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{dataset_label}', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(severity_levels, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Set y-axis limits - consistent across all subplots for comparison
    ax.set_ylim([0.4, 1.0])

# Add shared legend at the bottom center
fig.legend(handles, labels, 
          loc='lower center', 
          bbox_to_anchor=(0.5, -0.08),
          ncol=2, 
          fontsize=13,
          frameon=True,
          framealpha=0.9,
          edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('fedagain_vs_fedavg_per_dataset.png', dpi=300, bbox_inches='tight')
print("\nSaved: fedagain_vs_fedavg_per_dataset.png")
plt.close()

# =============================================================================
# Summary statistics per dataset
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: FedAgain vs FedAvg by Individual Dataset")
print("Configuration: 30% Corrupted Clients, Non-IID Label Distribution")
print("="*80)

for dataset in all_kidney_datasets:
    dataset_label = dataset_mapping[dataset]
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_label}")
    print(f"{'='*80}")
    
    dataset_data = df_kidney[df_kidney['dataset'] == dataset]
    
    if len(dataset_data) == 0:
        print("No data available")
        continue
    
    # Calculate overall improvement
    fedavg_overall = dataset_data[dataset_data['strategy'] == 'FedAvg']['final_accuracy'].mean()
    fedagain_overall = dataset_data[dataset_data['strategy'] == 'FedAgain']['final_accuracy'].mean()
    
    if not np.isnan(fedavg_overall) and not np.isnan(fedagain_overall):
        overall_improvement = ((fedagain_overall - fedavg_overall) / fedavg_overall) * 100
        print(f"\nOverall Average Accuracy:")
        print(f"  FedAvg:  {fedavg_overall:.4f}")
        print(f"  FedAgain: {fedagain_overall:.4f}")
        print(f"  Improvement: {overall_improvement:+.2f}%")
    
    print(f"\nAccuracy by Severity Level:")
    print("-" * 80)
    print(f"{'Severity':<12} {'FedAvg':<15} {'FedAgain':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for severity in severity_levels:
        fedavg_data = dataset_data[
            (dataset_data['strategy'] == 'FedAvg') & 
            (dataset_data['corruption_severity'] == severity)
        ]
        fedagain_data = dataset_data[
            (dataset_data['strategy'] == 'FedAgain') & 
            (dataset_data['corruption_severity'] == severity)
        ]
        
        if len(fedavg_data) > 0 and len(fedagain_data) > 0:
            fedavg_acc = fedavg_data['final_accuracy'].mean()
            fedagain_acc = fedagain_data['final_accuracy'].mean()
            improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
            
            print(f"{severity:<12} {fedavg_acc:<15.4f} {fedagain_acc:<15.4f} {improvement:+.2f}%")
        else:
            print(f"{severity:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

print("\n" + "="*80)
print("Analysis complete!")
print("Generated: fedagain_vs_fedavg_per_dataset.png")
print("="*80)

