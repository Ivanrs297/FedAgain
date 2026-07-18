"""
CIFAR-10 and MNIST: FedAvg vs FedAgain
2x2 grid — Left: CIFAR-10, Right: MNIST
Top: Corruption escalation, Bottom: Client scaling
Each dataset gets its own y-axis range for clarity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.style.use('seaborn-v0_8-paper')

# Read data
df = pd.read_csv('../results/experiments.csv')
for col in ['final_accuracy', 'corrupt_ratio', 'corruption_severity', 'num_clients']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

datasets = ['CIFAR10', 'MNIST']
df_filtered = df[df['dataset'].isin(datasets)]

strategies = ['FedAvg', 'FedAgain']
colors = {'FedAvg': '#e74c3c', 'FedAgain': '#2ecc71'}
markers = {'FedAvg': 's', 'FedAgain': 'o'}

# Corruption escalation scenarios
scenarios = [
    ('Light', 0.1, 3),
    ('Medium', 0.3, 4),
    ('Heavy', 0.4, 5),
]

# Client scaling: ratio=0.3, severity=3, non-iid-label
client_numbers = [5, 10, 20]

# =============================================================================
# Filter data
# =============================================================================

df_esc = df_filtered[
    (df_filtered['corrupt_clients'] == True) &
    (df_filtered['data_distribution'] == 'non-iid-label') &
    (df_filtered['num_clients'] == 10)
]

df_scaling = df_filtered[
    (df_filtered['corrupt_clients'] == True) &
    (df_filtered['corrupt_ratio'] == 0.3) &
    (df_filtered['corruption_severity'] == 3) &
    (df_filtered['data_distribution'] == 'non-iid-label')
]

# =============================================================================
# Create 2x2 figure
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

dataset_titles = {'CIFAR10': 'CIFAR-10', 'MNIST': 'MNIST'}

# --- Top row: Corruption Escalation ---
for col_idx, dataset in enumerate(datasets):
    ax = axes[0, col_idx]
    ds_esc = df_esc[df_esc['dataset'] == dataset]

    scenario_x = np.arange(len(scenarios))

    for strat in strategies:
        accs = []
        for _, ratio, sev in scenarios:
            sub = ds_esc[(ds_esc['strategy'] == strat) &
                         (ds_esc['corrupt_ratio'] == ratio) &
                         (ds_esc['corruption_severity'] == sev)]
            accs.append(sub['final_accuracy'].mean() if len(sub) > 0 else np.nan)

        lw = 2.8 if strat == 'FedAgain' else 1.8
        ms = 11 if strat == 'FedAgain' else 8
        zorder = 4 if strat == 'FedAgain' else 3
        ls = '-' if strat == 'FedAgain' else '--'

        ax.plot(scenario_x, accs, color=colors[strat], marker=markers[strat],
                markersize=ms, linewidth=lw, linestyle=ls, zorder=zorder,
                markeredgecolor='black', markeredgewidth=1.0)

    # Shade gap
    favg_accs = []
    fagain_accs = []
    for _, ratio, sev in scenarios:
        fa = ds_esc[(ds_esc['strategy'] == 'FedAvg') & (ds_esc['corrupt_ratio'] == ratio) & (ds_esc['corruption_severity'] == sev)]
        fg = ds_esc[(ds_esc['strategy'] == 'FedAgain') & (ds_esc['corrupt_ratio'] == ratio) & (ds_esc['corruption_severity'] == sev)]
        favg_accs.append(fa['final_accuracy'].mean() if len(fa) > 0 else np.nan)
        fagain_accs.append(fg['final_accuracy'].mean() if len(fg) > 0 else np.nan)

    favg_accs = np.array(favg_accs)
    fagain_accs = np.array(fagain_accs)
    valid = ~np.isnan(favg_accs) & ~np.isnan(fagain_accs)
    if valid.any():
        ax.fill_between(scenario_x[valid], favg_accs[valid], fagain_accs[valid],
                         color='#2ecc71', alpha=0.12)

    ax.set_title(dataset_titles[dataset], fontsize=15, fontweight='bold')
    ax.set_xticks(scenario_x)
    ax.set_xticklabels([s[0] for s in scenarios], fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=11)

# Set y-limits with enough padding for labels
axes[0, 0].set_ylim([0.76, 0.90])   # CIFAR-10
axes[0, 1].set_ylim([0.9915, 0.9950])  # MNIST

# --- Bottom row: Client Scaling ---
for col_idx, dataset in enumerate(datasets):
    ax = axes[1, col_idx]
    ds_scale = df_scaling[df_scaling['dataset'] == dataset]

    for strat in strategies:
        accs = []
        for nc in client_numbers:
            sub = ds_scale[(ds_scale['strategy'] == strat) & (ds_scale['num_clients'] == nc)]
            accs.append(sub['final_accuracy'].mean() if len(sub) > 0 else np.nan)

        lw = 2.8 if strat == 'FedAgain' else 1.8
        ms = 11 if strat == 'FedAgain' else 8
        zorder = 4 if strat == 'FedAgain' else 3
        ls = '-' if strat == 'FedAgain' else '--'

        ax.plot(client_numbers, accs, color=colors[strat], marker=markers[strat],
                markersize=ms, linewidth=lw, linestyle=ls, zorder=zorder,
                markeredgecolor='black', markeredgewidth=1.0)

    # Shade gap
    favg_accs = []
    fagain_accs = []
    for nc in client_numbers:
        fa = ds_scale[(ds_scale['strategy'] == 'FedAvg') & (ds_scale['num_clients'] == nc)]
        fg = ds_scale[(ds_scale['strategy'] == 'FedAgain') & (ds_scale['num_clients'] == nc)]
        favg_accs.append(fa['final_accuracy'].mean() if len(fa) > 0 else np.nan)
        fagain_accs.append(fg['final_accuracy'].mean() if len(fg) > 0 else np.nan)

    favg_accs = np.array(favg_accs)
    fagain_accs = np.array(fagain_accs)
    valid = ~np.isnan(favg_accs) & ~np.isnan(fagain_accs)
    if valid.any():
        ax.fill_between(np.array(client_numbers)[valid], favg_accs[valid], fagain_accs[valid],
                         color='#2ecc71', alpha=0.12)

    ax.set_xlabel('Number of Clients', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(client_numbers)
    ax.set_xticklabels([str(c) for c in client_numbers], fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=11)

# Set y-limits for scaling row
axes[1, 0].set_ylim([0.74, 0.87])   # CIFAR-10
axes[1, 1].set_ylim([0.9905, 0.9950])  # MNIST

# =============================================================================
# Shared legend
# =============================================================================

handles = [
    Line2D([0], [0], color=colors['FedAvg'], marker='s', markersize=8,
           linewidth=1.8, linestyle='--', markeredgecolor='black',
           markeredgewidth=1.0, label='FedAvg'),
    Line2D([0], [0], color=colors['FedAgain'], marker='o', markersize=11,
           linewidth=2.8, linestyle='-', markeredgecolor='black',
           markeredgewidth=1.0, label='FedAgain'),
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.03),
           ncol=2, fontsize=14, frameon=True, framealpha=0.9, edgecolor='black',
           columnspacing=3.0)

plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.10, hspace=0.20, wspace=0.30)
plt.savefig('cifar_mnist_simplified_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: cifar_mnist_simplified_analysis.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("CORRUPTION ESCALATION (10 clients, non-iid-label)")
print("=" * 80)
for label, ratio, sev in scenarios:
    print(f"\n{label.replace(chr(10), ' ')} (ratio={ratio}, severity={sev}):")
    for ds in datasets:
        print(f"  {ds}:")
        for strat in strategies:
            s = df_esc[(df_esc['dataset'] == ds) & (df_esc['strategy'] == strat) &
                       (df_esc['corrupt_ratio'] == ratio) & (df_esc['corruption_severity'] == sev)]
            if len(s) > 0:
                print(f"    {strat:12s}: {s['final_accuracy'].mean():.4f}")

print("\n" + "=" * 80)
print("CLIENT SCALING (ratio=0.3, severity=3, non-iid-label)")
print("=" * 80)
for nc in client_numbers:
    print(f"\n{nc} clients:")
    for ds in datasets:
        print(f"  {ds}:")
        for strat in strategies:
            s = df_scaling[(df_scaling['dataset'] == ds) & (df_scaling['strategy'] == strat) &
                           (df_scaling['num_clients'] == nc)]
            if len(s) > 0:
                print(f"    {strat:12s}: {s['final_accuracy'].mean():.4f}")
            else:
                print(f"    {strat:12s}: N/A")
