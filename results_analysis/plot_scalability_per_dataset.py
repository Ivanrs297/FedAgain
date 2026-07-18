import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.style.use('seaborn-v0_8-paper')

# Read data
df = pd.read_csv('../results/experiments.csv')
for col in ['final_accuracy', 'corrupt_ratio', 'corruption_severity', 'num_clients']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

kidney = ['DAUDON_MIX', 'JONATHAN_MIX', 'MYSTONE']
dataset_short = {
    'DAUDON_MIX': 'Dataset A',
    'JONATHAN_MIX': 'Dataset B',
    'MYSTONE': 'Dataset C'
}

# non-iid-label, ratio=0.3, severity=3
df_scale = df[
    (df['dataset'].isin(kidney)) &
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) &
    (df['corruption_severity'] == 3) &
    (df['data_distribution'] == 'non-iid-label') &
    (df['strategy'].isin(['FedAvg', 'FedAgain'])) &
    (df['num_clients'].isin([5, 10, 20]))
].copy()

# Deduplicate
df_scale = df_scale.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'num_clients'], keep='last'
)

client_sizes = [5, 10, 20]
colors = {'FedAvg': '#e74c3c', 'FedAgain': '#2ecc71'}
markers = {'FedAvg': 's', 'FedAgain': 'o'}

# =============================================================================
# Plot: one subplot per dataset, lines showing accuracy vs num_clients
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

for i, ds in enumerate(kidney):
    ax = axes[i]

    for strat in ['FedAvg', 'FedAgain']:
        accs = []
        for nc in client_sizes:
            sub = df_scale[(df_scale['strategy'] == strat) &
                           (df_scale['dataset'] == ds) &
                           (df_scale['num_clients'] == nc)]
            if len(sub) > 0:
                accs.append(sub['final_accuracy'].values[0])
            else:
                accs.append(np.nan)

        lw = 2.5 if strat == 'FedAgain' else 1.5
        ms = 9 if strat == 'FedAgain' else 7

        ax.plot(client_sizes, accs, color=colors[strat], marker=markers[strat],
                markersize=ms, linewidth=lw, markeredgecolor='black',
                markeredgewidth=1.0, zorder=4 if strat == 'FedAgain' else 3)

        # Value labels
        for nc, acc in zip(client_sizes, accs):
            if not np.isnan(acc):
                offset = 0.015 if strat == 'FedAgain' else -0.015
                ax.text(nc, acc + offset, f'{acc:.3f}',
                        ha='center', va='bottom' if strat == 'FedAgain' else 'top',
                        fontsize=7.5, fontweight='bold' if strat == 'FedAgain' else 'normal',
                        color=colors[strat])

    # Shade the gap between strategies
    favg_accs = []
    fagain_accs = []
    for nc in client_sizes:
        fa = df_scale[(df_scale['strategy'] == 'FedAvg') & (df_scale['dataset'] == ds) & (df_scale['num_clients'] == nc)]
        fg = df_scale[(df_scale['strategy'] == 'FedAgain') & (df_scale['dataset'] == ds) & (df_scale['num_clients'] == nc)]
        favg_accs.append(fa['final_accuracy'].values[0] if len(fa) > 0 else np.nan)
        fagain_accs.append(fg['final_accuracy'].values[0] if len(fg) > 0 else np.nan)

    favg_accs = np.array(favg_accs)
    fagain_accs = np.array(fagain_accs)
    valid = ~np.isnan(favg_accs) & ~np.isnan(fagain_accs)
    if valid.any():
        ax.fill_between(np.array(client_sizes)[valid],
                        favg_accs[valid], fagain_accs[valid],
                        color='#2ecc71', alpha=0.1)

    ax.set_title(dataset_short[ds], fontsize=14, fontweight='bold')
    ax.set_xticks(client_sizes)
    ax.set_xticklabels([str(c) for c in client_sizes], fontsize=11)
    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim([0.45, 0.90])

# Shared legend
handles = [Line2D([0], [0], color=colors[s], marker=markers[s],
                   markersize=9 if s == 'FedAgain' else 7,
                   linewidth=2.5 if s == 'FedAgain' else 1.5,
                   markeredgecolor='black', markeredgewidth=1.0, label=s)
           for s in ['FedAvg', 'FedAgain']]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=2, fontsize=13, frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('fedagain_scalability_per_dataset.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_scalability_per_dataset.png")
plt.close()

# Summary table
print("\nPer-dataset scalability (Accuracy):")
print(f"{'Dataset':>15s}  {'5 cli FedAvg':>12s}  {'5 cli FedAgain':>14s}  {'10 cli FedAvg':>13s}  {'10 cli FedAgain':>15s}  {'20 cli FedAvg':>13s}  {'20 cli FedAgain':>15s}")
print("-" * 100)
for ds in kidney:
    row = f"{ds:>15s}"
    for nc in client_sizes:
        for strat in ['FedAvg', 'FedAgain']:
            sub = df_scale[(df_scale['strategy'] == strat) & (df_scale['dataset'] == ds) & (df_scale['num_clients'] == nc)]
            if len(sub) > 0:
                row += f"  {sub['final_accuracy'].values[0]*100:>12.1f}%"
            else:
                row += f"  {'N/A':>12s}"
    print(row)
