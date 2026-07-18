import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-paper')

# Read data
df = pd.read_csv('../results/experiments.csv')
for col in ['final_accuracy', 'final_precision', 'final_recall', 'final_f1',
            'corrupt_ratio', 'corruption_severity', 'num_clients']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

kidney = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
          'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 'MYSTONE']

# non-iid-label, ratio=0.3, severity=3 — the setting with 5, 10, 20 clients
df_scale = df[
    (df['dataset'].isin(kidney)) &
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) &
    (df['corruption_severity'] == 3) &
    (df['data_distribution'] == 'non-iid-label') &
    (df['strategy'].isin(['FedAvg', 'FedAgain'])) &
    (df['num_clients'].isin([5, 10, 20]))
].copy()

# Deduplicate: keep latest per (strategy, dataset, num_clients)
df_scale = df_scale.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'num_clients'], keep='last'
)

client_sizes = [5, 10, 20]
strategies = ['FedAvg', 'FedAgain']
colors = {'FedAvg': '#e74c3c', 'FedAgain': '#2ecc71'}
markers = {'FedAvg': 's', 'FedAgain': 'o'}
metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# =============================================================================
# Plot: 2x2 line plot with error bands — scalability across client sizes
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    for strat in strategies:
        means = []
        stds = []
        for nc in client_sizes:
            sub = df_scale[(df_scale['strategy'] == strat) & (df_scale['num_clients'] == nc)]
            if len(sub) > 0:
                means.append(sub[metric].mean())
                stds.append(sub[metric].std())
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        lw = 2.5 if strat == 'FedAgain' else 1.8
        ms = 10 if strat == 'FedAgain' else 8
        zorder = 4 if strat == 'FedAgain' else 3

        ax.plot(client_sizes, means, color=colors[strat], marker=markers[strat],
                markersize=ms, linewidth=lw, label=strat, zorder=zorder,
                markeredgecolor='black', markeredgewidth=1.2)
        ax.fill_between(client_sizes, means - stds, means + stds,
                        color=colors[strat], alpha=0.15, zorder=2)

        # Value labels
        for nc, m in zip(client_sizes, means):
            if not np.isnan(m):
                offset = 0.012 if strat == 'FedAgain' else -0.018
                ax.text(nc, m + offset, f'{m:.3f}', ha='center',
                        va='bottom' if strat == 'FedAgain' else 'top',
                        fontsize=9, fontweight='bold' if strat == 'FedAgain' else 'normal',
                        color=colors[strat])

    # Compute and annotate improvement gap at each client size
    for nc in client_sizes:
        favg = df_scale[(df_scale['strategy'] == 'FedAvg') & (df_scale['num_clients'] == nc)]
        fagain = df_scale[(df_scale['strategy'] == 'FedAgain') & (df_scale['num_clients'] == nc)]
        if len(favg) > 0 and len(fagain) > 0:
            gap = (fagain[metric].mean() - favg[metric].mean()) / favg[metric].mean() * 100
            mid = (favg[metric].mean() + fagain[metric].mean()) / 2
            ax.annotate(f'+{gap:.1f}%', xy=(nc + 0.5, mid), fontsize=8,
                        fontweight='bold', color='#1a7a3a', ha='left', va='center')

    ax.set_xticks(client_sizes)
    ax.set_xticklabels([str(c) for c in client_sizes], fontsize=12)
    ax.set_xlabel('Number of Clients', fontsize=12)
    ax.set_ylabel(mlabel, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=11)

    # Consistent y-axis
    all_vals = df_scale[metric].dropna()
    ymin = max(all_vals.min() - 0.08, 0.45)
    ymax = min(all_vals.max() + 0.08, 1.0)
    ax.set_ylim([ymin, ymax])

# Shared legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=colors[s], marker=markers[s], markersize=10 if s == 'FedAgain' else 8,
                   linewidth=2.5 if s == 'FedAgain' else 1.8, markeredgecolor='black',
                   markeredgewidth=1.2, label=s)
           for s in strategies]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=13, frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.savefig('fedagain_scalability_clients.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_scalability_clients.png")
plt.close()

# Summary
print("\nScalability Summary (Non-IID Label, ratio=0.3, severity=3):")
print(f"{'Clients':>8s}  {'FedAvg Acc':>12s}  {'FedAgain Acc':>13s}  {'Gap':>8s}  {'FedAvg F1':>11s}  {'FedAgain F1':>12s}  {'Gap':>8s}")
print("-" * 80)
for nc in client_sizes:
    favg = df_scale[(df_scale['strategy'] == 'FedAvg') & (df_scale['num_clients'] == nc)]
    fagain = df_scale[(df_scale['strategy'] == 'FedAgain') & (df_scale['num_clients'] == nc)]
    if len(favg) > 0 and len(fagain) > 0:
        a1, a2 = favg['final_accuracy'].mean(), fagain['final_accuracy'].mean()
        f1, f2 = favg['final_f1'].mean(), fagain['final_f1'].mean()
        acc_gap = (a2 - a1) / a1 * 100
        f1_gap = (f2 - f1) / f1 * 100
        print(f"{nc:>8d}  {a1*100:>11.2f}%  {a2*100:>12.2f}%  {acc_gap:>+7.1f}%  {f1*100:>10.2f}%  {f2*100:>11.2f}%  {f1_gap:>+7.1f}%")
