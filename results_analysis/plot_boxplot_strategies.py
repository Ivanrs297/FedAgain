import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-paper')

# Read data
df = pd.read_csv('../results/experiments.csv')
for col in ['final_accuracy', 'final_precision', 'final_recall', 'final_f1',
            'corrupt_ratio', 'corruption_severity', 'num_clients']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

kidney = ['DAUDON_MIX', 'DAUDON_SEC', 'DAUDON_SUR',
          'JONATHAN_MIX', 'JONATHAN_SEC', 'JONATHAN_SUR', 'MYSTONE']

# Filter data (ratio=0.3 for iid/dirichlet, ratio=0.1 for non-iid-label)
df_iid_dir = df[
    (df['dataset'].isin(kidney)) & (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) & (df['corruption_severity'] == 3) &
    (df['data_distribution'].isin(['iid', 'non-iid-dirichlet'])) &
    (df['strategy'].isin(['FedAvg', 'FedMedian', 'FedAgain']))
].copy()

df_noniid = df[
    (df['dataset'].isin(kidney)) & (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.1) & (df['corruption_severity'] == 3) &
    (df['data_distribution'] == 'non-iid-label') &
    (df['strategy'].isin(['FedAvg', 'FedMedian', 'FedAgain']))
].copy()

df_all = pd.concat([df_iid_dir, df_noniid])
df_all = df_all.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'data_distribution'], keep='last'
)

distributions = ['iid', 'non-iid-label', 'non-iid-dirichlet']
dist_labels = {'iid': 'IID', 'non-iid-label': 'Non-IID (Label)', 'non-iid-dirichlet': 'Non-IID (Dirichlet)'}
strategies = ['FedAvg', 'FedMedian', 'FedAgain']
colors = {'FedAvg': '#e74c3c', 'FedMedian': '#f39c12', 'FedAgain': '#2ecc71'}
metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# =============================================================================
# Box plot: 2x2 grid, one per metric
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

n_strat = len(strategies)
box_width = 0.6

for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    data_to_plot = []
    positions = []
    box_colors = []
    group_centers = []

    pos = 0
    for dist in distributions:
        group_start = pos
        for strat in strategies:
            sub = df_all[(df_all['strategy'] == strat) & (df_all['data_distribution'] == dist)]
            if len(sub) > 0:
                data_to_plot.append(sub[metric].values)
                positions.append(pos)
                box_colors.append(colors[strat])
            pos += 1
        group_centers.append((group_start + pos - 1) / 2)
        pos += 1.5  # gap between groups

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5),
    )

    # Color boxes and highlight FedAgain
    for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
        patch.set_facecolor(color)
        strat_idx = i % n_strat
        if strategies[strat_idx] == 'FedAgain':
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(2.5)
        else:
            patch.set_alpha(0.65)

    # X-axis labels at group centers
    ax.set_xticks(group_centers)
    ax.set_xticklabels([dist_labels[d] for d in distributions], fontsize=12)
    ax.set_ylabel(mlabel, fontsize=14, fontweight='bold')
    ax.set_ylim([0.45, 0.95])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='y', labelsize=12)

# Shared legend
legend_patches = [Patch(facecolor=colors[s], edgecolor='black' if s == 'FedAgain' else 'gray',
                        label=s, alpha=0.85 if s == 'FedAgain' else 0.65,
                        linewidth=2.5 if s == 'FedAgain' else 1.2)
                  for s in strategies]
fig.legend(handles=legend_patches,
           loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, fontsize=13,
           frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.savefig('fedagain_boxplot_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_boxplot_distributions.png")
plt.close()
