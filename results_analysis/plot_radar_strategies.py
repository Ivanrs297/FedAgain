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
# Radar chart: one subplot per distribution
# =============================================================================

num_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), subplot_kw=dict(polar=True))

for ax, dist in zip(axes, distributions):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw grid and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0.55, 0.90)
    ax.set_yticks([0.60, 0.70, 0.80, 0.90])
    ax.set_yticklabels(['0.60', '0.70', '0.80', '0.90'], fontsize=8, color='gray')
    ax.set_rlabel_position(30)

    for strat in strategies:
        sub = df_all[(df_all['strategy'] == strat) & (df_all['data_distribution'] == dist)]
        if len(sub) == 0:
            continue

        values = [sub[m].mean() for m in metrics]
        values += values[:1]  # close polygon

        linewidth = 2.5 if strat == 'FedAgain' else 1.5
        linestyle = '-' if strat == 'FedAgain' else '--'

        ax.plot(angles, values, color=colors[strat], linewidth=linewidth,
                linestyle=linestyle, label=strat, zorder=3 if strat == 'FedAgain' else 2)
        ax.fill(angles, values, color=colors[strat],
                alpha=0.15 if strat == 'FedAgain' else 0.05)

        # Add value labels for FedAgain
        if strat == 'FedAgain':
            for angle, val in zip(angles[:-1], values[:-1]):
                ax.text(angle, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=colors[strat])

    ax.set_title(dist_labels[dist], fontsize=14, fontweight='bold', pad=20)

# Shared legend
handles = [plt.Line2D([0], [0], color=colors[s], linewidth=2.5 if s == 'FedAgain' else 1.5,
                       linestyle='-' if s == 'FedAgain' else '--', label=s)
           for s in strategies]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=13, frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.savefig('fedagain_radar_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_radar_distributions.png")
plt.close()
