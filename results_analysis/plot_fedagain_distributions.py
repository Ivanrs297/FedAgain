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

# Filter: ratio=0.3, severity=3, kidney datasets
df_k = df[
    (df['dataset'].isin(kidney)) &
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) &
    (df['corruption_severity'] == 3) &
    (df['strategy'].isin(['FedAvg', 'FedMedian', 'FedAgain']))
].copy()

# Deduplicate: keep latest per (strategy, dataset, distribution)
df_k = df_k.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'data_distribution'], keep='last'
)

distributions = ['iid', 'non-iid-label', 'non-iid-dirichlet']
dist_labels = {'iid': 'IID', 'non-iid-label': 'Non-IID\n(Label)', 'non-iid-dirichlet': 'Non-IID\n(Dirichlet)'}
strategies = ['FedAvg', 'FedMedian', 'FedAgain']
colors = {'FedAvg': '#e74c3c', 'FedMedian': '#f39c12', 'FedAgain': '#2ecc71'}

# =============================================================================
# Plot: Grouped bar chart — mean accuracy with std error bars per distribution
# FedAgain highlighted with hatching and bold edge
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

n_dist = len(distributions)
n_strat = len(strategies)
bar_width = 0.22
x = np.arange(n_dist)

for i, strat in enumerate(strategies):
    means = []
    stds = []
    has_data = []

    for dist in distributions:
        sub = df_k[(df_k['strategy'] == strat) & (df_k['data_distribution'] == dist)]
        if len(sub) > 0:
            means.append(sub['final_accuracy'].mean())
            stds.append(sub['final_accuracy'].std())
            has_data.append(True)
        else:
            means.append(0)
            stds.append(0)
            has_data.append(False)

    offset = (i - (n_strat - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset, means, bar_width,
        yerr=stds, capsize=4,
        color=colors[strat],
        edgecolor='black' if strat == 'FedAgain' else 'gray',
        linewidth=2.0 if strat == 'FedAgain' else 1.0,
        hatch='//' if strat == 'FedAgain' else None,
        alpha=0.85,
        error_kw=dict(lw=1.5, capthick=1.5),
        zorder=3 if strat == 'FedAgain' else 2,
    )

    # Hide bars with no data
    for bar, valid in zip(bars, has_data):
        if not valid:
            bar.set_alpha(0)
            bar.set_edgecolor('none')
            bar.set_height(0)

    # Add value labels on top of bars
    for bar, m, valid in zip(bars, means, has_data):
        if valid:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold' if strat == 'FedAgain' else 'normal')

# Add improvement annotations for FedAgain vs FedAvg
for j, dist in enumerate(distributions):
    fedavg_sub = df_k[(df_k['strategy'] == 'FedAvg') & (df_k['data_distribution'] == dist)]
    fedagain_sub = df_k[(df_k['strategy'] == 'FedAgain') & (df_k['data_distribution'] == dist)]
    if len(fedavg_sub) > 0 and len(fedagain_sub) > 0:
        avg_mean = fedavg_sub['final_accuracy'].mean()
        again_mean = fedagain_sub['final_accuracy'].mean()
        imp = ((again_mean - avg_mean) / avg_mean) * 100
        # Place annotation above FedAgain bar
        offset_again = ((n_strat - 1) / 2) * bar_width  # FedAgain is last
        # Offset to center between FedAvg and FedAgain
        ax.annotate(
            f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
            xy=(j + offset_again, again_mean + 0.025),
            fontsize=10, fontweight='bold', color='#1a7a3a',
            ha='center', va='bottom',
        )

ax.set_xticks(x)
ax.set_xticklabels([dist_labels[d] for d in distributions], fontsize=13)
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim([0.55, 0.90])
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
ax.tick_params(axis='y', labelsize=12)

# Legend
legend_patches = [Patch(facecolor=colors[s], edgecolor='black' if s == 'FedAgain' else 'gray',
                        label=s, alpha=0.85, hatch='//' if s == 'FedAgain' else None)
                  for s in strategies]
ax.legend(handles=legend_patches, fontsize=12, loc='upper right',
          frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('fedagain_distribution_performance.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_distribution_performance.png")
plt.close()

# Print summary
print("\nSummary (mean accuracy across 7 kidney datasets):")
print(f"{'':20s} {'FedAvg':>10s} {'FedMedian':>10s} {'FedAgain':>10s} {'vs FedAvg':>10s}")
print("-" * 65)
for dist in distributions:
    row = f"{dist_labels[dist].replace(chr(10),' '):20s}"
    vals = {}
    for s in strategies:
        sub = df_k[(df_k['strategy'] == s) & (df_k['data_distribution'] == dist)]
        if len(sub) > 0:
            vals[s] = sub['final_accuracy'].mean()
            row += f" {vals[s]:10.4f}"
        else:
            row += f" {'N/A':>10s}"
    if 'FedAvg' in vals and 'FedAgain' in vals:
        imp = ((vals['FedAgain'] - vals['FedAvg']) / vals['FedAvg']) * 100
        row += f" {imp:>+9.1f}%"
    print(row)
