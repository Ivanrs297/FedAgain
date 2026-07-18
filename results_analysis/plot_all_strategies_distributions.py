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

# For iid and non-iid-dirichlet: ratio=0.3, severity=3
df_iid_dir = df[
    (df['dataset'].isin(kidney)) &
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.3) &
    (df['corruption_severity'] == 3) &
    (df['data_distribution'].isin(['iid', 'non-iid-dirichlet'])) &
    (df['strategy'].isin(['FedAvg', 'FedMedian', 'FedAgain']))
].copy()

# For non-iid-label: ratio=0.1, severity=3 (all 3 strategies available)
df_noniid = df[
    (df['dataset'].isin(kidney)) &
    (df['corrupt_clients'] == True) &
    (df['corrupt_ratio'] == 0.1) &
    (df['corruption_severity'] == 3) &
    (df['data_distribution'] == 'non-iid-label') &
    (df['strategy'].isin(['FedAvg', 'FedMedian', 'FedAgain']))
].copy()

df_all = pd.concat([df_iid_dir, df_noniid])

# Deduplicate: keep latest per (strategy, dataset, distribution)
df_all = df_all.sort_values('timestamp').drop_duplicates(
    subset=['strategy', 'dataset', 'data_distribution'], keep='last'
)

distributions = ['iid', 'non-iid-label', 'non-iid-dirichlet']
dist_labels = {
    'iid': 'IID',
    'non-iid-label': 'Non-IID (Label)',
    'non-iid-dirichlet': 'Non-IID (Dirichlet)'
}
strategies = ['FedAvg', 'FedMedian', 'FedAgain']
colors = {'FedAvg': '#e74c3c', 'FedMedian': '#f39c12', 'FedAgain': '#2ecc71'}
hatches = {'FedAvg': None, 'FedMedian': None, 'FedAgain': '//'}

metrics = ['final_accuracy', 'final_recall', 'final_precision', 'final_f1']
metric_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

# =============================================================================
# Plot: 2x2 grouped bar chart — all 4 metrics, 3 strategies, 3 distributions
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

n_dist = len(distributions)
n_strat = len(strategies)
bar_width = 0.24
x = np.arange(n_dist)

for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]

    for i, strat in enumerate(strategies):
        means = []
        stds = []
        has_data = []

        for dist in distributions:
            sub = df_all[(df_all['strategy'] == strat) & (df_all['data_distribution'] == dist)]
            if len(sub) > 0:
                means.append(sub[metric].mean())
                stds.append(sub[metric].std())
                has_data.append(True)
            else:
                means.append(0)
                stds.append(0)
                has_data.append(False)

        offset = (i - (n_strat - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            color=colors[strat],
            edgecolor='black' if strat == 'FedAgain' else 'gray',
            linewidth=2.0 if strat == 'FedAgain' else 1.0,
            hatch=hatches[strat],
            alpha=0.85,
            error_kw=dict(lw=1.2, capthick=1.2),
            zorder=3 if strat == 'FedAgain' else 2,
        )

        # Hide bars with no data
        for bar, valid in zip(bars, has_data):
            if not valid:
                bar.set_alpha(0)
                bar.set_edgecolor('none')
                bar.set_height(0)

    # Add FedAgain improvement annotations vs FedAvg
    for j, dist in enumerate(distributions):
        fedavg_sub = df_all[(df_all['strategy'] == 'FedAvg') & (df_all['data_distribution'] == dist)]
        fedagain_sub = df_all[(df_all['strategy'] == 'FedAgain') & (df_all['data_distribution'] == dist)]
        if len(fedavg_sub) > 0 and len(fedagain_sub) > 0:
            avg_val = fedavg_sub[metric].mean()
            again_val = fedagain_sub[metric].mean()
            imp = ((again_val - avg_val) / avg_val) * 100
            # Place above the tallest bar in the group
            all_means = []
            for s in strategies:
                ss = df_all[(df_all['strategy'] == s) & (df_all['data_distribution'] == dist)]
                if len(ss) > 0:
                    all_means.append(ss[metric].mean() + ss[metric].std())
            top = max(all_means) if all_means else again_val
            ax.annotate(
                f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
                xy=(j, top + 0.02),
                fontsize=9, fontweight='bold',
                color='#1a7a3a' if imp > 0 else '#c0392b',
                ha='center', va='bottom',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels[d] for d in distributions], fontsize=11)
    ax.set_ylabel(mlabel, fontsize=13, fontweight='bold')
    ax.set_ylim([0.50, 0.95])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.tick_params(axis='y', labelsize=11)

# Shared legend
legend_patches = [Patch(facecolor=colors[s], edgecolor='black' if s == 'FedAgain' else 'gray',
                        label=s, alpha=0.85, hatch=hatches[s])
                  for s in strategies]
fig.legend(handles=legend_patches,
           loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, fontsize=13,
           frameon=True, framealpha=0.9, edgecolor='black')

plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.savefig('fedagain_all_strategies_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: fedagain_all_strategies_distributions.png")
plt.close()

# =============================================================================
# Summary table
# =============================================================================
print("\nComparison (mean ± std across 7 kidney datasets):")
print(f"{'Metric':<12} {'Distribution':<22} {'FedAvg':>12} {'FedMedian':>12} {'FedAgain':>12} {'FedAgain':>12}")
print(f"{'':12} {'':22} {'':12} {'':12} {'':12} {'vs FedAvg':>12}")
print("-" * 85)
for metric, mlabel in zip(metrics, metric_labels):
    for dist in distributions:
        row = f"{mlabel:<12} {dist_labels[dist]:<22}"
        vals = {}
        for s in strategies:
            sub = df_all[(df_all['strategy'] == s) & (df_all['data_distribution'] == dist)]
            if len(sub) > 0:
                m = sub[metric].mean()
                vals[s] = m
                row += f" {m:>11.4f}"
            else:
                row += f" {'N/A':>12}"
        if 'FedAvg' in vals and 'FedAgain' in vals:
            imp = ((vals['FedAgain'] - vals['FedAvg']) / vals['FedAvg']) * 100
            row += f" {imp:>+11.1f}%"
        print(row)
    print()

print("Note: Non-IID (Label) uses ratio=0.1; IID and Non-IID (Dirichlet) use ratio=0.3.")
print("      All experiments use severity=3, corruption enabled.")
