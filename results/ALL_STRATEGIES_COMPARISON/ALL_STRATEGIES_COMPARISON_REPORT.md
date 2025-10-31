# Comprehensive Strategy Comparison Report

**Analysis Date:** 2025-10-31 02:55:45
**Datasets Analyzed:** 9
**Strategies Compared:** Bulyan, FedAgain, FedAvg, FedMedian, FedProx
**Total Experiments:** 369

---

## Executive Summary

**Best Overall Strategy:** FedMedian (0.7994 mean accuracy)
**FedAgain Rank:** 3 of 5
**FedAgain Mean Accuracy:** 0.7799

---

# Detailed Comparison Tables


## Table 1: Overall Performance - All Strategies by Dataset

Mean accuracy ± standard deviation

| Dataset | Bulyan | FedAgain | FedAvg | FedMedian | FedProx | Best |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.8403±0.0760 | 0.8473±0.0367 | 0.8367±0.0434 | 0.8549±0.0572 | 0.8689±0.0563 | FedProx |
| DAUDON_MIX | 0.8145±0.0388 | 0.8019±0.0350 | 0.7827±0.0712 | 0.8169±0.0350 | 0.8214±0.0381 | FedProx |
| DAUDON_SEC | 0.7996±0.0642 | 0.7963±0.0539 | 0.7816±0.0505 | 0.8138±0.0469 | 0.8364±0.0600 | FedProx |
| DAUDON_SUR | 0.7089±0.0323 | 0.7147±0.0348 | 0.7022±0.0462 | 0.7451±0.0186 | 0.7381±0.0153 | FedMedian |
| JONATHAN_MIX | 0.7301±0.0598 | 0.7761±0.0310 | 0.7181±0.0674 | 0.8049±0.0202 | 0.7769±0.0739 | FedMedian |
| JONATHAN_SEC | 0.7852±0.1165 | 0.7889±0.0467 | 0.7276±0.0882 | 0.8519±0.0629 | 0.8014±0.0952 | FedMedian |
| JONATHAN_SUR | 0.7344±0.1051 | 0.7432±0.0519 | 0.7120±0.0835 | 0.7500±0.0547 | 0.7522±0.0874 | FedProx |
| MNIST | 0.9928±0.0033 | 0.9926±0.0028 | 0.9932±0.0015 | 0.9940±0.0011 | 0.9939±0.0028 | FedMedian |
| MYSTONE | 0.5756±0.1075 | 0.5583±0.0502 | 0.5482±0.0383 | 0.5629±0.0894 | 0.5944±0.0631 | FedProx |


## Table 2: FedAgain Head-to-Head Comparison

Improvement of FedAgain over each strategy (percentage)

| Dataset | FedAgain Acc | vs Bulyan | vs FedAvg | vs FedMedian | vs FedProx |
|---|---|---|---|---|---|
| CIFAR10 | 0.8473 | +0.84% | +1.28% | -0.88% | -2.48% |
| DAUDON_MIX | 0.8019 | -1.54% | +2.45% | -1.83% | -2.37% |
| DAUDON_SEC | 0.7963 | -0.41% | +1.88% | -2.15% | -4.80% |
| DAUDON_SUR | 0.7147 | +0.82% | +1.78% | -4.08% | -3.17% |
| JONATHAN_MIX | 0.7761 | +6.30% | +8.08% | -3.58% | -0.11% |
| JONATHAN_SEC | 0.7889 | +0.48% | +8.44% | -7.39% | -1.55% |
| JONATHAN_SUR | 0.7432 | +1.20% | +4.39% | -0.90% | -1.20% |
| MNIST | 0.9926 | -0.02% | -0.07% | -0.14% | -0.13% |
| MYSTONE | 0.5583 | -3.01% | +1.83% | -0.82% | -6.08% |


## Table 3: Strategy Rankings by Dataset

Rank based on mean accuracy (1=best)

| Dataset | FedProx | FedMedian | FedAgain | Bulyan | FedAvg |
|---|---|---|---|---|---|
| CIFAR10 | 1 | 2 | 3 | 4 | 5 |
| DAUDON_MIX | 1 | 2 | 4 | 3 | 5 |
| DAUDON_SEC | 1 | 2 | 4 | 3 | 5 |
| DAUDON_SUR | 2 | 1 | 3 | 4 | 5 |
| JONATHAN_MIX | 2 | 1 | 3 | 4 | 5 |
| JONATHAN_SEC | 2 | 1 | 3 | 4 | 5 |
| JONATHAN_SUR | 1 | 2 | 3 | 4 | 5 |
| MNIST | 2 | 1 | 5 | 4 | 3 |
| MYSTONE | 1 | 3 | 4 | 2 | 5 |


## Table 4: Performance by Corruption Level - All Strategies


### 10% Corruption

| Dataset | Bulyan | FedAgain | FedAvg | FedMedian | FedProx | Best |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.8600 | 0.8480 | 0.8673 | 0.8632 | 0.8420 | FedAvg |
| DAUDON_MIX | 0.8279 | 0.8046 | 0.8367 | 0.8496 | 0.8229 | FedMedian |
| DAUDON_SEC | 0.8125 | 0.8042 | 0.7842 | 0.7750 | 0.8150 | FedProx |
| DAUDON_SUR | 0.7563 | 0.7285 | 0.7302 | 0.7334 | 0.7555 | Bulyan |
| JONATHAN_MIX | 0.7175 | 0.7654 | 0.7725 | 0.8183 | 0.7825 | FedMedian |
| JONATHAN_SEC | 0.8500 | 0.8125 | 0.8592 | 0.8058 | 0.7700 | FedAvg |
| JONATHAN_SUR | 0.7567 | 0.7300 | 0.7408 | 0.7442 | 0.6817 | Bulyan |
| MNIST | 0.9934 | 0.9927 | 0.9934 | 0.9935 | 0.9943 | FedProx |
| MYSTONE | 0.5983 | 0.6050 | 0.5483 | 0.4525 | 0.6342 | FedProx |

### 30% Corruption

| Dataset | Bulyan | FedAgain | FedAvg | FedMedian | FedProx | Best |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.8357 | 0.8451 | 0.8312 | 0.8341 | 0.8310 | FedAgain |
| DAUDON_MIX | 0.8154 | 0.7938 | 0.7716 | 0.7942 | 0.7825 | Bulyan |
| DAUDON_SEC | 0.7567 | 0.7931 | 0.7802 | 0.7942 | 0.7900 | FedMedian |
| DAUDON_SUR | 0.7007 | 0.7165 | 0.7001 | 0.7588 | 0.7269 | FedMedian |
| JONATHAN_MIX | 0.7821 | 0.7754 | 0.7034 | 0.7958 | 0.7004 | FedMedian |
| JONATHAN_SEC | 0.7200 | 0.7744 | 0.7056 | 0.8133 | 0.7258 | FedMedian |
| JONATHAN_SUR | 0.7383 | 0.7352 | 0.6962 | 0.7133 | 0.7250 | Bulyan |
| MNIST | 0.9931 | 0.9923 | 0.9930 | 0.9932 | 0.9909 | FedMedian |
| MYSTONE | 0.6417 | 0.5504 | 0.5456 | 0.5567 | 0.5217 | Bulyan |

### 40% Corruption

| Dataset | Bulyan | FedAgain | FedAvg | FedMedian | FedProx | Best |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.7410 | 0.8020 | 0.7819 | 0.7932 | -- | FedAgain |
| DAUDON_MIX | 0.7612 | 0.8187 | 0.7533 | 0.7800 | -- | FedAgain |
| DAUDON_SEC | 0.7442 | 0.7508 | 0.7083 | 0.8042 | -- | FedMedian |
| DAUDON_SUR | 0.6942 | 0.6934 | 0.7056 | 0.7629 | -- | FedMedian |
| JONATHAN_MIX | 0.6508 | 0.7400 | 0.6700 | 0.7808 | -- | FedMedian |
| JONATHAN_SEC | 0.6583 | 0.8683 | 0.7200 | 0.8458 | -- | FedAgain |
| JONATHAN_SUR | 0.5942 | 0.7008 | 0.7117 | 0.7133 | -- | FedMedian |
| MNIST | 0.9883 | 0.9928 | 0.9923 | 0.9936 | -- | FedMedian |
| MYSTONE | 0.4175 | 0.5300 | 0.4958 | 0.5717 | -- | FedMedian |


## Table 5: Win Rate Analysis - All Strategies

How often each strategy achieves best performance

| Dataset | Total Scenarios | Bulyan | FedAgain | FedAvg | FedMedian | FedProx |
|---|---|---|---|---|---|---|
| CIFAR10 | 9 | 1 (11.1%) | 5 (55.6%) | 2 (22.2%) | 0 (0.0%) | 1 (11.1%) |
| DAUDON_MIX | 9 | 1 (11.1%) | 4 (44.4%) | 3 (33.3%) | 1 (11.1%) | 0 (0.0%) |
| DAUDON_SEC | 9 | 0 (0.0%) | 4 (44.4%) | 2 (22.2%) | 1 (11.1%) | 2 (22.2%) |
| DAUDON_SUR | 9 | 1 (11.1%) | 5 (55.6%) | 1 (11.1%) | 2 (22.2%) | 0 (0.0%) |
| JONATHAN_MIX | 9 | 0 (0.0%) | 6 (66.7%) | 1 (11.1%) | 2 (22.2%) | 0 (0.0%) |
| JONATHAN_SEC | 9 | 0 (0.0%) | 6 (66.7%) | 2 (22.2%) | 1 (11.1%) | 0 (0.0%) |
| JONATHAN_SUR | 10 | 1 (10.0%) | 5 (50.0%) | 3 (30.0%) | 1 (10.0%) | 0 (0.0%) |
| MNIST | 10 | 0 (0.0%) | 2 (20.0%) | 5 (50.0%) | 1 (10.0%) | 2 (20.0%) |
| MYSTONE | 9 | 1 (11.1%) | 2 (22.2%) | 3 (33.3%) | 2 (22.2%) | 1 (11.1%) |


## Table 6: Cross-Dataset Statistical Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Datasets |
|---|---|---|---|---|---|
| FedMedian | 0.7994 | 0.1083 | 0.5629 | 0.9940 | 9 |
| FedProx | 0.7982 | 0.1017 | 0.5944 | 0.9939 | 9 |
| FedAgain | 0.7799 | 0.1082 | 0.5583 | 0.9926 | 9 |
| Bulyan | 0.7757 | 0.1063 | 0.5756 | 0.9928 | 9 |
| FedAvg | 0.7558 | 0.1127 | 0.5482 | 0.9932 | 9 |

---

## Generated Visualizations

1. **all_strategies_comparison.png** - Overall comparison across datasets
2. **fedagain_vs_all_heatmap.png** - FedAgain improvement heatmap
3. **strategy_rankings.png** - Average rankings
4. **corruption_robustness_all_strategies.png** - Performance at corruption levels
5. **win_rate_all_strategies.png** - Win rate comparison
6. **strategy_consistency.png** - Consistency analysis

---

## Key Findings

### Win Rates Across All Datasets

- **Bulyan:** 5/83 scenarios (6.0%)
- **FedAgain:** 39/83 scenarios (47.0%)
- **FedAvg:** 22/83 scenarios (26.5%)
- **FedMedian:** 11/83 scenarios (13.3%)
- **FedProx:** 6/83 scenarios (7.2%)

---


*Report generated: 2025-10-31 02:55:45.174168*