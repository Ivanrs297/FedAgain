# FedAgain Cross-Dataset Analysis Report

**Analysis Date:** 2025-10-31 02:52:02
**Datasets Analyzed:** 9
**Total Experiments:** 369

---

## Executive Summary

**Average Improvement:** +3.34%
**Best Performance:** JONATHAN_SEC (+8.44%)
**Worst Performance:** MNIST (-0.07%)
**Datasets Improved:** 8/9

---

## Datasets Overview

### CIFAR10
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +1.28%

### DAUDON_MIX
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +2.45%

### DAUDON_SEC
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +1.88%

### DAUDON_SUR
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +1.78%

### JONATHAN_MIX
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +8.08%

### JONATHAN_SEC
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +8.44%

### JONATHAN_SUR
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +4.39%

### MNIST
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: -0.07%

### MYSTONE
- Experiments: 41
- Strategies: FedAvg, FedProx, FedMedian, Bulyan, FedAgain
- FedAgain Improvement: +1.83%

---

# Comparison Tables


## Table 1: Overall Performance by Dataset

| Dataset | FedAgain | FedAvg | Improvement | Best Strategy | Best Acc | Winner |
|---|---|---|---|---|---|---|
| CIFAR10 | 0.8473±0.0367 | 0.8367±0.0434 | +1.28% | FedProx | 0.8689 |  |
| DAUDON_MIX | 0.8019±0.0350 | 0.7827±0.0712 | +2.45% | FedProx | 0.8214 |  |
| DAUDON_SEC | 0.7963±0.0539 | 0.7816±0.0505 | +1.88% | FedProx | 0.8364 |  |
| DAUDON_SUR | 0.7147±0.0348 | 0.7022±0.0462 | +1.78% | FedMedian | 0.7451 |  |
| JONATHAN_MIX | 0.7761±0.0310 | 0.7181±0.0674 | +8.08% | FedMedian | 0.8049 |  |
| JONATHAN_SEC | 0.7889±0.0467 | 0.7276±0.0882 | +8.44% | FedMedian | 0.8519 |  |
| JONATHAN_SUR | 0.7432±0.0519 | 0.7120±0.0835 | +4.39% | FedProx | 0.7522 |  |
| MNIST | 0.9926±0.0028 | 0.9932±0.0015 | -0.07% | FedMedian | 0.9940 |  |
| MYSTONE | 0.5583±0.0502 | 0.5482±0.0383 | +1.83% | FedProx | 0.5944 |  |


## Table 2: FedAgain Performance by Corruption Level

| Dataset | Corruption | FedAgain | FedAvg | Improvement | Status |
|---|---|---|---|---|---|
| CIFAR10 | 10% | 0.8480 | 0.8673 | -2.23% | NO |
| CIFAR10 | 30% | 0.8451 | 0.8312 | +1.67% | OK |
| CIFAR10 | 40% | 0.8020 | 0.7819 | +2.57% | OK |
| DAUDON_MIX | 10% | 0.8046 | 0.8367 | -3.83% | NO |
| DAUDON_MIX | 30% | 0.7938 | 0.7716 | +2.88% | OK |
| DAUDON_MIX | 40% | 0.8187 | 0.7533 | +8.68% | OK |
| DAUDON_SEC | 10% | 0.8042 | 0.7842 | +2.55% | OK |
| DAUDON_SEC | 30% | 0.7931 | 0.7802 | +1.66% | OK |
| DAUDON_SEC | 40% | 0.7508 | 0.7083 | +6.00% | OK |
| DAUDON_SUR | 10% | 0.7285 | 0.7302 | -0.22% | NO |
| DAUDON_SUR | 30% | 0.7165 | 0.7001 | +2.35% | OK |
| DAUDON_SUR | 40% | 0.6934 | 0.7056 | -1.74% | NO |
| JONATHAN_MIX | 10% | 0.7654 | 0.7725 | -0.92% | NO |
| JONATHAN_MIX | 30% | 0.7754 | 0.7034 | +10.23% | OK |
| JONATHAN_MIX | 40% | 0.7400 | 0.6700 | +10.45% | OK |
| JONATHAN_SEC | 10% | 0.8125 | 0.8592 | -5.43% | NO |
| JONATHAN_SEC | 30% | 0.7744 | 0.7056 | +9.76% | OK |
| JONATHAN_SEC | 40% | 0.8683 | 0.7200 | +20.60% | OK |
| JONATHAN_SUR | 10% | 0.7300 | 0.7408 | -1.46% | NO |
| JONATHAN_SUR | 30% | 0.7352 | 0.6962 | +5.61% | OK |
| JONATHAN_SUR | 40% | 0.7008 | 0.7117 | -1.52% | NO |
| MNIST | 10% | 0.9927 | 0.9934 | -0.07% | NO |
| MNIST | 30% | 0.9923 | 0.9930 | -0.08% | NO |
| MNIST | 40% | 0.9928 | 0.9923 | +0.05% | OK |
| MYSTONE | 10% | 0.6050 | 0.5483 | +10.33% | OK |
| MYSTONE | 30% | 0.5504 | 0.5456 | +0.88% | OK |
| MYSTONE | 40% | 0.5300 | 0.4958 | +6.89% | OK |


## Table 3: FedAgain Win Rate by Dataset

| Dataset | Wins | Total Scenarios | Win Rate | Grade |
|---|---|---|---|---|
| CIFAR10 | 5 | 9 | 55.6% | B |
| DAUDON_MIX | 4 | 9 | 44.4% | C |
| DAUDON_SEC | 4 | 9 | 44.4% | C |
| DAUDON_SUR | 5 | 9 | 55.6% | B |
| JONATHAN_MIX | 6 | 9 | 66.7% | B |
| JONATHAN_SEC | 6 | 9 | 66.7% | B |
| JONATHAN_SUR | 5 | 10 | 50.0% | B |
| MNIST | 2 | 10 | 20.0% | D |
| MYSTONE | 2 | 9 | 22.2% | D |


## Table 4: Statistical Summary Across All Datasets

| Metric | Value |
|---|---|
| Average Improvement | +3.34% |
| Best Improvement | +8.44% |
| Worst Improvement | -0.07% |
| Std Deviation | 2.85% |
| Datasets Improved | 8/9 |

---

## Generated Visualizations

1. **cross_dataset_comparison.png** - Bar chart comparing FedAgain vs FedAvg
2. **improvement_heatmap.png** - Heatmap of improvements across corruption levels
3. **win_rate_analysis.png** - Win rate by dataset
4. **severity_comparison.png** - Performance across severity levels
5. **improvement_distribution.png** - Distribution of improvements

---

## Key Findings

1. **Best Corruption Level:** 40% corruption (avg improvement: +5.78%)
2. **Most Consistent Dataset:** JONATHAN_SEC
3. **Most Challenging Dataset:** MNIST
4. **Success Rate:** 8/9 datasets show improvement

---

## Recommendations

[MODEST] **FedAgain shows modest improvement** with an average 3.34% gain.

**Dataset-Specific Recommendations:**
- **JONATHAN_SEC:** Strongly recommended (improvement: +8.44%)
- **JONATHAN_MIX:** Strongly recommended (improvement: +8.08%)
- **JONATHAN_SUR:** Consider using (improvement: +4.39%)
- **DAUDON_MIX:** Consider using (improvement: +2.45%)
- **DAUDON_SEC:** Consider using (improvement: +1.88%)
- **MYSTONE:** Consider using (improvement: +1.83%)
- **DAUDON_SUR:** Consider using (improvement: +1.78%)
- **CIFAR10:** Consider using (improvement: +1.28%)
- **MNIST:** Use alternative strategies (decline: -0.07%)

---


*Report generated: 2025-10-31 02:52:02.110211*