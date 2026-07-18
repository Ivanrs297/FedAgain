# FedAgain Experimental Analysis: DAUDON_SUR
**Analysis Date: 2025-10-31 03:03:27**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the DAUDON_SUR dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.7147
- FedAvg Average Accuracy: 0.7022
- Overall Improvement: +1.78%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedMedian | 0.7451 | 0.0186 | 0.7253 | 0.7629 | 4 |
| FedProx | 0.7381 | 0.0153 | 0.7269 | 0.7555 | 3 |
| FedAgain | 0.7147 | 0.0348 | 0.6500 | 0.7784 | 15 |
| Bulyan | 0.7089 | 0.0323 | 0.6844 | 0.7563 | 4 |
| FedAvg | 0.7022 | 0.0462 | 0.5511 | 0.7490 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.7318 |
| FedMedian | 0.7253 |
| FedAgain | 0.7007 |
| FedAvg | 0.6966 |
| Bulyan | 0.6844 |

**FedAgain Rank:** 3 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| Bulyan | 0.7563 |
| FedProx | 0.7555 |
| FedMedian | 0.7334 |
| FedAvg | 0.7302 |
| FedAgain | 0.7285 |

**FedAgain vs FedAvg:** -0.22% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7588 |
| FedProx | 0.7269 |
| FedAgain | 0.7165 |
| Bulyan | 0.7007 |
| FedAvg | 0.7001 |

**FedAgain vs FedAvg:** +2.35% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7629 |
| FedAvg | 0.7056 |
| Bulyan | 0.6942 |
| FedAgain | 0.6934 |

**FedAgain vs FedAvg:** -1.74% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.7253 | 0.6966 | +4.11% |
| 2 | 0.7473 | 0.7326 | +2.01% |
| 3 | 0.7137 | 0.6914 | +3.23% |
| 4 | 0.7244 | 0.7150 | +1.32% |
| 5 | 0.6803 | 0.7016 | -3.03% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.7400 | 0.7212 | +2.61% |
| non-iid-label | 0.7200 | 0.7128 | +1.00% |
| non-iid-dirichlet | 0.6582 | 0.5511 | +19.44% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.7048 | 0.7040 | +0.12% |
| 10 | 0.7141 | 0.6948 | +2.79% |
| 20 | 0.7514 | 0.7490 | +0.33% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **30% Corruption:** +2.35% improvement over FedAvg

### Challenging Scenarios:
- **40% Corruption:** -1.74% vs FedAvg
- **10% Corruption:** -0.22% vs FedAvg

---

## Recommendations

Based on the analysis of DAUDON_SUR dataset:

✅ **FedAgain shows overall improvement** (+1.78%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for DAUDON_SUR dataset*

*Timestamp: 2025-10-31 03:03:27.685287*