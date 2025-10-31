# FedAgain Experimental Analysis: CIFAR10
**Analysis Date: 2025-10-31 03:03:19**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the CIFAR10 dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.8473
- FedAvg Average Accuracy: 0.8367
- Overall Improvement: +1.28%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedProx | 0.8689 | 0.0563 | 0.8310 | 0.9336 | 3 |
| FedMedian | 0.8549 | 0.0572 | 0.7932 | 0.9290 | 4 |
| FedAgain | 0.8473 | 0.0367 | 0.8000 | 0.9274 | 15 |
| Bulyan | 0.8403 | 0.0760 | 0.7410 | 0.9244 | 4 |
| FedAvg | 0.8367 | 0.0434 | 0.7691 | 0.9268 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.9336 |
| FedMedian | 0.9290 |
| FedAvg | 0.9268 |
| Bulyan | 0.9244 |
| FedAgain | 0.9194 |

**FedAgain Rank:** 5 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAvg | 0.8673 |
| FedMedian | 0.8632 |
| Bulyan | 0.8600 |
| FedAgain | 0.8480 |
| FedProx | 0.8420 |

**FedAgain vs FedAvg:** -2.23% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAgain | 0.8451 |
| Bulyan | 0.8357 |
| FedMedian | 0.8341 |
| FedAvg | 0.8312 |
| FedProx | 0.8310 |

**FedAgain vs FedAvg:** +1.67% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAgain | 0.8020 |
| FedMedian | 0.7932 |
| FedAvg | 0.7819 |
| Bulyan | 0.7410 |

**FedAgain vs FedAvg:** +2.57% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.8622 | 0.8372 | +2.99% |
| 2 | 0.8433 | 0.8118 | +3.88% |
| 3 | 0.8518 | 0.8359 | +1.91% |
| 4 | 0.8205 | 0.8214 | -0.10% |
| 5 | 0.8314 | 0.8309 | +0.06% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.9274 | 0.9103 | +1.88% |
| non-iid-label | 0.8346 | 0.8198 | +1.81% |
| non-iid-dirichlet | 0.8674 | 0.8661 | +0.15% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.8392 | 0.8408 | -0.19% |
| 10 | 0.8501 | 0.8364 | +1.64% |
| 20 | 0.8000 | 0.7691 | +4.02% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +2.57% improvement over FedAvg
- **30% Corruption:** +1.67% improvement over FedAvg

### Challenging Scenarios:
- **10% Corruption:** -2.23% vs FedAvg

---

## Recommendations

Based on the analysis of CIFAR10 dataset:

✅ **FedAgain shows overall improvement** (+1.28%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for CIFAR10 dataset*

*Timestamp: 2025-10-31 03:03:19.157321*