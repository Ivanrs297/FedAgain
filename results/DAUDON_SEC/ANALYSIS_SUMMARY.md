# FedAgain Experimental Analysis: DAUDON_SEC
**Analysis Date: 2025-10-31 03:03:25**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the DAUDON_SEC dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.7963
- FedAvg Average Accuracy: 0.7816
- Overall Improvement: +1.88%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedProx | 0.8364 | 0.0600 | 0.7900 | 0.9042 | 3 |
| FedMedian | 0.8137 | 0.0469 | 0.7750 | 0.8817 | 4 |
| Bulyan | 0.7996 | 0.0642 | 0.7442 | 0.8850 | 4 |
| FedAgain | 0.7963 | 0.0539 | 0.6950 | 0.9000 | 15 |
| FedAvg | 0.7816 | 0.0505 | 0.7075 | 0.8683 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.9042 |
| Bulyan | 0.8850 |
| FedMedian | 0.8817 |
| FedAgain | 0.8717 |
| FedAvg | 0.8683 |

**FedAgain Rank:** 4 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.8150 |
| Bulyan | 0.8125 |
| FedAgain | 0.8042 |
| FedAvg | 0.7842 |
| FedMedian | 0.7750 |

**FedAgain vs FedAvg:** +2.55% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7942 |
| FedAgain | 0.7931 |
| FedProx | 0.7900 |
| FedAvg | 0.7802 |
| Bulyan | 0.7567 |

**FedAgain vs FedAvg:** +1.66% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.8042 |
| FedAgain | 0.7508 |
| Bulyan | 0.7442 |
| FedAvg | 0.7083 |

**FedAgain vs FedAvg:** +6.00% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.8158 | 0.7800 | +4.59% |
| 2 | 0.7375 | 0.7617 | -3.17% |
| 3 | 0.7969 | 0.7874 | +1.21% |
| 4 | 0.8000 | 0.7729 | +3.50% |
| 5 | 0.7858 | 0.7633 | +2.95% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.9000 | 0.8483 | +6.09% |
| non-iid-label | 0.7922 | 0.7659 | +3.44% |
| non-iid-dirichlet | 0.6950 | 0.8550 | -18.71% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.8442 | 0.8400 | +0.50% |
| 10 | 0.7948 | 0.7815 | +1.71% |
| 20 | 0.7250 | 0.7075 | +2.47% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +6.00% improvement over FedAvg
- **10% Corruption:** +2.55% improvement over FedAvg
- **30% Corruption:** +1.66% improvement over FedAvg

### Challenging Scenarios:

---

## Recommendations

Based on the analysis of DAUDON_SEC dataset:

✅ **FedAgain shows overall improvement** (+1.88%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for DAUDON_SEC dataset*

*Timestamp: 2025-10-31 03:03:25.966653*