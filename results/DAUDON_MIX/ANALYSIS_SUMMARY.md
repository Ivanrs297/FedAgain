# FedAgain Experimental Analysis: DAUDON_MIX
**Analysis Date: 2025-10-31 03:03:24**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the DAUDON_MIX dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.8019
- FedAvg Average Accuracy: 0.7827
- Overall Improvement: +2.45%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedProx | 0.8214 | 0.0381 | 0.7825 | 0.8588 | 3 |
| FedMedian | 0.8169 | 0.0350 | 0.7800 | 0.8496 | 4 |
| Bulyan | 0.8145 | 0.0388 | 0.7612 | 0.8533 | 4 |
| FedAgain | 0.8019 | 0.0350 | 0.7492 | 0.8804 | 15 |
| FedAvg | 0.7827 | 0.0712 | 0.5613 | 0.8921 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAvg | 0.8921 |
| FedAgain | 0.8804 |
| FedProx | 0.8588 |
| Bulyan | 0.8533 |
| FedMedian | 0.8438 |

**FedAgain Rank:** 2 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.8496 |
| FedAvg | 0.8367 |
| Bulyan | 0.8279 |
| FedProx | 0.8229 |
| FedAgain | 0.8046 |

**FedAgain vs FedAvg:** -3.83% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| Bulyan | 0.8154 |
| FedMedian | 0.7942 |
| FedAgain | 0.7938 |
| FedProx | 0.7825 |
| FedAvg | 0.7716 |

**FedAgain vs FedAvg:** +2.88% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAgain | 0.8187 |
| FedMedian | 0.7800 |
| Bulyan | 0.7612 |
| FedAvg | 0.7533 |

**FedAgain vs FedAvg:** +8.68% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.7492 | 0.7688 | -2.55% |
| 2 | 0.8075 | 0.8046 | +0.36% |
| 3 | 0.8032 | 0.7601 | +5.66% |
| 4 | 0.7796 | 0.7727 | +0.89% |
| 5 | 0.7871 | 0.8192 | -3.92% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.8533 | 0.7954 | +7.28% |
| non-iid-label | 0.7905 | 0.7902 | +0.04% |
| non-iid-dirichlet | 0.7662 | 0.5613 | +36.53% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.8300 | 0.7817 | +6.18% |
| 10 | 0.7886 | 0.7722 | +2.12% |
| 20 | 0.8087 | 0.7546 | +7.18% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +8.68% improvement over FedAvg
- **30% Corruption:** +2.88% improvement over FedAvg

### Challenging Scenarios:
- **10% Corruption:** -3.83% vs FedAvg

---

## Recommendations

Based on the analysis of DAUDON_MIX dataset:

✅ **FedAgain shows overall improvement** (+2.45%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for DAUDON_MIX dataset*

*Timestamp: 2025-10-31 03:03:24.277043*