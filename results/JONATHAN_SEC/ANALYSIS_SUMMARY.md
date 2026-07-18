# FedAgain Experimental Analysis: JONATHAN_SEC
**Analysis Date: 2025-10-31 03:03:29**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the JONATHAN_SEC dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.7889
- FedAvg Average Accuracy: 0.7276
- Overall Improvement: +8.44%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedMedian | 0.8519 | 0.0629 | 0.8058 | 0.9425 | 4 |
| FedProx | 0.8014 | 0.0952 | 0.7258 | 0.9083 | 3 |
| FedAgain | 0.7889 | 0.0467 | 0.7333 | 0.8683 | 15 |
| Bulyan | 0.7852 | 0.1165 | 0.6583 | 0.9125 | 4 |
| FedAvg | 0.7276 | 0.0882 | 0.6242 | 0.8917 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.9425 |
| Bulyan | 0.9125 |
| FedProx | 0.9083 |
| FedAvg | 0.8675 |
| FedAgain | 0.8600 |

**FedAgain Rank:** 5 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAvg | 0.8592 |
| Bulyan | 0.8500 |
| FedAgain | 0.8125 |
| FedMedian | 0.8058 |
| FedProx | 0.7700 |

**FedAgain vs FedAvg:** -5.43% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.8133 |
| FedAgain | 0.7744 |
| FedProx | 0.7258 |
| Bulyan | 0.7200 |
| FedAvg | 0.7056 |

**FedAgain vs FedAvg:** +9.76% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAgain | 0.8683 |
| FedMedian | 0.8458 |
| FedAvg | 0.7200 |
| Bulyan | 0.6583 |

**FedAgain vs FedAvg:** +20.60% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.7592 | 0.7067 | +7.43% |
| 2 | 0.7558 | 0.7325 | +3.19% |
| 3 | 0.7730 | 0.7099 | +8.89% |
| 4 | 0.7858 | 0.6796 | +15.63% |
| 5 | 0.7958 | 0.6992 | +13.83% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.7767 | 0.8158 | -4.80% |
| non-iid-label | 0.7669 | 0.6759 | +13.46% |
| non-iid-dirichlet | 0.8475 | 0.8917 | -4.95% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.7350 | 0.6342 | +15.90% |
| 10 | 0.7750 | 0.7147 | +8.43% |
| 20 | 0.8083 | 0.6850 | +18.00% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +20.60% improvement over FedAvg
- **30% Corruption:** +9.76% improvement over FedAvg

### Challenging Scenarios:
- **10% Corruption:** -5.43% vs FedAvg

---

## Recommendations

Based on the analysis of JONATHAN_SEC dataset:

✅ **FedAgain shows overall improvement** (+8.44%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for JONATHAN_SEC dataset*

*Timestamp: 2025-10-31 03:03:29.394077*