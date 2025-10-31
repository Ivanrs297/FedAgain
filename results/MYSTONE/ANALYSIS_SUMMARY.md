# FedAgain Experimental Analysis: MYSTONE
**Analysis Date: 2025-10-31 03:03:17**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the MYSTONE dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.5583
- FedAvg Average Accuracy: 0.5482
- Overall Improvement: +1.83%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedProx | 0.5944 | 0.0631 | 0.5217 | 0.6342 | 3 |
| Bulyan | 0.5756 | 0.1075 | 0.4175 | 0.6450 | 4 |
| FedMedian | 0.5629 | 0.0894 | 0.4525 | 0.6708 | 4 |
| FedAgain | 0.5583 | 0.0502 | 0.4942 | 0.6525 | 15 |
| FedAvg | 0.5482 | 0.0383 | 0.4950 | 0.6317 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.6708 |
| Bulyan | 0.6450 |
| FedAgain | 0.6342 |
| FedAvg | 0.6317 |
| FedProx | 0.6275 |

**FedAgain Rank:** 3 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.6342 |
| FedAgain | 0.6050 |
| Bulyan | 0.5983 |
| FedAvg | 0.5483 |
| FedMedian | 0.4525 |

**FedAgain vs FedAvg:** +10.33% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| Bulyan | 0.6417 |
| FedMedian | 0.5567 |
| FedAgain | 0.5504 |
| FedAvg | 0.5456 |
| FedProx | 0.5217 |

**FedAgain vs FedAvg:** +0.88% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.5717 |
| FedAgain | 0.5300 |
| FedAvg | 0.4958 |
| Bulyan | 0.4175 |

**FedAgain vs FedAvg:** +6.89% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.6000 | 0.5208 | +15.20% |
| 2 | 0.5608 | 0.5500 | +1.97% |
| 3 | 0.5523 | 0.5493 | +0.54% |
| 4 | 0.5187 | 0.5337 | -2.81% |
| 5 | 0.5408 | 0.5642 | -4.14% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.6525 | 0.5733 | +13.81% |
| non-iid-label | 0.5443 | 0.5436 | +0.14% |
| non-iid-dirichlet | 0.5092 | 0.5383 | -5.42% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.6025 | 0.6058 | -0.55% |
| 10 | 0.5507 | 0.5424 | +1.54% |
| 20 | 0.4950 | 0.5175 | -4.35% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **10% Corruption:** +10.33% improvement over FedAvg
- **40% Corruption:** +6.89% improvement over FedAvg
- **30% Corruption:** +0.88% improvement over FedAvg

### Challenging Scenarios:

---

## Recommendations

Based on the analysis of MYSTONE dataset:

✅ **FedAgain shows overall improvement** (+1.83%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for MYSTONE dataset*

*Timestamp: 2025-10-31 03:03:17.482797*