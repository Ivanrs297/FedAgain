# FedAgain Experimental Analysis: JONATHAN_MIX
**Analysis Date: 2025-10-31 03:03:20**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the JONATHAN_MIX dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.7761
- FedAvg Average Accuracy: 0.7181
- Overall Improvement: +8.08%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedMedian | 0.8049 | 0.0202 | 0.7808 | 0.8246 | 4 |
| FedProx | 0.7769 | 0.0739 | 0.7004 | 0.8479 | 3 |
| FedAgain | 0.7761 | 0.0310 | 0.7371 | 0.8308 | 15 |
| Bulyan | 0.7301 | 0.0598 | 0.6508 | 0.7821 | 4 |
| FedAvg | 0.7181 | 0.0674 | 0.6342 | 0.8871 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAvg | 0.8871 |
| FedProx | 0.8479 |
| FedAgain | 0.8308 |
| FedMedian | 0.8246 |
| Bulyan | 0.7700 |

**FedAgain Rank:** 3 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.8183 |
| FedProx | 0.7825 |
| FedAvg | 0.7725 |
| FedAgain | 0.7654 |
| Bulyan | 0.7175 |

**FedAgain vs FedAvg:** -0.92% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7958 |
| Bulyan | 0.7821 |
| FedAgain | 0.7754 |
| FedAvg | 0.7034 |
| FedProx | 0.7004 |

**FedAgain vs FedAvg:** +10.23% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7808 |
| FedAgain | 0.7400 |
| FedAvg | 0.6700 |
| Bulyan | 0.6508 |

**FedAgain vs FedAvg:** +10.45% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.7546 | 0.6554 | +15.13% |
| 2 | 0.7371 | 0.6817 | +8.13% |
| 3 | 0.7835 | 0.7184 | +9.06% |
| 4 | 0.7902 | 0.6933 | +13.97% |
| 5 | 0.7488 | 0.6887 | +8.71% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.8229 | 0.7404 | +11.14% |
| non-iid-label | 0.7720 | 0.7054 | +9.45% |
| non-iid-dirichlet | 0.7621 | 0.6471 | +17.77% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.8208 | 0.7704 | +6.54% |
| 10 | 0.7702 | 0.6971 | +10.48% |
| 20 | 0.7825 | 0.6996 | +11.85% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +10.45% improvement over FedAvg
- **30% Corruption:** +10.23% improvement over FedAvg

### Challenging Scenarios:
- **10% Corruption:** -0.92% vs FedAvg

---

## Recommendations

Based on the analysis of JONATHAN_MIX dataset:

✅ **FedAgain shows overall improvement** (+8.08%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for JONATHAN_MIX dataset*

*Timestamp: 2025-10-31 03:03:20.913383*