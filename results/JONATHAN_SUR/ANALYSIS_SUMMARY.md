# FedAgain Experimental Analysis: JONATHAN_SUR
**Analysis Date: 2025-10-31 03:03:22**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the JONATHAN_SUR dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.7432
- FedAvg Average Accuracy: 0.7120
- Overall Improvement: +4.39%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedProx | 0.7522 | 0.0874 | 0.6817 | 0.8500 | 3 |
| FedMedian | 0.7500 | 0.0547 | 0.7133 | 0.8292 | 4 |
| FedAgain | 0.7432 | 0.0519 | 0.6783 | 0.8950 | 15 |
| Bulyan | 0.7344 | 0.1051 | 0.5942 | 0.8483 | 4 |
| FedAvg | 0.7120 | 0.0835 | 0.5950 | 0.8733 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedAgain | 0.8950 |
| FedAvg | 0.8733 |
| FedProx | 0.8500 |
| Bulyan | 0.8483 |
| FedMedian | 0.8292 |

**FedAgain Rank:** 1 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| Bulyan | 0.7567 |
| FedMedian | 0.7442 |
| FedAvg | 0.7408 |
| FedAgain | 0.7300 |
| FedProx | 0.6817 |

**FedAgain vs FedAvg:** -1.46% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| Bulyan | 0.7383 |
| FedAgain | 0.7352 |
| FedProx | 0.7250 |
| FedMedian | 0.7133 |
| FedAvg | 0.6962 |

**FedAgain vs FedAvg:** +5.61% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.7133 |
| FedAvg | 0.7117 |
| FedAgain | 0.7008 |
| Bulyan | 0.5942 |

**FedAgain vs FedAvg:** -1.52% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.7483 | 0.6275 | +19.26% |
| 2 | 0.7167 | 0.6692 | +7.10% |
| 3 | 0.7471 | 0.7215 | +3.55% |
| 4 | 0.6917 | 0.7058 | -2.01% |
| 5 | 0.7442 | 0.5950 | +25.07% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.7600 | 0.8508 | -10.68% |
| non-iid-label | 0.7274 | 0.6753 | +7.73% |
| non-iid-dirichlet | 0.7883 | 0.7508 | +4.99% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.7367 | 0.8083 | -8.87% |
| 10 | 0.7354 | 0.6896 | +6.65% |
| 20 | 0.7317 | 0.6500 | +12.56% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **30% Corruption:** +5.61% improvement over FedAvg

### Challenging Scenarios:
- **40% Corruption:** -1.52% vs FedAvg
- **10% Corruption:** -1.46% vs FedAvg

---

## Recommendations

Based on the analysis of JONATHAN_SUR dataset:

✅ **FedAgain shows overall improvement** (+4.39%) over FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for JONATHAN_SUR dataset*

*Timestamp: 2025-10-31 03:03:22.585204*