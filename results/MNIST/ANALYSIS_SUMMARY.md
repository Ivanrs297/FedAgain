# FedAgain Experimental Analysis: MNIST
**Analysis Date: 2025-10-31 03:03:31**
**Total Experiments: 41**

---

## Executive Summary

This analysis evaluates 5 federated learning strategies on the MNIST dataset.

**Overall Performance:**
- FedAgain Average Accuracy: 0.9926
- FedAvg Average Accuracy: 0.9932
- Overall Improvement: -0.07%

---

## Strategy Performance Summary

| Strategy | Mean Acc | Std Dev | Min | Max | Experiments |
|----------|----------|---------|-----|-----|-------------|
| FedMedian | 0.9940 | 0.0011 | 0.9932 | 0.9957 | 4 |
| FedProx | 0.9939 | 0.0028 | 0.9909 | 0.9965 | 3 |
| FedAvg | 0.9932 | 0.0015 | 0.9902 | 0.9961 | 15 |
| Bulyan | 0.9928 | 0.0033 | 0.9883 | 0.9964 | 4 |
| FedAgain | 0.9926 | 0.0028 | 0.9832 | 0.9957 | 15 |

---

## Test 1: Baseline (Clean Data)

**Setup:** No corrupted clients

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.9965 |
| Bulyan | 0.9964 |
| FedAvg | 0.9961 |
| FedAgain | 0.9957 |
| FedMedian | 0.9957 |

**FedAgain Rank:** 4 of 5


---

## Test 2: Light Corruption (10% Corrupted Clients)

**Setup:** 10% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedProx | 0.9943 |
| FedMedian | 0.9935 |
| Bulyan | 0.9934 |
| FedAvg | 0.9934 |
| FedAgain | 0.9927 |

**FedAgain vs FedAvg:** -0.07% improvement


---

## Test 3: Moderate Corruption (30% Corrupted Clients)

**Setup:** 30% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.9932 |
| Bulyan | 0.9931 |
| FedAvg | 0.9930 |
| FedAgain | 0.9923 |
| FedProx | 0.9909 |

**FedAgain vs FedAvg:** -0.08% improvement


---

## Test 4: Heavy Corruption (40% Corrupted Clients)

**Setup:** 40% corrupted clients, non-IID label distribution

**Results:**
| Strategy | Accuracy |
|----------|----------|
| FedMedian | 0.9936 |
| FedAgain | 0.9928 |
| FedAvg | 0.9923 |
| Bulyan | 0.9883 |

**FedAgain vs FedAvg:** +0.05% improvement


---

## Test 5: Corruption Severity Progression

**Setup:** 30% corrupted clients, varying severity levels

| Severity | FedAgain | FedAvg | Improvement |
|----------|----------|--------|-------------|
| 1 | 0.9926 | 0.9932 | -0.06% |
| 2 | 0.9932 | 0.9940 | -0.08% |
| 3 | 0.9916 | 0.9935 | -0.19% |
| 4 | 0.9935 | 0.9923 | +0.13% |
| 5 | 0.9931 | 0.9902 | +0.29% |


---

## Test 6: Non-IID + Corruption Analysis

**Setup:** 30% corrupted clients, various data distributions

| Distribution | FedAgain | FedAvg | Improvement |
|--------------|----------|--------|-------------|
| iid | 0.9946 | 0.9957 | -0.11% |
| non-iid-label | 0.9930 | 0.9926 | +0.04% |
| non-iid-dirichlet | 0.9832 | 0.9949 | -1.18% |


---

## Test 7: Scaling Number of Clients

**Setup:** 30% corrupted clients, varying client counts

| Clients | FedAgain | FedAvg | Improvement |
|---------|----------|--------|-------------|
| 5 | 0.9935 | 0.9939 | -0.04% |
| 10 | 0.9922 | 0.9931 | -0.09% |
| 20 | 0.9914 | 0.9917 | -0.03% |


---

## Key Findings

### Best Scenarios for FedAgain:
- **40% Corruption:** +0.05% improvement over FedAvg

### Challenging Scenarios:
- **30% Corruption:** -0.08% vs FedAvg
- **10% Corruption:** -0.07% vs FedAvg

---

## Recommendations

Based on the analysis of MNIST dataset:

⚠️ **FedAgain underperforms** (-0.07%) compared to FedAvg

### Suggested Use Cases:
- Consider dataset-specific characteristics when choosing FL strategy
- Evaluate Byzantine-robust methods (FedMedian, Bulyan) for heavy corruption
- Test with multiple data distributions to ensure robustness

---


*Analysis generated for MNIST dataset*

*Timestamp: 2025-10-31 03:03:31.066435*