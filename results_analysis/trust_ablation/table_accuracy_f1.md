# Accuracy (mean +/- std over seeds), % — by dataset x trust mode

| dataset | E only | D only | E*D (mult) | E+D (add) |
|---|---|---|---|---|
| CIFAR10 | 84.87±2.05 | 82.21±2.10 | 84.66±1.62 | 81.39±2.32 |
| DAUDON_MIX | 81.36±1.23 | 78.39±0.64 | 79.60±1.14 | 79.82±1.92 |
| JONATHAN_MIX | 74.72±3.05 | 69.53±4.16 | 75.24±6.11 | 70.00±1.65 |
| MYSTONE | 55.32±3.23 | 54.11±2.86 | 55.32±3.33 | 54.88±2.51 |

## weighted_additive variants (accuracy %)

| dataset | variant | acc | f1 |
|---|---|---|---|
| CIFAR10 | wADD(a0.5,b1.0) | 82.45±2.12 | 82.49 |
| CIFAR10 | wADD(a1.0,b2.0) | 82.02±1.54 | 82.00 |
| CIFAR10 | wADD(a2.0,b1.0) | 82.68±1.82 | 82.58 |
| DAUDON_MIX | wADD(a0.5,b1.0) | 79.22±0.84 | 79.29 |
| DAUDON_MIX | wADD(a1.0,b2.0) | 77.68±1.87 | 77.83 |
| DAUDON_MIX | wADD(a2.0,b1.0) | 78.58±1.01 | 78.76 |
| JONATHAN_MIX | wADD(a0.5,b1.0) | 67.47±3.50 | 66.00 |
| JONATHAN_MIX | wADD(a1.0,b2.0) | 68.82±4.88 | 67.07 |
| JONATHAN_MIX | wADD(a2.0,b1.0) | 65.42±2.32 | 63.25 |
| MYSTONE | wADD(a0.5,b1.0) | 53.13±2.07 | 50.98 |
| MYSTONE | wADD(a1.0,b2.0) | 54.35±2.46 | 51.90 |
| MYSTONE | wADD(a2.0,b1.0) | 52.44±3.68 | 51.14 |
