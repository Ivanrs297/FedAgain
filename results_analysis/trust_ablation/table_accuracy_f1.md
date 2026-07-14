# Accuracy (mean +/- std over seeds), % — by dataset x trust mode

| dataset | E only | D only | E*D (mult) | E+D (add) |
|---|---|---|---|---|
| CIFAR10 | 84.44±1.83 | 82.18±1.54 | 83.26±1.91 | 81.89±2.15 |
| DAUDON_MIX | 80.28±0.80 | 79.44±1.96 | 79.67±0.83 | 79.03±1.46 |
| JONATHAN_MIX | 76.81±2.95 | 67.11±4.46 | 77.97±0.99 | 69.99±2.44 |
| MYSTONE | 56.34±1.08 | 53.36±1.84 | 54.93±3.42 | 51.34±5.28 |

## weighted_additive variants (accuracy %)

| dataset | variant | acc | f1 |
|---|---|---|---|
| CIFAR10 | wADD(a0.5,b1.0) | 82.16±1.90 | 82.14 |
| CIFAR10 | wADD(a1.0,b2.0) | 81.80±2.14 | 81.79 |
| CIFAR10 | wADD(a2.0,b1.0) | 82.39±2.01 | 82.44 |
| DAUDON_MIX | wADD(a0.5,b1.0) | 78.83±0.39 | 78.98 |
| DAUDON_MIX | wADD(a1.0,b2.0) | 78.33±0.57 | 78.58 |
| DAUDON_MIX | wADD(a2.0,b1.0) | 79.04±0.86 | 78.99 |
