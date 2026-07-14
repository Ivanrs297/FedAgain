# Trust suppression — mean normalized trust weight (seed 44)

Higher benign/corrupt ratio = stronger down-weighting of corrupted clients.

| dataset | mode | benign | corrupt | benign/corrupt |
|---|---|---|---|---|
| CIFAR10 | additive | 0.10024 | 0.09945 | 1.008 |
| CIFAR10 | benchmark_only | 0.12417 | 0.04359 | 2.848 |
| CIFAR10 | combined_mult | 0.12415 | 0.04365 | 2.845 |
| CIFAR10 | divergence_only | 0.10002 | 0.09995 | 1.001 |
| CIFAR10 | weighted_additive | 0.10013 | 0.09971 | 1.004 |
| DAUDON_MIX | additive | 0.10110 | 0.09743 | 1.038 |
| DAUDON_MIX | benchmark_only | 0.12517 | 0.04126 | 3.033 |
| DAUDON_MIX | combined_mult | 0.12648 | 0.03822 | 3.309 |
| DAUDON_MIX | divergence_only | 0.10059 | 0.09863 | 1.020 |
| DAUDON_MIX | weighted_additive | 0.10090 | 0.09791 | 1.031 |
| JONATHAN_MIX | additive | 0.10091 | 0.09787 | 1.031 |
| JONATHAN_MIX | benchmark_only | 0.12937 | 0.03146 | 4.112 |
| JONATHAN_MIX | combined_mult | 0.13023 | 0.02946 | 4.421 |
| JONATHAN_MIX | divergence_only | 0.10039 | 0.09910 | 1.013 |
| MYSTONE | additive | 0.10359 | 0.09161 | 1.131 |
| MYSTONE | benchmark_only | 0.10915 | 0.07866 | 1.388 |
| MYSTONE | combined_mult | 0.11311 | 0.06942 | 1.629 |
| MYSTONE | divergence_only | 0.10339 | 0.09210 | 1.123 |
