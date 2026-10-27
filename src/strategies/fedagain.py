"""
FedAgain: Trust-based Federated Learning Strategy.

Weights clients by combining a benchmark-error signal (E_k) and a parameter
divergence signal (D_k) into a per-round trust weight T_k. The exact trust
formulation is delegated to :mod:`src.strategies.trust`, which supports the
ablation (#2) and multiplicative-vs-additive (#3) experiments.
"""

import csv
import numpy as np
from typing import List, Tuple

from . import trust as trust_fns


class FedAgainStrategy:
    """FedAgain trust-weighted aggregation.

    Trust weight (default): ``T_k = 1 / (E_k * D_k + epsilon)`` where
        E_k = benchmark error of client k,
        D_k = L2 divergence of the client update from the global model.

    The ``mode`` argument selects the trust formulation (see
    :data:`src.strategies.trust.TRUST_MODES`), enabling the ablation and
    sensitivity studies without changing the aggregation loop.
    """

    def __init__(self, epsilon: float = 1e-6, mode: str = "combined_mult",
                 alpha: float = 1.0, beta: float = 1.0,
                 log_trust: bool = False, norm_mode: str = "sum_to_one",
                 norm_temperature: float = 1.0, mix_lambda: float = 0.0):
        if mode not in trust_fns.TRUST_MODES:
            raise ValueError(
                f"Unknown trust mode '{mode}'. "
                f"Expected one of {trust_fns.TRUST_MODES}."
            )
        if not 0.0 <= mix_lambda <= 1.0:
            raise ValueError(f"mix_lambda must be in [0, 1], got {mix_lambda}.")
        if norm_mode not in trust_fns.NORM_MODES:
            raise ValueError(
                f"Unknown normalization mode '{norm_mode}'. "
                f"Expected one of {trust_fns.NORM_MODES}."
            )
        self.epsilon = epsilon
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.log_trust = log_trust
        self.norm_mode = norm_mode
        self.norm_temperature = norm_temperature
        self.mix_lambda = mix_lambda
        self.global_params = None
        # Per-round trust records, populated when log_trust is True.
        # Each entry: (round, client, E_k, D_k, T_raw, T_norm, w_final).
        self.trust_history: List[tuple] = []
        self._round = 0

    def aggregate(self, results: List[Tuple],
                  global_params: List[np.ndarray]) -> List[np.ndarray]:
        """Aggregate client updates using trust-based weighting.

        Args:
            results: list of ``(client_params, num_examples, metrics)``.
            global_params: current global model parameters.

        Returns:
            Aggregated parameters (falls back to ``global_params`` if empty).
        """
        if len(results) == 0:
            return global_params

        self._round += 1

        client_params_list = [r[0] for r in results]
        client_metrics = [r[2] for r in results]

        # Benchmark errors E_k (neutral 1.0 if a client did not report one).
        benchmark_errors = [
            m.get("benchmark_error", 1.0) for m in client_metrics
        ]

        # Divergences D_k.
        divergences = [
            trust_fns.compute_divergence(cp, global_params)
            for cp in client_params_list
        ]

        # Raw + normalized trust weights via the selected formulation.
        raw = trust_fns.raw_trust_scores(
            benchmark_errors, divergences,
            mode=self.mode, alpha=self.alpha, beta=self.beta,
            epsilon=self.epsilon,
        )
        trust_weights = trust_fns.normalize(
            raw, mode=self.norm_mode, temperature=self.norm_temperature)

        # Optional floor against benign distortion under real domain shift:
        # blend the trust weights with the data-size (FedAvg) weights,
        # w = (1 - lambda) * trust + lambda * n_k / n. lambda = 0 is FedAgain
        # unchanged; every client keeps at least lambda * n_k / n.
        if self.mix_lambda > 0.0:
            sizes = np.asarray([r[1] for r in results], dtype=np.float64)
            weights = ((1.0 - self.mix_lambda) * trust_weights
                       + self.mix_lambda * sizes / sizes.sum())
        else:
            weights = trust_weights

        if self.log_trust:
            for k, (E_k, D_k, t_raw, t_norm, w) in enumerate(
                zip(benchmark_errors, divergences, raw, trust_weights, weights)
            ):
                self.trust_history.append(
                    (self._round, k, float(E_k), float(D_k),
                     float(t_raw), float(t_norm), float(w))
                )

        # Trust-weighted average of the client parameters.
        num_params = len(client_params_list[0])
        aggregated = []
        for i in range(num_params):
            weighted_sum = None
            for client_params, w in zip(client_params_list, weights):
                contribution = client_params[i] * w
                weighted_sum = (contribution if weighted_sum is None
                                else weighted_sum + contribution)
            aggregated.append(weighted_sum)

        return aggregated

    def save_trust_log(self, path: str) -> None:
        """Write the per-round trust records to a CSV file (if any)."""
        if not self.trust_history:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["round", "client", "benchmark_error", "divergence",
                 "trust_raw", "trust_normalized", "weight"]
            )
            writer.writerows(self.trust_history)

    # Backwards-compatible alias for the old private method name.
    def _compute_divergence(self, client_params, global_params) -> float:
        return trust_fns.compute_divergence(client_params, global_params)


def compute_benchmark_error(model, benchmark_loader, device):
    """Average cross-entropy loss of ``model`` on ``benchmark_loader``.

    This is the benchmark error E_k each client computes on the incoming global
    model before local training. Cross-entropy (rather than accuracy) gives a
    continuous, calibration-sensitive reliability signal. Trust weights are
    normalized later, so no extra normalization is needed here.
    """
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in benchmark_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0
