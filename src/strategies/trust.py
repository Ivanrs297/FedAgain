"""
Trust-score computation for the FedAgain strategy.

This module isolates *how* the per-client trust weight ``T_k`` is derived from
the two trust signals used by FedAgain:

    * ``E_k`` -- benchmark (validation) error of the incoming global model on
      client ``k`` (data-quality signal).
    * ``D_k`` -- L2 parameter divergence of the client update from the global
      model (update-consistency signal).

Keeping this logic in one place lets us swap the trust formulation without
touching the aggregation loop. It supports the two paper experiments:

    * #2 Trust-score ablation  -> ``benchmark_only`` / ``divergence_only`` /
      ``combined_mult``.
    * #3 Multiplicative vs. additive sensitivity -> ``combined_mult`` /
      ``additive`` / ``weighted_additive``.
"""

import numpy as np

# Canonical set of supported trust formulations.
TRUST_MODES = (
    "combined_mult",      # T = 1 / (E * D + eps)         (default / current)
    "benchmark_only",     # T = 1 / (E + eps)             (ablation #2)
    "divergence_only",    # T = 1 / (D + eps)             (ablation #2)
    "additive",           # T = 1 / (E + D + eps)         (#3)
    "weighted_additive",  # T = 1 / (alpha*E + beta*D+eps)(#3)
)


def compute_divergence(client_params, global_params):
    """Return the L2 divergence ``D_k = ||w_k - w_server||_2``.

    Args:
        client_params: list of np.ndarray, the client's updated parameters.
        global_params: list of np.ndarray, the current global parameters.
    """
    squared = 0.0
    for c, g in zip(client_params, global_params):
        diff = c - g
        squared += float(np.sum(diff ** 2))
    return float(np.sqrt(squared))


def raw_trust_scores(benchmark_errors, divergences, mode="combined_mult",
                     alpha=1.0, beta=1.0, epsilon=1e-6):
    """Compute the raw (un-normalized) trust scores ``T_k`` for every client.

    Args:
        benchmark_errors: sequence of E_k (one per client).
        divergences: sequence of D_k (one per client).
        mode: one of :data:`TRUST_MODES`.
        alpha, beta: weights used only by ``weighted_additive``.
        epsilon: numerical-stability constant added to the denominator.

    Returns:
        np.ndarray of raw trust scores (higher = more trusted).
    """
    E = np.asarray(benchmark_errors, dtype=np.float64)
    D = np.asarray(divergences, dtype=np.float64)

    if mode == "combined_mult":
        denom = E * D
    elif mode == "benchmark_only":
        denom = E
    elif mode == "divergence_only":
        denom = D
    elif mode == "additive":
        denom = E + D
    elif mode == "weighted_additive":
        denom = alpha * E + beta * D
    else:
        raise ValueError(
            f"Unknown trust mode '{mode}'. Expected one of {TRUST_MODES}."
        )

    return 1.0 / (denom + epsilon)


# Normalization strategies mapping raw trust scores to aggregation weights.
# Reviewer 2.5 asked for alternatives to the deployed sum-to-one rule.
NORM_MODES = (
    "sum_to_one",  # w_k = T_k / sum_j T_j            (default / deployed)
    "softmax",     # w_k = softmax(T_k / tau)         (needs a temperature)
    "rank",        # w_k proportional to the rank of T_k (magnitude discarded)
)


def normalize(trust, mode="sum_to_one", temperature=1.0):
    """Normalize raw trust scores to aggregation weights summing to 1.

    Args:
        trust: sequence of raw trust scores T_k (higher = more trusted).
        mode: one of :data:`NORM_MODES`.
        temperature: softmax temperature tau; ignored by the other modes.

    Falls back to a uniform distribution if the total is non-positive
    (degenerate case, e.g. all-zero trust).
    """
    t = np.asarray(trust, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        return np.ones_like(t) / len(t)

    if mode == "sum_to_one":
        total = float(t.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.ones_like(t) / len(t)
        return t / total

    if mode == "softmax":
        # Raw trust spans several orders of magnitude, so softmax is applied to
        # the log scores; on the linear scale a single client saturates the
        # distribution regardless of tau. Shift-invariant, hence the max-subtract.
        z = np.log(np.maximum(t, np.finfo(np.float64).tiny)) / float(temperature)
        z -= z.max()
        e = np.exp(z)
        s = float(e.sum())
        if s <= 0.0 or not np.isfinite(s):
            return np.ones_like(t) / len(t)
        return e / s

    if mode == "rank":
        # Ranks 1..n by increasing trust, normalized: keeps the ordering but
        # discards how far apart the scores are.
        order = np.argsort(np.argsort(t))  # 0 = least trusted
        r = order.astype(np.float64) + 1.0
        return r / r.sum()

    raise ValueError(
        f"Unknown normalization mode '{mode}'. Expected one of {NORM_MODES}."
    )
