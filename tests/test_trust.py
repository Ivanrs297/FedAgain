"""
Unit tests for the modular trust-score computation (paper experiments #2/#3).

Runnable without a GPU or datasets:  python tests/test_trust.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies import trust  # noqa: E402
from src.strategies.fedagain import FedAgainStrategy  # noqa: E402

EPS = 1e-6


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_divergence():
    a = [np.array([3.0, 0.0])]
    b = [np.array([0.0, 4.0])]
    # ||(3,-4)||_2 = 5
    assert approx(trust.compute_divergence(a, b), 5.0)


def test_modes_formula():
    E = [0.25, 0.5]
    D = [0.8, 0.6]

    mult = trust.raw_trust_scores(E, D, mode="combined_mult", epsilon=EPS)
    assert approx(mult[0], 1.0 / (0.25 * 0.8 + EPS))
    assert approx(mult[1], 1.0 / (0.5 * 0.6 + EPS))

    bench = trust.raw_trust_scores(E, D, mode="benchmark_only", epsilon=EPS)
    assert approx(bench[0], 1.0 / (0.25 + EPS))
    # benchmark_only must ignore D entirely
    bench2 = trust.raw_trust_scores(E, [999.0, -5.0], mode="benchmark_only",
                                    epsilon=EPS)
    assert np.allclose(bench, bench2)

    div = trust.raw_trust_scores(E, D, mode="divergence_only", epsilon=EPS)
    assert approx(div[0], 1.0 / (0.8 + EPS))

    add = trust.raw_trust_scores(E, D, mode="additive", epsilon=EPS)
    assert approx(add[0], 1.0 / (0.25 + 0.8 + EPS))

    wadd = trust.raw_trust_scores(E, D, mode="weighted_additive",
                                  alpha=2.0, beta=0.5, epsilon=EPS)
    assert approx(wadd[0], 1.0 / (2.0 * 0.25 + 0.5 * 0.8 + EPS))


def test_normalize():
    w = trust.normalize([1.0, 3.0])
    assert approx(w.sum(), 1.0)
    assert approx(w[0], 0.25) and approx(w[1], 0.75)
    # degenerate -> uniform
    u = trust.normalize([0.0, 0.0])
    assert approx(u[0], 0.5) and approx(u[1], 0.5)


def test_unknown_mode_raises():
    try:
        trust.raw_trust_scores([1.0], [1.0], mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_strategy_aggregate_and_log():
    # Two clients, one 2-element parameter tensor each.
    gp = [np.array([0.0, 0.0])]
    c0 = [np.array([1.0, 1.0])]   # divergence sqrt(2)
    c1 = [np.array([2.0, 2.0])]   # divergence sqrt(8)
    results = [
        (c0, 10, {"benchmark_error": 0.2}),
        (c1, 10, {"benchmark_error": 0.5}),
    ]
    strat = FedAgainStrategy(mode="combined_mult", log_trust=True)
    agg = strat.aggregate(results, gp)

    # Reproduce expected weights independently.
    D = [np.sqrt(2.0), np.sqrt(8.0)]
    raw = trust.raw_trust_scores([0.2, 0.5], D, mode="combined_mult",
                                 epsilon=strat.epsilon)
    w = trust.normalize(raw)
    expected = c0[0] * w[0] + c1[0] * w[1]
    assert np.allclose(agg[0], expected)
    # trust history recorded for both clients in round 1
    assert len(strat.trust_history) == 2
    assert strat.trust_history[0][0] == 1  # round index


def test_empty_results_returns_global():
    gp = [np.array([1.0, 2.0])]
    strat = FedAgainStrategy()
    assert strat.aggregate([], gp) is gp


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
