"""
Unit tests for the resilient ExperimentTracker.  Run: python tests/test_tracking.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking import ExperimentTracker, make_run_key  # noqa: E402

CFG = {
    "strategy": "FedAgain", "dataset": "MNIST", "data_distribution": "iid",
    "num_clients": 2, "num_rounds": 3, "corrupt_ratio": 0.3,
    "corruption_severity": 3, "trust_mode": "additive", "seed": 42,
}


def test_run_key_deterministic_and_distinguishes_mode():
    k1 = make_run_key(CFG)
    k2 = make_run_key(dict(CFG))
    assert k1 == k2
    other = dict(CFG); other["trust_mode"] = "combined_mult"
    assert make_run_key(other) != k1
    assert "tm-additive" in k1


def test_incremental_and_completion():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(CFG, results_dir=d)
        assert not t.is_completed()
        t.start()
        # manifest exists and marked running after start
        assert t.manifest_path.exists()
        assert json.loads(t.manifest_path.read_text())["status"] == "running"

        for r in range(1, 4):
            t.log_round(r, {"accuracy": 0.9 + r / 100, "loss": 0.1})
        # per-round CSV has 3 data rows persisted live
        assert t.rounds_path.exists()
        assert len(t.rounds_path.read_text().strip().splitlines()) == 4  # +header

        t.finish({"accuracy": 0.93, "precision": 0.9, "recall": 0.9, "f1": 0.9})
        assert t.is_completed()
        man = json.loads(t.manifest_path.read_text())
        assert man["status"] == "completed"
        assert man["rounds_completed"] == 3
        assert man["final_metrics"]["accuracy"] == 0.93
        # ledger CSV written with one row
        assert t.ledger_csv.exists()
        assert len(t.ledger_csv.read_text().strip().splitlines()) == 2  # +header


def test_resume_skip():
    with tempfile.TemporaryDirectory() as d:
        ExperimentTracker(CFG, results_dir=d)  # first
        t1 = ExperimentTracker(CFG, results_dir=d)
        t1.start()
        t1.finish({"accuracy": 0.9})
        # a fresh tracker with the same config sees it as completed
        t2 = ExperimentTracker(CFG, results_dir=d)
        assert t2.is_completed()


def test_fail_records_failure():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(CFG, results_dir=d)
        t.start()
        t.log_round(1, {"accuracy": 0.5})
        t.fail("boom")
        man = json.loads(t.manifest_path.read_text())
        assert man["status"] == "failed"
        assert "boom" in man["error"]
        # a failed run must NOT be treated as completed (so it reruns)
        assert not ExperimentTracker(CFG, results_dir=d).is_completed()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
