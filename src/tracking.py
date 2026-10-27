"""
Resilient experiment tracking for long-running federated experiments.

These experiments are slow and can crash mid-way, so results are written
*incrementally*: every completed round is persisted immediately and each run's
status (running / completed / failed) is kept up to date. Completed runs can be
detected and skipped on resume.

Files written under ``results_dir`` (default ``"results"``):

    experiment_ledger.csv   append-only, one row per run with config + final
                            metrics + status (authoritative tabular log,
                            includes ``trust_mode`` so ablation runs are
                            distinguishable).
    runs/<run_key>.json     per-run manifest: full config, status, timings, and
                            every round's metrics; rewritten atomically each
                            round.
    rounds/<run_key>.csv    per-round metrics appended live as they finish.
    ledger.jsonl            append-only event stream (start / round / finish /
                            fail) for auditing.

The legacy ``results/experiments.csv`` (written by :class:`ExperimentLogger`)
is left untouched for backward compatibility with existing analysis scripts.
"""

import atexit
import csv
import json
import os
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Fields that define an experiment's identity (used for the run key and the
# skip-if-already-completed logic).
KEY_FIELDS = [
    "strategy", "dataset", "data_distribution", "num_clients", "num_rounds",
    "local_epochs", "corrupt_clients", "corrupt_ratio", "corruption_severity",
    "poison_mode", "trust_mode", "trust_alpha", "trust_beta", "epsilon",
    "norm_mode", "norm_temperature", "seed",
]

LEDGER_COLUMNS = [
    "run_key", "timestamp_start", "timestamp_end", "status", "host",
    "model", "strategy", "dataset", "data_distribution", "num_clients",
    "num_rounds", "local_epochs", "batch_size", "learning_rate",
    "use_pretrained", "corrupt_clients", "corrupt_ratio", "corruption_severity",
    "trust_mode", "trust_alpha", "trust_beta",
    "final_accuracy", "final_precision", "final_recall", "final_f1",
    "rounds_completed",
]


def make_run_key(config: dict) -> str:
    """Deterministic, compact, filename-safe identity string for an experiment.

    Uses short field prefixes so the key stays well under filesystem name
    limits even as more dimensions (datasets, baselines) are added.
    """
    c = config
    parts = [str(c.get("strategy", "")), str(c.get("dataset", "")),
             str(c.get("data_distribution", ""))]
    if c.get("num_clients") is not None:
        parts.append(f"k{c['num_clients']}")
    if c.get("num_rounds") is not None:
        parts.append(f"r{c['num_rounds']}")
    if c.get("local_epochs") is not None:
        parts.append(f"e{c['local_epochs']}")
    if c.get("corrupt_clients"):
        parts.append(f"cr{c.get('corrupt_ratio', '')}")
        parts.append(f"sev{c.get('corruption_severity', '')}")
        # Only tag non-default poisoning so image-corruption keys stay unchanged.
        pm = c.get("poison_mode")
        if pm and pm != "image":
            parts.append(f"pm-{pm}")
    else:
        parts.append("clean")
    tm = c.get("trust_mode")
    if tm:
        parts.append(f"tm-{tm}")
        if tm == "weighted_additive":
            parts.append(f"a{c.get('trust_alpha', '')}")
            parts.append(f"b{c.get('trust_beta', '')}")
        # Only tag a non-default epsilon so all pre-existing keys stay valid.
        eps = c.get("epsilon")
        if eps not in (None, "", 1e-6):
            parts.append(f"eps{eps}")
        # Same convention for the normalization rule: only tag non-defaults so
        # every pre-existing run key stays valid.
        nm = c.get("norm_mode")
        if nm not in (None, "", "sum_to_one"):
            parts.append(f"nm-{nm}")
            if nm == "softmax":
                parts.append(f"tau{c.get('norm_temperature', '')}")
        # Trust/data-size mixing (round 2): untagged at the default lambda = 0.
        mix = c.get("mix_lambda")
        if mix not in (None, "", 0, 0.0):
            parts.append(f"mix{mix}")
    if c.get("seed") is not None:
        parts.append(f"s{c['seed']}")
    raw = "_".join(p for p in parts if p != "")
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in raw)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


class ExperimentTracker:
    """Track one experiment run with crash-resilient, incremental logging."""

    def __init__(self, config: dict, results_dir: str = "results"):
        self.config = dict(config)
        self.results_dir = Path(results_dir)
        self.run_key = make_run_key(self.config)

        self.runs_dir = self.results_dir / "runs"
        self.rounds_dir = self.results_dir / "rounds"
        self.manifest_path = self.runs_dir / f"{self.run_key}.json"
        self.rounds_path = self.rounds_dir / f"{self.run_key}.csv"
        self.ledger_path = self.results_dir / "ledger.jsonl"
        self.ledger_csv = self.results_dir / "experiment_ledger.csv"

        self.rounds: list = []
        self.status = "pending"
        self.started_at = None
        self._started = False
        self._finished = False

    # -- resume / skip ---------------------------------------------------
    def is_completed(self) -> bool:
        """True if a previous run with the same identity finished cleanly."""
        if not self.manifest_path.exists():
            return False
        try:
            data = json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return data.get("status") == "completed"

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now()
        self.status = "running"
        self._started = True
        self._write_manifest()
        self._append_ledger_event("start")
        # Ensure an interrupted run is marked failed even if no except clause
        # runs (e.g. an unhandled exception or a hard exit).
        atexit.register(self._finalize_atexit)
        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

    def log_round(self, round_num: int, metrics: dict) -> None:
        """Persist a single round's metrics immediately."""
        row = {"round": round_num,
               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row.update({k: float(v) for k, v in metrics.items()})
        self.rounds.append(row)

        # append to the live per-round CSV (write header once)
        write_header = not self.rounds_path.exists()
        with open(self.rounds_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # refresh the manifest so partial progress survives a crash
        self._write_manifest()

    def finish(self, final_metrics: dict) -> None:
        self.status = "completed"
        self._finished = True
        self.final_metrics = dict(final_metrics)
        self._write_manifest(final_metrics=final_metrics)
        self._append_ledger_event("finish", extra={"final": final_metrics})
        self._append_ledger_csv(final_metrics)
        print(f"[tracker] run '{self.run_key}' completed "
              f"({len(self.rounds)} rounds logged).")

    def fail(self, error: str) -> None:
        self.status = "failed"
        self._finished = True
        self._write_manifest(error=error)
        self._append_ledger_event("fail", extra={"error": error})
        self._append_ledger_csv(None)
        print(f"[tracker] run '{self.run_key}' FAILED: {error}")

    # -- internals -------------------------------------------------------
    def _write_manifest(self, final_metrics=None, error=None) -> None:
        manifest = {
            "run_key": self.run_key,
            "status": self.status,
            "host": socket.gethostname(),
            "started_at": (self.started_at.strftime("%Y-%m-%d %H:%M:%S")
                           if self.started_at else None),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "rounds_completed": len(self.rounds),
            "rounds": self.rounds,
        }
        if final_metrics is not None:
            manifest["final_metrics"] = final_metrics
        if error is not None:
            manifest["error"] = error
        _atomic_write_json(self.manifest_path, manifest)

    def _append_ledger_event(self, event: str, extra: dict = None) -> None:
        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "run_key": self.run_key,
        }
        if extra:
            record.update(extra)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _append_ledger_csv(self, final_metrics) -> None:
        cfg = self.config
        fm = final_metrics or {}
        row = {
            "run_key": self.run_key,
            "timestamp_start": (self.started_at.strftime("%Y-%m-%d %H:%M:%S")
                                if self.started_at else ""),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": self.status,
            "host": socket.gethostname(),
            "model": cfg.get("model", ""),
            "strategy": cfg.get("strategy", ""),
            "dataset": cfg.get("dataset", ""),
            "data_distribution": cfg.get("data_distribution", ""),
            "num_clients": cfg.get("num_clients", ""),
            "num_rounds": cfg.get("num_rounds", ""),
            "local_epochs": cfg.get("local_epochs", ""),
            "batch_size": cfg.get("batch_size", ""),
            "learning_rate": cfg.get("learning_rate", ""),
            "use_pretrained": cfg.get("use_pretrained", ""),
            "corrupt_clients": cfg.get("corrupt_clients", ""),
            "corrupt_ratio": cfg.get("corrupt_ratio", ""),
            "corruption_severity": cfg.get("corruption_severity", ""),
            "trust_mode": cfg.get("trust_mode", ""),
            "trust_alpha": cfg.get("trust_alpha", ""),
            "trust_beta": cfg.get("trust_beta", ""),
            "final_accuracy": fm.get("accuracy", ""),
            "final_precision": fm.get("precision", ""),
            "final_recall": fm.get("recall", ""),
            "final_f1": fm.get("f1", ""),
            "rounds_completed": len(self.rounds),
        }
        write_header = not self.ledger_csv.exists()
        with open(self.ledger_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _excepthook(self, exc_type, exc, tb) -> None:
        if self._started and not self._finished:
            self.fail("".join(traceback.format_exception(exc_type, exc, tb)))
        self._prev_excepthook(exc_type, exc, tb)

    def _finalize_atexit(self) -> None:
        if self._started and not self._finished:
            self.fail("process exited before completion (interrupted)")
