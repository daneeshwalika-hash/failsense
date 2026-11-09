"""Prediction log in DuckDB: one row per ``POST /predict`` (what was asked, what was answered).

Why a log: it is the raw material of monitoring (alert rate, risk distribution over time, drift
check) and of audits ("what did the model say about truck 7 on Tuesday?"). It stores the answer
and the few request facts needed for that, NOT the 105 readout columns and NOT the raw model score.

DuckDB is an embedded analytical database (a single file, SQL, no server). One process may
write to the file at a time, so only the API opens it; every other consumer (dashboard, drift
check) reads through ``GET /predictions/latest`` or opens the file after the API has stopped.

SQL Server analogy: a small logging table with an IDENTITY column and a default timestamp.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from failsense.config import resolve_path

logger = logging.getLogger(__name__)

TABLE = "predictions_log"
MAX_LATEST = 1000

_DDL = (
    "CREATE SEQUENCE IF NOT EXISTS predictions_log_id START 1",
    f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        id            BIGINT PRIMARY KEY DEFAULT nextval('predictions_log_id'),
        logged_at     TIMESTAMP NOT NULL DEFAULT current_timestamp,
        vehicle_id    BIGINT,
        time_step     DOUBLE NOT NULL,
        has_previous  BOOLEAN NOT NULL,
        risk          DOUBLE NOT NULL,
        alert         BOOLEAN NOT NULL,
        threshold     DOUBLE NOT NULL,
        model_name    VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL
    )""",
)


@dataclass(frozen=True)
class PredictionRecord:
    """The facts of one prediction that are written to the log."""

    vehicle_id: int | None
    time_step: float
    has_previous: bool
    risk: float
    alert: bool
    threshold: float
    model_name: str
    model_version: str


class PredictionLog:
    """Append-only prediction log (thread-safe: FastAPI runs sync routes in a thread pool)."""

    def __init__(self, path: str | Path) -> None:
        """Open (and create if needed) the DuckDB file; ``":memory:"`` for a throw-away log."""
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(target)
        self._lock = threading.Lock()
        self.path = target
        with self._lock:
            for statement in _DDL:
                self._con.execute(statement)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PredictionLog | None:
        """The log named by ``api.prediction_log``; ``None`` (log disabled) if it is null."""
        configured = config["api"].get("prediction_log")
        if not configured:
            logger.info("Prediction log disabled (api.prediction_log is empty)")
            return None
        log = cls(resolve_path(configured))
        logger.info("Prediction log: %s (%d rows)", log.path, log.count())
        return log

    def log(self, record: PredictionRecord) -> None:
        """Append one row."""
        with self._lock:
            self._con.execute(
                f"""INSERT INTO {TABLE}
                    (vehicle_id, time_step, has_previous, risk, alert, threshold,
                     model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [record.vehicle_id, record.time_step, record.has_previous, record.risk,
                 record.alert, record.threshold, record.model_name, record.model_version],
            )  # fmt: skip

    def count(self) -> int:
        """Number of logged predictions."""
        with self._lock:
            return int(self._con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0])

    def latest(self, n: int = 20) -> list[dict[str, Any]]:
        """The ``n`` most recent predictions, newest first.

        Raises:
            ValueError: If ``n`` is not in 1..``MAX_LATEST``.
        """
        if not 1 <= n <= MAX_LATEST:
            raise ValueError(f"n must be in 1..{MAX_LATEST}, got {n}")
        with self._lock:
            frame = self._con.execute(f"SELECT * FROM {TABLE} ORDER BY id DESC LIMIT ?", [n]).df()
        frame["logged_at"] = frame["logged_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        frame["vehicle_id"] = frame["vehicle_id"].astype(object).where(frame["vehicle_id"].notna())
        return [
            {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
             for k, v in row.items()}
            for row in frame.to_dict("records")
        ]  # fmt: skip

    def close(self) -> None:
        """Close the file (releases the writer lock)."""
        with self._lock:
            self._con.close()
