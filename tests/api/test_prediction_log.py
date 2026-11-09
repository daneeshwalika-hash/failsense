"""DuckDB prediction log: schema, ordering, persistence, threads, validation."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from failsense.api.prediction_log import MAX_LATEST, PredictionLog, PredictionRecord


def record(i: int = 1, vehicle_id: int | None = 7, alert: bool = True) -> PredictionRecord:
    return PredictionRecord(
        vehicle_id=vehicle_id, time_step=float(i), has_previous=i % 2 == 0, risk=0.01 * i,
        alert=alert, threshold=0.0353, model_name="failsense-q1", model_version="3",
    )  # fmt: skip


def test_empty_log() -> None:
    log = PredictionLog(":memory:")
    assert log.count() == 0 and log.latest(5) == []


def test_latest_returns_newest_first_with_python_types() -> None:
    log = PredictionLog(":memory:")
    for i in range(1, 6):
        log.log(record(i))
    rows = log.latest(3)
    assert [r["time_step"] for r in rows] == [5.0, 4.0, 3.0]
    first = rows[0]
    assert (type(first["id"]), type(first["alert"]), type(first["risk"])) == (int, bool, float)
    assert first["logged_at"][:4] == "2026" and first["model_version"] == "3"
    assert log.count() == 5


def test_missing_vehicle_id_is_stored_as_null() -> None:
    log = PredictionLog(":memory:")
    log.log(record(1, vehicle_id=None))
    log.log(record(2, vehicle_id=7))
    rows = log.latest(2)
    assert rows[0]["vehicle_id"] == 7 and rows[1]["vehicle_id"] is None


def test_ids_increase_and_the_file_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "log.duckdb"  # the folder is created
    first = PredictionLog(path)
    first.log(record(1))
    first.log(record(2))
    first.close()
    second = PredictionLog(path)
    second.log(record(3))
    assert second.count() == 3
    assert [r["id"] for r in second.latest(3)] == [3, 2, 1]


@pytest.mark.parametrize("n", [0, -1, MAX_LATEST + 1])
def test_latest_rejects_bad_limits(n: int) -> None:
    with pytest.raises(ValueError, match="n must be"):
        PredictionLog(":memory:").latest(n)


def test_concurrent_writes_are_all_stored() -> None:
    log = PredictionLog(":memory:")

    def work(offset: int) -> None:
        for i in range(25):
            log.log(record(offset + i))

    threads = [threading.Thread(target=work, args=(k * 100,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert log.count() == 200


def test_from_config_disabled_and_enabled(tmp_path: Path) -> None:
    assert PredictionLog.from_config({"api": {"prediction_log": None}}) is None
    assert PredictionLog.from_config({"api": {}}) is None
    log = PredictionLog.from_config({"api": {"prediction_log": str(tmp_path / "p.duckdb")}})
    assert log is not None and log.count() == 0


def test_the_log_has_no_column_for_the_raw_score() -> None:
    log = PredictionLog(":memory:")
    log.log(record(1))
    assert "score" not in log.latest(1)[0]
