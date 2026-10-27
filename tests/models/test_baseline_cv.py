"""Tests for failsense.models.baseline_cv (helpers only, no real data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from failsense.data.labels import BINARY_COL
from failsense.models.baseline_cv import build_training_frame, parse_n_cutoffs, read_table

BOUNDS = (48.0, 24.0, 12.0, 6.0)


@pytest.fixture
def readouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 2, 2, 2],
            "time_step": [40.0, 60.0, 95.0, 40.0, 60.0, 95.0],
            "171_0": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def tte() -> pd.DataFrame:
    # vehicle 1 repaired at 100, vehicle 2 censored at 100
    return pd.DataFrame(
        {
            "vehicle_id": [1, 2],
            "length_of_study_time_step": [100.0, 100.0],
            "in_study_repair": [1, 0],
        }
    )


def test_training_frame_keeps_only_known_labels(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=None, seed=0)
    # vehicle 2 at t=60 (d=40) and t=95 (d=5) are unknown -> dropped
    assert len(frame) == 4
    assert frame[BINARY_COL].tolist() == [0, 1, 1, 0]


def test_training_frame_has_features_and_only_binary_label(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=1, seed=0)
    assert list(frame.columns) == ["vehicle_id", "time_step", "171_0", BINARY_COL]
    assert frame["vehicle_id"].value_counts().to_dict() == {1: 1, 2: 1}


def test_training_frame_rows_match_readouts(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=None, seed=0)
    pd.testing.assert_frame_equal(frame.drop(columns=[BINARY_COL]), readouts.loc[frame.index])


@pytest.mark.parametrize(("text", "expected"), [("5", 5), ("1", 1), ("all", None), ("ALL", None)])
def test_parse_n_cutoffs(text: str, expected: int | None) -> None:
    assert parse_n_cutoffs(text) == expected


def test_read_table_csv_and_parquet(tmp_path: Path, readouts) -> None:
    pytest.importorskip("pyarrow")  # Parquet engine; installed in the project env
    csv_path = tmp_path / "t.csv"
    parquet_path = tmp_path / "t.parquet"
    readouts.to_csv(csv_path, index=False)
    readouts.to_parquet(parquet_path, index=False)
    pd.testing.assert_frame_equal(read_table(csv_path), readouts)
    pd.testing.assert_frame_equal(read_table(parquet_path), readouts)
