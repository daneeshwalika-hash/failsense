"""Tests for failsense.rul.data (loader and data contract)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from failsense.rul.data import (
    COLUMNS,
    FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    load_train,
    rul_train_schema,
)

SCHEMA = rul_train_schema()


@pytest.fixture
def frame(make_cmapss_frame) -> pd.DataFrame:
    return make_cmapss_frame([4, 5, 3])


def test_column_names_in_one_place() -> None:
    assert len(COLUMNS) == 26
    assert COLUMNS[:2] == ("engine", "cycle")
    assert SETTING_COLUMNS == ("setting_1", "setting_2", "setting_3")
    assert SENSOR_COLUMNS[0] == "sensor_1" and SENSOR_COLUMNS[-1] == "sensor_21"
    assert len(FEATURE_COLUMNS) == 24
    assert not {"engine", "cycle", "rul"} & set(FEATURE_COLUMNS)


def test_loader_reads_whitespace_file_with_trailing_space(
    tmp_path: Path, frame: pd.DataFrame, write_cmapss_file
) -> None:
    path = write_cmapss_file(frame, tmp_path / "train_FD001.txt")
    assert path.read_text().splitlines()[0].endswith(" ")  # like the original file
    loaded = load_train(path)
    assert list(loaded.columns) == list(COLUMNS)
    assert len(loaded) == len(frame)
    assert str(loaded["engine"].dtype) == "int64" and str(loaded["cycle"].dtype) == "int64"
    assert all(str(loaded[c].dtype) == "float64" for c in FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(loaded, frame, check_exact=False, atol=1e-4)


def test_loader_missing_file_and_wrong_column_count(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_train(tmp_path / "nope.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("1 1 0.1 0.2\n1 2 0.1 0.2\n")
    with pytest.raises(ValueError, match="26 columns"):
        load_train(bad)


def test_valid_frame_passes(frame: pd.DataFrame) -> None:
    assert len(SCHEMA.validate(frame)) == len(frame)


def test_non_increasing_cycle_raises(frame: pd.DataFrame) -> None:
    bad = frame.copy()
    bad.loc[2, "cycle"] = 1  # engine 1: cycles 1, 2, 1, 4, 5
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(bad, lazy=True)


def test_duplicate_cycle_raises(frame: pd.DataFrame) -> None:
    bad = frame.copy()
    bad.loc[1, "cycle"] = 1
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(bad, lazy=True)


def test_interleaved_engines_with_increasing_cycles_pass(frame: pd.DataFrame) -> None:
    interleaved = frame.iloc[[0, 4, 1, 5, 2, 6, 3, 7, 8, 9, 10, 11]].reset_index(drop=True)
    assert len(SCHEMA.validate(interleaved)) == len(interleaved)


def test_nan_raises(frame: pd.DataFrame) -> None:
    bad = frame.copy()
    bad.loc[0, "sensor_3"] = float("nan")
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(bad, lazy=True)


def test_float_engine_and_empty_frame_raise(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(frame.astype({"engine": "float64"}), lazy=True)
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(frame.iloc[:0], lazy=True)


@pytest.mark.leakage
@pytest.mark.parametrize("label", ["rul", "max_cycle"])
def test_label_column_in_raw_table_is_rejected(frame: pd.DataFrame, label: str) -> None:
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(frame.assign(**{label: 1}), lazy=True)


def test_missing_sensor_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors):
        SCHEMA.validate(frame.drop(columns="sensor_7"), lazy=True)
