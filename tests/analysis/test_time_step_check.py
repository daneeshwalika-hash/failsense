"""Tests for failsense.analysis.time_step_check (small fixtures, no raw data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.analysis.time_step_check import (
    QUANTILES,
    ks_distance,
    last_readout_time_step,
    plot_overlay,
    quantile_table,
)


def test_last_readout_is_the_max_time_step_per_truck() -> None:
    readouts = pd.DataFrame(
        {"vehicle_id": [2, 1, 1, 2, 1], "time_step": [5.0, 10.0, 30.0, 15.0, 20.0]}
    )
    last = last_readout_time_step(readouts)
    assert last.to_dict() == {1: 30.0, 2: 15.0}  # independent of row order


def test_last_readout_single_row_and_empty_and_missing() -> None:
    one = pd.DataFrame({"vehicle_id": [1], "time_step": [7.0]})
    assert last_readout_time_step(one).tolist() == [7.0]
    empty = pd.DataFrame({"vehicle_id": pd.Series([], dtype=int), "time_step": []})
    assert last_readout_time_step(empty).empty
    with pytest.raises(ValueError, match="missing"):
        last_readout_time_step(one.drop(columns="time_step"))


def test_quantile_table_values() -> None:
    table = quantile_table({"a": np.arange(101.0), "b": [5.0]})
    a = table[table["sample"] == "a"].iloc[0]
    assert a["n"] == 101 and a["mean"] == 50.0
    assert a["q50"] == 50.0 and a["q05"] == 5.0 and a["q99"] == 99.0
    assert {f"q{int(q * 100):02d}" for q in QUANTILES} <= set(table.columns)
    assert table[table["sample"] == "b"].iloc[0]["q50"] == 5.0  # single value


def test_quantile_table_rejects_empty_and_nan() -> None:
    with pytest.raises(ValueError, match="empty"):
        quantile_table({"a": []})
    with pytest.raises(ValueError, match="NaN"):
        quantile_table({"a": [1.0, np.nan]})


def test_ks_is_zero_for_identical_and_one_for_separate_samples() -> None:
    same = np.arange(50.0)
    assert ks_distance(same, same)[0] == 0.0
    stat, p = ks_distance(np.arange(50.0), np.arange(100.0, 150.0))
    assert stat == 1.0 and p < 0.001


def test_ks_detects_a_shift() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(1, 1, 500)
    assert ks_distance(a, b)[0] > 0.3
    assert ks_distance(a, rng.normal(0, 1, 500))[0] < 0.15


def test_ks_rejects_empty_and_nan() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ks_distance([], [1.0])
    with pytest.raises(ValueError, match="NaN"):
        ks_distance([1.0, np.nan], [1.0])


def test_overlay_figure_is_written(tmp_path: Path) -> None:
    a = pd.Series(np.arange(1.0, 50.0))
    b = pd.Series(np.arange(10.0, 80.0))
    path = plot_overlay(a, b, "5", tmp_path / "sub" / "t.png")
    assert path.exists() and path.stat().st_size > 0
