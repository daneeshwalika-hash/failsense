"""Tests for failsense.rul.baseline (tiny engines, no MLflow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from failsense.rul.baseline import (
    best_by_lowest_mean,
    build_rul_pipeline,
    constant_columns,
    make_engine_folds,
    rmse_improvement,
    run,
    select_features,
    zero_in_all_folds,
)
from failsense.rul.data import FEATURE_COLUMNS


@pytest.fixture
def frame(make_cmapss_frame) -> pd.DataFrame:
    return make_cmapss_frame([30, 35, 40, 45, 50, 32, 38, 42])


@pytest.mark.leakage
def test_no_engine_in_two_folds_and_every_row_validated_once(frame: pd.DataFrame) -> None:
    folds = make_engine_folds(frame, 5)
    assert len(folds) == 5
    engines = frame["engine"].to_numpy()
    seen: list[int] = []
    for train_idx, val_idx in folds:
        assert set(engines[train_idx]).isdisjoint(engines[val_idx])
        seen.extend(val_idx.tolist())
    assert sorted(seen) == list(range(len(frame)))


@pytest.mark.leakage
def test_all_rows_of_an_engine_stay_together(frame: pd.DataFrame) -> None:
    for _, val_idx in make_engine_folds(frame, 4):
        val_engines = set(frame["engine"].iloc[val_idx])
        assert len(val_idx) == int(frame["engine"].isin(val_engines).sum())


def test_folds_need_enough_engines_and_the_engine_column(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 5 engines"):
        make_engine_folds(frame[frame["engine"] <= 3], 5)
    with pytest.raises(ValueError, match="engine"):
        make_engine_folds(frame.drop(columns="engine"), 5)


@pytest.mark.leakage
def test_labels_and_engine_id_never_reach_the_model(frame: pd.DataFrame) -> None:
    dirty = frame.assign(rul=1, max_cycle=2)
    for include_cycle in (False, True):
        columns = select_features(dirty, include_cycle=include_cycle)
        assert not {"engine", "rul", "max_cycle"} & set(columns)
        assert ("cycle" in columns) == include_cycle
    assert select_features(frame) == list(FEATURE_COLUMNS)


def test_select_features_missing_column_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        select_features(frame.drop(columns="sensor_4"))


def test_constant_columns_found(frame: pd.DataFrame) -> None:
    constant = constant_columns(frame, list(FEATURE_COLUMNS))
    assert {"setting_3", "sensor_1", "sensor_5"} <= set(constant)
    assert "sensor_2" not in constant
    assert constant_columns(frame.iloc[:1], ["sensor_2"]) == ["sensor_2"]  # single row


def test_zero_in_all_folds_finds_constant_columns_under_lasso(frame: pd.DataFrame) -> None:
    y = np.arange(len(frame), dtype=float)
    folds = make_engine_folds(frame, 4)
    fitted = [
        build_rul_pipeline(Lasso(alpha=0.5, max_iter=5000)).fit(
            frame[list(FEATURE_COLUMNS)].iloc[t], y[t]
        )
        for t, _ in folds
    ]
    zeros = zero_in_all_folds(fitted, list(FEATURE_COLUMNS))
    assert {"sensor_1", "sensor_5", "setting_3"} <= set(zeros)  # constant -> coefficient 0
    with pytest.raises(ValueError, match="at least one"):
        zero_in_all_folds([], list(FEATURE_COLUMNS))


def test_best_by_lowest_mean_and_improvement_sign() -> None:
    assert (
        best_by_lowest_mean(
            {"a": np.array([3.0, 3.0]), "b": np.array([2.0, 5.0]), "c": np.array([2.5, 2.4])}
        )
        == "c"
    )
    with pytest.raises(ValueError, match="no scores"):
        best_by_lowest_mean({})
    better = rmse_improvement(np.array([10.0, 11.0]), np.array([20.0, 21.0]))
    assert better["mean_diff"] == 10.0 and better["wins"] == 2  # lower RMSE = improvement
    worse = rmse_improvement(np.array([30.0, 31.0]), np.array([20.0, 21.0]))
    assert worse["mean_diff"] == -10.0 and worse["wins"] == 0


@pytest.mark.leakage
def test_run_never_reads_the_test_or_rul_files(make_mini_cmapss, tmp_path: Path) -> None:
    config = make_mini_cmapss()
    assert not Path(config["datasets"]["cmapss"]["test"]).exists()
    assert not Path(config["datasets"]["cmapss"]["rul"]).exists()
    summary = run(config, track=False)  # would raise FileNotFoundError if it tried to read them
    names = [n for n in summary if not n.startswith("_")]
    assert "dummy_mean" in names
    assert {f"ridge_a{a:g}" for a in (0.1, 1.0, 10.0, 100.0)} <= set(names)
    assert {f"lasso_a{a:g}" for a in (0.001, 0.01, 0.1, 1.0)} <= set(names)
    assert any(n.endswith("_cycle") for n in names)
    assert (tmp_path / "reports" / "rul_baseline_fd001.csv").exists()


def test_run_results_are_sensible(make_mini_cmapss) -> None:
    summary = run(make_mini_cmapss(), track=False)
    best_ridge = summary["_report"]["best_ridge"]
    # Sensors drift with wear, so a linear model must beat the mean predictor.
    assert summary[best_ridge]["rmse"] < summary["dummy_mean"]["rmse"]
    assert summary["dummy_mean"]["mae"] <= summary["dummy_mean"]["rmse"]  # MAE <= RMSE always
    assert {"setting_3", "sensor_1", "sensor_5"} <= set(summary["_report"]["constant_columns"])
    for name, s in summary.items():
        if not name.startswith("_"):
            assert s["rmse"] >= 0 and s["rmse_train"] >= 0
