"""RUL time-series features and models (mini C-MAPSS-shaped fixtures, no raw data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.rul import timeseries as ts
from failsense.rul.data import CYCLE_COL, ENGINE_COL


def tiny_frame() -> pd.DataFrame:
    """Two engines; sensor_2 grows 1, 2, 3, 4 (engine 1) and 10, 20, 30 (engine 2)."""
    return pd.DataFrame(
        {
            ENGINE_COL: [1, 1, 1, 1, 2, 2, 2],
            CYCLE_COL: [1, 2, 3, 4, 1, 2, 3],
            "sensor_2": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
        }
    )


# --- rolling features -----------------------------------------------------------------------
def test_rolling_mean_and_std_by_hand() -> None:
    out = ts.add_rolling_features(tiny_frame(), ["sensor_2"], [3])
    mean, std = out["sensor_2__roll3_mean"], out["sensor_2__roll3_std"]
    assert mean.tolist() == pytest.approx([1.0, 1.5, 2.0, 3.0, 10.0, 15.0, 20.0])
    assert std.iloc[0] == 0.0  # a single reading: nothing varied yet
    assert std.iloc[1] == pytest.approx(np.std([1, 2], ddof=1))
    assert std.iloc[3] == pytest.approx(1.0)  # window [2, 3, 4]
    assert std.iloc[4] == 0.0  # the first cycle of engine 2 starts fresh


def test_features_use_only_the_past_of_the_same_engine() -> None:
    base = tiny_frame()
    changed = base.copy()
    changed.loc[3, "sensor_2"] = 999.0  # the LAST cycle of engine 1
    changed.loc[4:, "sensor_2"] = -50.0  # all of engine 2
    a = ts.add_rolling_features(base, ["sensor_2"], [3])
    b = ts.add_rolling_features(changed, ["sensor_2"], [3])
    new = [c for c in a.columns if "__roll" in c]
    # cycles 1-3 of engine 1 do not see cycle 4 (the future) nor engine 2
    pd.testing.assert_frame_equal(a.loc[:2, new], b.loc[:2, new])
    assert not a.loc[3, new].equals(b.loc[3, new])  # sanity: the change is visible where allowed


def test_original_row_order_and_input_are_kept() -> None:
    frame = tiny_frame().sample(frac=1, random_state=0)
    before = frame.copy()
    out = ts.add_rolling_features(frame, ["sensor_2"], [2, 3])
    pd.testing.assert_frame_equal(frame, before)  # not modified
    assert list(out.index) == list(frame.index)
    assert list(out.columns[:3]) == [ENGINE_COL, CYCLE_COL, "sensor_2"]
    expected = ts.add_rolling_features(tiny_frame(), ["sensor_2"], [2, 3]).sort_index()
    pd.testing.assert_frame_equal(out.sort_index(), expected)
    assert {"sensor_2__roll2_mean", "sensor_2__roll2_std", "sensor_2__roll3_mean",
            "sensor_2__roll3_std"} <= set(out.columns)  # fmt: skip


@pytest.mark.parametrize("windows", [[], [1], [0, 5], [-3]])
def test_invalid_windows_raise(windows: list[int]) -> None:
    with pytest.raises(ValueError, match="windows"):
        ts.add_rolling_features(tiny_frame(), ["sensor_2"], windows)


def test_missing_columns_and_bad_cycles_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        ts.add_rolling_features(tiny_frame(), ["sensor_99"], [3])
    with pytest.raises(ValueError, match="missing columns"):
        ts.add_rolling_features(tiny_frame().drop(columns=ENGINE_COL), ["sensor_2"], [3])
    duplicate = pd.concat([tiny_frame(), tiny_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        ts.add_rolling_features(duplicate, ["sensor_2"], [3])


def test_empty_frame_gets_no_rows_but_the_columns() -> None:
    out = ts.add_rolling_features(tiny_frame().iloc[0:0], ["sensor_2"], [3])
    assert len(out) == 0 and "sensor_2__roll3_mean" in out.columns


# --- feature sets and pipeline --------------------------------------------------------------
def test_informative_sensors_drop_the_constant_ones(make_cmapss_frame) -> None:  # type: ignore[no-untyped-def]
    frame = make_cmapss_frame([30, 35, 40])
    informative = ts.informative_sensors(frame)
    assert (
        "sensor_1" not in informative and "sensor_5" not in informative
    )  # constant in the fixture
    assert "sensor_2" in informative and len(informative) == 19


def test_feature_sets_contain_no_label_cycle_or_engine(make_cmapss_frame) -> None:  # type: ignore[no-untyped-def]
    frame, sets = ts.feature_sets(make_cmapss_frame([30, 35, 40]), [5, 15])
    for columns in sets.values():
        assert not {ENGINE_COL, CYCLE_COL} & set(columns)
        assert not any("rul" in c.lower() or c == "max_cycle" for c in columns)
        assert set(columns) <= set(frame.columns)
    assert len(sets["raw"]) == 24
    assert len(sets["rolling"]) == 24 + 19 * 2 * 2  # informative sensors x (mean, std) x 2 windows


def test_pipeline_removes_constant_columns_fitted_on_the_training_rows(
    make_cmapss_frame,  # type: ignore[no-untyped-def]
) -> None:
    from failsense.rul.data import FEATURE_COLUMNS

    frame = make_cmapss_frame([30, 35, 40, 45])
    pipe = ts.make_pipeline(ts.Ridge(alpha=1.0)).fit(
        frame[list(FEATURE_COLUMNS)], np.arange(len(frame))
    )
    kept = pipe.named_steps["constant"].get_support().sum()
    assert kept == 24 - 3  # sensors 1 and 5 and setting_3 are constant in the fixture
    assert list(pipe.named_steps) == ["constant", "scale", "model"]


def test_lightgbm_model_is_configured_and_reproducible() -> None:
    model = ts.lightgbm_model({"n_estimators": 5, "num_leaves": 4}, seed=3)
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)))
    y = x[0] * 2 + 1
    a = model.fit(x, y).predict(x)
    b = ts.lightgbm_model({"n_estimators": 5, "num_leaves": 4}, seed=3).fit(x, y).predict(x)
    np.testing.assert_array_equal(a, b)


# --- the run --------------------------------------------------------------------------------
def test_run_on_mini_data_reports_all_models_and_never_reads_the_test_file(
    make_mini_cmapss,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    config = make_mini_cmapss()
    config["rul"].update(
        {
            "rolling_windows": [3, 5], "ridge_alpha": 1.0,
            "lightgbm": {"n_estimators": 10, "num_leaves": 4, "min_child_samples": 5},
        }
    )  # fmt: skip
    summary = ts.run(config, track=False)  # test/RUL paths in the fixture do not exist
    assert set(summary) == {"dummy_mean", "ridge_raw", "ridge_rolling", "lightgbm_raw",
                            "lightgbm_rolling"}  # fmt: skip
    assert summary["ridge_rolling"]["n_features"] > summary["ridge_raw"]["n_features"] == 24
    assert summary["lightgbm_rolling"]["paired"]["lightgbm_raw"]["n_folds"] == 5
    assert all(summary[m]["rmse"] > 0 for m in summary)
    report = pd.read_csv(Path(config["paths"]["reports"]) / ts.REPORT_FILE)
    assert len(report) == 5 and "rmse" in report.columns
