"""Tests for failsense.data.labels."""

from __future__ import annotations

import pandas as pd
import pytest

from failsense.data.labels import (
    BINARY_COL,
    CLASS_COL,
    DISTANCE_COL,
    LABEL_COLUMNS,
    TIME_COL,
    VEHICLE_COL,
    assign_class,
    build_labels,
    class_bounds_from_config,
    validate_class_bounds,
)

BOUNDS = (48, 24, 12, 6)


@pytest.fixture
def tte() -> pd.DataFrame:
    """Vehicle 1 repaired at t=100, vehicle 2 censored at t=100."""
    return pd.DataFrame(
        {
            "vehicle_id": [1, 2],
            "length_of_study_time_step": [100.0, 100.0],
            "in_study_repair": [1, 0],
        }
    )


@pytest.fixture
def readouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "time_step": [40.0, 52.0, 76.0, 90.0, 100.0, 40.0, 60.0, 100.0],
            "171_0": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        }
    )


def _int8(values: list) -> pd.Series:
    return pd.Series(values, dtype="Int8")


# --- validate_class_bounds -------------------------------------------------


def test_validate_class_bounds_returns_floats() -> None:
    assert validate_class_bounds(BOUNDS) == (48.0, 24.0, 12.0, 6.0)


@pytest.mark.parametrize("bad", [(), (48, 0), (48, -6), (24, 48), (48, 48, 12)])
def test_validate_class_bounds_rejects_invalid(bad: tuple) -> None:
    with pytest.raises(ValueError):
        validate_class_bounds(bad)


def test_class_bounds_from_config() -> None:
    assert class_bounds_from_config({"labels": {"class_bounds": [48, 24, 12, 6]}}) == (
        48.0,
        24.0,
        12.0,
        6.0,
    )


def test_class_bounds_from_config_missing_key() -> None:
    with pytest.raises(KeyError, match="labels.class_bounds"):
        class_bounds_from_config({})


# --- assign_class: every border value ----------------------------------------


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (60.0, 0),
        (48.1, 0),
        (48.0, 1),
        (24.1, 1),
        (24.0, 2),
        (12.1, 2),
        (12.0, 3),
        (6.1, 3),
        (6.0, 4),
        (0.0, 4),
    ],
)
def test_assign_class_borders_are_upper_inclusive(d: float, expected: int) -> None:
    assert assign_class(pd.Series([d]), BOUNDS).iloc[0] == expected


def test_assign_class_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="missing"):
        assign_class(pd.Series([10.0, None]), BOUNDS)


def test_assign_class_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        assign_class(pd.Series([-0.1]), BOUNDS)


def test_assign_class_rejects_non_numeric() -> None:
    with pytest.raises(TypeError):
        assign_class(pd.Series(["10"]), BOUNDS)


# --- build_labels -----------------------------------------------------------


def test_build_labels_repaired_vehicle(readouts: pd.DataFrame, tte: pd.DataFrame) -> None:
    labels = build_labels(readouts, tte, BOUNDS)
    repaired = labels[labels[VEHICLE_COL] == 1].reset_index(drop=True)

    assert repaired[DISTANCE_COL].tolist() == [60.0, 48.0, 24.0, 10.0, 0.0]
    pd.testing.assert_series_equal(repaired[CLASS_COL], _int8([0, 1, 2, 3, 4]), check_names=False)
    pd.testing.assert_series_equal(repaired[BINARY_COL], _int8([0, 1, 1, 1, 1]), check_names=False)


@pytest.mark.leakage
def test_build_labels_censored_vehicle_near_end_is_unknown(
    readouts: pd.DataFrame, tte: pd.DataFrame
) -> None:
    labels = build_labels(readouts, tte, BOUNDS)
    censored = labels[labels[VEHICLE_COL] == 2].reset_index(drop=True)

    # d = 60 is safely class 0; d = 40 and d = 0 are unknown, never 0.
    pd.testing.assert_series_equal(censored[CLASS_COL], _int8([0, pd.NA, pd.NA]), check_names=False)
    pd.testing.assert_series_equal(
        censored[BINARY_COL], _int8([0, pd.NA, pd.NA]), check_names=False
    )


@pytest.mark.leakage
def test_build_labels_censored_exactly_at_horizon_is_unknown() -> None:
    tte = pd.DataFrame(
        {"vehicle_id": [7], "length_of_study_time_step": [148.0], "in_study_repair": [0]}
    )
    readouts = pd.DataFrame({"vehicle_id": [7], "time_step": [100.0]})  # d = 48
    assert build_labels(readouts, tte, BOUNDS)[CLASS_COL].isna().all()


def test_build_labels_keeps_index_and_order(readouts: pd.DataFrame, tte: pd.DataFrame) -> None:
    shuffled = readouts.sample(frac=1.0, random_state=0)
    shuffled.index = shuffled.index + 1000
    labels = build_labels(shuffled, tte, BOUNDS)

    assert labels.index.equals(shuffled.index)
    assert labels[TIME_COL].tolist() == shuffled[TIME_COL].tolist()


@pytest.mark.leakage
def test_build_labels_output_has_no_feature_columns(
    readouts: pd.DataFrame, tte: pd.DataFrame
) -> None:
    labels = build_labels(readouts, tte, BOUNDS)

    assert "171_0" not in labels.columns
    assert set(labels.columns) == {VEHICLE_COL, TIME_COL, DISTANCE_COL, CLASS_COL, BINARY_COL}
    assert {DISTANCE_COL, CLASS_COL, BINARY_COL} <= LABEL_COLUMNS


def test_build_labels_empty_readouts(tte: pd.DataFrame) -> None:
    empty = pd.DataFrame(
        {"vehicle_id": pd.Series([], dtype="int64"), "time_step": pd.Series([], dtype="float64")}
    )
    labels = build_labels(empty, tte, BOUNDS)

    assert labels.empty
    assert set(labels.columns) == {VEHICLE_COL, TIME_COL, DISTANCE_COL, CLASS_COL, BINARY_COL}


def test_build_labels_missing_column(tte: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_labels(pd.DataFrame({"vehicle_id": [1]}), tte, BOUNDS)


def test_build_labels_wrong_dtype(tte: pd.DataFrame) -> None:
    readouts = pd.DataFrame({"vehicle_id": [1], "time_step": ["40"]})
    with pytest.raises(TypeError, match="non-numeric"):
        build_labels(readouts, tte, BOUNDS)


def test_build_labels_vehicle_without_tte(tte: pd.DataFrame) -> None:
    readouts = pd.DataFrame({"vehicle_id": [99], "time_step": [10.0]})
    with pytest.raises(ValueError, match="no tte row"):
        build_labels(readouts, tte, BOUNDS)


def test_build_labels_duplicate_tte_vehicle(readouts: pd.DataFrame, tte: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="one row per vehicle_id"):
        build_labels(readouts, pd.concat([tte, tte]), BOUNDS)


def test_build_labels_invalid_repair_flag(readouts: pd.DataFrame, tte: pd.DataFrame) -> None:
    tte = tte.assign(in_study_repair=[1, 2])
    with pytest.raises(ValueError, match="0 or 1"):
        build_labels(readouts, tte, BOUNDS)


def test_build_labels_readout_after_study_end(tte: pd.DataFrame) -> None:
    readouts = pd.DataFrame({"vehicle_id": [1], "time_step": [100.5]})
    with pytest.raises(ValueError, match="after the end of observation"):
        build_labels(readouts, tte, BOUNDS)
