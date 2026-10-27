"""Tests for failsense.data.split (group leakage protection)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.data.split import (
    assert_no_group_overlap,
    cv_params_from_config,
    make_group_folds,
)


@pytest.fixture
def train_rows() -> pd.DataFrame:
    """40 vehicles x 3 rows; 8 vehicles (20 %) are positive in every row.

    Small enough to read, large enough for 5 stratified folds.
    """
    vehicle_ids = np.repeat(np.arange(40), 3)
    positive = (vehicle_ids % 5 == 0).astype(int)  # vehicles 0, 5, 10, ... are positive
    return pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "time_step": np.tile([10.0, 20.0, 30.0], 40),
            "repair_within_horizon": pd.array(positive, dtype="Int8"),
        }
    )


# --- assert_no_group_overlap -------------------------------------------------


def test_no_overlap_passes() -> None:
    # Duplicates *inside* a split are fine: one truck has many readouts.
    assert_no_group_overlap({"train": [1, 1, 2, 3], "validation": [4, 5], "test": [6, 6]})


def test_overlap_raises_and_names_the_pair() -> None:
    with pytest.raises(ValueError, match="train/test: 1 shared") as info:
        assert_no_group_overlap({"train": [1, 2, 3], "validation": [4], "test": [3, 7]})
    assert "validation" not in str(info.value)  # only the overlapping pair is reported


def test_overlap_reports_every_pair() -> None:
    with pytest.raises(ValueError) as info:
        assert_no_group_overlap({"a": [1, 2], "b": [2, 3], "c": [3, 1]})
    message = str(info.value)
    assert "a/b" in message and "a/c" in message and "b/c" in message


def test_overlap_accepts_pandas_series() -> None:
    with pytest.raises(ValueError, match="group leakage"):
        assert_no_group_overlap({"train": pd.Series([1, 2]), "validation": pd.Series([2])})


def test_needs_at_least_two_splits() -> None:
    with pytest.raises(ValueError, match="at least two"):
        assert_no_group_overlap({"train": [1, 2]})


def test_empty_splits_do_not_overlap() -> None:
    assert_no_group_overlap({"train": [], "validation": []})


# --- make_group_folds --------------------------------------------------------


def test_folds_never_share_a_vehicle(train_rows: pd.DataFrame) -> None:
    folds = make_group_folds(train_rows, n_splits=5, seed=42)
    vehicles = train_rows["vehicle_id"].to_numpy()

    assert len(folds) == 5
    for train_pos, val_pos in folds:
        assert set(vehicles[train_pos]).isdisjoint(vehicles[val_pos])


def test_every_row_is_validated_exactly_once(train_rows: pd.DataFrame) -> None:
    folds = make_group_folds(train_rows, n_splits=5, seed=42)
    all_val = np.concatenate([val_pos for _, val_pos in folds])
    assert sorted(all_val.tolist()) == list(range(len(train_rows)))


def test_every_fold_has_positives(train_rows: pd.DataFrame) -> None:
    folds = make_group_folds(train_rows, n_splits=5, seed=42)
    y = train_rows["repair_within_horizon"].to_numpy(dtype=int)
    assert all(y[val_pos].sum() > 0 for _, val_pos in folds)


def test_same_seed_gives_same_folds(train_rows: pd.DataFrame) -> None:
    first = make_group_folds(train_rows, n_splits=5, seed=42)
    second = make_group_folds(train_rows, n_splits=5, seed=42)
    for (tr_a, va_a), (tr_b, va_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)


def test_folds_work_with_custom_group_column(train_rows: pd.DataFrame) -> None:
    engines = train_rows.rename(columns={"vehicle_id": "unit"})  # C-MAPSS naming
    folds = make_group_folds(engines, n_splits=5, seed=0, group_col="unit")
    assert len(folds) == 5


def test_unknown_labels_raise(train_rows: pd.DataFrame) -> None:
    train_rows.loc[0, "repair_within_horizon"] = pd.NA
    with pytest.raises(ValueError, match="missing values"):
        make_group_folds(train_rows, n_splits=5, seed=42)


def test_missing_column_raises(train_rows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        make_group_folds(train_rows.drop(columns=["vehicle_id"]), n_splits=5, seed=42)


def test_too_few_vehicles_raises(train_rows: pd.DataFrame) -> None:
    few = train_rows[train_rows["vehicle_id"] < 3]  # only 3 vehicles
    with pytest.raises(ValueError, match="at least 5 groups"):
        make_group_folds(few, n_splits=5, seed=42)


# --- config -------------------------------------------------------------------


def test_cv_params_from_config() -> None:
    assert cv_params_from_config({"cv": {"n_splits": 5, "seed": 42}}) == (5, 42)


@pytest.mark.parametrize(
    ("cv", "error"),
    [
        ({"seed": 42}, KeyError),
        ({"n_splits": 1, "seed": 42}, ValueError),
        ({"n_splits": "5", "seed": 42}, TypeError),
        ({"n_splits": 5, "seed": True}, TypeError),
    ],
)
def test_cv_params_invalid(cv: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        cv_params_from_config({"cv": cv})
