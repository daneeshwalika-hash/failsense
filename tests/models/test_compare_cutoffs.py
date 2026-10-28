"""Tests for failsense.models.compare_cutoffs (fixed-evaluation folds for k).

Small hand-made frames only; the full SCANIA data is never loaded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.data.labels import BINARY_COL, VEHICLE_COL
from failsense.models.compare_cutoffs import assign_vehicle_folds, build_fixed_eval_folds

N_SPLITS = 2


@pytest.fixture
def eval_frame() -> pd.DataFrame:
    """One row per vehicle: 10 vehicles, 4 positives (enough for 2 stratified folds)."""
    return pd.DataFrame(
        {
            VEHICLE_COL: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            BINARY_COL: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        }
    )


@pytest.fixture
def train_vehicles(eval_frame: pd.DataFrame) -> pd.Series:
    """Training sample with 3 cut-offs per vehicle (like k = 3)."""
    return pd.Series(np.repeat(eval_frame[VEHICLE_COL].to_numpy(), 3), name=VEHICLE_COL)


def test_assign_vehicle_folds_one_fold_per_vehicle(eval_frame: pd.DataFrame) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)

    assert sorted(vehicle_fold.index) == sorted(eval_frame[VEHICLE_COL])
    assert set(vehicle_fold) == {0, 1}  # every fold is used, no -1 left


def test_assign_vehicle_folds_is_reproducible(eval_frame: pd.DataFrame) -> None:
    first = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    second = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    pd.testing.assert_series_equal(first, second)


def test_assign_vehicle_folds_rejects_duplicate_vehicles(eval_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([eval_frame, eval_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per vehicle"):
        assign_vehicle_folds(duplicated, N_SPLITS, seed=42)


def test_fixed_eval_folds_split_train_and_eval_parts(
    eval_frame: pd.DataFrame, train_vehicles: pd.Series
) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    eval_vehicles = eval_frame[VEHICLE_COL]
    offset = len(train_vehicles)

    folds = build_fixed_eval_folds(train_vehicles, eval_vehicles, vehicle_fold)

    assert len(folds) == N_SPLITS
    all_val_pos: list[int] = []
    for train_pos, val_pos in folds:
        # Training rows only from the training sample, validation rows only
        # from the evaluation sample.
        assert (train_pos < offset).all()
        assert (val_pos >= offset).all()
        # No vehicle on both sides of a fold.
        train_ids = set(train_vehicles.to_numpy()[train_pos])
        val_ids = set(eval_vehicles.to_numpy()[val_pos - offset])
        assert train_ids.isdisjoint(val_ids)
        # All cut-offs of every training vehicle are used (3 per vehicle).
        assert len(train_pos) == 3 * len(train_ids)
        all_val_pos.extend(val_pos.tolist())

    # Every evaluation row is validated exactly once over all folds.
    assert sorted(all_val_pos) == list(range(offset, offset + len(eval_vehicles)))


def test_fixed_eval_folds_vehicle_without_fold_raises(
    eval_frame: pd.DataFrame, train_vehicles: pd.Series
) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    unknown_vehicle = pd.concat([train_vehicles, pd.Series([999])], ignore_index=True)

    with pytest.raises(ValueError, match="no fold"):
        build_fixed_eval_folds(unknown_vehicle, eval_frame[VEHICLE_COL], vehicle_fold)
