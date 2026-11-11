"""Cross-cutting data-leakage tests (``pytest -m leakage`` runs every leakage test).

The rules are in ``docs/problem_framing.md`` section 4 ("Data leakage").
Most leakage tests live next to the code they protect and carry the same marker;
the tests here check rules that span several modules, end to end, on a tiny
SCANIA-shaped dataset (see ``tests/conftest.py``) - never on the raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.data.labels import LABEL_COLUMNS, VEHICLE_COL
from failsense.data.schema import (
    counter_columns,
    histogram_columns,
    histogram_groups,
    validate_scania,
)
from failsense.data.split import make_group_folds
from failsense.features.build import attach_specifications, engineered_columns
from failsense.models import baseline_cv
from failsense.models.baseline_cv import build_training_frame, load_training_frame
from failsense.models.pipeline import FORBIDDEN_FEATURES, split_feature_columns
from failsense.models.train import run as train_run

pytestmark = pytest.mark.leakage

BOUNDS = (48.0, 24.0, 12.0, 6.0)


def _frame(frames: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    return build_training_frame(
        frames["readouts"],
        frames["tte"],
        BOUNDS,
        n_per_vehicle=None,
        seed=42,
        counter_cols=counter_columns(),
        hist_groups=histogram_groups(),
        **kwargs,
    )


# --- 1. no truck in more than one official split -------------------------------------
def test_validate_scania_accepts_disjoint_splits(make_mini_scania) -> None:
    validate_scania(make_mini_scania(), root=None)


def test_validate_scania_rejects_a_truck_in_two_splits(make_mini_scania) -> None:
    config = make_mini_scania(shared_vehicle=5)  # truck 5 is in train and validation
    with pytest.raises(ValueError, match="group leakage between splits"):
        validate_scania(config, root=None)


# --- 2. features at time t use only readouts <= t of the same truck -----------------
def test_later_readouts_never_change_earlier_features(mini_frames) -> None:
    """End to end (labels -> cut-offs -> deltas -> histogram features)."""
    before = _frame(mini_frames)

    changed = {k: v.copy() for k, v in mini_frames.items()}
    readouts = changed["readouts"]
    last = readouts[readouts[VEHICLE_COL] == 0].index[-1]  # last readout of truck 0
    value_columns = counter_columns() + histogram_columns()
    readouts.loc[last, value_columns] = readouts.loc[last, value_columns] * 1000 + 7
    after = _frame(changed)

    earlier = before.index.difference([last])
    assert len(before) == len(after)
    pd.testing.assert_frame_equal(before.loc[earlier], after.loc[earlier])


def test_features_of_one_truck_never_depend_on_another_truck(mini_frames) -> None:
    before = _frame(mini_frames)
    changed = {k: v.copy() for k, v in mini_frames.items()}
    other = changed["readouts"][VEHICLE_COL] == 1
    changed["readouts"].loc[other, counter_columns()] += 1e6
    after = _frame(changed)
    mine = before[VEHICLE_COL] != 1
    pd.testing.assert_frame_equal(before[mine], after[mine])


# --- 3. labels and ids never reach the model -------------------------------------------
def test_model_input_never_contains_labels_or_vehicle_id(mini_frames) -> None:
    frame = attach_specifications(_frame(mini_frames), mini_frames["specs"])
    # Worst case: every label column and the id are present in the frame.
    frame = frame.assign(**{c: 1 for c in LABEL_COLUMNS if c not in frame.columns})
    cols = split_feature_columns(frame)
    used = set(cols.raw) | set(cols.engineered) | set(cols.log1p) | set(cols.categorical)
    assert not used & FORBIDDEN_FEATURES
    assert VEHICLE_COL in frame.columns  # it was there, and was still excluded
    assert not set(engineered_columns(cols.raw))  # engineered columns are never "raw"


# --- 4. cross validation never puts a truck on both sides ----------------------------
def test_cv_folds_of_the_training_frame_share_no_truck(mini_frames) -> None:
    frame = _frame(mini_frames)
    assert frame[VEHICLE_COL].value_counts().max() > 1  # several cut-offs per truck
    for train_idx, val_idx in make_group_folds(frame, n_splits=3, seed=42):
        train_trucks = set(frame[VEHICLE_COL].iloc[train_idx])
        val_trucks = set(frame[VEHICLE_COL].iloc[val_idx])
        assert train_trucks.isdisjoint(val_trucks)


# --- 5. validation and test data are never used for fitting --------------------------
def test_training_entry_points_read_only_the_train_split(
    make_mini_scania, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    original = baseline_cv.table_path

    def spy(config, split, table):
        requested.append(split)
        return original(config, split, table)

    monkeypatch.setattr(baseline_cv, "table_path", spy)
    config = make_mini_scania()
    load_training_frame(config, 5)
    train_run(config, track=False)
    assert requested and set(requested) == {"train"}


def test_training_frame_has_no_rows_of_other_splits(make_mini_scania) -> None:
    config = make_mini_scania()
    frame = load_training_frame(config, 5)
    assert set(frame[VEHICLE_COL]) <= set(range(40))  # validation/test trucks are 100+/200+
    assert np.isfinite(frame["time_step"]).all()
