"""Tests for failsense.models.validation_data (last readout per validation truck)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import mini_readouts, mini_specs
from failsense.config import load_config
from failsense.data.labels import BINARY_COL, CLASS_COL, TIME_COL, VEHICLE_COL
from failsense.data.schema import counter_columns, histogram_groups
from failsense.features.build import DELTA_SUFFIX, RATE_SUFFIX, engineered_columns
from failsense.models.baseline_cv import frame_from_sampled
from failsense.models.pipeline import assert_no_forbidden_features, split_feature_columns
from failsense.models.validation_data import (
    build_last_readout_frame,
    class_counts,
    registry_input_columns,
    select_model_inputs,
)

IDS = [100, 101, 102, 103]
LABELS = pd.DataFrame({VEHICLE_COL: IDS, "class_label": [0, 3, 0, 4]})
TOP = 0.2


def build(readouts: pd.DataFrame, labels: pd.DataFrame = LABELS) -> pd.DataFrame:
    return build_last_readout_frame(
        readouts, mini_specs(IDS), labels, counter_columns(), histogram_groups(), TOP
    )


def test_one_row_per_truck_at_the_last_readout() -> None:
    frame = build(mini_readouts(IDS))

    assert list(frame[VEHICLE_COL]) == IDS
    assert (frame[TIME_COL] == 110.0).all()  # 12 readouts: 0, 10, ..., 110
    assert list(frame[CLASS_COL]) == [0, 3, 0, 4]
    assert list(frame[BINARY_COL]) == [0, 1, 0, 1]
    assert class_counts(frame) == {0: 2, 1: 0, 2: 0, 3: 1, 4: 1}


def test_delta_uses_the_previous_readout_even_if_rows_are_shuffled() -> None:
    readouts = mini_readouts(IDS)
    shuffled = readouts.sample(frac=1.0, random_state=1)  # input order must not matter
    counter = counter_columns()[0]

    frame = build(shuffled)

    for vehicle in IDS:
        rows = readouts[readouts[VEHICLE_COL] == vehicle].sort_values(TIME_COL)
        expected_delta = rows[counter].iloc[-1] - rows[counter].iloc[-2]
        got = frame.loc[frame[VEHICLE_COL] == vehicle, counter + DELTA_SUFFIX].iloc[0]
        assert got == pytest.approx(expected_delta)
        got_rate = frame.loc[frame[VEHICLE_COL] == vehicle, counter + RATE_SUFFIX].iloc[0]
        assert got_rate == pytest.approx(expected_delta / 10.0)


def test_features_use_no_later_readouts() -> None:
    """Features of a cut-off row are the same whether later readouts exist or not."""
    full = mini_readouts(IDS)
    kept = full[full[TIME_COL] <= 70.0]  # the truck "ends" at t = 70

    short = build(kept)

    # Reference: the same cut-off row (t = 70) inside the FULL history, built by the
    # training-side function. Later readouts exist there but must not change it.
    cutoffs = full[full[TIME_COL] == 70.0]
    sampled = cutoffs.assign(**{BINARY_COL: 0})[[BINARY_COL]]
    reference = frame_from_sampled(full, sampled, counter_columns(), histogram_groups(), TOP)
    reference = reference.sort_values(VEHICLE_COL).reset_index(drop=True)
    cols = [c for c in short.columns if c in reference.columns and c != BINARY_COL]
    assert (short[TIME_COL] == 70.0).all()
    pd.testing.assert_frame_equal(short[cols], reference[cols], check_exact=True)


def test_changing_the_last_readout_does_not_change_earlier_information() -> None:
    """The last row depends on the earlier readouts, never the reverse (no look-ahead)."""
    readouts = mini_readouts(IDS)
    changed = readouts.copy()
    last = changed[TIME_COL] == 110.0
    changed.loc[last, counter_columns()[0]] += 1000.0

    a, b = build(readouts), build(changed)

    # Row at t = 110 differs only through its own value; a truncated build is identical.
    drop = readouts[TIME_COL] <= 100.0
    pd.testing.assert_frame_equal(build(readouts[drop]), build(changed[drop]))
    assert not a[counter_columns()[0]].equals(b[counter_columns()[0]])


def test_model_input_columns_are_clean_and_ordered() -> None:
    frame = build(mini_readouts(IDS))
    cols = split_feature_columns(frame)
    columns = cols.raw + cols.engineered + cols.categorical

    assert_no_forbidden_features(columns)  # no vehicle_id, no label columns
    selected = select_model_inputs(frame, columns)
    assert list(selected.columns) == columns
    assert len(engineered_columns(columns)) == 125


def test_registered_input_columns_can_be_built() -> None:
    columns = registry_input_columns(load_config())
    frame = build(mini_readouts(IDS))

    assert len(columns) == 239
    assert_no_forbidden_features(columns)
    assert list(select_model_inputs(frame, columns).columns) == columns


def test_registry_input_columns_reads_the_reports_folder(tmp_path: Path) -> None:
    (tmp_path / "registry_q1.json").write_text(json.dumps({"input_columns": ["a", "b"]}))

    assert registry_input_columns({"paths": {"reports": str(tmp_path)}}) == ["a", "b"]


def test_registry_input_columns_rejects_duplicates(tmp_path: Path) -> None:
    (tmp_path / "registry_q1.json").write_text(json.dumps({"input_columns": ["a", "a"]}))

    with pytest.raises(ValueError, match="duplicate"):
        registry_input_columns({"paths": {"reports": str(tmp_path)}})


def test_select_model_inputs_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        select_model_inputs(build(mini_readouts(IDS)), ["time_step", "no_such_column"])


def test_truck_mismatch_between_readouts_and_labels_raises() -> None:
    with pytest.raises(ValueError, match="differ"):
        build(mini_readouts(IDS), LABELS.iloc[:3])


def test_duplicate_label_rows_raise() -> None:
    with pytest.raises(ValueError, match="more than one"):
        build(mini_readouts(IDS), pd.concat([LABELS, LABELS.iloc[:1]]))


@pytest.mark.parametrize("bad", [5, -1, 2.5])
def test_label_outside_0_to_4_raises(bad: float) -> None:
    labels = LABELS.copy()
    labels["class_label"] = labels["class_label"].astype(float)
    labels.loc[0, "class_label"] = bad

    with pytest.raises(ValueError, match="0..4"):
        build(mini_readouts(IDS), labels)


def test_missing_columns_and_empty_readouts_raise() -> None:
    with pytest.raises(ValueError, match="missing"):
        build(mini_readouts(IDS).drop(columns=[TIME_COL]))
    with pytest.raises(ValueError, match="missing"):
        build(mini_readouts(IDS), LABELS.drop(columns=["class_label"]))
    with pytest.raises(ValueError, match="empty"):
        build(mini_readouts(IDS).iloc[:0])


def test_duplicate_time_step_within_a_truck_raises() -> None:
    readouts = mini_readouts(IDS)
    readouts.loc[readouts.index[1], TIME_COL] = readouts.loc[readouts.index[0], TIME_COL]

    with pytest.raises(ValueError, match="strictly increasing"):
        build(readouts)


def test_class_counts_always_has_five_classes() -> None:
    frame = pd.DataFrame({CLASS_COL: [0, 0, 4]})

    assert class_counts(frame) == {0: 2, 1: 0, 2: 0, 3: 0, 4: 1}
    assert np.array_equal(list(class_counts(frame).keys()), [0, 1, 2, 3, 4])
