"""Tests for failsense.analysis.class_mix (hand-computed fixtures, no raw data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.analysis import class_mix as cm
from failsense.data.labels import CLASS_COL, DISTANCE_COL, TIME_COL, VEHICLE_COL

BOUNDS = (48, 24, 12, 6)


def labels_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Truck 1 repaired at 100 (readouts every 10), truck 2 repaired at 60, truck 3 censored."""
    rows = []
    for vehicle, length, step in ((1, 100, 10), (2, 60, 10), (3, 100, 10)):
        for t in range(step, length + 1, step):
            d = float(length - t)
            cls = int(sum(d <= b for b in BOUNDS))
            rows.append({VEHICLE_COL: vehicle, TIME_COL: float(t), DISTANCE_COL: d, CLASS_COL: cls})
    frame = pd.DataFrame(rows)
    frame[CLASS_COL] = frame[CLASS_COL].astype("Int8")
    frame.loc[(frame[VEHICLE_COL] == 3) & (frame[DISTANCE_COL] <= 48), CLASS_COL] = pd.NA
    return frame, pd.Series({1: 1, 2: 1, 3: 0})


def test_class_widths() -> None:
    assert cm.class_widths(BOUNDS) == {1: 24.0, 2: 12.0, 3: 6.0, 4: 6.0}


def test_repaired_known_keeps_only_repaired_trucks_sorted() -> None:
    labels, repaired = labels_frame()

    rows = cm.repaired_known(labels.sample(frac=1, random_state=0), repaired)

    assert set(rows[VEHICLE_COL]) == {1, 2}
    assert rows.groupby(VEHICLE_COL)[TIME_COL].apply(lambda t: t.is_monotonic_increasing).all()


def test_repaired_known_errors() -> None:
    labels, repaired = labels_frame()
    with pytest.raises(ValueError, match="missing"):
        cm.repaired_known(labels.drop(columns=[DISTANCE_COL]), repaired)
    with pytest.raises(ValueError, match="no known"):
        cm.repaired_known(labels, pd.Series({1: 0, 2: 0, 3: 0}))


def test_class_density_table_hand_computed() -> None:
    labels, repaired = labels_frame()
    rows = cm.repaired_known(labels, repaired)

    table = cm.class_density_table(rows, BOUNDS).set_index("class")

    # truck 1 (d = 90..0): d 40, 30 -> class 1; 20 -> 2; 10 -> 3; 0 -> 4 (6 and 12 are not hit)
    # truck 2 (d = 50..0): d 40, 30 -> class 1; 20 -> 2; 10 -> 3; 0 -> 4
    assert list(table.n_readouts) == [4, 2, 2, 2]
    assert table.loc[1, "readouts_per_unit"] == pytest.approx(4 / 24 / 2)
    assert table.loc[4, "share_of_trucks_with_readout"] == 1.0


def test_gap_by_distance() -> None:
    labels, repaired = labels_frame()
    rows = cm.repaired_known(labels, repaired)

    gaps = cm.gap_by_distance(rows, (6, 48))

    assert gaps["n_readouts"].sum() == len(rows)
    assert (gaps["median_gap"].dropna() == 10).all()  # every gap is 10


def test_expected_positive_counts_and_shift() -> None:
    labels, repaired = labels_frame()
    rows = cm.repaired_known(labels, repaired)

    base = cm.expected_positive_counts(rows, BOUNDS)

    # truck 1: 10 readouts (d = 90, 80, ..., 0): class 1 at d 40 and 30 -> 2/10
    # truck 2: 6 readouts (d = 50, 40, ..., 0): class 1 at d 40 and 30 -> 2/6
    assert base[0] == pytest.approx(2 / 10 + 2 / 6)
    assert base[3] == pytest.approx(1 / 10 + 1 / 6)  # class 4 = d 0 only
    assert base.sum() < 2  # at most one positive per truck
    later = cm.expected_positive_counts(rows, BOUNDS, shift=-100.0)  # everything after the repair
    assert later[3] == pytest.approx(2.0)  # d' < 0 counts as the last class


def test_chi_square_hand_computed_and_edge_cases() -> None:
    assert cm.chi_square([10, 10], [1, 1]) == 0.0
    assert cm.chi_square([20, 0], [1, 1]) == pytest.approx((10**2 / 10) * 2)
    assert cm.chi_square([5, 5], [1, 0]) == float("inf")
    with pytest.raises(ValueError):
        cm.chi_square([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        cm.chi_square([0, 0], [1, 1])


def test_positive_class_counts_fills_missing_classes() -> None:
    rows = pd.DataFrame({CLASS_COL: pd.array([0, 1, 1, 3], dtype="Int8")})

    assert cm.positive_class_counts(rows) == {1: 2, 2: 0, 3: 1, 4: 0}


def test_offset_table_is_reach_minus_target() -> None:
    reach = pd.Series([100.0, 200.0, 300.0])
    targets = [50.0, 150.0, 250.0]

    table = cm.offset_table(reach, targets, quantiles=(0.0, 0.5, 1.0))

    assert list(table.offset) == [50.0, 50.0, 50.0]


def test_distance_bins_of_chosen_counts_only_repaired_trucks() -> None:
    labels, repaired = labels_frame()
    chosen = labels[labels[TIME_COL] == 60.0]  # one row per truck

    out = cm.distance_bins_of_chosen(chosen, repaired, (6, 48))

    assert int(out.sum()) == 2  # trucks 1 and 2, not the censored truck 3
    assert np.isclose(chosen.loc[chosen[VEHICLE_COL] == 1, DISTANCE_COL].iloc[0], 40)
