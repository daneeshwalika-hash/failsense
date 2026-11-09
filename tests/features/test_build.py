"""Tests for failsense.features.build (specification join)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError

from failsense.data.labels import BINARY_COL, LABEL_COLUMNS, TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, counter_columns, histogram_groups
from failsense.features.build import (
    add_counter_deltas,
    add_histogram_features,
    attach_specifications,
    build_model_features,
    engineered_columns,
    features_params_from_config,
    n_top_bins,
)


@pytest.fixture
def rows() -> pd.DataFrame:
    """Three cut-offs of truck 7 and one of truck 3, with a non-default index."""
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array([7, 3, 7, 7], dtype="int64"),
            TIME_COL: [10.0, 4.0, 20.0, 30.0],
            BINARY_COL: pd.array([0, 1, 0, 1], dtype="Int8"),
        },
        index=[105, 42, 106, 107],
    )


@pytest.fixture
def specs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array([3, 7, 9], dtype="int64"),  # truck 9 has no rows: fine
            **{c: [f"Cat{i}", f"Cat{i + 10}", "Cat0"] for i, c in enumerate(SPEC_COLUMNS)},
        }
    )


def test_specs_are_attached_per_vehicle(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    result = attach_specifications(rows, specs)
    assert list(result.columns) == [*rows.columns, *SPEC_COLUMNS]
    # Truck 7 gets the second spec row, truck 3 the first.
    assert result.loc[[105, 106, 107], "Spec_0"].tolist() == ["Cat10"] * 3
    assert result.loc[42, "Spec_0"] == "Cat0"
    assert result.loc[42, "Spec_7"] == "Cat7"


def test_index_order_and_row_count_are_kept(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    result = attach_specifications(rows, specs)
    assert result.index.tolist() == rows.index.tolist()
    pd.testing.assert_frame_equal(result[rows.columns], rows)  # original columns untouched


def test_input_is_not_modified(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    before = rows.copy()
    attach_specifications(rows, specs)
    pd.testing.assert_frame_equal(rows, before)


def test_duplicate_spec_row_raises(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    # Truck 7 twice in specs would duplicate its cut-offs (row explosion).
    duplicated = pd.concat([specs, specs.iloc[[1]]], ignore_index=True)
    with pytest.raises(MergeError):
        attach_specifications(rows, duplicated)


def test_vehicle_without_spec_row_raises(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"1 vehicle\(s\) have no specification row, e\.g\. \[3\]"):
        attach_specifications(rows, specs[specs[VEHICLE_COL] != 3])


def test_spec_columns_already_present_raise(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    once = attach_specifications(rows, specs)
    with pytest.raises(ValueError, match="already contains spec columns"):
        attach_specifications(once, specs)


def test_missing_vehicle_column_raises(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="rows is missing"):
        attach_specifications(rows.drop(columns=VEHICLE_COL), specs)


def test_missing_spec_column_raises(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Spec_4"):
        attach_specifications(rows, specs.drop(columns="Spec_4"))


def test_empty_rows_give_empty_result(rows: pd.DataFrame, specs: pd.DataFrame) -> None:
    result = attach_specifications(rows.iloc[0:0], specs)
    assert result.empty
    assert list(result.columns) == [*rows.columns, *SPEC_COLUMNS]


# --- engineered features (30.10) ----------------------------------------------

C = "171_0"


@pytest.fixture
def readouts() -> pd.DataFrame:
    """Two trucks, interleaved rows, non-default index. Counter 171_0 grows."""
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array([1, 2, 1, 2, 1], dtype="int64"),
            TIME_COL: [10.0, 5.0, 20.0, 15.0, 40.0],
            C: [100.0, 7.0, 130.0, 9.0, 190.0],
        },
        index=[50, 10, 30, 20, 40],
    )


def test_deltas_and_rates_per_truck(readouts: pd.DataFrame) -> None:
    out = add_counter_deltas(readouts, [C])
    # truck 1: 100 -> 130 (dt 10) -> 190 (dt 20); truck 2: 7 -> 9 (dt 10)
    assert out.loc[30, f"{C}__delta"] == 30.0
    assert out.loc[30, f"{C}__rate"] == 3.0
    assert out.loc[40, f"{C}__delta"] == 60.0
    assert out.loc[40, f"{C}__rate"] == 3.0
    assert out.loc[20, f"{C}__delta"] == 2.0
    assert out.loc[20, f"{C}__rate"] == 0.2


@pytest.mark.leakage
def test_first_readout_of_each_truck_is_nan(readouts: pd.DataFrame) -> None:
    out = add_counter_deltas(readouts, [C])
    # Truck 2 starts at index 10, truck 1 at index 50: no delta from the other truck.
    assert np.isnan(out.loc[50, f"{C}__delta"])
    assert np.isnan(out.loc[10, f"{C}__delta"])
    assert np.isnan(out.loc[10, f"{C}__rate"])


@pytest.mark.leakage
def test_deltas_never_cross_the_truck_boundary() -> None:
    # Truck 1's last value is huge, truck 2's first value tiny: a delta across
    # the boundary would be hugely negative.
    df = pd.DataFrame(
        {VEHICLE_COL: [1, 1, 2, 2], TIME_COL: [1.0, 2.0, 1.0, 2.0], C: [0.0, 1000.0, 1.0, 2.0]}
    )
    out = add_counter_deltas(df, [C])
    assert out[f"{C}__delta"].iloc[2] != out[f"{C}__delta"].iloc[2]  # NaN
    assert out[f"{C}__delta"].iloc[3] == 1.0


def test_deltas_keep_index_order_and_rows(readouts: pd.DataFrame) -> None:
    out = add_counter_deltas(readouts, [C])
    assert out.index.tolist() == readouts.index.tolist()
    assert len(out) == len(readouts)
    assert list(out.columns) == [*readouts.columns, f"{C}__delta", f"{C}__rate"]
    pd.testing.assert_frame_equal(out[readouts.columns], readouts)
    assert f"{C}__delta" not in readouts.columns  # input not modified


@pytest.mark.leakage
def test_no_future_leakage_in_deltas(readouts: pd.DataFrame) -> None:
    changed = readouts.copy()
    changed.loc[40, C] = 9999.0  # the LAST readout of truck 1 changes
    before = add_counter_deltas(readouts, [C]).drop(index=40)
    after = add_counter_deltas(changed, [C]).drop(index=40)
    pd.testing.assert_frame_equal(before, after)


def test_one_readout_truck_gives_nan() -> None:
    df = pd.DataFrame({VEHICLE_COL: [5], TIME_COL: [3.0], C: [10.0]})
    out = add_counter_deltas(df, [C])
    assert out[[f"{C}__delta", f"{C}__rate"]].isna().all().all()


def test_empty_frame_gets_the_new_columns() -> None:
    df = pd.DataFrame(
        {VEHICLE_COL: pd.Series([], dtype="int64"), TIME_COL: pd.Series([], dtype=float), C: []}
    )
    out = add_counter_deltas(df, [C])
    assert out.empty
    assert f"{C}__delta" in out.columns


def test_missing_columns_raise(readouts: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        add_counter_deltas(readouts, ["nope"])
    with pytest.raises(ValueError, match="missing"):
        add_counter_deltas(readouts.drop(columns=TIME_COL), [C])
    with pytest.raises(ValueError, match="missing"):
        add_counter_deltas(readouts.drop(columns=VEHICLE_COL), [C])


@pytest.mark.parametrize("times", [[10.0, 5.0, 20.0, 15.0, 15.0], [10.0, 5.0, 10.0, 15.0, 40.0]])
@pytest.mark.leakage
def test_unsorted_or_duplicate_time_step_raises(readouts: pd.DataFrame, times: list[float]) -> None:
    bad = readouts.assign(**{TIME_COL: times})  # truck 1 goes backwards / stands still
    with pytest.raises(ValueError, match="strictly increasing"):
        add_counter_deltas(bad, [C])


def test_nan_in_keys_raises(readouts: pd.DataFrame) -> None:
    bad = readouts.copy()
    bad.loc[10, TIME_COL] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        add_counter_deltas(bad, [C])


def test_counter_decrease_is_kept_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({VEHICLE_COL: [1, 1], TIME_COL: [1.0, 2.0], C: [10.0, 4.0]})
    with caplog.at_level("WARNING"):
        out = add_counter_deltas(df, [C])
    assert out[f"{C}__delta"].iloc[1] == -6.0  # not clipped
    assert "decreases in 1 readouts" in caplog.text


def test_deltas_work_with_real_column_lists() -> None:
    cols = counter_columns()
    df = pd.DataFrame({VEHICLE_COL: [1, 1], TIME_COL: [1.0, 2.0], **{c: [1.0, 3.0] for c in cols}})
    out = add_counter_deltas(df, cols)
    assert len(out.columns) == 2 + 3 * len(cols)


# --- histograms ------------------------------------------------------------------
HG = {"h": ["h_0", "h_1", "h_2", "h_3"], "g": ["g_0", "g_1"]}


@pytest.fixture
def hist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            VEHICLE_COL: [1, 1, 2],
            "h_0": [1.0, 0.0, 2.0],
            "h_1": [1.0, 0.0, 0.0],
            "h_2": [2.0, 0.0, 0.0],
            "h_3": [0.0, 0.0, 2.0],
            "g_0": [3.0, 0.0, 1.0],
            "g_1": [1.0, 0.0, 1.0],
        },
        index=[7, 8, 9],
    )


def test_shares_sum_to_one(hist: pd.DataFrame) -> None:
    out = add_histogram_features(hist, HG, 0.5)
    shares = out[[f"h_{i}__share" for i in range(4)]]
    assert shares.iloc[0].tolist() == [0.25, 0.25, 0.5, 0.0]
    assert shares.iloc[0].sum() == pytest.approx(1.0)


def test_mean_bin_and_top_share(hist: pd.DataFrame) -> None:
    out = add_histogram_features(hist, HG, 0.5)  # 4 bins -> top 2 bins
    # row 0: 0*.25 + 1*.25 + 2*.5 = 1.25 ; row 2: 0*.5 + 3*.5 = 1.5
    assert out["h__mean_bin"].iloc[0] == pytest.approx(1.25)
    assert out["h__mean_bin"].iloc[2] == pytest.approx(1.5)
    assert out["h__top_share"].iloc[0] == pytest.approx(0.5)  # bins 2,3
    assert out["h__top_share"].iloc[2] == pytest.approx(0.5)  # bins 2,3 = 0 + .5
    assert out["g__top_share"].iloc[0] == pytest.approx(0.25)  # 2 bins * 0.5 -> 1 bin


def test_histogram_row_sum_zero_gives_nan(hist: pd.DataFrame) -> None:
    out = add_histogram_features(hist, HG, 0.5)
    new = [c for c in out.columns if c.endswith(("__share", "__mean_bin", "__top_share"))]
    assert out.loc[8, new].isna().all()  # the all-zero row
    assert not np.isinf(out[new].to_numpy(dtype=float)).any()


def test_histogram_nan_bin_gives_nan_row(hist: pd.DataFrame) -> None:
    hist.loc[7, "h_1"] = np.nan
    out = add_histogram_features(hist, HG, 0.5)
    assert out.loc[7, ["h_0__share", "h__mean_bin", "h__top_share"]].isna().all()
    assert out.loc[7, "g_0__share"] == 0.75  # other histogram unaffected


def test_histogram_keeps_index_order_rows(hist: pd.DataFrame) -> None:
    out = add_histogram_features(hist, HG, 0.5)
    assert out.index.tolist() == [7, 8, 9]
    pd.testing.assert_frame_equal(out[hist.columns], hist)


def test_histogram_empty_frame(hist: pd.DataFrame) -> None:
    out = add_histogram_features(hist.iloc[:0], HG, 0.5)
    assert out.empty
    assert "h__mean_bin" in out.columns


def test_histogram_missing_columns_and_bad_fraction(hist: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        add_histogram_features(hist.drop(columns="h_2"), HG, 0.5)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_fraction"):
            add_histogram_features(hist, HG, bad)


def test_histogram_second_call_raises(hist: pd.DataFrame) -> None:
    once = add_histogram_features(hist, HG, 0.5)
    with pytest.raises(ValueError, match="already contains"):
        add_histogram_features(once, HG, 0.5)


def test_real_histogram_groups_work() -> None:
    groups = histogram_groups()
    bins = [c for cols in groups.values() for c in cols]
    df = pd.DataFrame(np.ones((2, len(bins))), columns=bins)
    out = add_histogram_features(df, groups, 0.2)
    assert len(out.columns) == len(bins) * 2 + 2 * len(groups)
    assert out["397__top_share"].iloc[0] == pytest.approx(7 / 36)  # round(.2*36) = 7


def test_histogram_features_same_before_and_after_selection(hist: pd.DataFrame) -> None:
    """Row-wise features: computing on all rows then selecting == selecting then computing."""
    select = [7, 9]
    full_then_select = add_histogram_features(hist, HG, 0.5).loc[select]
    select_then_full = add_histogram_features(hist.loc[select], HG, 0.5)
    pd.testing.assert_frame_equal(full_then_select, select_then_full)


@pytest.mark.parametrize(
    ("n_bins", "fraction", "expected"), [(10, 0.2, 2), (36, 0.2, 7), (11, 0.2, 2), (3, 0.1, 1)]
)
def test_n_top_bins(n_bins: int, fraction: float, expected: int) -> None:
    assert n_top_bins(n_bins, fraction) == expected


def test_features_params_from_config() -> None:
    assert features_params_from_config({"features": {"top_fraction": 0.2}}) == 0.2
    with pytest.raises(KeyError):
        features_params_from_config({})
    with pytest.raises(ValueError, match="top_fraction"):
        features_params_from_config({"features": {"top_fraction": 0}})
    with pytest.raises(TypeError):
        features_params_from_config({"features": {"top_fraction": "0.2"}})


# --- leakage guards ----------------------------------------------------------------
@pytest.mark.leakage
def test_engineered_columns_never_contain_labels_or_vehicle_id(readouts: pd.DataFrame) -> None:
    out = add_counter_deltas(readouts, [C])
    new = engineered_columns(list(out.columns))
    assert new == [f"{C}__delta", f"{C}__rate"]
    assert not set(new) & (LABEL_COLUMNS | {VEHICLE_COL})
    assert VEHICLE_COL not in new


@pytest.mark.leakage
def test_label_columns_in_input_are_not_transformed(readouts: pd.DataFrame) -> None:
    # Only the listed counters are turned into features, nothing else.
    df = readouts.assign(in_study_repair=1, length_of_study_time_step=100.0)
    out = add_counter_deltas(df, [C])
    assert engineered_columns(list(out.columns)) == [f"{C}__delta", f"{C}__rate"]


# --- build_model_features: the one builder shared by training, validation and the API ----
def test_model_features_match_frame_from_sampled_without_the_label(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    from failsense.models.baseline_cv import frame_from_sampled

    readouts = mini_frames["readouts"]
    sampled = pd.DataFrame({BINARY_COL: 1}, index=readouts.index[::7])
    args = (counter_columns(), histogram_groups(), 0.2)

    features = build_model_features(readouts, sampled.index, *args)
    training = frame_from_sampled(readouts, sampled, *args)

    assert BINARY_COL not in features.columns  # the shared builder never touches labels
    pd.testing.assert_frame_equal(training.drop(columns=BINARY_COL), features)


def test_two_readouts_give_the_same_last_row_as_the_full_history(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    """Every feature needs at most the immediately preceding readout (API contract)."""
    args = (counter_columns(), histogram_groups(), 0.2)
    truck = mini_frames["readouts"].query("vehicle_id == 3").reset_index(drop=True)
    last = truck.index[-1]

    full = build_model_features(truck, pd.Index([last]), *args)
    two = truck.iloc[-2:].reset_index(drop=True)
    pair = build_model_features(two, pd.Index([1]), *args)

    pd.testing.assert_frame_equal(full.reset_index(drop=True), pair.reset_index(drop=True))


def test_single_readout_has_nan_deltas_like_a_trucks_first_readout(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    args = (counter_columns(), histogram_groups(), 0.2)
    truck = mini_frames["readouts"].query("vehicle_id == 3").reset_index(drop=True)

    first_in_history = build_model_features(truck, pd.Index([0]), *args)
    alone = build_model_features(truck.iloc[[0]], pd.Index([0]), *args)

    delta_cols = [c for c in alone.columns if c.endswith(("__delta", "__rate"))]
    assert len(delta_cols) == 2 * len(counter_columns())
    assert alone[delta_cols].isna().all().all()
    pd.testing.assert_frame_equal(first_in_history, alone)
