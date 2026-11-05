"""Tests for failsense.rul.labels (tiny engines)."""

from __future__ import annotations

import pandas as pd
import pytest

from failsense.rul.labels import (
    FORBIDDEN_FEATURES,
    assert_no_forbidden_features,
    compute_rul,
    rul_cap_from_config,
    validate_cap,
)


@pytest.fixture
def engines() -> pd.DataFrame:
    """Engine 1 runs 5 cycles, engine 2 runs 3; rows interleaved, odd index."""
    return pd.DataFrame(
        {
            "engine": [1, 2, 1, 2, 1, 2, 1, 1],
            "cycle": [1, 1, 2, 2, 3, 3, 4, 5],
        },
        index=[40, 10, 30, 20, 50, 60, 70, 80],
    )


def test_rul_is_max_cycle_minus_cycle(engines: pd.DataFrame) -> None:
    rul = compute_rul(engines)
    # engine 1: cycles 1..5 -> 4,3,2,1,0 ; engine 2: cycles 1..3 -> 2,1,0
    assert rul.loc[[40, 30, 50, 70, 80]].tolist() == [4, 3, 2, 1, 0]
    assert rul.loc[[10, 20, 60]].tolist() == [2, 1, 0]
    assert rul.name == "rul" and str(rul.dtype) == "int64"


def test_last_cycle_of_every_engine_has_rul_zero(engines: pd.DataFrame) -> None:
    rul = compute_rul(engines)
    last = engines.groupby("engine")["cycle"].transform("max") == engines["cycle"]
    assert (rul[last] == 0).all()


def test_cap_limits_the_early_life(engines: pd.DataFrame) -> None:
    rul = compute_rul(engines, cap=2)
    assert rul.loc[[40, 30, 50, 70, 80]].tolist() == [2, 2, 2, 1, 0]
    assert rul.max() == 2


def test_cap_larger_than_every_rul_changes_nothing(engines: pd.DataFrame) -> None:
    pd.testing.assert_series_equal(compute_rul(engines, cap=1000), compute_rul(engines))


def test_rul_does_not_depend_on_row_order(engines: pd.DataFrame) -> None:
    shuffled = engines.sample(frac=1.0, random_state=0)
    pd.testing.assert_series_equal(
        compute_rul(shuffled).sort_index(), compute_rul(engines).sort_index()
    )


def test_rul_keeps_index_and_order(engines: pd.DataFrame) -> None:
    assert compute_rul(engines).index.tolist() == engines.index.tolist()


def test_rul_does_not_add_max_cycle_to_the_frame(engines: pd.DataFrame) -> None:
    before = engines.copy()
    compute_rul(engines)
    pd.testing.assert_frame_equal(engines, before)  # max_cycle never becomes a column


def test_single_cycle_engine_and_empty_frame() -> None:
    one = pd.DataFrame({"engine": [3], "cycle": [1]})
    assert compute_rul(one).tolist() == [0]
    empty = pd.DataFrame(
        {"engine": pd.Series([], dtype="int64"), "cycle": pd.Series([], dtype="int64")}
    )
    assert compute_rul(empty).empty


def test_missing_columns_raise(engines: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        compute_rul(engines.drop(columns="cycle"))
    with pytest.raises(ValueError, match="missing"):
        compute_rul(engines.drop(columns="engine"))


@pytest.mark.parametrize("bad", [0, -5])
def test_invalid_cap_value_raises(engines: pd.DataFrame, bad: int) -> None:
    with pytest.raises(ValueError, match="cap"):
        compute_rul(engines, cap=bad)


@pytest.mark.parametrize("bad", [1.5, "125", True, None])
def test_validate_cap_rejects_non_ints(bad: object) -> None:
    with pytest.raises(TypeError, match="int"):
        validate_cap(bad)


def test_cap_from_config() -> None:
    assert rul_cap_from_config({"rul": {"cap": 125}}) == 125
    with pytest.raises(KeyError, match="rul.cap"):
        rul_cap_from_config({})
    with pytest.raises(ValueError):
        rul_cap_from_config({"rul": {"cap": 0}})


@pytest.mark.leakage
def test_forbidden_features_guard() -> None:
    assert {"rul", "max_cycle", "engine"} <= FORBIDDEN_FEATURES
    assert_no_forbidden_features(["sensor_2", "setting_1", "cycle"])  # cycle is allowed
    with pytest.raises(ValueError, match="leakage"):
        assert_no_forbidden_features(["sensor_2", "rul"])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_forbidden_features(["max_cycle"])
