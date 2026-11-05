"""Tests for failsense.data.sampling.

The fixtures are tiny hand-made label tables, never the real SCANIA data, so
every expected value can be checked by reading the test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from failsense.data.labels import CLASS_COL, VEHICLE_COL
from failsense.data.sampling import (
    sample_cutoffs,
    sampling_params_from_config,
    validate_n_per_vehicle,
)


@pytest.fixture
def labels() -> pd.DataFrame:
    """Three vehicles with different situations.

    * vehicle 1: 5 known readouts
    * vehicle 2: 2 known readouts + 2 unknown (<NA>, censored near the end)
    * vehicle 3: only unknown readouts -> must never appear in the sample
    """
    return pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            "time_step": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0],
            "proximity_class": pd.array(
                [0, 0, 1, 2, 4, 0, 0, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int8"
            ),
        },
        # A non-default index checks that the original index is kept.
        index=range(100, 111),
    )


@pytest.fixture
def big_labels() -> pd.DataFrame:
    """50 vehicles x 20 known readouts, large enough to compare seeds."""
    vehicles = [v for v in range(50) for _ in range(20)]
    return pd.DataFrame(
        {
            "vehicle_id": vehicles,
            "time_step": [float(t) for _ in range(50) for t in range(20)],
            "proximity_class": pd.array([0] * len(vehicles), dtype="Int8"),
        }
    )


# --- reproducibility --------------------------------------------------------


def test_same_seed_gives_same_sample(big_labels: pd.DataFrame) -> None:
    first = sample_cutoffs(big_labels, n_per_vehicle=3, seed=42)
    second = sample_cutoffs(big_labels, n_per_vehicle=3, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_sample(big_labels: pd.DataFrame) -> None:
    first = sample_cutoffs(big_labels, n_per_vehicle=3, seed=42)
    second = sample_cutoffs(big_labels, n_per_vehicle=3, seed=7)
    assert not first.index.equals(second.index)


# --- what is drawn ----------------------------------------------------------


@pytest.mark.leakage
def test_never_draws_unknown_labels(labels: pd.DataFrame) -> None:
    sampled = sample_cutoffs(labels, n_per_vehicle=10, seed=0)
    assert sampled[CLASS_COL].notna().all()
    assert 3 not in set(sampled[VEHICLE_COL])  # vehicle 3 has only unknown rows


def test_one_cutoff_per_vehicle(labels: pd.DataFrame) -> None:
    sampled = sample_cutoffs(labels, n_per_vehicle=1, seed=0)
    assert sampled[VEHICLE_COL].value_counts().to_dict() == {1: 1, 2: 1}


def test_vehicle_with_fewer_rows_than_k_keeps_all_without_duplicates(
    labels: pd.DataFrame,
) -> None:
    sampled = sample_cutoffs(labels, n_per_vehicle=3, seed=0)
    counts = sampled[VEHICLE_COL].value_counts().to_dict()
    assert counts == {1: 3, 2: 2}  # vehicle 2 has only 2 known rows
    assert not sampled.index.duplicated().any()


def test_none_keeps_all_known_rows(labels: pd.DataFrame) -> None:
    sampled = sample_cutoffs(labels, n_per_vehicle=None, seed=0)
    assert len(sampled) == 7  # 5 (vehicle 1) + 2 (vehicle 2)


def test_keeps_index_order_and_columns(labels: pd.DataFrame) -> None:
    sampled = sample_cutoffs(labels, n_per_vehicle=2, seed=0)
    assert list(sampled.columns) == list(labels.columns)
    assert sampled.index.isin(labels.index).all()
    assert sampled.index.is_monotonic_increasing  # original order restored


# --- edge cases ---------------------------------------------------------------


def test_empty_input_returns_empty_frame(labels: pd.DataFrame) -> None:
    sampled = sample_cutoffs(labels.iloc[0:0], n_per_vehicle=2, seed=0)
    assert sampled.empty
    assert list(sampled.columns) == list(labels.columns)


def test_missing_column_raises(labels: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        sample_cutoffs(labels.drop(columns=[CLASS_COL]), n_per_vehicle=1, seed=0)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_k_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        validate_n_per_vehicle(bad)


@pytest.mark.parametrize("bad", [1.5, "3", True])
def test_non_int_k_raises(bad: object) -> None:
    with pytest.raises(TypeError):
        validate_n_per_vehicle(bad)  # type: ignore[arg-type]


# --- config -------------------------------------------------------------------


def test_sampling_params_from_config() -> None:
    config = {"sampling": {"n_cutoffs_per_vehicle": 5, "seed": 42}}
    assert sampling_params_from_config(config) == (5, 42)


def test_sampling_params_null_means_all() -> None:
    config = {"sampling": {"n_cutoffs_per_vehicle": None, "seed": 42}}
    assert sampling_params_from_config(config) == (None, 42)


def test_sampling_params_missing_key() -> None:
    with pytest.raises(KeyError, match="sampling"):
        sampling_params_from_config({"sampling": {"seed": 42}})
