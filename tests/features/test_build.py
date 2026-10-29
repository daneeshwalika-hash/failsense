"""Tests for failsense.features.build (specification join)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError

from failsense.data.labels import BINARY_COL, TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS
from failsense.features.build import attach_specifications


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
