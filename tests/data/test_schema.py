"""Tests for the SCANIA data contract (failsense.data.schema).

All tests use tiny hand-made tables, never the raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from failsense.data.labels import LABEL_COLUMNS, REPAIR_COL, STUDY_LENGTH_COL, TIME_COL, VEHICLE_COL
from failsense.data.schema import (
    SPEC_COLUMNS,
    assert_specs_cover_readouts,
    counter_columns,
    feature_columns,
    histogram_columns,
    readouts_schema,
    specifications_schema,
    tte_schema,
    validate_frame,
)

# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def readouts() -> pd.DataFrame:
    """Two trucks, five readouts, all 105 feature columns, one missing value."""
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        VEHICLE_COL: np.array([1, 1, 1, 2, 2], dtype="int64"),
        TIME_COL: np.array([0.0, 3.0, 7.0, 0.0, 5.0]),
    }
    for column in feature_columns():
        data[column] = rng.uniform(0, 100, size=5)
    df = pd.DataFrame(data)
    df.loc[1, "397_5"] = np.nan  # missing features are allowed
    return df


@pytest.fixture
def specs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array([1, 2], dtype="int64"),
            **{c: ["Cat0", "Cat12"] for c in SPEC_COLUMNS},
        }
    )


@pytest.fixture
def tte() -> pd.DataFrame:
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array([1, 2], dtype="int64"),
            STUDY_LENGTH_COL: [7.0, 12.0],
            REPAIR_COL: np.array([1, 0], dtype="int64"),
        }
    )


def failed_checks(exc: pytest.ExceptionInfo[SchemaErrors]) -> set[str]:
    """Names of the checks that failed (from pandera's failure report)."""
    return set(exc.value.failure_cases["check"].astype(str))


# --- column lists ------------------------------------------------------------


def test_feature_columns_match_dataset_structure() -> None:
    assert len(counter_columns()) == 8
    assert len(histogram_columns()) == 97
    assert len(feature_columns()) == 105
    assert len(set(feature_columns())) == 105  # no duplicates


def test_feature_columns_contain_no_label_or_key_columns() -> None:
    forbidden = LABEL_COLUMNS | {VEHICLE_COL, TIME_COL}
    assert forbidden.isdisjoint(feature_columns())


# --- readouts ----------------------------------------------------------------


def test_valid_readouts_pass(readouts: pd.DataFrame) -> None:
    result = validate_frame(readouts, readouts_schema())
    pd.testing.assert_frame_equal(result, readouts)  # returned unchanged


def test_readouts_column_order_does_not_matter(readouts: pd.DataFrame) -> None:
    validate_frame(readouts[readouts.columns[::-1]], readouts_schema())


@pytest.mark.parametrize("label_col", sorted(LABEL_COLUMNS))
def test_label_column_in_readouts_is_rejected(readouts: pd.DataFrame, label_col: str) -> None:
    # Leakage guard: strict=True must reject every label column.
    readouts[label_col] = 1.0
    with pytest.raises(SchemaErrors, match=label_col):
        validate_frame(readouts, readouts_schema())


def test_missing_feature_column_is_rejected(readouts: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors, match="167_3"):
        validate_frame(readouts.drop(columns="167_3"), readouts_schema())


@pytest.mark.parametrize("column", ["171_0", "397_35", TIME_COL])
def test_negative_values_are_rejected(readouts: pd.DataFrame, column: str) -> None:
    readouts.loc[0, column] = -1.0
    with pytest.raises(SchemaErrors):
        validate_frame(readouts, readouts_schema())


@pytest.mark.parametrize("column", [VEHICLE_COL, TIME_COL])
def test_missing_key_values_are_rejected(readouts: pd.DataFrame, column: str) -> None:
    readouts[column] = readouts[column].astype("float64")
    readouts.loc[0, column] = np.nan
    with pytest.raises(SchemaErrors):
        validate_frame(readouts, readouts_schema())


def test_wrong_vehicle_dtype_is_rejected(readouts: pd.DataFrame) -> None:
    readouts[VEHICLE_COL] = readouts[VEHICLE_COL].astype("float64")
    with pytest.raises(SchemaErrors):
        validate_frame(readouts, readouts_schema())


def test_empty_readouts_are_rejected(readouts: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors) as exc:
        validate_frame(readouts.iloc[0:0], readouts_schema())
    assert "not_empty" in failed_checks(exc)


@pytest.mark.parametrize(
    "bad_times", [[0.0, 7.0, 3.0], [0.0, 3.0, 3.0]], ids=["unsorted", "duplicate"]
)
def test_time_step_not_increasing_is_rejected(
    readouts: pd.DataFrame, bad_times: list[float]
) -> None:
    readouts.loc[0:2, TIME_COL] = bad_times
    with pytest.raises(SchemaErrors) as exc:
        validate_frame(readouts, readouts_schema())
    assert "time_step_increasing_per_vehicle" in failed_checks(exc)


def test_interleaved_trucks_with_increasing_time_pass(readouts: pd.DataFrame) -> None:
    # Rows of different trucks may be mixed; only the order inside a truck counts.
    validate_frame(readouts.iloc[[0, 3, 1, 4, 2]], readouts_schema())


# --- specifications ----------------------------------------------------------


def test_valid_specs_pass(specs: pd.DataFrame) -> None:
    validate_frame(specs, specifications_schema())


def test_unseen_but_well_formed_category_passes(specs: pd.DataFrame) -> None:
    # Validation/test contain categories not seen in training: valid data.
    specs.loc[0, "Spec_3"] = "Cat99"
    validate_frame(specs, specifications_schema())


@pytest.mark.parametrize("bad_value", ["X1", "cat3", "", None])
def test_malformed_spec_value_is_rejected(specs: pd.DataFrame, bad_value: str | None) -> None:
    specs.loc[0, "Spec_0"] = bad_value
    with pytest.raises(SchemaErrors):
        validate_frame(specs, specifications_schema())


def test_numeric_spec_column_is_rejected(specs: pd.DataFrame) -> None:
    specs["Spec_1"] = [0, 1]
    with pytest.raises(SchemaErrors):
        validate_frame(specs, specifications_schema())


def test_duplicate_vehicle_in_specs_is_rejected(specs: pd.DataFrame) -> None:
    specs.loc[1, VEHICLE_COL] = 1
    with pytest.raises(SchemaErrors):
        validate_frame(specs, specifications_schema())


def test_specs_cover_readouts(readouts: pd.DataFrame, specs: pd.DataFrame) -> None:
    assert_specs_cover_readouts(readouts, specs)
    with pytest.raises(ValueError, match="no specification row"):
        assert_specs_cover_readouts(readouts, specs[specs[VEHICLE_COL] == 1])


# --- time-to-event -----------------------------------------------------------


def test_valid_tte_passes(tte: pd.DataFrame) -> None:
    validate_frame(tte, tte_schema())


def test_invalid_repair_flag_is_rejected(tte: pd.DataFrame) -> None:
    tte.loc[0, REPAIR_COL] = 2
    with pytest.raises(SchemaErrors):
        validate_frame(tte, tte_schema())


def test_negative_study_length_is_rejected(tte: pd.DataFrame) -> None:
    tte.loc[0, STUDY_LENGTH_COL] = -1.0
    with pytest.raises(SchemaErrors):
        validate_frame(tte, tte_schema())


def test_duplicate_vehicle_in_tte_is_rejected(tte: pd.DataFrame) -> None:
    tte.loc[1, VEHICLE_COL] = 1
    with pytest.raises(SchemaErrors):
        validate_frame(tte, tte_schema())


def test_histogram_groups_match_histogram_columns() -> None:
    from failsense.data.schema import histogram_groups

    groups = histogram_groups()
    assert [c for cols in groups.values() for c in cols] == histogram_columns()
    assert len(groups["397"]) == 36
