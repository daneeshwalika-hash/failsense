"""Tests for failsense.models.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_validate

from failsense.data.split import make_group_folds
from failsense.models.pipeline import (
    FORBIDDEN_FEATURES,
    assert_no_forbidden_features,
    build_dummy_classifier,
    build_logreg_pipeline,
    select_feature_columns,
)


@pytest.fixture
def rows() -> pd.DataFrame:
    """60 vehicles x 2 rows; the positive class has larger values in `171_0`.

    Feature `666_0` has missing values to exercise the imputer.
    """
    rng = np.random.default_rng(0)
    vehicle_ids = np.repeat(np.arange(60), 2)
    y = (vehicle_ids % 4 == 0).astype(int)  # 25 % positive vehicles
    feature = rng.normal(size=len(y)) + 2.0 * y
    other = rng.normal(size=len(y))
    other[::7] = np.nan
    return pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "time_step": np.tile([10.0, 20.0], 60),
            "171_0": feature,
            "666_0": other,
            "Spec_0": ["Cat0"] * len(y),  # non-numeric -> not a numeric feature
            "time_to_study_end": rng.uniform(0, 100, size=len(y)),  # label column
            "repair_within_horizon": pd.array(y, dtype="Int8"),
        }
    )


# --- feature selection: leakage guard ----------------------------------------


def test_select_excludes_labels_id_and_non_numeric(rows: pd.DataFrame) -> None:
    assert select_feature_columns(rows) == ["time_step", "171_0", "666_0"]


def test_select_respects_extra_exclude(rows: pd.DataFrame) -> None:
    assert select_feature_columns(rows, exclude=["time_step"]) == ["171_0", "666_0"]


def test_forbidden_features_contain_ids_and_labels() -> None:
    assert {"vehicle_id", "in_study_repair", "length_of_study_time_step"} <= FORBIDDEN_FEATURES


def test_assert_no_forbidden_features_raises() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_no_forbidden_features(["171_0", "vehicle_id"])


# --- pipeline behaviour --------------------------------------------------------


def test_pipeline_handles_missing_values(rows: pd.DataFrame) -> None:
    features = select_feature_columns(rows)
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    pipe = build_logreg_pipeline(random_state=42).fit(rows[features], y)

    proba = pipe.predict_proba(rows[features])[:, 1]
    assert proba.shape == (len(rows),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_imputer_learns_only_from_training_part(rows: pd.DataFrame) -> None:
    """The core reason for a Pipeline: statistics come from the fit data only."""
    features = select_feature_columns(rows)
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    train = rows.iloc[:60]  # pretend this is the training part of a fold

    pipe = build_logreg_pipeline(random_state=42).fit(train[features], y[:60])

    learned = pipe.named_steps["impute"].statistics_
    np.testing.assert_allclose(learned, train[features].median().to_numpy())
    assert not np.allclose(learned, rows[features].median().to_numpy())


def test_cross_validate_with_group_folds(rows: pd.DataFrame) -> None:
    features = select_feature_columns(rows)
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    folds = make_group_folds(rows, n_splits=5, seed=42)

    result = cross_validate(
        build_logreg_pipeline(random_state=42),
        rows[features],
        y,
        cv=folds,
        scoring="average_precision",
    )
    assert len(result["test_score"]) == 5
    # The signal in `171_0` is strong, so the model must beat the prior (0.25).
    assert result["test_score"].mean() > 0.5


def test_dummy_pr_auc_equals_prevalence(rows: pd.DataFrame) -> None:
    features = select_feature_columns(rows)
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    folds = make_group_folds(rows, n_splits=5, seed=42)

    result = cross_validate(
        build_dummy_classifier(), rows[features], y, cv=folds, scoring="average_precision"
    )
    # Constant scores -> average precision = share of positives in each fold.
    expected = [y[val].mean() for _, val in folds]
    np.testing.assert_allclose(result["test_score"], expected)
