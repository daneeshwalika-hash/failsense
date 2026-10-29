"""Tests for failsense.models.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_validate

from failsense.data.split import make_group_folds
from failsense.models.pipeline import (
    FORBIDDEN_FEATURES,
    SCALERS,
    assert_no_forbidden_features,
    build_dummy_classifier,
    build_logreg_pipeline,
    build_preprocessor,
    select_categorical_columns,
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

    numeric = pipe.named_steps["preprocess"].named_transformers_["num"]
    learned = numeric.named_steps["impute"].statistics_
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


# --- ColumnTransformer (29.10) -------------------------------------------------


@pytest.fixture
def mixed(rows: pd.DataFrame) -> pd.DataFrame:
    """Numeric features + two spec columns with a few categories."""
    df = rows.copy()
    df["Spec_0"] = np.where(np.arange(len(df)) % 3 == 0, "Cat1", "Cat2")
    df["Spec_1"] = np.where(np.arange(len(df)) % 2 == 0, "Cat0", "Cat5")
    return df


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    return df[select_feature_columns(df) + select_categorical_columns(df)]


def test_select_categorical_columns(mixed: pd.DataFrame) -> None:
    assert select_categorical_columns(mixed) == ["Spec_0", "Spec_1"]
    assert select_categorical_columns(mixed.drop(columns=["Spec_0", "Spec_1"])) == []


def test_preprocessor_output_width(mixed: pd.DataFrame) -> None:
    pre = build_preprocessor(["Spec_0", "Spec_1"]).fit(model_input(mixed))
    out = pre.transform(model_input(mixed))
    # 3 numeric columns + 2 categories of Spec_0 + 2 categories of Spec_1.
    assert out.shape == (len(mixed), 3 + 2 + 2)
    assert not np.isnan(out).any()  # imputer filled the NaN in 666_0


def test_unseen_category_is_encoded_as_zeros(mixed: pd.DataFrame) -> None:
    pre = build_preprocessor(["Spec_0", "Spec_1"]).fit(model_input(mixed))
    new = model_input(mixed).iloc[[0]].copy()
    new["Spec_0"] = "Cat99"  # never seen in training (happens in validation/test)
    out = pre.transform(new)
    spec0 = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("cat__Spec_0")]
    assert out[0, spec0].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("scaler", sorted(SCALERS))
def test_pipeline_with_specs_fits_and_predicts(mixed: pd.DataFrame, scaler: str) -> None:
    x = model_input(mixed)
    y = mixed["repair_within_horizon"].to_numpy(dtype=int)
    pipe = build_logreg_pipeline(42, categorical_columns=["Spec_0", "Spec_1"], scaler=scaler)
    proba = pipe.fit(x, y).predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)


def test_minmax_maps_training_data_to_unit_range(mixed: pd.DataFrame) -> None:
    x = model_input(mixed)
    pre = build_preprocessor(scaler="minmax").fit(x)
    out = pre.transform(x)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_scaler_learns_only_from_training_part(rows: pd.DataFrame) -> None:
    x = rows[select_feature_columns(rows)]
    pre = build_preprocessor(scaler="standard").fit(x.iloc[:60])
    scale = pre.named_transformers_["num"].named_steps["scale"]
    imputed_train = x.iloc[:60].fillna(x.iloc[:60].median())
    np.testing.assert_allclose(scale.mean_, imputed_train.mean().to_numpy())


def test_unknown_scaler_raises() -> None:
    with pytest.raises(ValueError, match="unknown scaler"):
        build_preprocessor(scaler="robust")


@pytest.mark.parametrize("leak", ["vehicle_id", "time_to_study_end"])
def test_forbidden_numeric_column_stops_fit(rows: pd.DataFrame, leak: str) -> None:
    x = rows[[*select_feature_columns(rows), leak]]
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    with pytest.raises(ValueError, match="leakage"):
        build_logreg_pipeline(42).fit(x, y)


def test_pipeline_with_specs_survives_parallel_cv(mixed: pd.DataFrame) -> None:
    # n_jobs=2 pickles the pipeline (incl. the column selector) to worker processes.
    x = model_input(mixed)
    y = mixed["repair_within_horizon"].to_numpy(dtype=int)
    folds = make_group_folds(mixed, n_splits=3, seed=42)
    pipe = build_logreg_pipeline(42, categorical_columns=["Spec_0", "Spec_1"])
    result = cross_validate(pipe, x, y, cv=folds, scoring="average_precision", n_jobs=2)
    assert len(result["test_score"]) == 3
