"""Tests for failsense.models.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate

from failsense.data.split import make_group_folds
from failsense.models.pipeline import (
    FORBIDDEN_FEATURES,
    IMBALANCE_STRATEGIES,
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


# --- engineered columns (30.10) ---------------------------------------------------
def test_selector_picks_engineered_columns_and_rejects_forbidden(rows: pd.DataFrame) -> None:
    from failsense.features.build import add_counter_deltas, add_histogram_features
    from failsense.models.pipeline import numeric_feature_selector

    eng = add_counter_deltas(rows[["vehicle_id", "time_step", "171_0"]], ["171_0"])
    eng = add_histogram_features(eng.assign(h_0=1.0, h_1=3.0), {"h": ["h_0", "h_1"]}, 0.5)
    cols = numeric_feature_selector(eng.drop(columns="vehicle_id"))
    assert {"171_0__delta", "171_0__rate", "h__mean_bin", "h_0__share"} <= set(cols)
    # select_feature_columns keeps them and still drops vehicle_id / labels
    selected = select_feature_columns(eng.assign(in_study_repair=1))
    assert "171_0__rate" in selected
    assert "vehicle_id" not in selected and "in_study_repair" not in selected
    with pytest.raises(ValueError, match="leakage"):
        numeric_feature_selector(eng)  # vehicle_id still inside


# --- imbalance strategies (30.10) ---------------------------------------------------


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[select_feature_columns(df)], df["repair_within_horizon"].to_numpy(dtype=int)


@pytest.mark.parametrize(
    ("strategy", "class_weight", "step_names"),
    [
        ("none", None, ["preprocess", "model"]),
        ("balanced", "balanced", ["preprocess", "model"]),
        ("smote", None, ["preprocess", "smote", "model"]),
    ],
)
def test_imbalance_strategy_builds_the_expected_pipeline(
    strategy: str, class_weight: str | None, step_names: list[str]
) -> None:
    pipe = build_logreg_pipeline(42, imbalance=strategy)
    assert [name for name, _ in pipe.steps] == step_names
    assert pipe.named_steps["model"].class_weight == class_weight
    assert isinstance(pipe, ImbPipeline) == (strategy == "smote")


def test_unknown_imbalance_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown imbalance strategy"):
        build_logreg_pipeline(42, imbalance="oversample")
    assert set(IMBALANCE_STRATEGIES) == {"none", "balanced", "smote"}


def test_default_strategy_reproduces_the_old_pipeline(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    default = build_logreg_pipeline(42).fit(x, y)
    explicit = build_logreg_pipeline(42, imbalance="balanced").fit(x, y)
    assert [n for n, _ in default.steps] == ["preprocess", "model"]
    assert default.named_steps["model"].class_weight == "balanced"
    np.testing.assert_allclose(default.predict_proba(x), explicit.predict_proba(x))


def test_smote_pipeline_fits_and_predict_proba_is_not_resampled(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    pipe = build_logreg_pipeline(42, imbalance="smote").fit(x, y)
    proba = pipe.predict_proba(x.iloc[:7])
    assert proba.shape == (7, 2)  # 7 rows in, 7 rows out: no resampling at predict time


class _SpySMOTE(SMOTE):
    """SMOTE that records how many rows it receives (class attribute survives clone)."""

    calls: list[int] = []  # noqa: RUF012

    def fit_resample(self, X, y, **params):  # noqa: N803
        _SpySMOTE.calls.append(len(X))
        return super().fit_resample(X, y, **params)


def test_smote_runs_only_on_the_training_part_of_each_fold(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    folds = make_group_folds(rows, n_splits=3, seed=42)
    pipe = build_logreg_pipeline(42, imbalance="smote")
    pipe.steps[1] = ("smote", _SpySMOTE(random_state=42))
    _SpySMOTE.calls.clear()
    cross_validate(pipe, x, y, cv=folds, scoring="average_precision", n_jobs=1)
    # One resampling per fold, on exactly the training rows; scoring the
    # validation part (predict_proba) never triggers it.
    assert _SpySMOTE.calls == [len(train) for train, _ in folds]


def test_smote_balances_only_the_training_data(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    sampler = SMOTE(random_state=42)
    x_res, y_res = sampler.fit_resample(
        build_logreg_pipeline(42).named_steps["preprocess"].fit_transform(x, y), y
    )
    assert len(y_res) > len(y)
    assert y_res.mean() == 0.5  # 50:50 after SMOTE; the original y is untouched
    assert y.mean() == 0.25


@pytest.mark.parametrize("strategy", ["none", "balanced", "smote"])
@pytest.mark.parametrize("leak", ["vehicle_id", "repair_within_horizon"])
def test_forbidden_column_stops_fit_for_every_strategy(
    rows: pd.DataFrame, strategy: str, leak: str
) -> None:
    x = rows[[*select_feature_columns(rows), leak]]
    y = rows["repair_within_horizon"].to_numpy(dtype=int)
    with pytest.raises(ValueError, match="leakage"):
        build_logreg_pipeline(42, imbalance=strategy).fit(x, y)


# --- missing values ----------------------------------------------------------------
def test_nan_at_predict_time_is_imputed(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    pipe = build_logreg_pipeline(42).fit(x.fillna(0.0), y)  # no NaN while fitting
    x_new = x.iloc[:5].copy()
    x_new[:] = np.nan  # every value missing at prediction time
    proba = pipe.predict_proba(x_new)
    assert proba.shape == (5, 2) and np.isfinite(proba).all()


def test_all_nan_column_in_training_is_dropped_not_an_error(rows: pd.DataFrame) -> None:
    """SimpleImputer cannot compute a median of nothing and drops the column.

    The pipeline keeps working (fit and predict use the same reduced width), but
    the model silently has one feature less. We keep this sklearn default on
    purpose: zero-filling (keep_empty_features=True) would feed constant noise.
    sklearn announces the drop with a UserWarning; the test expects it.
    """
    x, y = _xy(rows)
    x = x.copy()
    x["666_0"] = np.nan
    with pytest.warns(UserWarning, match="Skipping features without any observed values"):
        pipe = build_logreg_pipeline(42).fit(x, y)
        n_in = x.shape[1]
        assert pipe.named_steps["preprocess"].transform(x).shape[1] == n_in - 1
        x_new = x.iloc[:4].copy()
        x_new["666_0"] = 1.0  # values appear at predict time: ignored, no crash
        assert pipe.predict_proba(x_new).shape == (4, 2)


# --- log1p branch (30.10) -------------------------------------------------------------
@pytest.fixture
def skewed() -> pd.DataFrame:
    """Non-negative, right-skewed counters (+ time_step, one column with NaN)."""
    rng = np.random.default_rng(1)
    n = 80
    counter = rng.lognormal(mean=3.0, sigma=2.0, size=n)
    other = rng.lognormal(mean=1.0, sigma=1.0, size=n)
    other[::9] = np.nan
    return pd.DataFrame({"171_0": counter, "666_0": other, "time_step": rng.uniform(0, 100, n)})


def test_log1p_branch_matches_manual_log1p_then_scale(skewed: pd.DataFrame) -> None:
    pre = build_preprocessor(log1p_columns=["171_0"])
    out = pre.fit_transform(skewed)
    manual = np.log1p(skewed["171_0"].to_numpy())
    manual = (manual - manual.mean()) / manual.std()
    # transformer order: generic numeric branch first, then the log1p branch
    np.testing.assert_allclose(out[:, -1], manual, rtol=1e-6)
    assert [name for name, *_ in pre.transformers] == ["num", "num_log1p"]


def test_log1p_columns_are_not_processed_twice(skewed: pd.DataFrame) -> None:
    out = build_preprocessor(log1p_columns=["171_0"]).fit_transform(skewed)
    assert out.shape[1] == skewed.shape[1]  # 3 columns in, 3 out - not 4


def test_log1p_reduces_the_skew(skewed: pd.DataFrame) -> None:
    plain = build_preprocessor().fit_transform(skewed)[:, 0]
    logged = build_preprocessor(log1p_columns=["171_0"]).fit_transform(skewed)[:, -1]
    assert abs(pd.Series(logged).skew()) < abs(pd.Series(plain).skew())


def test_log1p_keeps_rows_and_nan_is_imputed_first(skewed: pd.DataFrame) -> None:
    pre = build_preprocessor(log1p_columns=["666_0"])  # the column with NaN
    out = pre.fit_transform(skewed)
    assert out.shape[0] == len(skewed) and np.isfinite(out).all()


def test_log1p_refuses_negative_values(skewed: pd.DataFrame) -> None:
    bad = skewed.copy()
    bad.loc[3, "171_0"] = -5.0
    with pytest.raises(ValueError, match="non-negative"):
        build_preprocessor(log1p_columns=["171_0"]).fit(bad)


def test_log1p_default_is_off_and_reproduces_the_old_model(rows: pd.DataFrame) -> None:
    x, y = _xy(rows)
    default = build_logreg_pipeline(42).fit(x, y)
    explicit = build_logreg_pipeline(42, log1p_columns=()).fit(x, y)
    assert [t[0] for t in default.named_steps["preprocess"].transformers] == ["num"]
    np.testing.assert_allclose(default.predict_proba(x), explicit.predict_proba(x))


@pytest.mark.parametrize("leak", ["vehicle_id", "repair_within_horizon"])
def test_log1p_columns_cannot_be_forbidden_columns(leak: str) -> None:
    with pytest.raises(ValueError, match="leakage"):
        build_preprocessor(log1p_columns=[leak])


def test_log1p_pipeline_fits_in_cross_validation(skewed: pd.DataFrame) -> None:
    y = (skewed["171_0"] > skewed["171_0"].median()).astype(int).to_numpy()
    groups = pd.DataFrame({"vehicle_id": np.arange(len(skewed)) // 2})
    folds = make_group_folds(groups.assign(repair_within_horizon=y), n_splits=3, seed=42)
    pipe = build_logreg_pipeline(42, log1p_columns=["171_0"])
    result = cross_validate(pipe, skewed, y, cv=folds, scoring="roc_auc", n_jobs=2)
    assert len(result["test_score"]) == 3
