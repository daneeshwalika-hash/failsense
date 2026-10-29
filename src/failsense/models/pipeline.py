"""First scikit-learn Pipeline for Q1 (repair within the horizon, binary).

Why a Pipeline
--------------
Preprocessing steps *learn* from data: ``SimpleImputer`` learns the median of
each column, ``StandardScaler`` learns mean and standard deviation. If they
learn from the whole training set **before** cross validation, every
validation fold has already "seen" its own statistics: information leaks
from validation into training and the score is too optimistic.

A ``Pipeline`` bundles preprocessing and model into **one estimator**. When
``cross_validate`` calls ``pipeline.fit`` on the training part of a fold,
every step learns only from that part; the validation part is only
*transformed* with what was learned. The same object is later saved and used
by the API, so training and serving apply exactly the same preprocessing
(no train/serving skew).

What is in this module
----------------------
* ``FORBIDDEN_FEATURES`` / ``select_feature_columns``: the feature list never
  contains label columns or ``vehicle_id`` (leakage rule 1 in
  ``docs/problem_framing.md``).
* ``build_preprocessor`` (29.10): a ``ColumnTransformer`` that sends each
  column type to its own preprocessing:

  - numeric columns -> median imputation -> scaler (``"standard"`` or
    ``"minmax"``, chosen by parameter so both can be compared on the same folds);
  - categorical specifications -> ``OneHotEncoder(handle_unknown="ignore")``.
    Validation/test contain categories that never occur in training; "ignore"
    encodes them as all zeros instead of crashing at prediction time.

* ``build_logreg_pipeline``: preprocessor -> logistic regression with
  ``class_weight="balanced"`` (about 2 % positives).
* ``build_dummy_classifier``: the reference point. With ``strategy="prior"``
  it predicts the same probability for every truck, so its PR-AUC equals the
  share of positives. A real model must beat this number.

Feature engineering and other models follow on 30.10 and in week 2.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from failsense.data.labels import LABEL_COLUMNS, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS

logger = logging.getLogger(__name__)

# Columns that must never reach a model:
# * every label column (they *are* the answer, or are computed from it),
# * vehicle_id (an identifier: the model would learn "truck 4711 breaks",
#   which is useless for a new truck and leaks group information).
FORBIDDEN_FEATURES: frozenset[str] = LABEL_COLUMNS | {VEHICLE_COL}

# Scalers that can be compared. A factory (the class) instead of an instance,
# so every pipeline gets its own fresh, unfitted scaler.
# * standard: (x - mean) / std. Unbounded output, keeps outliers as large values.
# * minmax:   (x - min) / (max - min) -> [0, 1] on the training data. One extreme
#   value squeezes all other values into a tiny range; new data outside the
#   training range ends up outside [0, 1].
SCALERS: dict[str, Callable[[], TransformerMixin]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    """Raise if any forbidden column is in the feature list.

    Called right before fitting, so a leakage mistake stops the run instead
    of producing a suspiciously good score.
    """
    found = sorted(set(columns) & FORBIDDEN_FEATURES)
    if found:
        raise ValueError(f"forbidden columns in features (leakage): {found}")


def select_feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Return the numeric feature columns of ``df`` in their original order.

    Args:
        df: Readouts (possibly already joined with labels).
        exclude: Extra columns to leave out, e.g. ``["time_step"]`` for the
            with/without comparison of the selection-effect check (30.10).

    The result never contains ``FORBIDDEN_FEATURES``. Non-numeric columns are
    skipped here; categorical specifications get their own encoder later.
    """
    skip = FORBIDDEN_FEATURES | set(exclude)
    columns = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    assert_no_forbidden_features(columns)  # defensive: should always pass
    logger.info(
        "Selected %d feature columns (%d excluded)", len(columns), len(df.columns) - len(columns)
    )
    return columns


def select_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the specification columns (``Spec_0`` ... ``Spec_7``) present in ``df``."""
    columns = [c for c in SPEC_COLUMNS if c in df.columns]
    assert_no_forbidden_features(columns)  # defensive: should always pass
    return columns


_select_numeric = make_column_selector(dtype_include="number")


def numeric_feature_selector(x: pd.DataFrame) -> list[str]:
    """Pick the numeric columns of ``x`` and refuse forbidden ones.

    Used inside the ``ColumnTransformer``. ``make_column_selector`` alone would
    take *any* numeric column, so passing the whole frame (with ``vehicle_id``
    or a label column) by mistake would leak silently. Here it stops the fit.
    A module-level function (not a lambda), so ``n_jobs=-1`` can pickle it.
    """
    columns = _select_numeric(x)
    assert_no_forbidden_features(columns)
    return columns


def build_preprocessor(
    categorical_columns: Sequence[str] = (), scaler: str = "standard"
) -> ColumnTransformer:
    """Route numeric and categorical columns to their own preprocessing.

    Args:
        categorical_columns: Text columns to one-hot encode (the specs).
            Empty = numeric features only (the setup of 27./28.10).
        scaler: Key of ``SCALERS`` (``"standard"`` or ``"minmax"``).

    Numeric columns are picked by dtype (``numeric_feature_selector``), so the
    caller only passes the feature matrix; the spec columns are text and are
    therefore never picked up as numeric. Forbidden columns raise a ValueError.

    * ``SimpleImputer(strategy="median")``: robust to the skewed, cumulative
      counters; missing values are rare (<= 0.86 %).
    * Scaler: logistic regression is sensitive to the scale of its inputs
      (regularisation treats all coefficients the same).
    * ``OneHotEncoder(handle_unknown="ignore")``: one 0/1 column per category;
      an unseen category becomes all zeros. One-hot columns are not scaled:
      they are already 0/1.

    ``remainder="drop"``: any column not named here never reaches the model.

    Raises:
        ValueError: If ``scaler`` is not a key of ``SCALERS``.
    """
    if scaler not in SCALERS:
        raise ValueError(f"unknown scaler {scaler!r}, choose one of {sorted(SCALERS)}")

    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", SCALERS[scaler]()),
        ]
    )
    transformers: list[tuple[str, object, object]] = [
        ("num", numeric, numeric_feature_selector),
    ]
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logreg_pipeline(
    random_state: int,
    categorical_columns: Sequence[str] = (),
    scaler: str = "standard",
) -> Pipeline:
    """Preprocessor -> logistic regression, as one estimator.

    With the defaults (no categorical columns, standard scaler) this is the
    same model as on 27./28.10, so earlier results stay reproducible.

    * ``class_weight="balanced"``: errors on the rare positive class weigh
      more, otherwise the model may just predict "no repair" for everyone.
    * ``max_iter=1000``: the default (100) often stops before converging on
      unscaled or many features; a ConvergenceWarning would be a red flag.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical_columns, scaler)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=random_state
                ),
            ),
        ]
    )


def build_dummy_classifier() -> DummyClassifier:
    """Baseline that ignores the features and predicts the class prior."""
    return DummyClassifier(strategy="prior")
