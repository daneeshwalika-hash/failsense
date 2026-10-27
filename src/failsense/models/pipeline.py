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

What is in this module (deliberately small, course day 27.10)
-------------------------------------------------------------
* ``FORBIDDEN_FEATURES`` / ``select_feature_columns``: the feature list never
  contains label columns or ``vehicle_id`` (leakage rule 1 in
  ``docs/problem_framing.md``).
* ``build_logreg_pipeline``: median imputation -> scaling -> logistic
  regression with ``class_weight="balanced"`` (about 2 % positives).
* ``build_dummy_classifier``: the reference point. With ``strategy="prior"``
  it predicts the same probability for every truck, so its PR-AUC equals the
  share of positives. A real model must beat this number.

ColumnTransformer (numeric + categorical specifications), feature
engineering and other models follow on 29.10, 30.10 and in week 2.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from failsense.data.labels import LABEL_COLUMNS, VEHICLE_COL

logger = logging.getLogger(__name__)

# Columns that must never reach a model:
# * every label column (they *are* the answer, or are computed from it),
# * vehicle_id (an identifier: the model would learn "truck 4711 breaks",
#   which is useless for a new truck and leaks group information).
FORBIDDEN_FEATURES: frozenset[str] = LABEL_COLUMNS | {VEHICLE_COL}


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


def build_logreg_pipeline(random_state: int) -> Pipeline:
    """Imputer -> scaler -> logistic regression, as one estimator.

    * ``SimpleImputer(strategy="median")``: the median is robust to the
      skewed, cumulative counters; missing values are rare (<= 0.86 %).
    * ``StandardScaler``: logistic regression is sensitive to the scale of
      the inputs (regularisation treats all coefficients the same).
    * ``class_weight="balanced"``: errors on the rare positive class weigh
      more, otherwise the model may just predict "no repair" for everyone.
    * ``max_iter=1000``: the default (100) often stops before converging on
      unscaled or many features; a ConvergenceWarning would be a red flag.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
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
