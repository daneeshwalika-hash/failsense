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

* ``build_logreg_pipeline``: preprocessor -> logistic regression. The
  imbalance strategy (about 2.8 % positives) is chosen by parameter:
  ``"none"``, ``"balanced"`` (default) or ``"smote"``.
* ``build_dummy_classifier``: the reference point. With ``strategy="prior"``
  it predicts the same probability for every truck, so its PR-AUC equals the
  share of positives. A real model must beat this number.

Feature engineering and other models follow on 30.10 and in week 2.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from failsense.data.labels import LABEL_COLUMNS, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, feature_columns
from failsense.features.build import engineered_columns

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


# Ways to handle the rare positive class (about 2.8 % of the rows):
# * none:     no correction; the model is rewarded for predicting "no repair".
# * balanced: class_weight="balanced" - errors on positives weigh ~35x more.
#             No new rows; only the loss changes.
# * smote:    synthetic positives are interpolated between real positives so
#             the training part becomes 50:50 (see build_logreg_pipeline).
IMBALANCE_STRATEGIES: tuple[str, ...] = ("none", "balanced", "smote")


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


@dataclass(frozen=True)
class FeatureColumns:
    """Which columns of a training frame go where (one definition for CV and training).

    * ``raw``: numeric raw columns incl. ``time_step`` (kept until the boosting
      re-check on 04.11); never labels or ``vehicle_id``.
    * ``engineered``: counter deltas/rates and histogram features.
    * ``log1p``: the raw counters and histogram counts (non-negative by the data
      contract) that get ``log1p`` before the scaler.
    * ``categorical``: the specification columns (one-hot encoded).
    """

    raw: list[str]
    engineered: list[str]
    log1p: list[str]
    categorical: list[str]


def split_feature_columns(frame: pd.DataFrame) -> FeatureColumns:
    """Sort the columns of a training frame into the groups of ``FeatureColumns``.

    ``baseline_cv`` (cross validation) and ``train`` (final fit) both call this,
    so the model that is evaluated and the model that is saved always use the
    same columns. Forbidden columns are dropped by ``select_feature_columns``.
    """
    engineered = engineered_columns(list(frame.columns))
    raw = select_feature_columns(frame, exclude=engineered)
    raw_names = set(feature_columns())
    return FeatureColumns(
        raw=raw,
        engineered=engineered,
        log1p=[c for c in raw if c in raw_names],
        categorical=select_categorical_columns(frame),
    )


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


def _numeric_excluding(x: pd.DataFrame, exclude: frozenset[str]) -> list[str]:
    """Numeric columns of ``x`` (forbidden ones refused) without ``exclude``."""
    return [c for c in numeric_feature_selector(x) if c not in exclude]


def _log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    """``log(1 + x)`` for non-negative values; raises on negative ones.

    The raw counters and histogram counts are >= 0 by the data contract. For
    x < -1 ``log1p`` would silently return NaN, so a contract violation stops
    the pipeline instead of becoming a missing value. NaN stays NaN.
    """
    arr = np.asarray(values, dtype=float)
    if (arr < 0).any():
        raise ValueError("log1p needs non-negative values, found negative ones")
    return np.log1p(arr)


def build_preprocessor(
    categorical_columns: Sequence[str] = (),
    scaler: str | None = "standard",
    log1p_columns: Sequence[str] = (),
) -> ColumnTransformer:
    """Route numeric and categorical columns to their own preprocessing.

    Args:
        categorical_columns: Text columns to one-hot encode (the specs).
            Empty = numeric features only (the setup of 27./28.10).
        scaler: Key of ``SCALERS`` (``"standard"`` or ``"minmax"``), or ``None`` for no
            scaling at all. Trees only compare a value with a threshold, so any
            monotone transformation leaves them unchanged; Gaussian naive Bayes
            standardises every feature on its own. Distance-based models (kNN) and
            penalised linear models need a scaler.
        log1p_columns: Non-negative numeric columns that get ``log1p`` between
            imputation and scaling (the cumulative counters and histogram
            counts are extremely right-skewed: a few old trucks have values
            far above the rest, which dominate a scaler's mean and variance).
            ``log1p`` keeps the order of the rows and every row, it only
            compresses the large values - no clipping, no removal, and no
            threshold learned from data, so it cannot leak. Empty = off.

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
        ValueError: If ``scaler`` is not a key of ``SCALERS``, or
            ``log1p_columns`` contains a forbidden column.
    """
    if scaler is not None and scaler not in SCALERS:
        raise ValueError(f"unknown scaler {scaler!r}, choose one of {sorted(SCALERS)}")
    assert_no_forbidden_features(log1p_columns)

    def scale_step() -> list[tuple[str, TransformerMixin]]:
        """The scaler as a pipeline step (a fresh instance each time), or nothing."""
        return [] if scaler is None else [("scale", SCALERS[scaler]())]

    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), *scale_step()])
    # The log columns get their own branch; the generic branch takes every
    # other numeric column, so no column is processed twice.
    transformers: list[tuple[str, object, object]] = [
        ("num", numeric, partial(_numeric_excluding, exclude=frozenset(log1p_columns))),
    ]
    if log1p_columns:
        logged = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one")),
                *scale_step(),
            ]
        )
        transformers.append(("num_log1p", logged, list(log1p_columns)))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


# Penalties of the logistic regression, in our own readable names.
# scikit-learn >= 1.8 no longer wants ``penalty=...`` (deprecated, removed in 1.10);
# the penalty is chosen by ``l1_ratio``: 0 = L2, 1 = L1, in between = ElasticNet.
PENALTIES: tuple[str, ...] = ("l2", "l1", "elasticnet")


def logreg_penalty_settings(
    penalty: str,
    C: float,  # noqa: N803 - `C` is the name in scikit-learn and the literature
    l1_ratio: float | None,
) -> dict[str, float | str]:
    """Translate (penalty, C, l1_ratio) into ``LogisticRegression`` arguments.

    * ``"l2"``: ``l1_ratio=0``, solver ``lbfgs`` (the default since 27.10, so earlier
      results reproduce exactly).
    * ``"l1"``: ``l1_ratio=1``, solver ``saga``.
    * ``"elasticnet"``: ``0 < l1_ratio < 1``, solver ``saga``.

    Why ``saga`` for L1 and ElasticNet: according to the scikit-learn 1.9 solver
    table, ``lbfgs`` handles only L2, ``liblinear`` only L1 or L2, and only ``saga``
    supports all three. Using ``saga`` for both L1 and ElasticNet means they differ
    only in the penalty, not in the optimiser. ``saga`` converges fast only on
    features of similar scale, which the scaler inside the Pipeline provides.

    Raises:
        ValueError: If ``penalty`` is unknown, ``C <= 0``, ``l1_ratio`` is missing
            or outside (0, 1) for ElasticNet, or given for L1/L2 (it would be ignored
            silently otherwise).
    """
    if penalty not in PENALTIES:
        raise ValueError(f"unknown penalty {penalty!r}, choose one of {list(PENALTIES)}")
    if not C > 0:
        raise ValueError(f"C must be positive, got {C}")
    if penalty == "elasticnet":
        if l1_ratio is None or not 0 < l1_ratio < 1:
            raise ValueError(f"elasticnet needs 0 < l1_ratio < 1, got {l1_ratio}")
        return {"C": C, "l1_ratio": float(l1_ratio), "solver": "saga"}
    if l1_ratio is not None:
        raise ValueError(f"l1_ratio is only used with penalty='elasticnet', got {penalty!r}")
    if penalty == "l1":
        return {"C": C, "l1_ratio": 1.0, "solver": "saga"}
    return {"C": C, "l1_ratio": 0.0, "solver": "lbfgs"}


def build_logreg_pipeline(
    random_state: int,
    categorical_columns: Sequence[str] = (),
    scaler: str = "standard",
    imbalance: str = "balanced",
    log1p_columns: Sequence[str] = (),
    penalty: str = "l2",
    C: float = 1.0,  # noqa: N803 - same name as in scikit-learn
    l1_ratio: float | None = None,
) -> Pipeline:
    """Preprocessor (-> SMOTE) -> logistic regression, as one estimator.

    With the defaults (no categorical columns, standard scaler, ``"balanced"``)
    this is the same model as on 27./28.10, so earlier results stay reproducible.

    Args:
        random_state: Seed of the model (and of SMOTE).
        categorical_columns: Text columns to one-hot encode (the specs).
        scaler: Key of ``SCALERS``.
        imbalance: One of ``IMBALANCE_STRATEGIES``.
        log1p_columns: Columns that get ``log1p`` before the scaler (see
            ``build_preprocessor``). Empty = off.
        penalty: ``"l2"`` (default, ridge: shrinks all coefficients), ``"l1"`` (lasso:
            drives some coefficients to exactly 0 = built-in feature selection) or
            ``"elasticnet"`` (mix of both, needs ``l1_ratio``).
        C: Inverse regularisation strength: smaller = stronger penalty. The default 1.0
            is the setting of all results before 02.11.
        l1_ratio: Share of L1 in ElasticNet, ``0 < l1_ratio < 1``; ``None`` otherwise.

    * ``"none"``: ``class_weight=None``.
    * ``"balanced"``: ``class_weight="balanced"``. Errors on the rare positive
      class weigh more, otherwise the model may just predict "no repair".
    * ``"smote"``: ``preprocess -> SMOTE -> LogisticRegression(class_weight=None)``.
      SMOTE creates synthetic positives on the line between a real positive and
      one of its 5 nearest positive neighbours, up to a 50:50 ratio.
      It must run **after** imputation and scaling (it computes distances, and
      cannot handle NaN or text) and **only on the training part of a fold**:
      the imblearn ``Pipeline`` applies the sampler in ``fit`` and skips it in
      ``predict`` / ``predict_proba``, so validation rows are never resampled.
      SMOTE *before* cross validation is leakage: synthetic points would be
      interpolated from validation positives and land in the training part, so
      the model would "know" its validation rows. The one-hot spec columns get
      fractional values (e.g. 0.4) in synthetic rows - a known limit of plain
      SMOTE (``SMOTENC`` would handle categories).

    ``max_iter=1000``: the default (100) often stops before converging on
    unscaled or many features; a ConvergenceWarning would be a red flag.

    Raises:
        ValueError: If ``imbalance`` is not in ``IMBALANCE_STRATEGIES`` or the
            penalty settings are invalid (see ``logreg_penalty_settings``).
    """
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"unknown imbalance strategy {imbalance!r}, choose one of {list(IMBALANCE_STRATEGIES)}"
        )
    penalty_settings = logreg_penalty_settings(penalty, C, l1_ratio)
    preprocess = build_preprocessor(categorical_columns, scaler, log1p_columns)
    model = LogisticRegression(
        class_weight="balanced" if imbalance == "balanced" else None,
        max_iter=1000,
        random_state=random_state,
        **penalty_settings,
    )
    if imbalance == "smote":
        return ImbPipeline(
            steps=[
                ("preprocess", preprocess),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


# Kind of a preprocessed column, from the name prefix the ColumnTransformer gives it.
# (prefix, kind when the numeric columns are scaled, kind without scaling)
_COLUMN_KINDS = (
    ("num_log1p__", "numeric, log1p + scaled", "numeric, log1p"),
    ("num__", "numeric, scaled", "numeric"),
    ("cat__", "one-hot, NOT scaled", "one-hot"),
)


def column_kind(name: str, scaled: bool = True) -> str:
    """Kind of a preprocessed column from its ColumnTransformer prefix (or ``"other"``).

    ``scaled=False`` is for models without a scaler (trees), where "scaled" would be wrong.
    """
    for prefix, kind_scaled, kind_plain in _COLUMN_KINDS:
        if name.startswith(prefix):
            return kind_scaled if scaled else kind_plain
    return "other"


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted logistic-regression pipeline with column names.

    Returns:
        One row per column the model sees (after preprocessing): ``feature``
        (``num__``/``num_log1p__``/``cat__`` prefix from the ColumnTransformer),
        ``coef`` and ``kind``. Numeric columns are standardized, so their
        coefficients are comparable with each other; one-hot columns are 0/1 and
        **not scaled**, so their coefficients live on a different scale.

    Raises:
        ValueError: If the pipeline has no fitted ``model`` step with ``coef_``.
    """
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        raise ValueError("pipeline needs a fitted logistic-regression step named 'model'")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = model.coef_.ravel()
    return pd.DataFrame({"feature": names, "coef": coefs, "kind": [column_kind(n) for n in names]})


def count_nonzero_coefficients(pipeline: Pipeline) -> tuple[int, int]:
    """Return ``(non-zero coefficients, all coefficients)`` of a fitted pipeline."""
    coefs = coefficient_table(pipeline)["coef"].to_numpy()
    return int((coefs != 0).sum()), int(coefs.size)


def build_model_pipeline(
    estimator: Any,
    categorical_columns: Sequence[str] = (),
    scaler: str | None = "standard",
    log1p_columns: Sequence[str] = (),
) -> Pipeline:
    """Preprocessor -> any classifier, as one estimator (steps ``preprocess``, ``model``).

    Used to compare model families on identical inputs. Every learned step (imputer,
    scaler, encoder) sits inside the Pipeline and is fitted per fold. ``scaler=None``
    for models that do not need one (trees, naive Bayes).
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical_columns, scaler, log1p_columns)),
            ("model", estimator),
        ]
    )


def build_dummy_classifier() -> DummyClassifier:
    """Baseline that ignores the features and predicts the class prior."""
    return DummyClassifier(strategy="prior")
