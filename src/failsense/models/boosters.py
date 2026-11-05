"""XGBoost / LightGBM with early stopping on a truck-grouped inner split (tuning helpers).

Early stopping decides the number of boosting rounds from a validation set. That set must
(1) be preprocessed like the training data (so it cannot sit inside a scikit-learn Pipeline)
and (2) contain whole trucks: a random split of the training *rows* puts readouts of the same
truck on both sides and the stopping point would be chosen on trucks the model has seen.
The inner split therefore lives in ``failsense.models.tuning_data`` (grouped by
``vehicle_id``) and these helpers only fit and predict.

The step-6 settings (``STEP6_PARAMS``) are the untuned reference: 300 fixed rounds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FAMILIES = ("xgboost", "lightgbm")

# The untuned step-6 models (compare_models_cv): learning rate 0.05, 300 rounds, no class
# weights, no early stopping. Used as the tuning anchor (trial 0) and the evaluation reference.
STEP6_PARAMS: dict[str, dict[str, float | int]] = {
    "xgboost": {
        "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1.0, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.001,
    },
    "lightgbm": {
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 0.01, "reg_alpha": 0.001,
    },
}  # fmt: skip
STEP6_ROUNDS = 300


def make_booster(
    family: str,
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    n_estimators: int,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier | LGBMClassifier:
    """A booster with ``params``; ``early_stopping_rounds`` is only for the tuned fits.

    No class weights anywhere. ``n_estimators`` is the maximum with early stopping and the
    exact number of rounds without it.

    Raises:
        ValueError: For an unknown family.
    """
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=n_estimators, tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=early_stopping_rounds, importance_type="gain",
            n_jobs=n_jobs, random_state=seed, verbosity=0, **params,
        )  # fmt: skip
    if family == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_estimators, subsample_freq=1, importance_type="gain",
            deterministic=True, n_jobs=n_jobs, random_state=seed, verbose=-1, **params,
        )  # fmt: skip
    raise ValueError(f"unknown family {family!r}, choose one of {list(FAMILIES)}")


def fit_early_stopping(
    family: str,
    params: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_jobs: int,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[XGBClassifier | LGBMClassifier, int]:
    """Fit on the inner-train rows, stop on the inner-validation rows (average precision).

    Returns:
        The fitted booster (it predicts with its best iteration) and the number of
        rounds that early stopping chose.
    """
    model = make_booster(family, params, seed, n_jobs, max_rounds, early_stopping_rounds)
    if family == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model, int(model.best_iteration) + 1
    model.fit(
        x_train, y_train, eval_X=x_val, eval_y=y_val, eval_metric="average_precision",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )  # fmt: skip
    return model, int(model.best_iteration_)


def fit_fixed_rounds(
    family: str,
    params: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    rounds: int,
    seed: int,
    n_jobs: int,
) -> XGBClassifier | LGBMClassifier:
    """Fit exactly ``rounds`` rounds, no validation data at all (untuned and final models)."""
    model = make_booster(family, params, seed, n_jobs, rounds)
    model.fit(x_train, y_train)
    return model
