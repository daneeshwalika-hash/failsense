"""Fit the current logistic-regression reference on the training split and log it.

Run it with::

    uv run python -m failsense.models.train             # or: make train
    uv run python -m failsense.models.train --no-mlflow # fit only, log nothing

CV vs. final fit
----------------
``baseline_cv`` answers *"how good is this model on trucks it has not seen?"*:
it trains five times on four fifths of the trucks. It does not produce a model
to keep. This script fits the **same pipeline once on all training trucks**
(cut-offs from the config), because the model that is saved, registered and
served later should learn from as much data as possible. The pipeline is
unchanged; only the number of fits differs.

Reference model (decided 05.11): specifications one-hot encoded, class weight
"balanced", ``log1p`` on the raw counters and histogram counts, no engineered
features (they did not help, see ``docs/problem_framing.md`` §6). ``time_step``
stays in until the boosting re-check on 04.11.

Only the **training** split is read. No validation or test data is loaded, so
this script cannot tune anything on them. The score logged here is the
**in-sample** PR-AUC on the rows the model was fitted on: a sanity check that
the fit worked, not an estimate of quality (that is the CV number).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from failsense.config import load_config
from failsense.data.labels import BINARY_COL, VEHICLE_COL
from failsense.data.sampling import sampling_params_from_config
from failsense.models.baseline_cv import load_training_frame, n_cutoffs_label
from failsense.models.pipeline import (
    FeatureColumns,
    build_logreg_pipeline,
    split_feature_columns,
)
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# MLflow saves sklearn models in the ``skops`` format, which refuses to load any
# type it does not know unless it is listed. These are exactly the project's own
# pipeline pieces (the log1p function, the numeric column selector with its
# ``partial`` and ``frozenset``) plus numpy's dtype. Listing them keeps the safer
# skops format instead of falling back to pickle, which can run arbitrary code on
# load. If a new pipeline piece is added, test_run_logs_... fails and says which.
SKOPS_TRUSTED_TYPES = [
    "builtins.frozenset",
    "functools.partial",
    "numpy.dtype",
    "failsense.models.pipeline._log1p_nonnegative",
    "failsense.models.pipeline._numeric_excluding",
]

RUN_NAME = "logreg_reference_train"
RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "train"}


def fit_reference_model(
    frame: pd.DataFrame, random_state: int
) -> tuple[Pipeline, FeatureColumns, pd.DataFrame]:
    """Fit specs + balanced + log1p logistic regression on all rows of ``frame``.

    Args:
        frame: Training frame from ``load_training_frame`` (features, specs and
            the binary label; ``vehicle_id`` may be present but is never used).
        random_state: Seed of the model.

    Returns:
        The fitted pipeline, the column groups it uses, and the model input
        ``x`` (raw numeric + specification columns) it was fitted on.

    Raises:
        ValueError: If ``frame`` is empty or has no label column.
    """
    if BINARY_COL not in frame.columns:
        raise ValueError(f"frame is missing the label column {BINARY_COL!r}")
    if frame.empty:
        raise ValueError("frame is empty: nothing to fit")
    cols = split_feature_columns(frame)
    x = frame[cols.raw + cols.categorical]
    y = frame[BINARY_COL].to_numpy()
    pipeline = build_logreg_pipeline(
        random_state=random_state,
        categorical_columns=cols.categorical,
        log1p_columns=cols.log1p,
    )
    pipeline.fit(x, y)
    return pipeline, cols, x


def run(config: Mapping[str, Any], track: bool = True) -> Pipeline:
    """Load the training cut-offs, fit the reference model, optionally log it to MLflow.

    ``setup_mlflow`` must have been called when ``track`` is true. Default for
    tests would be ``track=False`` so nothing writes to the project database.
    """
    n_per_vehicle, sampling_seed = sampling_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    pipeline, cols, x = fit_reference_model(frame, random_state=int(config["seed"]))
    y = frame[BINARY_COL].to_numpy()
    in_sample = float(average_precision_score(y, pipeline.predict_proba(x)[:, 1]))
    logger.info(
        "Fitted reference model on %d rows / %d trucks; in-sample PR-AUC %.4f "
        "(not a quality estimate)",
        len(frame),
        frame[VEHICLE_COL].nunique(),
        in_sample,
    )

    if track:
        with mlflow.start_run(run_name=RUN_NAME) as active:
            mlflow.set_tags(RUN_TAGS)
            mlflow.log_params(
                {
                    "model": "logreg_specs_balanced_log1p",
                    "imbalance_strategy": "balanced",
                    "log1p": True,
                    "engineered": False,
                    "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle),
                    "sampling_seed": sampling_seed,
                    "seed": int(config["seed"]),
                    "n_rows": len(frame),
                    "n_vehicles": int(frame[VEHICLE_COL].nunique()),
                    "n_numeric_features": len(cols.raw),
                    "n_log1p_features": len(cols.log1p),
                    "n_categorical_features": len(cols.categorical),
                    "positive_rate": round(float(y.mean()), 5),
                    "uses_time_step": "time_step" in cols.raw,
                }
            )
            mlflow.log_metric("pr_auc_in_sample", in_sample)
            # The example gives the model its input schema (column names and types).
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=x.head(5),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            logger.info("Logged MLflow run %r (%s)", RUN_NAME, active.info.run_id)
    return pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="fit only, do not log to MLflow",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(config, track=track)


if __name__ == "__main__":
    main()
