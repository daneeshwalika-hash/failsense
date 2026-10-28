"""MLflow helpers shared by every training script in FailSense.

Why this module exists
----------------------
Without tracking, results live in terminal output and PR descriptions. After
ten experiments nobody knows which number came from which settings. MLflow
stores, for every *run*:

* **params**  - the settings (k cut-offs, seeds, number of folds, model),
* **metrics** - the scores (per fold, mean, standard deviation),
* **tags**    - free labels (task, dataset, stage). MLflow also adds the Git
  commit automatically when the script runs inside a Git repository.

Comparable to a log table in SQL Server: one row per run, which you can
filter, sort and compare in the UI (``scripts/mlflow_ui.sh``).

Naming convention for metrics
-----------------------------
``cross_validate`` calls the held-out part of each fold ``test_*``. In
FailSense that part is **validation**, not the test split (which is used once
at the very end). To avoid confusion we log it as ``val_*``:

* ``val_<metric>_fold`` (one value per fold, ``step`` = fold number),
* ``val_<metric>_mean`` / ``val_<metric>_std``,
* ``train_<metric>_mean`` / ``train_<metric>_std`` if train scores exist,
* ``<metric>_gap_mean`` = train mean - validation mean. A large gap points to
  overfitting; low train *and* validation scores point to underfitting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import mlflow
import numpy as np

from failsense.config import resolve_path

logger = logging.getLogger(__name__)

# cross_validate key prefix -> name used in MLflow.
_SPLIT_LABELS: dict[str, str] = {"test": "val", "train": "train"}


def tracking_params_from_config(config: Mapping[str, Any]) -> tuple[str, str, str]:
    """Read ``mlflow.db_path``, ``mlflow.artifact_dir`` and ``mlflow.experiment``.

    Returns:
        ``(db_path, artifact_dir, experiment_name)`` as written in the config
        (paths still relative to the repo root).

    Raises:
        KeyError: If the ``mlflow`` section or one of its keys is missing.
    """
    try:
        section = config["mlflow"]
        db_path = section["db_path"]
        artifact_dir = section["artifact_dir"]
        experiment = section["experiment"]
    except KeyError as exc:
        raise KeyError(
            "config needs 'mlflow.db_path', 'mlflow.artifact_dir' and 'mlflow.experiment'"
        ) from exc
    return str(db_path), str(artifact_dir), str(experiment)


def setup_mlflow(config: Mapping[str, Any], experiment: str | None = None) -> str:
    """Point MLflow at the project's SQLite database and select an experiment.

    Args:
        config: Project config (needs the ``mlflow`` section).
        experiment: Experiment name; ``None`` uses ``mlflow.experiment``.

    Returns:
        The experiment id.

    The experiment is created on first use with an explicit artifact folder,
    so artifacts (plots, models) never end up in a random working directory.
    Calling this function again reuses the existing experiment.
    """
    db_path, artifact_dir, default_experiment = tracking_params_from_config(config)
    name = experiment or default_experiment

    # resolve_path makes relative paths absolute (relative to the repo root).
    # An absolute path gives four slashes: sqlite:////home/... - that is correct.
    db_file = resolve_path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_file}")

    if mlflow.get_experiment_by_name(name) is None:
        # Path.as_uri() turns /home/x/mlartifacts into file:///home/x/mlartifacts
        artifact_uri = resolve_path(artifact_dir).as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_uri)
        logger.info("Created MLflow experiment %r (artifacts in %s)", name, artifact_uri)

    experiment_obj = mlflow.set_experiment(name)
    logger.info("MLflow tracking: %s, experiment %r", db_file, name)
    return experiment_obj.experiment_id


def log_cv_run(
    run_name: str,
    cv_result: Mapping[str, Any],
    metrics: Iterable[str],
    params: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log the output of ``sklearn.model_selection.cross_validate`` as one run.

    Args:
        run_name: Human-readable name shown in the UI, e.g. ``"logreg_balanced_k5"``.
        cv_result: The dict returned by ``cross_validate`` (``test_<metric>``
            is required; ``train_<metric>`` and ``fit_time`` are optional).
        metrics: Metric names as used in ``scoring``, e.g. ``["pr_auc", "roc_auc"]``.
        params: Settings of this run (stored as strings by MLflow).
        tags: Optional labels, e.g. ``{"task": "q1_binary"}``.

    Returns:
        The MLflow run id.

    Raises:
        ValueError: If ``metrics`` is empty, a ``test_<metric>`` key is
            missing, or a metric has no fold scores. The checks run **before**
            a run is started, so a bad call never leaves a half-filled run.
    """
    metric_names = list(metrics)
    if not metric_names:
        raise ValueError("metrics must not be empty")
    missing = [f"test_{m}" for m in metric_names if f"test_{m}" not in cv_result]
    if missing:
        raise ValueError(f"cv_result is missing keys: {missing}")
    for m in metric_names:
        if len(cv_result[f"test_{m}"]) == 0:
            raise ValueError(f"no fold scores for metric {m!r}")

    # `with` ends the run automatically, also if an exception happens inside.
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.set_tags(dict(tags or {}))

        for m in metric_names:
            means: dict[str, float] = {}
            for prefix, label in _SPLIT_LABELS.items():
                key = f"{prefix}_{m}"
                if key not in cv_result:  # train scores are optional
                    continue
                scores = np.asarray(cv_result[key], dtype=float)
                # One point per fold -> the UI draws a small chart over the folds.
                for fold_no, score in enumerate(scores):
                    mlflow.log_metric(f"{label}_{m}_fold", float(score), step=fold_no)
                means[label] = float(scores.mean())
                mlflow.log_metrics(
                    {f"{label}_{m}_mean": means[label], f"{label}_{m}_std": float(scores.std())}
                )
            if "train" in means:
                # Positive gap = the model scores better on data it has seen.
                mlflow.log_metric(f"{m}_gap_mean", means["train"] - means["val"])

        if "fit_time" in cv_result:
            mlflow.log_metric("fit_time_mean_s", float(np.mean(cv_result["fit_time"])))

        logger.info("Logged MLflow run %r (%s)", run_name, run.info.run_id)
        return run.info.run_id
