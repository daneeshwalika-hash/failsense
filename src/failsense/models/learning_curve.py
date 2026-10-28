"""Learning curve for the Q1 baseline: is the model limited by bias or by variance?

Run it with::

    uv run python -m failsense.models.learning_curve
    uv run python -m failsense.models.learning_curve --no-mlflow

The idea (course day 28.10)
---------------------------
Train the same Pipeline on growing parts of the training data (2 %, 5 %,
10 %, 25 %, 50 %, 100 %) and score each fit twice:

* on the data it was trained on (**train** score),
* on the validation fold it has never seen (**validation** score).

The shape of the two curves tells you what to do next:

=====================================  ==========================  =========================
Picture                                Diagnosis                   What helps
=====================================  ==========================  =========================
big gap, validation still rising       high variance (overfitting) more data, regularisation
both low, close together, both flat    high bias (underfitting)    better features, stronger
                                                                   model - NOT more data
both high, close together              good fit                    stop here
=====================================  ==========================  =========================

Evaluation is the same as in ``compare_cutoffs``: training rows are the ``k``
cut-offs per truck from the config, validation is one cut-off per truck with
fixed folds by vehicle. Only the *training* part of each fold is reduced;
the validation part stays complete, so every point is measured on the same exam.

Outputs
-------
* ``reports/figures/learning_curve_q1_logreg.png`` (committed, for README/docs)
* ``reports/learning_curve_q1_logreg.csv`` (the numbers behind the plot)
* one MLflow run with the curve as metrics (``step`` = training rows) and both
  files as artifacts.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullLocator
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, VEHICLE_COL, class_bounds_from_config
from failsense.data.sampling import sampling_params_from_config
from failsense.data.split import Fold, cv_params_from_config
from failsense.models.baseline_cv import (
    build_training_frame,
    n_cutoffs_label,
    read_table,
    table_path,
)
from failsense.models.compare_cutoffs import assign_vehicle_folds, build_fixed_eval_folds
from failsense.models.pipeline import build_logreg_pipeline, select_feature_columns
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# Fractions of the (largest) training part of a fold.
DEFAULT_TRAIN_SIZES: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)

FIGURE_NAME = "learning_curve_q1_logreg.png"
TABLE_NAME = "learning_curve_q1_logreg.csv"

RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "learning_curve"}

# Colors: validated categorical slots 1/2 (blue/orange; CVD-safe pair) and
# neutral ink for text, grid and the dummy reference line.
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def compute_learning_curve(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: np.ndarray,
    folds: Sequence[Fold],
    train_sizes: Sequence[float] = DEFAULT_TRAIN_SIZES,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Score ``estimator`` on growing training subsets of every fold.

    Args:
        estimator: An unfitted estimator or Pipeline (cloned for every fit).
        x, y: All rows referenced by ``folds``.
        folds: ``(train_pos, val_pos)`` pairs, e.g. from ``build_fixed_eval_folds``.
        train_sizes: Fractions in (0, 1] of the training part of a fold.
        seed: Seed for shuffling the training rows before taking a subset.
            Without shuffling, "2 %" would be the first trucks by id only.
        n_jobs: Parallel fits (-1 = all cores).

    Returns:
        One row per size with ``train_rows``, ``train_mean``, ``train_std``,
        ``val_mean``, ``val_std`` (PR-AUC = average precision).

    Raises:
        ValueError: If ``train_sizes`` is empty or a value is not in (0, 1].
    """
    sizes = np.asarray(list(train_sizes), dtype=float)
    if sizes.size == 0 or (sizes <= 0).any() or (sizes > 1).any():
        raise ValueError(f"train_sizes must be fractions in (0, 1], got {list(train_sizes)}")

    sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        x,
        y,
        cv=list(folds),
        train_sizes=sizes,
        scoring="average_precision",
        shuffle=True,
        random_state=seed,
        n_jobs=n_jobs,
        # The default (error_score=nan) silently turns a failed fit into NaN,
        # e.g. a subset with only one class. We want a loud error instead.
        error_score="raise",
    )
    # Score arrays have the shape (n_sizes, n_folds): mean/std over the folds.
    return pd.DataFrame(
        {
            "train_rows": sizes_abs.astype(int),
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
        }
    )


def plot_learning_curve(
    curve: pd.DataFrame,
    baseline: float,
    path: Path,
    title: str = "Learning curve - Q1 logistic regression (PR-AUC)",
) -> Path:
    """Draw train and validation curves (mean +- std over folds) and save a PNG.

    Args:
        curve: Output of ``compute_learning_curve``.
        baseline: PR-AUC of a dummy model (= share of positives), drawn as a
            dashed reference line: everything above it is real signal.
        path: Target file; parent folders are created.
        title: Figure title.

    Uses ``matplotlib.figure.Figure`` directly (no ``pyplot``): no global
    state and no GUI backend, so it works in scripts, tests, Docker and CI.
    """
    fig = Figure(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    rows = curve["train_rows"].to_numpy()

    # (column prefix, label, color, vertical label offset in points): train label
    # above, validation label below, so they never overlap when the curves meet.
    series = (("train", "train", COLOR_TRAIN, 7), ("val", "validation", COLOR_VAL, -7))
    for prefix, label, color, dy in series:
        mean = curve[f"{prefix}_mean"].to_numpy()
        std = curve[f"{prefix}_std"].to_numpy()
        ax.fill_between(rows, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(rows, mean, color=color, linewidth=2, marker="o", markersize=6, label=label)
        # Direct label at the last point, so identity is not color-only.
        ax.annotate(
            f"{label} {mean[-1]:.3f}",
            xy=(rows[-1], mean[-1]),
            xytext=(8, dy),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    ax.axhline(baseline, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"dummy (share of positives) {baseline:.3f}",
        xy=(rows[0], baseline),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )

    # Log scale: the sizes grow by factors (2 % ... 100 %).
    ax.set_xscale("log")
    # Ticks exactly at the measured sizes, written as plain numbers (1,870 not 10^3).
    ax.set_xticks(rows)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.set_xlabel("training rows (cut-offs) per fold", color=INK_SECONDARY)
    ax.set_ylabel("PR-AUC (average precision)", color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=rows[-1] * 1.8)  # room for the direct labels
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved learning curve figure to %s", path)
    return path


def log_learning_curve(
    curve: pd.DataFrame,
    artifact_paths: Sequence[Path],
    params: Mapping[str, Any],
    run_name: str = "logreg_balanced_learning_curve",
) -> str:
    """Log the curve as one MLflow run: metrics per size + files as artifacts.

    Metrics use ``step`` = number of training rows, so the MLflow UI can draw
    the curve itself (x axis "Step"). ``setup_mlflow`` must be called first.

    Returns:
        The MLflow run id.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.set_tags(RUN_TAGS)
        for row in curve.itertuples(index=False):
            step = int(row.train_rows)
            mlflow.log_metric("lc_train_pr_auc", float(row.train_mean), step=step)
            mlflow.log_metric("lc_val_pr_auc", float(row.val_mean), step=step)
        last = curve.iloc[-1]
        mlflow.log_metrics(
            {
                "val_pr_auc_full": float(last["val_mean"]),
                "pr_auc_gap_full": float(last["train_mean"] - last["val_mean"]),
            }
        )
        for path in artifact_paths:
            # artifact_path = sub-folder inside the run's artifact store.
            mlflow.log_artifact(str(path), artifact_path="learning_curve")
        logger.info("Logged MLflow run %r (%s)", run_name, run.info.run_id)
        return run.info.run_id


def run(config: Mapping[str, Any], track: bool = False) -> pd.DataFrame:
    """Build data and fixed folds, compute, plot, save and (optionally) log the curve."""
    k, seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    bounds = class_bounds_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    eval_frame = build_training_frame(readouts, tte, bounds, 1, seed)
    train_frame = build_training_frame(readouts, tte, bounds, k, seed)
    del readouts

    vehicle_fold = assign_vehicle_folds(eval_frame, n_splits, cv_seed)
    folds = build_fixed_eval_folds(
        train_frame[VEHICLE_COL].reset_index(drop=True),
        eval_frame[VEHICLE_COL].reset_index(drop=True),
        vehicle_fold,
    )
    features = select_feature_columns(eval_frame)
    columns = [*features, BINARY_COL]
    combined = pd.concat([train_frame[columns], eval_frame[columns]], ignore_index=True)
    baseline = float(eval_frame[BINARY_COL].mean())

    curve = compute_learning_curve(
        build_logreg_pipeline(random_state=cv_seed),
        combined[features],
        combined[BINARY_COL].to_numpy(),
        folds,
        seed=cv_seed,
    )
    for row in curve.itertuples(index=False):
        logger.info(
            "rows %6d | train %.4f +- %.4f | val %.4f +- %.4f | gap %+.4f",
            row.train_rows,
            row.train_mean,
            row.train_std,
            row.val_mean,
            row.val_std,
            row.train_mean - row.val_mean,
        )
    # How much did the last doubling of the data help? A small number means
    # more data will not fix the model (high bias).
    gain = curve["val_mean"].iloc[-1] - curve["val_mean"].iloc[-2]
    logger.info("Validation gain from the last size step: %+.4f", gain)

    reports = resolve_path(config["paths"]["reports"])
    figure_path = plot_learning_curve(curve, baseline, reports / "figures" / FIGURE_NAME)
    table_path_out = reports / TABLE_NAME
    curve.to_csv(table_path_out, index=False, float_format="%.5f")
    logger.info("Saved learning curve table to %s", table_path_out)

    if track:
        log_learning_curve(
            curve,
            [figure_path, table_path_out],
            params={
                "model": "logreg_balanced",
                "n_cutoffs_per_vehicle": n_cutoffs_label(k),
                "evaluation": "1 cut-off per truck, fixed folds",
                "train_sizes": ",".join(f"{s:g}" for s in DEFAULT_TRAIN_SIZES),
                "cv_n_splits": n_splits,
                "cv_seed": cv_seed,
                "n_features": len(features),
                "eval_positive_rate": round(baseline, 5),
            },
        )
    return curve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="do not log to MLflow (quick local check)",
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
