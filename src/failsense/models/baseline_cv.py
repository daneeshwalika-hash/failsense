"""Baseline cross validation on the real SCANIA training data (Q1, binary).

Run it with::

    uv run python -m failsense.models.baseline_cv            # k from config
    uv run python -m failsense.models.baseline_cv --n-cutoffs 1
    uv run python -m failsense.models.baseline_cv --n-cutoffs all
    uv run python -m failsense.models.baseline_cv --no-mlflow  # quick check only

What happens, step by step
--------------------------
1. Load the training readouts and the time-to-event table (Parquet if
   ``make parquet`` was run, otherwise the raw CSV).
2. ``build_labels``: one label per readout; censored readouts close to the
   end of the study become ``<NA>``.
3. ``sample_cutoffs``: up to ``k`` random readouts per truck with a known
   label (the "moments we ask the model").
4. Features = the raw values of the readout at the cut-off. They are
   cumulative counters/histograms, so they only contain information up to
   that moment (no future information).
   Since 30.10 engineered features are added: counter deltas/rates (computed
   on the full readouts of each truck, then picked at the cut-offs) and
   histogram shares / mean bin / top-bin share (``failsense.features.build``).
   Since 29.10 the eight truck specifications are joined by ``vehicle_id``
   (``attach_specifications``) and one-hot encoded inside the Pipeline.
5. ``make_group_folds``: 5 stratified folds by ``vehicle_id``.
6. ``cross_validate`` for the Dummy baseline and the logistic-regression
   Pipelines (numeric only; + specs with StandardScaler; + specs with
   MinMaxScaler; + engineered features; each with and without ``time_step``) on
   exactly the same folds; PR-AUC (average precision) is the
   primary metric, ROC-AUC is shown for comparison only. Train scores are
   computed too (``return_train_score=True``): the gap between train and
   validation scores shows over- or underfitting (course day 28.10).
7. Paired comparison per fold (``paired_comparison``): the difference of two
   models fold by fold. Folds differ in difficulty (fold 2 is the hardest for
   every model), so per-fold differences are a fairer test than two means.
8. Every model becomes one MLflow run (``failsense.tracking``) with params,
   per-fold metrics, means, standard deviations and the train/val gap.

Only the **training** split is used. Validation is for threshold choice and
calibration later; the test split is used once at the very end.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    BINARY_COL,
    TIME_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.data.split import cv_params_from_config, make_group_folds
from failsense.features.build import (
    add_counter_deltas,
    add_histogram_features,
    attach_specifications,
    engineered_columns,
    features_params_from_config,
)
from failsense.models.pipeline import (
    build_dummy_classifier,
    build_logreg_pipeline,
    select_categorical_columns,
    select_feature_columns,
)
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

# Labels attached to every run of this script (filterable in the MLflow UI).
RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "baseline"}

# (candidate, reference) pairs compared fold by fold on the primary metric.
# 1) Do the specifications help?  2) Which scaler, with everything else equal?
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("logreg_specs_standard", "logreg_balanced"),
    ("logreg_specs_minmax", "logreg_specs_standard"),
    # 30.10: engineered features, and the selection effect of time_step.
    ("logreg_specs_eng", "logreg_specs_standard"),
    ("logreg_specs_no_ts", "logreg_specs_standard"),
    ("logreg_specs_eng_no_ts", "logreg_specs_eng"),
)


def table_path(config: Mapping[str, Any], split: str, table: str) -> Path:
    """Return the Parquet copy of a SCANIA table if it exists, else the raw CSV.

    The Parquet file has the same name as the CSV with ``.parquet``
    (``failsense.data.convert``), stored in ``datasets.scania.processed_dir``.
    """
    scania = config["datasets"]["scania"]
    csv_path = resolve_path(scania[split][table])
    parquet_path = resolve_path(scania["processed_dir"]) / f"{csv_path.stem}.parquet"
    if parquet_path.exists():
        return parquet_path
    logger.warning("No Parquet copy found, reading CSV %s (run `make parquet`)", csv_path)
    return csv_path


def read_table(path: Path) -> pd.DataFrame:
    """Read a Parquet or CSV file, chosen by the file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_training_frame(
    readouts: pd.DataFrame,
    tte: pd.DataFrame,
    class_bounds: tuple[float, ...],
    n_per_vehicle: int | None,
    seed: int,
    counter_cols: Sequence[str] = (),
    hist_groups: Mapping[str, Sequence[str]] | None = None,
    top_fraction: float = 0.2,
) -> pd.DataFrame:
    """Labels -> sampled cut-offs -> the readout rows at those cut-offs + Q1 label.

    The label table has the same index as ``readouts`` (see ``build_labels``),
    so ``readouts.loc[sampled.index]`` selects exactly the sampled readouts.
    Only the binary label is attached; the other label columns stay in the
    label table and never come near the features.

    Engineered features (both optional, off by default):

    * ``counter_cols``: deltas/rates are computed on the **full** readouts
      (they need the previous readout of the truck) and only then picked at
      the cut-off rows. Only the key columns and the counters are copied for
      this, not the whole ~1 GB table.
    * ``hist_groups``: shares and mean bin use one row at a time, so they are
      computed on the selected rows only (same result, far fewer rows).
    """
    labels = build_labels(readouts, tte, class_bounds)
    sampled = sample_cutoffs(labels, n_per_vehicle=n_per_vehicle, seed=seed)
    frame = readouts.loc[sampled.index].copy()
    if counter_cols:
        key_and_counters = readouts[[VEHICLE_COL, TIME_COL, *counter_cols]]
        with_deltas = add_counter_deltas(key_and_counters, counter_cols)
        new_cols = [c for c in with_deltas.columns if c not in key_and_counters.columns]
        frame = frame.join(with_deltas.loc[sampled.index, new_cols])
    if hist_groups:
        frame = add_histogram_features(frame, hist_groups, top_fraction)
    frame[BINARY_COL] = sampled[BINARY_COL].astype(int)
    return frame


def paired_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare two models on the same folds, fold by fold.

    Args:
        candidate: Validation score per fold of the new model.
        reference: Validation score per fold of the model to beat, from the
            **same** folds in the same order.

    Returns:
        ``mean_diff`` (candidate - reference), ``std_diff``, ``wins`` (folds
        where the candidate is better) and ``n_folds``. A ``mean_diff`` much
        smaller than the fold-to-fold noise is not a real improvement.

    Raises:
        ValueError: If the arrays are empty or have different lengths.
    """
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError(
            f"need two non-empty score arrays of equal length, got {a.shape}, {b.shape}"
        )
    diff = a - b
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "wins": int((diff > 0).sum()),
        "n_folds": int(diff.size),
    }


def n_cutoffs_label(n_per_vehicle: int | None) -> str:
    """Readable value for logs and run names: ``"5"`` or ``"all"``."""
    return "all" if n_per_vehicle is None else str(n_per_vehicle)


def run(
    config: Mapping[str, Any],
    n_per_vehicle: int | None,
    track: bool = False,
) -> dict[str, dict[str, float]]:
    """Load data, build folds and cross-validate Dummy and LogReg.

    Args:
        config: Project config.
        n_per_vehicle: Cut-offs per truck, ``None`` = all known readouts.
        track: Log every model as an MLflow run. ``setup_mlflow`` must have
            been called before. Default ``False`` so tests never write to the
            project's tracking database.

    Returns:
        Mean validation score per model and metric, e.g.
        ``{"logreg_balanced": {"pr_auc": 0.10, "roc_auc": 0.81}}``.
    """
    _, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    specs = read_table(table_path(config, "train", "specifications"))
    logger.info("Loaded %d readouts of %d trucks", len(readouts), readouts[VEHICLE_COL].nunique())

    # Engineered features are always built; each model below picks its columns.
    frame = build_training_frame(
        readouts,
        tte,
        class_bounds_from_config(config),
        n_per_vehicle,
        sampling_seed,
        counter_cols=counter_columns(),
        hist_groups=histogram_groups(),
        top_fraction=features_params_from_config(config),
    )
    del readouts  # about 1 GB in RAM; not needed any more
    frame = attach_specifications(frame, specs)

    # Numeric features are selected *before* the specs are counted: the specs
    # are text, so select_feature_columns skips them; they get their own list.
    # Four numeric column sets: raw / raw + engineered, each with and without
    # time_step (is it a real signal or only a selection effect of the cut-offs?).
    engineered = engineered_columns(list(frame.columns))
    raw_features = select_feature_columns(frame, exclude=engineered)
    raw_no_ts = [c for c in raw_features if c != TIME_COL]
    all_features = select_feature_columns(frame)
    all_no_ts = [c for c in all_features if c != TIME_COL]
    categorical = select_categorical_columns(frame)
    y = frame[BINARY_COL].to_numpy()
    logger.info(
        "Training frame: %d rows, %d raw + %d engineered numeric + %d categorical features, "
        "positive rate %.4f",
        len(frame),
        len(raw_features),
        len(engineered),
        len(categorical),
        y.mean(),
    )

    # Settings shared by every model of this run (they describe the data and CV).
    common_params = {
        "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle),
        "sampling_seed": sampling_seed,
        "cv_n_splits": n_splits,
        "cv_seed": cv_seed,
        "n_rows": len(frame),
        "n_vehicles": int(frame[VEHICLE_COL].nunique()),
        "n_categorical_features": len(categorical),
        "positive_rate": round(float(y.mean()), 5),
        "top_fraction": features_params_from_config(config),
    }

    # Same folds for every model -> fair comparison.
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    # name -> (estimator, numeric columns). A pipeline without categorical
    # columns drops the spec columns (remainder="drop"), so "logreg_balanced"
    # is exactly the model of 27./28.10 (reproducibility anchor).
    spec_kwargs = {"random_state": cv_seed, "categorical_columns": categorical}
    models: dict[str, tuple[Any, list[str]]] = {
        "dummy_prior": (build_dummy_classifier(), raw_features),
        "logreg_balanced": (build_logreg_pipeline(random_state=cv_seed), raw_features),
        "logreg_specs_standard": (build_logreg_pipeline(**spec_kwargs), raw_features),
        "logreg_specs_minmax": (
            build_logreg_pipeline(**spec_kwargs, scaler="minmax"),
            raw_features,
        ),
        "logreg_specs_eng": (build_logreg_pipeline(**spec_kwargs), all_features),
        "logreg_specs_no_ts": (build_logreg_pipeline(**spec_kwargs), raw_no_ts),
        "logreg_specs_eng_no_ts": (build_logreg_pipeline(**spec_kwargs), all_no_ts),
    }

    summary: dict[str, dict[str, float]] = {}
    fold_scores: dict[str, np.ndarray] = {}  # primary metric per fold, per model
    for name, (model, numeric) in models.items():
        x = frame[numeric + categorical]
        # return_train_score=True: also score each fold's *training* part.
        # Costs extra time, but train vs. validation is today's key diagnostic.
        result = cross_validate(
            model, x, y, cv=folds, scoring=SCORING, n_jobs=-1, return_train_score=True
        )
        summary[name] = {}
        fold_scores[name] = result["test_pr_auc"]
        for metric in SCORING:
            val = result[f"test_{metric}"]  # "test" here = the validation fold
            train = result[f"train_{metric}"]
            summary[name][metric] = float(val.mean())
            logger.info(
                "%-16s %-7s val %.4f +- %.4f | train %.4f | gap %+.4f | folds %s",
                name,
                metric,
                val.mean(),
                val.std(),
                train.mean(),
                train.mean() - val.mean(),
                [round(float(s), 4) for s in val],
            )

        if track:
            log_cv_run(
                run_name=f"{name}_k{n_cutoffs_label(n_per_vehicle)}",
                cv_result=result,
                metrics=SCORING,
                params={
                    **common_params,
                    "model": name,
                    "model_repr": repr(model),
                    "n_numeric_features": len(numeric),
                    "engineered": any(c in engineered for c in numeric),
                    "uses_time_step": TIME_COL in numeric,
                },
                tags=RUN_TAGS,
            )

    for candidate, reference in COMPARISONS:
        if candidate in fold_scores and reference in fold_scores:
            cmp = paired_comparison(fold_scores[candidate], fold_scores[reference])
            logger.info(
                "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
                candidate,
                reference,
                cmp["mean_diff"],
                cmp["std_diff"],
                cmp["wins"],
                cmp["n_folds"],
            )
    return summary


def parse_n_cutoffs(value: str) -> int | None:
    """argparse helper: an integer, or 'all' for every known readout."""
    if value.lower() in {"all", "none", "null"}:
        return None
    return int(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No `type=` here on purpose: argparse would also apply it to the default.
    # None = "not given on the command line" -> use the config value.
    parser.add_argument(
        "--n-cutoffs",
        default=None,
        help="cut-offs per truck: an integer or 'all' (default: value from config)",
    )
    # store_true: the flag has no value; present -> True, absent -> False.
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
    if args.n_cutoffs is None:
        n_per_vehicle, _ = sampling_params_from_config(config)
    else:
        n_per_vehicle = parse_n_cutoffs(args.n_cutoffs)
    logger.info("Cut-offs per truck: %s", n_cutoffs_label(n_per_vehicle))

    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(config, n_per_vehicle, track=track)


if __name__ == "__main__":
    main()
