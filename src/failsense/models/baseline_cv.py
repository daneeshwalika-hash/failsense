"""First end-to-end run on the real SCANIA training data (Q1, binary).

Run it with::

    uv run python -m failsense.models.baseline_cv            # k from config
    uv run python -m failsense.models.baseline_cv --n-cutoffs 1
    uv run python -m failsense.models.baseline_cv --n-cutoffs all

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
   that moment (no future information). Engineered features follow on 30.10.
5. ``make_group_folds``: 5 stratified folds by ``vehicle_id``.
6. ``cross_validate`` for the Dummy baseline and the logistic-regression
   Pipeline on exactly the same folds; PR-AUC (average precision) is the
   primary metric, ROC-AUC is shown for comparison only.

Only the **training** split is used. Validation is for threshold choice and
calibration later; the test split is used once at the very end.
This is a first measurement, not the final baseline: MLflow tracking,
learning curves and the comparison of ``k`` follow on 28.10.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import cross_validate

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, VEHICLE_COL, build_labels, class_bounds_from_config
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.data.split import cv_params_from_config, make_group_folds
from failsense.models.pipeline import (
    build_dummy_classifier,
    build_logreg_pipeline,
    select_feature_columns,
)

logger = logging.getLogger(__name__)

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


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
) -> pd.DataFrame:
    """Labels -> sampled cut-offs -> the readout rows at those cut-offs + Q1 label.

    The label table has the same index as ``readouts`` (see ``build_labels``),
    so ``readouts.loc[sampled.index]`` selects exactly the sampled readouts.
    Only the binary label is attached; the other label columns stay in the
    label table and never come near the features.
    """
    labels = build_labels(readouts, tte, class_bounds)
    sampled = sample_cutoffs(labels, n_per_vehicle=n_per_vehicle, seed=seed)
    frame = readouts.loc[sampled.index].copy()
    frame[BINARY_COL] = sampled[BINARY_COL].astype(int)
    return frame


def run(config: Mapping[str, Any], n_per_vehicle: int | None) -> dict[str, dict[str, float]]:
    """Load data, build folds and cross-validate Dummy and LogReg. Returns mean scores."""
    _, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    logger.info("Loaded %d readouts of %d trucks", len(readouts), readouts[VEHICLE_COL].nunique())

    frame = build_training_frame(
        readouts, tte, class_bounds_from_config(config), n_per_vehicle, sampling_seed
    )
    del readouts  # about 1 GB in RAM; not needed any more

    features = select_feature_columns(frame)
    x = frame[features]
    y = frame[BINARY_COL].to_numpy()
    logger.info(
        "Training frame: %d rows, %d features, positive rate %.4f",
        len(frame),
        len(features),
        y.mean(),
    )

    # Same folds for every model -> fair comparison.
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    models = {
        "dummy_prior": build_dummy_classifier(),
        "logreg_balanced": build_logreg_pipeline(random_state=cv_seed),
    }

    summary: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        result = cross_validate(model, x, y, cv=folds, scoring=SCORING, n_jobs=-1)
        summary[name] = {}
        for metric in SCORING:
            scores = result[f"test_{metric}"]
            summary[name][metric] = float(scores.mean())
            logger.info(
                "%-16s %-7s mean %.4f  std %.4f  folds %s",
                name,
                metric,
                scores.mean(),
                scores.std(),
                [round(float(s), 4) for s in scores],
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    if args.n_cutoffs is None:
        n_per_vehicle, _ = sampling_params_from_config(config)
    else:
        n_per_vehicle = parse_n_cutoffs(args.n_cutoffs)
    logger.info("Cut-offs per truck: %s", "all" if n_per_vehicle is None else n_per_vehicle)
    run(config, n_per_vehicle)


if __name__ == "__main__":
    main()
