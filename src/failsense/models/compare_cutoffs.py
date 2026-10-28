"""Compare the number of training cut-offs per truck (k) on a FIXED evaluation set.

Run it with::

    uv run python -m failsense.models.compare_cutoffs                # k = 1, 5, all
    uv run python -m failsense.models.compare_cutoffs --ks 1 3 5 10
    uv run python -m failsense.models.compare_cutoffs --no-mlflow

The question
------------
``k`` (``sampling.n_cutoffs_per_vehicle``) decides how many random readouts
per truck we use as training examples. It is a hyperparameter of the data
preparation, so it is chosen with cross validation, not guessed.

Why not simply run ``baseline_cv --n-cutoffs 1/5/all``?
-------------------------------------------------------
In ``baseline_cv`` the *validation* rows of each fold also come from the
sample with ``k`` cut-offs. Changing ``k`` would then change **what we
measure on**, not only what we train on:

* ``k = all`` validates on every readout of a truck (long histories dominate,
  different age distribution, different share of positives),
* ``k = 1`` validates on one readout per truck, like the real validation and
  test splits.

The scores would not be comparable (different exam for every student). This
module therefore fixes the exam:

1. **Evaluation sample:** exactly one random cut-off per truck (``k = 1``,
   seeded). This imitates the official validation/test splits.
2. **Fold assignment per vehicle:** ``StratifiedGroupKFold`` on that sample
   assigns every truck to one fold, once. The same assignment is used for
   every ``k``.
3. For every ``k``: train on the ``k`` cut-offs of the training trucks of a
   fold, validate on the single evaluation cut-off of the validation trucks.

Only the training sample changes between the runs, so differences in the
score are caused by ``k`` alone. Group leakage is impossible: a truck is in
the training part *or* the validation part of a fold, never both (checked in
every fold).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from failsense.config import load_config
from failsense.data.labels import BINARY_COL, VEHICLE_COL, class_bounds_from_config
from failsense.data.sampling import sampling_params_from_config
from failsense.data.split import (
    Fold,
    assert_no_group_overlap,
    cv_params_from_config,
    make_group_folds,
)
from failsense.models.baseline_cv import (
    SCORING,
    build_training_frame,
    n_cutoffs_label,
    parse_n_cutoffs,
    read_table,
    table_path,
)
from failsense.models.pipeline import build_logreg_pipeline, select_feature_columns
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "k_comparison"}

DEFAULT_KS = ("1", "5", "all")


def assign_vehicle_folds(eval_frame: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """Assign every vehicle to exactly one validation fold.

    Args:
        eval_frame: One row per vehicle (the evaluation sample, ``k = 1``),
            with ``vehicle_id`` and the binary label (used for stratification).
        n_splits: Number of folds.
        seed: Seed for ``StratifiedGroupKFold`` (same seed = same assignment).

    Returns:
        A Series ``vehicle_id -> fold number`` (0 .. n_splits-1).

    Raises:
        ValueError: If a vehicle appears more than once (then the sample is
            not "one cut-off per truck" and the stratification is wrong).
    """
    if eval_frame[VEHICLE_COL].duplicated().any():
        raise ValueError("eval_frame must contain exactly one row per vehicle")

    folds = make_group_folds(eval_frame, n_splits=n_splits, seed=seed)
    # Start with -1 ("no fold yet"); KFold puts every row into exactly one
    # validation part, so afterwards no -1 is left.
    vehicle_fold = pd.Series(-1, index=eval_frame[VEHICLE_COL].to_numpy(), name="fold")
    for fold_no, (_, val_pos) in enumerate(folds):
        vehicle_fold.iloc[val_pos] = fold_no
    return vehicle_fold


def build_fixed_eval_folds(
    train_vehicles: pd.Series,
    eval_vehicles: pd.Series,
    vehicle_fold: pd.Series,
) -> list[Fold]:
    """Folds for "train on sample A, validate on sample B", stacked as [A; B].

    The caller stacks the training sample (``len(train_vehicles)`` rows)
    on top of the evaluation sample. Position ``i`` < offset is a training
    candidate, position ``offset + j`` is evaluation row ``j``.

    Args:
        train_vehicles: ``vehicle_id`` of every row of the training sample.
        eval_vehicles: ``vehicle_id`` of every row of the evaluation sample.
        vehicle_fold: Output of ``assign_vehicle_folds``.

    Returns:
        ``(train_pos, val_pos)`` per fold, usable as ``cv=`` in scikit-learn:
        training rows of all vehicles *outside* fold ``f`` and evaluation rows
        of the vehicles *inside* fold ``f``.

    Raises:
        ValueError: If a vehicle has no fold, or (should never happen) a
            vehicle ends up in both parts of a fold.
    """
    train_fold = train_vehicles.map(vehicle_fold)
    eval_fold = eval_vehicles.map(vehicle_fold)
    no_fold = pd.concat([train_vehicles[train_fold.isna()], eval_vehicles[eval_fold.isna()]])
    if len(no_fold):
        examples = sorted(no_fold.unique().tolist())[:5]
        raise ValueError(f"{no_fold.nunique()} vehicles have no fold, e.g. {examples}")

    train_fold_arr = train_fold.to_numpy(dtype=int)
    eval_fold_arr = eval_fold.to_numpy(dtype=int)
    train_ids = train_vehicles.to_numpy()
    eval_ids = eval_vehicles.to_numpy()
    offset = len(train_vehicles)
    n_splits = int(vehicle_fold.max()) + 1

    folds: list[Fold] = []
    for f in range(n_splits):
        # np.flatnonzero(mask) = the positions where mask is True.
        train_pos = np.flatnonzero(train_fold_arr != f)
        eval_pos = np.flatnonzero(eval_fold_arr == f)
        # Defensive check, as in make_group_folds: trust, but verify.
        assert_no_group_overlap(
            {f"fold{f}_train": train_ids[train_pos], f"fold{f}_val": eval_ids[eval_pos]}
        )
        folds.append((train_pos, offset + eval_pos))
    return folds


def run(
    config: Mapping[str, Any],
    ks: Sequence[int | None],
    track: bool = False,
) -> dict[str, float]:
    """Cross-validate LogReg for every ``k`` on the same evaluation folds.

    Returns:
        Mean validation PR-AUC per ``k`` label, e.g. ``{"1": 0.09, "all": 0.11}``.
    """
    _, seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    bounds = class_bounds_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))

    # The fixed exam: one cut-off per truck, folds assigned per truck once.
    eval_frame = build_training_frame(readouts, tte, bounds, 1, seed)
    vehicle_fold = assign_vehicle_folds(eval_frame, n_splits, cv_seed)
    features = select_feature_columns(eval_frame)
    eval_prevalence = float(eval_frame[BINARY_COL].mean())
    logger.info(
        "Evaluation sample: %d trucks, positive rate %.4f (= PR-AUC of a dummy model)",
        len(eval_frame),
        eval_prevalence,
    )

    results: dict[str, float] = {}
    for k in ks:
        label = n_cutoffs_label(k)
        train_frame = build_training_frame(readouts, tte, bounds, k, seed)

        # Stack [training sample; evaluation sample] so that one x / y and one
        # list of folds can go into cross_validate. reset_index: positions and
        # labels must agree after the concat.
        columns = [*features, BINARY_COL]
        combined = pd.concat([train_frame[columns], eval_frame[columns]], ignore_index=True)
        folds = build_fixed_eval_folds(
            train_frame[VEHICLE_COL].reset_index(drop=True),
            eval_frame[VEHICLE_COL].reset_index(drop=True),
            vehicle_fold,
        )
        x = combined[features]
        y = combined[BINARY_COL].to_numpy()

        # k = all gives ~900k rows; every parallel worker copies its training
        # part during imputation/scaling. 2 workers keep RAM use safe.
        n_jobs = -1 if k is not None else 2
        logger.info("k=%s: %d training rows, n_jobs=%d", label, len(train_frame), n_jobs)
        result = cross_validate(
            build_logreg_pipeline(random_state=cv_seed),
            x,
            y,
            cv=folds,
            scoring=SCORING,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        val = result["test_pr_auc"]
        train = result["train_pr_auc"]
        results[label] = float(val.mean())
        logger.info(
            "k=%-4s pr_auc val %.4f +- %.4f | train %.4f | gap %+.4f | fit %.1f s | folds %s",
            label,
            val.mean(),
            val.std(),
            train.mean(),
            train.mean() - val.mean(),
            float(np.mean(result["fit_time"])),
            [round(float(s), 4) for s in val],
        )

        if track:
            log_cv_run(
                run_name=f"logreg_balanced_k{label}_fixed_eval",
                cv_result=result,
                metrics=SCORING,
                params={
                    "model": "logreg_balanced",
                    "n_cutoffs_per_vehicle": label,
                    "evaluation": "1 cut-off per truck, fixed folds",
                    "sampling_seed": seed,
                    "cv_n_splits": n_splits,
                    "cv_seed": cv_seed,
                    "n_train_rows": len(train_frame),
                    "n_eval_rows": len(eval_frame),
                    "n_features": len(features),
                    "eval_positive_rate": round(eval_prevalence, 5),
                },
                tags=RUN_TAGS,
            )
        del train_frame, combined, x  # free memory before the next (larger) k

    logger.info("Summary (val PR-AUC): %s", {k: round(v, 4) for k, v in results.items()})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # nargs="+": one or more values, e.g. --ks 1 5 all
    parser.add_argument(
        "--ks",
        nargs="+",
        default=list(DEFAULT_KS),
        help="values of k to compare: integers or 'all' (default: 1 5 all)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="do not log to MLflow (quick local check)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ks = [parse_n_cutoffs(value) for value in args.ks]
    config = load_config()

    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(config, ks, track=track)


if __name__ == "__main__":
    main()
