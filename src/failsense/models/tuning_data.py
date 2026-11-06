"""Data for the hyperparameter search and the evaluation: folds, inner splits, eval sets.

Everything the tuner and ``evaluate`` share, built once:

* the ``k = 5`` training sample (the decision of step 7a) with and without the engineered
  features,
* the two validation-like evaluation sets (``calibrated`` = main, ``literal`` = second
  reference) on the same truck-to-fold assignment as ``make cv``,
* per fold: the training rows of the other folds' trucks, the evaluation rows of the fold's
  trucks, both preprocessed with a preprocessor **fitted on the fold's training rows only**
  (median imputer + one-hot specs, no scaler), cached as float32,
* per fold: a fixed truck-grouped inner split of the *training* trucks for early stopping.
  The evaluation rows never take part in it: their trucks are in the fold's validation part,
  the inner split only contains trucks of the fold's training part.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from failsense.data.labels import (
    BINARY_COL,
    CLASS_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.data.split import cv_params_from_config, make_group_folds
from failsense.features.build import attach_specifications, features_params_from_config
from failsense.models.baseline_cv import frame_from_sampled, read_table, table_path
from failsense.models.eval_set import (
    build_eval_cutoffs_by_method,
    validation_time_steps,
    vehicle_fold_from_folds,
)
from failsense.models.pipeline import build_preprocessor, split_feature_columns

logger = logging.getLogger(__name__)

FEATURE_SETS = ("base", "engineered")
EVAL_METHODS = ("calibrated", "literal")


@dataclass(frozen=True)
class FoldData:
    """One fold: preprocessed arrays (float32) for both feature sets and both eval sets."""

    fold: int
    x_train: dict[str, np.ndarray]  # feature set -> training rows
    y_train: np.ndarray
    vehicles_train: np.ndarray
    inner_train: np.ndarray  # positions in x_train used for fitting while tuning
    inner_val: np.ndarray  # positions in x_train used for early stopping (whole trucks)
    x_eval: dict[str, dict[str, np.ndarray]]  # eval method -> feature set -> rows
    y_eval: dict[str, np.ndarray]
    vehicles_eval: dict[str, np.ndarray]


@dataclass(frozen=True)
class TuningData:
    folds: list[FoldData]
    seed: int
    cv_seed: int
    columns: dict[str, list[str]]  # feature set -> input columns of the model
    categorical: list[str]
    train_frame: pd.DataFrame  # the k5 training rows (engineered superset), all folds together
    eval_info: dict[str, dict[str, float]]


def inner_split(
    groups: np.ndarray, y: np.ndarray, fraction: float, seed: int, attempts: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Truck-grouped split into inner-train and inner-validation positions.

    ``GroupShuffleSplit`` by ``vehicle_id``: no truck is on both sides. A part that holds
    a single class cannot be scored with average precision, so the split is repeated with
    the next seed (deterministically) until both parts have both classes.

    Raises:
        ValueError: If ``fraction`` is not in (0, 1) or no valid split is found.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    for attempt in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + attempt)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return train_idx, val_idx
    raise ValueError(f"no inner split with both classes in both parts after {attempts} attempts")


def build_tuning_data(
    config: Mapping[str, Any], eval_methods: Sequence[str] = EVAL_METHODS
) -> TuningData:
    """Load the training split once and prepare every fold (see the module docstring).

    Only the training split and the validation readouts' ``time_step`` / one aggregate label
    rate (calibrated evaluation set) are used; the test split is never touched.
    """
    _, seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    tuning = config["tuning"]
    top_fraction = features_params_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    specs = read_table(table_path(config, "train", "specifications"))
    labels = build_labels(readouts, tte, class_bounds_from_config(config))
    targets = validation_time_steps(config)
    repaired = tte.set_index(VEHICLE_COL)["in_study_repair"]

    def frame(cutoffs: pd.DataFrame) -> pd.DataFrame:  # engineered superset of the columns
        built = frame_from_sampled(
            readouts, cutoffs, counter_cols=counter_columns(), hist_groups=histogram_groups(),
            top_fraction=top_fraction,
        )  # fmt: skip
        return attach_specifications(built, specs)

    train = frame(sample_cutoffs(labels, 5, seed)).reset_index(drop=True)
    cols = split_feature_columns(train)
    columns = {
        "base": cols.raw + cols.categorical,
        "engineered": cols.raw + cols.engineered + cols.categorical,
    }
    folds_idx = make_group_folds(train, n_splits=n_splits, seed=cv_seed)
    vehicle_fold = vehicle_fold_from_folds(train, folds_idx)

    eval_frames: dict[str, pd.DataFrame] = {}
    eval_info: dict[str, dict[str, float]] = {}
    for method in eval_methods:
        cut, info = build_eval_cutoffs_by_method(method, labels, targets, repaired, seed)
        eval_frames[method] = frame(cut).reset_index(drop=True)
        eval_info[method] = {**info, "n": float(len(cut)), "rate": float(cut[BINARY_COL].mean())}
        # class mix of the evaluation rows (TRAINING labels): used for the cost threshold
        for k in range(5):
            eval_info[method][f"n_class_{k}"] = float((cut[CLASS_COL] == k).sum())

    train_fold = train[VEHICLE_COL].map(vehicle_fold).to_numpy()
    eval_fold = {m: f[VEHICLE_COL].map(vehicle_fold).to_numpy() for m, f in eval_frames.items()}
    folds: list[FoldData] = []
    for f in range(n_splits):
        tr_mask = train_fold != f
        tr = train.loc[tr_mask]
        y_tr = tr[BINARY_COL].to_numpy(dtype=int)
        vehicles_tr = tr[VEHICLE_COL].to_numpy()
        x_train: dict[str, np.ndarray] = {}
        x_eval: dict[str, dict[str, np.ndarray]] = {m: {} for m in eval_frames}
        for fs in FEATURE_SETS:
            # fitted on this fold's training rows only; the evaluation rows are only transformed
            pre = build_preprocessor(cols.categorical, scaler=None)
            x_train[fs] = pre.fit_transform(tr[columns[fs]]).astype(np.float32)
            for m, ef in eval_frames.items():
                rows = ef.loc[eval_fold[m] == f]
                x_eval[m][fs] = pre.transform(rows[columns[fs]]).astype(np.float32)
        inner_tr, inner_val = inner_split(
            vehicles_tr, y_tr, float(tuning["inner_val_fraction"]), int(tuning["seed"]) + f
        )
        folds.append(
            FoldData(
                fold=f, x_train=x_train, y_train=y_tr, vehicles_train=vehicles_tr,
                inner_train=inner_tr, inner_val=inner_val, x_eval=x_eval,
                y_eval={m: ef.loc[eval_fold[m] == f, BINARY_COL].to_numpy(dtype=int)
                        for m, ef in eval_frames.items()},
                vehicles_eval={m: ef.loc[eval_fold[m] == f, VEHICLE_COL].to_numpy()
                               for m, ef in eval_frames.items()},
            )
        )  # fmt: skip
        logger.info(
            "Fold %d: %d training rows (%d / %d inner train / val), eval rows %s",
            f, len(y_tr), len(inner_tr), len(inner_val),
            {m: len(v) for m, v in folds[-1].y_eval.items()},
        )  # fmt: skip
    return TuningData(folds, seed, cv_seed, columns, list(cols.categorical), train, eval_info)
