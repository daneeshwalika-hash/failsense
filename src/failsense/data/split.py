"""Leakage-safe splitting by group (vehicle for SCANIA, engine for C-MAPSS).

Why this module exists
----------------------
Readouts of the same truck are almost copies of each other. If one truck
appears in two parts of the data (for example in training *and* in
validation), the model can "recognise the truck" instead of learning
general patterns. The score then looks better than it will be on new
trucks. This is group leakage, the most important risk in FailSense.

Two places need protection:

1. **Official splits** (train / validation / test). SCANIA already splits
   by truck. We do not trust that blindly: ``assert_no_group_overlap``
   checks it, and stops the pipeline if the check fails.
2. **Cross validation inside the training split.** ``make_group_folds``
   builds the folds with ``StratifiedGroupKFold``:

   * *Group*: all rows of one vehicle are in the same fold, so a vehicle is
     never in the training part and the validation part of one fold.
   * *Stratified*: every fold gets about the same share of positives.
     With roughly 2 % positives, a plain ``GroupKFold`` could produce a
     fold with almost no positives, which makes its score meaningless.

The functions are generic (``group_col`` is a parameter), so the same code
protects C-MAPSS on 11.11 with ``group_col="unit"`` (engine number).
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from failsense.data.labels import BINARY_COL, VEHICLE_COL

logger = logging.getLogger(__name__)

# A fold is a pair of integer row positions: (train positions, validation positions).
Fold = tuple[np.ndarray, np.ndarray]


def assert_no_group_overlap(
    groups_by_split: Mapping[str, Iterable[Hashable]],
    max_examples: int = 5,
) -> None:
    """Raise if any group id (e.g. a vehicle_id) appears in more than one split.

    Args:
        groups_by_split: Split name -> group ids of that split, for example
            ``{"train": train["vehicle_id"], "validation": val["vehicle_id"]}``.
            Duplicates inside one split are fine (many readouts per truck).
        max_examples: How many overlapping ids to show in the error message.

    Raises:
        ValueError: If fewer than two splits are given, or if two splits
            share at least one group id. The message names every pair of
            splits that overlaps, so the problem can be found quickly.
    """
    if len(groups_by_split) < 2:
        raise ValueError("need at least two splits to check for overlap")

    # set() removes the duplicates inside a split: we only care about which
    # vehicles occur, not how often.
    id_sets = {name: set(ids) for name, ids in groups_by_split.items()}

    problems: list[str] = []
    # combinations(["train", "val", "test"], 2) gives every pair once:
    # (train, val), (train, test), (val, test).
    for (name_a, ids_a), (name_b, ids_b) in combinations(id_sets.items(), 2):
        shared = ids_a & ids_b  # set intersection, like INNER JOIN on the id
        if shared:
            examples = sorted(shared, key=str)[:max_examples]
            problems.append(f"{name_a}/{name_b}: {len(shared)} shared, e.g. {examples}")

    if problems:
        raise ValueError("group leakage between splits -> " + "; ".join(problems))

    logger.info(
        "No group overlap between splits: %s",
        ", ".join(f"{name}={len(ids)}" for name, ids in id_sets.items()),
    )


def cv_params_from_config(config: Mapping[str, Any]) -> tuple[int, int]:
    """Read ``cv.n_splits`` and ``cv.seed`` from the project config."""
    try:
        n_splits = config["cv"]["n_splits"]
        seed = config["cv"]["seed"]
    except KeyError as exc:
        raise KeyError("config needs 'cv.n_splits' and 'cv.seed'") from exc
    for name, value in (("cv.n_splits", n_splits), ("cv.seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {value!r}")
    if n_splits < 2:
        raise ValueError(f"cv.n_splits must be >= 2, got {n_splits}")
    return n_splits, seed


def make_group_folds(
    data: pd.DataFrame,
    n_splits: int,
    seed: int,
    target_col: str = BINARY_COL,
    group_col: str = VEHICLE_COL,
) -> list[Fold]:
    """Build stratified, group-safe CV folds.

    Args:
        data: Training rows (usually the output of ``sample_cutoffs``).
            Needs ``target_col`` without missing values and ``group_col``.
        n_splits: Number of folds (5 in FailSense).
        seed: Random seed so the folds are identical in every run. All
            models must use the **same** folds for a fair comparison.
        target_col: Column used for stratification (binary label of Q1).
        group_col: Column whose values must not be split across folds.

    Returns:
        A list of ``n_splits`` pairs ``(train_pos, val_pos)`` with integer
        row positions, usable with ``data.iloc[...]`` and as ``cv=`` in
        scikit-learn (``cross_validate``, ``RandomizedSearchCV``).

    Raises:
        ValueError: On missing columns, missing target values, fewer groups
            than folds, or (should never happen) a group in two parts of a fold.
    """
    missing = [c for c in (target_col, group_col) if c not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if data[target_col].isna().any():
        # <NA> labels must be removed first (sample_cutoffs does that).
        raise ValueError(f"{target_col} contains missing values; drop unknown labels first")

    n_groups = data[group_col].nunique()
    if n_groups < n_splits:
        raise ValueError(f"need at least {n_splits} groups for {n_splits} folds, got {n_groups}")

    # shuffle=True + random_state: the assignment of vehicles to folds is
    # random but reproducible. Without shuffle the folds would depend on
    # the order of the vehicle ids.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = data[target_col].to_numpy(dtype=int)
    groups = data[group_col].to_numpy()

    folds: list[Fold] = []
    for fold_no, (train_pos, val_pos) in enumerate(splitter.split(data, y, groups)):
        # Defensive check: trust, but verify. Costs milliseconds, and turns a
        # silent leakage bug (e.g. a future code change) into a loud error.
        assert_no_group_overlap(
            {f"fold{fold_no}_train": groups[train_pos], f"fold{fold_no}_val": groups[val_pos]}
        )
        logger.info(
            "Fold %d: %d train rows, %d val rows, val positive rate %.3f",
            fold_no,
            len(train_pos),
            len(val_pos),
            float(y[val_pos].mean()),
        )
        folds.append((train_pos, val_pos))
    return folds
