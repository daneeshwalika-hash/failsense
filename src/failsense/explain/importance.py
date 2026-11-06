"""Permutation importance (single column and grouped) and gain importance.

**Permutation importance** answers "how much worse does the score get when this feature carries no
information?": shuffle the rows of one column (or of a whole group of columns, jointly), predict
again, and take the drop of the metric (here PR-AUC on the official validation). It needs labels,
so it is an *evaluation* of the fitted model, never a tool to select or drop features.

* **Repeats and spread:** the shuffle is random, so every column is shuffled ``n_repeats`` times.
  The standard deviation over the repeats shows only the *shuffle* noise; the sampling noise of
  136 positive trucks comes on top and is not in this number. A drop is called ``reliable`` only
  if ``mean - 2 * std > 0``; groups inside the spread are not ranked.
* **Correlated features:** the histogram bins and their engineered shares / mean bin / top share
  carry nearly the same information; shuffling one column leaves the others to compensate, so
  single-column importances are small and split. Shuffling a **group** (all columns of one
  variable, with the same row permutation) removes that information completely.

**Gain importance** is the XGBoost-internal sum of the loss reduction of all splits on a feature.
It is biased: it is computed on the training data (rewards what the trees overfit), favours
continuous columns with many split points, splits credit arbitrarily between correlated columns,
is unsigned and says nothing about held-out performance.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from failsense.data.schema import COUNTER_VARIABLES, HISTOGRAM_BINS

logger = logging.getLogger(__name__)

RELIABLE_STDS = 2.0  # a drop is reliable if mean - RELIABLE_STDS * std > 0


def variable_of(column: str) -> str:
    """Variable a column belongs to: ``167_3``, ``167__mean_bin``, ``167_3__share`` -> ``167``.

    ``time_step`` and ``Spec_k`` are their own variables.

    Raises:
        ValueError: If the column name has no known pattern.
    """
    if column == "time_step" or column.startswith("Spec_"):
        return column
    match = re.match(r"^(\d+)(?:_\d+)?(?:__\w+)?$", column)
    if match is None:
        raise ValueError(f"cannot assign column {column!r} to a variable")
    return match.group(1)


def group_name(variable: str) -> str:
    """Readable group name, e.g. ``167 (histogram)``, ``171 (counter)``, ``Spec_3``."""
    if variable in COUNTER_VARIABLES:
        return f"{variable} (counter)"
    if variable in HISTOGRAM_BINS:
        return f"{variable} (histogram)"
    return variable


def feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """All input columns of one variable form one group (raw + engineered columns together)."""
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(group_name(variable_of(column)), []).append(column)
    return groups


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    x: pd.DataFrame,
    y: Sequence[int] | np.ndarray,
    groups: Mapping[str, Sequence[str]],
    n_repeats: int,
    seed: int,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
) -> pd.DataFrame:
    """Drop of ``metric`` when a group of columns is shuffled (rows permuted jointly).

    Args:
        predict: Scores for a frame, e.g. ``lambda f: pipeline.predict_proba(f)[:, 1]``.
        x: Input frame (never modified).
        y: Binary labels.
        groups: Group name -> columns (a single column is a group of one).
        n_repeats: Shuffles per group (>= 2 so a spread exists).
        seed: Every (group, repeat) has its own generator derived from it: reproducible and
            independent of the order of the groups.
        metric: ``metric(y, scores)``, higher is better.

    Returns:
        One row per group: ``group, n_columns, baseline, mean_drop, std_drop, lower, reliable``.

    Raises:
        ValueError: For ``n_repeats < 2``, a length mismatch, one-class ``y``, or unknown columns.
    """
    labels = np.asarray(y)
    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2 to measure a spread")
    if len(labels) != len(x):
        raise ValueError("x and y must have the same length")
    if len(np.unique(labels)) < 2:
        raise ValueError("y must contain both classes")
    unknown = {c for cols in groups.values() for c in cols} - set(x.columns)
    if unknown:
        raise ValueError(f"groups contain unknown columns: {sorted(unknown)[:5]}")

    baseline = float(metric(labels, predict(x)))
    work = x.copy()  # shuffled in place, restored after every group
    rows = []
    for index, (name, columns) in enumerate(groups.items()):
        columns = list(columns)
        original = x[columns].copy()
        drops = []
        for repeat in range(n_repeats):
            order = np.random.default_rng((seed, index, repeat)).permutation(len(x))
            for column in columns:  # same row order for every column of the group
                work[column] = original[column].to_numpy()[order]
            drops.append(baseline - float(metric(labels, predict(work))))
        for column in columns:
            work[column] = original[column].to_numpy()
        mean, std = float(np.mean(drops)), float(np.std(drops, ddof=1))
        rows.append(
            {"group": name, "n_columns": len(columns), "baseline": baseline, "mean_drop": mean,
             "std_drop": std, "lower": mean - RELIABLE_STDS * std,
             "reliable": bool(mean - RELIABLE_STDS * std > 0)}
        )  # fmt: skip
    return pd.DataFrame(rows)


def gain_importance(model: object, feature_names: Sequence[str]) -> pd.Series:
    """Normalised gain importance of a fitted XGBoost model, indexed by model column name.

    The model must have been built with ``importance_type="gain"`` (our ``make_booster`` does).

    Raises:
        ValueError: If the number of names differs from the model's number of features.
    """
    values = np.asarray(model.feature_importances_, dtype=float)  # type: ignore[attr-defined]
    if len(values) != len(feature_names):
        raise ValueError("feature_names does not match the model's number of features")
    return pd.Series(values, index=list(feature_names), name="gain")


def aggregate(values: Sequence[float] | np.ndarray, keys: Sequence[str]) -> pd.Series:
    """Sum ``values`` by ``keys`` (e.g. gain of one-hot columns -> input column), order kept."""
    return pd.Series(np.asarray(values, dtype=float)).groupby(list(keys), sort=False).sum()
