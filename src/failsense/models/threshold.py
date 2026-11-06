"""Cost-based alert threshold: cost curve, cross-fitted choice, baselines, per-class report.

One decision per truck: alert if ``p >= t``. The total cost of a threshold is

    sum(alert cost of alerted trucks) + sum(miss cost of the other trucks)

with the per-truck costs from ``failsense.models.cost`` (class 0: alert = false alarm FP,
no alert = 0; class k >= 1: alert = TP_k, no alert = FN_k). Sorting the trucks by ``p`` and
taking cumulative sums gives the cost of *every* possible threshold in one pass (like a
running ``SUM() OVER (ORDER BY p DESC)`` in SQL Server).

Honesty rule: the minimum of the cost curve on the same trucks is **optimistic** (the
threshold is tuned on the data it is scored on). The reported cost is therefore
**cross-fitted**: the threshold is chosen on 4 folds of trucks and applied to the 5th.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from failsense.models.calibration import cross_fit_splits
from failsense.models.cost import BinaryCosts, total_cost

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdChoice:
    """Result of a threshold search.

    ``threshold`` is ``inf`` for "never alert" and ``0.0`` for "alert every truck".
    """

    threshold: float
    cost: float
    n_alerts: int


def per_truck_costs(
    true_class: Sequence[int] | np.ndarray, costs: BinaryCosts
) -> tuple[np.ndarray, np.ndarray]:
    """Cost of every truck if alerted and if not alerted (arrays in the input order)."""
    classes = np.asarray(true_class).astype(int)
    if classes.ndim != 1:
        raise ValueError("true_class must be 1-D")
    if classes.size and (classes.min() < 0 or classes.max() > 4):
        raise ValueError("true_class must be in 0..4")
    alert_cost = np.zeros(len(classes))
    miss_cost = np.zeros(len(classes))
    alert_cost[classes == 0] = costs.fp
    for k in range(1, 5):
        alert_cost[classes == k] = costs.tp[k - 1]
        miss_cost[classes == k] = costs.fn[k - 1]
    return alert_cost, miss_cost


def cost_curve(
    true_class: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray, costs: BinaryCosts
) -> pd.DataFrame:
    """Total cost for every distinct decision "alert the trucks with ``p >= t``".

    Rows run from "never alert" (``threshold = inf``) to "alert everyone". Trucks with equal
    ``p`` are always alerted together (ties never split). The threshold of a row is the midpoint
    to the next lower score, so it also separates unseen data sensibly.

    Raises:
        ValueError: On different lengths, empty input, or non-finite ``p``.
    """
    scores = np.asarray(p, dtype=float)
    classes = np.asarray(true_class)
    if scores.shape != classes.shape or scores.ndim != 1 or scores.size == 0:
        raise ValueError("true_class and p must be non-empty 1-D arrays of the same length")
    if not np.isfinite(scores).all():
        raise ValueError("p must be finite")
    alert_cost, miss_cost = per_truck_costs(classes, costs)
    order = np.argsort(-scores, kind="stable")
    s, a, m = scores[order], alert_cost[order], miss_cost[order]
    n = len(s)
    cum_alert = np.concatenate([[0.0], np.cumsum(a)])
    cum_miss = np.concatenate([[0.0], np.cumsum(m)])
    cost = cum_alert + (cum_miss[-1] - cum_miss)  # alert the top k trucks, k = 0..n

    # a cut after k trucks is allowed only between two different scores (or at the ends)
    ks = [0] + [k for k in range(1, n + 1) if k == n or s[k - 1] != s[k]]
    thresholds = []
    for k in ks:
        if k == 0:
            thresholds.append(math.inf)
        elif k == n:
            thresholds.append(0.0 if s[n - 1] >= 0 else float(s[n - 1]))
        else:
            mid = (s[k - 1] + s[k]) / 2
            thresholds.append(float(mid) if s[k] < mid <= s[k - 1] else float(s[k - 1]))
    return pd.DataFrame({"threshold": thresholds, "cost": cost[ks], "n_alerts": ks})


def choose_threshold(
    true_class: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray, costs: BinaryCosts
) -> ThresholdChoice:
    """Threshold with the lowest total cost; among equal costs the one with the fewest alerts."""
    curve = cost_curve(true_class, p, costs)
    best = curve["cost"].to_numpy().min()
    row = curve[curve["cost"] == best].iloc[0]  # curve runs from few to many alerts
    return ThresholdChoice(float(row["threshold"]), float(row["cost"]), int(row["n_alerts"]))


def apply_threshold(p: Sequence[float] | np.ndarray, threshold: float) -> np.ndarray:
    """Boolean alert vector: ``p >= threshold`` (``inf`` -> nobody is alerted)."""
    return np.asarray(p, dtype=float) >= threshold


def top_fraction_alerts(p: Sequence[float] | np.ndarray, fraction: float) -> np.ndarray:
    """Alert exactly the ``ceil(fraction * n)`` highest scores (stable order for ties)."""
    scores = np.asarray(p, dtype=float)
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = math.ceil(fraction * len(scores))
    alerts = np.zeros(len(scores), dtype=bool)
    alerts[np.argsort(-scores, kind="stable")[:k]] = True
    return alerts


@dataclass(frozen=True)
class CrossFitResult:
    """Cross-fitted decisions: every truck is judged by a threshold chosen without it."""

    alerts: np.ndarray  # pooled boolean decisions in the input order
    folds: pd.DataFrame  # per fold: threshold, n, n_alerts, cost, always_cost, saving
    cost: float  # pooled cost = sum over folds


def cross_fit_threshold(
    true_class: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    costs: BinaryCosts,
    n_splits: int,
    seed: int,
) -> CrossFitResult:
    """Choose the threshold on all folds but one, apply it to the held-out fold (5 times).

    Folds are ``StratifiedGroupKFold`` by truck on the binary label (class >= 1).
    """
    classes = np.asarray(true_class).astype(int)
    scores = np.asarray(p, dtype=float)
    alerts = np.zeros(len(scores), dtype=bool)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        cross_fit_splits((classes >= 1).astype(int), groups, n_splits, seed)
    ):
        choice = choose_threshold(classes[train_idx], scores[train_idx], costs)
        decided = apply_threshold(scores[test_idx], choice.threshold)
        alerts[test_idx] = decided
        always = total_cost(classes[test_idx], np.ones(len(test_idx), dtype=bool), costs)
        cost = total_cost(classes[test_idx], decided, costs)
        rows.append(
            {"fold": fold, "threshold": choice.threshold, "n": len(test_idx),
             "n_alerts": int(decided.sum()), "cost": cost, "always_cost": always,
             "saving_vs_always": always - cost}
        )  # fmt: skip
    folds = pd.DataFrame(rows)
    return CrossFitResult(alerts, folds, float(folds["cost"].sum()))


def saving_summary(folds: pd.DataFrame, n_trucks: int) -> dict[str, float]:
    """Saving vs. "always alert": total, per truck and the fold-to-fold spread.

    ``clear`` is True when the mean per-fold saving is larger than the standard deviation of
    the per-fold savings (ddof = 1) - a deliberately low bar; a smaller saving is noise.
    """
    saving = folds["saving_vs_always"].to_numpy(dtype=float)
    std = float(saving.std(ddof=1)) if len(saving) > 1 else float("nan")
    return {
        "total_saving": float(saving.sum()),
        "saving_per_truck": float(saving.sum() / n_trucks),
        "fold_mean": float(saving.mean()),
        "fold_std": std,
        "folds_better": int((saving > 0).sum()),
        "n_folds": int(len(saving)),
        "clear": bool(saving.mean() > std) if len(saving) > 1 else False,
    }


def baseline_table(
    true_class: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    costs: BinaryCosts,
    top_fraction: float,
    theoretical: float,
) -> pd.DataFrame:
    """Cost of the reference policies: never, always, top fraction, theoretical threshold."""
    classes = np.asarray(true_class).astype(int)
    n = len(classes)
    policies = {
        "never_alert": np.zeros(n, dtype=bool),
        "always_alert": np.ones(n, dtype=bool),
        f"top_{top_fraction:.0%}": top_fraction_alerts(p, top_fraction),
        "theoretical_threshold": apply_threshold(p, theoretical),
    }
    return pd.DataFrame(
        [
            {"policy": name, "n_alerts": int(a.sum()), "cost": total_cost(classes, a, costs)}
            for name, a in policies.items()
        ]
    )


def decision_report(true_class: Sequence[int] | np.ndarray, alert: np.ndarray) -> dict[str, Any]:
    """Confusion matrix, per-class recall and the trucks that were NOT alerted.

    ``not_alerted_positives`` = true class 1..4 trucks among the not alerted ones (the misses).
    """
    classes = np.asarray(true_class).astype(int)
    flags = np.asarray(alert, dtype=bool)
    if classes.shape != flags.shape:
        raise ValueError("true_class and alert must have the same length")
    pos = classes >= 1
    per_class = {}
    for k in range(1, 5):
        n_k = int((classes == k).sum())
        caught = int(((classes == k) & flags).sum())
        per_class[k] = {"n": n_k, "alerted": caught, "missed": n_k - caught,
                        "recall": caught / n_k if n_k else float("nan")}  # fmt: skip
    return {
        "tp": int((pos & flags).sum()), "fp": int((~pos & flags).sum()),
        "fn": int((pos & ~flags).sum()), "tn": int((~pos & ~flags).sum()),
        "n_alerted": int(flags.sum()), "n_not_alerted": int((~flags).sum()),
        "recall": float((pos & flags).sum() / pos.sum()) if pos.any() else float("nan"),
        "precision": float((pos & flags).sum() / flags.sum()) if flags.any() else float("nan"),
        "per_class": per_class,
        "not_alerted_positives": {k: v["missed"] for k, v in per_class.items()},
    }  # fmt: skip
