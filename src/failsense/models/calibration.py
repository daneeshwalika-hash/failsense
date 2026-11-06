"""Probability calibration: sigmoid (Platt) and isotonic, cross-fitting, reliability tables.

A classifier's ``predict_proba`` is only a *score* until it is calibrated: "0.10" should mean
that 10 % of the trucks with that score are repaired. A calibrator is a monotone map
``score -> probability``, fitted on (score, label) pairs the model did not train on:

* ``sigmoid`` (Platt): logistic regression on ``logit(score)``, 2 parameters, smooth. Needs a
  positive slope, so it never reverses the ranking.
* ``isotonic``: the best non-decreasing step function, many parameters. Flexible, but with few
  positives it overfits (plateaus of exactly 0 or equal values); it keeps the order only
  *weakly* (ties).

Leakage rules: a calibrator is fitted on out-of-fold scores of a set with its own labels
(``eval set`` of the training trucks, or the validation split in cross-fitted mode). When a
calibrator is fitted and scored on the same split, it is **cross-fitted**: fit on 4 folds of
trucks, apply to the 5th, so no truck is in both parts.

SQL analogy: cross-fitting is like computing a lookup table from 4/5 of the customers and
applying it to the remaining 1/5, five times.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from failsense.data.split import assert_no_group_overlap
from failsense.models.eval_set import wilson_interval

logger = logging.getLogger(__name__)

KINDS = ("sigmoid", "isotonic")
_LOGIT_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _check_scores(p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray | None = None):
    """Validate scores (finite, in [0, 1]) and labels (0/1, same length)."""
    scores = np.asarray(p, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("scores must be finite probabilities in [0, 1]")
    if y is None:
        return scores, None
    labels = np.asarray(y)
    if labels.shape != scores.shape:
        raise ValueError("scores and labels must have the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return scores, labels.astype(int)


@dataclass(frozen=True)
class Calibrator:
    """A fitted monotone map from model score to probability."""

    kind: str
    model: Any

    def predict(self, p: Sequence[float] | np.ndarray) -> np.ndarray:
        """Calibrated probabilities for new scores (same length, same order)."""
        scores, _ = _check_scores(p)
        if self.kind == "sigmoid":
            return self.model.predict_proba(_logit(scores))[:, 1]
        return np.asarray(self.model.predict(scores), dtype=float)


def sigmoid_parameters(calibrator: Calibrator) -> dict[str, Any]:
    """Slope and intercept of a sigmoid calibrator, so a service can apply it without sklearn.

    ``p_cal = 1 / (1 + exp(-(intercept + slope * logit(clip(score, eps, 1 - eps)))))``.

    Raises:
        ValueError: If the calibrator is not a sigmoid.
    """
    if calibrator.kind != "sigmoid":
        raise ValueError("only a sigmoid calibrator has slope / intercept parameters")
    return {
        "kind": "sigmoid",
        "slope": float(calibrator.model.coef_[0, 0]),
        "intercept": float(calibrator.model.intercept_[0]),
        "logit_clip_eps": _LOGIT_EPS,
    }


def apply_sigmoid_parameters(params: dict[str, Any], p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply stored sigmoid parameters (the inverse of ``sigmoid_parameters``)."""
    scores, _ = _check_scores(p)
    eps = float(params["logit_clip_eps"])
    clipped = np.clip(scores, eps, 1 - eps)
    z = params["intercept"] + params["slope"] * np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-z))


def fit_calibrator(
    kind: str, p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray
) -> Calibrator:
    """Fit a calibrator on out-of-fold scores ``p`` with their labels ``y``.

    Raises:
        ValueError: For an unknown kind, invalid inputs, one-class labels, or a sigmoid with
            a non-positive slope (it would reverse the ranking).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    scores, labels = _check_scores(p, y)
    if len(np.unique(labels)) < 2:
        raise ValueError("a calibrator needs both classes in the labels")
    if kind == "sigmoid":
        # C is huge: effectively unregularised Platt scaling (2 parameters).
        model = LogisticRegression(C=1e10, max_iter=1000).fit(_logit(scores), labels)
        if model.coef_[0, 0] <= 0:
            raise ValueError("sigmoid slope is not positive: the score carries no usable signal")
    else:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(scores, labels)
    return Calibrator(kind, model)


def cross_fit_splits(
    y: Sequence[int] | np.ndarray, groups: Sequence[Any] | np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``StratifiedGroupKFold`` splits by truck; verified to share no truck between the parts."""
    labels, group_ids = np.asarray(y), np.asarray(groups)
    if labels.shape != group_ids.shape:
        raise ValueError("y and groups must have the same length")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels, group_ids):
        assert_no_group_overlap({"fit": group_ids[train_idx], "apply": group_ids[test_idx]})
        splits.append((train_idx, test_idx))
    return splits


def cross_fit_calibrate(
    kind: str,
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Calibrated probabilities; each row is scored by a calibrator that never saw its truck."""
    scores, labels = _check_scores(p, y)
    out = np.full(len(scores), np.nan)
    for train_idx, test_idx in cross_fit_splits(labels, groups, n_splits, seed):
        calibrator = fit_calibrator(kind, scores[train_idx], labels[train_idx])
        out[test_idx] = calibrator.predict(scores[test_idx])
    return out


def probability_metrics(
    y: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray, eps: float = 1e-6
) -> dict[str, float]:
    """Brier score, log loss, mean predicted probability, true rate, PR-AUC, ROC-AUC.

    The log loss clips probabilities to ``[eps, 1 - eps]`` (an isotonic 0 would give infinity).
    """
    scores, labels = _check_scores(p, y)
    clipped = np.clip(scores, eps, 1 - eps)
    both = len(np.unique(labels)) == 2
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "mean_proba": float(scores.mean()),
        "true_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
    }


def constant_scores(
    y: Sequence[int] | np.ndarray, rate: float, eps: float = 1e-6
) -> dict[str, float]:
    """Brier and log loss of the base-rate predictor (the same probability for every truck)."""
    labels = np.asarray(y).astype(int)
    full = np.full(len(labels), float(rate))
    return {
        "brier": float(brier_score_loss(labels, full)),
        "log_loss": float(log_loss(labels, np.clip(full, eps, 1 - eps), labels=[0, 1])),
    }


def reliability_table(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    n_bins: int = 10,
    p_max: float | None = None,
) -> pd.DataFrame:
    """Reliability curve with quantile bins (equal number of trucks per bin).

    Equal-width bins would put nearly every truck in the first bin (most scores are tiny).
    Per bin: mean predicted probability, observed rate and a 95 % Wilson interval.

    Args:
        p_max: Keep only trucks with ``p < p_max`` (the low-probability zoom).

    Raises:
        ValueError: If fewer rows than bins remain.
    """
    scores, labels = _check_scores(p, y)
    if p_max is not None:
        keep = scores < p_max
        scores, labels = scores[keep], labels[keep]
    if len(scores) < n_bins:
        raise ValueError(f"only {len(scores)} rows for {n_bins} bins")
    order = np.argsort(scores, kind="stable")  # ties stay together in order, bins by rank
    rows = []
    for chunk in np.array_split(order, n_bins):
        n_pos, n = int(labels[chunk].sum()), len(chunk)
        low, high = wilson_interval(n_pos, n)
        rows.append(
            {"mean_pred": float(scores[chunk].mean()), "observed": n_pos / n, "ci_low": low,
             "ci_high": high, "n": n, "n_pos": n_pos}
        )  # fmt: skip
    return pd.DataFrame(rows)
