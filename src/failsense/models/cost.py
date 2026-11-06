"""SCANIA cost matrix and its binary (alert / no alert) reading for Q1.

The 5 x 5 matrix of the IDA 2024 challenge prices every (true class, predicted class)
pair. Q1 is binary: the model says "alert" or "no alert", so each alert has to be
mapped to one predicted class. Two mappings are supported (see ``configs/config.yaml``,
``cost.binary``):

* ``primary``: false alarm = ``Cost_0_<alert_class>`` (7), missed truck of class k =
  ``Cost_k_0`` (200 ... 500), alert on a truck of class 1-4 = 0.
* ``literal``: an alert *is* "predicted class ``alert_class``", so an alerted truck of
  class k costs ``Cost_k_<alert_class>`` (0, 200, 300, 400 for k = 1..4).

For a calibrated probability ``p`` an alert pays off when ``p * (FN - TP) > (1 - p) * FP``,
i.e. the theoretical cost-optimal threshold is ``t* = FP / (FP + (FN - TP))``. With
several positive classes ``FN - TP`` is averaged with the class mix of the positives.

SQL analogy: the matrix is a price table joined on (true, predicted); the total cost is
``SUM(price * COUNT(*)) GROUP BY true, predicted``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

N_CLASSES = 5
MAPPINGS = ("primary", "literal")


@dataclass(frozen=True)
class BinaryCosts:
    """Costs of the binary decision for true classes 1..4 (index 0 = class 1).

    Attributes:
        mapping: ``primary`` or ``literal``.
        fp: Cost of an alert on a truck of true class 0 (false alarm).
        fn: Cost of NO alert on a truck of true class k, k = 1..4.
        tp: Cost of an alert on a truck of true class k, k = 1..4.
    """

    mapping: str
    fp: float
    fn: np.ndarray
    tp: np.ndarray

    @property
    def benefit(self) -> np.ndarray:
        """Saving per positive truck of class k when it is alerted: ``FN_k - TP_k``."""
        return self.fn - self.tp


def cost_matrix_from_config(config: Mapping[str, Any]) -> np.ndarray:
    """Read and validate ``cost.matrix`` (5 x 5, numeric, non-negative, zero diagonal).

    Raises:
        KeyError: If the config has no ``cost.matrix``.
        ValueError: If the matrix has the wrong shape, negative or non-finite entries,
            or a non-zero diagonal (a correct prediction must be free).
    """
    raw = config["cost"]["matrix"]
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cost.matrix must be numeric: {exc}") from exc
    if matrix.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"cost.matrix must be {N_CLASSES}x{N_CLASSES}, got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("cost.matrix must contain finite, non-negative numbers")
    if np.diagonal(matrix).any():
        raise ValueError("cost.matrix must have a zero diagonal (correct predictions are free)")
    return matrix


def alert_class_from_config(config: Mapping[str, Any]) -> int:
    """``cost.binary.alert_class`` (1..4): the predicted class an alert stands for."""
    alert_class = config["cost"]["binary"]["alert_class"]
    if not isinstance(alert_class, int) or isinstance(alert_class, bool):
        raise ValueError(f"cost.binary.alert_class must be an integer, got {alert_class!r}")
    if not 1 <= alert_class < N_CLASSES:
        raise ValueError(f"cost.binary.alert_class must be in 1..4, got {alert_class}")
    return alert_class


def binary_costs(matrix: np.ndarray, alert_class: int, mapping: str = "primary") -> BinaryCosts:
    """Derive the binary costs from the 5 x 5 matrix (see the module docstring).

    Raises:
        ValueError: For an unknown mapping or an ``alert_class`` outside 1..4.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    if not 1 <= alert_class < N_CLASSES:
        raise ValueError(f"alert_class must be in 1..4, got {alert_class}")
    fn = matrix[1:, 0].copy()  # Cost_k_0: no alert on a truck of class k
    tp = np.zeros(N_CLASSES - 1) if mapping == "primary" else matrix[1:, alert_class].copy()
    return BinaryCosts(mapping=mapping, fp=float(matrix[0, alert_class]), fn=fn, tp=tp)


def class_mix(classes: Sequence[int] | np.ndarray) -> np.ndarray:
    """Share of the true classes 1..4 among the positive trucks (sums to 1).

    Raises:
        ValueError: If there is no positive truck or a class is outside 0..4.
    """
    values = np.asarray(classes)
    if values.size and (values.min() < 0 or values.max() >= N_CLASSES):
        raise ValueError("classes must be in 0..4")
    counts = np.bincount(values.astype(int), minlength=N_CLASSES)[1:].astype(float)
    if counts.sum() == 0:
        raise ValueError("no positive truck (class 1-4): the class mix is undefined")
    return counts / counts.sum()


def theoretical_threshold(costs: BinaryCosts, mix: Sequence[float] | np.ndarray) -> float:
    """Cost-optimal probability threshold ``FP / (FP + mean(FN - TP))`` for a calibrated model.

    Args:
        costs: Binary costs.
        mix: Class mix of the positives (classes 1..4, sums to 1), from TRAINING labels.

    Raises:
        ValueError: If ``mix`` is not a distribution over 4 classes or the mean saving of an
            alert is not positive (an alert would never pay off).
    """
    weights = np.asarray(mix, dtype=float)
    if (
        weights.shape != (N_CLASSES - 1,)
        or not np.isclose(weights.sum(), 1.0)
        or (weights < 0).any()
    ):
        raise ValueError("mix must be 4 non-negative weights that sum to 1")
    saving = float(weights @ costs.benefit)
    if saving <= 0:
        raise ValueError("mean saving of an alert is not positive: no threshold exists")
    return costs.fp / (costs.fp + saving)


def per_class_thresholds(costs: BinaryCosts) -> np.ndarray:
    """Theoretical threshold if all positives were of class k (k = 1..4)."""
    return costs.fp / (costs.fp + costs.benefit)


def total_cost(
    true_class: Sequence[int] | np.ndarray, alert: Sequence[bool] | np.ndarray, costs: BinaryCosts
) -> float:
    """Total cost of one alert decision per truck.

    Args:
        true_class: True class 0..4 per truck.
        alert: ``True`` where the truck is alerted.
        costs: Binary costs.

    Raises:
        ValueError: On different lengths, classes outside 0..4, or a non-boolean ``alert``.
    """
    classes = np.asarray(true_class)
    flags = np.asarray(alert)
    if classes.shape != flags.shape or classes.ndim != 1:
        raise ValueError("true_class and alert must be 1-D arrays of the same length")
    if flags.dtype != bool:
        raise ValueError(f"alert must be boolean, got dtype {flags.dtype}")
    if classes.size and (classes.min() < 0 or classes.max() >= N_CLASSES):
        raise ValueError("true_class must be in 0..4")
    classes = classes.astype(int)
    negative = classes == 0
    cost = costs.fp * np.count_nonzero(negative & flags)  # false alarms
    for k in range(1, N_CLASSES):
        of_class = classes == k
        cost += costs.fn[k - 1] * np.count_nonzero(of_class & ~flags)  # missed
        cost += costs.tp[k - 1] * np.count_nonzero(of_class & flags)  # caught (0 or literal)
    return float(cost)
