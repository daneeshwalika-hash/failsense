"""Tests for failsense.models.matched_check (decision rule and paired bootstraps)."""

from __future__ import annotations

import numpy as np

from failsense.models import matched_check as mc
from failsense.models.cost import binary_costs

MATRIX = np.array(
    [[0, 7, 8, 9, 10], [200, 0, 7, 8, 9], [300, 200, 0, 7, 8], [400, 300, 200, 0, 7],
     [500, 400, 300, 200, 0]],
    dtype=float,
)  # fmt: skip
COSTS = binary_costs(MATRIX, 1, "primary")


def test_decide_sampling_needs_both_metrics_above_the_spread() -> None:
    choice, reason = mc.decide_sampling(0.010, 0.005, 800.0, 500.0)
    assert choice == "matched" and "PR-AUC and in cost" in reason

    for pr_diff, saving in ((0.010, 100.0), (0.002, 800.0), (0.002, 100.0), (-0.01, 800.0)):
        choice, reason = mc.decide_sampling(pr_diff, 0.005, saving, 500.0)
        assert choice == "k5" and reason.startswith("keep k5")


def test_decide_sampling_is_strict_at_the_boundary() -> None:
    assert mc.decide_sampling(0.005, 0.005, 800.0, 500.0)[0] == "k5"
    assert mc.decide_sampling(0.0051, 0.005, 500.0, 500.0)[0] == "k5"


def synthetic(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes = rng.choice([0] * 25 + [1, 2, 3, 4], size=n)
    good = np.clip(0.03 + 0.12 * (classes > 0) + rng.normal(0, 0.04, n), 0.001, 0.99)
    weak = np.clip(0.03 + 0.02 * (classes > 0) + rng.normal(0, 0.04, n), 0.001, 0.99)
    return classes, np.arange(n), good, weak


def test_paired_bootstrap_prefers_the_better_scores_and_is_reproducible() -> None:
    classes, _, good, weak = synthetic()
    y = (classes >= 1).astype(int)

    a = mc.paired_bootstrap(y, good, weak, 60, seed=1)

    assert a == mc.paired_bootstrap(y, good, weak, 60, seed=1)
    assert a["diff_ci_low"] > 0 and a["share_a_better"] > 0.95 and a["diff_std"] > 0
    assert a["a_ci"][0] < a["a_ci"][1]


def test_paired_bootstrap_of_identical_scores_is_zero() -> None:
    classes, _, good, _ = synthetic()
    y = (classes >= 1).astype(int)

    out = mc.paired_bootstrap(y, good, good, 30, seed=2)

    assert out["diff_std"] == 0.0 and out["diff_ci_low"] == 0.0 == out["diff_ci_high"]


def test_paired_cost_bootstrap_better_scores_are_cheaper() -> None:
    classes, vehicles, good, weak = synthetic(1200)

    out = mc.paired_cost_bootstrap(classes, vehicles, good, weak, COSTS, 10, 5, seed=3)

    assert out["share_a_cheaper"] >= 0.8 and out["diff_std"] > 0
    assert out == mc.paired_cost_bootstrap(classes, vehicles, good, weak, COSTS, 10, 5, seed=3)


def test_paired_cost_bootstrap_of_identical_scores_is_zero() -> None:
    classes, vehicles, good, _ = synthetic(800)

    out = mc.paired_cost_bootstrap(classes, vehicles, good, good, COSTS, 5, 5, seed=4)

    assert out["diff_std"] == 0.0 and out["diff_ci_low"] == 0.0
