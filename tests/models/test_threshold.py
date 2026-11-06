"""Tests for failsense.models.threshold (cost curve, cross-fitting, baselines, reports)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from failsense.models.cost import binary_costs, total_cost
from failsense.models.threshold import (
    apply_threshold,
    baseline_table,
    choose_threshold,
    cost_curve,
    cross_fit_threshold,
    decision_report,
    per_truck_costs,
    saving_summary,
    top_fraction_alerts,
)

MATRIX = np.array(
    [[0, 7, 8, 9, 10], [200, 0, 7, 8, 9], [300, 200, 0, 7, 8], [400, 300, 200, 0, 7],
     [500, 400, 300, 200, 0]],
    dtype=float,
)  # fmt: skip
COSTS = binary_costs(MATRIX, 1, "primary")


def test_per_truck_costs() -> None:
    alert, miss = per_truck_costs([0, 1, 4], COSTS)

    np.testing.assert_array_equal(alert, [7, 0, 0])
    np.testing.assert_array_equal(miss, [0, 200, 500])


def test_cost_curve_hand_computed() -> None:
    # classes 0, 0, 1, 4 with scores 0.1, 0.4, 0.6, 0.9; alert the top k trucks:
    # k=0: 200+500=700, k=1: miss class 1 = 200, k=2: 0, k=3: one false alarm = 7, k=4: 14
    curve = cost_curve([0, 0, 1, 4], [0.1, 0.4, 0.6, 0.9], COSTS)

    assert list(curve.n_alerts) == [0, 1, 2, 3, 4]
    assert list(curve.cost) == [700, 200, 0, 7, 14]
    assert math.isinf(curve.threshold[0])
    assert curve.threshold[2] == pytest.approx(0.5)  # midpoint between 0.6 and 0.4
    assert choose_threshold([0, 0, 1, 4], [0.1, 0.4, 0.6, 0.9], COSTS).threshold == pytest.approx(
        0.5
    )


def test_cost_curve_matches_total_cost_of_every_threshold() -> None:
    rng = np.random.default_rng(3)
    classes = rng.choice([0, 0, 0, 0, 1, 2, 3, 4], size=300)
    p = np.round(rng.uniform(size=300), 1)  # many ties on purpose

    curve = cost_curve(classes, p, COSTS)

    for row in curve.itertuples():
        alerts = apply_threshold(p, row.threshold)
        assert alerts.sum() == row.n_alerts
        assert total_cost(classes, alerts, COSTS) == pytest.approx(row.cost)


def test_ties_are_never_split() -> None:
    curve = cost_curve([0, 1, 0], [0.5, 0.5, 0.5], COSTS)

    assert list(curve.n_alerts) == [0, 3]  # only "nobody" and "everybody"
    assert list(curve.cost) == [200, 14]


def test_no_positives_means_never_alert() -> None:
    choice = choose_threshold([0, 0, 0], [0.1, 0.5, 0.9], COSTS)

    assert math.isinf(choice.threshold) and choice.cost == 0 and choice.n_alerts == 0


def test_all_positives_means_alert_everyone() -> None:
    choice = choose_threshold([1, 4, 2], [0.1, 0.5, 0.9], COSTS)

    assert choice.threshold == 0.0 and choice.cost == 0 and choice.n_alerts == 3


def test_equal_costs_prefer_fewer_alerts() -> None:
    free = binary_costs(np.zeros((5, 5)), 1)  # every decision costs 0

    assert choose_threshold([0, 1, 0], [0.2, 0.9, 0.5], free).n_alerts == 0


def test_single_truck() -> None:
    assert choose_threshold([1], [0.3], COSTS).n_alerts == 1
    assert choose_threshold([0], [0.3], COSTS).n_alerts == 0


@pytest.mark.parametrize(
    ("classes", "p"),
    [([0, 1], [0.1]), ([], []), ([0, 1], [0.1, np.nan]), ([0, 7], [0.1, 0.2])],
)
def test_cost_curve_rejects_bad_input(classes: list, p: list) -> None:
    with pytest.raises(ValueError):
        cost_curve(classes, p, COSTS)


def test_apply_threshold_ends() -> None:
    p = np.array([0.0, 0.2, 1.0])

    assert not apply_threshold(p, math.inf).any()
    assert apply_threshold(p, 0.0).all()
    assert list(apply_threshold(p, 0.2)) == [False, True, True]


def test_top_fraction_alerts_counts_exactly_even_with_ties() -> None:
    alerts = top_fraction_alerts([0.5] * 10, 0.25)

    assert alerts.sum() == 3  # ceil(2.5)
    assert list(top_fraction_alerts([0.1, 0.9, 0.5, 0.7], 0.5)) == [False, True, False, True]
    with pytest.raises(ValueError):
        top_fraction_alerts([0.1], 0.0)


def make_fold_data(n_trucks: int = 300, rows_per_truck: int = 2, seed: int = 5):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_trucks), rows_per_truck)
    truck_class = rng.choice([0, 0, 0, 0, 0, 1, 2, 3, 4], size=n_trucks)
    classes = np.repeat(truck_class, rows_per_truck)
    p = np.clip(0.05 + 0.1 * (classes > 0) + rng.normal(0, 0.05, len(classes)), 0, 1)
    return classes, p, groups


def test_cross_fit_pools_the_fold_costs_and_uses_no_truck_twice() -> None:
    classes, p, groups = make_fold_data()

    result = cross_fit_threshold(classes, p, groups, COSTS, 5, 42)

    assert len(result.folds) == 5 and result.folds["n"].sum() == len(classes)
    assert result.cost == pytest.approx(result.folds["cost"].sum())
    assert result.cost == pytest.approx(total_cost(classes, result.alerts, COSTS))
    assert result.folds["n_alerts"].sum() == result.alerts.sum()
    assert (
        result.folds["saving_vs_always"] == result.folds["always_cost"] - result.folds["cost"]
    ).all()


def test_cross_fit_threshold_of_a_fold_ignores_the_labels_of_that_fold() -> None:
    classes, p, groups = make_fold_data()
    base = cross_fit_threshold(classes, p, groups, COSTS, 5, 42)
    from failsense.models.calibration import cross_fit_splits

    _, test_idx = cross_fit_splits((classes >= 1).astype(int), groups, 5, 42)[0]
    changed = classes.copy()
    # swap the positive classes 1<->4 and 2<->3 in fold 0: same binary labels (so the same
    # folds) but different costs; a threshold that looked at fold 0 would change
    positive = classes[test_idx] > 0
    changed[test_idx] = np.where(positive, 5 - classes[test_idx], 0)

    other = cross_fit_threshold(changed, p, groups, COSTS, 5, 42)

    assert other.folds.threshold[0] == base.folds.threshold[0]  # chosen without fold 0's labels


def test_saving_summary_hand_computed() -> None:
    import pandas as pd

    folds = pd.DataFrame({"saving_vs_always": [100.0, 200.0, 300.0, 400.0, 500.0]})
    s = saving_summary(folds, 1000)

    assert s["total_saving"] == 1500 and s["saving_per_truck"] == 1.5
    assert s["fold_mean"] == 300 and s["fold_std"] == pytest.approx(158.1139, abs=1e-3)
    assert s["clear"] and s["folds_better"] == 5

    noisy = saving_summary(
        pd.DataFrame({"saving_vs_always": [-100.0, 100.0, -100.0, 100.0, 0.0]}), 10
    )
    assert not noisy["clear"] and noisy["folds_better"] == 2


def test_baseline_table() -> None:
    classes = np.array([0, 0, 0, 0, 1, 4, 0, 0, 0, 0])
    p = np.array([0.1, 0.2, 0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1])

    table = baseline_table(classes, p, COSTS, top_fraction=0.2, theoretical=0.5).set_index("policy")

    assert table.loc["never_alert", "cost"] == 700
    assert table.loc["always_alert", "cost"] == 8 * 7
    assert table.loc["top_20%", "cost"] == 0 and table.loc["top_20%", "n_alerts"] == 2
    assert table.loc["theoretical_threshold", "n_alerts"] == 2


def test_decision_report_hand_computed() -> None:
    classes = np.array([0, 0, 0, 1, 2, 4, 4])
    alert = np.array([True, False, False, True, False, True, False])

    r = decision_report(classes, alert)

    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 2, 2)
    assert r["n_alerted"] == 3 and r["n_not_alerted"] == 4
    assert r["not_alerted_positives"] == {1: 0, 2: 1, 3: 0, 4: 1}
    assert r["per_class"][4]["recall"] == 0.5 and np.isnan(r["per_class"][3]["recall"])
    assert r["recall"] == 0.5 and r["precision"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        decision_report(classes, alert[:3])
