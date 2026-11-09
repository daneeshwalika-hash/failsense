"""Q2 helpers: priors, prior shift, expected-cost decision, costs (hand-computed, no raw data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.data.labels import CLASS_COL, VEHICLE_COL
from failsense.models import q2_classes as q2

MATRIX = np.array(
    [[0, 7, 8, 9, 10], [200, 0, 7, 8, 9], [300, 200, 0, 7, 8], [400, 300, 200, 0, 7],
     [500, 400, 300, 200, 0]], dtype=float,
)  # fmt: skip


def onehot(*classes: int) -> np.ndarray:
    out = np.zeros((len(classes), 5))
    out[np.arange(len(classes)), classes] = 1.0
    return out


# --- priors ---------------------------------------------------------------------------------
def test_class_prior_counts_and_validation() -> None:
    np.testing.assert_allclose(
        q2.class_prior([0, 0, 0, 1, 2, 3, 4, 4]), [3, 1, 1, 1, 2] / np.float64(8)
    )
    with pytest.raises(ValueError, match="every class must occur"):
        q2.class_prior([0, 1, 2])
    with pytest.raises(ValueError, match="no classes"):
        q2.class_prior([])
    with pytest.raises(ValueError, match="0..4"):
        q2.class_prior([0, 1, 2, 3, 4, 5])


def test_prior_from_counts() -> None:
    np.testing.assert_allclose(q2.prior_from_counts([4910, 16, 14, 30, 76]).sum(), 1.0)
    assert q2.prior_from_counts([4910, 16, 14, 30, 76])[4] == pytest.approx(76 / 5046)
    for bad in ([1, 2, 3], [-1, 1, 1, 1, 1], [0, 0, 0, 0, 0]):
        with pytest.raises(ValueError):
            q2.prior_from_counts(bad)


# --- prior shift ----------------------------------------------------------------------------
def test_prior_shift_is_bayes_rule_by_hand() -> None:
    proba = np.array([[0.5, 0.5, 0.0, 0.0, 0.0]])
    out = q2.prior_shift(proba, [0.5, 0.25, 0.125, 0.0625, 0.0625], [0.5, 0.5, 0.0, 0.0, 0.0])
    # weights: 1 and 2 -> 0.5*1 : 0.5*2 = 1 : 2
    np.testing.assert_allclose(out, [[1 / 3, 2 / 3, 0, 0, 0]])


def test_prior_shift_identity_and_normalisation() -> None:
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), size=20)
    prior = np.array([0.9, 0.04, 0.03, 0.02, 0.01])
    np.testing.assert_allclose(q2.prior_shift(proba, prior, prior), proba)
    shifted = q2.prior_shift(proba, prior, np.array([0.5, 0.1, 0.1, 0.1, 0.2]))
    np.testing.assert_allclose(shifted.sum(axis=1), 1.0)
    assert (shifted[:, 4] >= proba[:, 4]).all()  # class 4 became more common -> its share rises


def test_prior_shift_rejects_bad_inputs() -> None:
    good = np.full((2, 5), 0.2)
    ones = np.full(5, 0.2)
    with pytest.raises(ValueError, match="shape"):
        q2.prior_shift(np.full((2, 4), 0.25), ones, ones)
    with pytest.raises(ValueError, match="sum to 1"):
        q2.prior_shift(good * 2, ones, ones)
    with pytest.raises(ValueError, match="finite"):
        q2.prior_shift(np.full((2, 5), np.nan), ones, ones)
    with pytest.raises(ValueError, match="positive"):
        q2.prior_shift(good, [0, 0.25, 0.25, 0.25, 0.25], ones)
    with pytest.raises(ValueError, match="positive"):
        q2.prior_shift(good, ones, [-0.1, 0.3, 0.3, 0.3, 0.2])
    with pytest.raises(ValueError, match="length"):
        q2.prior_shift(good, ones[:4], ones)


def test_zero_target_prior_removes_the_class() -> None:
    proba = np.full((1, 5), 0.2)
    out = q2.prior_shift(proba, np.full(5, 0.2), [0.25, 0.25, 0.25, 0.25, 0.0])
    assert out[0, 4] == 0.0 and out.sum() == pytest.approx(1.0)


# --- decisions and costs --------------------------------------------------------------------
def test_certain_probabilities_give_the_zero_cost_class() -> None:
    pred = q2.expected_cost_decision(onehot(0, 1, 2, 3, 4), MATRIX)
    assert pred.tolist() == [0, 1, 2, 3, 4]


def test_uncertain_truck_is_alerted_early_because_a_miss_costs_more() -> None:
    # 10 % chance of class 4: staying at class 0 costs 0.1 * 500 = 50; predicting 1 costs
    # 0.9 * 7 + 0.1 * 400 = 46.3 -> the expected-cost rule alerts, the arg-max would not.
    proba = np.array([[0.9, 0.0, 0.0, 0.0, 0.1]])
    assert proba.argmax() == 0
    assert q2.expected_cost_decision(proba, MATRIX).tolist() == [4]  # 0.9*10 = 9 is cheapest
    expected = proba @ MATRIX
    assert expected.argmin() == 4


def test_decision_matches_a_brute_force_search() -> None:
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(5), size=50)
    pred = q2.expected_cost_decision(proba, MATRIX)
    for p, k in zip(proba, pred, strict=True):
        costs = [sum(p[c] * MATRIX[c, j] for c in range(5)) for j in range(5)]
        assert k == int(np.argmin(costs))


def test_total_cost_and_constant_baselines_by_hand() -> None:
    true = [0, 0, 1, 4]
    assert q2.total_cost(true, [0, 1, 0, 0], MATRIX) == 0 + 7 + 200 + 500
    baselines = q2.constant_baselines(true, MATRIX)
    assert baselines["always_0"] == 200 + 500
    assert baselines["always_4"] == 10 + 10 + 9 + 0
    assert set(baselines) == {f"always_{k}" for k in range(5)}


def test_official_validation_baselines_are_reproduced() -> None:
    true = np.repeat(np.arange(5), [4910, 16, 14, 30, 76])
    baselines = q2.constant_baselines(true, MATRIX)
    assert baselines["always_0"] == 57_400  # same as "never alert" in Q1
    assert baselines["always_1"] == 76_570 and baselines["always_4"] == 49_566


def test_cost_functions_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        q2.total_cost([0, 1], [0], MATRIX)
    with pytest.raises(ValueError, match="0..4"):
        q2.total_cost([0, 5], [0, 1], MATRIX)
    with pytest.raises(ValueError, match="5 x 5"):
        q2.total_cost([0], [0], np.zeros((4, 4)))
    with pytest.raises(ValueError, match="same length"):
        q2.confusion_table([0, 1], [0])


def test_confusion_table_counts() -> None:
    table = q2.confusion_table([0, 0, 1, 4, 4], [0, 1, 1, 0, 4])
    assert table.shape == (5, 5) and table.to_numpy().sum() == 5
    assert table.loc["true 0", "class 1"] == 1 and table.loc["true 4", "class 0"] == 1


# --- bootstrap, calibration table, metrics --------------------------------------------------
def test_paired_bootstrap_detects_a_clear_difference_and_is_reproducible() -> None:
    a, b = np.full(200, 1.0), np.full(200, 3.0)
    result = q2.paired_bootstrap_cost(a, b, 200, seed=1)
    assert result["diff"] == -400 and result["share_a_cheaper"] == 1.0
    assert result["std"] == pytest.approx(0.0) and result["ci_high"] < 0
    assert result == q2.paired_bootstrap_cost(a, b, 200, seed=1)
    noisy = q2.paired_bootstrap_cost(
        np.r_[np.ones(50), np.zeros(50)], np.r_[np.zeros(50), np.ones(50)], 300, 1
    )
    assert noisy["ci_low"] < 0 < noisy["ci_high"]
    with pytest.raises(ValueError, match="equal length"):
        q2.paired_bootstrap_cost(np.ones(3), np.ones(4), 10, 1)


def test_class_calibration_table() -> None:
    proba = np.tile([0.7, 0.1, 0.1, 0.05, 0.05], (10, 1))
    y = [0] * 7 + [1] * 1 + [2] * 1 + [4] * 1
    table = q2.class_calibration_table(y, proba)
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.7) and table.loc[0, "observed"] == 0.7
    assert np.isnan(table.loc[3, "ratio"])  # class 3 never observed: ratio undefined
    with pytest.raises(ValueError, match="same length"):
        q2.class_calibration_table([0], proba)


def test_decision_metrics() -> None:
    metrics = q2.decision_metrics(np.array([0, 1, 4]), np.array([0, 1, 0]), MATRIX)
    assert metrics["total_cost"] == 500 and metrics["predicted_counts"] == [2, 1, 0, 0, 0]
    assert 0 <= metrics["macro_f1"] <= 1


# --- data ---------------------------------------------------------------------------------
def test_training_frame_with_classes_matches_the_q1_frame(make_mini_scania) -> None:  # type: ignore[no-untyped-def]
    from failsense.models.baseline_cv import load_training_frame

    config = make_mini_scania()
    plain = load_training_frame(config, 5)
    with_classes = q2.load_training_frame_with_classes(config, 5)
    pd.testing.assert_frame_equal(plain, with_classes.drop(columns=CLASS_COL))
    assert ((with_classes[CLASS_COL] >= 1) == (plain["repair_within_horizon"] == 1)).all()
    assert set(with_classes[CLASS_COL]) <= {0, 1, 2, 3, 4}
    assert VEHICLE_COL in with_classes.columns  # kept for the CV groups; never an input column


def test_multiclass_pipeline_fits_on_five_classes() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=300), "Spec_0": rng.choice(["A", "B"], 300)})
    y = np.arange(300) % 5
    pipe = q2.make_multiclass_pipeline({"max_depth": 2, "learning_rate": 0.3}, 10, 0, ["Spec_0"])
    proba = pipe.fit(x, y).predict_proba(x)
    assert proba.shape == (300, 5) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_target_prior_from_config() -> None:
    prior = q2.target_prior_from_config({"q2": {"target_class_counts": [4910, 16, 14, 30, 76]}})
    assert prior.sum() == pytest.approx(1.0) and len(prior) == 5
