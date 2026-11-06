"""Tests for failsense.models.calibration (calibrators, cross-fitting, reliability table)."""

from __future__ import annotations

import numpy as np
import pytest

from failsense.models.calibration import (
    constant_scores,
    cross_fit_calibrate,
    cross_fit_splits,
    fit_calibrator,
    probability_metrics,
    reliability_table,
)


def synthetic(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Over-confident scores: the true risk is half of the score."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.4, n)
    return p, (rng.uniform(size=n) < p / 2).astype(int)


@pytest.mark.parametrize("kind", ["sigmoid", "isotonic"])
def test_calibrator_never_inverts_the_ranking(kind: str) -> None:
    p, y = synthetic()
    calibrated = fit_calibrator(kind, p, y).predict(p)

    order = np.argsort(p, kind="stable")
    assert (np.diff(calibrated[order]) >= -1e-12).all()  # non-decreasing (isotonic has ties)


def test_sigmoid_keeps_strict_order_for_distinct_scores() -> None:
    p, y = synthetic()
    grid = np.linspace(0.02, 0.4, 50)

    out = fit_calibrator("sigmoid", p, y).predict(grid)

    assert (np.diff(out) > 0).all()


@pytest.mark.parametrize("kind", ["sigmoid", "isotonic"])
def test_calibration_pulls_the_mean_to_the_true_rate(kind: str) -> None:
    p, y = synthetic()

    calibrated = fit_calibrator(kind, p, y).predict(p)

    assert abs(calibrated.mean() - y.mean()) < 0.005
    assert abs(p.mean() - y.mean()) > 0.05  # the raw scores were clearly too high


def test_isotonic_output_is_a_probability_and_clips_outside_the_range() -> None:
    p, y = synthetic()
    cal = fit_calibrator("isotonic", p, y)

    out = cal.predict([0.0, 0.5, 1.0])

    assert ((out >= 0) & (out <= 1)).all()


@pytest.mark.parametrize(
    ("kind", "p", "y", "message"),
    [
        ("other", [0.1, 0.2], [0, 1], "kind"),
        ("sigmoid", [0.1, 0.2], [0, 0], "both classes"),
        ("sigmoid", [0.1, 1.5], [0, 1], "probabilities"),
        ("sigmoid", [0.1, np.nan], [0, 1], "probabilities"),
        ("sigmoid", [0.1, 0.2, 0.3], [0, 1], "same length"),
        ("sigmoid", [0.1, 0.2], [0, 2], "0 or 1"),
        ("isotonic", [], [], "non-empty"),
    ],
)
def test_fit_calibrator_rejects_bad_input(kind: str, p: list, y: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_calibrator(kind, p, y)


def test_sigmoid_with_reversed_signal_raises() -> None:
    p = np.linspace(0.01, 0.5, 400)
    y = (p < 0.1).astype(int)  # high scores are the negatives

    with pytest.raises(ValueError, match="slope"):
        fit_calibrator("sigmoid", p, y)


def test_cross_fit_splits_share_no_truck_and_cover_every_row() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(200), 3)  # 3 rows per truck
    y = np.repeat(rng.uniform(size=200) < 0.2, 3).astype(int)

    splits = cross_fit_splits(y, groups, 5, seed=42)

    assert len(splits) == 5
    tested = np.concatenate([test for _, test in splits])
    assert sorted(tested) == list(range(len(y)))  # each row is applied exactly once
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])


def test_cross_fit_splits_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        cross_fit_splits([0, 1], [1, 2, 3], 2, 0)


def test_cross_fit_calibrate_scores_every_row_and_is_reproducible() -> None:
    p, y = synthetic(1500)
    groups = np.arange(len(p))

    a = cross_fit_calibrate("sigmoid", p, y, groups, 5, 42)
    b = cross_fit_calibrate("sigmoid", p, y, groups, 5, 42)

    assert not np.isnan(a).any()
    np.testing.assert_array_equal(a, b)


def test_probability_metrics_hand_computed() -> None:
    m = probability_metrics([0, 1], [0.2, 0.8])

    assert m["brier"] == pytest.approx(0.04)
    assert m["log_loss"] == pytest.approx(-np.log(0.8))
    assert m["mean_proba"] == pytest.approx(0.5)
    assert m["true_rate"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_probability_metrics_one_class_has_no_auc_and_clipped_log_loss() -> None:
    m = probability_metrics([0, 0], [0.0, 0.0])

    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])
    assert np.isfinite(m["log_loss"])


def test_constant_scores_hand_computed() -> None:
    scores = constant_scores([0, 0, 0, 1], 0.25)

    assert scores["brier"] == pytest.approx((3 * 0.0625 + 0.5625) / 4)


def test_reliability_table_bins_are_equally_filled_and_ordered() -> None:
    p, y = synthetic(1000)

    table = reliability_table(y, p, n_bins=10)

    assert list(table.n) == [100] * 10
    assert table.n.sum() == 1000 and table.n_pos.sum() == y.sum()
    assert table.mean_pred.is_monotonic_increasing
    assert ((table.ci_low <= table.observed) & (table.observed <= table.ci_high)).all()


def test_reliability_table_zoom_keeps_only_low_probabilities() -> None:
    p, y = synthetic(1000)

    table = reliability_table(y, p, n_bins=5, p_max=0.1)

    assert table.n.sum() == int((p < 0.1).sum())
    assert table.mean_pred.max() < 0.1


def test_reliability_table_needs_enough_rows() -> None:
    with pytest.raises(ValueError, match="bins"):
        reliability_table([0, 1], [0.1, 0.2], n_bins=10)


def test_sigmoid_parameters_reproduce_the_calibrator() -> None:
    from failsense.models.calibration import apply_sigmoid_parameters, sigmoid_parameters

    p, y = synthetic(2000)
    cal = fit_calibrator("sigmoid", p, y)

    params = sigmoid_parameters(cal)

    assert params["slope"] > 0
    np.testing.assert_allclose(apply_sigmoid_parameters(params, p), cal.predict(p), atol=1e-12)


def test_sigmoid_parameters_refuse_isotonic() -> None:
    from failsense.models.calibration import sigmoid_parameters

    p, y = synthetic(500)

    with pytest.raises(ValueError, match="sigmoid"):
        sigmoid_parameters(fit_calibrator("isotonic", p, y))
