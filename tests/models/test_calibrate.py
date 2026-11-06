"""Tests for failsense.models.calibrate (analysis on synthetic data, figures, JSON helper)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from failsense.config import load_config
from failsense.models import calibrate


def synthetic_validation(n: int = 1500, seed: int = 7):
    rng = np.random.default_rng(seed)
    classes = rng.choice([0] * 30 + [1, 2, 3, 4], size=n)
    base = np.clip(0.03 + 0.05 * (classes > 0) + rng.normal(0, 0.03, n), 0.001, 0.99)
    variants = {
        "raw": base, "sigmoid": base * 0.8, "isotonic": np.round(base * 0.8, 2),
        "sigmoid_cv": base * 0.7, "isotonic_cv": np.round(base * 0.7, 2),
    }  # fmt: skip
    return classes, np.arange(n), variants


@pytest.fixture
def analysis():
    classes, vehicles, variants = synthetic_validation()
    config = load_config()
    counts = {0: 900.0, 1: 40.0, 2: 10.0, 3: 5.0, 4: 1.0}
    return classes, variants, calibrate.analyze(classes, vehicles, variants, 0.03, counts, config)


def test_analyze_baselines_and_saving_are_consistent(analysis) -> None:
    classes, variants, result = analysis

    for pv in result["per_variant"].values():
        base = pv["baselines"].set_index("policy")["cost"]
        assert base["always_alert"] == 7 * (classes == 0).sum()
        saving = pv["cross_fit_saving"]
        assert saving["total_saving"] == pytest.approx(base["always_alert"] - pv["cross_fit"].cost)
        assert pv["in_sample"].cost <= base["always_alert"]  # "always" is one candidate
        decision = pv["cross_fit_decision"]
        assert decision["n_alerted"] + decision["n_not_alerted"] == len(classes)


def test_analyze_uses_the_eval_set_class_mix_for_the_threshold(analysis) -> None:
    _, _, result = analysis
    mix = np.array([40, 10, 5, 1]) / 56
    fn = np.array([200, 300, 400, 500])

    assert result["class_mix_eval_set"] == pytest.approx(mix)
    assert result["theoretical_threshold"] == pytest.approx(7 / (7 + mix @ fn))


def test_analyze_scores_include_both_base_rate_predictors(analysis) -> None:
    _, _, result = analysis
    names = list(result["scores"]["variant"])

    assert names[:5] == ["raw", "sigmoid", "isotonic", "sigmoid_cv", "isotonic_cv"]
    assert names[5:] == ["base_rate_eval_set", "base_rate_validation"]


def test_analyze_sensitivity_uses_the_literal_costs(analysis) -> None:
    classes, _, result = analysis
    s = result["sensitivity"]
    n = np.bincount(classes, minlength=5)

    assert s["never_alert"] == 200 * n[1] + 300 * n[2] + 400 * n[3] + 500 * n[4]
    assert s["always_alert"] == 7 * n[0] + 200 * n[2] + 300 * n[3] + 400 * n[4]
    assert s["saving_vs_best_trivial"] == pytest.approx(
        min(s["never_alert"], s["always_alert"]) - s["cross_fit_cost"]
    )


def test_threshold_rows_has_one_row_per_variant_and_class_columns(analysis) -> None:
    classes, _, result = analysis

    table = calibrate.threshold_rows(result, len(classes))

    assert len(table) == 5
    assert {"cf_missed_c1", "cf_missed_c4", "is_missed_c2", "cf_not_alerted", "clear"} <= set(table)
    assert (table.cf_not_alerted + table.cf_alerted == len(classes)).all()


def test_verdict_says_clearly_when_the_saving_is_not_larger_than_the_spread(analysis) -> None:
    _, _, result = analysis
    result["per_variant"]["sigmoid"]["cross_fit_saving"].update(clear=False, fold_std=999.0)

    assert "NOT larger" in calibrate.verdict(result, "sigmoid")
    result["per_variant"]["sigmoid"]["cross_fit_saving"].update(clear=True)
    assert "beats always-alert" in calibrate.verdict(result, "sigmoid")


def test_figures_are_written(analysis, tmp_path: Path) -> None:
    classes, variants, result = analysis
    y = (classes >= 1).astype(int)

    calibrate.plot_reliability(y, variants, "sigmoid", 10, 0.1, tmp_path / "rel.png")
    calibrate.plot_cost_curve(
        classes, variants["sigmoid"], result, "sigmoid", tmp_path / "cost.png"
    )

    assert (tmp_path / "rel.png").stat().st_size > 1000
    assert (tmp_path / "cost.png").stat().st_size > 1000


def test_bootstrap_pr_auc_is_reproducible_and_ordered() -> None:
    classes, _, variants = synthetic_validation()
    y = (classes >= 1).astype(int)

    a = calibrate.bootstrap_pr_auc(y, variants["raw"], 50, 1)

    assert a == calibrate.bootstrap_pr_auc(y, variants["raw"], 50, 1)
    assert a[0] < a[1]


def test_json_safe_handles_numpy_nan_and_inf() -> None:
    out = calibrate._json_safe(
        {1: np.int64(3), "a": np.float64(math.inf), "b": [np.nan, np.array([1.5])], "c": "x"}
    )

    assert out == {"1": 3, "a": None, "b": [None, [1.5]], "c": "x"}


def test_oof_scores_refuses_different_columns() -> None:
    data = SimpleNamespace(columns={"engineered": ["a", "b"]}, folds=[])
    registry = {"engineered": True, "input_columns": ["a", "c"]}

    with pytest.raises(ValueError, match="differ"):
        calibrate.oof_scores({"tuning": {"seed": 1}}, data, registry)


def test_deployment_record_states_the_choice_and_the_mapping_caveat(analysis) -> None:
    from failsense.models.calibration import fit_calibrator

    classes, variants, result = analysis
    y = (classes >= 1).astype(int)
    cal = fit_calibrator("sigmoid", variants["raw"], y)

    record = calibrate.deployment_record(cal, result, "sigmoid", {"model_uri": "models:/x@y"})

    pv = result["per_variant"]["sigmoid"]
    assert record["threshold"] == pv["in_sample"].threshold
    assert record["reported_cost_cross_fitted"] == pv["cross_fit"].cost  # not the optimistic one
    assert record["calibrator"]["slope"] > 0
    assert "stricter mapping" in record["caveat_literal_mapping"]
    assert "untouched" in record["test_split"]
