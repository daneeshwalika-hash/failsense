"""Tests for failsense.explain.explain_q1 and plots (selection rule, comparisons, figures)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import shap
from matplotlib import pyplot as plt

from conftest import TINY_INPUT_COLUMNS
from failsense.explain import explain_q1 as eq
from failsense.explain import plots
from failsense.explain.shap_utils import ShapExplainer

THRESHOLD = 0.035


def test_pick_story_rows_rule() -> None:
    classes = np.array([0, 4, 4, 2, 4, 1, 0])
    p_cal = np.array([0.5, 0.20, 0.01, 0.30, 0.02, 0.04, 0.001])

    tp, fn, note = eq.pick_story_rows(classes, p_cal, THRESHOLD)

    assert tp == 3  # highest p among positives (0.30); the class-0 truck with 0.5 is ignored
    assert fn == 2  # class-4 trucks below the threshold: 0.01 (pos 2) and 0.02 (pos 4)
    assert "class-4" in note


def test_pick_story_rows_falls_back_to_any_class_and_ties_take_the_first() -> None:
    classes = np.array([1, 2, 0, 3])
    p_cal = np.array([0.01, 0.01, 0.0, 0.2])

    tp, fn, note = eq.pick_story_rows(classes, p_cal, THRESHOLD)

    assert (tp, fn) == (3, 0) and "any class" in note


def test_pick_story_rows_errors() -> None:
    with pytest.raises(ValueError, match="no positive"):
        eq.pick_story_rows(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]), THRESHOLD)
    with pytest.raises(ValueError, match="no missed"):
        eq.pick_story_rows(np.array([1, 0]), np.array([0.5, 0.1]), THRESHOLD)


def test_group_shap_adds_inside_the_group_before_taking_the_absolute_value() -> None:
    by_input = pd.DataFrame({"a": [1.0, 1.0], "b": [-1.0, 1.0], "c": [0.5, -0.5]})

    out = eq.group_shap(by_input, {"ab": ["a", "b"], "c": ["c"]})

    assert out["ab"] == pytest.approx(1.0)  # rows: 0 and 2 -> mean 1
    assert out["c"] == pytest.approx(0.5)


def test_compare_rankings_ranks_only_reliable_groups() -> None:
    table = pd.DataFrame(
        {
            "group": ["A", "B", "C", "D"],
            "mean_drop": [0.05, 0.04, 0.03, 0.001],
            "reliable": [True, True, False, False],
            "shap_mean_abs": [0.9, 0.1, 0.8, 0.7],
        }
    )

    out = eq.compare_rankings(table, top_n=2)

    assert out["permutation_top_reliable"] == ["A", "B"]
    assert out["shap_top"] == ["A", "C"]
    assert out["in_both"] == ["A"] and out["only_permutation"] == ["B"]
    assert out["only_shap"] == ["C"] and out["inside_spread"] == ["C", "D"]
    assert out["n_reliable_groups"] == 2 and not math.isnan(out["spearman_all_groups"])


def test_neg_log_loss_hand_computed_and_clipped() -> None:
    assert eq.neg_log_loss(np.array([0, 1]), np.array([0.2, 0.8])) == pytest.approx(np.log(0.8))
    assert np.isfinite(eq.neg_log_loss(np.array([1, 0]), np.array([0.0, 1.0])))  # clipped


def test_log_loss_permutation_drop_is_the_rise_of_the_loss() -> None:
    from failsense.explain.importance import permutation_importance

    rng = np.random.default_rng(0)
    x = pd.DataFrame({"s": rng.normal(size=600), "n": rng.normal(size=600)})
    y = (x["s"] + rng.normal(size=600) > 0).astype(int).to_numpy()

    def predict(f: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-2 * f["s"].to_numpy()))

    table = permutation_importance(
        predict, x, y, {"s": ["s"], "n": ["n"]}, 6, 0, metric=eq.neg_log_loss
    ).set_index("group")

    assert table.loc["s", "mean_drop"] > 0.1 and table.loc["n", "mean_drop"] == 0.0
    assert -table.loc["s", "baseline"] == pytest.approx(-eq.neg_log_loss(y, predict(x)))


def test_clean_makes_values_json_safe() -> None:
    out = eq._clean({"a": np.float64(np.nan), "b": [np.int64(2), np.bool_(True)], 3: float("inf")})

    assert out == {"a": None, "b": [2, True], "3": None}


def test_story_has_factors_in_both_units_and_a_remainder(tiny_pipeline) -> None:
    pipeline, frame, _ = tiny_pipeline
    explainer = ShapExplainer(pipeline, TINY_INPUT_COLUMNS)
    row = frame.iloc[0]

    info = eq.story(explainer, row, slope=0.5, k=3, p_raw=0.2, p_cal=0.04, label="x")

    assert len(info["factors"]) == 3
    for factor in info["factors"]:
        assert factor["shap_calibrated"] == pytest.approx(0.5 * factor["shap_raw"])
    frame1 = explainer.prepare(row)
    values, _ = explainer.shap_values(frame1)
    total = sum(f["shap_raw"] for f in info["factors"]) + info["all_other_features_raw"]
    assert total == pytest.approx(values.sum(), abs=1e-5)
    assert info["margin"] == pytest.approx(info["base_value"] + values.sum(), abs=1e-4)


def test_saved_pyplot_figure_always_closes_figures(tmp_path) -> None:
    target = tmp_path / "a.png"
    with plots.saved_pyplot_figure(target):
        plt.plot([0, 1], [0, 1])
    assert target.exists() and plt.get_fignums() == []

    with pytest.raises(RuntimeError), plots.saved_pyplot_figure(tmp_path / "b.png"):
        plt.plot([0, 1], [0, 1])
        raise RuntimeError("boom")
    assert plt.get_fignums() == [] and not (tmp_path / "b.png").exists()


def test_all_figures_are_written(tiny_pipeline, tmp_path) -> None:
    pipeline, frame, _ = tiny_pipeline
    explainer = ShapExplainer(pipeline, TINY_INPUT_COLUMNS)
    x = explainer.prepare(frame.head(120))
    values, z = explainer.shap_values(x)
    expl = shap.Explanation(
        values=values, base_values=np.full(len(values), explainer.base_value), data=z,
        feature_names=explainer.feature_names,
    )  # fmt: skip

    plots.shap_beeswarm(expl, tmp_path / "bee.png", 8)
    plots.shap_bar(expl, tmp_path / "bar.png", 8)
    plots.shap_waterfall(expl[0], tmp_path / "wf.png", 6, "title")
    plots.shap_dependence(expl, "num__time_step", tmp_path / "dep.png")
    table = pd.DataFrame(
        {"group": ["a", "b"], "mean_drop": [0.1, 0.0], "std_drop": [0.01, 0.02],
         "reliable": [True, False], "shap_share": [0.7, 0.3], "gain_share": [0.6, 0.4]}
    )  # fmt: skip
    plots.plot_permutation(table, tmp_path / "perm.png", "t")
    plots.plot_group_comparison(table, tmp_path / "cmp.png")

    for name in ("bee", "bar", "wf", "dep", "perm", "cmp"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
    assert plt.get_fignums() == []
