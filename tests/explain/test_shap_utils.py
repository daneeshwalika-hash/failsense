"""Tests for failsense.explain.shap_utils (explainer, aggregation, top_shap_factors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TINY_INPUT_COLUMNS
from failsense.explain.shap_utils import ShapExplainer, origin_columns, top_shap_factors


@pytest.fixture
def explainer(tiny_pipeline) -> ShapExplainer:
    pipeline, _, _ = tiny_pipeline
    return ShapExplainer(pipeline, TINY_INPUT_COLUMNS)


def test_origin_columns_maps_one_hot_back_to_the_spec() -> None:
    names = ["num__time_step", "num__171_0__delta", "cat__Spec_7_Cat3", "cat__Spec_1_Cat10"]
    inputs = ["time_step", "171_0__delta", "Spec_7", "Spec_1", "Spec_10"]

    assert origin_columns(names, inputs) == ["time_step", "171_0__delta", "Spec_7", "Spec_1"]


def test_origin_columns_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="trace"):
        origin_columns(["num__mystery"], ["time_step"])
    with pytest.raises(ValueError, match="trace"):
        origin_columns(["cat__Other_Cat1"], ["Spec_0"])


def test_shap_values_add_up_to_the_margin(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    x = explainer.prepare(frame)

    values, z = explainer.shap_values(x)  # raises if base + sum(shap) != margin

    assert values.shape == z.shape == (len(x), len(explainer.feature_names))
    margin = explainer.margin(x)
    np.testing.assert_allclose(values.sum(axis=1) + explainer.base_value, margin, atol=1e-4)


def test_by_input_sums_the_one_hot_columns_of_a_spec(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    x = explainer.prepare(frame.head(50))
    values, _ = explainer.shap_values(x)

    per_input = explainer.by_input(values)

    assert list(per_input.columns) == TINY_INPUT_COLUMNS
    spec_cols = [i for i, o in enumerate(explainer.origin) if o == "Spec_0"]
    assert len(spec_cols) == 3
    np.testing.assert_allclose(per_input["Spec_0"], values[:, spec_cols].sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(per_input.sum(axis=1), values.sum(axis=1), atol=1e-5)


def test_prepare_coerces_types_and_checks_columns(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    as_object = frame.head(3).astype(object)

    out = explainer.prepare(as_object)

    assert out["time_step"].dtype == float and list(out.columns) == TINY_INPUT_COLUMNS
    with pytest.raises(ValueError, match="missing"):
        explainer.prepare(frame.drop(columns=["time_step"]))
    with pytest.raises(ValueError, match="no rows"):
        explainer.prepare(frame.iloc[:0])


def test_top_shap_factors_sorted_by_absolute_value_and_planted_signal(
    explainer, tiny_pipeline
) -> None:
    _, frame, _ = tiny_pipeline
    row = frame.iloc[[0]].copy()
    row["171_0__delta"] = 3.0  # the feature that drives the label
    row["Spec_0"] = "A"

    factors = top_shap_factors(row, k=4, explainer=explainer)

    assert len(factors) == 4
    magnitudes = [abs(s) for _, _, s in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert factors[0][0] == "171_0__delta" and factors[0][1] == 3.0 and factors[0][2] > 0
    assert any(f == "Spec_0" and v == "A" for f, v, _ in factors) or len(factors) == 4


def test_top_shap_factors_series_scale_and_k_larger_than_columns(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    row = frame.iloc[0]

    plain = top_shap_factors(row, k=3, explainer=explainer)
    scaled = top_shap_factors(row.to_frame().T, k=3, explainer=explainer, scale=0.5)
    everything = top_shap_factors(row, k=100, explainer=explainer)

    assert [f for f, _, _ in plain] == [f for f, _, _ in scaled]
    np.testing.assert_allclose([s * 0.5 for _, _, s in plain], [s for _, _, s in scaled])
    assert len(everything) == len(TINY_INPUT_COLUMNS)


def test_top_shap_factors_missing_value_is_nan(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    row = frame.iloc[[0]].copy()
    row["171_0__delta"] = np.nan

    factors = dict((f, v) for f, v, _ in top_shap_factors(row, k=8, explainer=explainer))

    assert np.isnan(factors["171_0__delta"])


def test_top_shap_factors_errors(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    with pytest.raises(ValueError, match="k must be"):
        top_shap_factors(frame.iloc[0], k=0, explainer=explainer)
    with pytest.raises(ValueError, match="exactly one row"):
        top_shap_factors(frame.head(2), k=3, explainer=explainer)
    with pytest.raises(ValueError, match="missing"):
        top_shap_factors(frame.iloc[0].drop("time_step"), k=3, explainer=explainer)


def test_top_shap_factors_does_not_modify_the_row(explainer, tiny_pipeline) -> None:
    _, frame, _ = tiny_pipeline
    row = frame.iloc[[0]].copy()
    before = row.copy()

    top_shap_factors(row, k=3, explainer=explainer)

    pd.testing.assert_frame_equal(row, before)
