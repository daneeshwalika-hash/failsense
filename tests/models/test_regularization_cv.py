"""Tests for failsense.models.regularization_cv (tiny fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.models.pipeline import build_logreg_pipeline
from failsense.models.regularization_cv import (
    ONE_HOT_NOTE,
    REFERENCE,
    aggregate_coefficients,
    best_by_mean,
    convergence_tags,
    model_name,
    render_markdown,
    run,
    top_features,
)


@pytest.fixture
def fold_models() -> list:
    """Three L1 pipelines fitted on different parts of the same data."""
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = ((2 * x["f0"] - x["f1"] + rng.normal(scale=0.5, size=300)) > 0).astype(int).to_numpy()
    models = []
    for part in (slice(0, 200), slice(50, 250), slice(100, 300)):
        pipe = build_logreg_pipeline(42, penalty="l1", C=0.05)
        models.append(pipe.fit(x.iloc[part], y[part]))
    return models


def test_model_name() -> None:
    assert model_name("l1", 0.1) == "l1_C0.1"
    assert model_name("l2", 1.0) == "l2_C1"
    assert model_name("l1", 0.01, engineered=True) == "l1_C0.01_engineered"


def test_convergence_tags() -> None:
    assert convergence_tags(420) == {"converged": "true"}
    assert convergence_tags(999) == {"converged": "true"}
    assert convergence_tags(1000) == {"converged": "false"}  # hit max_iter
    assert convergence_tags(50, max_iter=50) == {"converged": "false"}


def test_best_by_mean() -> None:
    scores = {"a": np.array([0.1, 0.1]), "b": np.array([0.2, 0.0]), "c": np.array([0.05, 0.3])}
    assert best_by_mean(scores) == "c"
    assert best_by_mean({"x": np.array([0.1]), "y": np.array([0.1])}) == "x"  # tie: first
    with pytest.raises(ValueError, match="no scores"):
        best_by_mean({})


def test_aggregate_coefficients_columns_and_order(fold_models: list) -> None:
    table = aggregate_coefficients(fold_models)
    assert list(table.columns) == [
        "feature", "mean_coef", "mean_abs_coef", "folds_nonzero", "n_folds", "kind",
    ]  # fmt: skip
    assert (table["n_folds"] == 3).all()
    assert table["folds_nonzero"].between(0, 3).all()
    assert table["mean_abs_coef"].is_monotonic_decreasing
    assert table["feature"].iloc[0] == "num__f0"  # the strongest real signal
    assert set(table["kind"]) == {"numeric, scaled"}


def test_aggregate_coefficients_treats_a_missing_feature_as_zero(fold_models: list) -> None:
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(100, 6)), columns=[f"f{i}" for i in range(6)])
    y = (x["f0"] > 0).astype(int).to_numpy()
    x_missing = x.assign(f5=np.nan)  # all-NaN: the imputer drops it in this fold
    with pytest.warns(UserWarning, match="Skipping features"):
        odd = build_logreg_pipeline(42, penalty="l1", C=0.05).fit(x_missing, y)
    table = aggregate_coefficients([*fold_models, odd])
    row = table.set_index("feature").loc["num__f5"]
    assert row["n_folds"] == 4 and row["folds_nonzero"] <= 3


def test_aggregate_coefficients_needs_models() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate_coefficients([])


def test_top_features_skips_all_zero_and_limits(fold_models: list) -> None:
    table = aggregate_coefficients(fold_models)
    top = top_features(table, n=2)
    assert len(top) == 2 and (top["folds_nonzero"] > 0).all()
    zeros = table.assign(folds_nonzero=0, mean_abs_coef=0.0)
    assert top_features(zeros).empty


def test_markdown_has_the_one_hot_note_and_ranks(fold_models: list) -> None:
    top = top_features(aggregate_coefficients(fold_models), n=3)
    text = render_markdown({"l1_C0.05": top})
    assert ONE_HOT_NOTE in text and "NOT scaled" in text
    assert "| 1 | num__f0 |" in text and "## l1_C0.05" in text


def test_run_on_tiny_dataset_writes_report_without_tracking(
    make_mini_scania, tmp_path: Path
) -> None:
    config = make_mini_scania()
    summary = run(config, n_per_vehicle=5, track=False)
    names = [n for n in summary if not n.startswith("_")]
    assert REFERENCE in names
    assert {"l2_C0.01", "l2_C0.1", "l1_C0.01", "l1_C0.1", "l1_C1"} <= set(names)
    assert any(n.startswith("elasticnet_") for n in names)
    assert any(n.endswith("_engineered") for n in names)
    for n in names:
        assert 0 <= summary[n]["nonzero_min"] <= summary[n]["nonzero_max"] <= summary[n]["n_coefs"]
    # L2 never zeroes a coefficient; L1 with the smallest C does zero some
    assert summary["l2_C0.1"]["nonzero_min"] == summary["l2_C0.1"]["n_coefs"]
    assert summary["l1_C0.01"]["nonzero_mean"] < summary["l1_C0.01"]["n_coefs"]
    assert (tmp_path / "reports" / "regularization_q1_l1_top_features.csv").exists()
    assert (
        ONE_HOT_NOTE in (tmp_path / "reports" / "regularization_q1_l1_top_features.md").read_text()
    )
