"""Tests for failsense.models.compare_models_cv (tiny fixtures, no MLflow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from failsense.models import compare_models_cv as cmp
from failsense.models.baseline_cv import load_training_frame
from failsense.models.compare_models_cv import (
    REFERENCE,
    aggregate_importances,
    best_by_mean,
    build_models,
    run,
    select_models,
    time_one_fold,
)
from failsense.models.pipeline import split_feature_columns

ALL_NAMES = [
    "logreg_reference",
    "knn_k25_uniform",
    "knn_k101_uniform",
    "knn_k101_distance",
    "gaussian_nb",
    "tree_depth4",
    "tree_depth8",
    "tree_depthNone",
]


@pytest.fixture
def data(make_mini_scania):
    config = make_mini_scania()
    frame = load_training_frame(config, 5)
    cols = split_feature_columns(frame)
    return config, frame, cols


def _x_y(frame: pd.DataFrame, cols):
    return frame[cols.raw + cols.categorical], frame["repair_within_horizon"].to_numpy(dtype=int)


def test_all_models_are_in_the_table_in_order(data) -> None:
    _, _, cols = data
    models = build_models(cols, random_state=42)
    assert list(models) == ALL_NAMES
    assert next(iter(models)) == REFERENCE
    assert {m.family for m in models.values()} == {"logreg", "knn", "naive_bayes", "tree"}


def test_knn_training_score_is_skipped_for_knn_only(data) -> None:
    _, _, cols = data
    for name, spec in build_models(cols, 42).items():
        assert spec.train_score == (spec.family != "knn"), name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_model_builds_fits_and_predicts_probabilities(data, name: str) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    model = build_models(cols, 42)[name].pipeline
    proba = model.fit(x, y).predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1.0)


def _numeric_branches(pipeline) -> dict:
    return {n: t for n, t, *_ in pipeline.named_steps["preprocess"].transformers if n != "cat"}


@pytest.mark.parametrize("name", ["tree_depth4", "tree_depth8", "tree_depthNone"])
def test_trees_get_no_scaler_and_no_log1p(data, name: str) -> None:
    _, _, cols = data
    pipeline = build_models(cols, 42)[name].pipeline
    branches = _numeric_branches(pipeline)
    assert list(branches) == ["num"]  # no log1p branch
    assert [step for step, _ in branches["num"].steps] == ["impute"]  # no scale step
    tree = pipeline.named_steps["model"]
    assert tree.class_weight == "balanced" and tree.min_samples_leaf == 100


@pytest.mark.parametrize("name", ["knn_k25_uniform", "knn_k101_uniform", "knn_k101_distance"])
def test_knn_gets_log1p_and_a_standard_scaler(data, name: str) -> None:
    _, _, cols = data
    pipeline = build_models(cols, 42)[name].pipeline
    branches = _numeric_branches(pipeline)
    assert set(branches) == {"num", "num_log1p"}
    for branch in branches.values():
        assert isinstance(branch.named_steps["scale"], StandardScaler)


def test_knn_settings(data) -> None:
    _, _, cols = data
    models = build_models(cols, 42)
    knn = {n: m.pipeline.named_steps["model"] for n, m in models.items() if m.family == "knn"}
    assert [(k.n_neighbors, k.weights) for k in knn.values()] == [
        (25, "uniform"), (101, "uniform"), (101, "distance"),
    ]  # fmt: skip


def test_gaussian_nb_has_log1p_but_no_scaler_and_no_class_weights(data) -> None:
    _, _, cols = data
    pipeline = build_models(cols, 42)["gaussian_nb"].pipeline
    branches = _numeric_branches(pipeline)
    assert set(branches) == {"num", "num_log1p"}
    assert all("scale" not in [s for s, _ in b.steps] for b in branches.values())
    assert pipeline.named_steps["model"].priors is None  # default: class frequencies


@pytest.mark.leakage
@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("leak", ["vehicle_id", "repair_within_horizon"])
def test_forbidden_column_stops_fit_for_every_model(data, name: str, leak: str) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    x = x.assign(**{leak: frame[leak].to_numpy()})
    model = build_models(cols, 42)[name].pipeline
    with pytest.raises(ValueError, match="leakage"):
        model.fit(x, y)


def test_select_models_and_best_by_mean(data) -> None:
    _, _, cols = data
    models = build_models(cols, 42)
    assert list(select_models(models, ["gaussian_nb"])) == ["gaussian_nb"]
    assert list(select_models(models, None)) == ALL_NAMES
    with pytest.raises(ValueError, match="unknown model"):
        select_models(models, ["xgboost"])
    assert best_by_mean({"a": np.array([0.1, 0.1]), "b": np.array([0.3, 0.0])}) == "b"
    with pytest.raises(ValueError, match="no scores"):
        best_by_mean({})


def test_worker_split_uses_all_cores_once(monkeypatch: pytest.MonkeyPatch) -> None:
    for cores, expected in ((16, (5, 3)), (4, (4, 1)), (1, (1, 1))):
        monkeypatch.setattr(cmp.os, "cpu_count", lambda c=cores: c)
        assert cmp._worker_split(5) == expected


def test_aggregate_importances(data) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    models = [build_models(cols, 42)["tree_depth4"].pipeline.fit(x.iloc[s], y[s]) for s in
              (slice(0, 120), slice(40, 160), slice(80, len(x)))]  # fmt: skip
    table = aggregate_importances(models)
    assert list(table.columns) == ["feature", "mean_importance", "folds_nonzero", "n_folds", "kind"]
    assert (table["n_folds"] == 3).all() and table["folds_nonzero"].between(0, 3).all()
    assert table["mean_importance"].is_monotonic_decreasing
    assert not any("scaled" in k for k in table["kind"])  # trees have no scaler
    assert table["mean_importance"].sum() <= 1.0 + 1e-9  # each tree sums to 1 (or 0 if no split)
    with pytest.raises(ValueError, match="at least one"):
        aggregate_importances([])
    knn_like = build_models(cols, 42)["gaussian_nb"].pipeline.fit(x, y)
    with pytest.raises(ValueError, match="feature_importances_"):
        aggregate_importances([knn_like])


def test_run_on_tiny_dataset_writes_the_table(data, tmp_path: Path) -> None:
    config, _, _ = data
    names = ["knn_k25_uniform", "gaussian_nb", "tree_depth4", "tree_depth8"]
    summary = run(config, names, track=False)
    assert list(summary)[0] == REFERENCE  # always included, always first
    assert set(names) <= set(summary)
    assert np.isnan(summary["knn_k25_uniform"]["train"])  # kNN: no training score
    assert not np.isnan(summary["tree_depth4"]["train"])
    assert summary["tree_depth4"]["paired"][REFERENCE]["n_folds"] == 3
    assert summary["_report"]["best_tree"] in {"tree_depth4", "tree_depth8"}
    reports = tmp_path / "reports"
    table = pd.read_csv(reports / "compare_models_q1.csv")
    assert set(table["model"]) == {REFERENCE, *names}
    assert (reports / "compare_models_q1_tree_importances.csv").exists()
    assert "biased" in (reports / "compare_models_q1_tree_importances.md").read_text()


def test_time_one_fold_projects_the_runtime(data) -> None:
    config, _, _ = data
    out = time_one_fold(config, ["knn_k25_uniform"])
    row = out["knn_k25_uniform"]
    assert row["fold_s"] > 0
    assert row["projected_sequential_s"] == pytest.approx(
        row["fold_s"] * 3
    )  # 3 folds in the fixture
    assert 0 <= row["mean_proba"] <= 1
