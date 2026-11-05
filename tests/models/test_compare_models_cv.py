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
    BOOSTING_FACTORIES,
    REFERENCE,
    aggregate_importances,
    best_by_mean,
    build_models,
    build_rechecks,
    permutation_top,
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
    "random_forest",
    "hist_gb",
    "xgboost",
    "lightgbm",
]
TREE_BASED = [
    "tree_depth4",
    "tree_depth8",
    "tree_depthNone",
    "random_forest",
    "hist_gb",
    "xgboost",
    "lightgbm",
]
BOOSTERS = ["hist_gb", "xgboost", "lightgbm"]


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
    assert {m.family for m in models.values()} == {
        "logreg", "knn", "naive_bayes", "tree", "forest", "boosting",
    }  # fmt: skip


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


@pytest.mark.parametrize("name", TREE_BASED)
def test_tree_models_get_no_scaler_and_no_log1p(data, name: str) -> None:
    _, _, cols = data
    pipeline = build_models(cols, 42)[name].pipeline
    branches = _numeric_branches(pipeline)
    assert list(branches) == ["num"]  # no log1p branch
    assert [step for step, _ in branches["num"].steps] == ["impute"]  # no scale step


def test_tree_settings(data) -> None:
    _, _, cols = data
    tree = build_models(cols, 42)["tree_depth8"].pipeline.named_steps["model"]
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
        select_models(models, ["catboost"])
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
    assert "Biased" in (reports / "compare_models_q1_tree_importances.md").read_text()


def test_time_one_fold_projects_the_runtime(data) -> None:
    config, _, _ = data
    out = time_one_fold(config, ["knn_k25_uniform"])
    row = out["knn_k25_uniform"]
    assert row["fold_s"] > 0
    assert row["projected_sequential_s"] == pytest.approx(
        row["fold_s"] * 3
    )  # 3 folds in the fixture
    assert 0 <= row["mean_proba"] <= 1


# --- ensembles and boosting (04.11) ---------------------------------------------------
def test_random_forest_settings(data) -> None:
    _, _, cols = data
    forest = build_models(cols, 42, model_jobs=3)["random_forest"].pipeline.named_steps["model"]
    assert forest.n_estimators == 300 and forest.class_weight == "balanced_subsample"
    assert forest.min_samples_leaf == 50 and forest.n_jobs == 3 and forest.random_state == 42


@pytest.mark.leakage
def test_no_early_stopping_anywhere_and_no_class_weights_at_first(data) -> None:
    _, _, cols = data
    models = build_models(cols, 42, model_jobs=2)
    hgb = models["hist_gb"].pipeline.named_steps["model"]
    # "auto" would switch early stopping on for > 10,000 rows with a random row split
    assert hgb.early_stopping is False and hgb.class_weight is None
    xgb = models["xgboost"].pipeline.named_steps["model"]
    lgbm = models["lightgbm"].pipeline.named_steps["model"]
    assert (
        xgb.early_stopping_rounds is None and lgbm.get_params().get("early_stopping_round") is None
    )
    assert xgb.scale_pos_weight == 1.0 and lgbm.scale_pos_weight == 1.0  # no class weights
    assert xgb.n_estimators == lgbm.n_estimators == hgb.max_iter == 300  # same fixed budget
    assert xgb.learning_rate == lgbm.learning_rate == hgb.learning_rate == 0.05
    assert xgb.n_jobs == lgbm.n_jobs == 2  # one level of parallelism: threads per model


@pytest.mark.parametrize("name", ["random_forest", *BOOSTERS])
def test_same_seed_gives_identical_predictions(data, name: str) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    first = build_models(cols, 42, model_jobs=2)[name].pipeline.fit(x, y).predict_proba(x)
    second = build_models(cols, 42, model_jobs=2)[name].pipeline.fit(x, y).predict_proba(x)
    # threads add up the trees' votes in varying order: equal up to float rounding
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)


def test_different_seed_changes_a_random_model(data) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    a = build_models(cols, 1, 2)["random_forest"].pipeline.fit(x, y).predict_proba(x)
    b = build_models(cols, 2, 2)["random_forest"].pipeline.fit(x, y).predict_proba(x)
    assert not np.array_equal(a, b)


@pytest.fixture
def recheck_inputs(data):
    _, _, cols = data
    return cols


@pytest.mark.parametrize("best", BOOSTERS)
def test_rechecks_variants(recheck_inputs, best: str) -> None:
    cols = recheck_inputs
    variants = build_rechecks(best, cols, 42, 2, positive_weight=35.0)
    assert list(variants) == [
        f"{best}__no_specs", f"{best}__no_time_step", f"{best}__engineered", f"{best}__pos_weight",
    ]  # fmt: skip
    default = set(cols.raw) | set(cols.categorical)
    no_specs = set(variants[f"{best}__no_specs"].columns)
    assert no_specs == set(cols.raw) and not no_specs & set(cols.categorical)
    no_ts = set(variants[f"{best}__no_time_step"].columns)
    assert no_ts == default - {"time_step"}
    engineered = set(variants[f"{best}__engineered"].columns)
    assert engineered == default | set(cols.engineered) and len(cols.engineered) == 125
    assert all(v.family == "boosting" for v in variants.values())


@pytest.mark.parametrize("best", BOOSTERS)
def test_pos_weight_variant_sets_the_weight_per_family(recheck_inputs, best: str) -> None:
    variants = build_rechecks(best, recheck_inputs, 42, 2, positive_weight=35.0)
    model = variants[f"{best}__pos_weight"].pipeline.named_steps["model"]
    if best == "hist_gb":
        assert model.class_weight == "balanced"
    else:
        assert model.scale_pos_weight == 35.0
    for other in ("no_specs", "no_time_step", "engineered"):  # the other variants stay unweighted
        m = variants[f"{best}__{other}"].pipeline.named_steps["model"]
        assert (
            getattr(m, "scale_pos_weight", 1.0) == 1.0 and getattr(m, "class_weight", None) is None
        )


def test_no_specs_variant_has_no_categorical_branch(recheck_inputs) -> None:
    variants = build_rechecks("xgboost", recheck_inputs, 42, 2, 35.0)
    pre = variants["xgboost__no_specs"].pipeline.named_steps["preprocess"]
    assert [name for name, *_ in pre.transformers] == ["num"]


@pytest.mark.leakage
@pytest.mark.parametrize("best", BOOSTERS)
def test_every_recheck_variant_refuses_forbidden_columns(data, best: str) -> None:
    _, frame, cols = data
    for variant in build_rechecks(best, cols, 42, 2, 35.0).values():
        x = frame[list(variant.columns)].assign(vehicle_id=frame["vehicle_id"].to_numpy())
        with pytest.raises(ValueError, match="leakage"):
            variant.pipeline.fit(x, frame["repair_within_horizon"].to_numpy(dtype=int))


def test_unknown_boosting_model_raises(recheck_inputs) -> None:
    with pytest.raises(ValueError, match="unknown boosting model"):
        build_rechecks("random_forest", recheck_inputs, 42, 2, 35.0)
    assert set(BOOSTING_FACTORIES) == set(BOOSTERS)


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "lightgbm"])
def test_importances_of_ensembles_are_normalised(data, name: str) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    models = [build_models(cols, 42, 2)[name].pipeline.fit(x.iloc[s], y[s])
              for s in (slice(0, 120), slice(40, len(x)))]  # fmt: skip
    table = aggregate_importances(models)
    total = table["mean_importance"].sum()
    # each fold model sums to 1, or to 0 if the tiny fixture allows no split at all
    assert 0.0 <= total <= 1.0 + 1e-9
    assert table["mean_importance"].is_monotonic_decreasing
    assert not any("scaled" in k for k in table["kind"])


def test_permutation_top_for_models_without_importances(data) -> None:
    _, frame, cols = data
    x, y = _x_y(frame, cols)
    model = build_models(cols, 42, 2)["hist_gb"].pipeline.fit(x, y)
    top = permutation_top(model, x.iloc[:80], y[:80], random_state=0, jobs=1, n_repeats=2)
    assert set(top["feature"]) == set(x.columns)  # the raw input columns, not preprocessed ones
    assert top["mean_importance"].is_monotonic_decreasing
    assert (top["n_folds"] == 2).all()


def test_run_with_boosting_writes_the_recheck_table(data, tmp_path: Path) -> None:
    config, _, _ = data
    summary = run(config, ["hist_gb", "xgboost", "lightgbm"], track=False)
    report = summary["_report"]
    assert report["best_boosting"] in BOOSTERS
    assert report["positive_weight"] == pytest.approx(
        (1 - summary_positive_rate(config)) / summary_positive_rate(config)
    )
    best = report["best_boosting"]
    for suffix in ("no_specs", "no_time_step", "engineered", "pos_weight"):
        entry = summary[f"{best}__{suffix}"]
        assert entry["recheck_of"] == best
        assert set(entry["paired"]) == {best, REFERENCE}
    reports = tmp_path / "reports"
    recheck = pd.read_csv(reports / "compare_models_q1_recheck.csv")
    assert set(recheck["base_model"]) == {best} and len(recheck) == 4
    main = pd.read_csv(reports / "compare_models_q1.csv")
    assert not main["model"].str.contains("__").any()  # variants are not in the main table
    assert (reports / "compare_models_q1_boosting_importances.csv").exists()
    assert "Biased" in (reports / "compare_models_q1_boosting_importances.md").read_text()


def summary_positive_rate(config) -> float:
    frame = load_training_frame(config, 5)
    return float(frame["repair_within_horizon"].mean())


def test_run_without_boosting_has_no_recheck(data, tmp_path: Path) -> None:
    config, _, _ = data
    summary = run(config, ["gaussian_nb"], track=False)
    assert summary["_report"]["best_boosting"] is None
    assert not (tmp_path / "reports" / "compare_models_q1_recheck.csv").exists()
    assert run(config, ["xgboost"], track=False, rechecks=False)["_report"]["best_boosting"] is None


def test_mlflow_runs_carry_the_comparison_set_tag_and_old_runs_are_kept(data) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    from failsense.tracking import setup_mlflow

    config, _, _ = data
    experiment_id = setup_mlflow(config)
    with mlflow.start_run(run_name="old_set_run") as old:  # a run of an earlier comparison
        mlflow.set_tag("stage", "comparison")
    run(config, ["xgboost"], track=True, comparison_set="vtest")
    client = MlflowClient()
    new_runs = client.search_runs([experiment_id], filter_string="tags.comparison_set = 'vtest'")
    names = {r.info.run_name for r in new_runs}
    assert {"logreg_reference_k5", "xgboost_k5"} <= names
    assert {
        f"xgboost__{s}_k5" for s in ("no_specs", "no_time_step", "engineered", "pos_weight")
    } <= names
    assert all(r.data.tags["stage"] == "comparison" for r in new_runs)
    variant = next(r for r in new_runs if r.info.run_name == "xgboost__no_specs_k5")
    assert (
        variant.data.tags["recheck"] == "no_specs" and variant.data.tags["base_model"] == "xgboost"
    )
    # the earlier run is untouched: still active and not part of the new set
    assert client.get_run(old.info.run_id).info.lifecycle_stage == "active"
    assert old.info.run_id not in {r.info.run_id for r in new_runs}
    best = next(r for r in new_runs if r.info.run_name == "xgboost_k5")
    assert any("boosting_importances" in a.path for a in client.list_artifacts(best.info.run_id))
