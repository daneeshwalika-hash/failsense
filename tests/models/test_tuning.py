"""Tests for failsense.models.tuning_data and tune (tiny fixtures, tiny budget)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from failsense.models.boosters import FAMILIES, STEP6_PARAMS, make_booster
from failsense.models.tune import (
    BEST_PARAMS_FILE,
    anchor_trial,
    make_objective,
    run_study,
    suggest_params,
    time_one_trial,
    write_best_params,
)
from failsense.models.tuning_data import build_tuning_data, inner_split

# The tiny fixture has ~1 positive evaluation row, so some folds have none and scikit-learn warns
# that average precision is undefined there; on the real data every fold has positives.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


@pytest.fixture
def data_and_config(make_mini_scania):
    config = make_mini_scania()
    return build_tuning_data(config), config


# --- inner split ----------------------------------------------------------------------------
def test_inner_split_has_no_truck_on_both_sides_and_both_classes() -> None:
    groups = np.repeat(np.arange(40), 3)
    y = np.repeat((np.arange(40) % 5 == 0).astype(int), 3)
    train_idx, val_idx = inner_split(groups, y, 0.25, seed=1)
    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert len(train_idx) + len(val_idx) == len(y)
    assert len(np.unique(y[train_idx])) == 2 and len(np.unique(y[val_idx])) == 2
    again = inner_split(groups, y, 0.25, seed=1)
    assert train_idx.tolist() == again[0].tolist() and val_idx.tolist() == again[1].tolist()


def test_inner_split_validates_its_input() -> None:
    groups = np.repeat(np.arange(20), 2)
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="fraction"):
            inner_split(groups, np.zeros(40, dtype=int), bad, seed=0)
    with pytest.raises(ValueError, match="no inner split"):
        inner_split(groups, np.zeros(40, dtype=int), 0.3, seed=0, attempts=3)  # a single class


@pytest.mark.leakage
def test_inner_split_and_eval_rows_never_share_a_truck_with_the_training_part(
    data_and_config,
) -> None:
    data, _ = data_and_config
    assert len(data.folds) == 3
    all_eval_trucks: set[int] = set()
    for fold in data.folds:
        train_trucks = set(fold.vehicles_train)
        inner_train = set(fold.vehicles_train[fold.inner_train])
        inner_val = set(fold.vehicles_train[fold.inner_val])
        assert inner_train.isdisjoint(inner_val)  # early stopping sees only unseen trucks
        assert inner_train | inner_val == train_trucks
        for method, trucks in fold.vehicles_eval.items():
            assert train_trucks.isdisjoint(set(trucks)), method  # evaluation rows stay unseen
        calibrated_trucks = set(fold.vehicles_eval["calibrated"])
        assert calibrated_trucks.isdisjoint(all_eval_trucks)  # every truck is evaluated once
        all_eval_trucks |= calibrated_trucks
    assert len(all_eval_trucks) == 40


def test_fold_arrays_are_consistent(data_and_config) -> None:
    data, _ = data_and_config
    assert set(data.columns) == {"base", "engineered"}
    assert len(data.columns["engineered"]) == len(data.columns["base"]) + 125
    for fold in data.folds:
        for fs in ("base", "engineered"):
            assert fold.x_train[fs].dtype == np.float32
            assert fold.x_train[fs].shape[0] == len(fold.y_train)
            for method in fold.x_eval:
                assert fold.x_eval[method][fs].shape == (
                    len(fold.y_eval[method]),
                    fold.x_train[fs].shape[1],
                )
        assert fold.x_train["engineered"].shape[1] > fold.x_train["base"].shape[1]


# --- search space and objective ----------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_search_space_ranges_and_anchor(family: str) -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(25):
        trial = study.ask()
        params = suggest_params(family, trial)
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert 0.5 <= params["subsample"] <= 1.0 and 0.4 <= params["colsample_bytree"] <= 1.0
        assert 1e-2 <= params["reg_lambda"] <= 50 and 1e-3 <= params["reg_alpha"] <= 10
        if family == "xgboost":
            assert 3 <= params["max_depth"] <= 8 and 1 <= params["min_child_weight"] <= 200
        else:
            assert 8 <= params["num_leaves"] <= 128 and 20 <= params["min_child_samples"] <= 400
        study.tell(trial, 0.0)
    anchor = anchor_trial(family)
    assert anchor["engineered"] is True
    assert {k: anchor[k] for k in STEP6_PARAMS[family]} == STEP6_PARAMS[family]
    # the anchor must lie inside the search space: FixedTrial raises otherwise on a re-suggest
    suggest_params(family, optuna.trial.FixedTrial(anchor))
    with pytest.raises(ValueError, match="unknown family"):
        suggest_params("catboost", optuna.trial.FixedTrial(anchor))


@pytest.mark.parametrize("family", FAMILIES)
def test_objective_is_deterministic_for_a_fixed_trial(data_and_config, family: str) -> None:
    data, config = data_and_config
    objective = make_objective(family, data, config["tuning"], n_jobs=2)
    first = objective(optuna.trial.FixedTrial(anchor_trial(family)))
    second = objective(optuna.trial.FixedTrial(anchor_trial(family)))
    assert first == second and 0.0 <= first <= 1.0


def test_objective_uses_the_feature_choice(data_and_config) -> None:
    data, config = data_and_config
    objective = make_objective("xgboost", data, config["tuning"], n_jobs=2)
    base = objective(optuna.trial.FixedTrial({**anchor_trial("xgboost"), "engineered": False}))
    eng = objective(optuna.trial.FixedTrial(anchor_trial("xgboost")))
    assert 0.0 <= base <= 1.0 and 0.0 <= eng <= 1.0


def test_boosters_have_no_class_weights_and_unknown_family_raises() -> None:
    xgb = make_booster("xgboost", STEP6_PARAMS["xgboost"], 1, 2, 10)
    lgbm = make_booster("lightgbm", STEP6_PARAMS["lightgbm"], 1, 2, 10)
    assert xgb.get_params().get("scale_pos_weight") in (None, 1.0)
    assert lgbm.get_params().get("scale_pos_weight") in (None, 1.0)
    assert (
        xgb.get_params().get("class_weight") is None
        and lgbm.get_params().get("class_weight") is None
    )
    assert xgb.early_stopping_rounds is None  # only the tuned fits stop early
    with pytest.raises(ValueError, match="unknown family"):
        make_booster("catboost", {}, 1, 2, 10)


# --- a tiny study ---------------------------------------------------------------------------
def test_study_runs_resumes_and_writes_the_reports(data_and_config, tmp_path: Path) -> None:
    data, config = data_and_config
    summary = run_study(config, data, "xgboost", n_trials=4, track=False)
    assert summary["n_trials"] == 4 and summary["n_complete"] + summary["n_pruned"] == 4
    assert summary["n_trials_target"] == 4 and summary["seed"] == 42
    assert summary["anchor_value"] is not None  # trial 0 is the untuned step-6 anchor
    assert set(summary["params"]) >= {"learning_rate", "max_depth", "min_child_weight"}
    assert isinstance(summary["engineered"], bool) and summary["n_rounds"] >= 1
    table = pd.read_csv(tmp_path / "reports" / "hpo_q1_xgboost_trials.csv")
    assert len(table) == 4 and table.loc[0, "learning_rate"] == 0.05  # the anchor
    best = json.loads((tmp_path / "reports" / BEST_PARAMS_FILE).read_text())
    assert best["xgboost"]["best_value"] == pytest.approx(summary["best_value"])
    assert {"n_complete", "n_pruned", "n_trials_target", "timed_out", "seed"} <= set(
        best["xgboost"]
    )
    # same target again: the study (SQLite) is resumed and no new trial is run
    again = run_study(config, data, "xgboost", n_trials=4, track=False)
    assert again["n_trials"] == 4 and again["best_value"] == summary["best_value"]


def test_second_model_is_merged_into_the_best_params_file(tmp_path: Path) -> None:
    path = tmp_path / BEST_PARAMS_FILE
    fake = {
        "params": {"a": 1}, "engineered": True, "n_rounds": 5, "best_value": 0.1, "n_trials": 3,
        "n_complete": 2, "n_pruned": 1, "n_trials_target": 3, "timed_out": False,
        "timeout_minutes": 1.0, "seed": 42, "best_rounds_per_fold": [5], "fold_scores": [0.1],
        "anchor_value": 0.09,
    }  # fmt: skip
    write_best_params(path, "xgboost", fake)
    write_best_params(path, "lightgbm", {**fake, "best_value": 0.2})
    merged = json.loads(path.read_text())
    assert set(merged) == {"xgboost", "lightgbm"} and merged["lightgbm"]["best_value"] == 0.2


def test_time_one_trial_projects_the_total(data_and_config) -> None:
    data, config = data_and_config
    out = time_one_trial(config, data, "lightgbm", n_trials=10)
    assert out["trial_seconds"] > 0
    assert out["projected_minutes"] == pytest.approx(out["trial_seconds"] * 10 / 60)
