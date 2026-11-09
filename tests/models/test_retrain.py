"""Champion / challenger: the promotion rule, the cost bookkeeping, a dry run and a promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from failsense.config import load_config
from failsense.models import retrain as rt
from failsense.models.boosters import STEP6_PARAMS
from failsense.models.cost import alert_class_from_config, binary_costs, cost_matrix_from_config

pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


def costs() -> Any:
    config = load_config()
    return binary_costs(cost_matrix_from_config(config), alert_class_from_config(config), "primary")


# --- the promotion rule -----------------------------------------------------------------------
def test_a_clearly_cheaper_challenger_is_promoted() -> None:
    champion = np.full(400, 10.0)
    challenger = np.full(400, 7.0)
    decision = rt.decide_promotion(challenger, champion, 200, seed=1)
    assert decision["promote"] is True and decision["diff"] == -1200 and decision["ci_high"] < 0


def test_an_equal_or_more_expensive_challenger_is_not_promoted() -> None:
    cost = np.random.default_rng(0).choice([0.0, 7.0, 300.0], 500, p=[0.4, 0.5, 0.1])
    assert rt.decide_promotion(cost, cost, 200, 1)["promote"] is False  # identical
    assert rt.decide_promotion(cost + 1.0, cost, 200, 1)["promote"] is False  # a bit worse
    assert rt.decide_promotion(cost * 2, cost, 200, 1)["promote"] is False  # much worse


def test_a_cheaper_challenger_inside_the_noise_is_not_promoted() -> None:
    """Cheaper in total, but only 3 trucks differ: the 95 % interval contains 0."""
    rng = np.random.default_rng(3)
    champion = rng.choice([0.0, 7.0, 400.0], 600, p=[0.45, 0.5, 0.05])
    challenger = champion.copy()
    misses = np.flatnonzero(champion == 400.0)
    challenger[misses[:3]] = 7.0  # the challenger catches three more failures
    decision = rt.decide_promotion(challenger, champion, 500, 2)
    assert decision["diff"] == pytest.approx(-3 * 393)  # cheaper on the data ...
    assert decision["ci_high"] >= 0 and decision["promote"] is False  # ... but not with confidence


def test_the_decision_names_the_rule_and_is_reproducible() -> None:
    a, b = np.arange(100.0), np.arange(100.0) + 3
    one, two = rt.decide_promotion(a, b, 100, 5), rt.decide_promotion(a, b, 100, 5)
    assert one == two and "95 %" in one["rule"]


# --- cost bookkeeping -------------------------------------------------------------------------
def test_per_truck_costs_add_up_to_the_cross_fitted_total() -> None:
    rng = np.random.default_rng(0)
    n = 600
    classes = np.where(rng.random(n) < 0.08, rng.integers(1, 5, n), 0)
    scores = np.clip(0.02 + 0.3 * (classes > 0) + rng.normal(0, 0.05, n), 0, 1)
    groups = np.arange(n) // 2  # two readouts per truck
    per_truck, total = rt.decision_costs(classes, scores, groups, costs(), 5, seed=1)
    assert per_truck.shape == (n,) and per_truck.sum() == pytest.approx(total)
    never = costs().fn[classes[classes > 0] - 1].sum()
    assert total < never  # a score that carries signal beats "never alert"
    assert set(np.unique(per_truck)) <= {0.0, 7.0, *costs().fn.tolist()}  # primary mapping prices


def test_a_better_score_costs_less_than_a_random_one() -> None:
    rng = np.random.default_rng(1)
    n = 800
    classes = np.where(rng.random(n) < 0.08, rng.integers(1, 5, n), 0)
    groups = np.arange(n)
    good = np.clip(0.02 + 0.4 * (classes > 0) + rng.normal(0, 0.05, n), 0, 1)
    random_scores = rng.random(n)
    _, cost_good = rt.decision_costs(classes, good, groups, costs(), 5, 1)
    _, cost_random = rt.decision_costs(classes, random_scores, groups, costs(), 5, 1)
    assert cost_good < cost_random


# --- recipes ----------------------------------------------------------------------------------
def test_champion_and_challenger_recipes_are_read_from_the_report_files(tmp_path: Path) -> None:
    (tmp_path / "registry_q1.json").write_text(json.dumps({
        "version": "3", "family": "xgboost", "params": {"max_depth": 4}, "n_rounds": 300,
        "engineered": True,
    }))  # fmt: skip
    champion = rt.champion_from_registry(tmp_path)
    assert (champion.name, champion.n_rounds, champion.engineered) == ("champion_v3", 300, True)
    hpo = tmp_path / "hpo.json"
    hpo.write_text(json.dumps({"xgboost": {"params": {"max_depth": 5}, "n_rounds": 163,
                                           "engineered": False}}))  # fmt: skip
    challenger = rt.challenger_from_file(hpo, "xgboost")
    assert (challenger.name, challenger.params, challenger.engineered) == (
        "challenger_xgboost", {"max_depth": 5}, False)  # fmt: skip
    with pytest.raises(FileNotFoundError, match="make tune"):
        rt.challenger_from_file(tmp_path / "missing.json", "xgboost")
    with pytest.raises(KeyError):
        rt.challenger_from_file(hpo, "lightgbm")


# --- the run ----------------------------------------------------------------------------------
@pytest.fixture
def retrain_config(make_mini_scania, tmp_path: Path) -> dict[str, Any]:
    config = make_mini_scania()
    real = load_config()
    config["cost"] = real["cost"]
    config["retrain"] = {"holdout_fold": 0, "n_bootstrap": 50, "mlflow_experiment": "test-retrain",
                         "challenger_file": str(tmp_path / "hpo.json"),
                         "challenger_model": "xgboost"}  # fmt: skip
    reports = Path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    champion = {"version": "1", "family": "xgboost", "params": dict(STEP6_PARAMS["xgboost"]),
                "n_rounds": 10, "engineered": True}  # fmt: skip
    (reports / "registry_q1.json").write_text(json.dumps(champion))
    challenger = {"params": {**STEP6_PARAMS["xgboost"], "max_depth": 2}, "n_rounds": 8,
                  "engineered": True}  # fmt: skip
    (tmp_path / "hpo.json").write_text(json.dumps({"xgboost": challenger}))
    return config


def test_dry_run_writes_the_summary_and_registers_nothing(retrain_config: dict[str, Any]) -> None:
    summary = rt.run(retrain_config, promote=False)
    assert set(summary) >= {"champion", "challenger", "decision", "promoted", "holdout_fold"}
    assert summary["promoted"] is False and summary["promote_requested"] is False
    assert summary["cost_never_alert"] >= 0 and summary["n_eval_rows"] > 0
    saved = json.loads((Path(retrain_config["paths"]["reports"]) / rt.SUMMARY_FILE).read_text())
    assert saved["decision"]["rule"].startswith("challenger cheaper")
    mlflow.set_tracking_uri(f"sqlite:///{retrain_config['mlflow']['db_path']}")
    assert MlflowClient().search_registered_models() == []  # nothing was registered


def test_promote_flag_without_a_win_registers_nothing(
    retrain_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rt, "decide_promotion", lambda *a, **k: {"diff": 5.0, "std": 1.0,
                        "ci_low": 3.0, "ci_high": 7.0, "share_a_cheaper": 0.0, "promote": False,
                        "rule": "x"})  # fmt: skip
    summary = rt.run(retrain_config, promote=True)
    assert summary["promoted"] is False and "new_version" not in summary
    mlflow.set_tracking_uri(f"sqlite:///{retrain_config['mlflow']['db_path']}")
    assert MlflowClient().search_registered_models() == []


def test_a_winning_challenger_is_registered_with_the_alias_when_promotion_is_requested(
    retrain_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rt, "decide_promotion", lambda *a, **k: {"diff": -50.0, "std": 5.0,
                        "ci_low": -60.0, "ci_high": -40.0, "share_a_cheaper": 1.0,
                        "promote": True, "rule": "forced for the test"})  # fmt: skip
    retrain_config["tuning"]["registered_model"] = "failsense-q1"
    summary = rt.run(retrain_config, promote=True)
    assert summary["promoted"] is True and summary["new_version"] == "1"
    mlflow.set_tracking_uri(f"sqlite:///{retrain_config['mlflow']['db_path']}")
    version = MlflowClient().get_model_version_by_alias("failsense-q1", "candidate")
    assert str(version.version) == "1"
    facts = json.loads((Path(retrain_config["paths"]["reports"]) / "registry_q1.json").read_text())
    assert facts["winner"] == "challenger_xgboost" and facts["params"]["max_depth"] == 2


def test_a_winner_in_a_dry_run_is_reported_but_not_registered(
    retrain_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rt, "decide_promotion", lambda *a, **k: {"diff": -50.0, "std": 5.0,
                        "ci_low": -60.0, "ci_high": -40.0, "share_a_cheaper": 1.0,
                        "promote": True, "rule": "forced"})  # fmt: skip
    summary = rt.run(retrain_config, promote=False)
    assert summary["decision"]["promote"] is True and summary["promoted"] is False
