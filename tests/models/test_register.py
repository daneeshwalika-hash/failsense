"""Tests for failsense.models.register: the registry round trip in a temporary MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from failsense.models.boosters import STEP6_PARAMS, STEP6_ROUNDS
from failsense.models.evaluate import WINNER_FILE
from failsense.models.pipeline import FORBIDDEN_FEATURES
from failsense.models.register import REGISTRY_FILE, load_winner, register_winner
from failsense.tracking import setup_mlflow

# The tiny fixture has ~1 positive evaluation row, so some folds have none and scikit-learn warns
# that average precision is undefined there; on the real data every fold has positives.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")


def winner(family: str, engineered: bool) -> dict:
    return {
        "winner": f"{family}_tuned", "family": family, "params": dict(STEP6_PARAMS[family]),
        "engineered": engineered, "n_rounds": 12, "reason": "test",
    }  # fmt: skip


@pytest.mark.parametrize(("family", "engineered"), [("xgboost", True), ("lightgbm", False)])
def test_registry_round_trip(
    make_mini_scania, tmp_path: Path, family: str, engineered: bool
) -> None:
    config = make_mini_scania()
    setup_mlflow(config)
    facts = register_winner(config, winner(family, engineered))
    client = MlflowClient()
    version = client.get_model_version_by_alias("failsense-q1", "candidate")
    assert str(version.version) == facts["version"] == "1"
    assert facts["model_uri"] == "models:/failsense-q1@candidate"
    saved = json.loads((tmp_path / "reports" / REGISTRY_FILE).read_text())
    assert saved["name"] == "failsense-q1" and saved["family"] == family
    columns = saved["input_columns"]
    assert not set(columns) & FORBIDDEN_FEATURES  # no label, no vehicle_id
    assert ("time_step" in columns) and any(c.endswith("__delta") for c in columns) == engineered
    # the registered model loads from the alias and predicts like a model fitted again
    model = mlflow.sklearn.load_model("models:/failsense-q1@candidate")
    again = register_winner(config, winner(family, engineered))  # version 2 moves the alias
    assert again["version"] == "2"
    assert str(client.get_model_version_by_alias("failsense-q1", "candidate").version) == "2"
    assert model.predict_proba(_sample(config, columns)).shape[1] == 2


def _sample(config, columns):
    from failsense.models.baseline_cv import load_training_frame

    return load_training_frame(config, 5)[columns].head(20)


def test_untuned_winner_uses_the_step6_rounds(make_mini_scania, tmp_path: Path) -> None:
    config = make_mini_scania()
    reports = tmp_path / "reports"
    reports.mkdir()
    spec = {
        "winner": "xgboost_step6_engineered",
        "family": "xgboost",
        "params": STEP6_PARAMS["xgboost"],
        "engineered": True,
        "n_rounds": STEP6_ROUNDS,
        "reason": "untuned wins",
    }
    (reports / WINNER_FILE).write_text(json.dumps(spec))
    assert load_winner(config)["n_rounds"] == 300
    setup_mlflow(config)
    facts = register_winner(config)  # reads the winner file
    assert facts["winner"] == "xgboost_step6_engineered" and facts["n_rounds"] == 300
    model = mlflow.sklearn.load_model(facts["model_uri"])
    assert model.named_steps["model"].n_estimators == 300


def test_load_winner_needs_evaluate(make_mini_scania) -> None:
    with pytest.raises(FileNotFoundError, match="make evaluate"):
        load_winner(make_mini_scania())
    assert np is not None
