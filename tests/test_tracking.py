"""Tests for failsense.tracking (MLflow helpers).

Every test writes to a temporary SQLite database, never to the project's
mlflow.db. Creating an MLflow database runs its schema migrations (about
1-2 s), so one database is shared per test module and each test uses its own
experiment (named after the test) to stay independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from failsense.tracking import log_cv_run, setup_mlflow, tracking_params_from_config


@pytest.fixture(scope="module")
def tracking_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One temporary folder (and database) for all tests in this module."""
    return tmp_path_factory.mktemp("mlflow")


@pytest.fixture
def config(tracking_dir: Path, request: pytest.FixtureRequest) -> dict[str, Any]:
    """Minimal config with absolute temp paths and one experiment per test."""
    return {
        "mlflow": {
            "db_path": str(tracking_dir / "mlflow.db"),
            "artifact_dir": str(tracking_dir / "artifacts"),
            "experiment": f"test-{request.node.name}",
        }
    }


@pytest.fixture
def cv_result() -> dict[str, np.ndarray]:
    """Small fake output of cross_validate with 3 folds."""
    return {
        "test_pr_auc": np.array([0.10, 0.12, 0.11]),
        "train_pr_auc": np.array([0.15, 0.16, 0.14]),
        "fit_time": np.array([1.0, 2.0, 3.0]),
    }


def test_tracking_params_missing_section_raises() -> None:
    with pytest.raises(KeyError, match="mlflow.db_path"):
        tracking_params_from_config({})


def test_tracking_params_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="mlflow.experiment"):
        tracking_params_from_config({"mlflow": {"db_path": "x.db", "artifact_dir": "a"}})


def test_setup_creates_database_and_experiment(config: dict, tracking_dir: Path) -> None:
    experiment_id = setup_mlflow(config)

    assert (tracking_dir / "mlflow.db").exists()
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == config["mlflow"]["experiment"]
    # Artifacts go to the configured folder, not to the working directory.
    assert experiment.artifact_location.startswith((tracking_dir / "artifacts").as_uri())


def test_setup_is_idempotent(config: dict) -> None:
    assert setup_mlflow(config) == setup_mlflow(config)


def test_setup_experiment_override(config: dict) -> None:
    experiment_id = setup_mlflow(config, experiment="override-experiment")
    assert mlflow.get_experiment(experiment_id).name == "override-experiment"


def test_log_cv_run_logs_params_metrics_and_tags(config: dict, cv_result: dict) -> None:
    setup_mlflow(config)
    run_id = log_cv_run("run", cv_result, ["pr_auc"], {"k": 5}, {"task": "q1"})

    data = MlflowClient().get_run(run_id).data
    assert data.params["k"] == "5"  # MLflow stores params as strings
    assert data.tags["task"] == "q1"
    assert data.metrics["val_pr_auc_mean"] == pytest.approx(0.11)
    assert data.metrics["val_pr_auc_std"] == pytest.approx(np.std([0.10, 0.12, 0.11]))
    assert data.metrics["train_pr_auc_mean"] == pytest.approx(0.15)
    assert data.metrics["pr_auc_gap_mean"] == pytest.approx(0.04)
    assert data.metrics["fit_time_mean_s"] == pytest.approx(2.0)


def test_log_cv_run_logs_one_point_per_fold(config: dict, cv_result: dict) -> None:
    setup_mlflow(config)
    run_id = log_cv_run("run", cv_result, ["pr_auc"], {})

    history = MlflowClient().get_metric_history(run_id, "val_pr_auc_fold")
    by_step = {m.step: m.value for m in history}
    assert by_step == pytest.approx({0: 0.10, 1: 0.12, 2: 0.11})


def test_log_cv_run_without_train_scores(config: dict) -> None:
    setup_mlflow(config)
    run_id = log_cv_run("run", {"test_pr_auc": np.array([0.2, 0.3])}, ["pr_auc"], {})

    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_pr_auc_mean" in metrics
    assert "train_pr_auc_mean" not in metrics
    assert "pr_auc_gap_mean" not in metrics


def test_log_cv_run_missing_metric_raises_before_run(config: dict, cv_result: dict) -> None:
    experiment_id = setup_mlflow(config)
    with pytest.raises(ValueError, match="test_roc_auc"):
        log_cv_run("run", cv_result, ["roc_auc"], {})
    # Validation happens before start_run: no half-filled run is left behind.
    assert mlflow.search_runs(experiment_ids=[experiment_id]).empty


@pytest.mark.parametrize(
    ("result", "metrics", "message"),
    [
        ({"test_pr_auc": np.array([])}, ["pr_auc"], "no fold scores"),
        ({"test_pr_auc": np.array([0.1])}, [], "must not be empty"),
    ],
)
def test_log_cv_run_invalid_input_raises(
    config: dict, result: dict, metrics: list[str], message: str
) -> None:
    setup_mlflow(config)
    with pytest.raises(ValueError, match=message):
        log_cv_run("run", result, metrics, {})
