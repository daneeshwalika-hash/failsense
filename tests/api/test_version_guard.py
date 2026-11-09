"""The API refuses to serve a model version its calibrator was not fitted for."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient

from failsense.api.service import ModelConfigError, ModelService
from failsense.models.register import TRUSTED_TYPES

CALIBRATOR = {"kind": "sigmoid", "slope": 0.5, "intercept": -2.0, "logit_clip_eps": 1e-6}


@pytest.fixture
def registry_config(tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path, monkeypatch) -> dict:  # type: ignore[no-untyped-def]
    """A temporary registry with failsense-q1@candidate = version 2 (two versions registered)."""
    monkeypatch.delenv("FAILSENSE_MLFLOW_URI", raising=False)
    pipeline, frame, _ = tiny_pipeline
    db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    experiment = mlflow.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    for _ in range(2):
        with mlflow.start_run(experiment_id=experiment):
            info = mlflow.sklearn.log_model(
                pipeline, name="model", input_example=frame.head(3),
                skops_trusted_types=TRUSTED_TYPES, registered_model_name="failsense-q1",
            )  # fmt: skip
    MlflowClient().set_registered_model_alias(
        "failsense-q1", "candidate", str(info.registered_model_version)
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "registry_q1.json").write_text(json.dumps({"input_columns": list(frame.columns)}))
    summary = tmp_path / "summary.json"

    def write(version: object | None) -> None:
        block: dict[str, Any] = {"model_uri": "models:/failsense-q1@candidate",
                                 "calibrator": CALIBRATOR, "threshold": 0.05}  # fmt: skip
        if version is not None:
            block["model_version"] = version
        summary.write_text(json.dumps({"deployment_choice": block}))

    write(None)
    config = {
        "api": {"summary_path": str(summary), "summary_key": "deployment_choice"},
        "tuning": {"registered_model": "failsense-q1", "alias": "candidate"},
        "mlflow": {"db_path": str(db), "artifact_dir": str(tmp_path / "art"), "experiment": "e"},
        "paths": {"reports": str(reports)},
        "features": {"top_fraction": 0.2},
        "_write_summary": write,
    }
    return config


def test_a_matching_version_loads_and_serves(registry_config: dict) -> None:
    registry_config["_write_summary"]("2")
    service = ModelService.from_config(registry_config)
    assert service.info.version == "2" and service.info.alias == "candidate"


def test_a_summary_for_another_version_is_refused_with_the_fix(registry_config: dict) -> None:
    registry_config["_write_summary"]("1")  # the alias points to version 2
    with pytest.raises(ModelConfigError, match=r"version 1.*version 2.*make calibrate"):
        ModelService.from_config(registry_config)


def test_the_version_may_be_stored_as_a_number(registry_config: dict) -> None:
    registry_config["_write_summary"](2)
    assert ModelService.from_config(registry_config).info.version == "2"


def test_an_old_summary_without_a_version_still_loads_with_a_warning(
    registry_config: dict, caplog: pytest.LogCaptureFixture
) -> None:
    registry_config["_write_summary"](None)
    with caplog.at_level("WARNING"):
        service = ModelService.from_config(registry_config)
    assert service.info.version == "2" and "does not name a model version" in caplog.text


def test_the_committed_summary_names_the_registered_version() -> None:
    from failsense.api.service import load_deployment
    from failsense.config import load_config, resolve_path
    from failsense.models.register import REGISTRY_FILE

    config = load_config()
    deployment = load_deployment(config)
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / REGISTRY_FILE).read_text(encoding="utf-8")
    )
    assert str(deployment["model_version"]) == str(registry["version"])
