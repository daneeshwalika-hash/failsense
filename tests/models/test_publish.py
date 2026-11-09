"""publish: copy a registered model between two MLflow registries and verify it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from failsense.models import publish
from failsense.models.register import TRUSTED_TYPES
from failsense.tracking import TRACKING_URI_ENV, setup_mlflow, tracking_uri_override


def registry(tmp_path: Path, name: str) -> str:
    """A fresh sqlite registry with its own artifact folder; returns its tracking URI."""
    folder = tmp_path / name
    folder.mkdir()
    return f"sqlite:///{folder / 'mlflow.db'}"


def register_model(uri: str, tmp_path: Path, pipeline: Any, frame: Any) -> None:
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment(
        f"src-{abs(hash(uri))}", artifact_location=(tmp_path / "src-artifacts").as_uri()
    )
    with mlflow.start_run(experiment_id=experiment):
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=frame.head(3),
            skops_trusted_types=TRUSTED_TYPES, registered_model_name="failsense-q1",
        )  # fmt: skip
    MlflowClient().set_registered_model_alias(
        "failsense-q1", "candidate", str(info.registered_model_version)
    )


def test_probe_frame_is_deterministic_and_has_the_columns_in_order() -> None:
    columns = ["time_step", "Spec_0", "171_0", "Spec_3"]
    a, b = publish.probe_frame(columns), publish.probe_frame(columns)
    assert list(a.columns) == columns and a.equals(b) and len(a) == 5
    assert (a["Spec_0"] == "Cat1").all() and a["171_0"].between(0, 100).all()


def test_copy_registers_the_alias_in_the_target_and_predicts_identically(
    tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    pipeline, frame, _ = tiny_pipeline
    source, target = registry(tmp_path, "source"), registry(tmp_path, "target")
    register_model(source, tmp_path, pipeline, frame)

    result = publish.copy_registered_model(source, target, "failsense-q1", "candidate",
                                           list(frame.columns))  # fmt: skip

    assert result == {
        "name": "failsense-q1",
        "alias": "candidate",
        "version": "1",
        "target": target,
    }
    mlflow.set_tracking_uri(target)
    version = MlflowClient().get_model_version_by_alias("failsense-q1", "candidate")
    assert str(version.version) == "1"
    copied = mlflow.sklearn.load_model("models:/failsense-q1@candidate")
    probe = publish.probe_frame(list(frame.columns))
    np.testing.assert_array_equal(copied.predict_proba(probe), pipeline.predict_proba(probe))
    # the same real rows too, not only the probe
    np.testing.assert_array_equal(
        copied.predict_proba(frame.head(50)), pipeline.predict_proba(frame.head(50))
    )


def test_publishing_twice_creates_version_2_and_moves_the_alias(
    tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    pipeline, frame, _ = tiny_pipeline
    source, target = registry(tmp_path, "source"), registry(tmp_path, "target")
    register_model(source, tmp_path, pipeline, frame)
    columns = list(frame.columns)
    publish.copy_registered_model(source, target, "failsense-q1", "candidate", columns)
    second = publish.copy_registered_model(source, target, "failsense-q1", "candidate", columns)
    assert second["version"] == "2"
    mlflow.set_tracking_uri(target)
    assert (
        str(MlflowClient().get_model_version_by_alias("failsense-q1", "candidate").version) == "2"
    )


def test_a_missing_source_model_is_an_error(tmp_path: Path) -> None:
    source, target = registry(tmp_path, "source"), registry(tmp_path, "target")
    with pytest.raises(Exception, match="(?i)not found|no such|does not exist|RESOURCE"):
        publish.copy_registered_model(source, target, "failsense-q1", "candidate", ["a"])


# --- the environment switch --------------------------------------------------------------------
def test_tracking_uri_override_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    assert tracking_uri_override() is None
    monkeypatch.setenv(TRACKING_URI_ENV, "")
    assert tracking_uri_override() is None
    monkeypatch.setenv(TRACKING_URI_ENV, "http://mlflow:5000")
    assert tracking_uri_override() == "http://mlflow:5000"


def test_setup_mlflow_uses_the_server_uri_when_the_override_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the override set the local database is not touched (here: a second sqlite URI)."""
    uri = registry(tmp_path, "server")
    monkeypatch.setenv(TRACKING_URI_ENV, uri)
    config = {"mlflow": {"db_path": str(tmp_path / "local.db"), "artifact_dir": str(tmp_path / "a"),
                         "experiment": "exp"}}  # fmt: skip
    experiment_id = setup_mlflow(config)
    assert mlflow.get_tracking_uri() == uri and experiment_id
    assert not (tmp_path / "local.db").exists()
