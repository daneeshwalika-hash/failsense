"""Seeding an empty registry from the bundled model (what a clean docker compose up relies on)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from failsense import freeze
from failsense.api.service import ModelConfigError, ModelService
from failsense.config import load_config, resolve_path
from failsense.models import seed_registry as sr


@pytest.fixture
def setup(
    tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """A bundle of the tiny pipeline, a config with a temp registry, summary version "1"."""
    monkeypatch.delenv("FAILSENSE_MLFLOW_URI", raising=False)
    pipeline, frame, _ = tiny_pipeline
    bundle = tmp_path / "bundle"
    types = freeze.save_model(pipeline, bundle / "model.skops")
    (bundle / "trusted_types.json").write_text(json.dumps(types))
    sha = hashlib.sha256((bundle / "model.skops").read_bytes()).hexdigest()
    (bundle / "source.json").write_text(json.dumps({"registry_version": "1", "model_sha256": sha}))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "registry_q1.json").write_text(json.dumps({"input_columns": list(frame.columns)}))
    summary = tmp_path / "summary.json"

    def write(version: object) -> None:
        summary.write_text(json.dumps({"deployment_choice": {
            "model_uri": "models:/failsense-q1@candidate", "model_version": version,
            "calibrator": {"kind": "sigmoid", "slope": 0.5, "intercept": -2.0,
                           "logit_clip_eps": 1e-6}, "threshold": 0.05}}))  # fmt: skip

    write("1")
    monkeypatch.setattr(sr, "BUNDLE_DIR", str(bundle))
    config = {
        "api": {"summary_path": str(summary), "summary_key": "deployment_choice"},
        "tuning": {"registered_model": "failsense-q1", "alias": "candidate"},
        "mlflow": {
            "db_path": str(tmp_path / "mlflow.db"),
            "artifact_dir": str(tmp_path / "art"),
            "experiment": "e",
        },  # fmt: skip
        "paths": {"reports": str(reports)},
        "features": {"top_fraction": 0.2},
    }
    return {"config": config, "write": write, "bundle": bundle, "uri": sr.tracking_uri(config)}


def versions(uri: str) -> list[str]:
    mlflow.set_tracking_uri(uri)
    return sorted(
        str(v.version) for v in MlflowClient().search_model_versions("name='failsense-q1'")
    )


def test_an_empty_registry_is_seeded_and_a_second_run_changes_nothing(setup: dict) -> None:
    first = sr.seed(setup["config"])
    assert first == {"seeded": True, "version": "1"}
    assert versions(setup["uri"]) == ["1"]
    second = sr.seed(setup["config"])  # e.g. a restart of the compose stack
    assert second == {"seeded": False, "version": "1"}
    assert versions(setup["uri"]) == ["1"]  # no second version


def test_the_seeded_model_is_served_by_the_api_loader(setup: dict) -> None:
    sr.seed(setup["config"])
    service = ModelService.from_config(setup["config"])  # the fail-fast checks all pass
    assert service.info.version == "1" and service.info.alias == "candidate"


def test_a_summary_for_another_version_stops_the_seed_with_the_reason(setup: dict) -> None:
    setup["write"]("2")
    with pytest.raises(ModelConfigError, match=r"seeded version 1.*version 2"):
        sr.seed(setup["config"])


def test_an_existing_alias_on_another_version_is_refused(setup: dict) -> None:
    sr.seed(setup["config"])
    setup["write"]("3")
    with pytest.raises(ModelConfigError, match="registry has version 1"):
        sr.seed(setup["config"])


def test_a_tampered_bundle_is_refused_before_anything_is_registered(setup: dict) -> None:
    (setup["bundle"] / "model.skops").write_bytes(b"tampered")
    with pytest.raises(ModelConfigError, match="does not match the hash"):
        sr.seed(setup["config"])
    assert versions(setup["uri"]) == []


def test_a_missing_bundle_is_refused(setup: dict) -> None:
    (setup["bundle"] / "source.json").unlink()
    with pytest.raises(ModelConfigError, match="bundled model is missing"):
        sr.seed(setup["config"])


def test_the_seed_uses_the_server_uri_when_the_environment_names_one(
    setup: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAILSENSE_MLFLOW_URI", setup["uri"])  # a sqlite URI stands in for a server
    assert sr.tracking_uri(setup["config"]) == setup["uri"]
    assert sr.seed(setup["config"])["seeded"] is True


# --- the real bundle ---------------------------------------------------------------------------
def test_the_committed_bundle_is_consistent_with_the_registry_and_the_summary() -> None:
    config = load_config()
    folder = resolve_path(sr.BUNDLE_DIR)
    pipeline, source = sr.load_bundle(folder)  # also checks the SHA-256
    registry = json.loads((resolve_path("reports") / "registry_q1.json").read_text())
    summary = json.loads((resolve_path("reports") / "calibration_q1_summary.json").read_text())
    assert (
        source["registry_version"]
        == registry["version"]
        == summary["deployment_choice"]["model_version"]
    )
    assert source["run_id"] == registry["run_id"]
    assert pipeline.named_steps["preprocess"].get_feature_names_out().size > 0
    assert config["tuning"]["alias"] == "candidate"
    frozen = resolve_path("artifacts/frozen/q1_model/model.skops")
    if frozen.is_file():  # the freeze copied the same registered model
        assert freeze.sha256_file(frozen) == source["model_sha256"]


def test_the_real_bundle_seeds_a_fresh_registry_and_the_api_loads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole clean-machine path: empty registry -> seed -> the production loader."""
    monkeypatch.delenv("FAILSENSE_MLFLOW_URI", raising=False)
    config = copy.deepcopy(load_config())
    config["mlflow"] = {"db_path": str(tmp_path / "mlflow.db"), "artifact_dir": str(tmp_path / "a"),
                        "experiment": "e"}  # fmt: skip
    assert sr.seed(config) == {"seeded": True, "version": "1"}
    service = ModelService.from_config(config)  # registered columns, calibrator, smoke test
    assert service.info.version == "1" and service.threshold == pytest.approx(0.0353238553943719)
