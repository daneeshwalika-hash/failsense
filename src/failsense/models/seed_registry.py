"""Seed an EMPTY MLflow registry with the bundled model, so a clean ``docker compose up`` works.

A new machine has no ``mlflow.db`` and no ``mlartifacts`` (both are git-ignored), so the compose
MLflow server starts with an empty registry and the API would stop at its fail-fast check
(``models:/failsense-q1@candidate`` not found). The seed step runs once before the API:

* the registry already has the alias -> nothing is changed (idempotent; a restart never creates a
  second version);
* otherwise the bundled model ``models/failsense-q1/`` (a 1.2 MB copy of the registered version 1:
  ``model.skops`` + the types skops may trust + ``source.json`` with its SHA-256) is checked against
  its recorded hash and **logged through the server** (not copied by path: MLflow stores absolute
  ``file:///`` artifact paths that do not exist in a container) and gets the alias.

The fail-fast check stays: the seeded version must equal ``deployment_choice.model_version`` of the
calibration summary. A fresh registry numbers its first model 1, which is the version the committed
calibrator and threshold belong to; any other outcome stops with a message instead of serving a
model with a wrong calibrator.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from failsense.api.service import ModelConfigError, load_deployment
from failsense.config import load_config, resolve_path
from failsense.freeze import load_model, sha256_file
from failsense.models.publish import register_pipeline
from failsense.tracking import tracking_params_from_config, tracking_uri_override

logger = logging.getLogger(__name__)

BUNDLE_DIR = "models/failsense-q1"


def tracking_uri(config: Mapping[str, Any]) -> str:
    """The MLflow server named by ``FAILSENSE_MLFLOW_URI``, else the local database."""
    db_path, _, _ = tracking_params_from_config(config)
    return tracking_uri_override() or f"sqlite:///{resolve_path(db_path)}"


def alias_version(uri: str, name: str, alias: str) -> str | None:
    """The version the alias points to in the registry at ``uri``, or ``None`` if there is none."""
    mlflow.set_tracking_uri(uri)
    try:
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def load_bundle(folder: Path) -> tuple[Any, dict[str, Any]]:
    """The bundled pipeline and its ``source.json``; the file hash must match the recorded one.

    Raises:
        ModelConfigError: If the bundle is missing or its hash differs from ``source.json``.
    """
    model_file, source_file = folder / "model.skops", folder / "source.json"
    if not model_file.is_file() or not source_file.is_file():
        raise ModelConfigError(f"the bundled model is missing in {folder}")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    if sha256_file(model_file) != source["model_sha256"]:
        raise ModelConfigError(f"{model_file} does not match the hash in {source_file}")
    trusted = json.loads((folder / "trusted_types.json").read_text(encoding="utf-8"))
    return load_model(model_file, trusted), source


def seed(config: Mapping[str, Any]) -> dict[str, Any]:
    """Register the bundled model if the registry has no ``name@alias`` yet.

    Returns:
        ``{"seeded": bool, "version": str}``.

    Raises:
        ModelConfigError: If the (new or existing) version is not the one the calibration summary
            names (the API would refuse it anyway; stopping here gives the reason early).
    """
    tuning = config["tuning"]
    name, alias = tuning["registered_model"], tuning["alias"]
    uri = tracking_uri(config)
    wanted = str(load_deployment(config).get("model_version"))

    existing = alias_version(uri, name, alias)
    if existing is not None:
        logger.info("Registry %s already has %s@%s (version %s): nothing to seed", uri, name, alias,
                    existing)  # fmt: skip
        if wanted != "None" and existing != wanted:
            raise ModelConfigError(
                f"the registry has version {existing}, the calibration summary belongs to {wanted}"
            )
        return {"seeded": False, "version": existing}

    pipeline, source = load_bundle(resolve_path(BUNDLE_DIR))
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / "registry_q1.json").read_text(encoding="utf-8")
    )
    result = register_pipeline(
        pipeline, uri, name, alias, registry["input_columns"],
        f"bundled {BUNDLE_DIR} (registry version {source['registry_version']}, "
        f"sha256 {source['model_sha256'][:12]})",
    )  # fmt: skip
    if wanted != "None" and result["version"] != wanted:
        raise ModelConfigError(
            f"seeded version {result['version']} but the calibration summary belongs to version "
            f"{wanted}: the registry was not empty or numbered differently"
        )
    return {"seeded": True, "version": result["version"]}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    seed(load_config())


if __name__ == "__main__":
    main()
