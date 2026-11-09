"""Copy the registered model (with its alias) to another MLflow registry, e.g. the compose server.

``make publish-model`` copies ``models:/<tuning.registered_model>@<tuning.alias>`` from the local
registry (``mlflow.db`` + ``mlartifacts``) to the MLflow server of ``docker compose``. Why a copy
and not "point the container at the local files": MLflow stores artifact paths as ABSOLUTE URIs
(``file:///home/.../mlartifacts``) which do not exist inside a container. Logging the model through
the server stores it in the server's artifact store (proxied over HTTP), so every container loads
it by name and alias, nothing else.

The copy is verified: the model is reloaded from the target by alias and must predict exactly like
the source on a probe frame. The calibration summary (calibrator, threshold) lives in the image
(``reports/calibration_q1_summary.json``) and still names ``models:/failsense-q1@candidate``, which
stays true: the same name and alias exist in the target.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from failsense.config import load_config, resolve_path
from failsense.data.schema import SPEC_COLUMNS
from failsense.models.register import REGISTRY_FILE, TRUSTED_TYPES
from failsense.tracking import tracking_params_from_config

logger = logging.getLogger(__name__)

DEFAULT_TARGET = "http://127.0.0.1:5000"


def probe_frame(input_columns: Sequence[str], n: int = 5, seed: int = 0) -> pd.DataFrame:
    """A small deterministic frame with the model's input columns (numbers + a spec category)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {c: rng.uniform(0, 100, n) for c in input_columns if c not in SPEC_COLUMNS}
    )
    for column in input_columns:
        if column in SPEC_COLUMNS:
            frame[column] = "Cat1"
    return frame[list(input_columns)]


def register_pipeline(
    pipeline: Any,
    target_tracking_uri: str,
    name: str,
    alias: str,
    input_columns: Sequence[str],
    source: str,
    experiment: str = "failsense-published",
) -> dict[str, Any]:
    """Log ``pipeline`` through ``target_tracking_uri``, register it, set the alias, reload-check.

    The model is LOGGED (not copied by path), so the target's artifact store holds it and no
    absolute ``file://`` path leaks into the registry.

    Raises:
        AssertionError: If the model reloaded from the target predicts differently.
    """
    probe = probe_frame(input_columns)
    expected = pipeline.predict_proba(probe)
    mlflow.set_tracking_uri(target_tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"publish_{name}_{alias}"):
        mlflow.set_tag("source", source)
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=probe,
            skops_trusted_types=TRUSTED_TYPES, registered_model_name=name,
        )  # fmt: skip
        version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(name, alias, version)
    reloaded = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    np.testing.assert_array_equal(
        reloaded.predict_proba(probe), expected,
        err_msg="the model reloaded from the target predicts differently from the source",
    )  # fmt: skip
    logger.info("Registered %s@%s as version %s in %s (reload verified)", name, alias, version,
                target_tracking_uri)  # fmt: skip
    return {"name": name, "alias": alias, "version": version, "target": target_tracking_uri}


def copy_registered_model(
    source_tracking_uri: str,
    target_tracking_uri: str,
    name: str,
    alias: str,
    input_columns: Sequence[str],
    experiment: str = "failsense-published",
) -> dict[str, Any]:
    """Load ``models:/name@alias`` from the source registry and register it in the target."""
    mlflow.set_tracking_uri(source_tracking_uri)
    source_uri = f"models:/{name}@{alias}"
    pipeline = mlflow.sklearn.load_model(source_uri)
    return register_pipeline(
        pipeline, target_tracking_uri, name, alias, input_columns,
        f"{source_tracking_uri} {source_uri}", experiment,
    )  # fmt: skip


def run(config: Mapping[str, Any], target: str) -> dict[str, Any]:
    """Publish the configured model from the local registry to ``target``."""
    tuning = config["tuning"]
    db_path, _, _ = tracking_params_from_config(config)
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / REGISTRY_FILE).read_text(encoding="utf-8")
    )
    return copy_registered_model(
        f"sqlite:///{resolve_path(db_path)}", target, tuning["registered_model"], tuning["alias"],
        registry["input_columns"],
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default=DEFAULT_TARGET, help="MLflow server of docker compose")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config(), args.target)


if __name__ == "__main__":
    main()
