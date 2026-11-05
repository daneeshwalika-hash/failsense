"""Fit the winner on the full training split and register it as ``failsense-q1``.

Run it with::

    uv run python -m failsense.models.register      # or: make register   (after make evaluate)

The winner comes from ``evaluate`` (``reports/evaluate_q1_winner.json``): a tuned model if it
is clearly better than the untuned engineered XGBoost on the calibrated set, otherwise the
untuned one. In both cases a model is registered with the alias ``candidate``, because the
FastAPI service loads ``models:/failsense-q1@candidate`` on 07.11.

What is fitted: one scikit-learn Pipeline (median imputer + one-hot specs, then the booster
with a **fixed number of rounds**: the median of the rounds that early stopping chose per fold
while tuning, 300 for the untuned model) on **all** ``k = 5`` training rows. **No validation or
test data** is used. The model is saved with MLflow's ``skops`` format; the pipeline's own types
and the XGBoost / LightGBM types are listed as trusted (skops refuses unknown types). After
saving, the registered model is loaded again and must predict **exactly** like the fitted one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL
from failsense.models.baseline_cv import load_training_frame
from failsense.models.boosters import make_booster
from failsense.models.evaluate import WINNER_FILE
from failsense.models.pipeline import build_model_pipeline, split_feature_columns
from failsense.models.train import SKOPS_TRUSTED_TYPES
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# skops trusts only the types it is told about: the pipeline's own pieces (train.py) plus the
# classifiers and the scikit-learn / numpy objects inside them.
TRUSTED_TYPES = [
    *SKOPS_TRUSTED_TYPES,
    "xgboost.sklearn.XGBClassifier",
    "xgboost.core.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
    "collections.OrderedDict",
    "numpy.float32",
    "numpy.int64",
]
REGISTRY_FILE = "registry_q1.json"


def load_winner(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the winner chosen by ``evaluate``.

    Raises:
        FileNotFoundError: If ``evaluate`` has not been run.
    """
    path = resolve_path(config["paths"]["reports"]) / WINNER_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make evaluate` first")
    return json.loads(path.read_text())


def register_winner(
    config: Mapping[str, Any], winner: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fit, log, register, set the alias and verify the round trip.

    ``setup_mlflow`` must have been called. Returns the registry facts (also written to
    ``reports/registry_q1.json``).

    Raises:
        AssertionError: If the reloaded model does not predict exactly like the fitted one.
    """
    tuning = config["tuning"]
    winner = dict(winner) if winner is not None else load_winner(config)
    seed, n_jobs = int(tuning["seed"]), os.cpu_count() or 1
    name, alias = tuning["registered_model"], tuning["alias"]

    frame = load_training_frame(config, 5)  # the k5 training split; no validation / test data
    cols = split_feature_columns(frame)
    columns = cols.raw + (cols.engineered if winner["engineered"] else []) + cols.categorical
    x, y = frame[columns], frame[BINARY_COL].to_numpy(dtype=int)
    booster = make_booster(
        winner["family"], winner["params"], seed, n_jobs, int(winner["n_rounds"])
    )
    pipeline = build_model_pipeline(booster, cols.categorical, scaler=None).fit(x, y)
    logger.info(
        "Fitted %s (%s, %d rounds, %d input columns) on %d training rows",
        winner["winner"], winner["family"], winner["n_rounds"], len(columns), len(x),
    )  # fmt: skip

    with mlflow.start_run(run_name=f"register_{winner['winner']}") as active:
        mlflow.set_tags({"task": "q1_binary", "stage": "registry", "family": winner["family"]})
        mlflow.log_params(
            {
                "winner": winner["winner"], "family": winner["family"],
                "engineered": winner["engineered"], "n_rounds": winner["n_rounds"],
                "n_train_rows": len(x), "n_input_columns": len(columns),
                "training_sample": "k5 (no validation / test data)",
                **{f"param_{k}": v for k, v in winner["params"].items()},
            }
        )  # fmt: skip
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=x.head(5),
            skops_trusted_types=TRUSTED_TYPES, registered_model_name=name,
        )  # fmt: skip
        version = str(info.registered_model_version)
        run_id = active.info.run_id
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, version)

    uri = f"models:/{name}@{alias}"
    reloaded = mlflow.sklearn.load_model(uri)
    sample = x.iloc[:: max(1, len(x) // 5000)]
    np.testing.assert_array_equal(
        reloaded.predict_proba(sample), pipeline.predict_proba(sample),
        err_msg="the reloaded model predicts differently from the fitted one",
    )  # fmt: skip
    logger.info(
        "Registered %s version %s, alias %r; reload from %s predicts identically on %d rows",
        name, version, alias, uri, len(sample),
    )  # fmt: skip

    facts = {
        "name": name, "version": version, "alias": alias, "run_id": run_id,
        "winner": winner["winner"], "reason": winner.get("reason"), "family": winner["family"],
        "engineered": winner["engineered"], "n_rounds": winner["n_rounds"],
        "params": winner["params"], "n_train_rows": len(x), "input_columns": columns,
        "model_uri": uri,
    }  # fmt: skip
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / REGISTRY_FILE).write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    return facts


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    setup_mlflow(config)
    register_winner(config)


if __name__ == "__main__":
    main()
