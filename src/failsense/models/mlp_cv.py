"""MLPClassifier in the Q1 comparison: same folds, paired against logreg and XGBoost.

Run: ``uv run python -m failsense.models.mlp_cv`` (``make mlp``, about 10 min). The full comparison
table (``make compare``) is NOT touched: this module writes ``reports/compare_mlp_q1.csv`` and logs
to MLflow with the tag ``comparison_set = v3_mlp``.

Models (5-fold StratifiedGroupKFold by truck, ``k5`` training cut-offs, PR-AUC, the folds of every
other comparison):

* ``logreg_reference`` and ``xgboost`` (raw features) and ``xgboost__engineered``: the references,
  refitted here so the paired differences are computed fold by fold on identical folds.
* ``mlp``: log1p of the counters / histograms, standard scaling, one-hot specs, then
  ``MLPClassifier`` (64-32, early stopping), raw features.
* ``mlp_engineered``: the same network with the engineered counter and histogram features added.

Everything learned (imputer, scaler, encoder, network) sits inside the Pipeline and is fitted per
fold. The test split is not touched.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, VEHICLE_COL
from failsense.data.sampling import sampling_params_from_config
from failsense.data.split import cv_params_from_config, make_group_folds
from failsense.models.baseline_cv import (
    SCORING_WITH_PROBA,
    load_training_frame,
    n_cutoffs_label,
    paired_comparison,
)
from failsense.models.compare_models_cv import (
    REFERENCE,
    ModelSpec,
    _worker_split,
    build_models,
    build_rechecks,
    make_mlp,
)
from failsense.models.pipeline import build_model_pipeline, split_feature_columns
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

COMPARISON_SET = "v3_mlp"
REPORT_FILE = "compare_mlp_q1.csv"
RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "comparison"}
PAIRS = (
    ("mlp", REFERENCE),
    ("mlp", "xgboost"),
    ("mlp_engineered", "mlp"),
    ("mlp_engineered", "xgboost__engineered"),
)


def build_specs(cols: Any, random_state: int, model_jobs: int) -> dict[str, ModelSpec]:
    """The five models of this comparison, references first."""
    models = build_models(cols, random_state, model_jobs)
    engineered = build_rechecks("xgboost", cols, random_state, model_jobs, 1.0)[
        "xgboost__engineered"
    ]
    mlp_engineered = ModelSpec(
        "mlp_engineered",
        "neural_network",
        build_model_pipeline(
            make_mlp(random_state), cols.categorical, scaler="standard", log1p_columns=cols.log1p
        ),
        {**models["mlp"].params, "features": "raw + engineered"},
        columns=tuple(cols.raw + list(cols.engineered) + list(cols.categorical)),
    )
    chosen = [models[REFERENCE], models["xgboost"], engineered, models["mlp"], mlp_engineered]
    return {spec.name: spec for spec in chosen}


def run(config: Mapping[str, Any], track: bool = False) -> dict[str, dict[str, Any]]:
    """Cross-validate the five models and write ``compare_mlp_q1.csv``."""
    n_per_vehicle, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    cols = split_feature_columns(frame)
    y = frame[BINARY_COL].to_numpy()
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    folds_parallel, model_jobs = _worker_split(n_splits)
    specs = build_specs(cols, cv_seed, model_jobs)
    default_columns = cols.raw + cols.categorical
    common = {
        "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle), "sampling_seed": sampling_seed,
        "cv_n_splits": n_splits, "cv_seed": cv_seed, "n_rows": len(frame),
        "n_vehicles": int(frame[VEHICLE_COL].nunique()), "comparison_set": COMPARISON_SET,
    }  # fmt: skip

    summary: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, np.ndarray] = {}
    for spec in specs.values():
        columns = list(spec.columns) if spec.columns is not None else default_columns
        started = time.perf_counter()
        result = cross_validate(
            spec.pipeline, frame[columns], y, cv=folds, scoring=SCORING_WITH_PROBA,
            n_jobs=folds_parallel, return_train_score=True,
        )  # fmt: skip
        val, train = result["test_pr_auc"], float(result["train_pr_auc"].mean())
        fold_scores[spec.name] = val
        summary[spec.name] = {
            "family": spec.family, "pr_auc": float(val.mean()), "pr_auc_std": float(val.std()),
            "train": train, "gap": train - float(val.mean()),
            "mean_proba": float(result["test_mean_proba"].mean()),
            "seconds": time.perf_counter() - started, "n_columns": len(columns),
        }  # fmt: skip
        logger.info(
            "%-22s val %.4f +- %.4f | train %.4f | mean proba %.4f | %.0f s",
            spec.name, val.mean(), val.std(), train, result["test_mean_proba"].mean(),
            summary[spec.name]["seconds"],
        )  # fmt: skip
        if track:
            log_cv_run(
                run_name=f"{spec.name}_k{n_cutoffs_label(n_per_vehicle)}", cv_result=result,
                metrics=SCORING_WITH_PROBA,
                params={**common, "model": spec.name, "family": spec.family,
                        "n_columns": len(columns), **spec.params},
                tags={**RUN_TAGS, "family": spec.family, "comparison_set": COMPARISON_SET},
            )  # fmt: skip

    for candidate, reference in PAIRS:
        cmp = paired_comparison(fold_scores[candidate], fold_scores[reference])
        summary[candidate].setdefault("paired", {})[reference] = cmp
        logger.info(
            "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
            candidate, reference, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    rows = []
    for name, s in summary.items():
        row = {"model": name, **{k: v for k, v in s.items() if k != "paired"}}
        for reference, cmp in s.get("paired", {}).items():
            row[f"diff_vs_{reference}"] = cmp["mean_diff"]
            row[f"diff_std_vs_{reference}"] = cmp["std_diff"]
            row[f"folds_won_vs_{reference}"] = cmp["wins"]
        rows.append(row)
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reports / REPORT_FILE, index=False, float_format="%.5f")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(config, track=track)


if __name__ == "__main__":
    main()
