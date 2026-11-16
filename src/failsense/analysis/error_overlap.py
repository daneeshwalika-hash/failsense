"""Which failures and false alarms do logreg, XGBoost and the MLP share? (validation only)

Run it with ``uv run python -m failsense.analysis.error_overlap`` (``make error-overlap``).

Why: the model comparison shows PR-AUC and cost, but not *which trucks* each model gets wrong.
If two models miss the same failures, an ensemble cannot help; if they miss different ones,
the models carry different information. The analysis is a plain set comparison, like comparing
which customer rows two SQL queries return (INTERSECT / EXCEPT).

What it does (the test split is never read):

* trains the logistic regression and the MLP on the TRAINING split (k = 5 cut-offs per truck, the
  same 239 input columns as the registered model) and loads the registered XGBoost model,
* scores the official validation (last readout per truck),
* chooses for every model its own cost-optimal threshold on the validation (``choose_threshold``,
  primary cost mapping). This threshold is chosen *in sample* on the same trucks that are
  analysed, so the numbers describe the models at their best operating point, not an honest
  out-of-sample cost,
* writes the overlap of the missed failures (FN) and of the false alarms (FP), and for every
  failure class 1-4 how many failures are found by all models, only by one, or by none.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, CLASS_COL, VEHICLE_COL
from failsense.data.sampling import sampling_params_from_config
from failsense.models.baseline_cv import load_training_frame
from failsense.models.compare_models_cv import make_mlp
from failsense.models.cost import alert_class_from_config, binary_costs, cost_matrix_from_config
from failsense.models.pipeline import (
    build_logreg_pipeline,
    build_model_pipeline,
    split_feature_columns,
)
from failsense.models.threshold import apply_threshold, choose_threshold
from failsense.models.validation_data import (
    build_validation_frame,
    registry_input_columns,
    select_model_inputs,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = ("logreg", "xgboost", "mlp")
OVERLAP_CSV = "error_overlap_q1_pairs.csv"
FOUND_CSV = "error_overlap_q1_found_by.csv"
SUMMARY_JSON = "error_overlap_q1.json"


def _check_alerts(classes: np.ndarray, alerts: Mapping[str, np.ndarray]) -> None:
    """All alert vectors must be boolean-like and as long as ``classes``."""
    if not alerts:
        raise ValueError("alerts is empty")
    for name, flags in alerts.items():
        if np.asarray(flags).shape != classes.shape:
            raise ValueError(f"alerts of '{name}' do not have the length of the classes")


def error_sets(classes: Sequence[int] | np.ndarray, alert: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices of the missed failures (``fn``) and of the false alarms (``fp``) of one model."""
    y = np.asarray(classes).astype(int)
    flags = np.asarray(alert, dtype=bool)
    if y.shape != flags.shape:
        raise ValueError("classes and alert must have the same length")
    return {"fn": np.flatnonzero((y >= 1) & ~flags), "fp": np.flatnonzero((y == 0) & flags)}


def pair_overlap(
    classes: Sequence[int] | np.ndarray, alerts: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Size of each error set per model, and the overlap of every pair and of all models.

    Returns one row per ``(error, group)`` with ``error`` in ``fn`` / ``fp`` and ``group`` a
    model name, ``a & b`` (both make the error) or ``all`` (every model makes it). ``jaccard`` =
    intersection / union (1 = same trucks, 0 = disjoint).
    """
    y = np.asarray(classes).astype(int)
    _check_alerts(y, alerts)
    sets = {name: error_sets(y, flags) for name, flags in alerts.items()}
    rows: list[dict[str, Any]] = []
    for error in ("fn", "fp"):
        members = {name: set(s[error].tolist()) for name, s in sets.items()}
        for name, m in members.items():
            rows.append({"error": error, "group": name, "n": len(m), "jaccard": np.nan})
        for a, b in itertools.combinations(members, 2):
            inter, union = members[a] & members[b], members[a] | members[b]
            rows.append({"error": error, "group": f"{a} & {b}", "n": len(inter),
                         "jaccard": len(inter) / len(union) if union else np.nan})  # fmt: skip
        if len(members) > 2:
            inter = set.intersection(*members.values())
            union = set.union(*members.values())
            rows.append({"error": error, "group": "all", "n": len(inter),
                         "jaccard": len(inter) / len(union) if union else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def found_by_table(
    classes: Sequence[int] | np.ndarray, alerts: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """For every failure class 1-4 (and the total): how many failures are found by which models.

    A failure is *found* by a model when the model alerts that truck. Columns: ``n`` failures,
    ``all`` (found by every model), ``only_<model>`` (found by exactly that model),
    ``other_combination`` (found by 2..n-1 models, not covered above) and ``none`` (missed by all).
    """
    y = np.asarray(classes).astype(int)
    _check_alerts(y, alerts)
    names = list(alerts)
    found = np.column_stack([np.asarray(alerts[n], dtype=bool) for n in names])
    n_found = found.sum(axis=1)
    rows = []
    for label, mask in [(str(k), y == k) for k in range(1, 5)] + [("total", y >= 1)]:
        row: dict[str, Any] = {"class": label, "n": int(mask.sum()),
                               "all": int((mask & (n_found == len(names))).sum())}  # fmt: skip
        for i, name in enumerate(names):
            row[f"only_{name}"] = int((mask & found[:, i] & (n_found == 1)).sum())
        row["none"] = int((mask & (n_found == 0)).sum())
        covered = row["all"] + row["none"] + sum(row[f"only_{n}"] for n in names)
        row["other_combination"] = row["n"] - covered
        rows.append(row)
    return pd.DataFrame(rows)


def cost_optimal_alerts(
    classes: np.ndarray, scores: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Alert vector and threshold of every model at its own cost-optimal threshold (in sample)."""
    costs = binary_costs(
        cost_matrix_from_config(config), alert_class_from_config(config), "primary"
    )
    alerts: dict[str, np.ndarray] = {}
    info: dict[str, dict[str, float]] = {}
    for name, p in scores.items():
        choice = choose_threshold(classes, p, costs)
        alerts[name] = apply_threshold(p, choice.threshold)
        info[name] = {
            "threshold": choice.threshold,
            "cost": choice.cost,
            "n_alerts": choice.n_alerts,
        }
    return alerts, info


def score_models(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Validation frame and scores of logreg, registered XGBoost and MLP (training data only)."""
    seed = int(config["seed"])
    input_columns = registry_input_columns(config)
    train = load_training_frame(config, sampling_params_from_config(config)[0])
    x_train = select_model_inputs(train, input_columns)
    y_train = train[BINARY_COL].to_numpy(dtype=int)
    cols = split_feature_columns(x_train)

    frame = build_validation_frame(config)
    x_val = select_model_inputs(frame, input_columns)
    scores: dict[str, np.ndarray] = {}

    logreg = build_logreg_pipeline(
        seed, categorical_columns=cols.categorical, log1p_columns=cols.log1p
    )
    scores["logreg"] = logreg.fit(x_train, y_train).predict_proba(x_val)[:, 1]
    mlp = build_model_pipeline(make_mlp(seed), cols.categorical, "standard", cols.log1p)
    scores["mlp"] = mlp.fit(x_train, y_train).predict_proba(x_val)[:, 1]

    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / "registry_q1.json").read_text(encoding="utf-8")
    )
    scores["xgboost"] = mlflow.sklearn.load_model(registry["model_uri"]).predict_proba(x_val)[:, 1]
    return frame, {name: scores[name] for name in MODEL_NAMES}


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Score the validation with the three models, write the two CSV tables and a summary JSON."""
    frame, scores = score_models(config)
    classes = frame[CLASS_COL].to_numpy(dtype=int)
    alerts, info = cost_optimal_alerts(classes, scores, config)
    pairs, found = pair_overlap(classes, alerts), found_by_table(classes, alerts)

    reports = resolve_path(config["paths"]["reports"])
    pairs.to_csv(reports / OVERLAP_CSV, index=False)
    found.to_csv(reports / FOUND_CSV, index=False)
    summary = {"thresholds": info, "n_trucks": len(frame), "pairs": pairs.to_dict("records"),
               "found_by": found.to_dict("records"), "vehicles_sorted": bool(
                   frame[VEHICLE_COL].is_monotonic_increasing)}  # fmt: skip
    Path(reports / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    logger.info("Thresholds and costs: %s", info)
    logger.info("Pair overlap:\n%s", pairs.to_string(index=False))
    logger.info("Found by:\n%s", found.to_string(index=False))
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
