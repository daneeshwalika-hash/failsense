"""Q4: anomaly detection on healthy trucks, and the anomaly score as an extra Q1 feature (ADR 0005).

The idea: in practice most components have no or very few failure examples. A detector learns
"normal" from trucks that were never repaired and flags readouts that look different. The repair
labels are used **only to evaluate** the scores, never to fit a detector.

* ``IsolationForest``: random trees isolate a point; anomalies need fewer splits (short paths).
  Cheap, no distance computation, copes with many columns.
* ``LocalOutlierFactor`` (novelty mode): compares the local density around a point with the density
  around its neighbours; distance based, so it gets the PCA-reduced inputs (distances in 239 columns
  lose meaning).

SQL analogy: a detector is a "what does a normal row look like" profile built from clean rows; the
score is how far a new row is from that profile.

Leakage rules
-------------
* Fit on healthy TRAINING trucks only; validation and test rows are only scored.
* When the score is used as a feature, the training rows are scored by a detector that never saw
  their truck (cross-fitting by truck, StratifiedGroupKFold on the healthy flag): in-sample scores
  of the training trucks would be systematically lower (more "normal") than the out-of-sample
  scores of new trucks, and the supervised model would learn from that mismatch.
* The healthy flag (``in_study_repair == 0``) is a label-derived fact of the TRAINING trucks only.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, CLASS_COL, REPAIR_COL, VEHICLE_COL
from failsense.data.schema import feature_columns
from failsense.data.split import assert_no_group_overlap
from failsense.models.baseline_cv import read_table, table_path
from failsense.models.boosters import make_booster
from failsense.models.evaluate import top_k_metrics
from failsense.models.pipeline import (
    build_model_pipeline,
    build_preprocessor,
    select_categorical_columns,
)
from failsense.models.q2_classes import load_training_frame_with_classes
from failsense.models.validation_data import (
    build_validation_frame,
    registry_input_columns,
    select_model_inputs,
)
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

DETECTORS = ("iforest_full", "iforest_pca", "lof_pca")
FEATURE_DETECTOR = "iforest_full"  # fixed in advance (ADR 0005), not chosen by the validation
FEATURE_NAME = "anomaly_score"
SUMMARY_FILE = "anomaly_q4_summary.json"


# --- detectors --------------------------------------------------------------------------------
def make_detector(
    kind: str,
    categorical: Sequence[str],
    log1p_columns: Sequence[str],
    seed: int,
    n_estimators: int = 300,
    max_samples: int = 256,
    n_components: int = 50,
    n_neighbors: int = 35,
) -> Pipeline:
    """Preprocessing (+ PCA) + detector as one Pipeline; everything is fitted on healthy rows only.

    Raises:
        ValueError: For an unknown ``kind``.
    """
    if kind not in DETECTORS:
        raise ValueError(f"kind must be one of {DETECTORS}, got {kind!r}")
    steps: list[tuple[str, Any]] = [
        ("preprocess", build_preprocessor(categorical, "standard", log1p_columns))
    ]
    if kind.endswith("_pca"):
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    if kind.startswith("iforest"):
        detector = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed,
            n_jobs=os.cpu_count() or 1,
        )  # fmt: skip
    else:
        detector = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, n_jobs=os.cpu_count() or 1
        )
    return Pipeline([*steps, ("detector", detector)])


def anomaly_score(detector: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Anomaly score per row; HIGHER = more anomalous (``score_samples`` is the opposite)."""
    return -np.asarray(detector.score_samples(x), dtype=float)


def healthy_mask(vehicle_ids: pd.Series, repaired_ids: Sequence[int]) -> np.ndarray:
    """True for rows of trucks that were never repaired during the study."""
    return ~vehicle_ids.isin(set(repaired_ids)).to_numpy()


def cross_fit_anomaly(
    make: Callable[[], Pipeline],
    x: pd.DataFrame,
    healthy: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Anomaly score of every row from a detector that never saw its truck.

    Per fold: fit on the healthy rows of the other trucks, score the rows of the held-out trucks
    (healthy or not). ``StratifiedGroupKFold`` keeps a truck in one fold and the healthy share
    similar in all folds.

    Raises:
        ValueError: On inconsistent lengths or if a fold has no healthy training row.
    """
    if not len(x) == len(healthy) == len(groups):
        raise ValueError("x, healthy and groups must have the same length")
    scores = np.full(len(x), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, healthy.astype(int), groups):
        assert_no_group_overlap({"fit": groups[train], "score": groups[test]})
        fit_rows = train[healthy[train]]
        if len(fit_rows) == 0:
            raise ValueError("a fold has no healthy training row to fit the detector")
        detector = make().fit(x.iloc[fit_rows])
        scores[test] = anomaly_score(detector, x.iloc[test])
    return scores


# --- evaluation -------------------------------------------------------------------------------
def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if y.sum() > 0 else float("nan")


def evaluate_scores(
    y: Sequence[int], score: Sequence[float], fraction: float, n_boot: int, seed: int
) -> dict[str, float]:
    """PR-AUC (with a 95 % bootstrap interval over trucks), ROC-AUC, lift and top-k recall.

    ``lift`` = PR-AUC divided by the prevalence (a random ranking has a lift of about 1).

    Raises:
        ValueError: If there is no positive or no negative truck, or lengths differ.
    """
    labels, s = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if labels.shape != s.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("y and score must be non-empty 1-D arrays of equal length")
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("need both positive and negative trucks")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        draws.append(_average_precision(labels[idx], s[idx]))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    prevalence = float(labels.mean())
    pr_auc = _average_precision(labels, s)
    top = top_k_metrics(labels, s, fraction)
    return {"pr_auc": pr_auc, "pr_auc_ci_low": float(low), "pr_auc_ci_high": float(high),
            "roc_auc": float(roc_auc_score(labels, s)), "prevalence": prevalence,
            "lift": pr_auc / prevalence, f"recall_top_{fraction:.0%}": top["recall"],
            f"precision_top_{fraction:.0%}": top["precision"]}  # fmt: skip


def paired_bootstrap_pr_auc(
    y: Sequence[int], score_a: Sequence[float], score_b: Sequence[float], n_boot: int, seed: int
) -> dict[str, float]:
    """PR-AUC(a) - PR-AUC(b) with its spread when the trucks are resampled (same draws for both)."""
    labels = np.asarray(y, dtype=int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    if not labels.shape == a.shape == b.shape or labels.size == 0:
        raise ValueError("y, score_a and score_b must be non-empty and of equal length")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].sum() == 0:
            continue
        diffs.append(
            _average_precision(labels[idx], a[idx]) - _average_precision(labels[idx], b[idx])
        )
    arr = np.asarray(diffs)
    return {
        "diff": _average_precision(labels, a) - _average_precision(labels, b),
        "std": float(arr.std(ddof=1)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "share_a_better": float((arr > 0).mean()),
    }


def adoption_decision(paired: Mapping[str, float]) -> bool:
    """ADR 0005: adopt only if the difference is positive and larger than the bootstrap std."""
    return bool(paired["diff"] > 0 and paired["diff"] > paired["std"])


# --- run --------------------------------------------------------------------------------------
def _fit_supervised(
    x: pd.DataFrame, y: np.ndarray, categorical: Sequence[str], registry: Mapping[str, Any],
    seed: int,
) -> Pipeline:  # fmt: skip
    booster = make_booster(
        "xgboost", registry["params"], seed, os.cpu_count() or 1, int(registry["n_rounds"])
    )
    return build_model_pipeline(booster, categorical, scaler=None).fit(x, y)


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Q4 evaluation of the three detectors and the ADR 0005 test of the anomaly feature."""
    section = config["anomaly"]
    seed, n_boot = int(config["seed"]), int(section["n_bootstrap"])
    fraction = float(section["top_fraction"])
    reports = resolve_path(config["paths"]["reports"])
    registry = json.loads((reports / "registry_q1.json").read_text(encoding="utf-8"))
    columns = registry_input_columns(config)

    frame = load_training_frame_with_classes(config, 5)
    tte = read_table(table_path(config, "train", "tte"))
    repaired_ids = tte.loc[tte[REPAIR_COL] == 1, VEHICLE_COL]
    healthy = healthy_mask(frame[VEHICLE_COL], repaired_ids)
    x, y = select_model_inputs(frame, columns), frame[BINARY_COL].to_numpy(dtype=int)
    groups = frame[VEHICLE_COL].to_numpy()
    categorical = select_categorical_columns(x)
    log1p_cols = [c for c in feature_columns() if c in x.columns]
    assert (frame.loc[healthy, CLASS_COL] == 0).all(), "healthy trucks must only have class 0 rows"
    logger.info("training rows %d, healthy rows %d (%d healthy trucks)", len(x), healthy.sum(),
                frame.loc[healthy, VEHICLE_COL].nunique())  # fmt: skip

    val = build_validation_frame(config)
    x_val, y_val = select_model_inputs(val, columns), val[BINARY_COL].to_numpy(dtype=int)

    def make(kind: str) -> Pipeline:
        return make_detector(
            kind, categorical, log1p_cols, seed, int(section["n_estimators"]),
            int(section["max_samples"]), int(section["pca_components"]),
            int(section["lof_neighbors"]),
        )  # fmt: skip

    # --- 1. the three detectors on the official validation ---------------------------------
    healthy_x = x[healthy]
    detectors, val_scores, rows = {}, {}, []
    for kind in DETECTORS:
        fit_x = healthy_x
        if kind.startswith("lof") and len(fit_x) > int(section["lof_max_rows"]):
            fit_x = fit_x.sample(n=int(section["lof_max_rows"]), random_state=seed)
        detectors[kind] = make(kind).fit(fit_x)
        val_scores[kind] = anomaly_score(detectors[kind], x_val)
        metrics = evaluate_scores(y_val, val_scores[kind], fraction, n_boot, seed)
        rows.append({"detector": kind, "n_fit_rows": len(fit_x), **metrics})
        logger.info("%s: %s", kind, {k: round(v, 4) for k, v in metrics.items()})
    table = pd.DataFrame(rows)

    # --- 2. the anomaly score as an extra feature (ADR 0005) --------------------------------
    oof = cross_fit_anomaly(
        lambda: make(FEATURE_DETECTOR), x, healthy, groups, int(section["cross_fit_splits"]), seed
    )
    x_plus = x.assign(**{FEATURE_NAME: oof})
    x_val_plus = x_val.assign(**{FEATURE_NAME: val_scores[FEATURE_DETECTOR]})
    proba = {"without": [], "with": []}
    for i in range(int(section["n_seeds"])):
        proba["without"].append(
            _fit_supervised(x, y, categorical, registry, seed + i).predict_proba(x_val)[:, 1]
        )
        proba["with"].append(
            _fit_supervised(x_plus, y, categorical, registry, seed + i).predict_proba(x_val_plus)[
                :, 1
            ]
        )
    mean = {k: np.mean(v, axis=0) for k, v in proba.items()}
    per_seed = {k: [_average_precision(y_val, p) for p in v] for k, v in proba.items()}
    supervised = {k: evaluate_scores(y_val, mean[k], fraction, n_boot, seed) for k in mean}
    paired = paired_bootstrap_pr_auc(y_val, mean["with"], mean["without"], n_boot, seed)
    adopt = adoption_decision(paired)
    oof_pr = _average_precision(y, oof)  # descriptive: OOF anomaly score vs. the training label
    logger.info("feature test: %s -> adopt=%s", paired, adopt)

    summary = {
        "n_train_rows": len(x), "n_healthy_rows": int(healthy.sum()),
        "n_validation_trucks": len(y_val), "n_validation_positives": int(y_val.sum()),
        "detectors": table.to_dict("records"),
        "supervised_reference_pr_auc_note": "Q1 candidate on the validation: PR-AUC 0.065",
        "feature_test": {
            "detector": FEATURE_DETECTOR, "n_seeds": int(section["n_seeds"]),
            "pr_auc_per_seed": per_seed, "averaged": supervised, "paired_bootstrap": paired,
            "adopt": adopt, "rule": "ADR 0005: diff > 0 and diff > paired bootstrap std",
            "oof_anomaly_score_pr_auc_vs_training_label": oof_pr,
        },
        "caveat": "136 positive validation trucks: all intervals are wide; test split untouched",
    }  # fmt: skip
    reports.mkdir(parents=True, exist_ok=True)
    (reports / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table.to_csv(reports / "anomaly_q4_detectors.csv", index=False, float_format="%.5f")
    _log_to_mlflow(config, summary)
    return summary


def _log_to_mlflow(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    import mlflow

    setup_mlflow(config, experiment=config["anomaly"]["mlflow_experiment"])
    with mlflow.start_run(run_name="q4_detectors_and_feature_test"):
        mlflow.set_tags({"task": "q4_anomaly", "split": "official validation"})
        for row in summary["detectors"]:
            mlflow.log_metric(f"pr_auc_{row['detector']}", row["pr_auc"])
            mlflow.log_metric(f"roc_auc_{row['detector']}", row["roc_auc"])
        test = summary["feature_test"]
        for name, metrics in test["averaged"].items():
            mlflow.log_metric(f"supervised_pr_auc_{name}_anomaly_feature", metrics["pr_auc"])
        mlflow.log_metric("feature_diff_pr_auc", test["paired_bootstrap"]["diff"])
        mlflow.log_metric("feature_diff_std", test["paired_bootstrap"]["std"])
        mlflow.log_param("feature_adopted", test["adopt"])
        mlflow.log_artifact(str(resolve_path(config["paths"]["reports"]) / SUMMARY_FILE))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
