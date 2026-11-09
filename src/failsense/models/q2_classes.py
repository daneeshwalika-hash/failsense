"""Q2: how close is the failure (classes 0-4)? Multi-class model + 5 x 5 cost matrix + prior shift.

The SCANIA cost matrix prices every (true class, predicted class) pair. A multi-class model gives
class probabilities ``p(c | x)``; the Bayes decision under a cost matrix is not the arg-max but the
class with the lowest **expected cost**::

    predicted = argmin_k  sum_c  p(c | x) * cost[c, k]

SQL analogy: ``p @ cost`` is a join of the probability row with the price table, summed per
predicted class; the decision is ``ORDER BY expected_cost LIMIT 1``.

Label shift (ADR 0004): the training cut-offs have another class mix than the official validation
(positives 48.6 / 24.2 / 11.6 / 15.6 % vs. 11.8 / 10.3 / 22.1 / 55.9 %). Bayes' rule gives the
correction ``p_target(c | x)  ~  p_train(c | x) * pi_target(c) / pi_train(c)`` (renormalised), valid
if ``p(x | c)`` is the same in both. ``pi_target`` is the class mix published in the dataset paper
(configs ``q2.target_class_counts``), never fitted on per-row validation labels.

Leakage rules: features are the registered model's inputs (no label, no ``vehicle_id``); training
uses ``k5`` cut-offs of the TRAINING trucks; cross-validation is StratifiedGroupKFold by truck; the
validation split is only scored (nothing is tuned on it); the test split is not read.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from xgboost import XGBClassifier

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    CLASS_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.features.build import (
    attach_specifications,
    features_params_from_config,
)
from failsense.models.baseline_cv import frame_from_sampled, read_table, table_path
from failsense.models.cost import N_CLASSES, cost_matrix_from_config
from failsense.models.pipeline import build_model_pipeline, select_categorical_columns
from failsense.models.validation_data import (
    build_validation_frame,
    registry_input_columns,
    select_model_inputs,
)
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

SUMMARY_FILE = "q2_classes_summary.json"


# --- priors and the label-shift correction ---------------------------------------------------
def class_prior(classes: Sequence[int] | np.ndarray) -> np.ndarray:
    """Share of each class 0..4 (sums to 1).

    Raises:
        ValueError: If empty, a class is outside 0..4, or a class does not occur (a zero prior
            cannot be corrected: the ratio would be undefined).
    """
    values = np.asarray(classes)
    if values.size == 0:
        raise ValueError("no classes given")
    if values.min() < 0 or values.max() >= N_CLASSES:
        raise ValueError(f"classes must be in 0..{N_CLASSES - 1}")
    counts = np.bincount(values.astype(int), minlength=N_CLASSES).astype(float)
    if (counts == 0).any():
        raise ValueError(f"every class must occur, counts: {counts.astype(int).tolist()}")
    return counts / counts.sum()


def prior_from_counts(counts: Sequence[float]) -> np.ndarray:
    """A prior from class counts (length 5, non-negative, not all zero)."""
    values = np.asarray(counts, dtype=float)
    if values.shape != (N_CLASSES,) or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"need {N_CLASSES} non-negative counts that are not all zero")
    return values / values.sum()


def _check_proba(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"proba must have shape (n, {N_CLASSES}), got {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("proba must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("every row of proba must sum to 1")
    return p


def prior_shift(
    proba: np.ndarray, source_prior: Sequence[float], target_prior: Sequence[float]
) -> np.ndarray:
    """Re-weight class probabilities from the training prior to the target prior.

    ``p'(c) = p(c) * target(c) / source(c)``, renormalised per row. A class with target prior 0
    gets probability 0.

    Raises:
        ValueError: On a wrong shape, a non-positive source prior or a negative target prior.
    """
    p = _check_proba(proba)
    source, target = np.asarray(source_prior, float), np.asarray(target_prior, float)
    if source.shape != (N_CLASSES,) or target.shape != (N_CLASSES,):
        raise ValueError(f"priors must have length {N_CLASSES}")
    if (source <= 0).any() or (target < 0).any():
        raise ValueError("source prior must be positive and target prior non-negative")
    weighted = p * (target / source)
    total = weighted.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("a row has zero mass after the correction")
    return weighted / total


# --- decisions and costs ----------------------------------------------------------------------
def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"cost matrix must be {N_CLASSES} x {N_CLASSES}")
    return m


def expected_cost_decision(proba: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Predicted class per truck = the class with the lowest expected cost (ties: lower class)."""
    expected = _check_proba(proba) @ _check_matrix(matrix)  # [n, k] = sum_c p_c * cost[c, k]
    return expected.argmin(axis=1)


def per_truck_cost(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """``cost[true, predicted]`` for every truck.

    Raises:
        ValueError: On different lengths or classes outside 0..4.
    """
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of the same length")
    for values in (true, pred):
        if values.size and (values.min() < 0 or values.max() >= N_CLASSES):
            raise ValueError(f"classes must be in 0..{N_CLASSES - 1}")
    return _check_matrix(matrix)[true.astype(int), pred.astype(int)]


def total_cost(y_true: Sequence[int], y_pred: Sequence[int], matrix: np.ndarray) -> float:
    """Sum of ``cost[true, predicted]`` over all trucks."""
    return float(per_truck_cost(y_true, y_pred, matrix).sum())


def confusion_table(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """5 x 5 counts, rows = true class, columns = predicted class."""
    per_truck_cost(y_true, y_pred, np.zeros((N_CLASSES, N_CLASSES)))  # validates the inputs
    table = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(table, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    labels = [f"class {k}" for k in range(N_CLASSES)]
    return pd.DataFrame(table, index=[f"true {k}" for k in range(N_CLASSES)], columns=labels)


def constant_baselines(y_true: Sequence[int], matrix: np.ndarray) -> dict[str, float]:
    """Total cost of "always predict class k" for k = 0..4 (the baselines to beat)."""
    true = np.asarray(y_true)
    return {
        f"always_{k}": total_cost(true, np.full(len(true), k), matrix) for k in range(N_CLASSES)
    }


def paired_bootstrap_cost(
    cost_a: np.ndarray, cost_b: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """Spread of ``total(a) - total(b)`` when the trucks are resampled with replacement.

    Args:
        cost_a, cost_b: Per-truck costs of two decisions on the same trucks.

    Returns:
        ``diff`` (a - b on the data, negative = a cheaper), ``std``, a 95 % percentile interval
        and ``share_a_cheaper`` (share of resamples where a is cheaper).
    """
    a, b = np.asarray(cost_a, float), np.asarray(cost_b, float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("need two non-empty per-truck cost arrays of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    sums = diff[idx].sum(axis=1)
    low, high = np.percentile(sums, [2.5, 97.5])
    return {"diff": float(diff.sum()), "std": float(sums.std(ddof=1)), "ci_low": float(low),
            "ci_high": float(high), "share_a_cheaper": float((sums < 0).mean())}  # fmt: skip


def class_calibration_table(y_true: Sequence[int], proba: np.ndarray) -> pd.DataFrame:
    """Per class: mean predicted probability vs. observed frequency (should match if calibrated)."""
    p = _check_proba(proba)
    y = np.asarray(y_true).astype(int)
    if len(y) != len(p):
        raise ValueError("y_true and proba must have the same length")
    observed = np.bincount(y, minlength=N_CLASSES) / len(y)
    return pd.DataFrame(
        {"class": range(N_CLASSES), "mean_predicted": p.mean(axis=0), "observed": observed}
    ).assign(ratio=lambda t: t["mean_predicted"] / t["observed"].replace(0, np.nan))


def decision_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, matrix: np.ndarray
) -> dict[str, float | list[int]]:
    """Total cost, macro F1 and how many trucks of each predicted class (documentation)."""
    return {
        "total_cost": total_cost(y_true, y_pred, matrix),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=range(N_CLASSES), average="macro", zero_division=0)
        ),  # fmt: skip
        "predicted_counts": np.bincount(y_pred, minlength=N_CLASSES).tolist(),
    }


# --- data and model ---------------------------------------------------------------------------
def load_training_frame_with_classes(
    config: Mapping[str, Any], n_per_vehicle: int | None
) -> pd.DataFrame:
    """Training cut-offs like ``load_training_frame`` plus the true class (0..4) of every row.

    Same rows and features as the Q1 training frame (same labels, same sampling seed); only the
    multi-class target is added. Only the TRAINING split is read.
    """
    _, seed = sampling_params_from_config(config)
    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    specs = read_table(table_path(config, "train", "specifications"))
    labels = build_labels(readouts, tte, class_bounds_from_config(config))
    sampled = sample_cutoffs(labels, n_per_vehicle=n_per_vehicle, seed=seed)
    frame = frame_from_sampled(
        readouts,
        sampled,
        counter_columns(),
        histogram_groups(),
        features_params_from_config(config),
    )
    del readouts
    frame[CLASS_COL] = sampled[CLASS_COL].astype(int)
    return attach_specifications(frame, specs)


def make_multiclass_pipeline(
    params: Mapping[str, Any], n_rounds: int, seed: int, categorical: Sequence[str]
) -> Any:
    """Preprocessing + multi-class XGBoost (``multi:softprob``) as one Pipeline."""
    booster = XGBClassifier(
        objective="multi:softprob", n_estimators=n_rounds, tree_method="hist",
        eval_metric="mlogloss", n_jobs=os.cpu_count() or 1, random_state=seed, verbosity=0,
        importance_type="gain", **params,
    )  # fmt: skip
    return build_model_pipeline(booster, categorical, scaler=None)


def target_prior_from_config(config: Mapping[str, Any]) -> np.ndarray:
    """``q2.target_class_counts`` (published validation mix) as a prior."""
    return prior_from_counts(config["q2"]["target_class_counts"])


# --- run --------------------------------------------------------------------------------------
def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit on the training trucks, check calibration out of fold, score the validation split."""
    section = config["q2"]
    seed, matrix = int(config["seed"]), cost_matrix_from_config(config)
    reports = resolve_path(config["paths"]["reports"])
    registry = json.loads((reports / "registry_q1.json").read_text(encoding="utf-8"))
    columns = registry_input_columns(config)

    frame = load_training_frame_with_classes(config, 5)
    x, y = select_model_inputs(frame, columns), frame[CLASS_COL].to_numpy(dtype=int)
    groups = frame[VEHICLE_COL].to_numpy()
    categorical = select_categorical_columns(x)
    prior_train, prior_target = class_prior(y), target_prior_from_config(config)
    logger.info("prior train %s | target %s", prior_train.round(4), prior_target.round(4))

    pipeline = make_multiclass_pipeline(registry["params"], registry["n_rounds"], seed, categorical)
    folds = StratifiedGroupKFold(
        n_splits=int(config["cv"]["n_splits"]), shuffle=True, random_state=int(config["cv"]["seed"])
    ).split(x, y, groups)
    oof = cross_val_predict(pipeline, x, y, cv=list(folds), method="predict_proba")
    oof_logloss = float(log_loss(y, oof, labels=range(N_CLASSES)))
    calibration = class_calibration_table(y, oof)
    logger.info("OOF log loss %.4f\n%s", oof_logloss, calibration.round(4).to_string(index=False))

    pipeline.fit(x, y)
    val = build_validation_frame(config)
    x_val, y_val = select_model_inputs(val, columns), val[CLASS_COL].to_numpy(dtype=int)
    proba = pipeline.predict_proba(x_val)
    corrected = prior_shift(proba, prior_train, prior_target)

    pred = {
        "uncorrected_expected_cost": expected_cost_decision(proba, matrix),
        "corrected_expected_cost": expected_cost_decision(corrected, matrix),
        "uncorrected_argmax": proba.argmax(axis=1),
    }
    metrics = {name: decision_metrics(y_val, p, matrix) for name, p in pred.items()}
    baselines = constant_baselines(y_val, matrix)
    best_const = min(baselines, key=baselines.get)
    costs = {name: per_truck_cost(y_val, p, matrix) for name, p in pred.items()}
    const_cost = per_truck_cost(y_val, np.full(len(y_val), int(best_const[-1])), matrix)
    n_boot = int(section["n_bootstrap"])
    paired = {
        "corrected_vs_uncorrected": paired_bootstrap_cost(
            costs["corrected_expected_cost"], costs["uncorrected_expected_cost"], n_boot, seed),
        f"corrected_vs_{best_const}": paired_bootstrap_cost(
            costs["corrected_expected_cost"], const_cost, n_boot, seed),
        f"uncorrected_vs_{best_const}": paired_bootstrap_cost(
            costs["uncorrected_expected_cost"], const_cost, n_boot, seed),
    }  # fmt: skip
    val_logloss = {
        "uncorrected": float(log_loss(y_val, proba, labels=range(N_CLASSES))),
        "corrected": float(log_loss(y_val, corrected, labels=range(N_CLASSES))),
    }
    summary = {
        "n_train_rows": len(x), "n_validation_trucks": len(y_val),
        "prior_train": prior_train.tolist(), "prior_target": prior_target.tolist(),
        "prior_target_source": "dataset paper, validation class counts (aggregate)",
        "oof_log_loss_train": oof_logloss, "validation_log_loss": val_logloss,
        "constant_baselines": baselines, "best_constant": best_const, "decisions": metrics,
        "paired_bootstrap_cost": paired, "n_bootstrap": n_boot,
        "caveat": "validation class mix is used for the correction and the cost is measured on "
                  "the same trucks: optimistic by construction (ADR 0004); test split untouched",
    }  # fmt: skip
    reports.mkdir(parents=True, exist_ok=True)
    (reports / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    calibration.to_csv(reports / "q2_calibration_train_oof.csv", index=False, float_format="%.5f")
    for name in ("uncorrected_expected_cost", "corrected_expected_cost"):
        confusion_table(y_val, pred[name]).to_csv(reports / f"q2_confusion_{name}.csv")
    _log_to_mlflow(config, summary)
    logger.info("Q2 summary:\n%s", json.dumps(summary, indent=2))
    return summary


def _log_to_mlflow(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    import mlflow

    setup_mlflow(config, experiment=config["q2"]["mlflow_experiment"])
    with mlflow.start_run(run_name="q2_multiclass_prior_shift"):
        mlflow.set_tags({"task": "q2_classes", "split": "official validation"})
        mlflow.log_params(
            {
                "model": "xgboost multi:softprob (Q1 winner params)",
                "training_sample": "k5",
                "n_train_rows": summary["n_train_rows"],
            }
        )
        for name, m in summary["decisions"].items():
            mlflow.log_metric(f"cost_{name}", m["total_cost"])
            mlflow.log_metric(f"macro_f1_{name}", m["macro_f1"])
        for name, cost in summary["constant_baselines"].items():
            mlflow.log_metric(f"cost_{name}", cost)
        mlflow.log_metric("oof_log_loss_train", summary["oof_log_loss_train"])
        mlflow.log_artifact(str(resolve_path(config["paths"]["reports"]) / SUMMARY_FILE))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
