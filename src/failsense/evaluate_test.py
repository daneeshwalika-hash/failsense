"""The single, frozen evaluation on the test splits: Q1, Q2 and RUL (plus a rehearsal).

See docs/test_evaluation_plan.md. Order of events: ``make freeze`` is re-run on the final commit,
``make freeze-verify`` must be clean, ``make rehearse`` proves the code path on the validation
split, THEN this script runs exactly once on test. It never trains, fits a calibrator, searches a
threshold or chooses anything: it only reads artifacts that ``failsense.freeze`` wrote and verified.

Protections of the single run (all before the first test byte is read):

1. ``--approved-single-run`` must be given;
2. ``reports/test_evaluation.lock`` must not exist (a second run is refused);
3. ``failsense.freeze.verify_manifest`` must report no problem: every frozen file, source file,
   config section, the five test files (by SHA-256), the library versions, a clean git tree, no
   code or config change since the freeze commit;
4. **preflight** (``load_frozen``): every frozen model (Q1, Q2, RUL LightGBM and Ridge), the
   calibrator and the threshold are loaded and checked (column lists, a probe prediction). A file
   that cannot be loaded stops the run with a clear message, **without a lock and without reading
   any test file**;
5. only then the lock is created, and only then the test files are read. A crash after that point
   uses up the run (the lock stays), which is why steps 3 and 4 and the rehearsal come first.

Rehearsal (``--rehearse-on-validation``, ``make rehearse``): the same functions run on the official
VALIDATION split (SCANIA) and on cut TRAINING engines (C-MAPSS has no validation file). It needs no
approval, creates no lock, opens no test file, does not hash the test files and writes nothing
inside the repository (the result goes to the log and to a file in the system temp folder).

The SCANIA files are read from the RAW csv files (``pd.read_csv`` with the same call as
``failsense.data.convert``), never from the Parquet copies in ``data/processed``.

Reported (the six decisions of the plan): Q1 PR-AUC with the test prevalence, ROC-AUC, cost at the
frozen threshold under both mappings next to the never-alert and always-alert baselines
(always-alert is the primary baseline); Q2 corrected vs. uncorrected vs. the constant predictors;
RUL last-cycle RMSE of LightGBM next to the frozen Ridge and mean baselines (primary: capped RUL;
also raw RUL and the NASA score). Everything is reported as it comes out: no re-tuning, no second
run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from failsense.config import find_repo_root, load_config, resolve_path
from failsense.data.schema import SPEC_COLUMNS, counter_columns, histogram_groups
from failsense.freeze import hash_tree, load_model, verify_manifest
from failsense.models.cost import N_CLASSES, binary_costs
from failsense.models.q2_classes import (
    confusion_table,
    constant_baselines,
    expected_cost_decision,
    paired_bootstrap_cost,
    per_truck_cost,
    prior_shift,
)
from failsense.models.threshold import per_truck_costs, total_cost

logger = logging.getLogger(__name__)

LOCK_FILE = "test_evaluation.lock"
RESULT_FILE = "test_evaluation_final.json"
LABEL_FILE_COLUMN = "class_label"
REHEARSAL_BANNER = "REHEARSAL – validation, not test"
REHEARSAL_FILE_PREFIX = "failsense_rehearsal_"
NASA_A_EARLY, NASA_A_LATE = 13.0, 10.0  # asymmetric NASA score constants (late is punished more)


class EvaluationRefusedError(RuntimeError):
    """A protection of the single run says no. Nothing was read."""


# --- protections ---------------------------------------------------------------------------
def guard(
    config: Mapping[str, Any],
    approved: bool,
    verify: Callable[[Mapping[str, Any]], list[str]] = verify_manifest,
    preflight: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Check approval, lock, manifest and the frozen artifacts; then create the lock.

    Args:
        config: Project config.
        approved: The ``--approved-single-run`` flag.
        verify: Returns the problems of the frozen state (empty list = verified).
        preflight: Called with the manifest after the verification and BEFORE the lock is
            created (``run`` passes the loader of the frozen models). If it raises, there is no
            lock and no test file has been read.

    Returns:
        The manifest.

    Raises:
        EvaluationRefusedError: Without the approval flag, with an existing lock, if the manifest
            does not verify, or if the preflight fails. No lock is created and no test file is
            read then.
    """
    reports = resolve_path(config["paths"]["reports"])
    lock = reports / LOCK_FILE
    if not approved:
        raise EvaluationRefusedError(
            "the single test run needs --approved-single-run (Daneesh approves it)"
        )
    if lock.exists():
        raise EvaluationRefusedError(f"{lock} exists: the single test run has already been started")
    problems = verify(config)
    if problems:
        raise EvaluationRefusedError("the frozen state does not verify: " + "; ".join(problems[:6]))
    manifest_path = resolve_path(config["freeze"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if preflight is not None:
        preflight(manifest)  # raises EvaluationRefusedError: no lock yet, nothing read
    reports.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "code_commit": manifest["code_commit"],
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "status": "started: no second run is allowed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _mark_lock(config: Mapping[str, Any], status: str) -> None:
    lock = resolve_path(config["paths"]["reports"]) / LOCK_FILE
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["status"] = status
    lock.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --- preflight: the frozen artifacts (no test file is involved) ------------------------------
@dataclass(frozen=True)
class FrozenArtifacts:
    """Everything the evaluation needs, loaded and checked before any data file is read."""

    q1_model: Any
    q1_columns: list[str]
    q1_calibrator: dict[str, Any]
    q1_threshold: float
    q2_model: Any
    q2_meta: dict[str, Any]
    rul_lightgbm: Any
    rul_ridge: Any
    rul_meta: dict[str, Any]


Q2_META_KEYS = (
    "input_columns", "prior_train", "prior_target", "cost_matrix",
    "best_constant_from_validation", "untrusted_types",
)  # fmt: skip
RUL_META_KEYS = (
    "cap", "rolling_windows", "informative_sensors", "columns_raw", "columns_rolling",
    "mean_rul_training", "untrusted_types",
)  # fmt: skip


def _probe_frame(columns: Sequence[str]) -> pd.DataFrame:
    """Two synthetic rows with the given input columns (numbers, a category for the specs)."""
    return pd.DataFrame(
        {c: ["Cat0", "Cat0"] if c in SPEC_COLUMNS else [1.0, 2.0] for c in columns}
    )[list(columns)]


def _check_columns(model: Any, columns: Sequence[str], what: str) -> None:
    """The model must have been fitted on exactly these columns, in this order."""
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError(f"{what}: the column list is empty or has duplicates")
    fitted = getattr(model, "feature_names_in_", None)
    if fitted is not None and [str(c) for c in fitted] != [str(c) for c in columns]:
        raise ValueError(
            f"{what}: the model was fitted on {len(fitted)} columns that differ from the "
            f"{len(columns)} recorded input columns"
        )


def _check_calibrator(calibrator: Mapping[str, Any], threshold: float) -> None:
    """A sigmoid calibrator with a positive slope and a threshold strictly between 0 and 1."""
    if calibrator.get("kind") != "sigmoid":
        raise ValueError(f"only a sigmoid calibrator is supported, got {calibrator.get('kind')!r}")
    slope, intercept = float(calibrator["slope"]), float(calibrator["intercept"])
    eps = float(calibrator["logit_clip_eps"])
    if not (math.isfinite(slope) and slope > 0 and math.isfinite(intercept)):
        raise ValueError("the calibrator needs a finite intercept and a finite positive slope")
    if not 0 < eps < 0.5:
        raise ValueError(f"logit_clip_eps must be in (0, 0.5), got {eps}")
    if not 0 < threshold < 1:
        raise ValueError(f"the alert threshold must be in (0, 1), got {threshold}")


def _require_keys(meta: Mapping[str, Any], keys: Sequence[str], what: str) -> None:
    missing = [k for k in keys if k not in meta]
    if missing:
        raise ValueError(f"{what} is missing {missing}")


def _load_q1(config: Mapping[str, Any], manifest: Mapping[str, Any], frozen: Path) -> tuple:
    import mlflow.sklearn

    from failsense.models.calibration import apply_sigmoid_parameters
    from failsense.models.cost import alert_class_from_config, cost_matrix_from_config
    from failsense.models.validation_data import registry_input_columns

    q1 = manifest["q1"]
    calibrator, threshold = dict(q1["calibrator"]), float(q1["threshold"])
    _check_calibrator(calibrator, threshold)
    # the config values the evaluation reads from the manifest must be there and valid
    float(manifest["config"]["features"]["top_fraction"])
    int(manifest["config"]["seed"])
    cost_config = {"cost": manifest["config"]["cost"]}
    cost_matrix_from_config(cost_config)
    alert_class_from_config(cost_config)
    columns = registry_input_columns(config)
    model = mlflow.sklearn.load_model(str(frozen / q1["folder"]))
    _check_columns(model, columns, "Q1 model")
    score = np.asarray(model.predict_proba(_probe_frame(columns)), dtype=float)
    if score.shape != (2, 2) or not np.isfinite(score).all():
        raise ValueError("Q1 model: the probe prediction is not a finite (n, 2) probability table")
    if not np.isfinite(apply_sigmoid_parameters(calibrator, score[:, 1])).all():
        raise ValueError("Q1 calibrator: the probe risk is not finite")
    return model, columns, calibrator, threshold


def _load_q2(frozen: Path) -> tuple:
    meta = json.loads((frozen / "q2_meta.json").read_text(encoding="utf-8"))
    _require_keys(meta, Q2_META_KEYS, "q2_meta.json")
    if not re.fullmatch(r"always_[0-4]", str(meta["best_constant_from_validation"])):
        raise ValueError(
            "q2_meta.json: best_constant_from_validation must be always_0 ... always_4"
        )
    matrix = np.asarray(meta["cost_matrix"], dtype=float)
    if matrix.shape != (N_CLASSES, N_CLASSES):
        raise ValueError("q2_meta.json: the cost matrix must be 5 x 5")
    model = load_model(frozen / "q2_model.skops", meta["untrusted_types"])
    _check_columns(model, meta["input_columns"], "Q2 model")
    proba = np.asarray(model.predict_proba(_probe_frame(meta["input_columns"])), dtype=float)
    if proba.shape != (2, N_CLASSES):
        raise ValueError(f"Q2 model: expected {N_CLASSES} class probabilities, got {proba.shape}")
    prior_shift(proba, meta["prior_train"], meta["prior_target"])  # validates the priors
    return model, meta


def _load_rul(frozen: Path) -> tuple:
    meta = json.loads((frozen / "rul_meta.json").read_text(encoding="utf-8"))
    _require_keys(meta, RUL_META_KEYS, "rul_meta.json")
    if int(meta["cap"]) < 1 or not math.isfinite(float(meta["mean_rul_training"])):
        raise ValueError("rul_meta.json: invalid cap or mean_rul_training")
    if not set(meta["informative_sensors"]) <= set(meta["columns_raw"]):
        raise ValueError("rul_meta.json: the informative sensors must be raw columns")
    models = []
    for file, columns in (
        ("rul_lightgbm.skops", meta["columns_rolling"]),
        ("rul_ridge.skops", meta["columns_raw"]),
    ):
        model = load_model(frozen / file, meta["untrusted_types"][file])
        _check_columns(model, columns, file)
        prediction = np.asarray(model.predict(_probe_frame(columns)), dtype=float)
        if prediction.shape != (2,) or not np.isfinite(prediction).all():
            raise ValueError(f"{file}: the probe prediction is not finite")
        models.append(model)
    return models[0], models[1], meta


def load_frozen(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> FrozenArtifacts:
    """Load and check ALL frozen models, the calibrator and the threshold (the preflight).

    Checked per model: the file loads with the recorded trusted types, it was fitted on exactly
    the recorded input columns, and a probe prediction on two synthetic rows is finite and has the
    expected shape. No data file (training, validation or test) is opened here.

    Raises:
        EvaluationRefusedError: If any artifact is missing, cannot be loaded or fails a check.
            The message names the artifact. The caller has not created a lock yet.
    """
    frozen = resolve_path(config["freeze"]["dir"])
    loaded: dict[str, tuple] = {}
    for name, loader in (
        ("Q1 model, calibrator and threshold", lambda: _load_q1(config, manifest, frozen)),
        ("Q2 model", lambda: _load_q2(frozen)),
        ("RUL models", lambda: _load_rul(frozen)),
    ):
        try:
            loaded[name] = loader()
        except Exception as exc:  # any failure here must stop the run BEFORE the lock
            raise EvaluationRefusedError(
                f"frozen artifact check failed ({name}): {type(exc).__name__}: {exc}"
            ) from exc
    q1_model, q1_columns, q1_calibrator, q1_threshold = loaded["Q1 model, calibrator and threshold"]
    q2_model, q2_meta = loaded["Q2 model"]
    rul_lightgbm, rul_ridge, rul_meta = loaded["RUL models"]
    logger.info(
        "Preflight ok: Q1 (%d columns, threshold %.6f), Q2 (%d columns), RUL (%d / %d columns)",
        len(q1_columns), q1_threshold, len(q2_meta["input_columns"]),
        len(rul_meta["columns_rolling"]), len(rul_meta["columns_raw"]),
    )  # fmt: skip
    return FrozenArtifacts(
        q1_model, list(q1_columns), q1_calibrator, q1_threshold, q2_model, q2_meta,
        rul_lightgbm, rul_ridge, rul_meta,
    )  # fmt: skip


# --- generic statistics --------------------------------------------------------------------
def bootstrap_interval(
    statistic: Callable[..., float], arrays: Sequence[np.ndarray], n_boot: int, seed: int
) -> tuple[float, float]:
    """95 % percentile interval of ``statistic(*arrays)`` over row resamples (same draws)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            value = statistic(*(a[idx] for a in arrays))
        except ValueError:  # e.g. a resample without a positive truck
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        raise ValueError("no positive")
    return float(average_precision_score(y, score))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() in (0, len(y)):
        raise ValueError("one class")
    return float(roc_auc_score(y, score))


# --- Q1 ------------------------------------------------------------------------------------
def q1_metrics(
    classes: np.ndarray, risk: np.ndarray, threshold: float, matrix: np.ndarray, alert_class: int,
    n_boot: int, seed: int,
) -> dict[str, Any]:  # fmt: skip
    """Q1 on the test trucks at the FROZEN threshold (nothing is fitted or chosen here).

    Returns PR-AUC (+interval) with the test prevalence and the lift, ROC-AUC, the confusion at
    the threshold, per-class recall and the total cost under the primary and the literal mapping
    next to the constant baselines (never alert, alert every truck = the primary baseline).

    Raises:
        ValueError: On empty or inconsistent input, or classes outside 0..4.
    """
    classes, risk = np.asarray(classes, dtype=int), np.asarray(risk, dtype=float)
    if classes.size == 0 or classes.shape != risk.shape:
        raise ValueError("classes and risk must be non-empty arrays of the same length")
    if classes.min() < 0 or classes.max() >= N_CLASSES:
        raise ValueError("classes must be in 0..4")
    y = (classes >= 1).astype(int)
    alert = risk >= threshold
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    fn, tn = int((~alert & (y == 1)).sum()), int((~alert & (y == 0)).sum())
    out: dict[str, Any] = {
        "n_trucks": int(len(y)), "n_positives": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": float(threshold), "alert_rate": float(alert.mean()),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "recall_per_class": {
            str(k): float(alert[classes == k].mean()) for k in range(1, N_CLASSES)
            if (classes == k).any()
        },
    }  # fmt: skip
    if 0 < y.sum() < len(y):
        pr, roc = _ap(y, risk), _auc(y, risk)
        out["pr_auc"] = {
            "value": pr, "ci95": bootstrap_interval(_ap, [y, risk], n_boot, seed),
            "prevalence": out["prevalence"], "lift": pr / out["prevalence"],
        }  # fmt: skip
        out["roc_auc"] = {"value": roc, "ci95": bootstrap_interval(_auc, [y, risk], n_boot, seed)}
    costs: dict[str, Any] = {}
    for mapping in ("primary", "literal"):
        c = binary_costs(matrix, alert_class, mapping)
        model = total_cost(classes, alert, c)
        never = total_cost(classes, np.zeros(len(y), dtype=bool), c)
        always = total_cost(classes, np.ones(len(y), dtype=bool), c)
        entry = {
            "model": model, "never_alert": never, "always_alert": always,
            "saving_vs_always_alert": always - model, "saving_vs_never_alert": never - model,
        }  # fmt: skip
        alert_cost, miss_cost = per_truck_costs(classes, c)
        model_i = np.where(alert, alert_cost, miss_cost)
        entry["model_minus_always_alert_bootstrap"] = paired_bootstrap_cost(
            model_i, alert_cost, n_boot, seed
        )
        costs[mapping] = entry
    out["cost"] = costs
    return out


# --- Q2 ------------------------------------------------------------------------------------
def q2_metrics(
    classes: np.ndarray, proba: np.ndarray, prior_train: Sequence[float],
    prior_target: Sequence[float], matrix: np.ndarray, frozen_best_constant: str, n_boot: int,
    seed: int,
) -> dict[str, Any]:  # fmt: skip
    """Q2: corrected and uncorrected expected-cost decisions against the constant predictors."""
    classes = np.asarray(classes, dtype=int)
    corrected = prior_shift(proba, prior_train, prior_target)
    decisions = {
        "uncorrected_expected_cost": expected_cost_decision(proba, matrix),
        "corrected_expected_cost": expected_cost_decision(corrected, matrix),
        "uncorrected_argmax": np.asarray(proba).argmax(axis=1),
    }
    constants = constant_baselines(classes, matrix)
    best_const_test = min(constants, key=constants.get)
    per_truck = {n: per_truck_cost(classes, d, matrix) for n, d in decisions.items()}

    def const_cost(name: str) -> np.ndarray:
        return per_truck_cost(classes, np.full(len(classes), int(name[-1])), matrix)

    def pair(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        return paired_bootstrap_cost(a, b, n_boot, seed)

    corrected_i = per_truck["corrected_expected_cost"]
    return {
        "n_trucks": int(len(classes)),
        "class_counts": np.bincount(classes, minlength=N_CLASSES).tolist(),
        "constant_baselines": constants, "frozen_best_constant": frozen_best_constant,
        "best_constant_on_test": best_const_test,
        "decisions": {
            n: {
                "total_cost": float(per_truck[n].sum()),
                "macro_f1": float(f1_score(classes, d, labels=range(N_CLASSES), average="macro",
                                           zero_division=0)),  # fmt: skip
                "predicted_counts": np.bincount(d, minlength=N_CLASSES).tolist(),
            }
            for n, d in decisions.items()
        },
        "paired_bootstrap_cost": {
            "corrected_vs_uncorrected": pair(corrected_i, per_truck["uncorrected_expected_cost"]),
            f"corrected_vs_{frozen_best_constant}": pair(
                corrected_i, const_cost(frozen_best_constant)
            ),
            f"corrected_vs_{best_const_test}": pair(corrected_i, const_cost(best_const_test)),
        },
        "confusion_corrected": confusion_table(
            classes, decisions["corrected_expected_cost"]
        ).values.tolist(),
        "confusion_uncorrected": confusion_table(
            classes, decisions["uncorrected_expected_cost"]
        ).values.tolist(),
    }  # fmt: skip


# --- RUL -----------------------------------------------------------------------------------
def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric NASA/PHM08 score: ``exp(-d/13) - 1`` if d < 0 else ``exp(d/10) - 1``, summed."""
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(np.where(d < 0, np.exp(-d / NASA_A_EARLY) - 1, np.exp(d / NASA_A_LATE) - 1).sum())


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def rul_metrics(
    true_raw: np.ndarray, cap: int, predictions: Mapping[str, np.ndarray], reference: str,
    n_boot: int, seed: int,
) -> dict[str, Any]:  # fmt: skip
    """RMSE at the last cycle per engine: primary vs. the CAPPED RUL, also raw RUL and NASA score.

    ``reference`` (the frozen Ridge) is compared with every other model by a paired bootstrap of the
    RMSE difference over the engines.
    """
    true_raw = np.asarray(true_raw, dtype=float)
    true_cap = np.minimum(true_raw, cap)
    out: dict[str, Any] = {"n_engines": int(len(true_raw)), "cap": int(cap), "models": {}}
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        if pred.shape != true_raw.shape:
            raise ValueError(f"{name}: one prediction per engine is needed")
        out["models"][name] = {
            "rmse_capped": _rmse(pred, true_cap),
            "rmse_capped_ci95": bootstrap_interval(_rmse, [pred, true_cap], n_boot, seed),
            "rmse_raw": _rmse(pred, true_raw),
            "mae_capped": float(np.mean(np.abs(pred - true_cap))),
            "nasa_score_raw": nasa_score(pred, true_raw),
        }
    ref = np.asarray(predictions[reference], dtype=float)
    out["reference"] = reference
    out["paired_rmse_difference_vs_reference"] = {}
    for name, pred in predictions.items():
        if name == reference:
            continue
        pred = np.asarray(pred, dtype=float)

        def diff(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
            return _rmse(a, t) - _rmse(b, t)

        out["paired_rmse_difference_vs_reference"][name] = {
            "diff": _rmse(pred, true_cap) - _rmse(ref, true_cap),
            "ci95": bootstrap_interval(diff, [pred, ref, true_cap], n_boot, seed),
        }
    return out


# --- reading the data files (the test files only after the guard) ----------------------------
def read_scania(
    config: Mapping[str, Any], split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Readouts, specifications, labels of one split from the RAW csv files (as ``convert`` does).

    Raises:
        ValueError: For a split that has no label file (only ``validation`` and ``test`` have one).
    """
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    files = config["datasets"]["scania"][split]
    return tuple(  # type: ignore[return-value]
        pd.read_csv(resolve_path(files[name]), engine="pyarrow")
        for name in ("readouts", "specifications", "labels")
    )


def read_scania_test(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The SCANIA test files (the hashed RAW csv files). Only the single run calls this."""
    return read_scania(config, "test")


def read_cmapss_test(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """The 100 truncated test engines and their true remaining life (one value per engine)."""
    from failsense.rul.data import load_train

    cmapss = config["datasets"]["cmapss"]
    engines = load_train(
        resolve_path(cmapss["test"])
    )  # same format and data contract as the training file
    rul = pd.read_csv(resolve_path(cmapss["rul"]), sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    if len(rul) != engines["engine"].nunique():
        raise ValueError("RUL file and test engines differ in number")
    return engines, rul.astype(float)


def rehearsal_cmapss(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """TRAINING engines cut before their end, shaped like the test files (rehearsal only).

    C-MAPSS has no validation file, so the rehearsal cuts every training engine a fixed number of
    cycles before its failure (10, 20 ... 100 by engine id, at most half of its life). The true
    remaining life is then the number of removed cycles. The frozen models were fitted on these
    engines, so the resulting error is **in-sample**: it proves that the code path works, it is
    not an estimate of the test error.
    """
    from failsense.rul.data import CYCLE_COL, ENGINE_COL, load_train

    train = load_train(resolve_path(config["datasets"]["cmapss"]["train"]))
    length = train.groupby(ENGINE_COL)[CYCLE_COL].transform("max")
    cut = np.minimum(10 + (train[ENGINE_COL] % 10) * 10, length // 2)
    kept = train[train[CYCLE_COL] <= length - cut].reset_index(drop=True)
    true_rul = cut.groupby(train[ENGINE_COL]).first().sort_index().to_numpy(dtype=float)
    return kept, true_rul


# --- the three evaluations (the same code for the test run and the rehearsal) ----------------
def last_readout_frame(
    scania: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], manifest: Mapping[str, Any]
) -> pd.DataFrame:
    """One row per truck (its last readout) with the model inputs and the true class."""
    from failsense.models.validation_data import build_last_readout_frame

    readouts, specs, labels = scania
    return build_last_readout_frame(
        readouts, specs, labels, counter_columns(), histogram_groups(),
        float(manifest["config"]["features"]["top_fraction"]),
    )  # fmt: skip


def evaluate_q1(
    manifest: Mapping[str, Any], frozen: FrozenArtifacts, frame: pd.DataFrame, n_boot: int
) -> dict[str, Any]:
    """Q1: last readouts -> frozen model -> frozen calibrator -> frozen threshold -> metrics."""
    from failsense.models.calibration import apply_sigmoid_parameters
    from failsense.models.cost import alert_class_from_config, cost_matrix_from_config
    from failsense.models.validation_data import select_model_inputs

    inputs = select_model_inputs(frame, frozen.q1_columns)
    risk = apply_sigmoid_parameters(
        frozen.q1_calibrator, frozen.q1_model.predict_proba(inputs)[:, 1]
    )
    classes = frame["proximity_class"].to_numpy(dtype=int)
    cost_config = {"cost": manifest["config"]["cost"]}
    return q1_metrics(
        classes, risk, frozen.q1_threshold, cost_matrix_from_config(cost_config),
        alert_class_from_config(cost_config), n_boot, int(manifest["config"]["seed"]),
    )  # fmt: skip


def evaluate_q2(
    manifest: Mapping[str, Any], frozen: FrozenArtifacts, frame: pd.DataFrame, n_boot: int
) -> dict[str, Any]:
    """Q2: the same last readouts -> frozen multi-class model -> expected-cost decisions."""
    from failsense.models.validation_data import select_model_inputs

    meta = frozen.q2_meta
    proba = frozen.q2_model.predict_proba(select_model_inputs(frame, meta["input_columns"]))
    return q2_metrics(
        frame["proximity_class"].to_numpy(dtype=int), proba, meta["prior_train"],
        meta["prior_target"],
        np.asarray(meta["cost_matrix"], dtype=float), meta["best_constant_from_validation"], n_boot,
        int(manifest["config"]["seed"]),
    )  # fmt: skip


def evaluate_rul(
    manifest: Mapping[str, Any], frozen: FrozenArtifacts, engines: pd.DataFrame,
    true_raw: np.ndarray, n_boot: int,
) -> dict[str, Any]:  # fmt: skip
    """RUL: the last cycle of every engine (rolling features from its past) -> frozen models."""
    from failsense.rul.timeseries import add_rolling_features

    meta = frozen.rul_meta
    frame = add_rolling_features(engines, meta["informative_sensors"], meta["rolling_windows"])
    last = frame.loc[frame.groupby("engine")["cycle"].idxmax()].sort_values(
        "engine"
    )  # last cycle each
    predictions = {
        "lightgbm_rolling": frozen.rul_lightgbm.predict(last[meta["columns_rolling"]]),
        "ridge_raw": frozen.rul_ridge.predict(last[meta["columns_raw"]]),
        "mean_predictor": np.full(len(last), meta["mean_rul_training"]),
    }
    return rul_metrics(true_raw, int(meta["cap"]), predictions, "ridge_raw", n_boot,
                       int(manifest["config"]["seed"]))  # fmt: skip


def _evaluate(
    manifest: Mapping[str, Any],
    frozen: FrozenArtifacts,
    scania_reader: Callable[[], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    cmapss_reader: Callable[[], tuple[pd.DataFrame, np.ndarray]],
    n_boot: int,
) -> dict[str, Any]:
    """Q1, Q2 and RUL with the given data readers (test files or rehearsal data)."""
    frame = last_readout_frame(scania_reader(), manifest)
    q1 = evaluate_q1(manifest, frozen, frame, n_boot)
    q2 = evaluate_q2(manifest, frozen, frame, n_boot)
    engines, true_raw = cmapss_reader()
    return {"q1": q1, "q2": q2, "rul": evaluate_rul(manifest, frozen, engines, true_raw, n_boot)}


# --- run -----------------------------------------------------------------------------------
def run(
    config: Mapping[str, Any],
    approved: bool,
    n_bootstrap: int = 1000,
    verify: Callable[[Mapping[str, Any]], list[str]] = verify_manifest,
    track: bool = False,
) -> dict[str, Any]:
    """The single run: guard + preflight, then Q1, Q2 and RUL, then the result file."""
    loaded: list[FrozenArtifacts] = []
    # the frozen models are loaded and checked BEFORE the lock exists and before any test read
    manifest = guard(
        config, approved, verify, preflight=lambda m: loaded.append(load_frozen(config, m))
    )
    frozen = loaded[0]
    reports = resolve_path(config["paths"]["reports"])
    try:
        result = {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_commit": manifest["code_commit"], "decisions": manifest["decisions"],
            "n_bootstrap": n_bootstrap,
            **_evaluate(
                manifest, frozen, lambda: read_scania_test(config),
                lambda: read_cmapss_test(config), n_bootstrap,
            ),
        }  # fmt: skip
    except Exception as exc:  # the lock stays: the run is used up, the reason is recorded
        _mark_lock(config, f"failed: {type(exc).__name__}: {exc}")
        raise
    (reports / RESULT_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _mark_lock(config, "finished")
    if track:
        _log_to_mlflow(config, result)
    logger.info("Single test evaluation finished: %s", reports / RESULT_FILE)
    return result


# --- rehearsal on the validation split -------------------------------------------------------
def rehearsal_output_path(output: str | Path | None = None) -> Path:
    """Where the rehearsal result is written: never inside the repository.

    ``None`` gives a new file in the system temp folder.

    Raises:
        EvaluationRefusedError: If ``output`` lies inside the repository (a new file there would
            make the git tree dirty and the manifest would no longer verify).
    """
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = Path(tempfile.gettempdir()) / f"{REHEARSAL_FILE_PREFIX}{stamp}.json"
    else:
        path = Path(output).expanduser()
    resolved, root = path.resolve(), find_repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise EvaluationRefusedError(
            f"the rehearsal writes nothing inside the repository ({root}); "
            f"choose an --output outside of it, got {resolved}"
        )
    return resolved


def rehearse(
    config: Mapping[str, Any], n_bootstrap: int = 1000, output: str | Path | None = None
) -> dict[str, Any]:
    """The same code path as the single run, on the official VALIDATION split.

    No approval, no lock, no test file is opened or hashed, no MLflow run, and no file is written
    inside the repository. SCANIA: the validation RAW csv files. C-MAPSS: cut training engines
    (``rehearsal_cmapss``, an in-sample plumbing check). The result is logged and written to
    ``output`` (default: a new file in the system temp folder).

    The Q1 numbers must equal what the calibration measured on the same trucks with the same
    threshold (``reports/calibration_q1_summary.json``: in-sample decision).

    Raises:
        EvaluationRefusedError: If the manifest is missing, a frozen artifact fails the preflight,
            or ``output`` lies inside the repository.
    """
    out_path = rehearsal_output_path(output)
    logger.warning("===== %s =====", REHEARSAL_BANNER)
    manifest_path = resolve_path(config["freeze"]["manifest"])
    if not manifest_path.is_file():
        raise EvaluationRefusedError(f"manifest {manifest_path} not found; run `make freeze` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = load_frozen(config, manifest)
    recorded = manifest.get("frozen_files")
    frozen_match = (
        None if recorded is None else hash_tree(resolve_path(config["freeze"]["dir"])) == recorded
    )
    if frozen_match is False:
        logger.warning("the frozen files differ from the manifest: run `make freeze-verify`")
    result = {
        "mode": REHEARSAL_BANNER, "split": "validation",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": manifest.get("code_commit"), "n_bootstrap": n_bootstrap,
        "frozen_files_match_manifest": frozen_match,
        "notes": [
            "no lock, no test file opened or hashed, nothing written inside the repository",
            "the full manifest verification is `make freeze-verify`, not part of the rehearsal",
            "rul: cut TRAINING engines, in-sample: a plumbing check, not an error estimate",
        ],
        **_evaluate(
            manifest, frozen, lambda: read_scania(config, "validation"),
            lambda: rehearsal_cmapss(config), n_bootstrap,
        ),
    }  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    q1 = result["q1"]
    confusion = q1["confusion"]
    logger.warning(
        "%s | Q1: %d trucks, %d positives, PR-AUC %s, %d alerts, cost (primary mapping) %.0f, "
        "threshold %.10f",
        REHEARSAL_BANNER, q1["n_trucks"], q1["n_positives"],
        f"{q1['pr_auc']['value']:.4f}" if "pr_auc" in q1 else "n/a",
        confusion["tp"] + confusion["fp"], q1["cost"]["primary"]["model"], q1["threshold"],
    )  # fmt: skip
    logger.warning(
        "%s | Q2: corrected cost %.0f, uncorrected %.0f | RUL (in-sample plumbing check): "
        "LightGBM RMSE %.2f, Ridge %.2f on %d cut training engines",
        REHEARSAL_BANNER,
        result["q2"]["decisions"]["corrected_expected_cost"]["total_cost"],
        result["q2"]["decisions"]["uncorrected_expected_cost"]["total_cost"],
        result["rul"]["models"]["lightgbm_rolling"]["rmse_capped"],
        result["rul"]["models"]["ridge_raw"]["rmse_capped"], result["rul"]["n_engines"],
    )  # fmt: skip
    logger.warning("%s | full result: %s", REHEARSAL_BANNER, out_path)
    return result


def _log_to_mlflow(config: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    import mlflow

    from failsense.tracking import setup_mlflow

    setup_mlflow(config, experiment="failsense-test-final")
    with mlflow.start_run(run_name="single_test_evaluation"):
        mlflow.set_tags(
            {"stage": "test", "single_run": "true", "code_commit": result["code_commit"]}
        )
        q1 = result["q1"]
        mlflow.log_metrics({
            "q1_pr_auc": q1["pr_auc"]["value"], "q1_prevalence": q1["prevalence"],
            "q1_cost_primary": q1["cost"]["primary"]["model"],
            "q1_cost_primary_always_alert": q1["cost"]["primary"]["always_alert"],
            "q2_cost_corrected": result["q2"]["decisions"]["corrected_expected_cost"]["total_cost"],
            "rul_rmse_lightgbm": result["rul"]["models"]["lightgbm_rolling"]["rmse_capped"],
            "rul_rmse_ridge": result["rul"]["models"]["ridge_raw"]["rmse_capped"],
        })  # fmt: skip
        mlflow.log_artifact(str(resolve_path(config["paths"]["reports"]) / RESULT_FILE))


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: the single run (``--approved-single-run``) or the rehearsal."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--approved-single-run", action="store_true",
                        help="Daneesh has approved the one test run")  # fmt: skip
    parser.add_argument(
        "--rehearse-on-validation", action="store_true",
        help="run the same code path on the VALIDATION split: no lock, no test file, nothing "
             "written inside the repository",
    )  # fmt: skip
    parser.add_argument(
        "--output", type=Path, default=None,
        help="rehearsal only: result file outside the repository (default: system temp folder)",
    )  # fmt: skip
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.rehearse_on_validation and args.approved_single_run:
        parser.error("--rehearse-on-validation and --approved-single-run exclude each other")
    if args.output is not None and not args.rehearse_on_validation:
        parser.error("--output belongs to --rehearse-on-validation")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        if args.rehearse_on_validation:
            rehearse(load_config(), args.n_bootstrap, args.output)
        else:
            run(load_config(), args.approved_single_run, args.n_bootstrap, track=True)
    except EvaluationRefusedError as exc:
        logger.error("refused: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
