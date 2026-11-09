"""The single, frozen evaluation on the test splits: Q1, Q2 and RUL.

See docs/test_evaluation_plan.md. Order of events (decided 09.11): ``make freeze`` is re-run on the
final commit on 14.11, THEN this script runs exactly once. It never trains, fits a calibrator,
searches a threshold or chooses anything: it only reads artifacts that ``failsense.freeze`` wrote
and verified.

Protections (all in ``guard``, before the first test byte is read):

1. ``--approved-single-run`` must be given;
2. ``reports/test_evaluation.lock`` must not exist (a second run is refused; the lock is created
   BEFORE the first test file is read, so a crash cannot be "retried with a quick fix" without
   anybody noticing);
3. ``failsense.freeze.verify_manifest`` must report no problem: every frozen file, source file,
   config section, the five test files (by SHA-256), the library versions, a clean git tree, no
   code or config change since the freeze commit.

The test files are read from the RAW files whose hashes are in the manifest (``pd.read_csv`` with
the same call as ``failsense.data.convert``), never from the Parquet copies in ``data/processed``.

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
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from failsense.config import load_config, resolve_path
from failsense.data.schema import counter_columns, histogram_groups
from failsense.freeze import load_model, verify_manifest
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
NASA_A_EARLY, NASA_A_LATE = 13.0, 10.0  # asymmetric NASA score constants (late is punished more)


class EvaluationRefusedError(RuntimeError):
    """A protection of the single run says no. Nothing was read."""


# --- protections ---------------------------------------------------------------------------
def guard(
    config: Mapping[str, Any],
    approved: bool,
    verify: Callable[[Mapping[str, Any]], list[str]] = verify_manifest,
) -> dict[str, Any]:
    """Check approval, lock and manifest; then create the lock. Returns the manifest.

    Raises:
        EvaluationRefusedError: Without the approval flag, with an existing lock, or if the
            manifest does not verify. No lock is created and no test file is read then.
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


# --- reading the test files (only here, only after the guard) ------------------------------
def read_scania_test(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Readouts, specifications, labels from the hashed RAW csv files (as ``convert`` does)."""
    test = config["datasets"]["scania"]["test"]
    return tuple(  # type: ignore[return-value]
        pd.read_csv(resolve_path(test[name]), engine="pyarrow")
        for name in ("readouts", "specifications", "labels")
    )


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


# --- the three evaluations -----------------------------------------------------------------
def evaluate_q1(
    config: Mapping[str, Any], manifest: Mapping[str, Any], n_boot: int
) -> dict[str, Any]:
    import mlflow.sklearn

    from failsense.models.calibration import apply_sigmoid_parameters
    from failsense.models.cost import alert_class_from_config, cost_matrix_from_config
    from failsense.models.validation_data import (
        build_last_readout_frame,
        registry_input_columns,
        select_model_inputs,
    )

    readouts, specs, labels = read_scania_test(config)
    frame = build_last_readout_frame(
        readouts, specs, labels, counter_columns(), histogram_groups(),
        float(manifest["config"]["features"]["top_fraction"]),
    )  # fmt: skip
    model = mlflow.sklearn.load_model(
        str(resolve_path(config["freeze"]["dir"]) / manifest["q1"]["folder"])
    )
    inputs = select_model_inputs(frame, registry_input_columns(config))
    risk = apply_sigmoid_parameters(manifest["q1"]["calibrator"], model.predict_proba(inputs)[:, 1])
    classes = frame["proximity_class"].to_numpy(dtype=int)
    cost_config = {"cost": manifest["config"]["cost"]}
    return q1_metrics(
        classes, risk, float(manifest["q1"]["threshold"]), cost_matrix_from_config(cost_config),
        alert_class_from_config(cost_config), n_boot, int(manifest["config"]["seed"]),
    )  # fmt: skip


def evaluate_q2(
    config: Mapping[str, Any], manifest: Mapping[str, Any], n_boot: int
) -> dict[str, Any]:
    from failsense.models.validation_data import build_last_readout_frame, select_model_inputs

    frozen = resolve_path(config["freeze"]["dir"])
    meta = json.loads((frozen / "q2_meta.json").read_text(encoding="utf-8"))
    readouts, specs, labels = read_scania_test(config)
    frame = build_last_readout_frame(
        readouts, specs, labels, counter_columns(), histogram_groups(),
        float(manifest["config"]["features"]["top_fraction"]),
    )  # fmt: skip
    model = load_model(frozen / "q2_model.skops", meta["untrusted_types"])
    proba = model.predict_proba(select_model_inputs(frame, meta["input_columns"]))
    return q2_metrics(
        frame["proximity_class"].to_numpy(dtype=int), proba, meta["prior_train"],
        meta["prior_target"],
        np.asarray(meta["cost_matrix"], dtype=float), meta["best_constant_from_validation"], n_boot,
        int(manifest["config"]["seed"]),
    )  # fmt: skip


def evaluate_rul(
    config: Mapping[str, Any], manifest: Mapping[str, Any], n_boot: int
) -> dict[str, Any]:
    from failsense.rul.timeseries import add_rolling_features

    frozen = resolve_path(config["freeze"]["dir"])
    meta = json.loads((frozen / "rul_meta.json").read_text(encoding="utf-8"))
    engines, true_raw = read_cmapss_test(config)
    frame = add_rolling_features(engines, meta["informative_sensors"], meta["rolling_windows"])
    last = frame.loc[frame.groupby("engine")["cycle"].idxmax()].sort_values(
        "engine"
    )  # last cycle each
    lgbm = load_model(frozen / "rul_lightgbm.skops", meta["untrusted_types"]["rul_lightgbm.skops"])
    ridge = load_model(frozen / "rul_ridge.skops", meta["untrusted_types"]["rul_ridge.skops"])
    predictions = {
        "lightgbm_rolling": lgbm.predict(last[meta["columns_rolling"]]),
        "ridge_raw": ridge.predict(last[meta["columns_raw"]]),
        "mean_predictor": np.full(len(last), meta["mean_rul_training"]),
    }
    return rul_metrics(true_raw, int(meta["cap"]), predictions, "ridge_raw", n_boot,
                       int(manifest["config"]["seed"]))  # fmt: skip


# --- run -----------------------------------------------------------------------------------
def run(
    config: Mapping[str, Any],
    approved: bool,
    n_bootstrap: int = 1000,
    verify: Callable[[Mapping[str, Any]], list[str]] = verify_manifest,
    track: bool = False,
) -> dict[str, Any]:
    """The single run: guard, then Q1, Q2 and RUL, then the result file (see the docstring)."""
    manifest = guard(config, approved, verify)  # creates the lock BEFORE any test file is read
    reports = resolve_path(config["paths"]["reports"])
    try:
        result = {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_commit": manifest["code_commit"], "decisions": manifest["decisions"],
            "n_bootstrap": n_bootstrap,
            "q1": evaluate_q1(config, manifest, n_bootstrap),
            "q2": evaluate_q2(config, manifest, n_bootstrap),
            "rul": evaluate_rul(config, manifest, n_bootstrap),
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--approved-single-run", action="store_true",
                        help="Daneesh has approved the one test run")  # fmt: skip
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        run(load_config(), args.approved_single_run, args.n_bootstrap, track=True)
    except EvaluationRefusedError as exc:
        logger.error("refused: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
