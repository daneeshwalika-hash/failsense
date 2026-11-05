"""Tuned vs. untuned boosting models on both validation-like evaluation sets.

Run it with::

    uv run python -m failsense.models.evaluate      # or: make evaluate   (after make tune)

Models (training sample ``k = 5``, same folds and evaluation rows as the tuning):

* ``xgboost_tuned`` / ``lightgbm_tuned``: the best Optuna trial of ``failsense.models.tune``,
  fitted per fold exactly like in the search (early stopping on a truck-grouped inner split).
* ``xgboost_step6_engineered``: the untuned step-6 XGBoost (300 fixed rounds) with the
  engineered features, the model the tuned ones have to beat.

Every model is fitted once per fold and predicts **both** evaluation sets: ``calibrated``
(main) and ``literal`` (second reference).

Metrics per fold, then mean and standard deviation over the 5 folds: PR-AUC, ROC-AUC, Brier
score, the mean predicted probability and, **when the 5 % riskiest trucks of a fold are
alerted**, precision, recall and F1. The confusion matrix at that cut is pooled over the folds
and is **provisional**: the real threshold is chosen by cost on the official validation on
06.11. There is no total cost today (it needs that threshold).

Honest reading: the tuned score is the best of N trials, selected on the same folds and
evaluation rows it is scored on, so it is **optimistic**. The tuned-vs-untuned comparison is
paired per fold, but the real check is the official validation on 06.11.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from failsense.config import load_config, resolve_path
from failsense.models.baseline_cv import paired_comparison
from failsense.models.boosters import (
    FAMILIES,
    STEP6_PARAMS,
    STEP6_ROUNDS,
    fit_early_stopping,
    fit_fixed_rounds,
)
from failsense.models.tune import BEST_PARAMS_FILE
from failsense.models.tuning_data import EVAL_METHODS, FoldData, TuningData, build_tuning_data
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

UNTUNED = "xgboost_step6_engineered"
PROVISIONAL = "provisional - threshold by cost on 06.11"
WINNER_FILE = "evaluate_q1_winner.json"


def top_k_metrics(
    y_true: Sequence[int], score: Sequence[float], fraction: float = 0.05
) -> dict[str, float]:
    """Alert the ``fraction`` riskiest trucks (highest score) and score the alert.

    Ties are broken by row position (stable), so the result is deterministic.

    Returns:
        ``alerts``, ``tp``, ``fp``, ``fn``, ``tn``, ``precision`` (share of the alerts that
        are positives), ``recall`` (share of the positives that are alerted) and ``f1``.

    Raises:
        ValueError: If the inputs are empty or differ in length, or ``fraction`` is not in (0, 1].
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.size == 0 or y.shape != s.shape:
        raise ValueError("need two non-empty arrays of equal length")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = max(1, int(np.ceil(fraction * y.size)))
    alerted = np.zeros(y.size, dtype=bool)
    alerted[np.argsort(-s, kind="stable")[:k]] = True
    tp = int(y[alerted].sum())
    fp = int(k - tp)
    fn = int(y.sum() - tp)
    tn = int(y.size - k - fn)
    precision = tp / k
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"alerts": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}  # fmt: skip


def fold_metrics(y_true: np.ndarray, proba: np.ndarray, fraction: float) -> dict[str, float]:
    """All metrics of one model on one fold of one evaluation set."""
    top = top_k_metrics(y_true, proba, fraction)
    # ROC-AUC needs both classes; a fold without positives (only possible in tiny test
    # fixtures) gives NaN instead of an error.
    roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_proba": float(np.mean(proba)),
        "top_precision": top["precision"], "top_recall": top["recall"], "top_f1": top["f1"],
        "tp": top["tp"], "fp": top["fp"], "fn": top["fn"], "tn": top["tn"],
    }  # fmt: skip


def choose_winner(
    paired_calibrated: Mapping[str, Mapping[str, float]], calibrated_mean: Mapping[str, float]
) -> dict[str, Any]:
    """The pre-set rule: a tuned model is registered only if it is clearly better.

    "Clearly better" than the untuned engineered XGBoost on the **calibrated** set: the mean of
    the per-fold differences is larger than their standard deviation **and** the tuned model
    wins in at least 4 of 5 folds. Of the models that qualify the one with the higher mean
    PR-AUC wins; if none qualifies the simpler untuned model is the winner.

    Args:
        paired_calibrated: ``paired_comparison`` of every tuned model vs. the untuned one.
        calibrated_mean: Mean calibrated PR-AUC of every model (tuned ones and ``UNTUNED``).
    """
    qualifying = [
        name for name, cmp in paired_calibrated.items()
        if cmp["mean_diff"] > cmp["std_diff"] and cmp["wins"] >= 4
    ]  # fmt: skip
    if qualifying:
        winner = max(qualifying, key=lambda n: calibrated_mean[n])
        reason = (
            f"{winner} is clearly better than {UNTUNED} on the calibrated set (paired "
            f"+{paired_calibrated[winner]['mean_diff']:.4f} > std "
            f"{paired_calibrated[winner]['std_diff']:.4f}, better in "
            f"{int(paired_calibrated[winner]['wins'])}/5 folds)"
        )
    else:
        winner = UNTUNED
        details = "; ".join(
            f"{n}: {c['mean_diff']:+.4f} vs. std {c['std_diff']:.4f}, {int(c['wins'])}/5 folds"
            for n, c in paired_calibrated.items()
        )
        reason = (
            f"no tuned model is clearly better than the untuned engineered XGBoost on the "
            f"calibrated set ({details}); the simpler untuned model is registered"
        )
    return {"winner": winner, "qualifying": qualifying, "reason": reason}


def _fit_predict(
    spec: Mapping[str, Any], fold: FoldData, seed: int, n_jobs: int, tuning: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Fit one model on one fold and predict every evaluation set of that fold."""
    fs = "engineered" if spec["engineered"] else "base"
    x = fold.x_train[fs]
    if spec["early_stopping"]:
        model, _ = fit_early_stopping(
            spec["family"], spec["params"],
            x[fold.inner_train], fold.y_train[fold.inner_train],
            x[fold.inner_val], fold.y_train[fold.inner_val],
            seed=seed, n_jobs=n_jobs, max_rounds=int(tuning["max_rounds"]),
            early_stopping_rounds=int(tuning["early_stopping_rounds"]),
        )  # fmt: skip
    else:
        model = fit_fixed_rounds(
            spec["family"], spec["params"], x, fold.y_train, spec["rounds"], seed, n_jobs
        )
    return {m: model.predict_proba(fold.x_eval[m][fs])[:, 1] for m in fold.x_eval}


def model_specs(best: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The three models of the comparison (tuned XGBoost / LightGBM, untuned step-6 XGBoost)."""
    specs = {
        f"{family}_tuned": {
            "family": family, "params": best[family]["params"],
            "engineered": best[family]["engineered"], "early_stopping": True,
            "rounds": best[family]["n_rounds"],
        }
        for family in FAMILIES
    }  # fmt: skip
    specs[UNTUNED] = {
        "family": "xgboost", "params": STEP6_PARAMS["xgboost"], "engineered": True,
        "early_stopping": False, "rounds": STEP6_ROUNDS,
    }  # fmt: skip
    return specs


def run(
    config: Mapping[str, Any], data: TuningData | None = None, track: bool = False
) -> dict[str, Any]:
    """Evaluate the three models on both evaluation sets and choose the winner.

    Args:
        config: Project config (needs ``reports/hpo_q1_best_params.json`` from ``tune``).
        data: Prepared folds (built from the config if omitted).
        track: Log the table and the winner to MLflow.

    Returns:
        ``summary`` (per eval set and model: metric means / stds), ``paired`` (tuned vs.
        untuned, both sets), ``confusion`` (pooled, provisional) and ``winner``.
    """
    tuning = config["tuning"]
    reports = resolve_path(config["paths"]["reports"])
    best = json.loads((reports / BEST_PARAMS_FILE).read_text())
    missing = [f for f in FAMILIES if f not in best]
    if missing:
        raise FileNotFoundError(f"no tuning result for {missing}; run `make tune` first")
    data = data or build_tuning_data(config, EVAL_METHODS)
    fraction = float(tuning["top_fraction"])
    seed, n_jobs = int(tuning["seed"]), os.cpu_count() or 1
    specs = model_specs(best)

    per_fold: dict[tuple[str, str], list[dict[str, float]]] = {
        (m, name): [] for m in EVAL_METHODS for name in specs
    }
    for fold in data.folds:
        for name, spec in specs.items():
            proba = _fit_predict(spec, fold, seed, n_jobs, tuning)
            for method, p in proba.items():
                per_fold[(method, name)].append(fold_metrics(fold.y_eval[method], p, fraction))
        logger.info("Fold %d done", fold.fold)

    rows, confusion, scores = [], [], {}
    for (method, name), folds in per_fold.items():
        frame = pd.DataFrame(folds)
        scores[(method, name)] = frame["pr_auc"].to_numpy()
        row = {"eval_set": method, "model": name}
        for metric in (
            "pr_auc",
            "roc_auc",
            "brier",
            "mean_proba",
            "top_precision",
            "top_recall",
            "top_f1",
        ):
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=0))
        rows.append(row)
        confusion.append(
            {"eval_set": method, "model": name, "alert_fraction": fraction, "status": PROVISIONAL,
             **{k: int(frame[k].sum()) for k in ("tp", "fp", "fn", "tn")}}
        )  # fmt: skip
    summary = pd.DataFrame(rows)

    paired: dict[str, dict[str, dict[str, float]]] = {m: {} for m in EVAL_METHODS}
    for method in EVAL_METHODS:
        for name in specs:
            if name != UNTUNED:
                paired[method][name] = paired_comparison(
                    scores[(method, name)], scores[(method, UNTUNED)]
                )
    calibrated_mean = {n: float(scores[("calibrated", n)].mean()) for n in specs}
    winner = choose_winner(paired["calibrated"], calibrated_mean)
    winner_spec = specs[winner["winner"]]
    winner.update(
        family=winner_spec["family"], params=dict(winner_spec["params"]),
        engineered=winner_spec["engineered"], n_rounds=int(winner_spec["rounds"]),
        early_stopping_in_evaluation=winner_spec["early_stopping"],
    )  # fmt: skip

    # the tuned fit reproduces the tuning score on the calibrated set (same seeds, same folds)
    for family in FAMILIES:
        got, expected = calibrated_mean[f"{family}_tuned"], float(best[family]["best_value"])
        logger.info("%s_tuned: calibrated PR-AUC %.5f (tuning study said %.5f, diff %+.2e)",
                    family, got, expected, got - expected)  # fmt: skip

    summary.to_csv(reports / "evaluate_q1.csv", index=False, float_format="%.5f")
    pd.DataFrame(confusion).to_csv(reports / "evaluate_q1_confusion.csv", index=False)
    paired_rows = [
        {"eval_set": m, "model": n, "reference": UNTUNED, **cmp}
        for m, by_model in paired.items() for n, cmp in by_model.items()
    ]  # fmt: skip
    pd.DataFrame(paired_rows).to_csv(
        reports / "evaluate_q1_paired.csv", index=False, float_format="%.5f"
    )
    (reports / WINNER_FILE).write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")

    logger.info("\n%s", summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m, by_model in paired.items():
        for n, cmp in by_model.items():
            logger.info(
                "[%s] paired PR-AUC %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
                m, n, UNTUNED, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
            )  # fmt: skip
    logger.info("Winner: %s - %s", winner["winner"], winner["reason"])
    logger.info("The tuned scores are the best of N trials chosen on these folds: optimistic. "
                "Confusion matrices are %s.", PROVISIONAL)  # fmt: skip
    if track:
        with mlflow.start_run(run_name="evaluate_tuned_vs_untuned") as active:
            mlflow.set_tags({"task": "q1_binary", "stage": "evaluate"})
            mlflow.log_params(
                {
                    "winner": winner["winner"],
                    "top_fraction": fraction,
                    "training_sample": "k5",
                    "confusion_status": PROVISIONAL,
                }
            )
            for _, r in summary.iterrows():
                for metric in (
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "mean_proba",
                    "top_precision",
                    "top_recall",
                ):
                    mlflow.log_metric(f"{r['eval_set']}_{r['model']}_{metric}", float(r[metric]))
            for filename in (
                "evaluate_q1.csv",
                "evaluate_q1_confusion.csv",
                "evaluate_q1_paired.csv",
                WINNER_FILE,
            ):
                mlflow.log_artifact(str(reports / filename))
            logger.info("Logged MLflow run evaluate_tuned_vs_untuned (%s)", active.info.run_id)
    return {"summary": summary, "paired": paired, "confusion": confusion, "winner": winner}


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
