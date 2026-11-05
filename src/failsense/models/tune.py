"""Hyperparameter search for XGBoost and LightGBM (Optuna) on the validation-like eval set.

Run it with::

    uv run python -m failsense.models.tune --time-one-trial      # time one trial, project the total
    uv run python -m failsense.models.tune                       # or: make tune (both models)
    uv run python -m failsense.models.tune --model xgboost --n-trials 10 --no-mlflow

What is optimised and how it is kept honest
-------------------------------------------
* **Objective:** the mean PR-AUC over the 5 folds on the **calibrated** validation-like
  evaluation set (step 7a; the ``literal`` set is a second reference in ``evaluate``).
* **Training sample:** ``k = 5`` cut-offs per truck (decision of step 7a).
* **Early stopping inside every fold** on a truck-grouped inner split of the *training*
  trucks (``GroupShuffleSplit`` by ``vehicle_id``, 15 % of the trucks); the evaluation rows
  never take part. The number of rounds is chosen by early stopping, not searched.
* **Search:** TPE sampler (seeded) and a ``MedianPruner`` per fold: after every fold the running
  mean is reported and a trial that is worse than the median of earlier trials at that fold
  stops. Trial 0 is the untuned step-6 setting (with engineered features), an anchor.
* **No class weights.** The engineered features are a categorical yes/no choice.
* **Optimistic by construction:** the best of N trials was selected on the same folds and
  evaluation rows it is scored on, so its score is higher than what the same settings reach
  on new trucks. The official validation (06.11) is the real check.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score

from failsense.config import load_config, resolve_path
from failsense.models.boosters import FAMILIES, STEP6_PARAMS, fit_early_stopping
from failsense.models.tuning_data import TuningData, build_tuning_data
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

BEST_PARAMS_FILE = "hpo_q1_best_params.json"


def suggest_params(family: str, trial: optuna.Trial) -> dict[str, float | int]:
    """The search space of one model (see the table in ``docs/problem_framing.md`` §6).

    Raises:
        ValueError: For an unknown family.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, choose one of {list(FAMILIES)}")
    params: dict[str, float | int] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    if family == "xgboost":
        params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 200.0, log=True)
    else:
        params["num_leaves"] = trial.suggest_int("num_leaves", 8, 128, log=True)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 20, 400, log=True)
    return params


def anchor_trial(family: str) -> dict[str, Any]:
    """Trial 0: the untuned step-6 settings with engineered features (inside the search space)."""
    return {**STEP6_PARAMS[family], "engineered": True}


def make_objective(
    family: str, data: TuningData, tuning: Mapping[str, Any], n_jobs: int
) -> Callable[[optuna.Trial], float]:
    """Objective: mean PR-AUC over the folds on the calibrated evaluation rows.

    Per fold the booster is fitted on the inner-train trucks with early stopping on the
    inner-validation trucks and scored on the fold's evaluation rows (never used for
    stopping). The running mean is reported after every fold for the pruner.
    """
    seed = int(tuning["seed"])

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        params = suggest_params(family, trial)
        engineered = trial.suggest_categorical("engineered", [True, False])
        feature_set = "engineered" if engineered else "base"
        scores: list[float] = []
        rounds: list[int] = []
        for step, fold in enumerate(data.folds):
            x = fold.x_train[feature_set]
            model, n_rounds = fit_early_stopping(
                family, params,
                x[fold.inner_train], fold.y_train[fold.inner_train],
                x[fold.inner_val], fold.y_train[fold.inner_val],
                seed=seed, n_jobs=n_jobs, max_rounds=int(tuning["max_rounds"]),
                early_stopping_rounds=int(tuning["early_stopping_rounds"]),
            )  # fmt: skip
            proba = model.predict_proba(fold.x_eval["calibrated"][feature_set])[:, 1]
            scores.append(float(average_precision_score(fold.y_eval["calibrated"], proba)))
            rounds.append(n_rounds)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                trial.set_user_attr("pruned_after_folds", step + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("best_rounds", rounds)
        trial.set_user_attr("seconds", time.perf_counter() - started)
        return float(np.mean(scores))

    return objective


def _n_jobs() -> int:
    """Folds run one after another, so one fit may use every core."""
    return os.cpu_count() or 1


def time_one_trial(
    config: Mapping[str, Any], data: TuningData, family: str, n_trials: int | None = None
) -> dict[str, float]:
    """Run the anchor trial (step-6 settings) once and project the total runtime.

    The projection is an upper bound for the study: the pruner stops weak trials early.
    """
    tuning = config["tuning"]
    n = int(n_trials if n_trials is not None else tuning["n_trials"])
    objective = make_objective(family, data, tuning, _n_jobs())
    started = time.perf_counter()
    score = objective(optuna.trial.FixedTrial(anchor_trial(family)))
    seconds = time.perf_counter() - started
    out = {"score": score, "trial_seconds": seconds, "projected_minutes": seconds * n / 60}
    logger.info(
        "%s: one trial (5 folds, early stopping) took %.1f s, PR-AUC %.4f -> about %.1f min for "
        "%d trials (upper bound: the pruner stops weak trials early)",
        family, seconds, score, out["projected_minutes"], n,
    )  # fmt: skip
    return out


def study_summary(study: optuna.Study) -> dict[str, Any]:
    """Best parameters, score, counts and the median early-stopping rounds of the best trial."""
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    best = study.best_trial
    engineered = bool(best.params["engineered"])
    return {
        "best_value": float(best.value),
        "best_trial": best.number,
        "params": {k: v for k, v in best.params.items() if k != "engineered"},
        "engineered": engineered,
        "n_rounds": int(np.median(best.user_attrs["best_rounds"])),
        "best_rounds_per_fold": list(best.user_attrs["best_rounds"]),
        "fold_scores": list(best.user_attrs["fold_scores"]),
        "n_trials": len(study.trials),
        "n_complete": len(complete),
        "n_pruned": len(pruned),
        "anchor_value": float(study.trials[0].value) if study.trials[0].value is not None else None,
    }


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: state, score, parameters, rounds and runtime (for ``reports/``)."""
    rows = []
    for t in study.trials:
        rounds = t.user_attrs.get("best_rounds")
        duration = t.duration.total_seconds() if t.duration else None
        rows.append(
            {
                "trial": t.number, "state": t.state.name, "value": t.value,
                "seconds": t.user_attrs.get("seconds", duration),
                "median_rounds": float(np.median(rounds)) if rounds else None,
                "pruned_after_folds": t.user_attrs.get("pruned_after_folds"),
                **t.params,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def run_study(
    config: Mapping[str, Any],
    data: TuningData,
    family: str,
    n_trials: int | None = None,
    timeout_minutes: float | None = None,
    track: bool = False,
) -> dict[str, Any]:
    """Create (or resume) the Optuna study of one model, optimise, write reports, log to MLflow.

    The study lives in a SQLite file, so a crash or a timeout loses nothing and a re-run
    continues until ``n_trials`` trials exist.
    """
    tuning = config["tuning"]
    target = int(n_trials if n_trials is not None else tuning["n_trials"])
    timeout = float(timeout_minutes if timeout_minutes is not None else tuning["timeout_minutes"])
    storage = f"sqlite:///{resolve_path(tuning['storage'])}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"failsense-q1-{family}", storage=storage, direction="maximize",
        sampler=TPESampler(seed=int(tuning["seed"])),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )  # fmt: skip
    if not study.trials:
        study.enqueue_trial(anchor_trial(family))  # trial 0: the untuned step-6 setting
    # the enqueued anchor is one of the planned trials; on a resume only finished trials count
    finished = sum(t.state.is_finished() for t in study.get_trials(deepcopy=False))
    remaining = max(0, target - finished)

    def log_trial(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info(
            "%s trial %d %s: %s%s", family, trial.number, trial.state.name,
            "-" if trial.value is None else f"PR-AUC {trial.value:.4f}",
            f" (best so far {study.best_value:.4f})" if trial.value is not None else "",
        )  # fmt: skip

    started = time.perf_counter()
    if remaining:
        study.optimize(
            make_objective(family, data, tuning, _n_jobs()), n_trials=remaining,
            timeout=timeout * 60, callbacks=[log_trial],
        )  # fmt: skip
    seconds = time.perf_counter() - started
    summary = {
        **study_summary(study),
        "family": family,
        "seconds": seconds,
        "n_trials_target": target,
        "timed_out": len(study.trials) < target,
        "timeout_minutes": timeout,
        "seed": int(tuning["seed"]),
    }
    logger.info(
        "%s: best PR-AUC %.4f (trial %d); %d trials = %d complete + %d pruned of %d planned%s; "
        "%.1f min (seed %d)",
        family, summary["best_value"], summary["best_trial"], summary["n_trials"],
        summary["n_complete"], summary["n_pruned"], target,
        " (time cap reached)" if summary["timed_out"] else "", seconds / 60, summary["seed"],
    )  # fmt: skip

    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    table_path_out = reports / f"hpo_q1_{family}_trials.csv"
    trials_table(study).to_csv(table_path_out, index=False, float_format="%.5f")
    write_best_params(reports / BEST_PARAMS_FILE, family, summary)
    if track:
        with mlflow.start_run(run_name=f"hpo_{family}") as active:
            mlflow.set_tags({"task": "q1_binary", "stage": "hpo", "family": family})
            mlflow.log_params(
                {f"best_{k}": v for k, v in summary["params"].items()}
                | {"best_engineered": summary["engineered"], "n_rounds": summary["n_rounds"],
                   "n_trials_target": target, "n_trials": summary["n_trials"],
                   "n_complete": summary["n_complete"], "n_pruned": summary["n_pruned"],
                   "timed_out": summary["timed_out"], "timeout_minutes": timeout,
                   "seed": summary["seed"],
                   "sampler": "TPE", "pruner": "MedianPruner", "evaluation": "calibrated",
                   "training_sample": "k5", "early_stopping": "truck-grouped inner split"}
            )  # fmt: skip
            mlflow.log_metrics(
                {
                    "best_pr_auc": summary["best_value"],
                    "n_complete": summary["n_complete"],
                    "n_pruned": summary["n_pruned"],
                    "seconds": seconds,
                }
            )
            if summary["anchor_value"] is not None:
                mlflow.log_metric("anchor_pr_auc", summary["anchor_value"])
            mlflow.log_artifact(str(table_path_out))
            logger.info("Logged MLflow run hpo_%s (%s)", family, active.info.run_id)
    return summary


def write_best_params(path: Path, family: str, summary: Mapping[str, Any]) -> None:
    """Merge the best parameters of one model into the shared JSON file."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[family] = {
        k: summary[k]
        for k in ("params", "engineered", "n_rounds", "best_value", "n_trials", "n_complete",
                  "n_pruned", "n_trials_target", "timed_out", "timeout_minutes", "seed",
                  "best_rounds_per_fold", "fold_scores", "anchor_value")
    }  # fmt: skip
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=[*FAMILIES, "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=None, help="trials per model (config)")
    parser.add_argument("--timeout-minutes", type=float, default=None, help="cap per model")
    parser.add_argument(
        "--time-one-trial",
        action="store_true",
        help="run the step-6 anchor trial once, print the projected runtime, then stop",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    families = FAMILIES if args.model == "both" else (args.model,)
    data = build_tuning_data(config, eval_methods=("calibrated",))
    if args.time_one_trial:
        total = 0.0
        for family in families:
            total += time_one_trial(config, data, family, args.n_trials)["projected_minutes"]
        logger.info("Projected total for %s: about %.1f min (upper bound)", list(families), total)
        return
    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    for family in families:
        run_study(config, data, family, args.n_trials, args.timeout_minutes, track=track)


if __name__ == "__main__":
    main()
