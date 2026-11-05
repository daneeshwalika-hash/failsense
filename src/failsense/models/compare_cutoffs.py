"""Compare the number of training cut-offs per truck (k) on a FIXED evaluation set.

Run it with::

    uv run python -m failsense.models.compare_cutoffs                # k = 1, 5, all
    uv run python -m failsense.models.compare_cutoffs --ks 1 3 5 10
    uv run python -m failsense.models.compare_cutoffs --no-mlflow

The question
------------
``k`` (``sampling.n_cutoffs_per_vehicle``) decides how many random readouts
per truck we use as training examples. It is a hyperparameter of the data
preparation, so it is chosen with cross validation, not guessed.

Why not simply run ``baseline_cv --n-cutoffs 1/5/all``?
-------------------------------------------------------
In ``baseline_cv`` the *validation* rows of each fold also come from the
sample with ``k`` cut-offs. Changing ``k`` would then change **what we
measure on**, not only what we train on:

* ``k = all`` validates on every readout of a truck (long histories dominate,
  different age distribution, different share of positives),
* ``k = 1`` validates on one readout per truck, like the real validation and
  test splits.

The scores would not be comparable (different exam for every student). This
module therefore fixes the exam:

1. **Evaluation sample:** exactly one random cut-off per truck (``k = 1``,
   seeded). This imitates the official validation/test splits.
2. **Fold assignment per vehicle:** ``StratifiedGroupKFold`` on that sample
   assigns every truck to one fold, once. The same assignment is used for
   every ``k``.
3. For every ``k``: train on the ``k`` cut-offs of the training trucks of a
   fold, validate on the single evaluation cut-off of the validation trucks.

Only the training sample changes between the runs, so differences in the
score are caused by ``k`` alone. Group leakage is impossible: a truck is in
the training part *or* the validation part of a fold, never both (checked in
every fold).

Validation-like evaluation (05.11, the protocol from step 7 on)
---------------------------------------------------------------
The random evaluation row above is not what the official validation split looks like:
its last readouts are much later (``time_step`` median 169 vs. 83). With
``--eval validation-like`` the evaluation row of every truck is chosen by calibrated rank
matching (``failsense.models.eval_set``): it imitates the ``time_step`` distribution of the
official validation last readouts **and** their positive rate (2.70 %). Used: the validation
readouts' ``time_step`` and one aggregate number of the validation labels (136 / 5,046, no
per-row labels); the test set is untouched. ``--eval-method`` selects ``calibrated`` (default),
``literal`` (one uniform random readout per truck, as the dataset paper states) or ``rank``
(``time_step`` marginal only, rejected: positive rate 9.6 %). The model is XGBoost with the
step-6 settings::

    uv run python -m failsense.models.compare_cutoffs --eval validation-like --eval-set-only

    uv run python -m failsense.models.compare_cutoffs --eval validation-like
    uv run python -m failsense.models.compare_cutoffs --eval validation-like --samplings kall \
        --time-one-fold

Training samplings compared on that fixed set: ``k5`` (current), ``k10``, ``kall``,
``ts70`` (only readouts with ``time_step >= 70``) and ``matched`` (cut-offs drawn to
follow the validation ``time_step`` distribution). For the best sampling XGBoost is
re-checked without ``time_step`` and with the engineered features, on the same rows. The
default (``--eval random``, logistic regression) is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from failsense.analysis.time_step_check import ks_distance
from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    BINARY_COL,
    TIME_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.data.split import (
    Fold,
    assert_no_group_overlap,
    cv_params_from_config,
    make_group_folds,
)
from failsense.features.build import attach_specifications, features_params_from_config
from failsense.models.baseline_cv import (
    SCORING,
    SCORING_WITH_PROBA,
    build_training_frame,
    frame_from_sampled,
    n_cutoffs_label,
    paired_comparison,
    parse_n_cutoffs,
    read_table,
    table_path,
)
from failsense.models.eval_set import (
    EVAL_METHODS,
    MIN_TIME_STEP,
    build_eval_cutoffs_by_method,
    cutoffs_from_time_step,
    eval_set_report,
    known_labels,
    matched_training_cutoffs,
    plot_eval_set,
    positive_rate_report,
    validation_time_steps,
    vehicle_fold_from_folds,
)
from failsense.models.pipeline import (
    build_logreg_pipeline,
    build_model_pipeline,
    select_feature_columns,
    split_feature_columns,
)
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "k_comparison"}
VL_RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "validation_like_eval"}
VL_SAMPLINGS = ("k5", "k10", "kall", "ts70", "matched")

DEFAULT_KS = ("1", "5", "all")


def assign_vehicle_folds(eval_frame: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """Assign every vehicle to exactly one validation fold.

    Args:
        eval_frame: One row per vehicle (the evaluation sample, ``k = 1``),
            with ``vehicle_id`` and the binary label (used for stratification).
        n_splits: Number of folds.
        seed: Seed for ``StratifiedGroupKFold`` (same seed = same assignment).

    Returns:
        A Series ``vehicle_id -> fold number`` (0 .. n_splits-1).

    Raises:
        ValueError: If a vehicle appears more than once (then the sample is
            not "one cut-off per truck" and the stratification is wrong).
    """
    if eval_frame[VEHICLE_COL].duplicated().any():
        raise ValueError("eval_frame must contain exactly one row per vehicle")

    folds = make_group_folds(eval_frame, n_splits=n_splits, seed=seed)
    # Start with -1 ("no fold yet"); KFold puts every row into exactly one
    # validation part, so afterwards no -1 is left.
    vehicle_fold = pd.Series(-1, index=eval_frame[VEHICLE_COL].to_numpy(), name="fold")
    for fold_no, (_, val_pos) in enumerate(folds):
        vehicle_fold.iloc[val_pos] = fold_no
    return vehicle_fold


def build_fixed_eval_folds(
    train_vehicles: pd.Series,
    eval_vehicles: pd.Series,
    vehicle_fold: pd.Series,
) -> list[Fold]:
    """Folds for "train on sample A, validate on sample B", stacked as [A; B].

    The caller stacks the training sample (``len(train_vehicles)`` rows)
    on top of the evaluation sample. Position ``i`` < offset is a training
    candidate, position ``offset + j`` is evaluation row ``j``.

    Args:
        train_vehicles: ``vehicle_id`` of every row of the training sample.
        eval_vehicles: ``vehicle_id`` of every row of the evaluation sample.
        vehicle_fold: Output of ``assign_vehicle_folds``.

    Returns:
        ``(train_pos, val_pos)`` per fold, usable as ``cv=`` in scikit-learn:
        training rows of all vehicles *outside* fold ``f`` and evaluation rows
        of the vehicles *inside* fold ``f``.

    Raises:
        ValueError: If a vehicle has no fold, or (should never happen) a
            vehicle ends up in both parts of a fold.
    """
    train_fold = train_vehicles.map(vehicle_fold)
    eval_fold = eval_vehicles.map(vehicle_fold)
    no_fold = pd.concat([train_vehicles[train_fold.isna()], eval_vehicles[eval_fold.isna()]])
    if len(no_fold):
        examples = sorted(no_fold.unique().tolist())[:5]
        raise ValueError(f"{no_fold.nunique()} vehicles have no fold, e.g. {examples}")

    train_fold_arr = train_fold.to_numpy(dtype=int)
    eval_fold_arr = eval_fold.to_numpy(dtype=int)
    train_ids = train_vehicles.to_numpy()
    eval_ids = eval_vehicles.to_numpy()
    offset = len(train_vehicles)
    n_splits = int(vehicle_fold.max()) + 1

    folds: list[Fold] = []
    for f in range(n_splits):
        # np.flatnonzero(mask) = the positions where mask is True.
        train_pos = np.flatnonzero(train_fold_arr != f)
        eval_pos = np.flatnonzero(eval_fold_arr == f)
        # Defensive check, as in make_group_folds: trust, but verify.
        assert_no_group_overlap(
            {f"fold{f}_train": train_ids[train_pos], f"fold{f}_val": eval_ids[eval_pos]}
        )
        folds.append((train_pos, offset + eval_pos))
    return folds


def run(
    config: Mapping[str, Any],
    ks: Sequence[int | None],
    track: bool = False,
) -> dict[str, float]:
    """Cross-validate LogReg for every ``k`` on the same evaluation folds.

    Returns:
        Mean validation PR-AUC per ``k`` label, e.g. ``{"1": 0.09, "all": 0.11}``.
    """
    _, seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    bounds = class_bounds_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))

    # The fixed exam: one cut-off per truck, folds assigned per truck once.
    eval_frame = build_training_frame(readouts, tte, bounds, 1, seed)
    vehicle_fold = assign_vehicle_folds(eval_frame, n_splits, cv_seed)
    features = select_feature_columns(eval_frame)
    eval_prevalence = float(eval_frame[BINARY_COL].mean())
    logger.info(
        "Evaluation sample: %d trucks, positive rate %.4f (= PR-AUC of a dummy model)",
        len(eval_frame),
        eval_prevalence,
    )

    results: dict[str, float] = {}
    for k in ks:
        label = n_cutoffs_label(k)
        train_frame = build_training_frame(readouts, tte, bounds, k, seed)

        # Stack [training sample; evaluation sample] so that one x / y and one
        # list of folds can go into cross_validate. reset_index: positions and
        # labels must agree after the concat.
        columns = [*features, BINARY_COL]
        combined = pd.concat([train_frame[columns], eval_frame[columns]], ignore_index=True)
        folds = build_fixed_eval_folds(
            train_frame[VEHICLE_COL].reset_index(drop=True),
            eval_frame[VEHICLE_COL].reset_index(drop=True),
            vehicle_fold,
        )
        x = combined[features]
        y = combined[BINARY_COL].to_numpy()

        # k = all gives ~900k rows; every parallel worker copies its training
        # part during imputation/scaling. 2 workers keep RAM use safe.
        n_jobs = -1 if k is not None else 2
        logger.info("k=%s: %d training rows, n_jobs=%d", label, len(train_frame), n_jobs)
        result = cross_validate(
            build_logreg_pipeline(random_state=cv_seed),
            x,
            y,
            cv=folds,
            scoring=SCORING,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        val = result["test_pr_auc"]
        train = result["train_pr_auc"]
        results[label] = float(val.mean())
        logger.info(
            "k=%-4s pr_auc val %.4f +- %.4f | train %.4f | gap %+.4f | fit %.1f s | folds %s",
            label,
            val.mean(),
            val.std(),
            train.mean(),
            train.mean() - val.mean(),
            float(np.mean(result["fit_time"])),
            [round(float(s), 4) for s in val],
        )

        if track:
            log_cv_run(
                run_name=f"logreg_balanced_k{label}_fixed_eval",
                cv_result=result,
                metrics=SCORING,
                params={
                    "model": "logreg_balanced",
                    "n_cutoffs_per_vehicle": label,
                    "evaluation": "1 cut-off per truck, fixed folds",
                    "sampling_seed": seed,
                    "cv_n_splits": n_splits,
                    "cv_seed": cv_seed,
                    "n_train_rows": len(train_frame),
                    "n_eval_rows": len(eval_frame),
                    "n_features": len(features),
                    "eval_positive_rate": round(eval_prevalence, 5),
                },
                tags=RUN_TAGS,
            )
        del train_frame, combined, x  # free memory before the next (larger) k

    logger.info("Summary (val PR-AUC): %s", {k: round(v, 4) for k, v in results.items()})
    return results


# --- validation-like evaluation (05.11) ----------------------------------------------------
def sampling_cutoffs(
    name: str, labels: pd.DataFrame, targets: Sequence[float], seed: int
) -> pd.DataFrame:
    """The training cut-offs of one sampling strategy (rows of the label table).

    ``k5`` / ``k10``: uniform random cut-offs per truck; ``kall``: every known-label
    readout; ``ts70``: 5 random cut-offs from readouts with ``time_step >= 70``;
    ``matched``: 5 cut-offs per truck whose ``time_step`` follows ``targets``.

    Raises:
        ValueError: For an unknown sampling name.
    """
    if name == "k5":
        return sample_cutoffs(labels, 5, seed)
    if name == "k10":
        return sample_cutoffs(labels, 10, seed)
    if name == "kall":
        return known_labels(labels)
    if name == "ts70":
        return cutoffs_from_time_step(labels, MIN_TIME_STEP, 5, seed)
    if name == "matched":
        return matched_training_cutoffs(labels, targets, 5, seed)
    raise ValueError(f"unknown sampling {name!r}, choose one of {list(VL_SAMPLINGS)}")


def choose_sampling(fold_scores: Mapping[str, np.ndarray], n_rows: Mapping[str, int]) -> str:
    """Cheapest sampling that is not clearly worse than the best one.

    The rule is fixed before the run: take the sampling with the highest mean score;
    every sampling whose paired difference to it (best - candidate, per fold) is at most
    one standard deviation of those differences qualifies; among the qualifying ones the
    one with the fewest training rows wins (ties: higher mean score).

    Raises:
        ValueError: If ``fold_scores`` is empty.
    """
    if not fold_scores:
        raise ValueError("no sampling to choose from")
    best = max(fold_scores, key=lambda n: float(np.mean(fold_scores[n])))
    qualifying = []
    for name, scores in fold_scores.items():
        diff = np.asarray(fold_scores[best]) - np.asarray(scores)
        if name == best or float(diff.mean()) <= float(diff.std()):
            qualifying.append(name)
    return min(qualifying, key=lambda n: (n_rows[n], -float(np.mean(fold_scores[n]))))


def _fixed_eval_cv(
    pipeline: Any,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    columns: Sequence[str],
    vehicle_fold: pd.Series,
    n_jobs: int,
) -> dict[str, Any]:
    """Cross-validate: train on the training rows of the other folds, score on the eval rows.

    ``train_frame`` and ``eval_frame`` are stacked [train; eval]; the folds put the
    training rows of trucks outside fold f and the evaluation rows of trucks inside f
    together (``build_fixed_eval_folds`` checks that no truck is on both sides).
    """
    cols = [*columns, BINARY_COL]
    combined = pd.concat([train_frame[cols], eval_frame[cols]], ignore_index=True)
    folds = build_fixed_eval_folds(
        train_frame[VEHICLE_COL].reset_index(drop=True),
        eval_frame[VEHICLE_COL].reset_index(drop=True),
        vehicle_fold,
    )
    return cross_validate(
        pipeline, combined[list(columns)], combined[BINARY_COL].to_numpy(), cv=folds,
        scoring=SCORING_WITH_PROBA, n_jobs=n_jobs, return_train_score=True,
    )  # fmt: skip


def _make_pipeline(model: str, cols: Any, seed: int, threads: int) -> Any:
    """XGBoost (step-6 settings, raw + specs, no scaler) or the logistic-regression reference."""
    if model == "xgboost":
        from failsense.models.compare_models_cv import make_xgboost  # lazy: heavy import

        return build_model_pipeline(make_xgboost(seed, threads), cols.categorical, scaler=None)
    if model == "logreg":
        return build_logreg_pipeline(
            seed, categorical_columns=cols.categorical, log1p_columns=cols.log1p
        )
    raise ValueError(f"unknown model {model!r}, choose xgboost or logreg")


def _cores_plan(heavy: bool, n_splits: int) -> tuple[int, int]:
    """(parallel folds, threads per model); ``heavy`` (k = all) runs fewer folds at once."""
    cores = os.cpu_count() or 1
    folds_parallel = min(2 if heavy else n_splits, cores)
    return folds_parallel, max(1, cores // folds_parallel)


def _summarise(result: Mapping[str, Any], seconds: float) -> dict[str, float]:
    val, train = result["test_pr_auc"], result["train_pr_auc"]
    return {
        "pr_auc": float(val.mean()), "pr_auc_std": float(val.std()),
        "train": float(train.mean()), "gap": float(train.mean() - val.mean()),
        "mean_proba": float(result["test_mean_proba"].mean()), "seconds": seconds,
    }  # fmt: skip


def _log_paired(candidate: str, reference: str, cmp: Mapping[str, float]) -> None:
    logger.info(
        "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
        candidate,
        reference,
        cmp["mean_diff"],
        cmp["std_diff"],
        cmp["wins"],
        cmp["n_folds"],
    )


def _positive_rate_table(positives: Mapping[str, Any]) -> pd.DataFrame:
    """Flat name/value table of the positive-rate analysis (for the CSV in ``reports/``)."""
    overall = dict(positives["overall"])
    low, high = overall.pop("official_interval")
    values = {f"overall_{k}": v for k, v in overall.items()}
    values.update({"official_ci_low": low, "official_ci_high": high})
    values.update(positives["repaired_share"])
    return pd.DataFrame({"name": list(values), "value": list(values.values())})


def _sampling_table(model: str, summary: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """One row per run with the paired difference to its reference (k5 or the base sampling)."""
    rows = []
    for name, entry in summary.items():
        paired = entry.get("paired_vs_k5") or entry.get("paired_vs_base") or {}
        scalars = {k: v for k, v in entry.items() if not isinstance(v, dict)}
        rows.append(
            {
                "model": model,
                "name": name,
                **scalars,
                "diff_vs_reference": paired.get("mean_diff"),
                "diff_std": paired.get("std_diff"),
                "folds_won": paired.get("wins"),
            }
        )
    return pd.DataFrame(rows)


def _analyse_eval_set(
    eval_cut: pd.DataFrame,
    k5_cut: pd.DataFrame,
    random_cut: pd.DataFrame,
    targets: Sequence[float],
    tte: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """KS / quantiles against the official validation and the positive-rate analysis."""
    report = eval_set_report(
        {
            "validation-like eval set": eval_cut[TIME_COL],
            "old random eval (k=1)": random_cut[TIME_COL],
            "training cut-offs k=5": k5_cut[TIME_COL],
        },
        targets,
    )
    return report, positive_rate_report(eval_cut, k5_cut, tte)


def _log_eval_set(
    method: str,
    info: Mapping[str, float],
    report: pd.DataFrame,
    positives: Mapping[str, Any],
    k5_cut: pd.DataFrame,
    random_cut: pd.DataFrame,
) -> None:
    """Print the evaluation-set check; warn loudly if the positive rate is not realistic."""
    overall = positives["overall"]
    logger.info("[%s] evaluation set vs. official validation (time_step):\n%s", method,
                report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))  # fmt: skip
    logger.info("[%s] positive rate by time_step quartile of the evaluation set:\n%s", method,
                positives["by_quartile"].to_string(float_format=lambda v: f"{v:.4f}"))  # fmt: skip
    logger.info("[%s] repaired-truck shares: %s", method, positives["repaired_share"])
    if info:
        logger.info("[%s] calibration: %s", method, info)
    logger.info(
        "[%s] positive rate: evaluation set %.4f (%d of %d) | old CV evaluation (k=5 rows) %.4f | "
        "old random k=1 eval %.4f | official validation %.4f (95%% interval %.4f-%.4f)",
        method, overall["rate"], overall["positives"], overall["n"], k5_cut[BINARY_COL].mean(),
        random_cut[BINARY_COL].mean(), overall["official_rate"], *overall["official_interval"],
    )  # fmt: skip
    if overall["far_from_official"]:
        logger.warning(
            "[%s] EVALUATION SET POSITIVE RATE IS FAR FROM THE OFFICIAL VALIDATION RATE: %.4f vs. "
            "%.4f (x%.2f, outside the 95%% interval %.4f-%.4f)",
            method, overall["rate"], overall["official_rate"], overall["ratio_to_official"],
            *overall["official_interval"],
        )  # fmt: skip


def eval_set_only(
    config: Mapping[str, Any], methods: Sequence[str] = EVAL_METHODS
) -> dict[str, dict[str, Any]]:
    """Build and check the evaluation set(s) alone: KS and positive rate, no model is trained.

    Writes ``reports/eval_set_methods_q1.csv`` (one row per method), the positive-rate
    tables of the first method and the overlay figure of all methods.

    Raises:
        ValueError: For an unknown method.
    """
    unknown = [m for m in methods if m not in EVAL_METHODS]
    if unknown:
        raise ValueError(f"unknown eval method(s) {unknown}; choose from {list(EVAL_METHODS)}")
    _, seed = sampling_params_from_config(config)
    bounds = class_bounds_from_config(config)
    readouts = read_table(table_path(config, "train", "readouts"), columns=[VEHICLE_COL, TIME_COL])
    tte = read_table(table_path(config, "train", "tte"))
    labels = build_labels(readouts, tte, bounds)
    targets = validation_time_steps(config)
    repaired = tte.set_index(VEHICLE_COL)["in_study_repair"]
    k5_cut = sample_cutoffs(labels, 5, seed)
    random_cut = sample_cutoffs(labels, 1, seed)

    out: dict[str, dict[str, Any]] = {}
    rows, curves = [], {}
    for method in methods:
        cut, info = build_eval_cutoffs_by_method(method, labels, targets, repaired, seed)
        report, positives = _analyse_eval_set(cut, k5_cut, random_cut, targets, tte)
        _log_eval_set(method, info, report, positives, k5_cut, random_cut)
        overall = positives["overall"]
        ks = float(report["ks_vs_validation"].iloc[0])
        out[method] = {"cut": cut, "info": info, "report": report, "positives": positives, "ks": ks}
        curves[f"eval set: {method}"] = cut[TIME_COL]
        rows.append(
            {
                "method": method, "n": overall["n"], "ks_vs_validation": ks,
                "positive_rate": overall["rate"], "official_rate": overall["official_rate"],
                "ratio_to_official": overall["ratio_to_official"],
                "far_from_official": overall["far_from_official"],
                "median_time_step": float(cut[TIME_COL].median()),
                "repaired_trucks_positive": positives["repaired_share"][
                    "positive_rate_in_repaired_trucks_eval"],
                **{f"calibration_{k}": v for k, v in info.items()},
            }
        )  # fmt: skip
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(reports / "eval_set_methods_q1.csv", index=False, float_format="%.5f")
    for method, entry in out.items():
        entry["positives"]["by_quartile"].to_csv(
            reports / f"eval_set_q1_positive_by_quartile_{method}.csv", float_format="%.5f"
        )
    plot_eval_set(
        {**curves, "old random eval (k=1)": random_cut[TIME_COL]},
        targets,
        reports / "figures" / "eval_set_time_step.png",
    )
    logger.info("Evaluation sets side by side:\n%s",
                table.drop(columns=[c for c in table if c.startswith("calibration_")]).to_string(
                    index=False, float_format=lambda v: f"{v:.4f}"))  # fmt: skip
    return out


def run_validation_like(
    config: Mapping[str, Any],
    model: str = "xgboost",
    samplings: Sequence[str] = VL_SAMPLINGS,
    track: bool = False,
    time_one_fold: bool = False,
    eval_method: str = "calibrated",
) -> dict[str, Any]:
    """Compare training samplings on the fixed validation-like evaluation set.

    Args:
        config: Project config.
        model: ``"xgboost"`` (step-6 settings) or ``"logreg"``.
        samplings: Subset of ``VL_SAMPLINGS``.
        eval_method: ``"calibrated"`` (default), ``"literal"`` or ``"rank"``, see
            ``failsense.models.eval_set``.
        track: Log every model as an MLflow run (``setup_mlflow`` must have been called).
        time_one_fold: Only fit and score fold 0 of every sampling, print the projected
            runtime, then stop (no table, no MLflow).

    Returns:
        ``eval_set`` (report, positive-rate analysis), ``samplings`` (score table),
        ``best_sampling``, ``rechecks`` and ``old_vs_new`` entries.
    """
    unknown = [n for n in samplings if n not in VL_SAMPLINGS]
    if unknown:
        raise ValueError(f"unknown sampling(s) {unknown}; choose from {list(VL_SAMPLINGS)}")
    _, seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    bounds = class_bounds_from_config(config)
    top_fraction = features_params_from_config(config)

    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    specs = read_table(table_path(config, "train", "specifications"))
    labels = build_labels(readouts, tte, bounds)
    targets = validation_time_steps(config)  # time_step only; no validation labels

    def frame(cutoffs: pd.DataFrame, engineered: bool = False) -> pd.DataFrame:
        built = frame_from_sampled(
            readouts, cutoffs,
            counter_cols=counter_columns() if engineered else (),
            hist_groups=histogram_groups() if engineered else None,
            top_fraction=top_fraction,
        )  # fmt: skip
        return attach_specifications(built, specs)

    # --- the fixed evaluation set and the fold assignment ------------------------------
    repaired = tte.set_index(VEHICLE_COL)["in_study_repair"]
    eval_cut, eval_info = build_eval_cutoffs_by_method(eval_method, labels, targets, repaired, seed)
    k5_cut = sample_cutoffs(labels, 5, seed)
    random_cut = sample_cutoffs(labels, 1, seed)  # the old fixed evaluation (random row)
    eval_frame, k5_frame = frame(eval_cut), frame(k5_cut)
    old_folds = make_group_folds(k5_frame, n_splits=n_splits, seed=cv_seed)  # same as `make cv`
    vehicle_fold = vehicle_fold_from_folds(k5_frame, old_folds)
    cols = split_feature_columns(k5_frame)
    columns = cols.raw + cols.categorical

    report, positives = _analyse_eval_set(eval_cut, k5_cut, random_cut, targets, tte)
    _log_eval_set(eval_method, eval_info, report, positives, k5_cut, random_cut)
    overall = positives["overall"]

    # --- timing mode ----------------------------------------------------------------------
    if time_one_fold:
        out: dict[str, Any] = {}
        for name in samplings:
            heavy = name == "kall"
            folds_parallel, threads = _cores_plan(heavy, n_splits)
            train_frame = frame(sampling_cutoffs(name, labels, targets, seed))
            cols_t = [*columns, BINARY_COL]
            combined = pd.concat([train_frame[cols_t], eval_frame[cols_t]], ignore_index=True)
            folds = build_fixed_eval_folds(
                train_frame[VEHICLE_COL].reset_index(drop=True),
                eval_frame[VEHICLE_COL].reset_index(drop=True),
                vehicle_fold,
            )
            train_idx, val_idx = folds[0]
            pipeline = _make_pipeline(model, cols, cv_seed, threads)
            t0 = time.perf_counter()
            pipeline.fit(
                combined[columns].iloc[train_idx], combined[BINARY_COL].to_numpy()[train_idx]
            )
            t_fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            pipeline.predict_proba(combined[columns].iloc[val_idx])
            t_pred = time.perf_counter() - t1
            per_fold = t_fit + t_pred
            rounds = -(-n_splits // folds_parallel)  # ceil: how many waves of parallel folds
            out[name] = {"rows": len(train_frame), "fold_s": per_fold,
                         "projected_total_s": per_fold * rounds}  # fmt: skip
            logger.info(
                "%-8s one fold: %d training rows, fit %.1f s + predict %.1f s = %.1f s -> "
                "~%.0f s for %d folds (%d in parallel, %d threads each; plus training-score "
                "predictions and data building)",
                name, len(train_frame), t_fit, t_pred, per_fold, per_fold * rounds, n_splits,
                folds_parallel, threads,
            )  # fmt: skip
            del train_frame, combined
        return out

    # --- stage 1: training samplings on the fixed evaluation set ---------------------------
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    common = {
        "model": model, "evaluation": f"validation-like ({eval_method})", "cv_n_splits": n_splits,
        "cv_seed": cv_seed, "sampling_seed": seed, "n_eval_rows": len(eval_frame),
        "eval_positive_rate": round(overall["rate"], 5),
        "eval_ks_vs_validation": round(float(report["ks_vs_validation"].iloc[0]), 4),
        "n_numeric_features": len(cols.raw), "n_categorical_features": len(cols.categorical),
    }  # fmt: skip
    summary: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, np.ndarray] = {}
    n_rows: dict[str, int] = {}

    def record(
        name: str, result: Mapping[str, Any], seconds: float, extra: Mapping[str, Any]
    ) -> None:
        summary[name] = {**_summarise(result, seconds), **extra}
        fold_scores[name] = result["test_pr_auc"]
        s = summary[name]
        logger.info(
            "%-22s val %.4f +- %.4f | train %.4f | gap %+.4f | mean proba %.4f | %s | %.0f s",
            name, s["pr_auc"], s["pr_auc_std"], s["train"], s["gap"], s["mean_proba"],
            ", ".join(f"{k} {v}" for k, v in extra.items()), seconds,
        )  # fmt: skip
        if track:
            log_cv_run(
                run_name=f"{model}_{name}_validation_like_{eval_method}",
                cv_result=result, metrics=SCORING_WITH_PROBA,
                params={**common, "sampling": name, **{k: str(v) for k, v in extra.items()}},
                tags={**VL_RUN_TAGS, "eval_method": eval_method},
            )  # fmt: skip

    train_frames: dict[str, pd.DataFrame] = {}
    for name in samplings:
        heavy = name == "kall"
        folds_parallel, threads = _cores_plan(heavy, n_splits)
        cutoffs = sampling_cutoffs(name, labels, targets, seed)
        train_frame = frame(cutoffs)
        ks_train = ks_distance_to(train_frame[TIME_COL], targets)
        n_rows[name] = len(train_frame)
        started = time.perf_counter()
        result = _fixed_eval_cv(
            _make_pipeline(model, cols, cv_seed, threads), train_frame, eval_frame, columns,
            vehicle_fold, folds_parallel,
        )  # fmt: skip
        record(
            name, result, time.perf_counter() - started,
            {"train_rows": len(train_frame), "ks_train_vs_validation": round(ks_train, 3),
             "train_median_time_step": round(float(train_frame[TIME_COL].median()), 1),
             "train_positive_rate": round(float(train_frame[BINARY_COL].mean()), 4)},
        )  # fmt: skip
        if not heavy:
            train_frames[name] = train_frame
        del train_frame

    for name in summary:
        if name != "k5" and "k5" in fold_scores:
            cmp = paired_comparison(fold_scores[name], fold_scores["k5"])
            summary[name]["paired_vs_k5"] = cmp
            _log_paired(name, "k5", cmp)

    best = choose_sampling(fold_scores, n_rows)
    logger.info(
        "Best sampling by the pre-set rule (cheapest not clearly worse than the best): %s", best
    )

    # --- stage 2: time_step and engineered features for the best sampling -------------------
    folds_parallel, threads = _cores_plan(best == "kall", n_splits)
    best_cut = sampling_cutoffs(best, labels, targets, seed)
    best_train = train_frames.get(best)
    if best_train is None:
        best_train = frame(best_cut)
    rechecks: dict[str, dict[str, float]] = {}
    variants = {
        "no_time_step": (best_train, eval_frame, [c for c in columns if c != TIME_COL]),
    }
    eng_train, eng_eval = frame(best_cut, engineered=True), frame(eval_cut, engineered=True)
    cols_eng = split_feature_columns(eng_train)
    variants["engineered"] = (
        eng_train,
        eng_eval,
        cols_eng.raw + cols_eng.engineered + cols_eng.categorical,
    )
    for suffix, (tr, ev, variant_columns) in variants.items():
        started = time.perf_counter()
        result = _fixed_eval_cv(
            _make_pipeline(model, cols, cv_seed, threads), tr, ev, variant_columns,
            vehicle_fold, folds_parallel,
        )  # fmt: skip
        name = f"{best}__{suffix}"
        record(name, result, time.perf_counter() - started, {"base_sampling": best})
        cmp = paired_comparison(fold_scores[name], fold_scores[best])
        rechecks[suffix] = cmp
        summary[name]["paired_vs_base"] = cmp
        _log_paired(name, best, cmp)

    # --- how much does the evaluation choice change the score? -------------------------------
    old_vs_new: dict[str, dict[str, float]] = {}
    for kind in ("xgboost", "logreg"):
        _, kind_threads = _cores_plan(False, n_splits)
        pipe = _make_pipeline(kind, cols, cv_seed, kind_threads)
        old = cross_validate(
            pipe, k5_frame[columns], k5_frame[BINARY_COL].to_numpy(), cv=old_folds,
            scoring=SCORING_WITH_PROBA, n_jobs=min(n_splits, os.cpu_count() or 1),
        )  # fmt: skip
        if kind == model and "k5" in fold_scores:
            new_scores = fold_scores["k5"]
        else:
            new_scores = _fixed_eval_cv(
                _make_pipeline(kind, cols, cv_seed, kind_threads), k5_frame, eval_frame, columns,
                vehicle_fold, min(n_splits, os.cpu_count() or 1),
            )["test_pr_auc"]  # fmt: skip
        old_vs_new[kind] = {
            "old_cv_mean": float(old["test_pr_auc"].mean()),
            "old_cv_std": float(old["test_pr_auc"].std()),
            "validation_like_mean": float(new_scores.mean()),
            "validation_like_std": float(new_scores.std()),
        }
        o = old_vs_new[kind]
        logger.info(
            "%-8s k=5 training: old CV evaluation %.4f +- %.4f | validation-like %.4f +- %.4f",
            kind, o["old_cv_mean"], o["old_cv_std"], o["validation_like_mean"],
            o["validation_like_std"],
        )  # fmt: skip

    # --- write the reports -------------------------------------------------------------------
    # file names carry the evaluation method, so a second method never overwrites the first
    report.to_csv(reports / f"eval_set_q1_{eval_method}.csv", index=False, float_format="%.4f")
    positives["by_quartile"].to_csv(
        reports / f"eval_set_q1_positive_by_quartile_{eval_method}.csv", float_format="%.5f"
    )
    _positive_rate_table(positives).to_csv(
        reports / f"eval_set_q1_positive_rate_{eval_method}.csv", index=False, float_format="%.5f"
    )
    _sampling_table(model, summary).to_csv(
        reports / f"compare_sampling_q1_{eval_method}.csv", index=False, float_format="%.5f"
    )
    pd.DataFrame(old_vs_new).T.rename_axis("model").to_csv(
        reports / f"eval_old_vs_new_q1_{eval_method}.csv", float_format="%.5f"
    )
    plot_eval_set(
        {
            "validation-like eval set": eval_cut[TIME_COL],
            "old random eval (k=1)": random_cut[TIME_COL],
            "training cut-offs k=5": k5_cut[TIME_COL],
        },
        targets,
        reports / "figures" / f"eval_set_time_step_{eval_method}.png",
    )
    return {
        "eval_set": {"report": report, "positives": positives},
        "samplings": summary,
        "best_sampling": best,
        "rechecks": rechecks,
        "old_vs_new": old_vs_new,
    }


def ks_distance_to(values: pd.Series, targets: Sequence[float]) -> float:
    """KS statistic between a sample's ``time_step`` and the validation targets."""

    return ks_distance(values, targets)[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # nargs="+": one or more values, e.g. --ks 1 5 all
    parser.add_argument(
        "--ks",
        nargs="+",
        default=list(DEFAULT_KS),
        help="values of k to compare: integers or 'all' (default: 1 5 all)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="do not log to MLflow (quick local check)",
    )
    parser.add_argument(
        "--eval",
        choices=["random", "validation-like"],
        default="random",
        help="evaluation rows: one random cut-off per truck (old default) or the "
        "validation-like set (rank matching to the official validation time_step)",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "xgboost"],
        default=None,
        help="model of the validation-like comparison (default xgboost); the random "
        "evaluation always uses the logistic regression",
    )
    parser.add_argument(
        "--samplings",
        nargs="+",
        default=list(VL_SAMPLINGS),
        help=f"training samplings of the validation-like comparison, from {list(VL_SAMPLINGS)}",
    )
    parser.add_argument(
        "--eval-method",
        choices=[*EVAL_METHODS, "all"],
        default="calibrated",
        help="how the validation-like evaluation rows are chosen (default: calibrated rank "
        "matching; 'all' only with --eval-set-only)",
    )
    parser.add_argument(
        "--eval-set-only",
        action="store_true",
        help="validation-like mode: build and check the evaluation set(s) only (KS, positive "
        "rate), train no model",
    )
    parser.add_argument(
        "--time-one-fold",
        action="store_true",
        help="validation-like mode: fit/score fold 0 only and print the projected runtime",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()

    track = not args.no_mlflow and not args.time_one_fold and not args.eval_set_only
    if track:
        setup_mlflow(config)
    if args.eval == "validation-like":
        if args.eval_set_only:
            methods = EVAL_METHODS if args.eval_method == "all" else (args.eval_method,)
            eval_set_only(config, methods)
            return
        if args.eval_method == "all":
            parser.error("--eval-method all is only for --eval-set-only")
        run_validation_like(
            config, args.model or "xgboost", args.samplings, track=track,
            time_one_fold=args.time_one_fold, eval_method=args.eval_method,
        )  # fmt: skip
        return
    ks = [parse_n_cutoffs(value) for value in args.ks]
    run(config, ks, track=track)


if __name__ == "__main__":
    main()
