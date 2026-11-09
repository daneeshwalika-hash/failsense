"""One comparison table for several model families (Q1, same folds for everyone).

Run it with::

    uv run python -m failsense.models.compare_models_cv                 # or: make compare
    uv run python -m failsense.models.compare_models_cv --time-one-fold --models knn_k25_uniform
    uv run python -m failsense.models.compare_models_cv --models gaussian_nb tree_depth4
    uv run python -m failsense.models.compare_models_cv --no-mlflow

Every model sees exactly the same rows and the same 5 ``StratifiedGroupKFold`` folds by
``vehicle_id`` (the training split, ``k = 5`` cut-offs per truck, raw features +
specifications) and is scored with the same metrics, so the numbers are directly
comparable and every model is compared fold by fold with the reference. New families are
added in ``build_models`` and appear in the same table (boosting follows on 04.11).

The model families and why they are set up as they are
------------------------------------------------------
* **Reference: logistic regression** (specs, balanced, ``log1p``, L2, C = 1).
* **kNN** predicts the share of positive trucks among the k most similar trucks.
  "Similar" is a distance, so all features must have the same scale (``log1p`` first
  because of the heavy tails, then standardisation, both inside the Pipeline). The
  probability is a fraction with denominator k, so k must be large (25, 101); with a
  small k almost every probability is exactly 0 or 1. kNN has no class weights, so its
  probabilities stay near the true rate of 2.8 %. Its **training score is not
  computed**: every training point is its own nearest neighbour (distance 0), so the
  in-sample score would be meaningless and it would cost more time than the validation.
* **GaussianNB** multiplies the class prior with one Gaussian per feature, assuming the
  features are independent given the class. That is wrong here (neighbouring histogram
  bins are strongly correlated, one-hot columns are 0/1), so its probabilities are
  expected to be extreme. It has no class-weight parameter: it is generative, the prior
  defaults to the class frequency, and we leave it. ``log1p`` helps (heavy tails); a
  scaler would change nothing (each feature is standardised on its own).
* **Decision tree** with ``class_weight="balanced"``, ``max_depth`` in {4, 8, None} and
  ``min_samples_leaf = 100`` (with 2.8 % positives a leaf needs enough rows for a stable
  probability; unlimited depth with leaves of size 1 just memorises the training data). A
  split only asks "is the value above a threshold?", so any monotone transformation
  (scaling, ``log1p``) gives the same tree: none is used.

* **RandomForest, HistGradientBoosting, XGBoost, LightGBM** (04.11): the same inputs as the
  trees (median imputer + one-hot specs, no scaler, no ``log1p``). The three boosting
  models get the **same fixed budget** (learning rate 0.05, 300 rounds, trees of about 16-31
  leaves), not tuned, and **no class weights** at first. **No early stopping anywhere**:
  it needs validation data, and a random split of the training *rows* puts readouts of the
  same truck on both sides (leakage). scikit-learn's HistGradientBoosting turns early
  stopping on by default for more than 10,000 rows (``early_stopping="auto"``), so it is
  switched off explicitly. Tuning with a group-aware inner split comes in step 7.
* **Parallelism, one level:** the 5 folds run in parallel and every model gets
  ``cores // folds`` threads (``model_jobs``), so folds x threads <= cores. HistGradientBoosting
  has no ``n_jobs`` parameter (OpenMP); joblib's loky workers limit the threads of such
  libraries to the same share.

The best boosting model is then re-checked fold by fold with and without the specifications,
``time_step``, engineered features and a positive-class weight (neg / pos).

Importances of the best tree and of the best boosting model are reported, but they are
biased (impurity: towards features with many distinct values; gain: towards features used
early) and computed on the training data; SHAP (06.11) is the proper tool. Only the training
split is used.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL, TIME_COL, VEHICLE_COL
from failsense.data.sampling import sampling_params_from_config
from failsense.data.split import cv_params_from_config, make_group_folds
from failsense.models.baseline_cv import (
    SCORING_WITH_PROBA,
    load_training_frame,
    n_cutoffs_label,
    paired_comparison,
)
from failsense.models.pipeline import (
    FeatureColumns,
    build_logreg_pipeline,
    build_model_pipeline,
    column_kind,
    split_feature_columns,
)
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "comparison"}
# Tag of the runs of this table. v1 = the step-5 table (logreg, kNN, NB, trees); v2 adds
# the ensemble models. Old runs are kept; reports only look at the tag of the current set.
COMPARISON_SET = "v2"
REFERENCE = "logreg_reference"
TREE_MIN_SAMPLES_LEAF = 100
TOP_N = 10
BOOSTING = ("hist_gb", "xgboost", "lightgbm")  # the models of the re-checks
N_ROUNDS = 300  # fixed, no early stopping
LEARNING_RATE = 0.05


@dataclass(frozen=True)
class ModelSpec:
    """One row of the comparison table."""

    name: str
    family: str
    pipeline: Any
    params: Mapping[str, Any] = field(default_factory=dict)
    train_score: bool = True  # False for kNN (a point is its own neighbour)
    columns: tuple[str, ...] | None = None  # None = raw numeric + specification columns
    keep_estimators: bool = False  # keep the fitted fold models (importances)


def make_random_forest(random_state: int, jobs: int) -> RandomForestClassifier:
    """300 trees, balanced bootstrap classes, leaves of >= 50 rows (stable probabilities)."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced_subsample",  # weights from each tree's own bootstrap sample
        n_jobs=jobs,
        random_state=random_state,
    )


def make_hist_gb(
    random_state: int, jobs: int = 1, positive_weight: float | None = None
) -> HistGradientBoostingClassifier:
    """scikit-learn histogram boosting, 300 rounds, no early stopping.

    ``early_stopping=False`` is essential: the default ``"auto"`` switches it on for more
    than 10,000 rows and uses a random 10 % of the *training rows* as validation data,
    which puts readouts of the same truck on both sides. ``jobs`` is ignored (OpenMP,
    no parameter). ``positive_weight`` is not None -> ``class_weight="balanced"``.
    """
    return HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=N_ROUNDS,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        early_stopping=False,
        class_weight="balanced" if positive_weight is not None else None,
        random_state=random_state,
    )


def make_xgboost(
    random_state: int, jobs: int, positive_weight: float | None = None
) -> XGBClassifier:
    """XGBoost, 300 rounds of depth-4 trees, no early stopping, gain-based importances."""
    return XGBClassifier(
        n_estimators=N_ROUNDS,
        learning_rate=LEARNING_RATE,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        importance_type="gain",
        scale_pos_weight=1.0 if positive_weight is None else positive_weight,
        n_jobs=jobs,
        random_state=random_state,
        verbosity=0,
    )


def make_lightgbm(
    random_state: int, jobs: int, positive_weight: float | None = None
) -> LGBMClassifier:
    """LightGBM, 300 rounds of up to 31 leaves, no early stopping, gain-based importances."""
    return LGBMClassifier(
        n_estimators=N_ROUNDS,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,  # subsample is only used if subsample_freq > 0
        colsample_bytree=0.8,
        importance_type="gain",
        scale_pos_weight=1.0 if positive_weight is None else positive_weight,
        n_jobs=jobs,
        random_state=random_state,
        deterministic=True,
        verbose=-1,
    )


def make_mlp(random_state: int) -> MLPClassifier:
    """Small neural network: two hidden layers (64, 32), ReLU, Adam, L2 penalty, early stopping.

    ``early_stopping=True`` holds out a random 10 % of the *training rows* and stops after 10
    epochs without improvement of the validation score. That inner split is by row, not by truck
    (``MLPClassifier`` cannot take groups), so readouts of the same truck are on both sides and the
    stopping point is chosen a little optimistically. The outer cross-validation (the reported
    PR-AUC) is still grouped by truck. There are no class weights: ``MLPClassifier`` has none; the
    probabilities are uncalibrated like those of the boosting models.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,  # L2 penalty
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )


BOOSTING_FACTORIES = {"hist_gb": make_hist_gb, "xgboost": make_xgboost, "lightgbm": make_lightgbm}


def build_models(
    cols: FeatureColumns, random_state: int, model_jobs: int = 1
) -> dict[str, ModelSpec]:
    """All model families of the comparison, in table order (the reference first).

    Args:
        cols: Column groups from ``split_feature_columns``.
        random_state: Seed of every model that uses randomness.
        model_jobs: Threads of each kNN / forest / boosting model (the folds already run
            in parallel, so this is the share of cores per fold).
    """
    cat, log1p = cols.categorical, cols.log1p
    specs: list[ModelSpec] = [
        ModelSpec(
            REFERENCE,
            "logreg",
            build_logreg_pipeline(random_state, categorical_columns=cat, log1p_columns=log1p),
            {"penalty": "l2", "C": 1.0, "class_weight": "balanced"},
        )
    ]
    for k, weights in ((25, "uniform"), (101, "uniform"), (101, "distance")):
        specs.append(
            ModelSpec(
                f"knn_k{k}_{weights}",
                "knn",
                # log1p, then standard scaling: distances need comparable scales.
                build_model_pipeline(
                    KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=model_jobs),
                    cat,
                    scaler="standard",
                    log1p_columns=log1p,
                ),
                {"n_neighbors": k, "weights": weights},
                train_score=False,
            )
        )
    specs.append(
        ModelSpec(
            "gaussian_nb",
            "naive_bayes",
            # log1p for the heavy tails; no scaler (NB standardises each feature itself).
            build_model_pipeline(GaussianNB(), cat, scaler=None, log1p_columns=log1p),
            {"priors": "class frequency (default)"},
        )
    )
    for depth in (4, 8, None):
        specs.append(
            ModelSpec(
                f"tree_depth{depth}",
                "tree",
                # No scaler and no log1p: a tree only compares values with thresholds.
                build_model_pipeline(
                    DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                    cat,
                    scaler=None,
                ),
                {
                    "max_depth": "None" if depth is None else depth,
                    "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
                    "class_weight": "balanced",
                },
                keep_estimators=True,
            )
        )
    # Ensembles: median imputer + one-hot specs, nothing else (trees are invariant).
    specs.append(
        ModelSpec(
            "random_forest",
            "forest",
            build_model_pipeline(make_random_forest(random_state, model_jobs), cat, scaler=None),
            {"n_estimators": 300, "min_samples_leaf": 50, "class_weight": "balanced_subsample"},
            keep_estimators=True,
        )
    )
    specs.append(
        ModelSpec(
            "mlp",
            "neural_network",
            # log1p for the heavy tails, then standard scaling: a network needs inputs on one scale.
            build_model_pipeline(
                make_mlp(random_state), cat, scaler="standard", log1p_columns=log1p
            ),
            {
                "hidden_layers": "64-32",
                "alpha": 1e-3,
                "early_stopping": True,
                "class_weight": "none",
            },
        )
    )
    for name, factory in BOOSTING_FACTORIES.items():
        specs.append(
            ModelSpec(
                name,
                "boosting",
                build_model_pipeline(factory(random_state, model_jobs), cat, scaler=None),
                {
                    "n_rounds": N_ROUNDS,
                    "learning_rate": LEARNING_RATE,
                    "class_weight": "none",
                    "early_stopping": False,
                },  # fmt: skip
                keep_estimators=True,
            )
        )
    return {s.name: s for s in specs}


RECHECKS = ("no_specs", "no_time_step", "engineered", "pos_weight")


def build_rechecks(
    best: str,
    cols: FeatureColumns,
    random_state: int,
    model_jobs: int,
    positive_weight: float,
) -> dict[str, ModelSpec]:
    """Variants of the best boosting model for the paired re-checks (same folds).

    * ``no_specs``: the one-hot specification columns are dropped.
    * ``no_time_step``: ``time_step`` is dropped from the raw columns.
    * ``engineered``: the engineered counter / histogram features are added.
    * ``pos_weight``: the positive class is weighted by ``positive_weight`` (neg / pos).

    Raises:
        ValueError: If ``best`` is not one of ``BOOSTING``.
    """
    if best not in BOOSTING_FACTORIES:
        raise ValueError(f"unknown boosting model {best!r}, choose one of {list(BOOSTING)}")
    factory = BOOSTING_FACTORIES[best]
    raw, cat = list(cols.raw), list(cols.categorical)

    def variant(
        suffix: str, columns: list[str], categorical: Sequence[str], weight: float | None = None
    ) -> ModelSpec:
        pipeline = build_model_pipeline(
            factory(random_state, model_jobs, weight), categorical, scaler=None
        )
        return ModelSpec(
            f"{best}__{suffix}",
            "boosting",
            pipeline,
            {"base_model": best, "recheck": suffix, "positive_weight": weight or 1.0},
            columns=tuple(columns + list(categorical)),
        )

    variants = [
        variant("no_specs", raw, ()),
        variant("no_time_step", [c for c in raw if c != TIME_COL], cat),
        variant("engineered", raw + list(cols.engineered), cat),
        variant("pos_weight", raw, cat, positive_weight),
    ]
    return {v.name: v for v in variants}


def aggregate_importances(estimators: Sequence[Any]) -> pd.DataFrame:
    """Importances of fitted tree-based pipelines, averaged over the folds.

    The importance is whatever the model reports (impurity decrease for trees and forests,
    gain for XGBoost / LightGBM); each fold model is normalised to sum to 1 so that the
    folds and models are comparable.

    Returns:
        One row per feature, sorted by ``mean_importance`` (descending), with
        ``folds_nonzero`` (fold models that use the feature at all), ``n_folds`` and
        ``kind``. A feature missing in a fold counts as 0 there.

    Raises:
        ValueError: If ``estimators`` is empty or a pipeline has no fitted tree.
    """
    if not estimators:
        raise ValueError("need at least one fitted pipeline")
    series = []
    for est in estimators:
        tree = est.named_steps["model"]
        if not hasattr(tree, "feature_importances_"):
            raise ValueError("pipeline needs a fitted tree with feature_importances_")
        names = est.named_steps["preprocess"].get_feature_names_out()
        values = np.asarray(tree.feature_importances_, dtype=float)
        total = values.sum()
        series.append(pd.Series(values / total if total > 0 else values, index=names))
    table = pd.concat(series, axis=1).fillna(0.0)
    out = pd.DataFrame(
        {
            "mean_importance": table.mean(axis=1),
            "folds_nonzero": (table > 0).sum(axis=1),
            "n_folds": len(series),
            "kind": [column_kind(n, scaled=False) for n in table.index],  # trees: no scaler
        }
    )
    out.index.name = "feature"
    return out.sort_values("mean_importance", ascending=False).reset_index()


def best_by_mean(scores: Mapping[str, np.ndarray]) -> str:
    """Name with the highest mean fold score (first one wins a tie)."""
    if not scores:
        raise ValueError("no scores to choose from")
    return max(scores, key=lambda name: float(np.mean(scores[name])))


def select_models(
    models: Mapping[str, ModelSpec], names: Sequence[str] | None
) -> dict[str, ModelSpec]:
    """Keep only ``names`` (``None`` = all); unknown names raise ``ValueError``."""
    if not names:
        return dict(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(models)}")
    return {n: models[n] for n in names}


def _worker_split(n_splits: int) -> tuple[int, int]:
    """Parallel folds and threads per model so that folds x threads <= cores."""
    cores = os.cpu_count() or 1
    folds_parallel = min(n_splits, cores)
    return folds_parallel, max(1, cores // folds_parallel)


def time_one_fold(
    config: Mapping[str, Any], names: Sequence[str] | None
) -> dict[str, dict[str, float]]:
    """Fit and score fold 0 only and project the total runtime (no MLflow, no table).

    Returns per model: seconds for one fold (fit + predicting the validation part, no
    training score), the projected total when the 5 folds run in parallel (about the
    time of one fold if enough cores are free) and when they run one after another.
    """
    n_per_vehicle, _ = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    cols = split_feature_columns(frame)
    _, model_jobs = _worker_split(n_splits)
    models = select_models(build_models(cols, cv_seed, model_jobs), names)
    x = frame[cols.raw + cols.categorical]
    y = frame[BINARY_COL].to_numpy()
    train_idx, val_idx = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)[0]
    out: dict[str, dict[str, float]] = {}
    for name, spec in models.items():
        model = clone(spec.pipeline)
        t0 = time.perf_counter()
        model.fit(x.iloc[train_idx], y[train_idx])
        t_fit = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = model.predict_proba(x.iloc[val_idx])[:, 1]
        t_pred = time.perf_counter() - t1
        total = t_fit + t_pred
        out[name] = {"fit_s": t_fit, "predict_s": t_pred, "fold_s": total,
                     "projected_parallel_s": total, "projected_sequential_s": total * n_splits,
                     "mean_proba": float(proba.mean())}  # fmt: skip
        logger.info(
            "%-20s one fold: fit %.1f s + predict %.1f s = %.1f s -> total ~%.0f s if the %d "
            "folds run in parallel (%d threads per model), ~%.0f s one after another",
            name, t_fit, t_pred, total, total, n_splits, model_jobs, total * n_splits,
        )  # fmt: skip
    return out


def permutation_top(
    pipeline: Any, x_val: pd.DataFrame, y_val: np.ndarray, random_state: int, jobs: int,
    n_repeats: int = 3,
) -> pd.DataFrame:  # fmt: skip
    """Permutation importance of the raw input columns (PR-AUC drop) on one validation fold.

    For models without ``feature_importances_`` (HistGradientBoosting). Every input column
    is shuffled ``n_repeats`` times and the drop of the PR-AUC is measured; this needs
    validation labels, so it is an analysis of a fitted model, never used to choose one.
    """
    result = permutation_importance(
        pipeline, x_val, y_val, scoring="average_precision", n_repeats=n_repeats,
        random_state=random_state, n_jobs=jobs,
    )  # fmt: skip
    out = pd.DataFrame(
        {
            "feature": list(x_val.columns),
            "mean_importance": result.importances_mean,
            "folds_nonzero": (result.importances > 0).sum(axis=1),
            "n_folds": n_repeats,
            "kind": ["raw input column"] * x_val.shape[1],
        }
    )
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def _importance_method(family: str, name: str) -> str:
    """Words for the report: which kind of importance a model reports."""
    if name == "hist_gb":
        return "permutation importance (PR-AUC drop on the validation part of fold 0, 3 repeats)"
    if name in ("xgboost", "lightgbm") or name.split("__")[0] in ("xgboost", "lightgbm"):
        return "gain-based importance (share of the total gain, mean over the fold models)"
    return "impurity-based importance (mean over the fold models)"


def write_importances(top: pd.DataFrame, model: str, method: str, reports: Path, stem: str) -> None:
    """Write a top-N importance table as CSV and Markdown (with the bias warning)."""
    top = top.head(TOP_N)
    top.assign(model=model).to_csv(reports / f"{stem}.csv", index=False, float_format="%.5f")
    lines = [
        f"# Top {TOP_N} importances of {model}",
        "",
        f"Method: {method}.",
        "",
        "**Biased:** impurity and gain importances favour features with many distinct values "
        "(counters over one-hot columns) and features used early; they are computed on "
        "training data. They are pointers, not explanations: SHAP (06.11) is the proper tool.",
        "",
        "| rank | feature | kind | importance | non-zero in folds |",
        "|---|---|---|---|---|",
    ]
    for rank, r in enumerate(top.itertuples(), start=1):
        lines.append(
            f"| {rank} | {r.feature} | {r.kind} | {r.mean_importance:.4f} | "
            f"{r.folds_nonzero}/{r.n_folds} |"
        )
    (reports / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(summary: Mapping[str, Mapping[str, Any]], reports: Path) -> None:
    """Write the comparison table (base models) and the re-check table (variants)."""
    reports.mkdir(parents=True, exist_ok=True)
    main_rows, recheck_rows = [], []
    for name, s in summary.items():
        paired = s.get("paired", {})
        vs_ref = paired.get(REFERENCE, {})
        row = {
            "model": name, "family": s["family"], "pr_auc": s["pr_auc"],
            "pr_auc_std": s["pr_auc_std"], "train": s["train"], "gap": s["gap"],
            "mean_proba": s["mean_proba"], "seconds": s["seconds"],
            "diff_vs_reference": vs_ref.get("mean_diff", np.nan),
            "diff_std": vs_ref.get("std_diff", np.nan),
            "folds_won": vs_ref.get("wins", np.nan),
        }  # fmt: skip
        base = s.get("recheck_of")
        if base is None:
            main_rows.append(row)
        else:
            vs_base = paired.get(base, {})
            recheck_rows.append(
                {
                    **row,
                    "base_model": base,
                    "diff_vs_base": vs_base.get("mean_diff", np.nan),
                    "diff_vs_base_std": vs_base.get("std_diff", np.nan),
                    "folds_won_vs_base": vs_base.get("wins", np.nan),
                }  # fmt: skip
            )
    pd.DataFrame(main_rows).to_csv(
        reports / "compare_models_q1.csv", index=False, float_format="%.5f"
    )
    if recheck_rows:
        pd.DataFrame(recheck_rows).to_csv(
            reports / "compare_models_q1_recheck.csv", index=False, float_format="%.5f"
        )


def run(
    config: Mapping[str, Any],
    names: Sequence[str] | None = None,
    track: bool = False,
    comparison_set: str = COMPARISON_SET,
    rechecks: bool = True,
) -> dict[str, dict[str, Any]]:
    """Cross-validate the model families, re-check the best boosting model, write the tables.

    Args:
        config: Project config.
        names: Models to run (``None`` = all). The reference is always included.
        track: Log every model as an MLflow run (``setup_mlflow`` must have been called).
        comparison_set: MLflow tag ``comparison_set`` of the runs of this table; the
            artifacts are attached to the run with this tag, so older sets are untouched.
        rechecks: Run the paired re-checks of the best boosting model (needs at least one
            boosting model among ``names``).

    Returns:
        Per model: ``pr_auc`` mean/std, ``train`` and ``gap`` (``NaN`` for kNN),
        ``mean_proba``, ``seconds``, the paired comparison with the reference (and, for
        re-check variants, with their base model) and a ``_report`` entry.
    """
    n_per_vehicle, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    cols = split_feature_columns(frame)
    y = frame[BINARY_COL].to_numpy()
    default_columns = cols.raw + cols.categorical
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    folds_parallel, model_jobs = _worker_split(n_splits)
    all_models = build_models(cols, cv_seed, model_jobs)
    chosen = select_models(all_models, names)
    if REFERENCE not in chosen:  # every paired comparison needs it
        chosen = {REFERENCE: all_models[REFERENCE], **chosen}

    common = {
        "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle),
        "sampling_seed": sampling_seed,
        "cv_n_splits": n_splits,
        "cv_seed": cv_seed,
        "n_rows": len(frame),
        "n_vehicles": int(frame[VEHICLE_COL].nunique()),
        "positive_rate": round(float(y.mean()), 5),
        "comparison_set": comparison_set,
    }
    summary: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, np.ndarray] = {}
    fitted: dict[str, list[Any]] = {}

    def evaluate(spec: ModelSpec) -> None:
        columns = list(spec.columns) if spec.columns is not None else default_columns
        started = time.perf_counter()
        result = cross_validate(
            spec.pipeline, frame[columns], y, cv=folds, scoring=SCORING_WITH_PROBA,
            n_jobs=folds_parallel, return_train_score=spec.train_score,
            return_estimator=spec.keep_estimators,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        val = result["test_pr_auc"]
        train = float(result["train_pr_auc"].mean()) if spec.train_score else float("nan")
        fold_scores[spec.name] = val
        if spec.keep_estimators:
            fitted[spec.name] = result["estimator"]
        summary[spec.name] = {
            "family": spec.family,
            "pr_auc": float(val.mean()),
            "pr_auc_std": float(val.std()),
            "train": train,
            "gap": train - float(val.mean()),
            "mean_proba": float(result["test_mean_proba"].mean()),
            "seconds": seconds,
            "fit_seconds_sum": float(np.sum(result["fit_time"])),
            "n_columns": len(columns),
        }
        logger.info(
            "%-26s val %.4f +- %.4f | train %s | gap %s | mean proba %.4f | %.0f s",
            spec.name, val.mean(), val.std(),
            "n/a" if np.isnan(train) else f"{train:.4f}",
            "n/a" if np.isnan(train) else f"{train - val.mean():+.4f}",
            result["test_mean_proba"].mean(), seconds,
        )  # fmt: skip
        if track:
            tags = {**RUN_TAGS, "family": spec.family, "comparison_set": comparison_set}
            if "recheck" in spec.params:
                tags.update({"recheck": str(spec.params["recheck"]),
                             "base_model": str(spec.params["base_model"])})  # fmt: skip
            log_cv_run(
                run_name=f"{spec.name}_k{n_cutoffs_label(n_per_vehicle)}",
                cv_result=result,
                metrics=SCORING_WITH_PROBA,
                params={**common, "model": spec.name, "family": spec.family,
                        "n_columns": len(columns), **spec.params},
                tags=tags,
            )  # fmt: skip

    def paired(candidate: str, reference: str) -> None:
        cmp = paired_comparison(fold_scores[candidate], fold_scores[reference])
        summary[candidate].setdefault("paired", {})[reference] = cmp
        logger.info(
            "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
            candidate, reference, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    for spec in chosen.values():
        evaluate(spec)
    for name in chosen:
        if name != REFERENCE:
            paired(name, REFERENCE)

    # --- paired re-checks of the best boosting model ------------------------------------
    best_boosting, positive_weight = None, None
    boosting_scores = {n: fold_scores[n] for n in chosen if n in BOOSTING}
    if rechecks and boosting_scores:
        best_boosting = best_by_mean(boosting_scores)
        positive_weight = float((1 - y.mean()) / y.mean())  # neg / pos, about 35
        logger.info(
            "Best boosting model: %s. Re-checks on the same folds (positive weight %.1f, the "
            "overall neg/pos ratio of the training labels, not fitted per fold)",
            best_boosting, positive_weight,
        )  # fmt: skip
        variants = build_rechecks(best_boosting, cols, cv_seed, model_jobs, positive_weight)
        for variant in variants.values():
            evaluate(variant)
            summary[variant.name]["recheck_of"] = best_boosting
            paired(variant.name, best_boosting)
            paired(variant.name, REFERENCE)

    # --- importances ----------------------------------------------------------------------
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    tree_scores = {n: fold_scores[n] for n in chosen if chosen[n].family == "tree"}
    best_tree = best_by_mean(tree_scores) if tree_scores else None
    if best_tree is not None:
        top = aggregate_importances(fitted[best_tree])
        method = _importance_method("tree", best_tree)
        write_importances(top, best_tree, method, reports, "compare_models_q1_tree_importances")
        written.append("compare_models_q1_tree_importances.csv")
        logger.info("Top %d importances of %s (%s):\n%s", TOP_N, best_tree, method,
                    top.head(TOP_N).to_string(index=False))  # fmt: skip
    if best_boosting is not None:
        estimators = fitted[best_boosting]
        if hasattr(estimators[0].named_steps["model"], "feature_importances_"):
            top = aggregate_importances(estimators)
        else:  # HistGradientBoosting has no built-in importances
            val_idx = folds[0][1]
            top = permutation_top(
                estimators[0], frame[default_columns].iloc[val_idx], y[val_idx],
                random_state=cv_seed, jobs=folds_parallel,
            )  # fmt: skip
        method = _importance_method("boosting", best_boosting)
        write_importances(
            top, best_boosting, method, reports, "compare_models_q1_boosting_importances"
        )
        written.append("compare_models_q1_boosting_importances.csv")
        logger.info("Top %d importances of %s (%s):\n%s", TOP_N, best_boosting, method,
                    top.head(TOP_N).to_string(index=False))  # fmt: skip
    write_tables(summary, reports)

    if track and written:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment"])
        for model, filename in ((best_tree, written[0]), (best_boosting, written[-1])):
            if model is None:
                continue
            run_name = f"{model}_k{n_cutoffs_label(n_per_vehicle)}"
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string=(
                    f"tags.comparison_set = '{comparison_set}' and "
                    f"tags.mlflow.runName = '{run_name}'"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                client.log_artifact(runs[0].info.run_id, str(reports / filename))
    summary["_report"] = {
        "best_tree": best_tree,
        "best_boosting": best_boosting,
        "positive_weight": positive_weight,
        "comparison_set": comparison_set,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="run only these models (default: all)")
    parser.add_argument(
        "--time-one-fold", action="store_true",
        help="fit and score fold 0 only, print the projected total runtime, then stop",
    )  # fmt: skip
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    parser.add_argument(
        "--comparison-set", default=COMPARISON_SET,
        help=f"MLflow tag comparison_set of the runs (default: {COMPARISON_SET})",
    )  # fmt: skip
    parser.add_argument(
        "--no-rechecks", action="store_true", help="skip the re-checks of the best boosting model"
    )  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    if args.time_one_fold:
        time_one_fold(config, args.models)
        return
    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(
        config, args.models, track=track, comparison_set=args.comparison_set,
        rechecks=not args.no_rechecks,
    )  # fmt: skip


if __name__ == "__main__":
    main()
