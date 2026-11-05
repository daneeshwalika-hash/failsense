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

Impurity-based feature importances of the best tree are reported, but they are biased
towards features with many distinct values and are computed on the training data; SHAP
(06.11) is the proper tool. Only the training split is used.
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
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
REFERENCE = "logreg_reference"
TREE_MIN_SAMPLES_LEAF = 100
TOP_N = 10


@dataclass(frozen=True)
class ModelSpec:
    """One row of the comparison table."""

    name: str
    family: str
    pipeline: Any
    params: Mapping[str, Any] = field(default_factory=dict)
    train_score: bool = True  # False for kNN (a point is its own neighbour)


def build_models(
    cols: FeatureColumns, random_state: int, knn_jobs: int = 1
) -> dict[str, ModelSpec]:
    """All model families of the comparison, in table order (the reference first).

    Args:
        cols: Column groups from ``split_feature_columns``.
        random_state: Seed of every model that uses randomness.
        knn_jobs: Threads of each kNN model (the folds already run in parallel).
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
                    KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=knn_jobs),
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
            )
        )
    return {s.name: s for s in specs}


def aggregate_importances(estimators: Sequence[Any]) -> pd.DataFrame:
    """Impurity-based importances of fitted tree pipelines, averaged over the folds.

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
        series.append(pd.Series(tree.feature_importances_, index=names))
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
    """Parallel folds and threads per kNN model so that all cores are used once."""
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
    _, knn_jobs = _worker_split(n_splits)
    models = select_models(build_models(cols, cv_seed, knn_jobs), names)
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
            name, t_fit, t_pred, total, total, n_splits, knn_jobs, total * n_splits,
        )  # fmt: skip
    return out


def write_tables(
    summary: Mapping[str, Mapping[str, Any]],
    importances: pd.DataFrame | None,
    best_tree: str | None,
    reports: Path,
) -> None:
    """Write the comparison table and the tree importances to ``reports/``."""
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, s in summary.items():
        paired = s.get("paired", {})
        rows.append(
            {
                "model": name, "family": s["family"], "pr_auc": s["pr_auc"],
                "pr_auc_std": s["pr_auc_std"], "train": s["train"], "gap": s["gap"],
                "mean_proba": s["mean_proba"], "seconds": s["seconds"],
                "diff_vs_reference": paired.get(REFERENCE, {}).get("mean_diff", np.nan),
                "diff_std": paired.get(REFERENCE, {}).get("std_diff", np.nan),
                "folds_won": paired.get(REFERENCE, {}).get("wins", np.nan),
            }
        )  # fmt: skip
    pd.DataFrame(rows).to_csv(reports / "compare_models_q1.csv", index=False, float_format="%.5f")
    if importances is not None and best_tree is not None:
        top = importances.head(TOP_N)
        top.assign(model=best_tree).to_csv(
            reports / "compare_models_q1_tree_importances.csv", index=False, float_format="%.5f"
        )
        lines = [
            f"# Top {TOP_N} impurity-based importances of {best_tree}",
            "",
            "Averaged over the 5 fold models. **Impurity importances are biased** towards "
            "features with many distinct values (counters over one-hot columns) and are "
            "computed on the training data; SHAP (06.11) is the proper tool.",
            "",
            "| rank | feature | kind | mean importance | non-zero in folds |",
            "|---|---|---|---|---|",
        ]
        for rank, r in enumerate(top.itertuples(), start=1):
            lines.append(
                f"| {rank} | {r.feature} | {r.kind} | {r.mean_importance:.4f} | "
                f"{r.folds_nonzero}/{r.n_folds} |"
            )
        (reports / "compare_models_q1_tree_importances.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def run(
    config: Mapping[str, Any], names: Sequence[str] | None = None, track: bool = False
) -> dict[str, dict[str, Any]]:
    """Cross-validate the model families and write the comparison table.

    Args:
        config: Project config.
        names: Models to run (``None`` = all). The reference is always included.
        track: Log every model as an MLflow run (``setup_mlflow`` must have been called).

    Returns:
        Per model: ``pr_auc`` mean/std, ``train`` and ``gap`` (``NaN`` for kNN),
        ``mean_proba``, ``seconds`` and the paired comparison with the reference.
    """
    n_per_vehicle, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    cols = split_feature_columns(frame)
    y = frame[BINARY_COL].to_numpy()
    x = frame[cols.raw + cols.categorical]
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    folds_parallel, knn_jobs = _worker_split(n_splits)
    chosen = select_models(build_models(cols, cv_seed, knn_jobs), names)
    if REFERENCE not in chosen:  # every paired comparison needs it
        chosen = {REFERENCE: build_models(cols, cv_seed, knn_jobs)[REFERENCE], **chosen}

    common = {
        "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle),
        "sampling_seed": sampling_seed,
        "cv_n_splits": n_splits,
        "cv_seed": cv_seed,
        "n_rows": len(frame),
        "n_vehicles": int(frame[VEHICLE_COL].nunique()),
        "n_numeric_features": len(cols.raw),
        "n_categorical_features": len(cols.categorical),
        "positive_rate": round(float(y.mean()), 5),
    }
    summary: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, np.ndarray] = {}
    fitted: dict[str, list[Any]] = {}
    for name, spec in chosen.items():
        started = time.perf_counter()
        result = cross_validate(
            spec.pipeline, x, y, cv=folds, scoring=SCORING_WITH_PROBA, n_jobs=folds_parallel,
            return_train_score=spec.train_score, return_estimator=spec.family == "tree",
        )  # fmt: skip
        seconds = time.perf_counter() - started
        val = result["test_pr_auc"]
        train = float(result["train_pr_auc"].mean()) if spec.train_score else float("nan")
        fold_scores[name] = val
        if spec.family == "tree":
            fitted[name] = result["estimator"]
        summary[name] = {
            "family": spec.family,
            "pr_auc": float(val.mean()),
            "pr_auc_std": float(val.std()),
            "train": train,
            "gap": train - float(val.mean()),
            "mean_proba": float(result["test_mean_proba"].mean()),
            "seconds": seconds,
            "fit_seconds_sum": float(np.sum(result["fit_time"])),
        }
        logger.info(
            "%-20s val %.4f +- %.4f | train %s | gap %s | mean proba %.4f | %.0f s",
            name, val.mean(), val.std(),
            "n/a" if np.isnan(train) else f"{train:.4f}",
            "n/a" if np.isnan(train) else f"{train - val.mean():+.4f}",
            result["test_mean_proba"].mean(), seconds,
        )  # fmt: skip
        if track:
            log_cv_run(
                run_name=f"{name}_k{n_cutoffs_label(n_per_vehicle)}",
                cv_result=result,
                metrics=SCORING_WITH_PROBA,
                params={**common, "model": name, "family": spec.family, **spec.params},
                tags={**RUN_TAGS, "family": spec.family},
            )

    for name in chosen:
        if name == REFERENCE:
            continue
        cmp = paired_comparison(fold_scores[name], fold_scores[REFERENCE])
        summary[name]["paired"] = {REFERENCE: cmp}
        logger.info(
            "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
            name, REFERENCE, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    importances, best_tree = None, None
    tree_scores = {n: fold_scores[n] for n in fitted}
    if tree_scores:
        best_tree = best_by_mean(tree_scores)
        importances = aggregate_importances(fitted[best_tree])
        logger.info("Best tree: %s. Top %d impurity importances (biased; SHAP on 06.11):\n%s",
                    best_tree, TOP_N, importances.head(TOP_N).to_string(index=False))  # fmt: skip
    write_tables(summary, importances, best_tree, resolve_path(config["paths"]["reports"]))
    if track and best_tree is not None:
        # attach the importances to the run of the best tree
        client = mlflow.tracking.MlflowClient()
        run_name = f"{best_tree}_k{n_cutoffs_label(n_per_vehicle)}"
        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment"])
        runs = client.search_runs(
            [experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"], max_results=1,
        )  # fmt: skip
        if runs:
            path = (
                resolve_path(config["paths"]["reports"]) / "compare_models_q1_tree_importances.csv"
            )
            client.log_artifact(runs[0].info.run_id, str(path))
    summary["_report"] = {"best_tree": best_tree}
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="run only these models (default: all)")
    parser.add_argument(
        "--time-one-fold", action="store_true",
        help="fit and score fold 0 only, print the projected total runtime, then stop",
    )  # fmt: skip
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
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
    run(config, args.models, track=track)


if __name__ == "__main__":
    main()
