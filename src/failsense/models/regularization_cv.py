"""Regularization of the logistic-regression reference: L2, L1, ElasticNet, C.

Run it with::

    uv run python -m failsense.models.regularization_cv      # or: make regularization
    uv run python -m failsense.models.regularization_cv --no-mlflow

Why regularize
--------------
A logistic regression with 240 inputs can fit noise. A penalty on the size of the
coefficients keeps it simpler:

* **L2** (ridge) shrinks all coefficients a little, none to exactly zero.
* **L1** (lasso) drives some coefficients to **exactly zero** - a built-in feature
  selection. The surviving features are the ones the model "chose".
* **ElasticNet** mixes both (``l1_ratio`` = share of L1).
* **C** is the *inverse* strength: small C = strong penalty.

Penalties act on coefficient size, so the inputs must be on the same scale; the
scaler inside the Pipeline (fitted per fold) takes care of that.

What this script compares (same 5 folds, reference = specs + balanced + log1p)
------------------------------------------------------------------------------
L2 with C in {0.01, 0.1} (C = 1 *is* the reference), L1 with C in {0.01, 0.1, 1},
ElasticNet (l1_ratio 0.5) with the best L1 C, and L1 on raw + engineered features
(can L1 pick the useful engineered ones?). The "best C" is chosen on the same CV
that reports the score, so it is **optimistic**: real tuning with a nested or
separate evaluation is step 7. Only the training split is used.

One-hot specification columns are 0/1 and **not scaled**; their coefficients are not
on the same scale as the standardized numeric ones and are marked in the top-10 table.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

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
    build_logreg_pipeline,
    coefficient_table,
    count_nonzero_coefficients,
    split_feature_columns,
)
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

RUN_TAGS = {"task": "q1_binary", "dataset": "scania_train", "stage": "regularization"}
REFERENCE = "l2_C1_reference"
L2_GRID = (0.01, 0.1)  # C = 1 is the reference itself
L1_GRID = (0.01, 0.1, 1.0)
ELASTICNET_L1_RATIO = 0.5
TOP_N = 10
MAX_ITER = 1000  # must match build_logreg_pipeline; used to detect non-convergence

ONE_HOT_NOTE = (
    "Note: one-hot specification columns (`cat__*`, kind 'one-hot, NOT scaled') are 0/1 and "
    "not standardized, so their coefficients are not on the same scale as the standardized "
    "numeric columns; compare them only with each other."
)


def convergence_tags(n_iter_max: int, max_iter: int = MAX_ITER) -> dict[str, str]:
    """MLflow tag ``converged``: ``"false"`` if any fold hit ``max_iter`` (solver stopped early).

    The score of such a model is kept (exploratory step), but it is marked so nobody
    reads it as a fully optimised solution.
    """
    return {"converged": "true" if n_iter_max < max_iter else "false"}


def model_name(penalty: str, C: float, engineered: bool = False) -> str:  # noqa: N803
    """Readable, stable run name, e.g. ``l1_C0.1`` or ``l1_C0.1_engineered``."""
    base = f"{penalty}_C{C:g}"
    return base + "_engineered" if engineered else base


def best_by_mean(scores: Mapping[str, np.ndarray]) -> str:
    """Name with the highest mean fold score (first one wins a tie).

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("no scores to choose from")
    return max(scores, key=lambda name: float(np.mean(scores[name])))


def aggregate_coefficients(estimators: Sequence[Any]) -> pd.DataFrame:
    """Combine the coefficients of the fold models (one fitted pipeline per fold).

    Returns:
        One row per feature that appears in any fold model, sorted by
        ``mean_abs_coef`` (descending): ``mean_coef`` (signed), ``mean_abs_coef``,
        ``folds_nonzero`` (in how many fold models the coefficient is not 0),
        ``n_folds`` and ``kind``. A feature missing in a fold (dropped by the
        imputer because it was entirely NaN) counts as 0 there.

    Raises:
        ValueError: If ``estimators`` is empty.
    """
    if not estimators:
        raise ValueError("need at least one fitted pipeline")
    tables = [coefficient_table(est).set_index("feature") for est in estimators]
    coefs = pd.concat([t["coef"] for t in tables], axis=1).fillna(0.0)
    kinds = pd.concat([t["kind"] for t in tables]).groupby(level=0).first()
    out = pd.DataFrame(
        {
            "mean_coef": coefs.mean(axis=1),
            "mean_abs_coef": coefs.abs().mean(axis=1),
            "folds_nonzero": (coefs != 0).sum(axis=1),
            "n_folds": len(tables),
            "kind": kinds.reindex(coefs.index),
        }
    )
    out.index.name = "feature"
    return out.sort_values(["mean_abs_coef", "folds_nonzero"], ascending=False).reset_index()


def top_features(coefs: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """First ``n`` rows of ``aggregate_coefficients``, skipping features that are 0 in all folds."""
    return coefs[coefs["folds_nonzero"] > 0].head(n).reset_index(drop=True)


def render_markdown(tops: Mapping[str, pd.DataFrame]) -> str:
    """Markdown with one top-10 table per model and the one-hot note."""
    lines = ["# Regularization - top features of the best L1 models", "", ONE_HOT_NOTE, ""]
    for name, table in tops.items():
        lines += [
            f"## {name}",
            "",
            "| rank | feature | kind | mean coef | mean abs coef | non-zero in folds |",
            "|---|---|---|---|---|---|",
        ]
        for rank, r in enumerate(table.itertuples(), start=1):
            lines.append(
                f"| {rank} | {r.feature} | {r.kind} | {r.mean_coef:+.4f} | "
                f"{r.mean_abs_coef:.4f} | {r.folds_nonzero}/{r.n_folds} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(
    config: Mapping[str, Any], n_per_vehicle: int | None, track: bool = False
) -> dict[str, dict[str, Any]]:
    """Cross-validate the regularization grid and write the top-feature report.

    Args:
        config: Project config.
        n_per_vehicle: Cut-offs per truck, ``None`` = all known readouts.
        track: Log every model as an MLflow run (``setup_mlflow`` must have been called).

    Returns:
        Per model: ``pr_auc`` mean/std, ``train``, ``gap``, non-zero coefficient
        statistics, ``n_iter_max`` and ``seconds``.
    """
    _, sampling_seed = sampling_params_from_config(config)
    n_splits, cv_seed = cv_params_from_config(config)
    frame = load_training_frame(config, n_per_vehicle)
    cols = split_feature_columns(frame)
    y = frame[BINARY_COL].to_numpy()
    folds = make_group_folds(frame, n_splits=n_splits, seed=cv_seed)
    raw = cols.raw + cols.categorical
    with_engineered = cols.raw + cols.engineered + cols.categorical

    common_params = {
        "n_cutoffs_per_vehicle": n_cutoffs_label(n_per_vehicle),
        "sampling_seed": sampling_seed,
        "cv_n_splits": n_splits,
        "cv_seed": cv_seed,
        "n_rows": len(frame),
        "n_vehicles": int(frame[VEHICLE_COL].nunique()),
        "positive_rate": round(float(y.mean()), 5),
        "imbalance_strategy": "balanced",
        "log1p": True,
    }
    summary: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, np.ndarray] = {}
    estimators: dict[str, list[Any]] = {}
    run_ids: dict[str, str] = {}

    def evaluate(
        name: str,
        penalty: str,
        c_value: float,
        l1_ratio: float | None = None,
        engineered: bool = False,
    ) -> None:
        model = build_logreg_pipeline(
            random_state=cv_seed,
            categorical_columns=cols.categorical,
            log1p_columns=cols.log1p,  # engineered columns are never log-transformed
            penalty=penalty,
            C=c_value,
            l1_ratio=l1_ratio,
        )
        x = frame[with_engineered if engineered else raw]
        started = time.perf_counter()
        result = cross_validate(
            model, x, y, cv=folds, scoring=SCORING_WITH_PROBA, n_jobs=-1,
            return_train_score=True, return_estimator=True,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        fitted = result["estimator"]
        nonzero = [count_nonzero_coefficients(est)[0] for est in fitted]
        n_total = count_nonzero_coefficients(fitted[0])[1]
        n_iter_max = int(max(np.max(est.named_steps["model"].n_iter_) for est in fitted))
        val, train = result["test_pr_auc"], result["train_pr_auc"]
        fold_scores[name], estimators[name] = val, fitted
        summary[name] = {
            "pr_auc": float(val.mean()),
            "pr_auc_std": float(val.std()),
            "train": float(train.mean()),
            "gap": float(train.mean() - val.mean()),
            "nonzero_mean": float(np.mean(nonzero)),
            "nonzero_min": int(min(nonzero)),
            "nonzero_max": int(max(nonzero)),
            "n_coefs": n_total,
            "mean_proba": float(result["test_mean_proba"].mean()),
            "n_iter_max": n_iter_max,
            "seconds": seconds,
        }
        converged = n_iter_max < MAX_ITER
        logger.info(
            "%-24s val %.4f +- %.4f | train %.4f | gap %+.4f | non-zero %.0f of %d "
            "(folds %d-%d) | n_iter max %d%s | %.0f s",
            name, val.mean(), val.std(), train.mean(), train.mean() - val.mean(),
            np.mean(nonzero), n_total, min(nonzero), max(nonzero), n_iter_max,
            "" if converged else "  ** NOT CONVERGED **", seconds,
        )  # fmt: skip
        if not converged:
            logger.warning("%s reached max_iter=%d without converging", name, MAX_ITER)
        if track:
            run_ids[name] = log_cv_run(
                run_name=f"{name}_k{n_cutoffs_label(n_per_vehicle)}",
                cv_result=result,
                metrics=SCORING_WITH_PROBA,
                params={
                    **common_params,
                    "model": name,
                    "penalty": penalty,
                    "C": c_value,
                    "l1_ratio": "" if l1_ratio is None else l1_ratio,
                    "solver": str(model.named_steps["model"].solver),
                    "engineered": engineered,
                    "n_features_after_preprocessing": n_total,
                    "n_nonzero_coefs_mean": round(float(np.mean(nonzero)), 1),
                    "n_nonzero_coefs_min": min(nonzero),
                    "n_nonzero_coefs_max": max(nonzero),
                    "n_iter_max": n_iter_max,
                },
                tags={**RUN_TAGS, **convergence_tags(n_iter_max)},
            )

    # Stage 1: reference and the L2 / L1 grids.
    evaluate(REFERENCE, "l2", 1.0)
    for c in L2_GRID:
        evaluate(model_name("l2", c), "l2", c)
    for c in L1_GRID:
        evaluate(model_name("l1", c), "l1", c)

    # Stage 2 needs the best L1 C (chosen on this same CV: optimistic).
    l1_names = {model_name("l1", c): c for c in L1_GRID}
    best_l1 = best_by_mean({n: fold_scores[n] for n in l1_names})
    best_c = l1_names[best_l1]
    logger.info("Best L1 on this CV: C = %g (chosen on the same folds: optimistic)", best_c)
    evaluate(model_name("elasticnet", best_c), "elasticnet", best_c, ELASTICNET_L1_RATIO)
    evaluate(model_name("l1", best_c, engineered=True), "l1", best_c, engineered=True)

    # Paired comparisons: best of each family against the reference.
    best_l2 = best_by_mean({model_name("l2", c): fold_scores[model_name("l2", c)] for c in L2_GRID})
    en_name = model_name("elasticnet", best_c)
    eng_name = model_name("l1", best_c, engineered=True)
    for candidate, reference in (
        (best_l2, REFERENCE),
        (best_l1, REFERENCE),
        (en_name, REFERENCE),
        (en_name, best_l1),
        (eng_name, best_l1),
        (eng_name, REFERENCE),
    ):
        cmp = paired_comparison(fold_scores[candidate], fold_scores[reference])
        summary[candidate].setdefault("paired", {})[reference] = cmp
        logger.info(
            "paired pr_auc %s vs %s: %+.4f +- %.4f, better in %d/%d folds",
            candidate, reference, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    # Top-10 features of the best L1 model (and of L1 + engineered), averaged over fold models.
    tops = {
        best_l1: top_features(aggregate_coefficients(estimators[best_l1])),
        eng_name: top_features(aggregate_coefficients(estimators[eng_name])),
    }
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = write_top_features(tops, reports)
    for name, table in tops.items():
        logger.info("Top %d features of %s:\n%s", TOP_N, name, table.to_string(index=False))
    logger.info(ONE_HOT_NOTE)
    if track:
        with mlflow.start_run(run_id=run_ids[best_l1]):
            mlflow.log_artifact(str(csv_path))
    summary["_best"] = {"l1": best_l1, "l1_c": best_c, "top_features_csv": str(csv_path)}
    return summary


def write_top_features(tops: Mapping[str, pd.DataFrame], reports: Path) -> Path:
    """Write the top-feature tables as CSV (+ a Markdown version with the one-hot note)."""
    combined = pd.concat([t.assign(model=name) for name, t in tops.items()], ignore_index=True)
    csv_path = reports / "regularization_q1_l1_top_features.csv"
    combined.to_csv(csv_path, index=False, float_format="%.5f")
    (reports / "regularization_q1_l1_top_features.md").write_text(
        render_markdown(tops), encoding="utf-8"
    )
    logger.info("Saved top features to %s", csv_path)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    n_per_vehicle, _ = sampling_params_from_config(config)
    track = not args.no_mlflow
    if track:
        setup_mlflow(config)
    run(config, n_per_vehicle, track=track)


if __name__ == "__main__":
    main()
