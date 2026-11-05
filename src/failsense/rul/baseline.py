"""Linear baselines for RUL on C-MAPSS FD001: Dummy (mean), Ridge, Lasso.

Run it with::

    uv run python -m failsense.rul.baseline            # or: make rul-baseline
    uv run python -m failsense.rul.baseline --no-mlflow

What happens
------------
1. Load the FD001 **training** file and check the data contract.
2. RUL = max cycle of the engine - cycle, capped at ``rul.cap`` (125).
3. ``GroupKFold(5)`` by engine: every engine is in exactly one validation fold. The
   rows of one engine are almost copies of each other (neighbouring cycles), so
   the same engine on both sides of a split would leak - the same reason as
   ``vehicle_id`` in SCANIA.
4. Models inside one Pipeline each (the scaler is fitted per fold, on the training
   engines only): ``DummyRegressor(mean)``, ``Ridge`` and ``Lasso`` with a small alpha
   grid. Inputs: the 3 operational settings and the 21 sensors; **all** sensors are
   kept, so Lasso shows which ones it switches off. Dropping constant sensors is
   planned for 11.11.
5. Metrics: RMSE (primary; punishes big errors) and MAE, validation mean +- std and
   training value. Reported: which columns are constant in FD001 and which
   coefficients Lasso sets to 0 in every fold.
6. A variant "best Ridge + cycle" shows what the engine age adds. ``cycle`` is a legitimate
   input (it is known at prediction time), but the training rows cover the whole life of
   each engine while the official test engines are cut at a random point before failure,
   so its distribution differs between training and test - the same question as
   ``time_step`` in SCANIA. It is therefore left out of the main models until that is
   examined.

alpha is chosen on the same CV that reports the score, which is optimistic; real
tuning comes later. The **test file and RUL_FD001.txt are never read** (used once at
the end).
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from failsense.config import load_config, resolve_path
from failsense.data.split import assert_no_group_overlap, cv_params_from_config
from failsense.models.baseline_cv import paired_comparison
from failsense.rul.data import CYCLE_COL, ENGINE_COL, FEATURE_COLUMNS, load_train
from failsense.rul.labels import (
    assert_no_forbidden_features,
    compute_rul,
    rul_cap_from_config,
)
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
RUN_TAGS = {"task": "q3_rul", "dataset": "cmapss_fd001_train", "stage": "baseline"}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (0.001, 0.01, 0.1, 1.0)
LASSO_MAX_ITER = 10_000  # the default (1000) is often too few for correlated sensors


def select_features(df: pd.DataFrame, include_cycle: bool = False) -> list[str]:
    """Model input columns: settings and sensors (+ ``cycle`` for the variant).

    Raises:
        ValueError: If a column is missing from ``df`` or a forbidden column
            (label, ``max_cycle``, engine id) would be selected.
    """
    columns = list(FEATURE_COLUMNS) + ([CYCLE_COL] if include_cycle else [])
    assert_no_forbidden_features(columns)  # defensive: the lists above never contain them
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing feature columns: {missing[:5]}")
    return columns


def make_engine_folds(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``GroupKFold`` by engine; raises if an engine would be on both sides of a fold.

    Raises:
        ValueError: If ``engine`` is missing, there are fewer engines than folds, or
            (defensive check) an engine appears in train and validation of a fold.
    """
    if ENGINE_COL not in df.columns:
        raise ValueError(f"df is missing {ENGINE_COL!r}")
    if df[ENGINE_COL].nunique() < n_splits:
        raise ValueError(f"need at least {n_splits} engines for {n_splits} folds")
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=df[ENGINE_COL]))
    engines = df[ENGINE_COL].to_numpy()
    for i, (train_idx, val_idx) in enumerate(folds):
        assert_no_group_overlap(
            {f"fold{i}_train": set(engines[train_idx]), f"fold{i}_val": set(engines[val_idx])}
        )
    return folds


def build_rul_pipeline(model: Any) -> Pipeline:
    """Scaler -> regressor in one estimator (the scaler is fitted per fold)."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def constant_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns that hold a single value (no information, zero variance)."""
    return [c for c in columns if df[c].nunique(dropna=True) <= 1]


def zero_in_all_folds(estimators: Sequence[Pipeline], feature_names: Sequence[str]) -> list[str]:
    """Features whose coefficient is exactly 0 in every fitted fold model.

    Raises:
        ValueError: If ``estimators`` is empty.
    """
    if not estimators:
        raise ValueError("need at least one fitted pipeline")
    coefs = np.vstack([est.named_steps["model"].coef_.ravel() for est in estimators])
    zero = (coefs == 0).all(axis=0)
    return [name for name, is_zero in zip(feature_names, zero, strict=True) if is_zero]


def best_by_lowest_mean(scores: Mapping[str, np.ndarray]) -> str:
    """Name with the lowest mean error (first one wins a tie)."""
    if not scores:
        raise ValueError("no scores to choose from")
    return min(scores, key=lambda name: float(np.mean(scores[name])))


def rmse_improvement(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Paired comparison of two RMSE arrays; ``mean_diff`` > 0 means the candidate is better.

    ``paired_comparison`` counts a win when the first argument is larger, which is
    right for scores but wrong for errors, so the arguments are swapped here.
    """
    return paired_comparison(reference, candidate)


def run(config: Mapping[str, Any], track: bool = False) -> dict[str, dict[str, Any]]:
    """Cross-validate Dummy, Ridge and Lasso on FD001 and report the findings.

    Args:
        config: Project config (needs ``datasets.cmapss.train``, ``rul``, ``cv``).
        track: Log every model as an MLflow run (``setup_mlflow`` must have been
            called with the RUL experiment). Default ``False`` so tests never write
            to the project's tracking database.

    Returns:
        Per model: RMSE/MAE validation mean and std, training values, gap
        (validation - training; positive = overfitting), plus a ``_report`` entry with
        the constant columns, the best alphas and the Lasso zero coefficients.
    """
    n_splits, _ = cv_params_from_config(config)
    cap = rul_cap_from_config(config)
    frame = load_train(resolve_path(config["datasets"]["cmapss"]["train"]))
    y = compute_rul(frame, cap).to_numpy()
    features = select_features(frame)
    features_cycle = select_features(frame, include_cycle=True)
    folds = make_engine_folds(frame, n_splits)
    logger.info(
        "RUL target: cap %d, mean %.1f, std %.1f over %d rows of %d engines",
        cap, y.mean(), y.std(), len(frame), frame[ENGINE_COL].nunique(),
    )  # fmt: skip

    common = {
        "rul_cap": cap,
        "cv_n_splits": n_splits,
        "n_rows": len(frame),
        "n_engines": int(frame[ENGINE_COL].nunique()),
    }
    summary: dict[str, dict[str, Any]] = {}
    val_rmse: dict[str, np.ndarray] = {}
    fitted: dict[str, list[Pipeline]] = {}

    def evaluate(name: str, model: Any, columns: list[str], extra: Mapping[str, Any]) -> None:
        started = time.perf_counter()
        result = cross_validate(
            build_rul_pipeline(model), frame[columns], y, cv=folds, scoring=SCORING,
            n_jobs=-1, return_train_score=True, return_estimator=True,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        # sklearn's scorers return negative errors ("greater is better"); flip the sign
        # so RMSE and MAE are the plain positive numbers everyone expects.
        for key in ("test_rmse", "train_rmse", "test_mae", "train_mae"):
            result[key] = -result[key]
        val, train = result["test_rmse"], result["train_rmse"]
        val_rmse[name], fitted[name] = val, result["estimator"]
        summary[name] = {
            "rmse": float(val.mean()), "rmse_std": float(val.std()),
            "rmse_train": float(train.mean()), "gap": float(val.mean() - train.mean()),
            "mae": float(result["test_mae"].mean()), "mae_std": float(result["test_mae"].std()),
            "n_features": len(columns), "seconds": seconds,
        }  # fmt: skip
        logger.info(
            "%-22s RMSE val %.2f +- %.2f | train %.2f | gap %+.2f | MAE %.2f +- %.2f | %.0f s",
            name, val.mean(), val.std(), train.mean(), val.mean() - train.mean(),
            result["test_mae"].mean(), result["test_mae"].std(), seconds,
        )  # fmt: skip
        if track:
            # Note: log_cv_run logs "<metric>_gap_mean" as train - val, which is
            # negative for an overfitting error metric.
            log_cv_run(
                run_name=name, cv_result=result, metrics=list(SCORING),
                params={**common, "model": name, "n_features": len(columns), **extra},
                tags=RUN_TAGS,
            )  # fmt: skip

    evaluate("dummy_mean", DummyRegressor(strategy="mean"), features, {"family": "dummy"})
    for alpha in RIDGE_ALPHAS:
        evaluate(f"ridge_a{alpha:g}", Ridge(alpha=alpha), features,
                 {"family": "ridge", "alpha": alpha, "include_cycle": False})  # fmt: skip
    lasso_info: dict[str, dict[str, Any]] = {}
    for alpha in LASSO_ALPHAS:
        name = f"lasso_a{alpha:g}"
        model = Lasso(alpha=alpha, max_iter=LASSO_MAX_ITER)
        evaluate(name, model, features,
                 {"family": "lasso", "alpha": alpha, "include_cycle": False})  # fmt: skip

    best_ridge = best_by_lowest_mean(
        {f"ridge_a{a:g}": val_rmse[f"ridge_a{a:g}"] for a in RIDGE_ALPHAS}
    )
    best_lasso = best_by_lowest_mean(
        {f"lasso_a{a:g}": val_rmse[f"lasso_a{a:g}"] for a in LASSO_ALPHAS}
    )
    ridge_alpha = float(best_ridge.removeprefix("ridge_a"))
    cycle_name = f"{best_ridge}_cycle"
    evaluate(cycle_name, Ridge(alpha=ridge_alpha), features_cycle,
             {"family": "ridge", "alpha": ridge_alpha, "include_cycle": True})  # fmt: skip

    for name in (f"lasso_a{a:g}" for a in LASSO_ALPHAS):
        zeros = zero_in_all_folds(fitted[name], features)
        n_iter = max(int(est.named_steps["model"].n_iter_) for est in fitted[name])
        lasso_info[name] = {"zero_in_all_folds": zeros, "n_iter_max": n_iter}
        logger.info("%s: %d of %d coefficients are 0 in all folds; n_iter max %d%s",
                    name, len(zeros), len(features), n_iter,
                    "  ** NOT CONVERGED **" if n_iter >= LASSO_MAX_ITER else "")  # fmt: skip

    for candidate, reference in (
        (best_ridge, "dummy_mean"),
        (best_lasso, "dummy_mean"),
        (best_lasso, best_ridge),
        (cycle_name, best_ridge),
    ):
        cmp = rmse_improvement(val_rmse[candidate], val_rmse[reference])
        summary[candidate].setdefault("paired", {})[reference] = cmp
        logger.info(
            "paired RMSE %s vs %s: %+.2f (positive = better) +- %.2f, better in %d/%d folds",
            candidate, reference, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    constant = constant_columns(frame, list(FEATURE_COLUMNS))
    logger.info("Constant columns in FD001: %s", constant)
    logger.info("Best alphas on this CV (optimistic): ridge %s, lasso %s", best_ridge, best_lasso)
    logger.info(
        "Lasso (%s) zero in all folds: %s", best_lasso, lasso_info[best_lasso]["zero_in_all_folds"]
    )

    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_dict(
        {n: {k: v for k, v in s.items() if k != "paired"} for n, s in summary.items()},
        orient="index",
    ).rename_axis("model")
    table.to_csv(reports / "rul_baseline_fd001.csv", float_format="%.4f")
    summary["_report"] = {
        "constant_columns": constant,
        "best_ridge": best_ridge,
        "best_lasso": best_lasso,
        "lasso": lasso_info,
        "target_std": float(y.std()),
    }
    return summary


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
        setup_mlflow(config, experiment=config["rul"]["mlflow_experiment"])
    run(config, track=track)


if __name__ == "__main__":
    main()
