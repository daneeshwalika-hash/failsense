"""RUL on C-MAPSS FD001 with time-series features: rolling mean / std, Ridge vs LightGBM.

Run: ``uv run python -m failsense.rul.timeseries`` (``make rul-timeseries``).

Why time-series features: a single cycle of a noisy sensor says little; the *trend* over the last
cycles says more. For every informative sensor the trailing mean (level) and std (noise / change)
over a short and a longer window are added. SQL analogy: ``AVG(x) OVER (PARTITION BY engine ORDER BY
cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)``.

Leakage rules
-------------
* Windows are **trailing** (the current and the previous cycles of the SAME engine only), so a
  feature at cycle t never sees a later cycle and never mixes two engines. A test pins this.
* Rolling features use no label. The RUL target is capped (``rul.cap``, 125) as before.
* Cross validation is ``GroupKFold`` by engine (the rows of an engine are near copies).
* Constant columns are removed by ``VarianceThreshold`` **inside the Pipeline** (fitted per fold);
  the scaler is inside as well. ``cycle`` and the engine id are never features.
* Only the TRAINING file is read. ``test_FD001.txt`` and ``RUL_FD001.txt`` are used once, at
  the end.

Caveat: the score is the RMSE over all cycles of held-out training engines (RUL capped), not the
official test protocol (the last cycle of truncated test engines), so it is not comparable with
published FD001 numbers. C-MAPSS is simulated data.
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
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from failsense.config import load_config, resolve_path
from failsense.data.split import cv_params_from_config
from failsense.rul.baseline import SCORING, make_engine_folds, rmse_improvement
from failsense.rul.data import CYCLE_COL, ENGINE_COL, FEATURE_COLUMNS, SENSOR_COLUMNS, load_train
from failsense.rul.labels import assert_no_forbidden_features, compute_rul, rul_cap_from_config
from failsense.tracking import log_cv_run, setup_mlflow

logger = logging.getLogger(__name__)

RUN_TAGS = {"task": "q3_rul", "dataset": "cmapss_fd001_train", "stage": "timeseries"}
REPORT_FILE = "rul_timeseries_fd001.csv"


def informative_sensors(df: pd.DataFrame, sensors: Sequence[str] = SENSOR_COLUMNS) -> list[str]:
    """Sensors that take more than one value (constant sensors carry no information)."""
    return [c for c in sensors if df[c].nunique(dropna=True) > 1]


def add_rolling_features(
    df: pd.DataFrame, columns: Sequence[str], windows: Sequence[int]
) -> pd.DataFrame:
    """Add trailing rolling mean and std per engine: ``<col>__roll<w>_mean`` / ``_std``.

    The first cycles use the shorter history available (``min_periods=1``); the std of a single
    reading is 0 (nothing varied yet). The frame is processed in ``(engine, cycle)`` order and
    returned in its original row order.

    Raises:
        ValueError: On missing columns, an empty or invalid window list, or cycles that are not
            strictly increasing within an engine.
    """
    missing = [c for c in (ENGINE_COL, CYCLE_COL, *columns) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns: {missing[:5]}")
    if not windows or any(int(w) < 2 for w in windows):
        raise ValueError("windows must be a non-empty list of integers >= 2")
    ordered = df.sort_values([ENGINE_COL, CYCLE_COL], kind="stable")
    if (ordered.groupby(ENGINE_COL)[CYCLE_COL].diff() <= 0).any():
        raise ValueError("cycle must be strictly increasing within each engine")
    by_engine = ordered.groupby(ENGINE_COL, sort=False)[list(columns)]
    new: dict[str, pd.Series] = {}
    for window in windows:
        rolling = by_engine.rolling(window=int(window), min_periods=1)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
        for col in columns:
            new[f"{col}__roll{window}_mean"] = mean[col]
            new[f"{col}__roll{window}_std"] = std[col]
    return pd.concat([df, pd.DataFrame(new, index=ordered.index).reindex(df.index)], axis=1)


def feature_sets(
    df: pd.DataFrame, windows: Sequence[int]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """The frame with rolling features and the column lists of the two feature sets.

    ``raw``: 3 settings + 21 sensors (constant ones are dropped inside the Pipeline).
    ``rolling``: ``raw`` plus the rolling mean / std of the informative sensors.
    """
    sensors = informative_sensors(df)
    frame = add_rolling_features(df, sensors, windows)
    raw = list(FEATURE_COLUMNS)
    rolling = raw + [c for c in frame.columns if "__roll" in c]
    for columns in (raw, rolling):
        assert_no_forbidden_features(columns)
    return frame, {"raw": raw, "rolling": rolling}


def make_pipeline(model: Any) -> Pipeline:
    """Constant-column filter -> scaler -> regressor; every step is fitted per fold."""
    return Pipeline(
        [("constant", VarianceThreshold(0.0)), ("scale", StandardScaler()), ("model", model)]
    )


def lightgbm_model(params: Mapping[str, Any], seed: int) -> LGBMRegressor:
    """A LightGBM regressor with the configured (untuned) settings."""
    return LGBMRegressor(
        **params, deterministic=True, n_jobs=os.cpu_count() or 1, random_state=seed, verbose=-1
    )


def run(config: Mapping[str, Any], track: bool = False) -> dict[str, dict[str, Any]]:
    """Cross-validate Dummy, Ridge and LightGBM on raw and rolling features."""
    n_splits, seed = cv_params_from_config(config)
    cap = rul_cap_from_config(config)
    section = config["rul"]
    frame, sets = feature_sets(
        load_train(resolve_path(config["datasets"]["cmapss"]["train"])), section["rolling_windows"]
    )
    y = compute_rul(frame, cap).to_numpy()
    folds = make_engine_folds(frame, n_splits)
    n_engines = frame[ENGINE_COL].nunique()
    logger.info(
        "%d rows, %d engines, %d raw / %d rolling columns, cap %d",
        len(frame), n_engines, len(sets["raw"]), len(sets["rolling"]), cap,
    )  # fmt: skip

    summary: dict[str, dict[str, Any]] = {}
    val_rmse: dict[str, np.ndarray] = {}

    def evaluate(name: str, model: Any, columns: list[str], extra: Mapping[str, Any]) -> None:
        started = time.perf_counter()
        result = cross_validate(
            make_pipeline(model), frame[columns], y, cv=folds, scoring=SCORING, n_jobs=1,
            return_train_score=True,
        )  # fmt: skip
        for key in ("test_rmse", "train_rmse", "test_mae", "train_mae"):
            result[key] = -result[key]  # sklearn returns negative errors
        val, train = result["test_rmse"], result["train_rmse"]
        val_rmse[name] = val
        summary[name] = {
            "rmse": float(val.mean()),
            "rmse_std": float(val.std()),
            "rmse_train": float(train.mean()),
            "gap": float(val.mean() - train.mean()),
            "mae": float(result["test_mae"].mean()),
            "n_features": len(columns),
            "seconds": time.perf_counter() - started,
        }
        logger.info(
            "%-18s RMSE val %.2f +- %.2f | train %.2f | MAE %.2f | %d columns",
            name, val.mean(), val.std(), train.mean(), summary[name]["mae"], len(columns),
        )  # fmt: skip
        if track:
            log_cv_run(
                run_name=name, cv_result=result, metrics=list(SCORING),
                params={"rul_cap": cap, "cv_n_splits": n_splits, "model": name,
                        "n_features": len(columns), **extra},
                tags=RUN_TAGS,
            )  # fmt: skip

    lgbm = dict(section["lightgbm"])
    evaluate("dummy_mean", DummyRegressor(strategy="mean"), sets["raw"], {"family": "dummy"})
    alpha = float(section["ridge_alpha"])
    evaluate("ridge_raw", Ridge(alpha=alpha), sets["raw"], {"family": "ridge", "features": "raw"})
    evaluate("ridge_rolling", Ridge(alpha=alpha), sets["rolling"],
             {"family": "ridge", "features": "rolling"})  # fmt: skip
    evaluate("lightgbm_raw", lightgbm_model(lgbm, seed), sets["raw"],
             {"family": "lightgbm", "features": "raw", **lgbm})  # fmt: skip
    evaluate("lightgbm_rolling", lightgbm_model(lgbm, seed), sets["rolling"],
             {"family": "lightgbm", "features": "rolling", **lgbm})  # fmt: skip

    for candidate, reference in (
        ("ridge_rolling", "ridge_raw"),
        ("lightgbm_raw", "ridge_raw"),
        ("lightgbm_rolling", "lightgbm_raw"),
        ("lightgbm_rolling", "ridge_rolling"),
        ("lightgbm_rolling", "ridge_raw"),
    ):
        cmp = rmse_improvement(val_rmse[candidate], val_rmse[reference])
        summary[candidate].setdefault("paired", {})[reference] = cmp
        logger.info(
            "paired RMSE %s vs %s: %+.2f (positive = better) +- %.2f, better in %d/%d folds",
            candidate, reference, cmp["mean_diff"], cmp["std_diff"], cmp["wins"], cmp["n_folds"],
        )  # fmt: skip

    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_dict(
        {n: {k: v for k, v in s.items() if k != "paired"} for n, s in summary.items()},
        orient="index",
    ).rename_axis("model")
    table.to_csv(reports / REPORT_FILE, float_format="%.4f")
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
