"""One-time check: ``matched`` vs ``k5`` training sample on the OFFICIAL validation (06.11).

Run it with ``uv run python -m failsense.models.matched_check`` (``make matched-check``). The test
split is not touched.

Question left open on 05.11: the ``matched`` training sample looked better on the calibrated
evaluation set (+0.0207, 5/5 folds) but not on the literal one (+0.0012), and it is built by the
*same* rank matching that produced the evaluation set (and its 607 / 26 / 2 / 0 class-mix
artefact). The official validation shares no mechanism with either.

Both models use the identical recipe of the registered candidate (XGBoost, 300 rounds, the
registry's parameters, engineered features, no class weights, same 239 input columns); only the
training rows differ. ``k5`` = the registered model itself (5 random cut-offs per truck, 117,031
rows), ``matched`` = 5 cut-offs per truck whose ``time_step`` follows the validation last-readout
``time_step`` (rank matching with noise, ``matched_training_cutoffs``).

**Rule, fixed before the run:** switch to ``matched`` only if it is better in BOTH

* PR-AUC (point estimate on the validation) by more than the spread of the paired bootstrap
  difference (std over 500 resamples of the validation trucks), and
* the cross-fitted total cost (threshold chosen on 4 folds, applied to the 5th; primary mapping)
  by more than the spread of its paired bootstrap difference;

otherwise keep ``k5``. Both models are scored on the same resampled trucks and the same folds.
The validation is used here for a selection (and already for the threshold), so numbers on it
are slightly optimistic from now on; the test split stays untouched until the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import average_precision_score

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    BINARY_COL,
    CLASS_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sampling_params_from_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.features.build import attach_specifications, features_params_from_config
from failsense.models.baseline_cv import frame_from_sampled, read_table, table_path
from failsense.models.boosters import make_booster
from failsense.models.calibrate import score_validation
from failsense.models.cost import alert_class_from_config, binary_costs, cost_matrix_from_config
from failsense.models.eval_set import matched_training_cutoffs, validation_time_steps
from failsense.models.pipeline import build_model_pipeline, split_feature_columns
from failsense.models.threshold import cross_fit_threshold
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

SUMMARY_FILE = "matched_check_q1.json"
N_ROUNDS_KEY = "n_rounds"


def decide_sampling(
    pr_diff: float, pr_spread: float, saving_diff: float, cost_spread: float
) -> tuple[str, str]:
    """Apply the rule fixed in advance (see the module docstring).

    Args:
        pr_diff: PR-AUC(matched) - PR-AUC(k5).
        pr_spread: Std of the paired bootstrap difference of the PR-AUC.
        saving_diff: cost(k5) - cost(matched) (positive = matched is cheaper).
        cost_spread: Std of the paired bootstrap difference of the cost.

    Returns:
        ``(choice, reason)`` with ``choice`` in ``{"matched", "k5"}``.
    """
    pr_ok = pr_diff > pr_spread
    cost_ok = saving_diff > cost_spread
    if pr_ok and cost_ok:
        return "matched", "better in PR-AUC and in cost by more than the bootstrap spread"
    parts = []
    parts.append(
        f"PR-AUC {pr_diff:+.4f} vs. spread {pr_spread:.4f} ({'ok' if pr_ok else 'not enough'})"
    )
    parts.append(
        f"cost saving {saving_diff:+,.0f} vs. spread {cost_spread:,.0f} "
        f"({'ok' if cost_ok else 'not enough'})"
    )
    return "k5", "keep k5: " + "; ".join(parts)


def paired_bootstrap(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b) over trucks (draws without a positive skipped)."""
    rng = np.random.default_rng(seed)
    diffs, a_vals, b_vals = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if not y[idx].any():
            continue
        a = average_precision_score(y[idx], p_a[idx])
        b = average_precision_score(y[idx], p_b[idx])
        a_vals.append(a)
        b_vals.append(b)
        diffs.append(a - b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_std": float(np.std(diffs, ddof=1)), "diff_ci_low": float(low),
        "diff_ci_high": float(high), "share_a_better": float(np.mean(np.array(diffs) > 0)),
        "a_ci": [float(v) for v in np.percentile(a_vals, [2.5, 97.5])],
        "b_ci": [float(v) for v in np.percentile(b_vals, [2.5, 97.5])],
    }  # fmt: skip


def paired_cost_bootstrap(
    classes: np.ndarray, vehicles: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    costs: Any, n_boot: int, n_splits: int, seed: int,
) -> dict[str, float]:  # fmt: skip
    """Paired bootstrap of the cross-fitted cost: cost(b) - cost(a) (positive = a is cheaper).

    Every resample draws trucks with replacement and runs the cross-fitted threshold for both
    models on the same trucks and the same folds (duplicates of a truck stay in one fold).
    """
    rng = np.random.default_rng(seed)
    savings = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(classes), len(classes))
        c, v = classes[idx], vehicles[idx]
        if not (c >= 1).any():
            continue
        cost_a = cross_fit_threshold(c, p_a[idx], v, costs, n_splits, seed).cost
        cost_b = cross_fit_threshold(c, p_b[idx], v, costs, n_splits, seed).cost
        savings.append(cost_b - cost_a)
    low, high = np.percentile(savings, [2.5, 97.5])
    return {
        "diff_std": float(np.std(savings, ddof=1)), "diff_ci_low": float(low),
        "diff_ci_high": float(high), "share_a_cheaper": float(np.mean(np.array(savings) > 0)),
    }  # fmt: skip


def fit_matched_model(config: Mapping[str, Any], registry: Mapping[str, Any]) -> tuple[Any, dict]:
    """Fit the registry recipe on the ``matched`` training sample (training split only)."""
    seed = sampling_params_from_config(config)[1]
    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    specs = read_table(table_path(config, "train", "specifications"))
    labels = build_labels(readouts, tte, class_bounds_from_config(config))
    targets = validation_time_steps(config)
    cutoffs = matched_training_cutoffs(labels, targets, 5, seed)
    frame = frame_from_sampled(
        readouts, cutoffs, counter_cols=counter_columns(), hist_groups=histogram_groups(),
        top_fraction=features_params_from_config(config),
    )  # fmt: skip
    del readouts
    frame = attach_specifications(frame, specs).reset_index(drop=True)
    cols = split_feature_columns(frame)
    columns = cols.raw + cols.engineered + cols.categorical
    if columns != registry["input_columns"]:
        raise ValueError("matched training columns differ from the registered input columns")
    y = frame[BINARY_COL].to_numpy(dtype=int)
    booster = make_booster(
        registry["family"], registry["params"], int(config["tuning"]["seed"]),
        os.cpu_count() or 1, int(registry[N_ROUNDS_KEY]),
    )  # fmt: skip
    pipeline = build_model_pipeline(booster, cols.categorical, scaler=None).fit(frame[columns], y)
    info = {"n_rows": len(frame), "positive_rate": float(y.mean()),
            "n_trucks": int(frame[VEHICLE_COL].nunique())}  # fmt: skip
    logger.info("Fitted the registry recipe on the matched sample: %s", info)
    return pipeline, info


def run(config: Mapping[str, Any], track: bool = True) -> dict[str, Any]:
    cal = config["calibration"]
    reports = resolve_path(config["paths"]["reports"])
    registry = json.loads((reports / "registry_q1.json").read_text(encoding="utf-8"))
    n_boot, n_splits, seed = int(cal["n_bootstrap"]), int(cal["n_splits"]), int(cal["seed"])
    t0 = time.perf_counter()

    frame, p_k5 = score_validation(config, registry)  # the registered (k5) model
    classes = frame[CLASS_COL].to_numpy(dtype=int)
    vehicles = frame[VEHICLE_COL].to_numpy()
    y = (classes >= 1).astype(int)
    matched_pipeline, matched_info = fit_matched_model(config, registry)
    from failsense.models.validation_data import registry_input_columns, select_model_inputs

    inputs = select_model_inputs(frame, registry_input_columns(config))
    p_matched = matched_pipeline.predict_proba(inputs)[:, 1]

    costs = binary_costs(
        cost_matrix_from_config(config), alert_class_from_config(config), "primary"
    )
    cf_k5 = cross_fit_threshold(classes, p_k5, vehicles, costs, n_splits, seed)
    cf_matched = cross_fit_threshold(classes, p_matched, vehicles, costs, n_splits, seed)
    always = float(cf_k5.folds["always_cost"].sum())
    pr_k5, pr_matched = (float(average_precision_score(y, p)) for p in (p_k5, p_matched))

    start = time.perf_counter()
    cross_fit_threshold(classes, p_k5, vehicles, costs, n_splits, seed)
    one = time.perf_counter() - start
    logger.info(
        "Cost bootstrap: %d resamples x 2 models, estimated %.0f s", n_boot, 2 * n_boot * one
    )
    pr_boot = paired_bootstrap(y, p_matched, p_k5, n_boot, seed)  # matched - k5
    cost_boot = paired_cost_bootstrap(
        classes, vehicles, p_matched, p_k5, costs, n_boot, n_splits, seed
    )  # k5 cost - matched cost: positive = matched cheaper

    pr_diff, saving = pr_matched - pr_k5, cf_k5.cost - cf_matched.cost
    choice, reason = decide_sampling(pr_diff, pr_boot["diff_std"], saving, cost_boot["diff_std"])
    summary = {
        "question": "matched vs k5 training sample on the official validation (one-time check)",
        "rule": "switch only if matched is better in BOTH PR-AUC and cross-fitted cost by more "
        "than the std of the paired bootstrap difference; otherwise keep k5",
        "n_trucks": len(y), "n_positives": int(y.sum()), "n_bootstrap": n_boot,
        "k5": {"n_rows": registry["n_train_rows"], "pr_auc": pr_k5, "pr_auc_95ci": pr_boot["b_ci"],
               "cost_cross_fitted": cf_k5.cost, "fold_costs": cf_k5.folds["cost"].tolist()},
        "matched": {**matched_info, "pr_auc": pr_matched, "pr_auc_95ci": pr_boot["a_ci"],
                    "cost_cross_fitted": cf_matched.cost,
                    "fold_costs": cf_matched.folds["cost"].tolist()},
        "always_alert_cost": always,
        "pr_auc_diff_matched_minus_k5": {"point": pr_diff, **{k: v for k, v in pr_boot.items()
                                                            if k not in ("a_ci", "b_ci")}},
        "cost_saving_matched_vs_k5": {"point": saving, **cost_boot},
        "decision": choice, "reason": reason,
        "timing_seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (reports / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    logger.info("Decision: %s (%s)", choice, reason)
    logger.info("%s", json.dumps(summary, indent=1, default=float))
    if track:
        with mlflow.start_run(run_name="matched_vs_k5_validation") as active:
            mlflow.set_tags({"task": "q1_binary", "stage": "sampling_check", "split": "validation"})
            mlflow.log_params({"decision": choice, "n_bootstrap": n_boot})
            for name, value in (("pr_auc_k5", pr_k5), ("pr_auc_matched", pr_matched),
                                ("cost_k5", cf_k5.cost), ("cost_matched", cf_matched.cost),
                                ("pr_auc_diff_std", pr_boot["diff_std"]),
                                ("cost_diff_std", cost_boot["diff_std"])):  # fmt: skip
                mlflow.log_metric(name, float(value))
            mlflow.log_artifact(str(reports / SUMMARY_FILE))
            logger.info("Logged MLflow run matched_vs_k5_validation (%s)", active.info.run_id)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    setup_mlflow(config)
    run(config, track=not args.no_mlflow)


if __name__ == "__main__":
    main()
