"""SHAP, permutation and gain importance of ``failsense-q1@candidate`` on the official validation.

Run it with ``uv run python -m failsense.explain.explain_q1`` (``make explain``; needs
``make calibrate`` for the deployed calibrator and threshold). The test split is not touched.

* **SHAP** (no labels needed) on the 5,046 validation last readouts, ``TreeExplainer`` with the
  default ``tree_path_dependent`` perturbation (see ``shap_utils`` for the limitation). Values are
  raw log-odds of the UNCALIBRATED model; the calibrated contribution is ``slope * shap``.
* **Permutation importance** (needs labels, so evaluation only, never feature selection): PR-AUC
  drop on the validation, 20 repeats, mean and spread; for single columns and for groups of
  correlated columns (all columns of one variable, shuffled jointly).
* **Gain importance** of XGBoost as the biased reference.
* Two waterfall stories, rows fixed by a rule before looking at the data: the highest-calibrated-p
  true positive and the lowest-scoring class-4 truck below the deployed threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss

from failsense.config import load_config, resolve_path
from failsense.data.labels import CLASS_COL, VEHICLE_COL
from failsense.explain import plots
from failsense.explain.importance import (
    aggregate,
    feature_groups,
    gain_importance,
    group_name,
    permutation_importance,
    variable_of,
)
from failsense.explain.shap_utils import ShapExplainer, top_shap_factors
from failsense.models.calibrate import SUMMARY_FILE as CALIBRATION_SUMMARY
from failsense.models.calibration import apply_sigmoid_parameters
from failsense.models.validation_data import (
    build_validation_frame,
    registry_input_columns,
    select_model_inputs,
)
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

SUMMARY_FILE = "explain_q1_summary.json"
GROUPS_FILE = "explain_q1_groups.csv"
COLUMNS_FILE = "explain_q1_columns.csv"


def pick_story_rows(
    classes: np.ndarray, p_cal: np.ndarray, threshold: float
) -> tuple[int, int, str]:
    """Positions of the two waterfall rows (rule fixed in advance).

    * TP: the true positive (class >= 1) with the highest calibrated probability.
    * FN: the lowest-scoring class-4 truck below the deployed threshold (the worst miss); if no
      class-4 truck is missed, the lowest-scoring missed positive of any class.

    Ties go to the first row (stable order). Returns ``(tp_position, fn_position, note)``.

    Raises:
        ValueError: If there is no positive truck or no missed positive.
    """
    classes, p_cal = np.asarray(classes), np.asarray(p_cal, dtype=float)
    positives = np.flatnonzero(classes >= 1)
    if positives.size == 0:
        raise ValueError("no positive truck")
    tp = int(positives[np.argmax(p_cal[positives])])
    missed = positives[p_cal[positives] < threshold]
    if missed.size == 0:
        raise ValueError("no missed positive below the threshold")
    class4 = missed[classes[missed] == 4]
    pool, note = (class4, "class-4 miss") if class4.size else (missed, "no class-4 miss: any class")
    fn = int(pool[np.argmin(p_cal[pool])])
    return tp, fn, note


def neg_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Negative log loss (higher is better, as ``permutation_importance`` expects).

    Probabilities are clipped to [1e-6, 1 - 1e-6] like in the calibration step.
    """
    return -float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def group_shap(by_input: pd.DataFrame, groups: Mapping[str, list[str]]) -> pd.Series:
    """Mean |SHAP| of a group: per truck the group's SHAP values are added up, then |.| averaged.

    Adding first keeps additivity (the group value is the group's contribution to the margin)
    and lets opposite contributions inside a group cancel.
    """
    return pd.Series(
        {name: float(by_input[cols].sum(axis=1).abs().mean()) for name, cols in groups.items()}
    )


def compare_rankings(table: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """Where grouped permutation and grouped mean |SHAP| agree and disagree.

    Only groups whose permutation drop is reliable (``mean - 2 std > 0``) are ranked by
    permutation; groups inside the spread are listed separately and never ranked.
    """
    reliable = table[table["reliable"]].sort_values("mean_drop", ascending=False)
    perm_top = list(reliable["group"].head(top_n))
    shap_top = list(table.sort_values("shap_mean_abs", ascending=False)["group"].head(top_n))
    rho = float(spearmanr(table["mean_drop"], table["shap_mean_abs"]).statistic)
    return {
        "permutation_top_reliable": perm_top,
        "shap_top": shap_top,
        "in_both": [g for g in perm_top if g in shap_top],
        "only_permutation": [g for g in perm_top if g not in shap_top],
        "only_shap": [g for g in shap_top if g not in perm_top],
        "n_reliable_groups": int(len(reliable)),
        "inside_spread": sorted(table.loc[~table["reliable"], "group"]),
        "spearman_all_groups": rho,
    }


def story(
    explainer: ShapExplainer,
    row: pd.Series,
    slope: float,
    k: int,
    p_raw: float,
    p_cal: float,
    label: str,
) -> dict[str, Any]:
    """One truck: scores, the top-k factors (raw and calibrated log-odds) and the rest."""
    factors = top_shap_factors(row, k=k, explainer=explainer)
    frame = explainer.prepare(row)
    values, _ = explainer.shap_values(frame)
    per_input = explainer.by_input(values).iloc[0]
    others = float(per_input.sum() - sum(f[2] for f in factors))
    return {
        "label": label,
        "p_raw": p_raw,
        "p_calibrated": p_cal,
        "margin": float(explainer.margin(frame)[0]),
        "base_value": explainer.base_value,
        "factors": [
            {"feature": f, "value": v, "shap_raw": s, "shap_calibrated": s * slope}
            for f, v, s in factors
        ],
        "all_other_features_raw": others,
    }


def run(config: Mapping[str, Any], track: bool = True) -> dict[str, Any]:
    """Compute and write everything; returns the summary dict."""
    cfg = config["explain"]
    reports = resolve_path(config["paths"]["reports"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    registry = json.loads((reports / "registry_q1.json").read_text(encoding="utf-8"))
    summary_path = reports / CALIBRATION_SUMMARY
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run `make calibrate` first")
    deployment = json.loads(summary_path.read_text(encoding="utf-8"))["deployment_choice"]
    calibrator, threshold = deployment["calibrator"], float(deployment["threshold"])
    slope = float(calibrator["slope"])

    frame = build_validation_frame(config)
    columns = registry_input_columns(config)
    x = select_model_inputs(frame, columns).reset_index(drop=True)
    classes = frame[CLASS_COL].to_numpy(dtype=int)
    y = (classes >= 1).astype(int)

    pipeline = mlflow.sklearn.load_model(registry["model_uri"])
    explainer = ShapExplainer(pipeline, columns)
    x = explainer.prepare(x)
    values, z = explainer.shap_values(x)
    by_input = explainer.by_input(values)
    p_raw = pipeline.predict_proba(x)[:, 1]
    p_cal = apply_sigmoid_parameters(calibrator, p_raw)
    logger.info("SHAP for %d trucks done in %.1f s", len(x), time.perf_counter() - t0)

    groups = feature_groups(columns)
    single = {c: [c] for c in columns}

    def predict(f: pd.DataFrame) -> np.ndarray:
        return pipeline.predict_proba(f)[:, 1]

    n_rep, seed = int(cfg["n_repeats"]), int(cfg["seed"])
    start = time.perf_counter()
    perm_groups = permutation_importance(predict, x, y, groups, n_rep, seed)
    logger.info("Grouped permutation: %.1f s", time.perf_counter() - start)
    # same groups and seed (= the same shuffles), scored by the log loss of CALIBRATED probabilities
    start = time.perf_counter()
    perm_ll = permutation_importance(
        lambda f: apply_sigmoid_parameters(calibrator, predict(f)), x, y, groups, n_rep, seed,
        metric=neg_log_loss,
    )  # fmt: skip
    logger.info("Grouped permutation (log loss): %.1f s", time.perf_counter() - start)
    start = time.perf_counter()
    perm_single = permutation_importance(predict, x, y, single, n_rep, seed)
    logger.info("Single-column permutation: %.1f s", time.perf_counter() - start)

    gain_model = gain_importance(explainer.model, explainer.feature_names)
    gain_by_input = aggregate(gain_model.to_numpy(), explainer.origin)
    shap_by_group = group_shap(by_input, groups)

    table = perm_groups.set_index("group")
    ll = perm_ll.set_index("group")
    table["ll_baseline"] = -ll["baseline"]  # the log loss itself (lower is better)
    table["ll_mean_increase"] = ll["mean_drop"]  # rise of the log loss when shuffled
    table["ll_std"] = ll["std_drop"]
    table["ll_reliable"] = ll["reliable"]
    table["shap_mean_abs"] = shap_by_group
    table["shap_share"] = table["shap_mean_abs"] / table["shap_mean_abs"].sum()
    table["gain"] = pd.Series({g: float(gain_by_input.reindex(c).sum()) for g, c in groups.items()})
    table["gain_share"] = table["gain"] / table["gain"].sum()
    table = table.reset_index()
    for col, rank_of in (("perm_rank", "mean_drop"), ("shap_rank", "shap_mean_abs"),
                         ("gain_rank", "gain")):  # fmt: skip
        table[col] = table[rank_of].rank(ascending=False).astype(int)
    table.loc[~table["reliable"], "perm_rank"] = np.nan  # groups inside the spread are not ranked
    comparison = compare_rankings(table, int(cfg["top_groups"]))

    cols_table = perm_single.rename(columns={"group": "column"}).set_index("column")
    cols_table["shap_mean_abs"] = by_input.abs().mean()
    cols_table["gain"] = gain_by_input.reindex(cols_table.index)
    cols_table["group"] = [group_name(variable_of(c)) for c in cols_table.index]
    cols_table = cols_table.reset_index()

    tp_pos, fn_pos, fn_note = pick_story_rows(classes, p_cal, threshold)
    k = int(cfg["story_k"])
    rows = {"true_positive": tp_pos, "missed_positive": fn_pos}
    stories = {}
    for name, pos in rows.items():
        info = story(explainer, x.iloc[pos], slope, k, float(p_raw[pos]), float(p_cal[pos]), name)
        info |= {"vehicle_id": int(frame[VEHICLE_COL].iloc[pos]), "true_class": int(classes[pos]),
                 "time_step": float(x["time_step"].iloc[pos]), "position": int(pos)}  # fmt: skip
        stories[name] = info
    stories["missed_positive"]["selection_note"] = fn_note

    table.to_csv(reports / GROUPS_FILE, index=False, float_format="%.6f")
    cols_table.to_csv(reports / COLUMNS_FILE, index=False, float_format="%.6f")

    # --- figures -----------------------------------------------------------------------------
    expl = shap.Explanation(
        values=values, base_values=np.full(len(values), explainer.base_value), data=z,
        feature_names=explainer.feature_names,
    )  # fmt: skip
    max_display, wf = int(cfg["max_display"]), int(cfg["waterfall_display"])
    plots.shap_beeswarm(expl, figures / "explain_q1_shap_beeswarm.png", max_display)
    plots.shap_bar(expl, figures / "explain_q1_shap_bar.png", max_display)
    titles = {"true_positive": "Highest-risk true positive", "missed_positive": "Worst miss"}
    for name, pos in rows.items():
        s = stories[name]
        plots.shap_waterfall(
            expl[pos], figures / f"explain_q1_waterfall_{name}.png", wf,
            f"{titles[name]}: truck {s['vehicle_id']}, class {s['true_class']}, "
            f"calibrated p = {s['p_calibrated']:.3f} (threshold {threshold:.4f})",
        )  # fmt: skip
    plots.shap_dependence(
        expl, "num__time_step", figures / "explain_q1_shap_dependence_time_step.png"
    )
    plots.plot_permutation(perm_groups, figures / "explain_q1_permutation_groups.png",
                           "Grouped permutation importance (validation, PR-AUC)")  # fmt: skip
    plots.plot_permutation(perm_single, figures / "explain_q1_permutation_columns.png",
                           "Single-column permutation importance, top 20", top=20)  # fmt: skip
    plots.plot_group_comparison(table, figures / "explain_q1_group_comparison.png")

    summary = {
        "model": registry["model_uri"], "n_trucks": len(x), "n_positives": int(y.sum()),
        "baseline_pr_auc": float(average_precision_score(y, p_raw)),
        "units": "SHAP in raw log-odds of the uncalibrated model; calibrated = slope * shap",
        "base_value": explainer.base_value,
        "base_probability_raw": float(1 / (1 + np.exp(-explainer.base_value))),
        "calibrator_slope": slope, "deployed_threshold": threshold,
        "shap_explainer": "TreeExplainer, feature_perturbation=tree_path_dependent (no background)",
        "n_repeats": n_rep, "reliable_rule": "mean_drop - 2 * std_drop > 0",
        "groups": table.to_dict("records"),
        "comparison": comparison,
        "stories": stories,
        "timing_seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (reports / SUMMARY_FILE).write_text(
        json.dumps(_clean(summary), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Group table:\n%s", table.round(4).to_string(index=False))
    logger.info("Comparison: %s", comparison)

    if track:
        with mlflow.start_run(run_name="explain_q1") as active:
            mlflow.set_tags({"task": "q1_binary", "stage": "explain", "split": "validation"})
            mlflow.log_params({"model_uri": registry["model_uri"], "n_repeats": n_rep, "seed": seed,
                               "shap_perturbation": "tree_path_dependent"})  # fmt: skip
            mlflow.log_metric("baseline_pr_auc", summary["baseline_pr_auc"])
            for r in table.to_dict("records"):
                tag = r["group"].replace(" ", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(f"perm_drop_{tag}", r["mean_drop"])
                mlflow.log_metric(f"shap_mean_abs_{tag}", r["shap_mean_abs"])
            for name in (SUMMARY_FILE, GROUPS_FILE, COLUMNS_FILE):
                mlflow.log_artifact(str(reports / name))
            for path in sorted(figures.glob("explain_q1_*.png")):
                mlflow.log_artifact(str(path))
            logger.info("Logged MLflow run explain_q1 (%s)", active.info.run_id)
    return summary


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy numbers to Python, NaN / inf to null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
