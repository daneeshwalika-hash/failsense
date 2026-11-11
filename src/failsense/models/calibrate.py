"""Calibration and cost-based threshold of the registered Q1 model on the OFFICIAL validation.

Run it with::

    uv run python -m failsense.models.calibrate      # or: make calibrate

What it does (the test split is never touched):

A. **Label-free calibrators (the real check of the 05.11 protocol).** The registered model
   (``failsense-q1@candidate``) scores the official validation. A sigmoid and an isotonic
   calibrator are fitted on **out-of-fold** scores of the calibrated validation-like evaluation
   set (training trucks only, same 5 folds as the tuning) and applied to the validation scores.
   Validation labels are used only to *score* them: Brier, log loss, mean probability vs true
   rate, reliability curve (quantile bins + zoom on p < 0.1), against the base-rate predictor.
B. **Reference:** calibrators cross-fitted on the validation itself (5 folds by truck).
C. **Threshold by total cost.** SCANIA cost matrix, binary reading from ``failsense.models.cost``.
   Reported cost = cross-fitted (threshold chosen on 4 folds, applied to the 5th). The in-sample
   optimum is shown separately and labelled optimistic. Baselines: never alert, always alert
   (primary), alert the top 5 %, theoretical threshold ``FP / (FP + mean(FN - TP))`` with the
   class mix of the calibrated evaluation set (training labels only).
D. One MLflow run, a summary JSON and CSV tables in ``reports/``, figures in ``reports/figures/``.

Which labels decide what: the THEORETICAL threshold uses the class mix of the evaluation set
(training labels only), and the validation class counts appear in the report for documentation
only. The DEPLOYED threshold (``deployment_choice``) is learned on the validation labels: the
true classes 0-4 give the per-truck costs, and the cost-optimal threshold on the whole validation
set is stored. That is why the cost to quote is the cross-fitted one.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score

from failsense.config import load_config, resolve_path
from failsense.data.labels import CLASS_COL, VEHICLE_COL
from failsense.models.boosters import fit_fixed_rounds
from failsense.models.calibration import (
    KINDS,
    constant_scores,
    cross_fit_calibrate,
    fit_calibrator,
    probability_metrics,
    reliability_table,
    sigmoid_parameters,
)
from failsense.models.cost import (
    alert_class_from_config,
    binary_costs,
    class_mix,
    cost_matrix_from_config,
    theoretical_threshold,
    total_cost,
)
from failsense.models.threshold import (
    apply_threshold,
    baseline_table,
    choose_threshold,
    cost_curve,
    cross_fit_threshold,
    decision_report,
    saving_summary,
)
from failsense.models.tuning_data import TuningData, build_tuning_data
from failsense.models.validation_data import (
    build_validation_frame,
    registry_input_columns,
    select_model_inputs,
)
from failsense.plotting import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

SUMMARY_FILE = "calibration_q1_summary.json"
SCORES_FILE = "calibration_q1_scores.csv"
THRESHOLD_FILE = "threshold_q1.csv"
RELIABILITY_FILE = "calibration_q1_reliability.csv"
FIG_RELIABILITY = "calibration_q1_reliability.png"
FIG_COST = "threshold_q1_cost_curve.png"
VARIANTS_A = ("raw", "sigmoid", "isotonic")  # label-free (fitted on the evaluation set)
VARIANTS_B = ("sigmoid_cv", "isotonic_cv")  # cross-fitted on the validation itself
LOGGED_COST_COLUMNS = (
    "cost_cross_fitted", "cost_always", "cost_never", "saving_total", "saving_per_truck",
    "cost_in_sample_optimistic", "cf_not_alerted",
)  # fmt: skip
SUMMARY_COLUMNS = (
    "variant", "cost_never", "cost_always", "cost_top_5%", "cost_theoretical",
    "cost_cross_fitted", "saving_total", "saving_per_truck", "saving_fold_mean",
    "saving_fold_std", "cf_alerted", "cf_not_alerted", "cost_in_sample_optimistic",
)  # fmt: skip
REFERENCE_FILE = "evaluate_q1.csv"
UNTUNED = "xgboost_step6_engineered"


def _json_safe(value: Any) -> Any:
    """Make numpy numbers, NaN and inf JSON-safe (non-finite numbers become null)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def oof_scores(
    config: Mapping[str, Any], data: TuningData, registry: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold scores on the calibrated evaluation set (same model recipe as the registry).

    Every fold fits the registered recipe (family, parameters, rounds) on the other folds'
    training trucks and scores the evaluation rows of its own trucks.

    Returns:
        ``(p, y, vehicles)`` pooled over the folds.
    """
    feature_set = "engineered" if registry["engineered"] else "base"
    if data.columns[feature_set] != registry["input_columns"]:
        raise ValueError("the tuning-data columns differ from the registered input columns")
    seed, n_jobs = int(config["tuning"]["seed"]), os.cpu_count() or 1
    scores, labels, vehicles = [], [], []
    for fold in data.folds:
        start = time.perf_counter()
        model = fit_fixed_rounds(
            registry["family"], registry["params"], fold.x_train[feature_set], fold.y_train,
            int(registry["n_rounds"]), seed, n_jobs,
        )  # fmt: skip
        scores.append(model.predict_proba(fold.x_eval["calibrated"][feature_set])[:, 1])
        labels.append(fold.y_eval["calibrated"])
        vehicles.append(fold.vehicles_eval["calibrated"])
        logger.info("OOF fold %d: %.1f s", fold.fold, time.perf_counter() - start)
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(vehicles)


def score_validation(
    config: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Scores of the REGISTERED model (loaded from the MLflow registry) on the validation frame."""
    frame = build_validation_frame(config)
    inputs = select_model_inputs(frame, registry_input_columns(config))
    model = mlflow.sklearn.load_model(registry["model_uri"])
    return frame, model.predict_proba(inputs)[:, 1]


def build_variants(
    p_eval: np.ndarray, y_eval: np.ndarray, p_val: np.ndarray, y_val: np.ndarray,
    vehicles_val: np.ndarray, n_splits: int, seed: int,
) -> dict[str, np.ndarray]:  # fmt: skip
    """The five probability variants on the validation trucks (A: label-free, B: cross-fitted)."""
    variants = {"raw": p_val}
    for kind in KINDS:
        variants[kind] = fit_calibrator(kind, p_eval, y_eval).predict(p_val)
    for kind in KINDS:
        variants[f"{kind}_cv"] = cross_fit_calibrate(
            kind, p_val, y_val, vehicles_val, n_splits, seed
        )
    return variants


def analyze(
    classes: np.ndarray,
    vehicles: np.ndarray,
    variants: Mapping[str, np.ndarray],
    eval_rate: float,
    eval_class_counts: Mapping[int, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Scores, thresholds, baselines and decision reports for every variant (pure function)."""
    cal = config["calibration"]
    n_splits, seed = int(cal["n_splits"]), int(cal["seed"])
    eps, top = float(cal["log_loss_eps"]), float(cal["top_fraction"])
    matrix = cost_matrix_from_config(config)
    alert_class = alert_class_from_config(config)
    primary_costs = binary_costs(matrix, alert_class, "primary")
    literal_costs = binary_costs(matrix, alert_class, "literal")
    y = (classes >= 1).astype(int)

    mix = class_mix(np.repeat(np.arange(1, 5), [int(eval_class_counts[k]) for k in range(1, 5)]))
    t_star = theoretical_threshold(primary_costs, mix)
    t_star_literal = theoretical_threshold(literal_costs, mix)

    scores = []
    for name, p in variants.items():
        scores.append({"variant": name, **probability_metrics(y, p, eps)})
    for name, rate in (("base_rate_eval_set", eval_rate), ("base_rate_validation", y.mean())):
        scores.append({"variant": name, "mean_proba": rate, "true_rate": y.mean(),
                       **constant_scores(y, rate, eps)})  # fmt: skip

    per_variant: dict[str, Any] = {}
    for name, p in variants.items():
        baselines = baseline_table(classes, p, primary_costs, top, t_star)
        cross = cross_fit_threshold(classes, p, vehicles, primary_costs, n_splits, seed)
        in_sample = choose_threshold(classes, p, primary_costs)
        in_sample_alerts = apply_threshold(p, in_sample.threshold)
        always = float(baselines.set_index("policy").loc["always_alert", "cost"])
        per_variant[name] = {
            "baselines": baselines,
            "cross_fit": cross,
            "cross_fit_saving": saving_summary(cross.folds, len(classes)),
            "cross_fit_saving_total_check": always - cross.cost,
            "in_sample": in_sample,
            "in_sample_decision": decision_report(classes, in_sample_alerts),
            "cross_fit_decision": decision_report(classes, cross.alerts),
        }

    # sensitivity: literal reading "alert = predicted class 1" (primary variant only)
    primary = str(cal["primary"])
    p_primary = variants[primary]
    literal_cross = cross_fit_threshold(classes, p_primary, vehicles, literal_costs, n_splits, seed)
    never = total_cost(classes, np.zeros(len(classes), dtype=bool), literal_costs)
    always_l = total_cost(classes, np.ones(len(classes), dtype=bool), literal_costs)
    sensitivity = {
        "variant": primary, "mapping": f"literal (alert = predicted class {alert_class})",
        "theoretical_threshold": t_star_literal, "never_alert": never, "always_alert": always_l,
        "cross_fit_cost": literal_cross.cost,
        "saving_vs_always": always_l - literal_cross.cost,
        "saving_vs_best_trivial": min(never, always_l) - literal_cross.cost,
        "cross_fit_saving": saving_summary(literal_cross.folds, len(classes)),
    }  # fmt: skip
    return {
        "scores": pd.DataFrame(scores), "per_variant": per_variant, "sensitivity": sensitivity,
        "class_mix_eval_set": mix, "theoretical_threshold": t_star,
        "validation_class_counts": {k: int((classes == k).sum()) for k in range(5)},
        "costs": primary_costs, "literal_costs": literal_costs,
    }  # fmt: skip


def threshold_rows(result: Mapping[str, Any], n_trucks: int) -> pd.DataFrame:
    """One row per variant: baselines, cross-fitted and in-sample cost, saving, missed trucks."""
    rows = []
    for name, pv in result["per_variant"].items():
        base = pv["baselines"].set_index("policy")["cost"]
        saving, cf, ins = pv["cross_fit_saving"], pv["cross_fit_decision"], pv["in_sample_decision"]
        row = {
            "variant": name,
            "cost_never": base["never_alert"], "cost_always": base["always_alert"],
            "cost_top_5%": base.get("top_5%", np.nan),
            "cost_theoretical": base["theoretical_threshold"],
            "cost_cross_fitted": pv["cross_fit"].cost,
            "saving_total": saving["total_saving"], "saving_per_truck": saving["saving_per_truck"],
            "saving_fold_mean": saving["fold_mean"], "saving_fold_std": saving["fold_std"],
            "folds_better": saving["folds_better"], "clear": saving["clear"],
            "cf_alerted": cf["n_alerted"], "cf_not_alerted": cf["n_not_alerted"],
            "cf_recall": cf["recall"],
            "cost_in_sample_optimistic": pv["in_sample"].cost,
            "in_sample_threshold": pv["in_sample"].threshold,
            "is_alerted": ins["n_alerted"], "is_not_alerted": ins["n_not_alerted"],
            "is_recall": ins["recall"],
        }  # fmt: skip
        for k in range(1, 5):
            row[f"cf_missed_c{k}"] = cf["not_alerted_positives"][k]
            row[f"cf_recall_c{k}"] = cf["per_class"][k]["recall"]
            row[f"is_missed_c{k}"] = ins["not_alerted_positives"][k]
            row[f"is_recall_c{k}"] = ins["per_class"][k]["recall"]
        rows.append(row)
    return pd.DataFrame(rows)


def deployment_record(
    calibrator: Any, result: Mapping[str, Any], primary: str, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """The choice that goes into the service on 07.11 (decided 06.11).

    Label-free sigmoid calibrator (fitted on out-of-fold scores of the calibrated evaluation
    set) + the threshold learned on the WHOLE validation set (primary mapping, TP = 0). The
    cost to quote is the cross-fitted one; the in-sample cost of the final threshold is
    optimistic. The positive result depends on the primary mapping.
    """
    pv = result["per_variant"][primary]
    base = pv["baselines"].set_index("policy")["cost"]
    saving, sens = pv["cross_fit_saving"], result["sensitivity"]
    return {
        "model_uri": registry["model_uri"],
        # the calibrator and the threshold belong to THIS version; the API refuses to serve another
        "model_version": registry["version"],
        "calibrator": sigmoid_parameters(calibrator),
        "calibrator_fitted_on": "out-of-fold scores of the calibrated validation-like set "
        "(training trucks, no validation labels)",
        "threshold": pv["in_sample"].threshold,
        "threshold_learned_on": "the whole official validation set (alert if calibrated p >= "
        "threshold)",
        "mapping": "primary (false alarm 7, miss 200-500, alert on a positive 0)",
        "reported_cost_cross_fitted": pv["cross_fit"].cost,
        "cost_in_sample_optimistic": pv["in_sample"].cost,
        "cost_always_alert": float(base["always_alert"]),
        "cost_never_alert": float(base["never_alert"]),
        "saving_vs_always_alert": saving["total_saving"],
        "folds_better": f"{saving['folds_better']}/{saving['n_folds']}",
        "caveat_literal_mapping": (
            f"under the stricter mapping (alert = predicted class 1) the model saves only "
            f"{sens['saving_vs_best_trivial']:,.0f} vs. never-alert; the positive result depends "
            f"on the primary mapping"
        ),
        "test_split": "untouched; used once at the very end",
    }


def verdict(result: Mapping[str, Any], primary: str) -> str:
    """Plain-language answer to "does the model beat always-alert?" (primary variant)."""
    s = result["per_variant"][primary]["cross_fit_saving"]
    text = (
        f"Primary variant {primary}: cross-fitted saving vs. always-alert "
        f"{s['total_saving']:+,.0f} in total ({s['saving_per_truck']:+.3f} per truck), per fold "
        f"{s['fold_mean']:+,.0f} +- {s['fold_std']:,.0f} (std over {s['n_folds']} folds), "
        f"better in {s['folds_better']}/{s['n_folds']} folds. "
    )
    if s["clear"]:
        return text + "The saving is larger than the fold spread: the model beats always-alert."
    return text + (
        "The saving is NOT larger than the fold spread: the model does NOT clearly beat "
        "always-alert on this validation split."
    )


# --- figures -----------------------------------------------------------------------------------


def _style(ax) -> None:  # noqa: ANN001 - matplotlib Axes
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)


def plot_reliability(
    y: np.ndarray, variants: Mapping[str, np.ndarray], primary: str, n_bins: int, zoom_max: float,
    path: Any,
) -> None:  # fmt: skip
    """Reliability curves (quantile bins): all trucks (left), zoom on p < ``zoom_max`` (right)."""
    style = {"raw": (INK_MUTED, "o"), "sigmoid": (COLOR_TRAIN, "s"), "isotonic": (COLOR_VAL, "^")}
    fig = Figure(figsize=(11, 4.8), facecolor=SURFACE)
    for ax, p_max, title in (
        (fig.add_subplot(1, 2, 1), None, "All trucks (10 quantile bins)"),
        (fig.add_subplot(1, 2, 2), zoom_max, f"Zoom: p < {zoom_max} (bins of the trucks below)"),
    ):
        _style(ax)
        top = 0.0
        for name in VARIANTS_A:
            table = reliability_table(y, variants[name], n_bins, p_max)
            colour, marker = style[name]
            if name == primary:  # intervals only for the headline variant (readability)
                err = np.vstack([table.observed - table.ci_low, table.ci_high - table.observed])
                ax.errorbar(table.mean_pred, table.observed, yerr=err, fmt="none",
                            ecolor=colour, alpha=0.35, capsize=2)  # fmt: skip
            ax.plot(table.mean_pred, table.observed, marker=marker, color=colour, label=name,
                    linewidth=1.4, markersize=5)  # fmt: skip
            top = max(top, table.mean_pred.max(), table.observed.max())
        limit = top * 1.08 if p_max is None else max(top * 1.08, 0.001)
        ax.plot([0, limit], [0, limit], "--", color=INK_PRIMARY, linewidth=1, label="perfect")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit if p_max is not None else limit)
        ax.set_xlabel("mean predicted probability", color=INK_SECONDARY)
        ax.set_ylabel("observed repair rate", color=INK_SECONDARY)
        ax.set_title(title, color=INK_PRIMARY, fontsize=10)
    fig.axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Reliability on the official validation split (bars: 95 % Wilson, primary)",
                 color=INK_PRIMARY, fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, dpi=130)


def plot_cost_curve(
    classes: np.ndarray, p: np.ndarray, result: Mapping[str, Any], primary: str, path: Any
) -> None:  # fmt: skip
    """Total cost over all thresholds (log x), with baselines, t*, and the optimistic optimum."""
    pv = result["per_variant"][primary]
    curve = cost_curve(classes, p, result["costs"])
    inner = curve[np.isfinite(curve.threshold) & (curve.threshold > 0)]
    fig = Figure(figsize=(8, 4.8), facecolor=SURFACE)
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    ax.plot(inner.threshold, inner.cost, color=COLOR_TRAIN, linewidth=1.6, label="total cost")
    base = pv["baselines"].set_index("policy")["cost"]
    ax.axhline(base["always_alert"], color=COLOR_VAL, linestyle="--", label="always alert")
    ax.axhline(base["never_alert"], color=INK_MUTED, linestyle="--", label="never alert")
    ax.axvline(result["theoretical_threshold"], color=INK_SECONDARY, linestyle=":",
               label=f"theoretical t* = {result['theoretical_threshold']:.4f}")  # fmt: skip
    best = pv["in_sample"]
    if np.isfinite(best.threshold) and best.threshold > 0:
        ax.plot([best.threshold], [best.cost], "o", color=INK_PRIMARY,
                label="in-sample optimum (optimistic)")  # fmt: skip
    ax.axhline(pv["cross_fit"].cost, color=COLOR_TRAIN, linestyle="-.", linewidth=1,
               label="cross-fitted cost")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel(f"alert if calibrated probability ({primary}) >= threshold", color=INK_SECONDARY)
    ax.set_ylabel("total cost on validation", color=INK_SECONDARY)
    ax.set_title("Cost over thresholds (official validation)", color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, dpi=130)


# --- orchestration -----------------------------------------------------------------------------


def bootstrap_pr_auc(
    y: np.ndarray, p: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:  # fmt: skip
    """95 % interval of PR-AUC by resampling trucks (draws without a positive are skipped)."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].any():
            values.append(average_precision_score(y[idx], p[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def reference_estimates(config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """PR-AUC of the untuned model on the two 05.11 evaluation sets (from evaluate_q1.csv)."""
    path = resolve_path(config["paths"]["reports"]) / REFERENCE_FILE
    table = pd.read_csv(path)
    rows = table[table["model"] == UNTUNED].set_index("eval_set")
    return {
        m: {"pr_auc": float(r["pr_auc"]), "std": float(r["pr_auc_std"])} for m, r in rows.iterrows()
    }


def run(config: Mapping[str, Any], track: bool = True) -> dict[str, Any]:
    """Whole step: OOF scores -> calibrators -> validation scoring -> thresholds -> reports."""
    cal = config["calibration"]
    reports = resolve_path(config["paths"]["reports"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    registry = json.loads((reports / "registry_q1.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()

    data = build_tuning_data(config, ("calibrated",))
    logger.info("Tuning data built in %.0f s", time.perf_counter() - t0)
    p_eval, y_eval, _ = oof_scores(config, data, registry)
    eval_info = data.eval_info["calibrated"]
    eval_counts = {k: eval_info[f"n_class_{k}"] for k in range(5)}

    frame, p_val = score_validation(config, registry)
    classes = frame[CLASS_COL].to_numpy(dtype=int)
    vehicles = frame[VEHICLE_COL].to_numpy()
    y_val = (classes >= 1).astype(int)
    n_splits, seed = int(cal["n_splits"]), int(cal["seed"])
    variants = build_variants(p_eval, y_eval, p_val, y_val, vehicles, n_splits, seed)
    primary_calibrator = fit_calibrator(str(cal["primary"]), p_eval, y_eval)

    result = analyze(classes, vehicles, variants, float(y_eval.mean()), eval_counts, config)
    primary = str(cal["primary"])
    table = threshold_rows(result, len(classes))
    scores = result["scores"]
    reliability = pd.concat(
        [reliability_table(y_val, variants[v], int(cal["n_bins"])).assign(variant=v, view="all")
         for v in VARIANTS_A]
        + [reliability_table(y_val, variants[v], int(cal["n_bins"]), float(cal["zoom_max"]))
           .assign(variant=v, view=f"p<{cal['zoom_max']}") for v in VARIANTS_A]
    )  # fmt: skip
    ci = bootstrap_pr_auc(y_val, p_val, int(cal["n_bootstrap"]), seed)
    reference = reference_estimates(config)
    text = verdict(result, primary)

    scores.to_csv(reports / SCORES_FILE, index=False, float_format="%.5f")
    table.to_csv(reports / THRESHOLD_FILE, index=False, float_format="%.5f")
    reliability.to_csv(reports / RELIABILITY_FILE, index=False, float_format="%.5f")
    plot_reliability(y_val, variants, primary, int(cal["n_bins"]), float(cal["zoom_max"]),
                     figures / FIG_RELIABILITY)  # fmt: skip
    plot_cost_curve(classes, variants[primary], result, primary, figures / FIG_COST)

    pv = result["per_variant"][primary]
    by_variant = scores.set_index("variant")
    summary = {
        "model": registry["model_uri"], "model_version": registry["version"],
        "primary_variant": primary, "n_trucks": len(classes), "n_positives": int(y_val.sum()),
        "validation_class_counts_documentation_only": result["validation_class_counts"],
        "eval_set": {"n": len(y_eval), "positive_rate": float(y_eval.mean()),
                     "class_counts": eval_counts,
                     "class_mix": result["class_mix_eval_set"]},  # fmt: skip
        "theoretical_threshold_primary": result["theoretical_threshold"],
        "binary_costs_primary": {"fp": result["costs"].fp, "fn": result["costs"].fn,
                                 "tp": result["costs"].tp},  # fmt: skip
        "ranking": {"pr_auc": float(by_variant.loc["raw", "pr_auc"]),
                    "pr_auc_95ci": ci, "roc_auc": float(by_variant.loc["raw", "roc_auc"]),
                    "prevalence": float(y_val.mean()),
                    "estimate_0511": reference},  # fmt: skip
        "scores": scores.to_dict("records"),
        "threshold_table": table.to_dict("records"),
        "primary_baselines": pv["baselines"].to_dict("records"),
        "primary_cross_fit_folds": pv["cross_fit"].folds.to_dict("records"),
        "primary_decision_cross_fitted": pv["cross_fit_decision"],
        "primary_decision_in_sample_optimistic": pv["in_sample_decision"],
        "sensitivity_literal": result["sensitivity"],
        "deployment_choice": deployment_record(primary_calibrator, result, primary, registry),
        "verdict": text,
        "timing_seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (reports / SUMMARY_FILE).write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    log_summary(scores, table, summary, text)

    if track:
        with mlflow.start_run(run_name="calibrate_threshold_q1") as active:
            mlflow.set_tags({"task": "q1_binary", "stage": "calibration", "split": "validation"})
            mlflow.log_params(
                {
                    "model_uri": registry["model_uri"], "primary": primary,
                    "n_splits": n_splits, "seed": seed,
                    "theoretical_threshold": result["theoretical_threshold"],
                }
            )  # fmt: skip
            for r in scores.to_dict("records"):
                for k in ("brier", "log_loss", "mean_proba", "pr_auc", "roc_auc"):
                    if k in r and pd.notna(r[k]):
                        mlflow.log_metric(f"{r['variant']}_{k}", float(r[k]))
            for r in table.to_dict("records"):
                for k in LOGGED_COST_COLUMNS:
                    mlflow.log_metric(f"{r['variant']}_{k}", float(r[k]))
            for path in (SUMMARY_FILE, SCORES_FILE, THRESHOLD_FILE, RELIABILITY_FILE):
                mlflow.log_artifact(str(reports / path))
            for path in (FIG_RELIABILITY, FIG_COST):
                mlflow.log_artifact(str(figures / path))
            logger.info("Logged MLflow run calibrate_threshold_q1 (%s)", active.info.run_id)
    return summary


def log_summary(
    scores: pd.DataFrame, table: pd.DataFrame, summary: Mapping[str, Any], text: str
) -> None:
    """Print the result tables to the log (the same numbers go to the files and to MLflow)."""

    def fmt(v: float) -> str:
        return f"{v:,.4f}"

    logger.info("Probability scores:\n%s", scores.to_string(index=False, float_format=fmt))
    shown = table[list(SUMMARY_COLUMNS)]
    logger.info("Thresholds by cost:\n%s", shown.to_string(index=False, float_format=fmt))
    logger.info("Ranking: %s", summary["ranking"])
    logger.info("VERDICT: %s", text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    setup_mlflow(config)  # also needed to load the registered model
    run(config, track=not args.no_mlflow)


if __name__ == "__main__":
    main()
