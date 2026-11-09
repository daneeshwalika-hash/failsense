"""Headline result figures for the README, drawn from the saved report files.

Nothing is re-fitted: every figure reads a CSV / JSON that an earlier ``make`` target wrote to
``reports/`` (``make cv``/``compare``, ``make calibrate``, ``make q2``, ``make rul-baseline``), so
the figures can never disagree with the tables in ``docs/results.md``. Run: ``make figures``.

Design rules (one axis per chart, values labelled directly, the highlighted bar is the model the
text is about, everything else is neutral blue; grid and axes recessive). Figures are built with
``matplotlib.figure.Figure`` directly (no global state, works in CI and Docker).
"""

from __future__ import annotations

import json
import logging
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.figure import Figure

from failsense.config import load_config, resolve_path
from failsense.plotting import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

logger = logging.getLogger(__name__)

FIGURE_DIR = "figures"
Q1_MODEL_LABELS = {
    "logreg_reference": "Logistic regression", "knn_k101_distance": "kNN (k=101)",
    "gaussian_nb": "Gaussian naive Bayes", "tree_depth8": "Decision tree (depth 8)",
    "random_forest": "Random forest", "hist_gb": "HistGradientBoosting",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "xgboost__engineered": "XGBoost + engineered features",
}  # fmt: skip


def _need(path: Path, make_target: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make {make_target}` first")
    return path


def _style(ax: Any, grid_axis: str = "x") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    highlight: str | None = None,
    errors: Sequence[float] | None = None,
    value_format: str = "{:,.0f}",
    reference: tuple[float, str] | None = None,
    subtitle: str | None = None,
    lower_is_better: bool = False,
) -> Path:
    """Horizontal bars with the BEST value at the top; one bar may be highlighted.

    ``lower_is_better`` flips the order for costs and errors (best = smallest at the top).

    The values are written at the end of the bars. ``reference`` draws a dashed vertical line
    with a label (e.g. a baseline cost).
    """
    if len(labels) != len(values) or not len(labels):
        raise ValueError("labels and values must be non-empty and of equal length")
    if errors is not None and len(errors) != len(values):
        raise ValueError("errors must have one entry per value")
    if highlight is not None and highlight not in labels:
        raise ValueError(f"highlight {highlight!r} is not one of the labels")
    # matplotlib draws the first bar at the bottom: ascending puts the largest value on top
    order = np.argsort(values)[::-1] if lower_is_better else np.argsort(values)
    names = [labels[i] for i in order]
    vals = np.asarray([values[i] for i in order], dtype=float)
    errs = None if errors is None else np.asarray([errors[i] for i in order], dtype=float)
    colors = [COLOR_VAL if n == highlight else COLOR_TRAIN for n in names]

    fig = Figure(figsize=(8.5, 0.55 * len(names) + 1.9), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    ax.barh(names, vals, color=colors, height=0.6, xerr=errs,
            error_kw={"ecolor": INK_SECONDARY, "capsize": 3, "linewidth": 1})  # fmt: skip
    span = float(vals.max() + (0 if errs is None else errs.max()))
    for y, value in enumerate(vals):
        pad = 0.01 * span + (0 if errs is None else errs[y])
        ax.text(value + pad, y, value_format.format(value), va="center", fontsize=9,
                color=INK_PRIMARY)  # fmt: skip
    if reference is not None:
        ax.axvline(reference[0], color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.text(reference[0], len(names) - 0.4, f" {reference[1]}", color=INK_SECONDARY,
                fontsize=8, va="bottom")  # fmt: skip
    ax.set_xlim(0, span * 1.15)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    top = 0.93
    if subtitle:
        wrapped = textwrap.fill(subtitle, 115)
        fig.text(0.01, 0.93, wrapped, ha="left", va="top", fontsize=9, color=INK_SECONDARY)
        top = 0.93 - 0.035 * (wrapped.count("\n") + 1) - 0.02
    _style(ax)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def confusion_panels(tables: Mapping[str, pd.DataFrame], path: Path, title: str) -> Path:
    """Confusion matrices side by side on one colour scale (single-hue sequential)."""
    if not tables:
        raise ValueError("need at least one confusion table")
    top = max(float(t.to_numpy().max()) for t in tables.values())
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, COLOR_TRAIN])
    # A power scale keeps the small cells (16, 30, 76 trucks) visible next to the 4,336.
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=top)
    fig = Figure(figsize=(5.2 * len(tables), 5.0), dpi=150, facecolor=SURFACE)
    for ax, (name, table) in zip(fig.subplots(1, len(tables), squeeze=False)[0], tables.items(),
                                 strict=True):  # fmt: skip
        values = table.to_numpy()
        ax.imshow(values, cmap=cmap, norm=norm)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                dark = norm(values[i, j]) > 0.6
                ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", fontsize=9,
                        color=SURFACE if dark else INK_PRIMARY)  # fmt: skip
        ax.set_xticks(range(values.shape[1]), [c.replace("class ", "") for c in table.columns])
        ax.set_yticks(range(values.shape[0]), [r.replace("true ", "") for r in table.index])
        ax.set_xlabel("predicted class", color=INK_SECONDARY)
        ax.set_ylabel("true class", color=INK_SECONDARY)
        ax.set_title(name, loc="left", fontsize=10, color=INK_PRIMARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


# --- loaders: one per source file ---------------------------------------------------------------
def q1_model_comparison(reports: Path) -> tuple[list[str], list[float], list[float]]:
    """PR-AUC (mean, std) of the cross-validated Q1 models; the engineered XGBoost comes from
    the recheck file."""
    main = pd.read_csv(_need(reports / "compare_models_q1.csv", "compare"))
    recheck = pd.read_csv(_need(reports / "compare_models_q1_recheck.csv", "cv"))
    frame = pd.concat([main, recheck[recheck["model"] == "xgboost__engineered"]])
    frame = frame[frame["model"].isin(Q1_MODEL_LABELS)]
    return (
        [Q1_MODEL_LABELS[m] for m in frame["model"]],
        frame["pr_auc"].tolist(),
        frame["pr_auc_std"].tolist(),
    )


def q1_costs(reports: Path) -> dict[str, float]:
    """Total cost of the Q1 strategies on the official validation (sigmoid variant)."""
    table = pd.read_csv(_need(reports / "threshold_q1.csv", "calibrate"))
    row = table[table["variant"] == "sigmoid"].iloc[0]
    return {
        "Never alert": float(row["cost_never"]),
        "Alert the riskiest 5 %": float(row["cost_top_5%"]),
        "Alert every truck": float(row["cost_always"]),
        "Model, cost-based threshold (cross-fitted)": float(row["cost_cross_fitted"]),
    }


def q2_costs(reports: Path) -> tuple[dict[str, float], str, dict[str, pd.DataFrame]]:
    """Q2 costs (constants and decisions), the label of the corrected model and its confusions."""
    summary = json.loads(_need(reports / "q2_classes_summary.json", "q2").read_text("utf-8"))
    costs = {f"Always class {k[-1]}": v for k, v in summary["constant_baselines"].items()}
    names = {
        "uncorrected_argmax": "Model, arg-max class",
        "uncorrected_expected_cost": "Model, expected cost, uncorrected",
        "corrected_expected_cost": "Model, expected cost, prior-shift corrected",
    }
    for key, label in names.items():
        costs[label] = summary["decisions"][key]["total_cost"]
    tables = {
        title: pd.read_csv(_need(reports / f"q2_confusion_{key}.csv", "q2"), index_col=0)
        for title, key in (
            ("Uncorrected", "uncorrected_expected_cost"),
            ("Prior-shift corrected", "corrected_expected_cost"),
        )  # fmt: skip
    }
    return costs, names["corrected_expected_cost"], tables


def q3_rmse(reports: Path) -> tuple[list[str], list[float], list[float]]:
    """RMSE (mean, std over folds) of the FD001 models from the time-series run."""
    frame = pd.read_csv(_need(reports / "rul_timeseries_fd001.csv", "rul-timeseries"))
    frame = frame.set_index("model")
    picks = {
        "dummy_mean": "Mean predictor",
        "ridge_raw": "Ridge, sensors",
        "ridge_rolling": "Ridge, + rolling mean/std",
        "lightgbm_raw": "LightGBM, sensors",
        "lightgbm_rolling": "LightGBM, + rolling mean/std",
    }
    rows = [
        (label, frame.loc[m, "rmse"], frame.loc[m, "rmse_std"])
        for m, label in picks.items()
        if m in frame.index
    ]
    if not rows:
        raise ValueError("none of the expected RUL models is in the report file")
    labels, values, stds = zip(*rows, strict=True)
    return list(labels), list(values), list(stds)


# --- orchestration ------------------------------------------------------------------------------
def run(config: Mapping[str, Any]) -> list[Path]:
    """Draw all headline figures into ``reports/figures``; returns their paths."""
    reports = resolve_path(config["paths"]["reports"])
    out = reports / FIGURE_DIR
    paths = []

    labels, values, stds = q1_model_comparison(reports)
    paths.append(bar_chart(
        labels, values, out / "results_q1_model_comparison.png",
        "Q1: PR-AUC of the models (5-fold CV by truck, random cut-offs)",
        "PR-AUC (higher is better)",
        highlight="XGBoost + engineered features", errors=stds, value_format="{:.3f}",
        subtitle="Mean and standard deviation over 5 folds; the validation-like sets give lower "
                 "values (docs/results.md, section 3)."))  # fmt: skip

    costs = q1_costs(reports)
    paths.append(bar_chart(
        list(costs), list(costs.values()), out / "results_q1_cost.png",
        "Q1: total cost on the official validation (5,046 trucks)",
        "total cost (lower is better)",
        highlight="Model, cost-based threshold (cross-fitted)", lower_is_better=True,
        subtitle="SCANIA cost matrix, primary mapping. The saving depends on the mapping "
                 "(docs/results.md, section 5)."))  # fmt: skip

    q2, best, tables = q2_costs(reports)
    paths.append(bar_chart(
        list(q2), list(q2.values()), out / "results_q2_cost.png",
        "Q2: total cost of the 5 x 5 matrix on the official validation",
        "total cost (lower is better)",
        highlight=best, lower_is_better=True,
        subtitle="Cheapest constant predictor: 'always class 4'. The correction is within noise of "
                 "the uncorrected model (ADR 0004)."))  # fmt: skip
    paths.append(
        confusion_panels(
            tables,
            out / "results_q2_confusion.png",
            "Q2: expected-cost decisions put almost every truck into class 4",
        )
    )

    labels, values, stds = q3_rmse(reports)
    paths.append(
        bar_chart(
            labels,
            values,
            out / "results_q3_rmse.png",
            "Q3: remaining useful life, NASA C-MAPSS FD001 (GroupKFold by engine)",
            "RMSE in cycles (lower is better)",
            highlight="LightGBM, + rolling mean/std",
            errors=stds,
            lower_is_better=True,
            value_format="{:.1f}",
            subtitle=(
                "Training engines only, RMSE over all cycles (capped RUL), not the official test "
                "protocol; simulated data, test file untouched."
            ),
        )
    )
    return paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
