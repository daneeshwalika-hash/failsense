"""Outlier analysis of the Q1 training cut-offs (training split only).

Run it with::

    uv run python -m failsense.analysis.outliers

Why look at outliers at all
---------------------------
The counters are cumulative and the histograms are counts, so they are
extremely right-skewed: most trucks have moderate values, a few old or heavily
used trucks have values far above the rest. Two very different things can hide
in such tails:

* **errors** (sensor glitches, resets) - they should be fixed or removed;
* **failure signal** (a worn truck really is extreme) - removing or clipping
  them would delete exactly what the model needs.

The data cannot tell us which one it is by itself, so we measure it: if the
repair rate in the outlier rows is clearly above the rate in the normal rows,
the tail carries signal. A tail with a normal repair rate is harmless noise.

What the analysis does *not* do
-------------------------------
No row is removed and nothing is clipped. The fence (Q3 + 3 * IQR) is computed
from the sampled training cut-offs without looking at labels and is used only
to **describe** the data. Any transformation that goes into a model (``log1p``)
sits inside the Pipeline and has no threshold. Thresholds chosen by looking at
validation scores would be tuning on validation.

Outputs (small, no raw data; committed for the docs):

* ``reports/outliers_q1_features.csv``  - one row per feature
* ``reports/outliers_q1_repair_rate.csv`` - outlier vs. normal rows, top features
* ``reports/outliers_q1_summary.md``
* ``reports/figures/outliers_q1_skew_vs_share.png``
* ``reports/figures/outliers_q1_repair_rate.png``
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from failsense.config import load_config, resolve_path
from failsense.data.labels import BINARY_COL
from failsense.data.sampling import sampling_params_from_config
from failsense.features.build import engineered_columns
from failsense.models.baseline_cv import load_training_frame, n_cutoffs_label
from failsense.models.learning_curve import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)
from failsense.models.pipeline import assert_no_forbidden_features, select_feature_columns

logger = logging.getLogger(__name__)

# "Far out" fence of Tukey's rule: Q3 + 3 * IQR (1.5 would be "outside", 3 is "far out").
FENCE_FACTOR = 3.0
# A repair rate from fewer outlier rows than this is too noisy to interpret.
MIN_OUTLIER_ROWS = 30
TOP_N_FEATURES = 10

SUMMARY_COLUMNS = [
    "feature",
    "n",
    "n_missing",
    "p50",
    "p99",
    "p999",
    "max",
    "skew",
    "iqr",
    "n_above_fence",
    "share_above_fence",
]


def summarize_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Describe the tail of every feature: quantiles, max, skewness, outlier share.

    Args:
        df: Rows (the sampled cut-offs).
        columns: Numeric columns to describe. Label columns and ``vehicle_id``
            are refused (they are never features).

    Returns:
        One row per feature with ``SUMMARY_COLUMNS``. ``share_above_fence`` is
        the share of non-missing rows above Q3 + 3 * IQR. If IQR = 0 (typical
        for zero-heavy histogram bins) the fence equals Q3 and *every* value
        above it would count as an outlier, which is misleading; the share is
        then ``NaN`` and ``n_above_fence`` is 0. Constant columns, a single row
        or an empty frame give ``NaN`` statistics, not an error.

    Raises:
        ValueError: If a column is missing or forbidden.
    """
    columns = list(columns)
    assert_no_forbidden_features(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns: {missing[:5]}")

    x = df[columns].astype(float)
    q = x.quantile([0.25, 0.5, 0.75, 0.99, 0.999])  # NaN values are skipped
    q1, q2, q3, p99, p999 = (q.loc[v] for v in (0.25, 0.5, 0.75, 0.99, 0.999))
    iqr = q3 - q1
    fence = q3 + FENCE_FACTOR * iqr
    n_valid = x.notna().sum()
    n_above = x.gt(fence, axis="columns").sum()
    usable = (iqr > 0) & (n_valid > 0)
    n_above = n_above.where(usable, 0)
    share = (n_above / n_valid.replace(0, np.nan)).where(usable)

    out = pd.DataFrame(
        {
            "feature": columns,
            "n": n_valid.to_numpy(),
            "n_missing": (len(df) - n_valid).to_numpy(),
            "p50": q2.to_numpy(),
            "p99": p99.to_numpy(),
            "p999": p999.to_numpy(),
            "max": x.max().to_numpy(),
            "skew": x.skew().to_numpy(),
            "iqr": iqr.to_numpy(),
            "n_above_fence": n_above.to_numpy(dtype=int),
            "share_above_fence": share.to_numpy(),
        }
    )
    return out[SUMMARY_COLUMNS]


def select_top_features(
    summary: pd.DataFrame, n: int = TOP_N_FEATURES, min_outliers: int = MIN_OUTLIER_ROWS
) -> list[str]:
    """Features with the highest outlier share that can be interpreted.

    Only features with a defined fence (IQR > 0) and at least ``min_outliers``
    outlier rows qualify; with fewer rows the repair rate is pure noise.
    """
    ok = summary[summary["share_above_fence"].notna() & (summary["n_above_fence"] >= min_outliers)]
    top = ok.sort_values("share_above_fence", ascending=False).head(n)
    return top["feature"].tolist()


def repair_rate_by_outlier(
    df: pd.DataFrame,
    features: Sequence[str],
    label_col: str = BINARY_COL,
    min_outliers: int = MIN_OUTLIER_ROWS,
) -> pd.DataFrame:
    """Repair rate in outlier rows vs. normal rows, per feature.

    Outlier = above Q3 + 3 * IQR of the feature (same fence as
    ``summarize_features``); normal = at or below it; NaN rows are in neither.
    ``ratio`` > 1 means the tail has more failures than the rest: signal, not
    error. Features without a fence (IQR = 0) or with fewer than
    ``min_outliers`` outlier rows are skipped.

    Returns:
        One row per remaining feature: ``feature``, ``n_outlier``,
        ``rate_outlier``, ``n_normal``, ``rate_normal``, ``ratio`` (``NaN`` if
        the normal rate is 0).

    Raises:
        ValueError: If a column is missing or forbidden, or ``label_col`` is
            not in ``df`` or not 0/1.
    """
    features = list(features)
    assert_no_forbidden_features(features)  # the label is only the target, not an input
    missing = [c for c in (label_col, *features) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns: {missing[:5]}")
    y = df[label_col].to_numpy()
    if len(y) and not np.isin(y, [0, 1]).all():
        raise ValueError(f"{label_col!r} must contain only 0 and 1")

    rows = []
    for feature in features:
        x = df[feature].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not iqr > 0:
            continue
        fence = q3 + FENCE_FACTOR * iqr
        outlier = (x > fence).to_numpy()
        normal = (x <= fence).to_numpy()
        if outlier.sum() < min_outliers or normal.sum() == 0:
            continue
        rate_out = float(y[outlier].mean())
        rate_norm = float(y[normal].mean())
        rows.append(
            {
                "feature": feature,
                "n_outlier": int(outlier.sum()),
                "rate_outlier": rate_out,
                "n_normal": int(normal.sum()),
                "rate_normal": rate_norm,
                "ratio": rate_out / rate_norm if rate_norm > 0 else np.nan,
            }
        )
    columns = ["feature", "n_outlier", "rate_outlier", "n_normal", "rate_normal", "ratio"]
    return pd.DataFrame(rows, columns=columns)


def render_markdown(
    summary: pd.DataFrame, repair: pd.DataFrame, base_rate: float, n_rows: int, k_label: str
) -> str:
    """Small Markdown report: the most skewed features and the repair-rate table."""
    skewed = summary.dropna(subset=["skew"]).sort_values("skew", ascending=False).head(5)
    lines = [
        "# Outlier analysis - Q1 training cut-offs",
        "",
        f"{n_rows:,} sampled cut-offs (k = {k_label} per truck), training split only. "
        f"Overall repair rate (label = repair within the horizon): **{base_rate:.4f}**.",
        f"Outlier = above Q3 + {FENCE_FACTOR:g} x IQR. Nothing was removed or clipped.",
        "",
        "## Most skewed features",
        "",
        "| feature | skew | p50 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for r in skewed.itertuples():
        lines.append(f"| {r.feature} | {r.skew:.1f} | {r.p50:.3g} | {r.p99:.3g} | {r.max:.3g} |")
    lines += [
        "",
        f"## Repair rate in outlier vs. normal rows (top {len(repair)} by outlier share)",
        "",
        "| feature | outlier rows | rate outlier | rate normal | ratio |",
        "|---|---|---|---|---|",
    ]
    for r in repair.itertuples():
        ratio = "n/a" if pd.isna(r.ratio) else f"{r.ratio:.2f}"
        lines.append(
            f"| {r.feature} | {r.n_outlier:,} | {r.rate_outlier:.4f} "
            f"| {r.rate_normal:.4f} | {ratio} |"
        )
    lines.append("")
    return "\n".join(lines)


def _style_axes(ax) -> None:  # noqa: ANN001 - matplotlib Axes
    """Same quiet look as the learning-curve figure."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_skew_vs_share(summary: pd.DataFrame, path: Path) -> Path:
    """Scatter: skewness vs. outlier share per feature (raw vs. engineered).

    Only features with a defined fence (IQR > 0) are drawn. Uses
    ``matplotlib.figure.Figure`` directly (no global state, works in CI).
    """
    data = summary.dropna(subset=["skew", "share_above_fence"])
    is_eng = data["feature"].isin(engineered_columns(data["feature"].tolist()))
    fig = Figure(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    for mask, label, color in ((~is_eng, "raw", COLOR_TRAIN), (is_eng, "engineered", COLOR_VAL)):
        ax.scatter(
            data.loc[mask, "skew"],
            data.loc[mask, "share_above_fence"] * 100,
            s=22,
            color=color,
            alpha=0.75,
            label=f"{label} ({int(mask.sum())})",
        )
    ax.set_xlabel("skewness", color=INK_SECONDARY)
    ax.set_ylabel(f"rows above Q3 + {FENCE_FACTOR:g} x IQR (%)", color=INK_SECONDARY)
    ax.set_title(
        "Q1 training cut-offs: skewness vs. outlier share per feature",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path


def plot_repair_rate(repair: pd.DataFrame, base_rate: float, path: Path) -> Path:
    """Horizontal bars: repair rate in normal vs. outlier rows for the top features."""
    fig = Figure(figsize=(8, max(3.0, 0.45 * len(repair) + 1.5)), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    y = np.arange(len(repair))
    h = 0.38
    ax.barh(y - h / 2, repair["rate_normal"] * 100, h, color=COLOR_TRAIN, label="normal rows")
    ax.barh(y + h / 2, repair["rate_outlier"] * 100, h, color=COLOR_VAL, label="outlier rows")
    ax.axvline(base_rate * 100, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"overall {base_rate * 100:.1f} %",
        xy=(base_rate * 100, len(repair) - 0.4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_yticks(y, repair["feature"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("repair rate within the horizon (%)", color=INK_SECONDARY)
    ax.set_title(
        "Repair rate: outlier vs. normal rows (top features by outlier share)",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--top", type=int, default=TOP_N_FEATURES, help="features for the repair rate"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    k, _ = sampling_params_from_config(config)
    frame = load_training_frame(config, k)

    features = select_feature_columns(frame)  # raw + engineered numeric; never labels/ids
    summary = summarize_features(frame, features)
    top = select_top_features(summary, n=args.top)
    repair = repair_rate_by_outlier(frame, top)
    base_rate = float(frame[BINARY_COL].mean())
    logger.info(
        "%d features described, %d with a repair-rate comparison", len(summary), len(repair)
    )

    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports / "outliers_q1_features.csv", index=False, float_format="%.5g")
    repair.to_csv(reports / "outliers_q1_repair_rate.csv", index=False, float_format="%.5f")
    (reports / "outliers_q1_summary.md").write_text(
        render_markdown(summary, repair, base_rate, len(frame), n_cutoffs_label(k)),
        encoding="utf-8",
    )
    plot_skew_vs_share(summary, reports / "figures" / "outliers_q1_skew_vs_share.png")
    plot_repair_rate(repair, base_rate, reports / "figures" / "outliers_q1_repair_rate.png")


if __name__ == "__main__":
    main()
