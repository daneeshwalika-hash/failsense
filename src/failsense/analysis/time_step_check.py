"""Do the training cut-offs imitate the validation situation? (time_step check)

Run it with::

    uv run python -m failsense.analysis.time_step_check

The question
------------
In validation and test each truck is seen **once**, at the end of its
(truncated) history, and the label says how far the repair is away from *that*
moment. Training draws ``k`` random readouts per truck instead (see
``failsense.data.sampling``). If the two moments are distributed differently
along ``time_step`` - e.g. validation always sits late in a truck's life while
random cut-offs sit anywhere - the model trains on one situation and is tested
on another (a selection effect, see ``docs/problem_framing.md`` §8).

The check compares ``time_step`` of

* the training cut-offs (``k`` from the config, labelled readouts only), with
* the **last readout per truck** in the validation split.

It uses the validation **readouts only, no labels**: this is a description of
the input, not a look at validation results. Output: quantiles, the
Kolmogorov-Smirnov statistic (largest vertical gap between the two empirical
distribution functions: 0 = identical, 1 = completely separate) and one overlay
figure in ``reports/``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    CLASS_COL,
    TIME_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.sampling import sample_cutoffs, sampling_params_from_config
from failsense.models.baseline_cv import n_cutoffs_label, read_table, table_path
from failsense.plotting import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def last_readout_time_step(
    readouts: pd.DataFrame, group_col: str = VEHICLE_COL, time_col: str = TIME_COL
) -> pd.Series:
    """``time_step`` of the last readout of every truck (one value per truck).

    Raises:
        ValueError: If a column is missing.
    """
    missing = [c for c in (group_col, time_col) if c not in readouts.columns]
    if missing:
        raise ValueError(f"readouts is missing columns: {missing}")
    return readouts.groupby(group_col)[time_col].max()


def quantile_table(samples: Mapping[str, Sequence[float] | pd.Series]) -> pd.DataFrame:
    """Row per sample: ``n``, ``mean`` and the quantiles of ``QUANTILES``.

    Raises:
        ValueError: If a sample is empty or contains NaN.
    """
    rows = []
    for name, values in samples.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"sample {name!r} is empty")
        if np.isnan(arr).any():
            raise ValueError(f"sample {name!r} contains NaN")
        row = {"sample": name, "n": arr.size, "mean": float(arr.mean())}
        row.update({f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in QUANTILES})
        rows.append(row)
    return pd.DataFrame(rows)


def ks_distance(
    a: Sequence[float] | pd.Series, b: Sequence[float] | pd.Series
) -> tuple[float, float]:
    """Two-sample KS statistic and p-value (``scipy.stats.ks_2samp``).

    The statistic is the size of the difference and the number to read. The
    p-value is not: with thousands of rows even a tiny, irrelevant gap is
    "significant".

    Raises:
        ValueError: If a sample is empty or contains NaN.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("both samples need at least one value")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("samples must not contain NaN")
    result = stats.ks_2samp(x, y)
    return float(result.statistic), float(result.pvalue)


def plot_overlay(train: pd.Series, validation: pd.Series, k_label: str, path: Path) -> Path:
    """Overlay histogram (density) and empirical CDF of the two samples."""
    fig = Figure(figsize=(10, 4.2), dpi=150, facecolor=SURFACE)
    ax_hist, ax_cdf = fig.subplots(1, 2)
    upper = float(max(train.max(), validation.max()))
    bins = np.linspace(0, upper, 41)
    samples = (
        (train, f"train cut-offs (k = {k_label})", COLOR_TRAIN),
        (validation, "validation: last readout per truck", COLOR_VAL),
    )
    for values, label, color in samples:
        ax_hist.hist(
            values, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=color
        )
        ax_hist.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
        sorted_values = np.sort(values.to_numpy(dtype=float))
        ax_cdf.step(
            sorted_values,
            np.arange(1, len(sorted_values) + 1) / len(sorted_values),
            color=color,
            linewidth=1.6,
        )
    ax_hist.set_ylabel("density", color=INK_SECONDARY)
    ax_cdf.set_ylabel("share of rows with time_step <= x", color=INK_SECONDARY)
    ax_hist.set_title("time_step distribution", color=INK_PRIMARY, fontsize=11, loc="left")
    ax_cdf.set_title("empirical CDF", color=INK_PRIMARY, fontsize=11, loc="left")
    for ax in (ax_hist, ax_cdf):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("time_step", color=INK_SECONDARY)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    ax_hist.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    k, seed = sampling_params_from_config(config)

    # Train: only key + time columns are needed, not the 106 feature columns.
    readouts = read_table(table_path(config, "train", "readouts"), columns=[VEHICLE_COL, TIME_COL])
    tte = read_table(table_path(config, "train", "tte"))
    labels = build_labels(readouts, tte, class_bounds_from_config(config))
    cutoffs = readouts.loc[sample_cutoffs(labels, n_per_vehicle=k, seed=seed).index, TIME_COL]
    all_labelled = readouts.loc[labels.index[labels[CLASS_COL].notna()], TIME_COL]
    # Validation: readouts only. No labels are loaded.
    val_readouts = read_table(
        table_path(config, "validation", "readouts"), columns=[VEHICLE_COL, TIME_COL]
    )
    val_last = last_readout_time_step(val_readouts)

    k_label = n_cutoffs_label(k)
    samples = {
        f"train cut-offs (k={k_label})": cutoffs,
        "train, all labelled readouts": all_labelled,
        "validation, last readout per truck": val_last,
    }
    table = quantile_table(samples)
    stat, pvalue = ks_distance(cutoffs, val_last)
    stat_all, _ = ks_distance(all_labelled, val_last)
    table["ks_vs_validation"] = [stat, stat_all, 0.0]
    logger.info("\n%s", table.to_string(index=False, float_format="%.1f"))
    logger.info("KS statistic cut-offs vs. validation last readout: %.4f (p = %.3g)", stat, pvalue)
    logger.info("KS statistic all labelled readouts vs. validation: %.4f", stat_all)

    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "time_step_check_q1.csv", index=False, float_format="%.4g")
    plot_overlay(cutoffs, val_last, k_label, reports / "figures" / "time_step_check_q1.png")


if __name__ == "__main__":
    main()
