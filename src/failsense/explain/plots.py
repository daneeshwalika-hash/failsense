"""Figures of the explanation step.

pyplot exception: everywhere else in this repo figures are built with ``matplotlib.figure.Figure``
(no global state). The SHAP plots (beeswarm, bar, waterfall, scatter) exist only as pyplot
functions inside ``shap.plots``, so they are used here under the headless ``Agg`` backend. Every
figure is saved and **always closed** (``plt.close("all")`` in a ``finally``), so no state leaks
into the rest of the process. The permutation and comparison plots are plain ``Figure`` objects.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from failsense.plotting import (  # noqa: E402
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

logger = logging.getLogger(__name__)


@contextmanager
def saved_pyplot_figure(path: Path, title: str | None = None) -> Iterator[None]:
    """Run SHAP plotting code, save the current figure to ``path`` and close everything."""
    try:
        yield
        figure = plt.gcf()
        if title:
            figure.suptitle(title, fontsize=10, color=INK_PRIMARY)
        figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=130)
    finally:
        plt.close("all")


def shap_beeswarm(explanation: shap.Explanation, path: Path, max_display: int) -> None:
    with saved_pyplot_figure(path, "SHAP (raw log-odds of the uncalibrated model), validation"):
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)


def shap_bar(explanation: shap.Explanation, path: Path, max_display: int) -> None:
    with saved_pyplot_figure(path, "Mean |SHAP| (raw log-odds), validation"):
        shap.plots.bar(explanation, max_display=max_display, show=False)


def shap_waterfall(explanation: shap.Explanation, path: Path, max_display: int, title: str) -> None:
    with saved_pyplot_figure(path, title):
        shap.plots.waterfall(explanation, max_display=max_display, show=False)


def shap_dependence(explanation: shap.Explanation, feature: str, path: Path) -> None:
    """SHAP value of one feature against its value, coloured by the strongest interaction."""
    with saved_pyplot_figure(path):
        shap.plots.scatter(explanation[:, feature], color=explanation, show=False)


def _style(ax) -> None:  # noqa: ANN001 - matplotlib Axes
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)


def plot_permutation(table: pd.DataFrame, path: Path, title: str, top: int | None = None) -> None:
    """Horizontal bars: PR-AUC drop with the repeat spread; unreliable drops in grey."""
    shown = table.sort_values("mean_drop", ascending=False)
    if top is not None:
        shown = shown.head(top)
    shown = shown.iloc[::-1]  # largest on top
    fig = Figure(figsize=(8, max(3.0, 0.32 * len(shown) + 1.2)), facecolor=SURFACE)
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    colors = [COLOR_TRAIN if r else INK_MUTED for r in shown["reliable"]]
    ax.barh(shown["group"], shown["mean_drop"], xerr=shown["std_drop"], color=colors,
            ecolor=INK_PRIMARY, error_kw={"elinewidth": 1, "capsize": 2})  # fmt: skip
    ax.axvline(0, color=INK_PRIMARY, linewidth=0.8)
    ax.set_xlabel("drop of PR-AUC when shuffled (mean, bar = std of the repeats)",
                  color=INK_SECONDARY)  # fmt: skip
    ax.set_title(f"{title}\nblue = mean - 2 std > 0 (reliable), grey = inside the spread",
                 color=INK_PRIMARY, fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, dpi=130)


def plot_group_comparison(table: pd.DataFrame, path: Path) -> None:
    """Shares of grouped SHAP and grouped gain next to the permutation drop (reliable only)."""
    shown = table.sort_values("shap_share", ascending=False).iloc[::-1]
    fig = Figure(figsize=(9, max(4.0, 0.34 * len(shown) + 1.5)), facecolor=SURFACE)
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    y = np.arange(len(shown))
    height = 0.27
    perm = shown["mean_drop"].clip(lower=0).where(shown["reliable"], 0.0)
    perm_share = perm / perm.sum() if perm.sum() > 0 else perm
    ax.barh(y + height, shown["shap_share"], height, color=COLOR_TRAIN, label="grouped mean |SHAP|")
    ax.barh(y, perm_share, height, color=COLOR_VAL, label="grouped permutation (reliable only)")
    ax.barh(y - height, shown["gain_share"], height, color=INK_MUTED, label="gain (biased)")
    ax.set_yticks(y, shown["group"])
    ax.set_xlabel("share of the total (each method sums to 1)", color=INK_SECONDARY)
    ax.set_title("Which variables matter: three views", color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, dpi=130)
