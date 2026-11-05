"""The validation-like evaluation set and the training samplings that go with it.

What the source says
--------------------
Kharazian et al. (2025), *Scientific Data* 12, 493, and the IDA 2024 challenge document
(both shipped with the dataset): vehicles are split at random 70 / 15 / 15 into
train / validation / test; in validation and test "the last readout ... is selected
randomly among all readouts for each vehicle", labelled by the time window before the
failure (class 0 for censored vehicles); validation has 136 of 5,046 vehicles in classes
1-4 (2.70 %), test 142 of 5,045 (2.81 %). Per vehicle that is one uniformly random readout.

What we measured on the training trucks (``docs/problem_framing.md`` section 6, v1.1)
-------------------------------------------------------------------------------------
* Imitating that procedure literally (``literal``) reproduces the **positive rate**
  (2.72 %) but not the ``time_step`` distribution (KS 0.31, median 108 vs. 169).
* The validation trucks keep 82 % of a mean training history (38.9 of 47.7 readouts per
  truck), a uniform cut of training-length histories would keep 51 %: the official
  validation and test histories are simply longer. The time_step shift is a property of
  the data, not of the cutting procedure.
* In the training data ``time_step`` and the distance to the end of the study are tied
  together per truck. Matching only the ``time_step`` marginal (``rank``, below) puts every
  repaired truck at its own end: KS 0.012 but a positive rate of 9.6 %, 3.6 x the official
  one. A model scored on that set would face an easier problem than the official one.

The final protocol: calibrated rank matching (``calibrated``)
-------------------------------------------------------------
Imitate **both** properties: the ``time_step`` distribution of the official validation
last readouts and their positive rate. Only the ``time_step`` of the validation
**readouts** and **one aggregate number of the validation labels (136 / 5,046 = 2.70 %,
no per-row labels, public in the dataset paper)** are used; the test set is untouched.

1. *Target:* the sorted ``time_step`` of the last readout of every official validation truck.
2. *Reach:* for every training truck the largest ``time_step`` of the readouts it may be
   evaluated on. Censored near-end readouts count as class 0 (as in the official labels).
3. *Calibration:* a random share of the repaired trucks is "end-near" (may be evaluated at
   its end, i.e. as a positive); every other repaired truck may only be evaluated on
   readouts more than 48 steps before its repair (a negative, like the official
   repaired-but-not-yet-failing trucks). The share is searched by bisection so that the
   positive rate of the finished evaluation set equals 2.70 %. The trucks are ordered by a
   fixed seeded random number, so the sets are nested and the search is monotone.
4. *Rank matching:* sort the n trucks by reach, give them the n quantiles of the target in
   that order, and let each truck take the allowed readout whose ``time_step`` is closest to
   its target (a truck that cannot reach it takes its latest allowed readout).

For *training* samples (``matched``) the same rank matching runs once per round with a
seeded multiplicative noise on the reach, and the rounds are united.

Fixed and seeded: the evaluation rows depend only on the data, the validation
``time_step`` values, the aggregate rate and the seed, never on a model or on the fold
assignment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from failsense.analysis.time_step_check import (
    ks_distance,
    last_readout_time_step,
    quantile_table,
)
from failsense.data.labels import (
    BINARY_COL,
    CLASS_COL,
    REPAIR_COL,
    TIME_COL,
    VEHICLE_COL,
)
from failsense.data.sampling import sample_cutoffs
from failsense.data.split import Fold
from failsense.models.baseline_cv import read_table, table_path
from failsense.plotting import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

logger = logging.getLogger(__name__)

EVAL_METHODS = ("calibrated", "literal", "rank")
MATCH_NOISE = 0.25  # sigma of the log-normal noise on the reach in training rounds
MIN_TIME_STEP = 70  # "ts70": the official validation q05 is 78
# Positive rate of the official validation split: 136 of 5,046 trucks are in classes 1-4
# (EDA, notebook section 14 / docs section 3). A reference for the sanity check only; it is
# neither computed from validation labels here nor used to build any sample.
OFFICIAL_VALIDATION_POSITIVES = (136, 5046)


def validation_time_steps(config: Mapping[str, Any]) -> pd.Series:
    """``time_step`` of the last readout of every official validation truck.

    Reads only ``vehicle_id`` and ``time_step`` of the validation **readouts**; the
    validation labels file is never touched.
    """
    readouts = read_table(
        table_path(config, "validation", "readouts"), columns=[VEHICLE_COL, TIME_COL]
    )
    return last_readout_time_step(readouts)


def known_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known label (censored readouts near the end of study are dropped)."""
    missing = [c for c in (VEHICLE_COL, TIME_COL, CLASS_COL) if c not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required columns: {missing}")
    return labels[labels[CLASS_COL].notna()]


def match_cutoffs(
    known: pd.DataFrame, targets: Sequence[float], seed: int, noise: float = 0.0
) -> pd.Index:
    """One readout per truck by rank matching (see the module docstring).

    Args:
        known: Known-label rows with ``vehicle_id`` and ``time_step``.
        targets: ``time_step`` values to imitate (e.g. last validation readouts).
        seed: Seed for the tie-break between trucks with equal reach (and for ``noise``).
        noise: Sigma of a log-normal factor on the reach; 0 = deterministic matching
            (the evaluation set), > 0 = a different pairing per seed (training rounds).

    Returns:
        The index labels of the chosen rows, in the original row order.

    Raises:
        ValueError: If ``known`` or ``targets`` is empty, ``targets`` has NaN, or
            ``noise < 0``.
    """
    if known.empty:
        raise ValueError("no known-label rows to choose from")
    if not known.index.is_unique:
        # the index labels identify the chosen readouts; duplicates would pick several rows
        raise ValueError("known must have a unique index (one label per readout)")
    target_values = np.asarray(targets, dtype=float)
    if target_values.size == 0 or np.isnan(target_values).any():
        raise ValueError("targets must be a non-empty array without NaN")
    if noise < 0:
        raise ValueError(f"noise must be >= 0, got {noise}")

    reach = known.groupby(VEHICLE_COL)[TIME_COL].max()
    n = len(reach)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(n)
    key = reach.to_numpy(dtype=float)
    if noise > 0:
        key = key * np.exp(noise * rng.standard_normal(n))
    order = np.lexsort((tiebreak, key))  # by key, ties by the seeded random number
    levels = (np.arange(n) + 0.5) / n
    assigned = np.empty(n)
    assigned[order] = np.quantile(target_values, levels)
    target_of = pd.Series(assigned, index=reach.index)

    work = known[[VEHICLE_COL, TIME_COL]].copy()
    work["_distance"] = (work[TIME_COL] - work[VEHICLE_COL].map(target_of)).abs()
    chosen = (
        work.sort_values([VEHICLE_COL, "_distance", TIME_COL], kind="stable")
        .groupby(VEHICLE_COL, sort=False)
        .head(1)
    )
    return known.index[known.index.isin(chosen.index)]


def build_eval_cutoffs(labels: pd.DataFrame, targets: Sequence[float], seed: int) -> pd.DataFrame:
    """The fixed validation-like evaluation set: one known-label row per truck."""
    known = known_labels(labels)
    return labels.loc[match_cutoffs(known, targets, seed)]


def matched_training_cutoffs(
    labels: pd.DataFrame,
    targets: Sequence[float],
    k: int,
    seed: int,
    noise: float = MATCH_NOISE,
) -> pd.DataFrame:
    """Up to ``k`` cut-offs per truck whose ``time_step`` follows the target distribution.

    ``k`` rounds of rank matching with different seeds and a noisy reach; the chosen
    rows are united (a truck whose rounds pick the same row keeps it once).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    known = known_labels(labels)
    chosen: set[Any] = set()
    for round_no in range(k):
        chosen.update(match_cutoffs(known, targets, seed + round_no, noise).tolist())
    return labels.loc[labels.index.isin(chosen)]


def cutoffs_from_time_step(
    labels: pd.DataFrame, minimum: float, k: int | None, seed: int
) -> pd.DataFrame:
    """``k`` random cut-offs per truck, drawn only from readouts with ``time_step >= minimum``.

    Trucks without such a readout have no training rows.
    """
    known = known_labels(labels)
    return sample_cutoffs(known[known[TIME_COL] >= minimum], n_per_vehicle=k, seed=seed)


def with_censored_tail(labels: pd.DataFrame) -> pd.DataFrame:
    """Label table for *evaluation*: censored near-end readouts become class 0.

    The official validation / test labels treat the readout of a censored vehicle as
    class 0 wherever it was drawn. For *training* these readouts stay unknown (the truck
    may fail right after the study ended); for scoring we imitate the official labels.
    """
    out = labels.copy()
    out[CLASS_COL] = out[CLASS_COL].astype("Int8").fillna(0)
    out[BINARY_COL] = (out[CLASS_COL] > 0).astype("Int8")
    return out


def build_literal_eval_cutoffs(
    labels: pd.DataFrame, seed: int, include_censored_tail: bool = True
) -> pd.DataFrame:
    """The procedure as the dataset paper states it: one random readout per truck.

    Uniform over all readouts of each truck (``include_censored_tail``: including the
    censored near-end ones, labelled class 0). Reproduces the official positive rate but
    not the ``time_step`` distribution (the official histories are longer).
    """
    pool = with_censored_tail(labels) if include_censored_tail else known_labels(labels)
    return sample_cutoffs(pool, n_per_vehicle=1, seed=seed)


def build_calibrated_eval_cutoffs(
    labels: pd.DataFrame,
    targets: Sequence[float],
    repaired: pd.Series,
    target_rate: float,
    seed: int,
    include_censored_tail: bool = True,
    iterations: int = 24,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The final evaluation set: rank matching with a calibrated share of end-near trucks.

    Args:
        labels: ``build_labels`` output of the training readouts.
        targets: Official validation last-readout ``time_step`` values.
        repaired: ``vehicle_id`` -> ``in_study_repair`` (0 / 1).
        target_rate: Positive rate to reach (one aggregate number: 136 / 5,046).
        seed: Seed of the random order of the repaired trucks and of the tie-breaks.
        include_censored_tail: Count censored near-end readouts as class 0.
        iterations: Bisection steps.

    Returns:
        ``(rows, info)``: one row per truck, and ``info`` with the ``end_near_share`` that
        was found, the achieved ``rate`` and the number of ``early_cut_trucks``.

    Raises:
        ValueError: If ``target_rate`` is not in (0, 1).
    """
    if not 0 < target_rate < 1:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    pool = with_censored_tail(labels) if include_censored_tail else known_labels(labels).copy()
    pool[BINARY_COL] = pool[BINARY_COL].astype(int)

    flags = repaired.reindex(pool[VEHICLE_COL].unique()).fillna(0).astype(int)
    repaired_ids = np.sort(flags.index[flags == 1].to_numpy())
    # one fixed random number per repaired truck: a larger share only adds trucks (nested)
    u = pd.Series(np.random.default_rng(seed).random(len(repaired_ids)), index=repaired_ids)
    has_negative = set(pool.loc[pool[BINARY_COL] == 0, VEHICLE_COL])
    positive_rows = pool[BINARY_COL] == 1

    def select(share: float) -> tuple[pd.DataFrame, int]:
        early = {v for v in repaired_ids if u[v] >= share and v in has_negative}
        keep = ~(pool[VEHICLE_COL].isin(early) & positive_rows)
        chosen = match_cutoffs(pool[keep], targets, seed)
        return pool.loc[chosen], len(early)

    lo, hi = 0.0, 1.0
    best_rows, best_early = select(1.0)
    best_share = 1.0
    if best_rows[BINARY_COL].mean() > target_rate:  # even all-end-near is too high: bisect
        best_gap = abs(best_rows[BINARY_COL].mean() - target_rate)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            rows, n_early = select(mid)
            rate = rows[BINARY_COL].mean()
            if abs(rate - target_rate) < best_gap:
                best_rows, best_early, best_share, best_gap = (
                    rows,
                    n_early,
                    mid,
                    abs(rate - target_rate),
                )
            if rate > target_rate:
                hi = mid
            else:
                lo = mid
    else:
        logger.warning(
            "target rate %.4f is above the maximum reachable %.4f: all repaired trucks end-near",
            target_rate,
            best_rows[BINARY_COL].mean(),
        )
    info = {
        "end_near_share": float(best_share),
        "rate": float(best_rows[BINARY_COL].mean()),
        "early_cut_trucks": int(best_early),
    }
    logger.info("Calibrated evaluation set: %s", info)
    return best_rows, info


def build_eval_cutoffs_by_method(
    method: str,
    labels: pd.DataFrame,
    targets: Sequence[float],
    repaired: pd.Series,
    seed: int,
    target_rate: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dispatch to ``calibrated`` (default protocol), ``literal`` or ``rank``.

    Raises:
        ValueError: For an unknown method.
    """
    if method == "calibrated":
        rate = (
            target_rate
            if target_rate is not None
            else OFFICIAL_VALIDATION_POSITIVES[0] / OFFICIAL_VALIDATION_POSITIVES[1]
        )
        rows, info = build_calibrated_eval_cutoffs(labels, targets, repaired, rate, seed)
    elif method == "literal":
        rows, info = build_literal_eval_cutoffs(labels, seed), {}
    elif method == "rank":
        rows, info = build_eval_cutoffs(labels, targets, seed), {}
    else:
        raise ValueError(f"unknown eval method {method!r}, choose one of {list(EVAL_METHODS)}")
    rows = restrict_to_known_trucks(rows, labels)
    if info:
        info = {**info, "rate": float(rows[BINARY_COL].mean())}  # rate of the final rows
    return rows, info


def restrict_to_known_trucks(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Keep only trucks with at least one known-label readout.

    A truck whose readouts are all censored near-end ones (class 0 only for scoring) has
    no training row and is in no fold; it is dropped from the evaluation set so that the
    training and evaluation trucks are the same set (one truck of 23,550 in SCANIA).
    """
    trucks = set(known_labels(labels)[VEHICLE_COL])
    kept = rows[rows[VEHICLE_COL].isin(trucks)]
    if len(kept) < len(rows):
        logger.info(
            "Dropped %d truck(s) without any known-label readout from the evaluation set",
            len(rows) - len(kept),
        )
    return kept


def vehicle_fold_from_folds(frame: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """Map every truck to the one fold in which it is validated.

    Args:
        frame: The frame the folds were built on (``vehicle_id`` per row).
        folds: ``(train_idx, val_idx)`` per fold, e.g. from ``make_group_folds``.

    Raises:
        ValueError: If a truck is validated in more than one fold or in none.
    """
    vehicle = frame[VEHICLE_COL].to_numpy()
    fold_of: dict[Any, int] = {}
    for fold_no, (_, val_idx) in enumerate(folds):
        for v in np.unique(vehicle[val_idx]):
            if v in fold_of:
                raise ValueError(f"vehicle {v} is validated in more than one fold")
            fold_of[v] = fold_no
    missing = set(np.unique(vehicle)) - set(fold_of)
    if missing:
        raise ValueError(f"{len(missing)} vehicle(s) are in no validation part")
    return pd.Series(fold_of, name="fold")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion (95 % for ``z = 1.96``)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def positive_rate_report(
    eval_rows: pd.DataFrame, train_rows: pd.DataFrame, tte: pd.DataFrame
) -> dict[str, Any]:
    """Is the evaluation set realistic? Positive rate overall and by ``time_step`` quartile.

    Late evaluation rows can only come from trucks that really ran long, and
    censored trucks lose their last 48 steps, so late rows may come mostly from
    repaired trucks and inflate the positive rate. This reports it.

    Args:
        eval_rows: Evaluation cut-offs (one per truck) with ``vehicle_id``,
            ``time_step`` and the binary label.
        train_rows: Training cut-offs (e.g. ``k = 5``) for the repaired share.
        tte: ``vehicle_id`` and ``in_study_repair`` of the training trucks.

    Returns:
        ``overall`` (rate, official reference, Wilson interval, ``far`` flag),
        ``by_quartile`` (DataFrame) and ``repaired_share`` (eval vs. training vs. all).
    """
    repaired = tte.set_index(VEHICLE_COL)[REPAIR_COL]
    ev = eval_rows.assign(repaired=eval_rows[VEHICLE_COL].map(repaired).to_numpy())
    tr = train_rows.assign(repaired=train_rows[VEHICLE_COL].map(repaired).to_numpy())

    positives, n = int(ev[BINARY_COL].sum()), len(ev)
    off_pos, off_n = OFFICIAL_VALIDATION_POSITIVES
    lo, hi = wilson_interval(off_pos, off_n)
    rate = positives / n
    overall = {
        "n": n,
        "positives": positives,
        "rate": rate,
        "official_rate": off_pos / off_n,
        "official_interval": (lo, hi),
        "ratio_to_official": rate / (off_pos / off_n),
        "far_from_official": not (lo <= rate <= hi),
    }

    # Quartiles by rank (ties broken by row order): equal sizes even when many rows share
    # one time_step, where value-based quartile edges would collapse.
    quartile = pd.qcut(
        ev[TIME_COL].rank(method="first"),
        4,
        labels=["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"],
    )
    grouped = ev.groupby(quartile, observed=True)
    by_quartile = pd.DataFrame(
        {
            "time_step_min": grouped[TIME_COL].min(),
            "time_step_max": grouped[TIME_COL].max(),
            "n": grouped.size(),
            "positives": grouped[BINARY_COL].sum(),
            "positive_rate": grouped[BINARY_COL].mean(),
            "repaired_truck_share": grouped["repaired"].mean(),
        }
    )
    train_trucks = tr.drop_duplicates(VEHICLE_COL)
    repaired_share = {
        "eval_set_trucks": float(ev["repaired"].mean()),
        "training_trucks": float(train_trucks["repaired"].mean()),
        "training_rows": float(tr["repaired"].mean()),
        "all_trucks_with_tte": float(repaired.mean()),
        "positive_rate_in_repaired_trucks_eval": float(
            ev.loc[ev["repaired"] == 1, BINARY_COL].mean()
        ),
    }
    return {"overall": overall, "by_quartile": by_quartile, "repaired_share": repaired_share}


def eval_set_report(
    samples: Mapping[str, Sequence[float] | pd.Series], reference: Sequence[float] | pd.Series
) -> pd.DataFrame:
    """Quantiles of every sample plus its KS statistic against ``reference``."""
    table = quantile_table({**samples, "official validation (last readout)": reference})
    table["ks_vs_validation"] = [
        ks_distance(values, reference)[0] if name != "official validation (last readout)" else 0.0
        for name, values in {**samples, "official validation (last readout)": reference}.items()
    ]
    return table


def plot_eval_set(
    samples: Mapping[str, pd.Series], reference: pd.Series, path: Path
) -> Path:  # fmt: skip
    """Histogram and ECDF: validation reference vs. the evaluation set (and other samples)."""
    fig = Figure(figsize=(10, 4.2), dpi=150, facecolor=SURFACE)
    ax_hist, ax_cdf = fig.subplots(1, 2)
    upper = float(max(reference.max(), *(s.max() for s in samples.values())))
    bins = np.linspace(0, upper, 41)
    colors = [COLOR_TRAIN, INK_MUTED, INK_SECONDARY]
    curves = [("official validation (last readout)", reference, COLOR_VAL)]
    curves += [(name, s, colors[i % len(colors)]) for i, (name, s) in enumerate(samples.items())]
    for label, values, color in curves:
        ax_hist.hist(values, bins=bins, density=True, histtype="step", linewidth=1.6,
                     color=color, label=label)  # fmt: skip
        sorted_values = np.sort(values.to_numpy(dtype=float))
        ax_cdf.step(sorted_values, np.arange(1, len(sorted_values) + 1) / len(sorted_values),
                    color=color, linewidth=1.6)  # fmt: skip
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
    ax_hist.legend(frameon=False, fontsize=7, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path
