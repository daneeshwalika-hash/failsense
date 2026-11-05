"""Build the model input table for SCANIA Component X.

* 29.10: attach the eight truck specifications to every row.
* 30.10: counter deltas/rates (``add_counter_deltas``) and histogram shares
  (``add_histogram_features``).

Why a join and not a feature per readout
----------------------------------------
The specifications (``Spec_0`` ... ``Spec_7``) describe the truck, not a
readout: there is exactly one row per truck. Every cut-off of a truck gets the
same eight values. In SQL Server this is a classic fact-to-dimension join:

    SELECT r.*, s.Spec_0, ..., s.Spec_7
    FROM cutoffs AS r
    LEFT JOIN specifications AS s ON s.vehicle_id = r.vehicle_id

Two things can go wrong in such a join, and both are checked here:

1. **Row explosion**: if a truck had two spec rows, every cut-off of that truck
   would appear twice, and the model would see duplicated training examples.
   ``validate="many_to_one"`` makes pandas raise instead.
2. **Silent row loss / silent NaN**: an inner join drops trucks without spec
   row; a left join gives them NaN specs. Neither is acceptable without notice,
   so a truck without spec row raises a ``ValueError``.

Leakage note: the specs are static truck properties known from the first day,
so they carry no information from the future.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from failsense.data.labels import TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS

logger = logging.getLogger(__name__)

_MERGE_INDICATOR = "_spec_merge"


def attach_specifications(rows: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Add the eight ``Spec_*`` columns to every row by ``vehicle_id``.

    Args:
        rows: Any table with a ``vehicle_id`` column, e.g. the sampled
            cut-offs (many rows per truck) or the last readout per truck.
        specs: Specification table, one row per truck (validated by
            ``failsense.data.schema.specifications_schema``).

    Returns:
        A new DataFrame with all columns of ``rows`` plus ``Spec_0`` ...
        ``Spec_7``; same index, same row order and same number of rows.

    Raises:
        ValueError: If a required column is missing, if ``rows`` already
            contains spec columns, or if a truck in ``rows`` has no spec row.
        pandas.errors.MergeError: If ``specs`` has more than one row for a
            truck (the join would duplicate rows).
    """
    if VEHICLE_COL not in rows.columns:
        raise ValueError(f"rows is missing required column {VEHICLE_COL!r}")
    missing_spec_cols = [c for c in (VEHICLE_COL, *SPEC_COLUMNS) if c not in specs.columns]
    if missing_spec_cols:
        raise ValueError(f"specs is missing required columns: {missing_spec_cols}")
    already_there = [c for c in SPEC_COLUMNS if c in rows.columns]
    if already_there:
        # A second join would create Spec_0_x / Spec_0_y columns.
        raise ValueError(f"rows already contains spec columns: {already_there}")

    merged = rows.merge(
        specs[[VEHICLE_COL, *SPEC_COLUMNS]],
        on=VEHICLE_COL,
        how="left",  # keeps every row of `rows`, in its original order
        validate="many_to_one",  # each truck at most once in specs
        indicator=_MERGE_INDICATOR,  # "both" or "left_only" per row
    )
    # A left merge keeps the row order of `rows`, so the index can be restored.
    merged.index = rows.index

    no_spec = merged[_MERGE_INDICATOR] == "left_only"
    if no_spec.any():
        vehicles = sorted(int(v) for v in merged.loc[no_spec, VEHICLE_COL].unique())
        raise ValueError(
            f"{len(vehicles)} vehicle(s) have no specification row, e.g. {vehicles[:5]}"
        )

    result = merged.drop(columns=_MERGE_INDICATOR)
    logger.info(
        "Attached %d spec columns to %d rows (%d vehicles)",
        len(SPEC_COLUMNS),
        len(result),
        result[VEHICLE_COL].nunique(),
    )
    return result


# --- counter deltas ----------------------------------------------------------
# Suffixes of the engineered columns. "__" cannot occur in a raw SCANIA column
# name (``171_0``), so an engineered column never collides with a raw one.
DELTA_SUFFIX = "__delta"
RATE_SUFFIX = "__rate"
SHARE_SUFFIX = "__share"
MEAN_BIN_SUFFIX = "__mean_bin"
TOP_SHARE_SUFFIX = "__top_share"


ENGINEERED_SUFFIXES = (DELTA_SUFFIX, RATE_SUFFIX, SHARE_SUFFIX, MEAN_BIN_SUFFIX, TOP_SHARE_SUFFIX)


def engineered_columns(columns: Sequence[str]) -> list[str]:
    """Return those of ``columns`` that were created by the functions of this module."""
    return [c for c in columns if c.endswith(ENGINEERED_SUFFIXES)]


def add_counter_deltas(
    df: pd.DataFrame,
    counter_cols: Sequence[str],
    group_col: str = VEHICLE_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Add ``<col>__delta`` and ``<col>__rate`` for every counter column.

    The counters are cumulative ("total fuel used since day 1"), so the raw
    value mostly says how old the truck is. What changed since the previous
    readout is the more direct signal of the current state:

    * ``delta`` = value - value of the previous readout **of the same truck**
    * ``rate``  = delta / (time_step - previous time_step of the same truck)

    Leakage: ``groupby(...).diff()`` looks only backwards, and it restarts at
    every truck, so a delta never mixes two trucks and never uses a later
    readout. The first readout of a truck has no predecessor -> NaN (the
    imputer in the Pipeline fills it, fitted per fold).

    In SQL Server terms: ``value - LAG(value) OVER (PARTITION BY vehicle_id
    ORDER BY time_step)``.

    Call this on the **full** readouts, then select the cut-off rows. After
    selection the previous readout would be missing and the delta wrong.

    Args:
        df: Readouts with ``group_col``, ``time_col`` and the counter columns.
            Rows of one truck must be in ascending ``time_step`` order.
        counter_cols: Counter columns, e.g. ``schema.counter_columns()``.
        group_col: Truck identifier.
        time_col: Readout time.

    Returns:
        A new DataFrame: all input columns plus 2 per counter; same index,
        order and row count.

    Raises:
        ValueError: If a column is missing, ``group_col``/``time_col`` contain
            NaN, or ``time_col`` is not strictly increasing within a truck.
    """
    counter_cols = list(counter_cols)
    missing = [c for c in (group_col, time_col, *counter_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    clash = [c + suffix for c in counter_cols for suffix in (DELTA_SUFFIX, RATE_SUFFIX)]
    clash = [c for c in clash if c in df.columns]
    if clash:
        raise ValueError(f"df already contains engineered columns: {clash}")
    if df[[group_col, time_col]].isna().any().any():
        raise ValueError(f"{group_col!r} and {time_col!r} must not contain NaN")

    by_truck = df.groupby(group_col, sort=False)
    # Time between two readouts of the same truck; NaN for the first readout.
    dt = by_truck[time_col].diff()
    if (dt <= 0).any():
        n_bad = int((dt <= 0).sum())
        raise ValueError(
            f"{time_col!r} must be strictly increasing within each {group_col!r} "
            f"({n_bad} readouts are not later than their predecessor)"
        )

    # Build all new columns first and concatenate once (adding 16 columns one by
    # one would fragment the frame).
    new: dict[str, pd.Series] = {}
    for col in counter_cols:
        delta = by_truck[col].diff()
        new[col + DELTA_SUFFIX] = delta
        new[col + RATE_SUFFIX] = delta / dt
    result = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # Counters should only grow. A decrease means a reset or a data problem; we
    # report it and keep the value (no silent clipping).
    for col in counter_cols:
        n_down = int((new[col + DELTA_SUFFIX] < 0).sum())
        if n_down:
            logger.warning("Counter %s decreases in %d readouts (kept as is)", col, n_down)
    logger.info("Added delta/rate for %d counters to %d rows", len(counter_cols), len(df))
    return result


# --- histogram features --------------------------------------------------------
def n_top_bins(n_bins: int, top_fraction: float) -> int:
    """Number of highest bins that form the "top" of a histogram (at least 1)."""
    return max(1, round(top_fraction * n_bins))


def features_params_from_config(config: Mapping[str, Any]) -> float:
    """Read ``features.top_fraction`` (0 < fraction <= 1) from the config."""
    try:
        fraction = config["features"]["top_fraction"]
    except KeyError as exc:
        raise KeyError("config needs 'features.top_fraction'") from exc
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise TypeError(f"features.top_fraction must be a number, got {fraction!r}")
    if not 0 < fraction <= 1:
        raise ValueError(f"features.top_fraction must be in (0, 1], got {fraction}")
    return float(fraction)


def add_histogram_features(
    df: pd.DataFrame,
    histogram_groups: Mapping[str, Sequence[str]],
    top_fraction: float,
) -> pd.DataFrame:
    """Add shares, weighted mean bin and top-bin share for every histogram.

    A histogram column holds how often a value fell into a bin, so the raw
    numbers grow with the number of measurements. Dividing by the row sum
    gives the *distribution* (shares sum to 1) independent of how long the
    truck has been measured. Per histogram this adds:

    * ``<bin>__share`` = bin / row sum (one per bin)
    * ``<hist>__mean_bin`` = sum(bin_index * share), index from 0: where the
      mass sits, one number instead of up to 36
    * ``<hist>__top_share`` = share of the ``n_top_bins`` highest bins: how
      much time is spent in the extreme range

    Rows with sum 0 (no measurement) or with a NaN bin get NaN for all of
    these; the imputer handles them. A NaN bin must not be skipped: the
    remaining bins would then sum to 1 and look like a complete histogram.

    This works on one row at a time (no past information), so it gives the
    same result before or after selecting the cut-off rows.

    Args:
        df: Rows containing all bin columns.
        histogram_groups: ``{histogram name: bin columns in order}``, e.g.
            ``schema.histogram_groups()``.
        top_fraction: Share of bins counted as "top", ``0 < f <= 1``.

    Returns:
        A new DataFrame with the input columns plus the new ones; same index,
        order and row count.

    Raises:
        ValueError: If bin columns are missing, a histogram has no bins,
            ``top_fraction`` is out of range, or new columns already exist.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    all_bins = [c for cols in histogram_groups.values() for c in cols]
    missing = [c for c in all_bins if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing histogram columns: {missing[:5]}")
    if any(len(cols) == 0 for cols in histogram_groups.values()):
        raise ValueError("every histogram needs at least one bin column")

    new: dict[str, np.ndarray] = {}
    for name, cols in histogram_groups.items():
        counts = df[list(cols)].to_numpy(dtype=float)
        total = counts.sum(axis=1)  # NaN if any bin is NaN
        total = np.where(total > 0, total, np.nan)  # sum 0 -> undefined shares
        shares = counts / total[:, None]
        n_top = n_top_bins(len(cols), top_fraction)
        new[name + MEAN_BIN_SUFFIX] = shares @ np.arange(len(cols))
        new[name + TOP_SHARE_SUFFIX] = shares[:, -n_top:].sum(axis=1)
        for j, col in enumerate(cols):
            new[col + SHARE_SUFFIX] = shares[:, j]
    clash = [c for c in new if c in df.columns]
    if clash:
        raise ValueError(f"df already contains engineered columns: {clash[:5]}")
    logger.info("Added %d histogram features to %d rows", len(new), len(df))
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
