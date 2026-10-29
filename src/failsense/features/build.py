"""Build the model input table for SCANIA Component X.

Today (29.10): attach the eight truck specifications to every row.
Tomorrow (30.10): counter rates and histogram shares are added here too.

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

import pandas as pd

from failsense.data.labels import VEHICLE_COL
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
