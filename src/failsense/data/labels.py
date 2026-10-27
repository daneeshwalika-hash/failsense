"""Build proximity-to-repair labels for SCANIA Component X.

The training split only ships time-to-event data (``train_tte.csv``): the
length of observation per vehicle and whether a repair happened during the
study. This module turns that into one label per readout.

For a readout at ``time_step = t`` of a vehicle observed until ``L``:

* ``d = L - t`` is the time left until the repair (repaired vehicles) or
  until the end of observation (censored vehicles).
* Proximity class (IDA 2024 challenge), with the **upper bound inclusive**:
  ``0`` if ``d > 48``, ``1`` if ``24 < d <= 48``, ``2`` if ``12 < d <= 24``,
  ``3`` if ``6 < d <= 12`` and ``4`` if ``0 <= d <= 6``.
* Censored vehicles (``in_study_repair == 0``): the repair, if any, happens
  after ``L``, so ``d`` is only a lower bound. Readouts with ``d > 48`` are
  safely class 0; readouts with ``d <= 48`` are **unknown** (``<NA>``) and
  must be excluded from training, never labelled 0.

The result is a separate label table keyed by ``(vehicle_id, time_step)`` so
label columns never end up next to the features by accident.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VEHICLE_COL = "vehicle_id"
TIME_COL = "time_step"
STUDY_LENGTH_COL = "length_of_study_time_step"
REPAIR_COL = "in_study_repair"

DISTANCE_COL = "time_to_study_end"
CLASS_COL = "proximity_class"
BINARY_COL = "repair_within_horizon"

# Columns that carry target information and must never be used as features.
LABEL_COLUMNS: frozenset[str] = frozenset(
    {STUDY_LENGTH_COL, REPAIR_COL, DISTANCE_COL, CLASS_COL, BINARY_COL}
)


def validate_class_bounds(bounds: Sequence[float]) -> tuple[float, ...]:
    """Return the class bounds as floats after checking them.

    Args:
        bounds: Upper bounds of classes 1..n, e.g. ``(48, 24, 12, 6)``.

    Raises:
        ValueError: If the bounds are empty, not positive or not strictly
            decreasing.
    """
    values = tuple(float(b) for b in bounds)
    if not values:
        raise ValueError("class_bounds must not be empty")
    if any(b <= 0 for b in values):
        raise ValueError(f"class_bounds must be positive, got {values}")
    if any(a <= b for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"class_bounds must be strictly decreasing, got {values}")
    return values


def class_bounds_from_config(config: Mapping[str, Any]) -> tuple[float, ...]:
    """Read and validate ``labels.class_bounds`` from the project config."""
    try:
        raw = config["labels"]["class_bounds"]
    except KeyError as exc:
        raise KeyError("config is missing 'labels.class_bounds'") from exc
    return validate_class_bounds(raw)


def assign_class(distance: pd.Series, bounds: Sequence[float]) -> pd.Series:
    """Map the time left ``d`` to a proximity class (upper bound inclusive).

    The class is the number of bounds with ``d <= bound``: with bounds
    ``(48, 24, 12, 6)``, ``d = 48`` gives 1, ``d = 24`` gives 2 and
    ``d = 50`` gives 0.

    Raises:
        TypeError: If ``distance`` is not numeric.
        ValueError: If ``distance`` has missing or negative values.
    """
    checked = validate_class_bounds(bounds)
    if not pd.api.types.is_numeric_dtype(distance):
        raise TypeError(f"distance must be numeric, got dtype {distance.dtype}")
    if distance.isna().any():
        raise ValueError("distance must not contain missing values")
    if (distance < 0).any():
        raise ValueError("distance must not be negative")

    hits = distance.to_numpy(dtype=float)[:, np.newaxis] <= np.asarray(checked)
    return pd.Series(hits.sum(axis=1), index=distance.index, dtype="Int8", name=CLASS_COL)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_numeric(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    wrong = {c: str(df[c].dtype) for c in columns if not pd.api.types.is_numeric_dtype(df[c])}
    if wrong:
        raise TypeError(f"{name} has non-numeric columns: {wrong}")


def build_labels(
    readouts: pd.DataFrame,
    tte: pd.DataFrame,
    class_bounds: Sequence[float],
) -> pd.DataFrame:
    """Build one label row per readout from time-to-event data.

    Args:
        readouts: Operational readouts with at least ``vehicle_id`` and
            ``time_step``. Other columns are ignored.
        tte: One row per vehicle with ``vehicle_id``,
            ``length_of_study_time_step`` and ``in_study_repair`` (0/1).
        class_bounds: Upper bounds of classes 1..n, e.g. ``(48, 24, 12, 6)``.
            ``class_bounds[0]`` is the horizon of the binary label.

    Returns:
        DataFrame with the same index and row order as ``readouts`` and the
        columns ``vehicle_id``, ``time_step``, ``time_to_study_end``,
        ``proximity_class`` (Int8, ``<NA>`` = unknown) and
        ``repair_within_horizon`` (Int8, ``<NA>`` = unknown).

    Raises:
        ValueError: On missing columns, duplicate or incomplete tte rows,
            invalid repair flags, vehicles without a tte row, or readouts
            after the end of observation.
        TypeError: On non-numeric time columns.
    """
    bounds = validate_class_bounds(class_bounds)
    _require_columns(readouts, (VEHICLE_COL, TIME_COL), "readouts")
    _require_columns(tte, (VEHICLE_COL, STUDY_LENGTH_COL, REPAIR_COL), "tte")
    _require_numeric(readouts, (TIME_COL,), "readouts")
    _require_numeric(tte, (STUDY_LENGTH_COL, REPAIR_COL), "tte")

    if readouts[TIME_COL].isna().any():
        raise ValueError("readouts.time_step must not contain missing values")
    if tte[VEHICLE_COL].duplicated().any():
        raise ValueError("tte must have exactly one row per vehicle_id")
    if tte[[STUDY_LENGTH_COL, REPAIR_COL]].isna().any().any():
        raise ValueError("tte must not contain missing values")
    invalid_flags = set(tte[REPAIR_COL].unique()) - {0, 1}
    if invalid_flags:
        raise ValueError(f"in_study_repair must be 0 or 1, got {sorted(invalid_flags)}")

    merged = readouts[[VEHICLE_COL, TIME_COL]].merge(
        tte[[VEHICLE_COL, STUDY_LENGTH_COL, REPAIR_COL]],
        on=VEHICLE_COL,
        how="left",
        validate="many_to_one",
    )
    # A left merge keeps the row order of `readouts`, so the index can be restored.
    merged.index = readouts.index

    missing = merged.loc[merged[STUDY_LENGTH_COL].isna(), VEHICLE_COL].unique()
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} vehicle(s) in readouts have no tte row, e.g. {list(missing[:5])}"
        )

    distance = (merged[STUDY_LENGTH_COL] - merged[TIME_COL]).rename(DISTANCE_COL)
    negative = int((distance < 0).sum())
    if negative:
        raise ValueError(f"{negative} readout(s) lie after the end of observation (d < 0)")

    proximity = assign_class(distance, bounds)
    censored_near_end = (merged[REPAIR_COL] == 0) & (distance <= bounds[0])
    proximity = proximity.mask(censored_near_end)
    binary = (proximity > 0).astype("Int8").rename(BINARY_COL)

    labels = pd.DataFrame(
        {
            VEHICLE_COL: merged[VEHICLE_COL],
            TIME_COL: merged[TIME_COL],
            DISTANCE_COL: distance,
            CLASS_COL: proximity,
            BINARY_COL: binary,
        },
        index=readouts.index,
    )

    logger.info(
        "Built %d labels: %d positive (d <= %.0f), %d unknown (censored near end)",
        len(labels),
        int((binary == 1).sum()),
        bounds[0],
        int(proximity.isna().sum()),
    )
    return labels
