"""Turn a truck's readouts into the API's request body (and score a risk trajectory).

The body follows ``failsense.api.schemas``: ``current`` readout, ``previous`` = the readout
IMMEDIATELY before it, the specifications. NaN becomes ``null``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from failsense.api.schemas import previous_keys, readout_keys
from failsense.dashboard.client import ApiClient
from failsense.data.labels import TIME_COL
from failsense.data.schema import SPEC_COLUMNS

logger = logging.getLogger(__name__)


def _number(value: Any) -> float | None:
    """A JSON-safe number: NaN / missing -> ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return float(value)


def _readout(row: pd.Series, keys: set[str]) -> dict[str, float | None]:
    missing = sorted(keys - set(row.index))
    if missing:
        raise ValueError(f"readout is missing columns, e.g. {missing[:5]}")
    return {key: _number(row[key]) for key in sorted(keys)}


def sorted_readouts(readouts: pd.DataFrame) -> pd.DataFrame:
    """One truck's readouts in ascending ``time_step`` (positions 0..n-1)."""
    if readouts.empty:
        raise ValueError("the truck has no readouts")
    if readouts[TIME_COL].duplicated().any():
        raise ValueError("time_step must be unique within a truck")
    return readouts.sort_values(TIME_COL).reset_index(drop=True)


def build_payload(
    readouts: pd.DataFrame,
    specifications: Mapping[str, str],
    position: int = -1,
    include_previous: bool = True,
    vehicle_id: int | None = None,
) -> dict[str, Any]:
    """Request body for the readout at ``position`` of one truck.

    Args:
        readouts: All readouts of ONE truck (any order; sorted here).
        specifications: ``Spec_0`` ... ``Spec_7`` of the truck.
        position: Which readout to score (0 = first, -1 = last).
        include_previous: Send the readout immediately before it. The first readout of a
            truck has none, so ``previous`` is then omitted (the API gives NaN deltas).
        vehicle_id: Only for the API's log; never a model input.
    """
    ordered = sorted_readouts(readouts)
    n = len(ordered)
    if not -n <= position < n:
        raise ValueError(f"position {position} outside 0..{n - 1}")
    position %= n
    body: dict[str, Any] = {
        "current": _readout(ordered.iloc[position], readout_keys()),
        "specifications": {c: str(specifications[c]) for c in SPEC_COLUMNS},
    }
    if include_previous and position > 0:
        body["previous"] = _readout(ordered.iloc[position - 1], previous_keys())
    if vehicle_id is not None:
        body["vehicle_id"] = int(vehicle_id)
    return body


def risk_history(
    client: ApiClient,
    readouts: pd.DataFrame,
    specifications: Mapping[str, str],
    n_latest: int,
    vehicle_id: int | None = None,
) -> pd.DataFrame:
    """Score the latest ``n_latest`` readouts (each with its predecessor) through the API.

    Returns ``time_step, risk, threshold, alert`` in time order.
    """
    ordered = sorted_readouts(readouts)
    first = max(0, len(ordered) - n_latest)
    rows = []
    for position in range(first, len(ordered)):
        body = build_payload(ordered, specifications, position, True, vehicle_id)
        answer = client.predict(body)
        rows.append({TIME_COL: float(ordered.loc[position, TIME_COL]), "risk": answer["risk"],
                     "threshold": answer["threshold"], "alert": answer["alert"]})  # fmt: skip
    return pd.DataFrame(rows)
