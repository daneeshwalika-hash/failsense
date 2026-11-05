"""RUL labels for C-MAPSS training engines.

In the training file every engine runs until it fails, so the last cycle of an engine
is its failure cycle and

    RUL = max_cycle - cycle            (cycles still to go)

capped at ``rul.cap`` (config, 125). Why the cap: early in an engine's life the sensors
show no degradation, so the true RUL there is not recoverable from the data; a model
that has to tell 300 from 200 cycles only learns noise and its error is dominated by
that range. Capping gives the "piece-wise linear" target commonly used for FD001. The
cap changes the *target*, so RMSE values are only comparable with the same cap.

Leakage rule: ``max_cycle`` (known only because the engine ran to failure) and the RUL
itself are labels, never features. ``max_cycle`` is used inside ``compute_rul`` and is
**not** stored in the frame.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from failsense.rul.data import CYCLE_COL, ENGINE_COL

logger = logging.getLogger(__name__)

RUL_COL = "rul"
MAX_CYCLE_COL = "max_cycle"
# Columns that must never reach a model: the label, the value it is built from, and
# the engine id (only the CV group).
FORBIDDEN_FEATURES: frozenset[str] = frozenset({RUL_COL, MAX_CYCLE_COL, ENGINE_COL})


def rul_cap_from_config(config: Mapping[str, Any]) -> int:
    """Read ``rul.cap`` (a positive integer) from the config."""
    try:
        cap = config["rul"]["cap"]
    except KeyError as exc:
        raise KeyError("config needs 'rul.cap'") from exc
    return validate_cap(cap)


def validate_cap(cap: object) -> int:
    """Return ``cap`` if it is a positive int (``bool`` is rejected), else raise."""
    if isinstance(cap, bool) or not isinstance(cap, int):
        raise TypeError(f"rul cap must be an int, got {cap!r}")
    if cap < 1:
        raise ValueError(f"rul cap must be >= 1, got {cap}")
    return cap


def compute_rul(df: pd.DataFrame, cap: int | None = None) -> pd.Series:
    """RUL per row: cycles until the last cycle of the same engine, optionally capped.

    Args:
        df: Training rows with ``engine`` and ``cycle`` (every engine complete up to
            its failure).
        cap: Upper limit of the RUL; ``None`` = no cap.

    Returns:
        A Series named ``rul`` with the same index and order as ``df`` (int64).
        The result does not depend on the row order: the last cycle is the
        maximum of the engine, not the last row.

    Raises:
        ValueError: If ``engine`` or ``cycle`` is missing, or ``cap`` is invalid.
    """
    missing = [c for c in (ENGINE_COL, CYCLE_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if cap is not None:
        validate_cap(cap)
    max_cycle = df.groupby(ENGINE_COL)[CYCLE_COL].transform("max")
    rul = (max_cycle - df[CYCLE_COL]).astype("int64")
    if cap is not None:
        rul = rul.clip(upper=cap)
    return rul.rename(RUL_COL)


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    """Raise if a label or the engine id is in the feature list."""
    found = sorted(set(columns) & FORBIDDEN_FEATURES)
    if found:
        raise ValueError(f"forbidden columns in features (leakage): {found}")
