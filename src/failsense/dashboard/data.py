"""Demo trucks for the dashboard: a fixed sample of the OFFICIAL VALIDATION split.

The test split is never read here. The validation labels are returned only so the page can show
the known outcome next to the prediction; they are not sent to the API.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from failsense.config import resolve_path
from failsense.data.labels import VEHICLE_COL

logger = logging.getLogger(__name__)


def _table_path(config: Mapping[str, Any], table: str) -> Path:
    """Parquet copy of a validation table if it exists, else the raw CSV.

    (Same rule as ``failsense.models.baseline_cv.table_path``, written out here so the dashboard
    does not import the training stack: scikit-learn, MLflow ... stay out of this page.)
    """
    scania = config["datasets"]["scania"]
    csv_path = resolve_path(scania["validation"][table])
    parquet_path = resolve_path(scania["processed_dir"]) / f"{csv_path.stem}.parquet"
    return parquet_path if parquet_path.exists() else csv_path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def sample_vehicle_ids(vehicle_ids: pd.Series, n: int, seed: int) -> list[int]:
    """A reproducible sample of ``n`` distinct truck ids (all of them if fewer exist)."""
    unique = pd.Series(vehicle_ids.unique())
    if unique.empty:
        raise ValueError("no trucks to sample from")
    picked = unique.sample(n=min(n, len(unique)), random_state=seed)
    return sorted(int(v) for v in picked)


def load_demo_trucks(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Readouts, specifications and the class label of the sampled validation trucks.

    Returns:
        ``(readouts, specs, label)``: readouts of the sampled trucks, their specification rows,
        and ``class_label`` (0..4) indexed by ``vehicle_id``.
    """
    section = config["dashboard"]
    specs_all = _read(_table_path(config, "specifications"))
    ids = sample_vehicle_ids(specs_all[VEHICLE_COL], section["n_demo_trucks"], section["seed"])
    path = _table_path(config, "readouts")
    if path.suffix == ".parquet":  # predicate pushdown: only the sampled trucks are read
        readouts = pd.read_parquet(path, filters=[(VEHICLE_COL, "in", ids)])
    else:
        readouts = _read(path)
        readouts = readouts[readouts[VEHICLE_COL].isin(ids)]
    labels = _read(_table_path(config, "labels"))
    label = labels.set_index(VEHICLE_COL)["class_label"].loc[ids]
    specs = specs_all[specs_all[VEHICLE_COL].isin(ids)].reset_index(drop=True)
    logger.info("Demo trucks: %d trucks, %d readouts (validation split)", len(ids), len(readouts))
    return readouts.reset_index(drop=True), specs, label
