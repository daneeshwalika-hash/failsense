"""Features of the OFFICIAL validation split: the last readout of every truck.

The challenge scores the validation split at one readout per truck (the last one in the
file). This module builds exactly the model input the registered model expects:

* the last readout per truck (largest ``time_step``),
* counter Δ / Δ-per-time-step from the **previous readouts of the same truck** only
  (``groupby(vehicle).diff()`` looks backwards; the deltas are computed on the full sorted
  readouts and the last rows are selected afterwards),
* histogram features from the row itself, the 8 ``Spec_*`` columns,
* the model input columns in the exact order stored in ``reports/registry_q1.json``.

Labels (``class_label`` 0..4) are attached as ``proximity_class`` / ``repair_within_horizon``
for scoring only; they are never part of the model input.
The test split is not touched here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from failsense.config import resolve_path
from failsense.data.labels import BINARY_COL, CLASS_COL, TIME_COL, VEHICLE_COL
from failsense.data.schema import counter_columns, histogram_groups
from failsense.features.build import attach_specifications, features_params_from_config
from failsense.models.baseline_cv import frame_from_sampled, read_table, table_path
from failsense.models.register import REGISTRY_FILE

logger = logging.getLogger(__name__)

LABEL_FILE_COLUMN = "class_label"
N_CLASSES = 5


def registry_input_columns(config: Mapping[str, Any]) -> list[str]:
    """Input columns of the registered model, from ``<reports>/registry_q1.json``."""
    path = resolve_path(config["paths"]["reports"]) / REGISTRY_FILE
    columns = json.loads(path.read_text(encoding="utf-8"))["input_columns"]
    if len(set(columns)) != len(columns):
        raise ValueError(f"{path} lists duplicate input columns")
    return list(columns)


def build_last_readout_frame(
    readouts: pd.DataFrame,
    specs: pd.DataFrame,
    labels: pd.DataFrame,
    counter_cols: Sequence[str],
    hist_groups: Mapping[str, Sequence[str]],
    top_fraction: float,
) -> pd.DataFrame:
    """One row per truck: its last readout with engineered features, specs and labels.

    Args:
        readouts: Validation readouts (``vehicle_id``, ``time_step``, raw columns).
        specs: One specification row per truck.
        labels: ``vehicle_id`` and ``class_label`` (0..4), one row per truck.
        counter_cols, hist_groups, top_fraction: Engineered features as in training.

    Returns:
        Frame indexed 0..n-1 sorted by ``vehicle_id`` with the model inputs plus
        ``proximity_class`` (0..4) and ``repair_within_horizon`` (class >= 1).

    Raises:
        ValueError: On missing columns, duplicate or mismatching trucks between readouts
            and labels, or labels outside 0..4.
    """
    for name, frame, needed in (
        ("readouts", readouts, (VEHICLE_COL, TIME_COL)),
        ("labels", labels, (VEHICLE_COL, LABEL_FILE_COLUMN)),
    ):
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    if readouts.empty:
        raise ValueError("readouts is empty")
    if labels[VEHICLE_COL].duplicated().any():
        raise ValueError("labels has more than one row for a truck")
    if set(labels[VEHICLE_COL]) != set(readouts[VEHICLE_COL]):
        raise ValueError("trucks in readouts and labels differ")
    if not labels[LABEL_FILE_COLUMN].isin(range(N_CLASSES)).all():
        raise ValueError(f"{LABEL_FILE_COLUMN} must be an integer in 0..{N_CLASSES - 1}")

    # Sort so that "previous readout" really means "earlier time_step"; the strict
    # increase is checked again inside add_counter_deltas.
    ordered = readouts.sort_values([VEHICLE_COL, TIME_COL], kind="stable").reset_index(drop=True)
    last_idx = ordered.groupby(VEHICLE_COL)[TIME_COL].idxmax()  # row of the last readout
    cls = labels.set_index(VEHICLE_COL)[LABEL_FILE_COLUMN]
    vehicles = ordered.loc[last_idx.to_numpy(), VEHICLE_COL]
    sampled = pd.DataFrame(
        {BINARY_COL: (cls.loc[vehicles].to_numpy() >= 1).astype(int)}, index=last_idx.to_numpy()
    )
    frame = frame_from_sampled(ordered, sampled, counter_cols, hist_groups, top_fraction)
    frame[CLASS_COL] = cls.loc[frame[VEHICLE_COL]].to_numpy().astype(int)
    frame = attach_specifications(frame, specs)
    return frame.sort_values(VEHICLE_COL).reset_index(drop=True)


def select_model_inputs(frame: pd.DataFrame, input_columns: Sequence[str]) -> pd.DataFrame:
    """The model input in the registered column order; refuses missing columns.

    Raises:
        ValueError: If a registered input column is missing from ``frame``.
    """
    missing = [c for c in input_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{len(missing)} registered input columns missing, e.g. {missing[:5]}")
    return frame[list(input_columns)]


def build_validation_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    """Official validation split -> one row per truck (inputs + labels), 5,046 rows.

    Reads the validation readouts, specifications and labels named in the config.
    """
    readouts = read_table(table_path(config, "validation", "readouts"))
    specs = read_table(table_path(config, "validation", "specifications"))
    labels_path = resolve_path(config["datasets"]["scania"]["validation"]["labels"])
    labels = read_table(labels_path)
    frame = build_last_readout_frame(
        readouts, specs, labels, counter_columns(), histogram_groups(),
        features_params_from_config(config),
    )  # fmt: skip
    logger.info(
        "Validation frame: %d trucks, %d positives (%.2f %%)",
        len(frame), int(frame[BINARY_COL].sum()), 100 * frame[BINARY_COL].mean(),
    )  # fmt: skip
    return frame


def class_counts(frame: pd.DataFrame) -> dict[int, int]:
    """Number of trucks per true class 0..4 (documentation only, never a threshold input)."""
    counts = np.bincount(frame[CLASS_COL].to_numpy(dtype=int), minlength=N_CLASSES)
    return {k: int(n) for k, n in enumerate(counts)}
