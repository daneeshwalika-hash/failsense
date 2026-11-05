"""Data contracts (pandera schemas) for the SCANIA Component X files.

Why this module exists
----------------------
A data contract is the Python version of ``NOT NULL``, ``CHECK`` and
``UNIQUE`` constraints in SQL Server: it states what valid input looks like,
and the pipeline stops with a clear message *before* bad data reaches a model.
Without it, a negative counter or a renamed column would silently produce a
worse model, and nobody would know why.

What the contract checks
------------------------
Operational readouts (one row per truck and readout):

* exactly the expected columns: ``vehicle_id``, ``time_step`` and 105 feature
  columns (8 counters + 6 histograms with 10-36 bins). ``strict=True`` rejects
  any extra column, so a label column such as ``in_study_repair`` can never
  end up next to the features by accident (leakage guard);
* dtypes as measured on 29.10.2026 (pandas 3.1): ``vehicle_id`` int64,
  ``time_step`` and all features float64;
* ``time_step`` and every feature are non-negative; features may be missing
  (44 of 105 columns contain NaN), ids and time steps may not;
* ``time_step`` is strictly increasing per truck (no duplicate readouts, no
  unsorted history), because the features of 30.10 are built from "previous
  readouts" and would be wrong on unsorted data;
* the table is not empty.

Specifications (one row per truck): unique ``vehicle_id``, the eight
``Spec_*`` columns as text matching ``Cat<number>``. The category *list* is not
fixed on purpose: the EDA showed categories in validation/test that do not
occur in training, which is valid data (``OneHotEncoder(handle_unknown="ignore")``
handles it), not a contract violation.

Time-to-event (training only): unique ``vehicle_id``, non-negative
``length_of_study_time_step``, ``in_study_repair`` in {0, 1}.

Usage:
    uv run python -m failsense.data.schema     # validate the Parquet files of all splits
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from failsense.config import find_repo_root, load_config, resolve_path
from failsense.data.labels import REPAIR_COL, STUDY_LENGTH_COL, TIME_COL, VEHICLE_COL
from failsense.data.split import assert_no_group_overlap

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")

# Structure of the anonymised SCANIA data (Kharazian et al., 2025), confirmed
# on the real files on 29.10.2026. These describe the dataset itself, not a
# tunable setting, so they live in code next to the other column names.
COUNTER_VARIABLES: tuple[str, ...] = ("171", "666", "427", "837", "309", "835", "370", "100")
HISTOGRAM_BINS: dict[str, int] = {"397": 36, "459": 20, "291": 11, "167": 10, "272": 10, "158": 10}
SPEC_COLUMNS: tuple[str, ...] = tuple(f"Spec_{i}" for i in range(8))
SPEC_CATEGORY_PATTERN = r"^Cat\d+$"


def counter_columns() -> list[str]:
    """Return the 8 counter columns, e.g. ``171_0`` (one column per variable)."""
    return [f"{variable}_0" for variable in COUNTER_VARIABLES]


def histogram_columns() -> list[str]:
    """Return the 97 histogram bin columns, e.g. ``167_0 ... 167_9``."""
    return [f"{variable}_{i}" for variable, n_bins in HISTOGRAM_BINS.items() for i in range(n_bins)]


def histogram_groups() -> dict[str, list[str]]:
    """Return the bin columns of every histogram, bins in order.

    Example: ``{"167": ["167_0", ..., "167_9"], ...}``. The bin order matters:
    the position of a bin is its index in the weighted mean (feature engineering).
    """
    return {
        variable: [f"{variable}_{i}" for i in range(n_bins)]
        for variable, n_bins in HISTOGRAM_BINS.items()
    }


def feature_columns() -> list[str]:
    """Return all 105 raw feature columns of the operational readouts."""
    return counter_columns() + histogram_columns()


# --- custom checks -----------------------------------------------------------
# pandera calls a DataFrame-level check with the whole DataFrame and a column
# check with the Series. A check returns a bool or a bool Series (one value per
# row); False marks a failure and shows up in the error report.


def _not_empty(df: pd.DataFrame) -> bool:
    return len(df) > 0


def _time_step_increasing(df: pd.DataFrame) -> pd.Series:
    # diff() inside each truck: current time_step minus the previous one of the
    # same truck (like LAG() OVER (PARTITION BY vehicle_id) in SQL Server).
    # The first readout of a truck has no predecessor -> NaN -> allowed.
    step = df.groupby(VEHICLE_COL, sort=False)[TIME_COL].diff()
    return step.isna() | (step > 0)


def _is_string_dtype(series: pd.Series) -> bool:
    # pandas 3 reads text as the new "str" dtype; older code paths may give
    # "object". Both are text, a numeric column is not.
    return pd.api.types.is_string_dtype(series.dtype)


# No `error=` text: pandera then reports the check *name*, which tests can assert on.
NOT_EMPTY = pa.Check(_not_empty, name="not_empty")


# --- schemas -----------------------------------------------------------------


def readouts_schema() -> pa.DataFrameSchema:
    """Contract for ``*_operational_readouts`` (train, validation and test)."""
    features = {
        column: pa.Column("float64", pa.Check.ge(0), nullable=True) for column in feature_columns()
    }
    return pa.DataFrameSchema(
        columns={
            VEHICLE_COL: pa.Column("int64", pa.Check.ge(0), nullable=False),
            TIME_COL: pa.Column("float64", pa.Check.ge(0), nullable=False),
            **features,
        },
        checks=[
            NOT_EMPTY,
            pa.Check(_time_step_increasing, name="time_step_increasing_per_vehicle"),
        ],
        strict=True,  # extra columns (e.g. label columns) are a violation
        coerce=False,  # never "repair" dtypes silently; wrong dtype = error
        name="scania_readouts",
    )


def specifications_schema() -> pa.DataFrameSchema:
    """Contract for ``*_specifications`` (one row per truck)."""
    specs = {
        column: pa.Column(
            None,  # dtype checked by _is_string_dtype (works for "str" and "object")
            [
                pa.Check(_is_string_dtype, name="is_string_dtype"),
                pa.Check.str_matches(SPEC_CATEGORY_PATTERN),
            ],
            nullable=False,
        )
        for column in SPEC_COLUMNS
    }
    return pa.DataFrameSchema(
        columns={VEHICLE_COL: pa.Column("int64", unique=True, nullable=False), **specs},
        checks=[NOT_EMPTY],
        strict=True,
        coerce=False,
        name="scania_specifications",
    )


def tte_schema() -> pa.DataFrameSchema:
    """Contract for ``train_tte`` (time-to-event, one row per truck)."""
    return pa.DataFrameSchema(
        columns={
            VEHICLE_COL: pa.Column("int64", unique=True, nullable=False),
            STUDY_LENGTH_COL: pa.Column("float64", pa.Check.ge(0), nullable=False),
            REPAIR_COL: pa.Column("int64", pa.Check.isin([0, 1]), nullable=False),
        },
        checks=[NOT_EMPTY],
        strict=True,
        coerce=False,
        name="scania_tte",
    )


# --- validation helpers ------------------------------------------------------


def validate_frame(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate ``df`` against ``schema`` and return it unchanged.

    ``lazy=True`` collects *all* violations before raising, instead of stopping
    at the first one, so one run shows every problem in the file.

    Raises:
        pandera.errors.SchemaErrors: If at least one check fails. The
            ``failure_cases`` attribute lists column, check and failing value.
    """
    try:
        validated = schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases
        logger.error(
            "%s: %d contract violation(s), first ones:\n%s",
            schema.name,
            len(cases),
            cases.head(20).to_string(),
        )
        raise
    logger.info("%s: %d rows x %d columns pass the contract", schema.name, *df.shape)
    return validated


def assert_specs_cover_readouts(readouts: pd.DataFrame, specs: pd.DataFrame) -> None:
    """Raise if a truck in ``readouts`` has no specification row.

    The specs are joined to the readouts by ``vehicle_id`` later; a missing
    spec row would turn into NaN features without any warning.
    """
    missing = set(readouts[VEHICLE_COL]) - set(specs[VEHICLE_COL])
    if missing:
        raise ValueError(
            f"{len(missing)} vehicle(s) have readouts but no specification row, "
            f"e.g. {sorted(missing)[:5]}"
        )


def _read_parquet(processed_dir: Path, raw_path: str) -> pd.DataFrame:
    path = processed_dir / f"{Path(raw_path).stem}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make parquet` first")
    return pd.read_parquet(path)


def validate_scania(config: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Validate the Parquet copies of all SCANIA splits against the contract.

    Splits are loaded one at a time to keep peak memory low (the training
    readouts alone are about 1 GB in memory).
    """
    root = root or find_repo_root()
    scania = config["datasets"]["scania"]
    processed_dir = resolve_path(scania["processed_dir"], root)

    vehicles_by_split: dict[str, set[int]] = {}
    for split in SPLITS:
        files = scania[split]
        readouts = validate_frame(
            _read_parquet(processed_dir, files["readouts"]), readouts_schema()
        )
        specs = validate_frame(
            _read_parquet(processed_dir, files["specifications"]), specifications_schema()
        )
        assert_specs_cover_readouts(readouts, specs)
        if "tte" in files:
            validate_frame(_read_parquet(processed_dir, files["tte"]), tte_schema())
        vehicles_by_split[split] = set(readouts[VEHICLE_COL])
        del readouts, specs  # free memory before the next split

    assert_no_group_overlap(vehicles_by_split)
    logger.info("All SCANIA splits pass the data contract")


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description="Validate the SCANIA data contract.").parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    validate_scania(load_config())


if __name__ == "__main__":
    main()
