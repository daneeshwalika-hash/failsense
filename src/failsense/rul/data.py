"""Load and validate the C-MAPSS FD001 **training** file.

The file has no header: one row per engine and cycle, 26 numbers separated by spaces
(a trailing space ends each line): engine id, cycle, 3 operational settings, 21 sensors
(readme.txt of the dataset). The column names are defined once, here.

Only the training file is loaded. ``test_FD001.txt`` and ``RUL_FD001.txt`` are used
once, at the very end, and deliberately have no loader in this package.

C-MAPSS is **simulated** data (NASA Ames Prognostics Center of Excellence; Saxena et al.,
2008): the RUL part of FailSense demonstrates the method, not real engine performance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from failsense.data.schema import validate_frame

logger = logging.getLogger(__name__)

ENGINE_COL = "engine"
CYCLE_COL = "cycle"
SETTING_COLUMNS: tuple[str, ...] = tuple(f"setting_{i}" for i in range(1, 4))
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"sensor_{i}" for i in range(1, 22))
COLUMNS: tuple[str, ...] = (ENGINE_COL, CYCLE_COL, *SETTING_COLUMNS, *SENSOR_COLUMNS)
# Model inputs: operating conditions and sensors. Engine id and cycle are not here.
FEATURE_COLUMNS: tuple[str, ...] = (*SETTING_COLUMNS, *SENSOR_COLUMNS)


def _cycle_increasing(df: pd.DataFrame) -> pd.Series:
    # Like LAG() OVER (PARTITION BY engine ORDER BY row): the cycle must be larger
    # than the previous cycle of the same engine. The first cycle has no
    # predecessor (NaN) and is allowed. Duplicates and going backwards fail.
    step = df.groupby(ENGINE_COL, sort=False)[CYCLE_COL].diff()
    return step.isna() | (step > 0)


def _not_empty(df: pd.DataFrame) -> bool:
    return len(df) > 0


def rul_train_schema() -> pa.DataFrameSchema:
    """Data contract for the FD001 training table (strict, no type coercion).

    ``strict=True`` rejects extra columns: a label such as ``rul`` or ``max_cycle``
    can never arrive in the raw table unnoticed.
    """
    columns: dict[str, pa.Column] = {
        ENGINE_COL: pa.Column("int64", pa.Check.ge(1), nullable=False),
        CYCLE_COL: pa.Column("int64", pa.Check.ge(1), nullable=False),
    }
    columns.update(
        {c: pa.Column("float64", nullable=False) for c in (*SETTING_COLUMNS, *SENSOR_COLUMNS)}
    )
    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            pa.Check(_not_empty, name="not_empty"),
            pa.Check(_cycle_increasing, name="cycle_increasing_per_engine"),
        ],
        strict=True,
        coerce=False,
        name="cmapss_fd001_train",
    )


def load_train(path: str | Path) -> pd.DataFrame:
    """Read ``train_FD001.txt``, set names and dtypes, and check the data contract.

    Args:
        path: Path to the whitespace-separated file without header.

    Returns:
        DataFrame with ``COLUMNS``; ``engine`` and ``cycle`` are int64, everything
        else float64 (some sensors are whole numbers in the file; one dtype keeps
        the contract simple).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have exactly 26 columns.
        pandera.errors.SchemaErrors: If the data contract is violated.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    # sep=r"\s+" also swallows the trailing space at the end of every line.
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns in {path.name}, found {raw.shape[1]}")
    raw.columns = list(COLUMNS)
    typed = raw.astype({c: "float64" for c in (*SETTING_COLUMNS, *SENSOR_COLUMNS)})
    typed = typed.astype({ENGINE_COL: "int64", CYCLE_COL: "int64"})
    validate_frame(typed, rul_train_schema())
    logger.info(
        "Loaded %d rows of %d engines from %s", len(typed), typed[ENGINE_COL].nunique(), path.name
    )
    return typed
