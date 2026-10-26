"""Convert the SCANIA Component X CSV files to Parquet.

Why (measured on 26.10.2026, 16 cores, pandas 3.1, file in page cache):

- Reading a few columns is much faster: 4 of 107 columns from the 1.2 GB training
  readouts take 0.94 s from CSV but 0.04 s from Parquet (about 23x). Parquet is
  columnar, so only the requested columns are read; CSV must scan every line.
- Reading the full file is only about 2x faster (1.1 s vs. 0.5 s).
- Column types are stored in the file (no type guessing on every read).
- The files are about half the size (1.2 GB -> 566 MB) and DuckDB can query them directly.

The raw CSV files stay untouched in ``data/raw/`` (raw data is immutable);
Parquet files go to ``data/processed/``.

Usage:
    uv run python -m failsense.data.convert            # skip files that are up to date
    uv run python -m failsense.data.convert --force    # convert everything again
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from failsense.config import find_repo_root, load_config, resolve_path

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def is_up_to_date(source: Path, target: Path) -> bool:
    """Return True if ``target`` exists and is not older than ``source``."""
    return target.is_file() and target.stat().st_mtime >= source.stat().st_mtime


def csv_to_parquet(csv_path: Path, parquet_path: Path, *, force: bool = False) -> int | None:
    """Convert one CSV file to Parquet.

    Args:
        csv_path: Source CSV file with a header row.
        parquet_path: Target Parquet file. Parent folders are created if needed.
        force: Convert even if the Parquet file is already up to date.

    Returns:
        Number of rows written, or ``None`` if the file was skipped.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV file has no data rows.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not force and is_up_to_date(csv_path, parquet_path):
        logger.info("skip %s (Parquet is up to date)", csv_path.name)
        return None

    start = time.perf_counter()
    df = pd.read_csv(csv_path, engine="pyarrow")
    if df.empty:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    logger.info(
        "%s -> %s: %d rows x %d columns, %.1f MB -> %.1f MB in %.1f s",
        csv_path.name,
        parquet_path.name,
        len(df),
        df.shape[1],
        csv_path.stat().st_size / 1e6,
        parquet_path.stat().st_size / 1e6,
        time.perf_counter() - start,
    )
    return len(df)


def scania_csv_files(config: dict[str, Any]) -> list[str]:
    """List all SCANIA CSV paths from the config (train, validation, test)."""
    scania = config["datasets"]["scania"]
    return [path for split in SPLITS for path in scania[split].values()]


def convert_scania(
    config: dict[str, Any], *, root: Path | None = None, force: bool = False
) -> list[Path]:
    """Convert every SCANIA CSV file listed in the config to Parquet.

    Args:
        config: Loaded configuration (see ``configs/config.yaml``).
        root: Repository root. Defaults to the root found from the working directory.
        force: Convert even if a Parquet file is already up to date.

    Returns:
        The Parquet paths, in the same order as the CSV files in the config.
    """
    root = root or find_repo_root()
    processed_dir = resolve_path(config["datasets"]["scania"]["processed_dir"], root)

    parquet_paths = []
    for relative_csv in scania_csv_files(config):
        csv_path = resolve_path(relative_csv, root)
        parquet_path = processed_dir / f"{csv_path.stem}.parquet"
        csv_to_parquet(csv_path, parquet_path, force=force)
        parquet_paths.append(parquet_path)
    return parquet_paths


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Convert SCANIA CSV files to Parquet.")
    parser.add_argument("--force", action="store_true", help="convert all files again")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    paths = convert_scania(load_config(), force=args.force)
    logger.info("done: %d Parquet files in %s", len(paths), paths[0].parent)


if __name__ == "__main__":
    main()
