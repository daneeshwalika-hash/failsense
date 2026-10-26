"""Tests for failsense.data.convert (small fixture files only, never the raw data)."""

import os
from pathlib import Path

import pandas as pd
import pytest

from failsense.data.convert import convert_scania, csv_to_parquet, scania_csv_files

SPLIT_FILES = {
    "train": {"readouts": "train_r.csv", "tte": "train_tte.csv"},
    "validation": {"readouts": "val_r.csv", "labels": "val_labels.csv"},
    "test": {"readouts": "test_r.csv", "labels": "test_labels.csv"},
}


@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    """A tiny readouts-like CSV with an int id, a float time step and a missing value."""
    path = tmp_path / "small.csv"
    path.write_text("vehicle_id,time_step,171_0\n0,1.5,10.0\n0,2.8,\n7,1.0,3.5\n")
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, dict]:
    """A minimal repo layout with six small CSV files and a matching config."""
    raw = tmp_path / "data" / "raw" / "scania"
    raw.mkdir(parents=True)
    scania: dict = {"processed_dir": "data/processed/scania"}
    for split, files in SPLIT_FILES.items():
        scania[split] = {}
        for key, name in files.items():
            (raw / name).write_text("vehicle_id,value\n1,0.5\n2,1.5\n")
            scania[split][key] = f"data/raw/scania/{name}"
    return tmp_path, {"datasets": {"scania": scania}}


def test_csv_to_parquet_round_trip(small_csv: Path, tmp_path: Path) -> None:
    target = tmp_path / "out" / "small.parquet"

    rows = csv_to_parquet(small_csv, target)

    assert rows == 3
    pd.testing.assert_frame_equal(pd.read_parquet(target), pd.read_csv(small_csv))


def test_csv_to_parquet_keeps_dtypes_and_missing_values(small_csv: Path, tmp_path: Path) -> None:
    target = tmp_path / "small.parquet"
    csv_to_parquet(small_csv, target)

    df = pd.read_parquet(target)

    assert pd.api.types.is_integer_dtype(df["vehicle_id"])
    assert pd.api.types.is_float_dtype(df["time_step"])
    assert df["171_0"].isna().sum() == 1


def test_csv_to_parquet_skips_up_to_date_file(small_csv: Path, tmp_path: Path) -> None:
    target = tmp_path / "small.parquet"
    csv_to_parquet(small_csv, target)

    assert csv_to_parquet(small_csv, target) is None
    assert csv_to_parquet(small_csv, target, force=True) == 3


def test_csv_to_parquet_reconverts_when_csv_is_newer(small_csv: Path, tmp_path: Path) -> None:
    target = tmp_path / "small.parquet"
    csv_to_parquet(small_csv, target)
    newer = target.stat().st_mtime + 10
    os.utime(small_csv, (newer, newer))

    assert csv_to_parquet(small_csv, target) == 3


def test_csv_to_parquet_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        csv_to_parquet(tmp_path / "missing.csv", tmp_path / "x.parquet")


def test_csv_to_parquet_header_only_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("vehicle_id,time_step\n")

    with pytest.raises(ValueError):
        csv_to_parquet(empty, tmp_path / "empty.parquet")


def test_scania_csv_files_lists_all_splits(fake_repo: tuple[Path, dict]) -> None:
    _, config = fake_repo

    assert len(scania_csv_files(config)) == 6


def test_convert_scania_writes_one_parquet_per_csv(
    fake_repo: tuple[Path, dict],
) -> None:
    root, config = fake_repo

    paths = convert_scania(config, root=root)

    assert [p.name for p in paths] == [
        "train_r.parquet",
        "train_tte.parquet",
        "val_r.parquet",
        "val_labels.parquet",
        "test_r.parquet",
        "test_labels.parquet",
    ]
    assert all(p.parent == root / "data" / "processed" / "scania" for p in paths)
    assert all(p.is_file() for p in paths)
