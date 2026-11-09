"""Demo-data sampling, and: the dashboard never imports the model stack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from failsense.dashboard import data


def test_sample_is_reproducible_distinct_and_capped() -> None:
    ids = pd.Series([5, 5, 7, 9, 11, 13])
    a = data.sample_vehicle_ids(ids, 3, seed=42)
    assert a == data.sample_vehicle_ids(ids, 3, seed=42) and len(set(a)) == 3
    assert data.sample_vehicle_ids(ids, 100, seed=1) == [5, 7, 9, 11, 13]
    with pytest.raises(ValueError, match="no trucks"):
        data.sample_vehicle_ids(pd.Series([], dtype=int), 3, seed=1)


def test_load_demo_trucks_reads_only_the_validation_split(make_mini_scania, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = make_mini_scania()
    processed = Path(config["datasets"]["scania"]["processed_dir"])
    labels = pd.DataFrame({"vehicle_id": [100, 101, 102, 103], "class_label": [0, 1, 2, 4]})
    labels.to_parquet(processed / "validation_labels.parquet")
    scania = config["datasets"]["scania"]
    scania["validation"]["labels"] = "validation_labels.csv"
    config["dashboard"] = {"n_demo_trucks": 2, "seed": 42}
    # A test split that cannot be read: any access would raise.
    for name in ("test_operational_readouts", "test_specifications"):
        (processed / f"{name}.parquet").unlink()

    readouts, specs, label = data.load_demo_trucks(config)

    assert set(readouts["vehicle_id"]) == set(specs["vehicle_id"]) == set(label.index)
    assert len(label) == 2 and set(label.index) <= {100, 101, 102, 103}
    assert readouts["vehicle_id"].nunique() == 2


def test_dashboard_helpers_do_not_import_the_model_stack() -> None:
    code = (
        "import sys\n"
        "import failsense.dashboard.client\n"
        "import failsense.dashboard.payloads\n"
        "import failsense.dashboard.data\n"
        "bad = [m for m in ('mlflow', 'xgboost', 'lightgbm', 'shap') if m in sys.modules]\n"
        "print(bad)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("[]"), out.stdout
