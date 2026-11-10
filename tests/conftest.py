"""Shared fixtures: a tiny SCANIA-shaped dataset on disk (never the raw data).

``make_mini_scania`` writes Parquet copies of all three splits into a temporary
folder and returns a config that points at them, so tests of the loaders, the
data contract and the training entry points run without ``data/``.
"""

from __future__ import annotations

import builtins
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from failsense.data.labels import REPAIR_COL, STUDY_LENGTH_COL, TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, feature_columns

N_READOUTS = 12  # readouts per truck, time_step 0, 10, ..., 110
STUDY_LENGTH = 115.0


def mini_readouts(vehicle_ids: list[int], seed: int = 0) -> pd.DataFrame:
    """Readouts that pass the data contract: increasing time, counters that only grow."""
    rng = np.random.default_rng(seed)
    parts = []
    for vid in vehicle_ids:
        part = {
            VEHICLE_COL: np.full(N_READOUTS, vid, dtype="int64"),
            TIME_COL: np.arange(N_READOUTS, dtype=float) * 10.0,
        }
        for column in feature_columns():
            part[column] = np.cumsum(rng.uniform(1, 5, size=N_READOUTS))  # non-negative
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)


def mini_specs(vehicle_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array(vehicle_ids, dtype="int64"),
            **{c: [f"Cat{v % 3}" for v in vehicle_ids] for c in SPEC_COLUMNS},
        }
    )


def mini_tte(vehicle_ids: list[int]) -> pd.DataFrame:
    """Every 4th truck is repaired at the end of its study; the others are censored."""
    return pd.DataFrame(
        {
            VEHICLE_COL: np.array(vehicle_ids, dtype="int64"),
            STUDY_LENGTH_COL: np.full(len(vehicle_ids), STUDY_LENGTH),
            REPAIR_COL: np.array([int(v % 4 == 0) for v in vehicle_ids], dtype="int64"),
        }
    )


@pytest.fixture
def mini_frames() -> dict[str, pd.DataFrame]:
    """Readouts, specs and tte of 40 trucks (what the training loader would read)."""
    ids = list(range(40))
    return {"readouts": mini_readouts(ids), "specs": mini_specs(ids), "tte": mini_tte(ids)}


@pytest.fixture
def make_mini_scania(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Factory: write a tiny dataset as Parquet and return a matching config.

    ``shared_vehicle``: a truck id that is put into train **and** validation, to
    test the split-overlap protection.
    """

    def factory(shared_vehicle: int | None = None) -> dict[str, Any]:
        processed = tmp_path / "processed"
        processed.mkdir()
        split_ids = {
            "train": list(range(40)),
            "validation": [100, 101, 102, 103],
            "test": [200, 201, 202, 203],
        }
        if shared_vehicle is not None:
            split_ids["validation"].append(shared_vehicle)
        files: dict[str, dict[str, str]] = {}
        for split, ids in split_ids.items():
            mini_readouts(ids, seed=len(ids)).to_parquet(
                processed / f"{split}_operational_readouts.parquet"
            )
            mini_specs(ids).to_parquet(processed / f"{split}_specifications.parquet")
            files[split] = {
                "readouts": f"{split}_operational_readouts.csv",
                "specifications": f"{split}_specifications.csv",
            }
        mini_tte(split_ids["train"]).to_parquet(processed / "train_tte.parquet")
        files["train"]["tte"] = "train_tte.csv"
        return {
            "datasets": {"scania": {"processed_dir": str(processed), **files}},
            "labels": {"class_bounds": [48, 24, 12, 6]},
            "sampling": {"n_cutoffs_per_vehicle": 5, "seed": 42},
            "features": {"top_fraction": 0.2},
            "cv": {"n_splits": 3, "seed": 42},
            "seed": 42,
            "paths": {"reports": str(tmp_path / "reports")},
            "tuning": {
                "n_trials": 4,
                "timeout_minutes": 2,
                "early_stopping_rounds": 5,
                "max_rounds": 40,
                "inner_val_fraction": 0.3,
                "seed": 42,
                "top_fraction": 0.2,
                "storage": str(tmp_path / "optuna.db"),
                "registered_model": "failsense-q1",
                "alias": "candidate",
            },
            "mlflow": {
                "db_path": str(tmp_path / "mlflow.db"),
                "artifact_dir": str(tmp_path / "artifacts"),
                "experiment": "test-train",
            },
        }

    return factory


# --- tiny C-MAPSS FD001-shaped data ---------------------------------------------------
def mini_cmapss_frame(engine_lengths: list[int], seed: int = 0) -> pd.DataFrame:
    """Engines with 1..n cycles, 3 settings and 21 sensors (some constant, like FD001)."""
    from failsense.rul.data import COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS

    rng = np.random.default_rng(seed)
    parts = []
    for engine, n in enumerate(engine_lengths, start=1):
        cycle = np.arange(1, n + 1)
        wear = cycle / n  # sensors drift with wear, so RUL is learnable
        part = {"engine": np.full(n, engine, dtype="int64"), "cycle": cycle.astype("int64")}
        for i, c in enumerate(SETTING_COLUMNS):
            part[c] = np.full(n, 100.0) if i == 2 else rng.normal(0, 0.01, n)  # setting_3 constant
        for i, c in enumerate(SENSOR_COLUMNS):
            part[c] = (
                np.full(n, 518.67)  # constant sensors 1, 5
                if i in (0, 4)
                else 500 + (i % 5) * 10 * wear + rng.normal(0, 0.5, n)
            )
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)[list(COLUMNS)]


def write_cmapss_txt(frame: pd.DataFrame, path: Path) -> Path:
    """Write like the original file: no header, spaces, a trailing space on every line."""
    lines = [
        " ".join(f"{v:g}" if not isinstance(v, str) else v for v in row) + " "
        for row in frame.to_numpy()
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def make_mini_cmapss(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Factory: write a tiny FD001-like training file and return a matching config.

    The config's test and RUL files point to paths that do not exist: any code that
    tries to read them fails, which is what the leakage test relies on.
    """

    def factory(engine_lengths: list[int] | None = None) -> dict[str, Any]:
        lengths = engine_lengths or [30, 35, 40, 45, 50, 32, 38, 42, 48, 36]
        train = write_cmapss_txt(mini_cmapss_frame(lengths), tmp_path / "train_FD001.txt")
        return {
            "datasets": {
                "cmapss": {
                    "train": str(train),
                    "test": str(tmp_path / "does_not_exist_test_FD001.txt"),
                    "rul": str(tmp_path / "does_not_exist_RUL_FD001.txt"),
                }
            },
            "rul": {"cap": 25, "mlflow_experiment": "test-rul"},
            "cv": {"n_splits": 5, "seed": 42},
            "paths": {"reports": str(tmp_path / "reports")},
            "mlflow": {
                "db_path": str(tmp_path / "mlflow.db"),
                "artifact_dir": str(tmp_path / "artifacts"),
                "experiment": "test-rul",
            },
        }

    return factory


@pytest.fixture
def make_cmapss_frame() -> Callable[..., pd.DataFrame]:
    """The ``mini_cmapss_frame`` function as a fixture (tests is not an importable package)."""
    return mini_cmapss_frame


@pytest.fixture
def write_cmapss_file() -> Callable[[pd.DataFrame, Path], Path]:
    """The ``write_cmapss_txt`` function as a fixture."""
    return write_cmapss_txt


TINY_INPUT_COLUMNS = [
    "time_step", "171_0", "171_0__delta", "171_0__rate",
    "167_0", "167_1", "167__mean_bin", "Spec_0",
]  # fmt: skip


def tiny_explain_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Small frame with the naming of the real inputs; the label depends on a delta and a spec."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "time_step": rng.uniform(10, 200, n),
            "171_0": rng.uniform(0, 100, n),
            "171_0__delta": rng.normal(0, 1, n),
            "171_0__rate": rng.normal(0, 1, n),
            "167_0": rng.uniform(0, 50, n),
            "167_1": rng.uniform(0, 50, n),
            "167__mean_bin": rng.uniform(0, 9, n),
            "Spec_0": rng.choice(["A", "B", "C"], n),
        }
    )
    logit = 2.0 * frame["171_0__delta"] + 1.5 * (frame["Spec_0"] == "A") - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame, y


@pytest.fixture
def tiny_pipeline() -> tuple[Any, pd.DataFrame, np.ndarray]:
    """A fitted ``Pipeline(preprocess, XGBoost)`` plus its training frame and labels."""
    from xgboost import XGBClassifier

    from failsense.models.pipeline import build_model_pipeline

    frame, y = tiny_explain_frame()
    model = XGBClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.2, importance_type="gain",
        n_jobs=1, random_state=0, verbosity=0,
    )  # fmt: skip
    pipeline = build_model_pipeline(model, ["Spec_0"], scaler=None).fit(frame, y)
    return pipeline, frame, y


# --- prediction API ---------------------------------------------------------------------
API_CALIBRATOR = {"kind": "sigmoid", "slope": 0.5, "intercept": -2.0, "logit_clip_eps": 1e-6}


@pytest.fixture
def service(tiny_pipeline: tuple[Any, Any, Any]) -> Any:
    """Service with the tiny pipeline; its input columns are the tiny frame's columns."""
    from failsense.api.service import ModelInfo, ModelService

    pipeline, frame, _ = tiny_pipeline
    return ModelService(
        pipeline=pipeline,
        calibrator=API_CALIBRATOR,
        threshold=0.2,
        input_columns=list(frame.columns),
        info=ModelInfo(name="failsense-q1", version="3", alias="candidate"),
        top_fraction=0.2,
    )


@pytest.fixture
def make_payload() -> Callable[..., dict[str, Any]]:
    """Factory for a valid /predict body (full 105-column current readout)."""

    from failsense.data.schema import SPEC_COLUMNS, counter_columns, feature_columns

    def factory(with_previous: bool = True, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "vehicle_id": 7,
            "current": {
                "time_step": 100.0,
                **{c: 10.0 + i for i, c in enumerate(feature_columns())},
            },
            "specifications": {c: "Cat1" for c in SPEC_COLUMNS},
        }
        if with_previous:
            body["previous"] = {"time_step": 90.0, **{c: 8.0 for c in counter_columns()}}
        body.update(overrides)
        return body

    return factory


# --- spy on data reads (guards of the test split) ---------------------------------------
DATA_SUFFIXES = (".csv", ".parquet", ".txt")


def is_test_split_file(path: str) -> bool:
    """True for a data file whose NAME marks it as a test-split file.

    ``test_operational_readouts.csv``, ``test_labels.parquet``, ``test_FD001.txt`` and
    ``RUL_FD001.txt`` match. Only the file name counts: pytest's temp folders contain the
    name of the running test (``test_...``) in their path.
    """
    name = Path(path).name
    return name.endswith(DATA_SUFFIXES) and ("test_" in name or "RUL_FD" in name)


class ReadSpy:
    """Every path given to ``pd.read_csv``, ``pd.read_parquet`` or ``open`` while it is active."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def record(self, target: Any) -> None:
        if isinstance(target, (str, bytes, os.PathLike)):
            self.paths.append(os.fsdecode(target))

    def test_split_reads(self) -> list[str]:
        """The recorded paths that are test-split files (must be empty for training code)."""
        return [p for p in self.paths if is_test_split_file(p)]


@pytest.fixture
def read_spy(monkeypatch: pytest.MonkeyPatch) -> ReadSpy:
    """Record every file that is read through pandas or ``open`` (the reads still happen)."""
    spy = ReadSpy()

    def wrap(original: Callable[..., Any], keyword: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            spy.record(args[0] if args else kwargs.get(keyword))
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(pd, "read_csv", wrap(pd.read_csv, "filepath_or_buffer"))
    monkeypatch.setattr(pd, "read_parquet", wrap(pd.read_parquet, "path"))
    monkeypatch.setattr(builtins, "open", wrap(builtins.open, "file"))
    monkeypatch.setattr(io, "open", wrap(io.open, "file"))  # pathlib.Path.open uses io.open
    return spy
