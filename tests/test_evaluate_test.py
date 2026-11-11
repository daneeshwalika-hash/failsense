"""The single test evaluation: protections, metrics by hand and a full run on FIXTURES only.

No real test file is read here: the fixtures write their own tiny "test" files into a temp folder.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from failsense import evaluate_test as et
from failsense import freeze
from failsense.config import load_config
from failsense.data.schema import counter_columns, histogram_groups
from failsense.models.boosters import STEP6_PARAMS
from failsense.models.cost import alert_class_from_config, cost_matrix_from_config
from failsense.models.q2_classes import make_multiclass_pipeline
from failsense.models.register import TRUSTED_TYPES
from failsense.models.validation_data import build_last_readout_frame

MATRIX = cost_matrix_from_config(load_config())
ALERT_CLASS = alert_class_from_config(load_config())
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "failsense"


# --- metrics by hand ---------------------------------------------------------------------
def test_q1_metrics_by_hand_primary_and_literal_mapping() -> None:
    classes = np.array([0, 0, 0, 0, 1, 4])
    risk = np.array([0.01, 0.05, 0.20, 0.04, 0.30, 0.02])
    out = et.q1_metrics(classes, risk, 0.035, MATRIX, ALERT_CLASS, n_boot=50, seed=1)
    # alerted: trucks 1, 2, 3 (false alarms) and 4 (the class-1 truck); missed: the class-4 truck
    assert out["confusion"] == {"tp": 1, "fp": 3, "fn": 1, "tn": 1}
    assert out["prevalence"] == pytest.approx(2 / 6) and out["alert_rate"] == pytest.approx(4 / 6)
    assert out["precision"] == pytest.approx(1 / 4) and out["recall"] == pytest.approx(1 / 2)
    assert out["recall_per_class"] == {"1": 1.0, "4": 0.0}
    primary, literal = out["cost"]["primary"], out["cost"]["literal"]
    assert (primary["model"], primary["never_alert"], primary["always_alert"]) == (521, 700, 28)
    assert primary["saving_vs_always_alert"] == 28 - 521 < 0  # worse than "alert every truck"
    assert primary["saving_vs_never_alert"] == 700 - 521
    assert (literal["model"], literal["never_alert"], literal["always_alert"]) == (521, 700, 428)
    assert out["pr_auc"]["value"] == pytest.approx(average_precision_score(classes >= 1, risk))
    assert out["pr_auc"]["prevalence"] == out["prevalence"]  # the prevalence next to the PR-AUC
    assert out["pr_auc"]["lift"] == pytest.approx(out["pr_auc"]["value"] / out["prevalence"])


def test_q1_reports_the_constant_baselines_under_both_mappings() -> None:
    classes = np.array([0] * 8 + [2, 3])
    out = et.q1_metrics(classes, np.linspace(0, 0.1, 10), 0.05, MATRIX, ALERT_CLASS, 20, 1)
    for mapping in ("primary", "literal"):
        entry = out["cost"][mapping]
        assert {"never_alert", "always_alert", "model", "saving_vs_always_alert"} <= set(entry)
        assert entry["never_alert"] == 300 + 400  # misses of class 2 and 3
    assert out["cost"]["primary"]["always_alert"] == 8 * 7
    assert "model_minus_always_alert_bootstrap" in out["cost"]["primary"]


def test_q1_without_positives_has_no_pr_auc_but_still_the_costs() -> None:
    out = et.q1_metrics(np.zeros(5, int), np.linspace(0, 0.1, 5), 0.05, MATRIX, ALERT_CLASS, 10, 1)
    assert "pr_auc" not in out and out["cost"]["primary"]["never_alert"] == 0


@pytest.mark.parametrize(
    "classes, risk",
    [([], []), ([0, 1], [0.1]), ([0, 5], [0.1, 0.2]), ([-1, 1], [0.1, 0.2])],
)
def test_q1_rejects_bad_input(classes: list[int], risk: list[float]) -> None:
    with pytest.raises(ValueError):
        et.q1_metrics(np.array(classes), np.array(risk), 0.05, MATRIX, ALERT_CLASS, 10, 1)


def test_q2_metrics_by_hand_constants_decisions_and_pairs() -> None:
    classes = np.array([0, 0, 1, 4])
    proba = np.array([[0.9, 0.05, 0.02, 0.02, 0.01]] * 2 + [[0.1, 0.7, 0.1, 0.05, 0.05]]
                     + [[0.05, 0.05, 0.1, 0.1, 0.7]])  # fmt: skip
    prior = [0.9, 0.04, 0.03, 0.02, 0.01]
    out = et.q2_metrics(classes, proba, prior, prior, MATRIX, "always_4", n_boot=30, seed=1)
    assert out["constant_baselines"]["always_0"] == 200 + 500
    assert out["constant_baselines"]["always_4"] == 10 + 10 + 9 + 0
    assert out["frozen_best_constant"] == "always_4" and out["best_constant_on_test"] == "always_4"
    assert out["class_counts"] == [2, 0 + 1, 0, 0, 1][:1] + [1, 0, 0, 1]
    assert set(out["decisions"]) == {"uncorrected_expected_cost", "corrected_expected_cost",
                                     "uncorrected_argmax"}  # fmt: skip
    # identical priors: the correction changes nothing
    assert (
        out["decisions"]["corrected_expected_cost"] == out["decisions"]["uncorrected_expected_cost"]
    )
    assert out["paired_bootstrap_cost"]["corrected_vs_uncorrected"]["diff"] == 0
    assert "corrected_vs_always_4" in out["paired_bootstrap_cost"]
    assert np.array(out["confusion_corrected"]).sum() == 4


def test_nasa_score_and_rul_metrics_by_hand() -> None:
    assert et.nasa_score(np.array([87.0]), np.array([100.0])) == pytest.approx(
        np.exp(1) - 1
    )  # d=-13
    assert et.nasa_score(np.array([110.0]), np.array([100.0])) == pytest.approx(
        np.exp(1) - 1
    )  # d=10
    true_raw = np.array([100.0, 200.0])  # capped at 125: [100, 125]
    out = et.rul_metrics(
        true_raw, 125,
        {"lightgbm_rolling": np.array([100.0, 125.0]), "ridge_raw": np.array([90.0, 125.0]),
         "mean_predictor": np.array([80.0, 80.0])},
        "ridge_raw", n_boot=40, seed=1,
    )  # fmt: skip
    models = out["models"]
    assert models["lightgbm_rolling"]["rmse_capped"] == 0
    assert models["ridge_raw"]["rmse_capped"] == pytest.approx(np.sqrt(100 / 2))
    assert models["ridge_raw"]["rmse_raw"] == pytest.approx(
        np.sqrt((100 + 75**2) / 2)
    )  # vs RAW RUL
    diff = out["paired_rmse_difference_vs_reference"]["lightgbm_rolling"]["diff"]
    assert diff == pytest.approx(0 - np.sqrt(50)) and out["reference"] == "ridge_raw"
    assert "ridge_raw" not in out["paired_rmse_difference_vs_reference"]
    with pytest.raises(ValueError, match="one prediction per engine"):
        et.rul_metrics(true_raw, 125, {"a": np.array([1.0])}, "a", 10, 1)


def test_bootstrap_interval_is_reproducible_and_tolerates_impossible_resamples() -> None:
    y = np.array([0] * 9 + [1])
    s = np.linspace(0, 1, 10)
    a = et.bootstrap_interval(et._ap, [y, s], 100, 3)
    assert a == et.bootstrap_interval(et._ap, [y, s], 100, 3) and a[0] <= a[1]
    nothing = et.bootstrap_interval(et._ap, [np.zeros(5, int), np.ones(5)], 20, 1)
    assert np.isnan(nothing[0]) and np.isnan(nothing[1])


# --- the protections ---------------------------------------------------------------------
@pytest.fixture
def guarded(tmp_path: Path) -> dict[str, Any]:
    reports = tmp_path / "reports"
    reports.mkdir()
    manifest = tmp_path / "reports" / "frozen_manifest.json"
    manifest.write_text(json.dumps({"code_commit": "abc", "decisions": ["d"]}))
    config = {"paths": {"reports": str(reports)}, "freeze": {"manifest": str(manifest)}}
    return {"config": config, "lock": reports / et.LOCK_FILE}


def test_without_the_approval_flag_nothing_happens(guarded: dict[str, Any]) -> None:
    called = []
    with pytest.raises(et.EvaluationRefusedError, match="--approved-single-run"):
        et.guard(guarded["config"], False, lambda c: called.append(1) or [])
    assert not guarded["lock"].exists() and called == []  # not even the manifest was verified


def test_a_manifest_that_does_not_verify_refuses_and_creates_no_lock(
    guarded: dict[str, Any],
) -> None:
    with pytest.raises(et.EvaluationRefusedError, match="does not verify: test file changed"):
        et.guard(guarded["config"], True, lambda c: ["test file changed: scania_labels", "x"])
    assert not guarded["lock"].exists()


def test_an_existing_lock_refuses_before_anything_else(guarded: dict[str, Any]) -> None:
    guarded["lock"].write_text("{}")
    called = []
    with pytest.raises(et.EvaluationRefusedError, match="already been started"):
        et.guard(guarded["config"], True, lambda c: called.append(1) or [])
    assert called == []


def test_a_successful_guard_writes_the_lock_with_commit_and_manifest_hash(
    guarded: dict[str, Any],
) -> None:
    manifest = et.guard(guarded["config"], True, lambda c: [])
    lock = json.loads(guarded["lock"].read_text())
    assert manifest["code_commit"] == lock["code_commit"] == "abc"
    assert len(lock["manifest_sha256"]) == 64 and "no second run" in lock["status"]
    with pytest.raises(et.EvaluationRefusedError, match="already been started"):
        et.guard(guarded["config"], True, lambda c: [])  # the second call is refused


def test_the_preflight_runs_after_the_verification_and_before_the_lock(
    guarded: dict[str, Any],
) -> None:
    seen: dict[str, Any] = {}
    et.guard(
        guarded["config"], True, lambda c: [],
        preflight=lambda m: seen.update(lock=guarded["lock"].exists(), commit=m["code_commit"]),
    )  # fmt: skip
    assert seen == {"lock": False, "commit": "abc"}  # the manifest is given, no lock yet
    assert guarded["lock"].exists()


def test_a_failing_preflight_creates_no_lock(guarded: dict[str, Any]) -> None:
    def broken(manifest: Any) -> None:
        raise et.EvaluationRefusedError("frozen artifact check failed (Q1 model): boom")

    with pytest.raises(et.EvaluationRefusedError, match="frozen artifact check failed"):
        et.guard(guarded["config"], True, lambda c: [], preflight=broken)
    assert not guarded["lock"].exists()


def test_the_preflight_is_not_reached_when_the_manifest_does_not_verify(
    guarded: dict[str, Any],
) -> None:
    called = []
    with pytest.raises(et.EvaluationRefusedError, match="does not verify"):
        et.guard(guarded["config"], True, lambda c: ["changed"], preflight=called.append)
    assert called == [] and not guarded["lock"].exists()


def test_the_cli_refuses_without_the_flag_with_exit_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(et, "load_config", lambda: {"paths": {"reports": "/nonexistent"}})
    with pytest.raises(SystemExit) as info:
        et.main([])
    assert info.value.code == 2


# --- who may read the test split -----------------------------------------------------------
# Every module that can reach a test-split file, and why. Nothing else may.
DECLARED_TEST_SPLIT_READERS = {
    "freeze.py": "hashes the five test files (bytes only, nothing is parsed)",
    "evaluate_test.py": "the single run (and its rehearsal, which reads validation instead)",
    "data/convert.py": "make parquet: format conversion CSV -> Parquet of all nine SCANIA files",
    "data/schema.py": "make validate: data contract and split overlap (readouts, specifications)",
}
# a literal test path, or a loop over a SPLITS tuple (which contains "test")
TEST_PATH_PATTERN = re.compile(
    r"""\["test"\]|table_path\([^)]*["']test["']|read_scania\([^)]*["']test["']|\bSPLITS\b"""
)


def test_only_the_declared_modules_can_reach_the_test_split() -> None:
    """Static guard: literal test paths and loops over ``SPLITS`` exist only where declared.

    ``convert`` and ``schema`` are on the list on purpose: they read test files for format
    conversion and schema checks. No training, tuning, calibration or analysis module is.
    """
    found = {
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if TEST_PATH_PATTERN.search(p.read_text(encoding="utf-8"))
    }
    assert found == set(DECLARED_TEST_SPLIT_READERS)


def test_the_static_guard_sees_a_loop_over_the_splits() -> None:
    """The old pattern looked only for ``["test"]`` and was blind to ``scania[split]``."""
    assert TEST_PATH_PATTERN.search("for split in SPLITS:\n    files = scania[split]")
    assert TEST_PATH_PATTERN.search('read_table(table_path(config, "test", "readouts"))')
    assert TEST_PATH_PATTERN.search('scania = config["datasets"]["scania"]["test"]')
    assert not TEST_PATH_PATTERN.search('read_table(table_path(config, "train", "readouts"))')


def assert_no_test_split_read(spy: Any) -> None:
    """The dynamic guard: fails and names the files if a test-split file was opened."""
    reads = spy.test_split_reads()
    assert reads == [], f"test-split file(s) were read: {sorted({Path(p).name for p in reads})}"


@pytest.fixture
def training_config(
    make_mini_scania, make_mini_cmapss, tmp_path: Path  # type: ignore[no-untyped-def]
) -> dict[str, Any]:  # fmt: skip
    """Mini SCANIA + C-MAPSS with all three splits on disk, validation labels included.

    The fixture folder DOES contain ``test_*.parquet`` files, so a loader that walks over all
    splits would be caught; the C-MAPSS test and RUL paths do not exist.
    """
    config = make_mini_scania()
    cmapss = make_mini_cmapss()
    config["datasets"]["cmapss"] = cmapss["datasets"]["cmapss"]
    config["rul"] = {
        **cmapss["rul"], "rolling_windows": [3, 5], "ridge_alpha": 1.0,
        "lightgbm": {"n_estimators": 10, "num_leaves": 4, "min_child_samples": 5},
    }  # fmt: skip
    processed = Path(config["datasets"]["scania"]["processed_dir"])
    ids = sorted(pd.read_parquet(processed / "validation_specifications.parquet")["vehicle_id"])
    labels = tmp_path / "validation_labels.csv"
    pd.DataFrame({"vehicle_id": ids, "class_label": [0, 0, 2, 4]}).to_csv(labels, index=False)
    config["datasets"]["scania"]["validation"]["labels"] = str(labels)
    return config


def _train(config: dict[str, Any], tmp_path: Path) -> None:
    from failsense.models.train import run

    run(config, track=False)  # make train


def _tuning_and_calibration_data(config: dict[str, Any], tmp_path: Path) -> None:
    """Every data read of make tune / evaluate / calibrate / retrain / matched-check."""
    from failsense.models.tuning_data import build_tuning_data
    from failsense.models.validation_data import build_validation_frame

    build_tuning_data(config)  # training rows, folds, both validation-like evaluation sets
    build_validation_frame(config)  # the official validation (calibration, threshold)


def _q2_data(config: dict[str, Any], tmp_path: Path) -> None:
    """Every data read of make q2 / anomaly / explain."""
    from failsense.models.q2_classes import load_training_frame_with_classes
    from failsense.models.validation_data import build_validation_frame

    load_training_frame_with_classes(config, 5)
    build_validation_frame(config)


def _rul(config: dict[str, Any], tmp_path: Path) -> None:
    from failsense.rul import baseline, timeseries

    baseline.run(config, track=False)  # make rul-baseline
    timeseries.run(config, track=False)  # make rul-timeseries


def _rul_freeze(config: dict[str, Any], tmp_path: Path) -> None:
    freeze.freeze_rul(config, tmp_path / "frozen_rul")  # the RUL training of make freeze


@pytest.mark.leakage
@pytest.mark.parametrize(
    "entry_point",
    [_train, _tuning_and_calibration_data, _q2_data, _rul, _rul_freeze],
    ids=lambda f: f.__name__.strip("_"),
)
def test_training_entry_points_never_open_a_test_split_file(
    entry_point: Callable[..., None], training_config: dict[str, Any], read_spy: Any,
    tmp_path: Path,
) -> None:  # fmt: skip
    """Train, calibrate, Q2 and RUL on fixtures: no opened file is named ``test_*`` / ``RUL_FD*``.

    ``train`` and the RUL scripts run completely. For ``calibrate`` and ``q2`` the functions
    through which they read ALL their data are run (their full ``run()`` needs a registry model).
    """
    read_spy.paths.clear()
    entry_point(training_config, tmp_path)
    assert any(Path(p).name.startswith("train_") for p in read_spy.paths)  # the spy was active
    assert_no_test_split_read(read_spy)


@pytest.mark.leakage
def test_the_guard_fails_on_a_fake_reader_of_the_test_split(
    training_config: dict[str, Any], read_spy: Any, tmp_path: Path
) -> None:
    """Proof that the guard above can fail: three careless 'training' loaders are caught."""
    processed = Path(training_config["datasets"]["scania"]["processed_dir"])

    def loop_over_all_splits() -> None:  # the mistake a regular expression cannot see
        for split in ("train", "validation", "test"):
            pd.read_parquet(processed / f"{split}_specifications.parquet")

    def read_a_csv_copy() -> None:
        copy = tmp_path / "test_labels.csv"
        copy.write_text("vehicle_id,class_label\n200,0\n")
        pd.read_csv(copy)

    def open_the_rul_file() -> None:
        rul = tmp_path / "RUL_FD001.txt"
        rul.write_text("112 \n")
        with rul.open() as handle:
            handle.read()

    for fake_reader, name in (
        (loop_over_all_splits, "test_specifications.parquet"),
        (read_a_csv_copy, "test_labels.csv"),
        (open_the_rul_file, "RUL_FD001.txt"),
    ):
        read_spy.paths.clear()
        fake_reader()
        with pytest.raises(AssertionError, match=re.escape(name)):
            assert_no_test_split_read(read_spy)


@pytest.mark.leakage
def test_the_declared_readers_convert_and_validate_are_caught_by_the_same_guard(
    training_config: dict[str, Any], read_spy: Any
) -> None:
    """``make validate`` does read test files (schema check): that is why it is declared."""
    from failsense.data.schema import validate_scania

    read_spy.paths.clear()
    validate_scania(training_config, root=None)
    caught = {Path(p).name for p in read_spy.test_split_reads()}
    assert caught == {"test_operational_readouts.parquet", "test_specifications.parquet"}


def test_the_spy_judges_a_file_by_its_name_not_by_its_folder() -> None:
    from conftest import is_test_split_file

    assert is_test_split_file("/data/raw/scania/test_labels.csv")
    assert is_test_split_file("data/processed/scania/test_operational_readouts.parquet")
    assert is_test_split_file("/data/raw/cmapss/test_FD001.txt")
    assert is_test_split_file("/data/raw/cmapss/RUL_FD001.txt")
    assert not is_test_split_file("/tmp/pytest/test_run_0/train_operational_readouts.parquet")
    assert not is_test_split_file("/tmp/pytest/test_run_0/validation_labels.csv")
    assert not is_test_split_file("tests/test_evaluate_test.py")  # code, not data


# --- a full run on fixtures --------------------------------------------------------------
@pytest.fixture
def stack(
    make_mini_scania,
    make_mini_cmapss,
    make_cmapss_frame,
    write_cmapss_file,  # type: ignore[no-untyped-def]
    tiny_pipeline: tuple[Any, Any, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Frozen artifacts (Q1, Q2, RUL) and 'test' files, all built from fixtures in a temp folder."""
    from conftest import tiny_explain_frame

    config = make_mini_scania()
    cmapss = make_mini_cmapss()
    real = load_config()
    config["datasets"]["cmapss"] = cmapss["datasets"]["cmapss"]
    lgbm = {"n_estimators": 10, "num_leaves": 4, "min_child_samples": 5}
    config["rul"] = {**cmapss["rul"], "rolling_windows": [3, 5], "ridge_alpha": 1.0,
                     "lightgbm": lgbm}  # fmt: skip
    config["cost"], config["q2"], config["seed"] = real["cost"], real["q2"], 42
    reports = Path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    config["freeze"] = {"dir": str(frozen), "manifest": str(reports / "frozen_manifest.json")}

    # --- fixture "test" files: SCANIA csv (trucks 200..203) and a truncated C-MAPSS test set
    processed = Path(config["datasets"]["scania"]["processed_dir"])
    test_dir = tmp_path / "testfiles"
    test_dir.mkdir()
    paths = {}
    for name in ("operational_readouts", "specifications"):
        frame = pd.read_parquet(processed / f"test_{name}.parquet")
        paths[name] = test_dir / f"test_{name}.csv"
        frame.to_csv(paths[name], index=False)
    ids = sorted(pd.read_parquet(processed / "test_specifications.parquet")["vehicle_id"])
    labels = pd.DataFrame({"vehicle_id": ids, "class_label": [0, 1, 3, 4]})
    paths["labels"] = test_dir / "test_labels.csv"
    labels.to_csv(paths["labels"], index=False)
    config["datasets"]["scania"]["test"] = {
        "readouts": str(paths["operational_readouts"]),
        "specifications": str(paths["specifications"]), "labels": str(paths["labels"]),
    }  # fmt: skip
    # --- fixture VALIDATION csv files (trucks 100..103) for the rehearsal
    val_dir = tmp_path / "valfiles"
    val_dir.mkdir()
    val_paths = {}
    for name in ("operational_readouts", "specifications"):
        val_paths[name] = val_dir / f"validation_{name}.csv"
        pd.read_parquet(processed / f"validation_{name}.parquet").to_csv(
            val_paths[name], index=False
        )
    val_ids = sorted(pd.read_parquet(processed / "validation_specifications.parquet")["vehicle_id"])
    val_paths["labels"] = val_dir / "validation_labels.csv"
    pd.DataFrame({"vehicle_id": val_ids, "class_label": [0, 0, 2, 4]}).to_csv(
        val_paths["labels"], index=False
    )
    config["datasets"]["scania"]["validation"] = {
        "readouts": str(val_paths["operational_readouts"]),
        "specifications": str(val_paths["specifications"]), "labels": str(val_paths["labels"]),
    }  # fmt: skip
    full = make_cmapss_frame([30, 35, 40, 45, 50], seed=5)
    cut = pd.concat([g.iloc[: len(g) - 7] for _, g in full.groupby("engine")])
    write_cmapss_file(cut, test_dir / "test_FD001.txt")
    (test_dir / "RUL_FD001.txt").write_text("\n".join(["12", "30", "7", "55", "20"]) + "\n")
    config["datasets"]["cmapss"]["test"] = str(test_dir / "test_FD001.txt")
    config["datasets"]["cmapss"]["rul"] = str(test_dir / "RUL_FD001.txt")

    # --- frozen Q1: the tiny pipeline as an MLflow model folder + registered input columns
    pipeline, frame, _ = tiny_pipeline
    mlflow.sklearn.save_model(pipeline, str(frozen / "q1_model"), skops_trusted_types=TRUSTED_TYPES)
    (reports / "registry_q1.json").write_text(json.dumps({"input_columns": list(frame.columns)}))
    # --- frozen Q2: a 5-class pipeline on the same columns
    x, _ = tiny_explain_frame()
    classes = np.arange(len(x)) % 5
    q2 = make_multiclass_pipeline({**STEP6_PARAMS["xgboost"], "max_depth": 2}, 5, 0, ["Spec_0"])
    q2.fit(x, classes)
    types = freeze.save_model(q2, frozen / "q2_model.skops")
    prior = (np.bincount(classes, minlength=5) / len(classes)).tolist()
    counts = np.array(real["q2"]["target_class_counts"], dtype=float)
    (frozen / "q2_meta.json").write_text(json.dumps({
        "input_columns": list(x.columns), "prior_train": prior,
        "prior_target": (counts / counts.sum()).tolist(),
        "cost_matrix": MATRIX.tolist(), "best_constant_from_validation": "always_4",
        "untrusted_types": types,
    }))  # fmt: skip
    # --- frozen RUL: the real freeze code on the mini training engines
    freeze.freeze_rul(config, frozen)
    manifest = {
        "code_commit": "fixture", "decisions": ["fixture"],
        "config": {"features": {"top_fraction": 0.2}, "cost": real["cost"], "seed": 42},
        "q1": {"folder": "q1_model", "threshold": 0.4,
               "calibrator": {"kind": "sigmoid", "slope": 1.0, "intercept": 0.0,
                              "logit_clip_eps": 1e-6}},  # fmt: skip
    }  # fmt: skip
    Path(config["freeze"]["manifest"]).write_text(json.dumps(manifest))
    return {"config": config, "reports": reports, "test_dir": test_dir, "val_dir": val_dir,
            "frozen": frozen}  # fmt: skip


def test_a_full_run_on_fixtures_writes_the_result_and_closes_the_lock(
    stack: dict[str, Any],
) -> None:
    result = et.run(stack["config"], True, n_bootstrap=20, verify=lambda c: [])
    assert set(result) >= {"q1", "q2", "rul", "code_commit", "decisions", "n_bootstrap"}
    q1 = result["q1"]
    assert q1["n_trucks"] == 4 and q1["n_positives"] == 3 and q1["prevalence"] == 0.75
    assert {"primary", "literal"} == set(q1["cost"]) and q1["pr_auc"]["prevalence"] == 0.75
    assert q1["cost"]["primary"]["never_alert"] == 200 + 400 + 500  # the three positive trucks
    assert result["q2"]["n_trucks"] == 4 and result["q2"]["class_counts"] == [1, 1, 0, 1, 1]
    rul = result["rul"]
    assert rul["n_engines"] == 5 and set(rul["models"]) == {"lightgbm_rolling", "ridge_raw",
                                                            "mean_predictor"}  # fmt: skip
    assert rul["models"]["mean_predictor"]["rmse_capped"] > 0 and rul["reference"] == "ridge_raw"
    saved = json.loads((stack["reports"] / et.RESULT_FILE).read_text())
    assert saved["q1"]["confusion"] == q1["confusion"]
    lock = json.loads((stack["reports"] / et.LOCK_FILE).read_text())
    assert lock["status"] == "finished"
    with pytest.raises(et.EvaluationRefusedError, match="already been started"):
        et.run(stack["config"], True, n_bootstrap=5, verify=lambda c: [])  # no second run


def test_the_lock_exists_before_the_first_test_file_is_read(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def first_read(config: Any) -> Any:
        seen["lock_existed"] = (stack["reports"] / et.LOCK_FILE).exists()
        raise RuntimeError("simulated crash while reading")

    monkeypatch.setattr(et, "read_scania_test", first_read)
    with pytest.raises(RuntimeError, match="simulated crash"):
        et.run(stack["config"], True, 10, lambda c: [])
    assert seen["lock_existed"] is True
    lock = json.loads((stack["reports"] / et.LOCK_FILE).read_text())
    assert lock["status"].startswith("failed: RuntimeError")  # the used-up run is documented
    assert not (stack["reports"] / et.RESULT_FILE).exists()
    with pytest.raises(et.EvaluationRefusedError):  # a crash does not buy a quick retry
        et.run(stack["config"], True, 10, lambda c: [])


def test_nothing_is_read_when_a_protection_refuses(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test file was read although the guard refused")

    for name in ("read_scania_test", "read_cmapss_test"):
        monkeypatch.setattr(et, name, boom)
    with pytest.raises(et.EvaluationRefusedError):
        et.run(stack["config"], False, 10, lambda c: [])
    with pytest.raises(et.EvaluationRefusedError):
        et.run(stack["config"], True, 10, lambda c: ["tampered"])
    assert not (stack["reports"] / et.LOCK_FILE).exists()


def test_the_scania_test_files_are_read_as_raw_csv_never_as_parquet(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_parquet(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parquet copies must not be used for the test evaluation")

    monkeypatch.setattr(pd, "read_parquet", no_parquet)
    result = et.run(stack["config"], True, 10, lambda c: [])
    assert result["q1"]["n_trucks"] == 4


def test_the_rul_files_must_match_in_number(stack: dict[str, Any]) -> None:
    (stack["test_dir"] / "RUL_FD001.txt").write_text("12\n30\n")  # 2 values for 5 engines
    with pytest.raises(ValueError, match="differ in number"):
        et.read_cmapss_test(stack["config"])


def test_the_rul_evaluation_uses_the_last_cycle_of_each_engine(stack: dict[str, Any]) -> None:
    engines, rul = et.read_cmapss_test(stack["config"])
    assert len(rul) == 5 and engines["engine"].nunique() == 5
    manifest = json.loads(Path(stack["config"]["freeze"]["manifest"]).read_text())
    frozen = et.load_frozen(stack["config"], manifest)
    result = et.evaluate_rul(manifest, frozen, engines, rul, 10)
    assert result["n_engines"] == 5 and result["cap"] == 25


# --- preflight: the frozen artifacts are loaded and checked before the lock ----------------
def _manifest(stack: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(stack["config"]["freeze"]["manifest"]).read_text())


def _edit_json(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_the_preflight_loads_every_frozen_artifact(stack: dict[str, Any]) -> None:
    frozen = et.load_frozen(stack["config"], _manifest(stack))
    assert frozen.q1_threshold == 0.4 and frozen.q1_calibrator["kind"] == "sigmoid"
    assert frozen.q1_columns == list(frozen.q1_model.feature_names_in_)
    assert frozen.q2_meta["best_constant_from_validation"] == "always_4"
    assert set(frozen.rul_meta["columns_raw"]) < set(frozen.rul_meta["columns_rolling"])
    assert hasattr(frozen.rul_lightgbm, "predict") and hasattr(frozen.rul_ridge, "predict")


def _break_q1_folder(stack: dict[str, Any]) -> None:
    shutil.rmtree(stack["frozen"] / "q1_model")


def _break_q2_file(stack: dict[str, Any]) -> None:
    (stack["frozen"] / "q2_model.skops").write_bytes(b"this is not a model file")


def _break_rul_file(stack: dict[str, Any]) -> None:
    (stack["frozen"] / "rul_lightgbm.skops").unlink()


def _break_calibrator(stack: dict[str, Any]) -> None:
    _edit_json(
        Path(stack["config"]["freeze"]["manifest"]),
        lambda m: m["q1"]["calibrator"].update(slope=-1.0),
    )


def _break_threshold(stack: dict[str, Any]) -> None:
    _edit_json(Path(stack["config"]["freeze"]["manifest"]), lambda m: m["q1"].update(threshold=1.5))


def _break_columns(stack: dict[str, Any]) -> None:
    _edit_json(
        stack["reports"] / "registry_q1.json",
        lambda r: r.update(input_columns=r["input_columns"][::-1]),
    )


def _break_q2_meta(stack: dict[str, Any]) -> None:
    _edit_json(stack["frozen"] / "q2_meta.json", lambda m: m.pop("prior_target"))


def _break_rul_meta(stack: dict[str, Any]) -> None:
    _edit_json(
        stack["frozen"] / "rul_meta.json",
        lambda m: m.update(columns_raw=m["columns_raw"][:-1]),
    )


BREAKERS = [
    _break_q1_folder, _break_q2_file, _break_rul_file, _break_calibrator, _break_threshold,
    _break_columns, _break_q2_meta, _break_rul_meta,
]  # fmt: skip


def _forbid_every_data_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a data file was read although a frozen artifact is broken")

    for name in ("read_scania", "read_scania_test", "read_cmapss_test", "rehearsal_cmapss"):
        monkeypatch.setattr(et, name, boom)


@pytest.mark.parametrize("breaker", BREAKERS, ids=lambda f: f.__name__)
def test_a_broken_artifact_stops_the_single_run_before_the_lock_and_before_any_read(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch, breaker: Callable[..., None]
) -> None:
    breaker(stack)
    _forbid_every_data_read(monkeypatch)
    with pytest.raises(et.EvaluationRefusedError, match="frozen artifact check failed"):
        et.run(stack["config"], True, 10, lambda c: [])
    assert not (stack["reports"] / et.LOCK_FILE).exists()  # the run is NOT used up
    assert not (stack["reports"] / et.RESULT_FILE).exists()


@pytest.mark.parametrize("breaker", BREAKERS, ids=lambda f: f.__name__)
def test_a_broken_artifact_stops_the_rehearsal_before_any_read(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    breaker: Callable[..., None],
) -> None:  # fmt: skip
    breaker(stack)
    _forbid_every_data_read(monkeypatch)
    out = tmp_path / "rehearsal.json"
    with pytest.raises(et.EvaluationRefusedError, match="frozen artifact check failed"):
        et.rehearse(stack["config"], 10, out)
    assert not out.exists()


def test_the_error_names_the_broken_artifact(stack: dict[str, Any]) -> None:
    _break_q2_file(stack)
    with pytest.raises(et.EvaluationRefusedError, match=r"\(Q2 model\)"):
        et.load_frozen(stack["config"], _manifest(stack))


def test_after_a_repaired_artifact_the_single_run_is_still_available(
    stack: dict[str, Any],
) -> None:
    backup = (stack["frozen"] / "q2_model.skops").read_bytes()
    _break_q2_file(stack)
    with pytest.raises(et.EvaluationRefusedError):
        et.run(stack["config"], True, 10, lambda c: [])
    (stack["frozen"] / "q2_model.skops").write_bytes(backup)
    result = et.run(stack["config"], True, 10, lambda c: [])  # no lock was left behind
    assert result["q1"]["n_trucks"] == 4


# --- the rehearsal on the validation split ---------------------------------------------------
def test_the_rehearsal_reproduces_hand_computed_metrics_on_the_validation_fixture(
    stack: dict[str, Any], tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out" / "rehearsal.json"
    result = et.rehearse(stack["config"], n_bootstrap=20, output=out)

    # the expected numbers, computed here without evaluate_test
    pipeline, frame, _ = tiny_pipeline
    val = stack["val_dir"]
    built = build_last_readout_frame(
        pd.read_csv(val / "validation_operational_readouts.csv"),
        pd.read_csv(val / "validation_specifications.csv"),
        pd.read_csv(val / "validation_labels.csv"),
        counter_columns(), histogram_groups(), 0.2,
    )  # fmt: skip
    risk = pipeline.predict_proba(built[list(frame.columns)])[:, 1]  # identity calibrator
    classes = built["proximity_class"].to_numpy()
    alert = risk >= 0.4
    positive = classes >= 1
    miss_cost = {2: 300, 4: 500}
    cost = 7 * int((alert & ~positive).sum()) + sum(
        miss_cost[c] for c, a in zip(classes, alert, strict=True) if c >= 1 and not a
    )

    q1 = result["q1"]
    assert (q1["n_trucks"], q1["n_positives"]) == (4, 2)
    assert q1["threshold"] == 0.4
    assert q1["confusion"] == {
        "tp": int((alert & positive).sum()), "fp": int((alert & ~positive).sum()),
        "fn": int((~alert & positive).sum()), "tn": int((~alert & ~positive).sum()),
    }  # fmt: skip
    assert q1["cost"]["primary"]["model"] == cost
    assert q1["cost"]["primary"]["never_alert"] == 300 + 500
    assert q1["cost"]["primary"]["always_alert"] == 2 * 7
    assert q1["pr_auc"]["value"] == pytest.approx(average_precision_score(positive, risk))
    assert result["q2"]["n_trucks"] == 4 and result["q2"]["class_counts"] == [2, 0, 1, 0, 1]
    assert result["mode"] == et.REHEARSAL_BANNER == "REHEARSAL \u2013 validation, not test"
    assert result["split"] == "validation"
    assert json.loads(out.read_text())["q1"]["confusion"] == q1["confusion"]


def test_the_rehearsal_takes_no_lock_writes_no_result_file_and_needs_no_approval(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_verify(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the rehearsal must not verify the manifest (it hashes test files)")

    monkeypatch.setattr(et, "verify_manifest", no_verify)
    before = sorted(p.name for p in stack["reports"].iterdir())
    et.rehearse(stack["config"], 10, tmp_path / "rehearsal.json")
    assert sorted(p.name for p in stack["reports"].iterdir()) == before
    assert not (stack["reports"] / et.LOCK_FILE).exists()
    assert not (stack["reports"] / et.RESULT_FILE).exists()
    # the single run is still available afterwards
    assert et.run(stack["config"], True, 10, lambda c: [])["q1"]["n_trucks"] == 4


def test_the_rehearsal_opens_no_test_split_file(
    stack: dict[str, Any], read_spy: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the rehearsal called a reader of the test split")

    for name in ("read_scania_test", "read_cmapss_test"):
        monkeypatch.setattr(et, name, boom)
    read_spy.paths.clear()
    et.rehearse(stack["config"], 10, tmp_path / "rehearsal.json")
    names = {Path(p).name for p in read_spy.paths}
    # the spy really saw the reads of the rehearsal ...
    assert {"validation_operational_readouts.csv", "validation_labels.csv", "train_FD001.txt",
            "q2_meta.json"} <= names  # fmt: skip
    # ... and none of them is a test-split file
    assert read_spy.test_split_reads() == []
    assert not any("test" in name for name in names)


def test_the_spy_does_catch_the_readers_of_the_single_run(
    stack: dict[str, Any], read_spy: Any
) -> None:
    """The guard above can fail: the real test readers are seen by the same spy."""
    read_spy.paths.clear()
    et.read_scania_test(stack["config"])
    et.read_cmapss_test(stack["config"])
    caught = {Path(p).name for p in read_spy.test_split_reads()}
    assert caught == {
        "test_operational_readouts.csv", "test_specifications.csv", "test_labels.csv",
        "test_FD001.txt", "RUL_FD001.txt",
    }  # fmt: skip


def test_the_rehearsal_refuses_an_output_inside_the_repository(stack: dict[str, Any]) -> None:
    inside = REPO / "reports" / "rehearsal_must_not_exist.json"
    with pytest.raises(et.EvaluationRefusedError, match="nothing inside the repository"):
        et.rehearse(stack["config"], 5, inside)
    assert not inside.exists()
    with pytest.raises(et.EvaluationRefusedError, match="nothing inside the repository"):
        et.rehearsal_output_path(REPO)


def _git_status() -> str:
    done = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True,
        check=False,
    )  # fmt: skip
    return done.stdout if done.returncode == 0 else "git not available"


def test_the_default_rehearsal_output_is_one_new_file_in_the_temp_folder_not_in_the_repo(
    stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temp = tmp_path / "system_temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    status_before = _git_status()
    result = et.rehearse(stack["config"], 5)
    files = list(temp.iterdir())
    assert len(files) == 1 and files[0].name.startswith(et.REHEARSAL_FILE_PREFIX)
    assert json.loads(files[0].read_text())["mode"] == result["mode"]
    assert _git_status() == status_before  # nothing appeared in the repository


def test_the_rehearsal_engines_are_cut_training_engines_with_a_known_remaining_life(
    stack: dict[str, Any], make_cmapss_frame: Any
) -> None:
    engines, true_rul = et.rehearsal_cmapss(stack["config"])
    lengths = [30, 35, 40, 45, 50, 32, 38, 42, 48, 36]  # the default mini training engines
    cuts = [min(10 + (e % 10) * 10, n // 2) for e, n in enumerate(lengths, start=1)]
    assert true_rul.tolist() == [float(c) for c in cuts]
    kept = engines.groupby("engine")["cycle"].max().tolist()
    assert kept == [n - c for n, c in zip(lengths, cuts, strict=True)]
    assert engines.groupby("engine")["cycle"].min().eq(1).all()  # only the END is cut


def test_the_rehearsal_reports_the_rul_part_as_an_in_sample_plumbing_check(
    stack: dict[str, Any], tmp_path: Path
) -> None:
    result = et.rehearse(stack["config"], 10, tmp_path / "rehearsal.json")
    assert result["rul"]["n_engines"] == 10
    assert set(result["rul"]["models"]) == {"lightgbm_rolling", "ridge_raw", "mean_predictor"}
    assert any("in-sample" in note for note in result["notes"])
    assert result["frozen_files_match_manifest"] is None  # the fixture manifest has no hashes


def test_the_rehearsal_needs_a_manifest(stack: dict[str, Any], tmp_path: Path) -> None:
    Path(stack["config"]["freeze"]["manifest"]).unlink()
    with pytest.raises(et.EvaluationRefusedError, match="make freeze"):
        et.rehearse(stack["config"], 5, tmp_path / "rehearsal.json")


def test_read_scania_knows_only_the_two_labelled_splits(stack: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="'validation' or 'test'"):
        et.read_scania(stack["config"], "train")


def test_the_cli_rehearsal_flag_starts_the_rehearsal_and_never_the_single_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(et, "load_config", lambda: {"fixture": True})
    monkeypatch.setattr(et, "rehearse", lambda c, n, o: calls.update(args=(c, n, o)) or {})
    monkeypatch.setattr(et, "run", lambda *a, **k: pytest.fail("the single run must not start"))
    et.main(["--rehearse-on-validation", "--n-bootstrap", "7", "--output", "/tmp/r.json"])
    assert calls["args"] == ({"fixture": True}, 7, Path("/tmp/r.json"))


@pytest.mark.parametrize(
    "argv",
    [["--rehearse-on-validation", "--approved-single-run"], ["--output", "/tmp/r.json"]],
)
def test_the_cli_rejects_contradicting_flags(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(et, "run", lambda *a, **k: pytest.fail("the single run must not start"))
    monkeypatch.setattr(et, "rehearse", lambda *a, **k: pytest.fail("no rehearsal either"))
    with pytest.raises(SystemExit) as info:
        et.main(argv)
    assert info.value.code == 2


def test_make_rehearse_exists_is_in_make_help_and_never_approves_the_single_run() -> None:
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    target = next(line for line in makefile.splitlines() if line.startswith("rehearse:"))
    assert "## " in target and "VALIDATION" in target  # shown by `make help`
    assert "scripts/evaluate_test_once.py --rehearse-on-validation" in makefile
    assert " rehearse " in makefile.splitlines()[0]  # .PHONY
    assert "--approved-single-run" not in makefile  # no make target can start the single run
