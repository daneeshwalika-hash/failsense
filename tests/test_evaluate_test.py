"""The single test evaluation: protections, metrics by hand and a full run on FIXTURES only.

No real test file is read here: the fixtures write their own tiny "test" files into a temp folder.
"""

from __future__ import annotations

import json
import re
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
from failsense.models.boosters import STEP6_PARAMS
from failsense.models.cost import alert_class_from_config, cost_matrix_from_config
from failsense.models.q2_classes import make_multiclass_pipeline
from failsense.models.register import TRUSTED_TYPES

MATRIX = cost_matrix_from_config(load_config())
ALERT_CLASS = alert_class_from_config(load_config())
SRC = Path(__file__).resolve().parents[1] / "src" / "failsense"


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


def test_the_cli_refuses_without_the_flag_with_exit_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(et, "load_config", lambda: {"paths": {"reports": "/nonexistent"}})
    with pytest.raises(SystemExit) as info:
        et.main([])
    assert info.value.code == 2


def test_no_other_module_reads_the_test_split() -> None:
    """Only freeze (hashes) and evaluate_test (the single run) touch the config's test paths."""
    allowed = {"freeze.py", "evaluate_test.py"}
    pattern = re.compile(r"""\["test"\]|table_path\([^)]*["']test["']""")
    offenders = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if p.name not in allowed and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


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
    return {"config": config, "reports": reports, "test_dir": test_dir}


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
    result = et.evaluate_rul(stack["config"], manifest, 10)
    assert result["n_engines"] == 5 and result["cap"] == 25
