"""Freeze: hashing, model files, the manifest and its verification, the training-only freezes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from failsense import freeze as fz


# --- hashing and model files --------------------------------------------------------------------
def test_sha256_is_the_known_digest_and_hash_tree_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"\x00\xff garbage that is not csv")
    assert fz.sha256_file(tmp_path / "b.txt") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    tree = fz.hash_tree(tmp_path)
    assert list(tree) == ["b.txt", "sub/a.bin"]  # relative POSIX paths, sorted


def test_a_skops_model_round_trips_and_lists_its_untrusted_types(
    tiny_pipeline: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    pipeline, frame, _ = tiny_pipeline
    trusted = fz.save_model(pipeline, tmp_path / "m" / "model.skops")
    assert trusted and all(isinstance(t, str) for t in trusted)
    loaded = fz.load_model(tmp_path / "m" / "model.skops", trusted)
    np.testing.assert_array_equal(loaded.predict_proba(frame), pipeline.predict_proba(frame))
    with pytest.raises(Exception, match="(?i)untrusted|trust"):
        fz.load_model(tmp_path / "m" / "model.skops", [])  # nothing trusted -> refuses


def test_canonical_json_and_config_hash_ignore_key_order() -> None:
    config = {"seed": 1, "labels": {"a": 1, "b": 2}, "features": 3, "sampling": 4, "cv": 5,
              "cost": 6, "q2": 7, "rul": 8, "other": "ignored"}  # fmt: skip
    shuffled = {k: config[k] for k in reversed(list(config))}
    shuffled["labels"] = {"b": 2, "a": 1}
    assert fz.config_hash(config) == fz.config_hash(shuffled)
    changed = {**config, "q2": 99}
    assert fz.config_hash(changed) != fz.config_hash(config)
    assert fz.config_hash({**config, "other": "x"}) == fz.config_hash(
        config
    )  # not a frozen section


# --- manifest and verification in a throw-away git repository -------------------------------------
def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "repo"
    for rel, text in {
        "src/failsense/x.py": "x = 1\n", "configs/config.yaml": "seed: 42\n",
        "pyproject.toml": "[project]\n", "uv.lock": "lock\n", "README.md": "readme\n",
        "reports/registry_q1.json": "{}", "reports/calibration_q1_summary.json": "{}",
    }.items():  # fmt: skip
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for name in ("test_a.csv", "test_b.csv", "test_labels.csv", "test_FD001.txt", "RUL_FD001.txt"):
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / name).write_bytes(b"not parsed " + name.encode())
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("artifacts/\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "freeze")
    out = root / "artifacts" / "frozen"
    out.mkdir(parents=True)
    (out / "q2_model.skops").write_bytes(b"model")
    config = {
        "seed": 42, "labels": {}, "features": {}, "sampling": {}, "cv": {}, "cost": {}, "q2": {},
        "rul": {}, "paths": {"reports": str(root / "reports")},
        "freeze": {"dir": str(out), "manifest": str(root / "reports" / "frozen_manifest.json")},
        "datasets": {
            "scania": {"test": {"readouts": str(root / "data/test_a.csv"),
                                "specifications": str(root / "data/test_b.csv"),
                                "labels": str(root / "data/test_labels.csv")}},
            "cmapss": {"test": str(root / "data/test_FD001.txt"),
                       "rul": str(root / "data/RUL_FD001.txt")},
        },
    }  # fmt: skip
    manifest = fz.assemble_manifest(root, config, out, {"v": 1}, {"v": 2}, {"v": 3})
    Path(config["freeze"]["manifest"]).write_text(json.dumps(manifest))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "manifest")
    return {"root": root, "config": config, "out": out, "manifest": manifest}


def test_a_fresh_freeze_verifies(repo: dict[str, Any]) -> None:
    assert fz.verify_manifest(repo["config"], repo["root"]) == []
    manifest = repo["manifest"]
    expected = {
        "scania_readouts",
        "scania_specifications",
        "scania_labels",
        "cmapss_test",
        "cmapss_rul",
    }
    assert set(manifest["test_files"]) == expected
    assert len(manifest["code_commit"]) == 40 and manifest["frozen_files"]["q2_model.skops"]


def test_the_test_files_are_hashed_not_parsed(repo: dict[str, Any]) -> None:
    entry = repo["manifest"]["test_files"]["scania_labels"]
    assert entry["sha256"] == fz.sha256_file(repo["root"] / "data" / "test_labels.csv")
    assert (
        entry["path"] == "data/test_labels.csv"
    )  # the file content was binary garbage: no parser ran


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: (r["out"] / "q2_model.skops").write_bytes(b"other"), "frozen file changed"),
        (lambda r: (r["out"] / "q2_model.skops").unlink(), "frozen file missing"),
        (lambda r: (r["out"] / "extra.skops").write_bytes(b"x"), "unexpected frozen file"),
        (lambda r: (r["root"] / "reports/registry_q1.json").write_text('{"v": 2}'),
         "source file changed"),
        (lambda r: (r["root"] / "data/test_labels.csv").write_bytes(b"changed"),
         "test file changed"),
        (lambda r: (r["root"] / "uv.lock").write_text("other lock"), "uv.lock changed"),
    ],
)  # fmt: skip
def test_verification_finds_every_kind_of_change(
    repo: dict[str, Any], mutate: Any, expected: str
) -> None:
    mutate(repo)
    problems = fz.verify_manifest(repo["config"], repo["root"])
    assert any(expected in p for p in problems), problems


def test_a_changed_config_section_is_found(repo: dict[str, Any]) -> None:
    changed = {**repo["config"], "q2": {"target_class_counts": [1, 2, 3, 4, 5]}}
    assert any("config sections changed" in p for p in fz.verify_manifest(changed, repo["root"]))
    ignored = {**repo["config"], "dashboard": {"anything": 1}}  # not a frozen section
    assert fz.verify_manifest(ignored, repo["root"]) == []


def test_an_untracked_or_modified_file_makes_the_tree_dirty(repo: dict[str, Any]) -> None:
    (repo["root"] / "notes.txt").write_text("x")
    assert any("not clean" in p for p in fz.verify_manifest(repo["config"], repo["root"]))


def test_a_committed_code_change_after_the_freeze_is_found_but_docs_may_change(
    repo: dict[str, Any],
) -> None:
    root = repo["root"]
    (root / "README.md").write_text("docs changed\n")
    git(root, "commit", "-qam", "docs")
    assert fz.verify_manifest(repo["config"], root) == []  # documentation is allowed to change
    (root / "src/failsense/x.py").write_text("x = 2\n")
    git(root, "commit", "-qam", "code")
    problems = fz.verify_manifest(repo["config"], root)
    assert any("code changed since the freeze: src/failsense/x.py" in p for p in problems)


def test_a_missing_manifest_names_the_make_target(repo: dict[str, Any]) -> None:
    Path(repo["config"]["freeze"]["manifest"]).unlink()
    assert "make freeze" in fz.verify_manifest(repo["config"], repo["root"])[0]


def test_the_five_test_files_come_from_the_config(repo: dict[str, Any]) -> None:
    paths = fz.test_file_paths(repo["config"])
    assert len(paths) == 5 and all(p.name.startswith(("test_", "RUL_")) for p in paths.values())


def test_freeze_all_refuses_a_dirty_tree_and_an_existing_manifest(
    repo: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fz, "find_repo_root", lambda: repo["root"])
    with pytest.raises(fz.FreezeError, match="exists; use --replace"):
        fz.freeze_all(repo["config"])
    (repo["root"] / "dirty.txt").write_text("x")
    with pytest.raises(fz.FreezeError, match="not clean"):
        fz.freeze_all(repo["config"], replace=True)


# --- the training-only freezes --------------------------------------------------------------
def test_rul_freeze_trains_on_training_engines_only_and_saves_three_baselines(
    make_mini_cmapss,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    config = make_mini_cmapss()  # the test and RUL paths of this fixture do not exist
    config["rul"].update({"rolling_windows": [3, 5], "ridge_alpha": 1.0,
                          "lightgbm": {"n_estimators": 10, "num_leaves": 4,
                                       "min_child_samples": 5}})  # fmt: skip
    config["seed"] = 42
    out = tmp_path / "frozen"
    summary = fz.freeze_rul(config, out)
    meta = json.loads((out / "rul_meta.json").read_text())
    assert summary["n_train_engines"] == 10 and summary["cap"] == 25
    assert meta["rolling_windows"] == [3, 5] and meta["mean_rul_training"] > 0
    assert "sensor_1" not in meta["informative_sensors"]  # constant in the fixture
    for name in ("rul_lightgbm.skops", "rul_ridge.skops"):
        model = fz.load_model(out / name, meta["untrusted_types"][name])
        columns = meta["columns_rolling"] if "lightgbm" in name else meta["columns_raw"]
        assert model.n_features_in_ == len(columns)


def test_q2_freeze_saves_a_model_and_the_priors_from_training_rows(
    make_mini_scania,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from failsense.config import load_config
    from failsense.models.baseline_cv import load_training_frame
    from failsense.models.boosters import STEP6_PARAMS
    from failsense.models.pipeline import split_feature_columns

    config = make_mini_scania()
    real = load_config()
    config["cost"], config["q2"] = real["cost"], real["q2"]
    frame = load_training_frame(config, 5)
    cols = split_feature_columns(frame)
    reports = Path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "registry_q1.json").write_text(json.dumps({
        "params": STEP6_PARAMS["xgboost"], "n_rounds": 5,
        "input_columns": cols.raw + cols.engineered + cols.categorical,
    }))  # fmt: skip
    from failsense.models import q2_classes

    # the 40-truck fixture has no truck in every class: give the rows all five classes (labels only)
    classes = np.arange(len(frame)) % 5
    monkeypatch.setattr(
        q2_classes, "load_training_frame_with_classes",
        lambda cfg, n: frame.assign(proximity_class=classes),
    )  # fmt: skip
    out = tmp_path / "frozen"
    out.mkdir()
    summary = fz.freeze_q2(config, out)
    meta = json.loads((out / "q2_meta.json").read_text())
    assert summary["n_train_rows"] == meta["n_train_rows"] > 0
    assert abs(sum(meta["prior_train"]) - 1) < 1e-9 and abs(sum(meta["prior_target"]) - 1) < 1e-9
    assert meta["prior_target"][4] == pytest.approx(76 / 5046)  # the published validation mix
    assert meta["best_constant_from_validation"] == "always_4"
