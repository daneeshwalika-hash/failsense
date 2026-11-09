"""Freeze everything the single test evaluation needs (docs/test_evaluation_plan.md).

``make freeze`` trains the models that exist only in memory (Q2 multi-class, RUL LightGBM, Ridge and
mean baselines) on **training data only**, copies the registered Q1 model, and writes
``reports/frozen_manifest.json``: the git commit, SHA-256 hashes of every frozen file, of the config
sections that matter and of the five test files (the test files are only HASHED: their bytes are not
parsed), the library versions and the decisions.

The evaluation script (written after the plan is approved) calls ``verify_manifest`` first and
refuses to run on any difference. This module never reads test content and never evaluates anything
on test.

Why skops and not pickle: the frozen models are loaded again later; ``skops`` lists the types it
would need to trust, they are recorded in the manifest, and the file hash is checked before loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from skops import io as skops_io

from failsense.config import find_repo_root, load_config, resolve_path

logger = logging.getLogger(__name__)

CODE_PATHS = (
    "src", "configs", "pyproject.toml", "uv.lock", "reports/registry_q1.json",
    "reports/calibration_q1_summary.json",
)  # fmt: skip
CONFIG_SECTIONS = ("seed", "labels", "features", "sampling", "cv", "cost", "q2", "rul")
LIBRARIES = (
    "python", "numpy", "pandas", "scikit-learn", "xgboost-cpu", "lightgbm", "mlflow", "skops",
)  # fmt: skip
DECISIONS = (
    "RUL model: LightGBM + rolling mean/std; baselines: Ridge (raw) and the mean predictor",
    "RUL primary reference: the RUL capped at rul.cap; secondary: raw RUL and the NASA score",
    "Q2 target mix: the class mix published for the validation, never the test counts (ADR 0004)",
    "Q1 headline: primary cost mapping first, literal next to it; never / always alert baselines",
    "Not evaluated on test: MLP, LightGBM for Q1, anomaly detectors and feature, the challenger",
    "Results are reported as they are: no re-tuning and no second run",
)


class FreezeError(RuntimeError):
    """The freeze (or its verification) cannot be trusted."""


# --- hashing and git -----------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (streamed; the content is not interpreted)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(folder: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of every file under ``folder`` (sorted, POSIX separators)."""
    return {
        p.relative_to(folder).as_posix(): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_commit(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def git_dirty_files(root: Path) -> list[str]:
    """Tracked or untracked (not ignored) files that differ from the commit."""
    return [line for line in _git(root, "status", "--porcelain").splitlines() if line.strip()]


def code_changes_since(root: Path, commit: str) -> list[str]:
    """Code/config files that differ between ``commit`` and the working tree."""
    out = _git(root, "diff", "--name-only", commit, "--", *CODE_PATHS)
    return [line for line in out.splitlines() if line.strip()]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config sections that determine the frozen results (others are irrelevant to them)."""
    return {name: config[name] for name in CONFIG_SECTIONS}


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the frozen config sections."""
    return hashlib.sha256(canonical_json(config_snapshot(config)).encode()).hexdigest()


def library_versions() -> dict[str, str]:
    import platform

    versions = {"python": platform.python_version()}
    for name in LIBRARIES[1:]:
        versions[name] = metadata.version(name)
    return versions


def test_file_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """The five test files of the config (SCANIA readouts, specs, labels; C-MAPSS test, RUL)."""
    scania, cmapss = config["datasets"]["scania"]["test"], config["datasets"]["cmapss"]
    return {
        "scania_readouts": resolve_path(scania["readouts"]),
        "scania_specifications": resolve_path(scania["specifications"]),
        "scania_labels": resolve_path(scania["labels"]),
        "cmapss_test": resolve_path(cmapss["test"]),
        "cmapss_rul": resolve_path(cmapss["rul"]),
    }


# --- model files ---------------------------------------------------------------------------------
def save_model(model: Any, path: Path) -> list[str]:
    """Save with skops; returns the types skops would need to trust to load it again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    skops_io.dump(model, path)
    return sorted(skops_io.get_untrusted_types(file=path))


def load_model(path: Path, trusted: Sequence[str]) -> Any:
    """Load a frozen model, trusting exactly the recorded types."""
    return skops_io.load(path, trusted=list(trusted))


# --- the three freezes ---------------------------------------------------------------------------
def freeze_q1(config: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Copy the registered Q1 model and record calibrator, threshold and version."""
    import mlflow
    import mlflow.artifacts

    from failsense.api.service import load_deployment
    from failsense.tracking import tracking_params_from_config, tracking_uri_override

    tuning = config["tuning"]
    uri = f"models:/{tuning['registered_model']}@{tuning['alias']}"
    db_path, _, _ = tracking_params_from_config(config)
    mlflow.set_tracking_uri(tracking_uri_override() or f"sqlite:///{resolve_path(db_path)}")
    target = out / "q1_model"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(target))
    deployment = load_deployment(config)
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / "registry_q1.json").read_text("utf-8")
    )
    if str(deployment["model_version"]) != str(registry["version"]):
        raise FreezeError("the calibration summary and the registry file name different versions")
    return {
        "model_uri": uri, "model_version": str(registry["version"]), "run_id": registry["run_id"],
        "calibrator": deployment["calibrator"], "threshold": float(deployment["threshold"]),
        "n_input_columns": len(registry["input_columns"]),
        "folder": "q1_model",
    }  # fmt: skip


def freeze_q2(config: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Fit the multi-class model on all ``k5`` TRAINING rows and save it with the priors."""
    from failsense.data.labels import CLASS_COL
    from failsense.models.cost import cost_matrix_from_config
    from failsense.models.pipeline import select_categorical_columns
    from failsense.models.q2_classes import (
        class_prior,
        load_training_frame_with_classes,
        make_multiclass_pipeline,
        target_prior_from_config,
    )
    from failsense.models.validation_data import registry_input_columns, select_model_inputs

    seed = int(config["seed"])
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / "registry_q1.json").read_text("utf-8")
    )
    columns = registry_input_columns(config)
    frame = load_training_frame_with_classes(config, 5)
    x, y = select_model_inputs(frame, columns), frame[CLASS_COL].to_numpy(dtype=int)
    pipeline = make_multiclass_pipeline(
        registry["params"], int(registry["n_rounds"]), seed, select_categorical_columns(x)
    ).fit(x, y)
    trusted = save_model(pipeline, out / "q2_model.skops")
    meta = {
        "n_train_rows": len(x), "input_columns": columns,
        "prior_train": class_prior(y).tolist(),
        "prior_target": target_prior_from_config(config).tolist(),
        "prior_target_source": "class mix published for the validation split (not the test counts)",
        "cost_matrix": cost_matrix_from_config(config).tolist(),
        "best_constant_from_validation": "always_4", "params": registry["params"],
        "n_rounds": int(registry["n_rounds"]), "untrusted_types": trusted,
    }  # fmt: skip
    (out / "q2_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {k: meta[k] for k in ("n_train_rows", "prior_train", "prior_target")}


def freeze_rul(config: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """Fit LightGBM + rolling features and the Ridge baseline on ALL training engines."""
    from sklearn.linear_model import Ridge

    from failsense.rul.data import load_train
    from failsense.rul.labels import compute_rul, rul_cap_from_config
    from failsense.rul.timeseries import (
        feature_sets,
        informative_sensors,
        lightgbm_model,
        make_pipeline,
    )

    section, seed, cap = config["rul"], int(config["seed"]), rul_cap_from_config(config)
    train = load_train(resolve_path(config["datasets"]["cmapss"]["train"]))
    windows = [int(w) for w in section["rolling_windows"]]
    frame, sets = feature_sets(train, windows)
    y = compute_rul(frame, cap).to_numpy()
    lgbm = make_pipeline(lightgbm_model(dict(section["lightgbm"]), seed)).fit(
        frame[sets["rolling"]], y
    )
    ridge = make_pipeline(Ridge(alpha=float(section["ridge_alpha"]))).fit(frame[sets["raw"]], y)
    untrusted = {
        "rul_lightgbm.skops": save_model(lgbm, out / "rul_lightgbm.skops"),
        "rul_ridge.skops": save_model(ridge, out / "rul_ridge.skops"),
    }
    meta = {
        "cap": cap, "rolling_windows": windows, "informative_sensors": informative_sensors(train),
        "columns_raw": sets["raw"], "columns_rolling": sets["rolling"],
        "mean_rul_training": float(np.mean(y)), "n_train_rows": len(frame),
        "n_train_engines": int(frame["engine"].nunique()),
        "ridge_alpha": float(section["ridge_alpha"]),
        "lightgbm": dict(section["lightgbm"]), "untrusted_types": untrusted,
    }  # fmt: skip
    (out / "rul_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {k: meta[k] for k in ("cap", "rolling_windows", "n_train_engines", "mean_rul_training")}


# --- manifest ------------------------------------------------------------------------------------
def assemble_manifest(
    root: Path, config: Mapping[str, Any], out: Path, q1: Mapping[str, Any],
    q2: Mapping[str, Any], rul: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """The manifest dict: commit, hashes of frozen / source / test files, config, libraries."""
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": git_commit(root),
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "config": config_snapshot(config),
        "config_sha256": config_hash(config),
        "frozen_files": hash_tree(out),
        "source_files": {rel: sha256_file(root / rel) for rel in CODE_PATHS[4:]},
        # the test files are only hashed (bytes), never parsed
        "test_files": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for name, path in test_file_paths(config).items()
        },
        "q1": dict(q1), "q2": dict(q2), "rul": dict(rul),
        "libraries": library_versions(), "decisions": list(DECISIONS),
        "note": "trained on training data only; no test file was parsed, nothing evaluated on test",
    }  # fmt: skip


def freeze_all(config: Mapping[str, Any], replace: bool = False) -> dict[str, Any]:
    """Train and copy everything on TRAINING data, write the manifest, return it.

    Raises:
        FreezeError: If the working tree is dirty, or a manifest exists and ``replace`` is False.
    """
    root = find_repo_root()
    manifest_path = resolve_path(config["freeze"]["manifest"])
    out = resolve_path(config["freeze"]["dir"])
    dirty = git_dirty_files(root)
    if dirty:
        raise FreezeError(f"the working tree is not clean, commit first: {dirty[:5]}")
    if manifest_path.exists() and not replace:
        raise FreezeError(f"{manifest_path} exists; use --replace to freeze again (this is logged)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    q1 = freeze_q1(config, out)
    q2 = freeze_q2(config, out)
    rul = freeze_rul(config, out)

    manifest = assemble_manifest(root, config, out, q1, q2, rul)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Frozen at commit %s; manifest %s", manifest["code_commit"][:10], manifest_path)
    return manifest


def verify_manifest(config: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Problems that make the frozen state untrustworthy (empty list = verified).

    Checks the frozen files, the source files, the config sections, ``uv.lock``, the five test
    files, that the working tree is clean and that no code or config file changed since the commit.
    """
    root = root or find_repo_root()
    path = resolve_path(config["freeze"]["manifest"])
    if not path.is_file():
        return [f"manifest {path} not found; run `make freeze`"]
    manifest = json.loads(path.read_text("utf-8"))
    out = resolve_path(config["freeze"]["dir"])
    problems: list[str] = []

    if not out.is_dir():
        problems.append(f"frozen folder {out} is missing")
    else:
        now = hash_tree(out)
        for name, digest in manifest["frozen_files"].items():
            if name not in now:
                problems.append(f"frozen file missing: {name}")
            elif now[name] != digest:
                problems.append(f"frozen file changed: {name}")
        problems += [
            f"unexpected frozen file: {n}" for n in sorted(set(now) - set(manifest["frozen_files"]))
        ]
    for rel, digest in manifest["source_files"].items():
        if not (root / rel).is_file() or sha256_file(root / rel) != digest:
            problems.append(f"source file changed: {rel}")
    if sha256_file(root / "uv.lock") != manifest["uv_lock_sha256"]:
        problems.append("uv.lock changed")
    if (
        hashlib.sha256(canonical_json(config_snapshot(config)).encode()).hexdigest()
        != manifest["config_sha256"]
    ):
        problems.append("the frozen config sections changed")
    for name, entry in manifest["test_files"].items():
        file = root / entry["path"]
        if not file.is_file() or sha256_file(file) != entry["sha256"]:
            problems.append(f"test file changed or missing: {name}")
    if library_versions() != manifest["libraries"]:
        problems.append("library versions differ from the freeze")
    dirty = git_dirty_files(root)
    if dirty:
        problems.append(f"working tree not clean: {dirty[:5]}")
    try:
        problems += [
            f"code changed since the freeze: {f}"
            for f in code_changes_since(root, manifest["code_commit"])
        ]
    except FreezeError as exc:
        problems.append(str(exc))
    return problems


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replace", action="store_true", help="freeze again over an existing manifest"
    )
    parser.add_argument("--verify", action="store_true", help="only verify the existing manifest")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    if args.verify:
        problems = verify_manifest(config)
        for problem in problems:
            logger.error(problem)
        logger.info("verified" if not problems else f"{len(problems)} problem(s)")
        raise SystemExit(1 if problems else 0)
    freeze_all(config, replace=args.replace)


if __name__ == "__main__":
    main()
