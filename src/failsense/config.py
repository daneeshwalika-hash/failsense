"""Load the central YAML configuration and resolve paths relative to the repo root.

All code reads paths from ``configs/config.yaml`` through this module, so no
path is hardcoded and the code works from any working directory
(scripts, tests, notebooks).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT_MARKER = "pyproject.toml"
ROOT_ENV_VAR = "FAILSENSE_ROOT"
DEFAULT_CONFIG = Path("configs") / "config.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root (the folder that contains ``pyproject.toml``).

    Search order:

    1. ``start``, if given: walk up from there.
    2. The environment variable ``FAILSENSE_ROOT``, if set (e.g. in Docker).
    3. The current working directory and its parents.
    4. The location of this module (works from any folder after ``uv sync``,
       because the package is installed in editable mode from ``src/``).

    Args:
        start: Folder to start from.

    Returns:
        The repository root folder.

    Raises:
        FileNotFoundError: If no candidate folder contains ``pyproject.toml``.
    """
    if start is not None:
        candidates = [start]
    else:
        env_root = os.environ.get(ROOT_ENV_VAR)
        candidates = [Path(env_root)] if env_root else []
        candidates += [Path.cwd(), Path(__file__).parent]

    for candidate in candidates:
        resolved = candidate.resolve()
        for folder in (resolved, *resolved.parents):
            if (folder / ROOT_MARKER).is_file():
                return folder

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"{ROOT_MARKER} not found above: {searched}")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration.

    Args:
        config_path: Path to a YAML file. Defaults to ``configs/config.yaml``
            in the repository root.

    Returns:
        The configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        TypeError: If the file is empty or not a YAML mapping.
    """
    path = config_path or find_repo_root() / DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")

    logger.debug("Loaded config from %s", path)
    return config


def resolve_path(relative: str | Path, root: Path | None = None) -> Path:
    """Turn a path from the config (relative to the repo root) into an absolute path.

    Args:
        relative: Path as written in the config, e.g. ``data/raw/scania/train_tte.csv``.
            Absolute paths are returned unchanged.
        root: Repository root. Defaults to :func:`find_repo_root`.

    Returns:
        The absolute path.
    """
    path = Path(relative)
    if path.is_absolute():
        return path
    return (root or find_repo_root()) / path
