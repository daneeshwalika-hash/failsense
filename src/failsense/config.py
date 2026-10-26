"""Load the central YAML configuration and resolve paths relative to the repo root.

All code reads paths from ``configs/config.yaml`` through this module, so no
path is hardcoded and the code works from any working directory
(scripts, tests, notebooks).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT_MARKER = "pyproject.toml"
DEFAULT_CONFIG = Path("configs") / "config.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until the folder containing ``pyproject.toml`` is found.

    Args:
        start: Folder to start from. Defaults to the current working directory.

    Returns:
        The repository root folder.

    Raises:
        FileNotFoundError: If no parent folder contains ``pyproject.toml``.
    """
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / ROOT_MARKER).is_file():
            return folder
    raise FileNotFoundError(f"{ROOT_MARKER} not found above {start} - are you inside the repo?")


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
