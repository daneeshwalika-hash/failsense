"""Tests for failsense.config."""

from pathlib import Path

import pytest

from failsense.config import find_repo_root, load_config, resolve_path


def test_find_repo_root_from_nested_folder(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    nested = tmp_path / "notebooks" / "deep"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == tmp_path.resolve()


def test_find_repo_root_raises_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)


def test_find_repo_root_uses_env_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    monkeypatch.setenv("FAILSENSE_ROOT", str(tmp_path))

    assert find_repo_root() == tmp_path.resolve()


def test_find_repo_root_falls_back_to_package_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAILSENSE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # a folder outside any repo, like /tmp

    root = find_repo_root()

    assert (root / "configs" / "config.yaml").is_file()


def test_find_repo_root_finds_this_repo() -> None:
    root = find_repo_root(Path(__file__).parent)

    assert (root / "pyproject.toml").is_file()
    assert (root / "configs" / "config.yaml").is_file()


def test_load_config_reads_project_config() -> None:
    config = load_config()

    assert "datasets" in config
    assert "scania" in config["datasets"]
    assert config["seed"] == 42


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("content", ["", "- just\n- a list\n"])
def test_load_config_rejects_non_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content)

    with pytest.raises(TypeError):
        load_config(path)


def test_resolve_path_relative_and_absolute(tmp_path: Path) -> None:
    assert resolve_path("data/raw/x.csv", root=tmp_path) == tmp_path / "data/raw/x.csv"
    assert resolve_path(tmp_path / "abs.csv", root=Path("/other")) == tmp_path / "abs.csv"
