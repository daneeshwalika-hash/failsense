"""Smoke test: the package imports and exposes a valid version string."""

import re

import failsense


def test_package_has_semver_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", failsense.__version__)
