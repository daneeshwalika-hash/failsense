# ADR 0002: Install the `failsense` package into the environment with `uv_build`

- **Status:** Accepted
- **Date:** 2026-10-26
- **Deciders:** Daneesh Walika
- **Supersedes:** the rule "no `[build-system]` section" from the project setup (24.10.2026)

## Context

FailSense uses the `src/` layout (`src/failsense/`). Until now `pyproject.toml` had no
`[build-system]`, because FailSense is an application and will not be published to PyPI.
pytest found the code through `pythonpath = ["src"]`.

On 26.10.2026 the first module outside the tests (`failsense.data.convert`) had to run
from the terminal. It failed with `ModuleNotFoundError: No module named 'failsense'`:
`pythonpath` only works inside pytest. Notebooks, `make` targets, Docker and CI would all
need the same workaround.

## Options considered

| Option | How | Drawback |
| --- | --- | --- |
| A: keep the rule | Set `PYTHONPATH=src` in the Makefile, a `.env` file for VS Code/Jupyter, the Dockerfile and CI | Easy to forget in each new place; the same error comes back |
| **B: add a build backend** | `[build-system]` with `uv_build`; `uv sync` installs `failsense` as an editable package | Three lines in `pyproject.toml` |

## Decision

Option B. `pyproject.toml` gets a `[build-system]` with `uv_build` (the build backend
that ships with uv and expects the `src/` layout by default). `uv sync` installs
`failsense` into `.venv` in editable mode: changes in `src/` are visible immediately.
`pythonpath = ["src"]` is removed from the pytest settings, so the tests import the
installed package exactly like every other caller.

Installing is not publishing: FailSense is still an application and is not uploaded
to PyPI.

## Consequences

- `import failsense` works in the terminal, in notebooks, in `make` targets, in Docker and in CI.
- Tests now also catch packaging mistakes (for example a missing `__init__.py`).
- New contributors run `uv sync` (already part of `make setup`); no extra environment variables.
- The project instructions are updated: the rule "no `[build-system]`" is replaced by
  "`[build-system]` with `uv_build` only for the editable install; never publish to PyPI".
