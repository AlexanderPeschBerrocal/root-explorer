# Development and contributing

ROOT Explorer uses [uv](https://docs.astral.sh/uv/) for reproducible development.
Python 3.12 (3.12.4 or later) is the supported interpreter; CI tests both that
floor and the `.python-version` release shipped in Docker. The committed
`uv.lock` is the source of resolved dependencies for CI and Docker, while
`pyproject.toml` maintains broad compatibility requirements for ordinary
installations.

```bash
uv sync --locked
uv run --no-sync pytest
uv run --no-sync mypy
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Without uv, pip 25.1 or newer installs the same dependency groups, resolved
from `pyproject.toml` rather than the lock, for example
`python -m pip install -e . --group dev`. The pre-commit hooks and the release
helper still call uv and use the project's `.venv`.

To reproduce individual CI jobs without installing all development tools:

```bash
uv sync --locked --no-default-groups --group test
MPLBACKEND=Agg uv run --no-sync pytest -q --cov=root_explorer --cov-report=term-missing

uv sync --locked --no-default-groups --group typing
uv run --no-sync mypy

uv sync --locked --no-default-groups --group docs
uv run --no-sync python -m sphinx -n -W --keep-going -b html docs/source docs/_build/html
```

The standard application environment does not install the optional XRootD
backend. Users needing it can install `pip install "root-explorer[xrootd]"`;
in an editable checkout use `uv sync --locked --extra xrootd`. XRootD may
require additional native libraries.

## Dependency updates

```bash
uv lock --check             # Verify that pyproject.toml and lock agree
uv lock --upgrade           # Intentionally refresh the resolved environment
uv lock --upgrade-package numpy  # Update one dependency and any required peers
```

Commit changes to `pyproject.toml` and `uv.lock` together. Dependabot
opens reviewed weekly PRs for uv, Python base images, GitHub Actions and
pre-commit. CI validates the lock, tests, documentation, Docker and the
independent wheel installation. In particular, a lockfile update is **not**
equivalent to changing the published package's compatibility metadata.

`.python-version` pins the exact interpreter that uv uses locally and in CI.
It must match the Python version of the Docker base image, which the Docker
workflow checks; update both together when the base image changes.

uv itself is pinned to the same version in the Dockerfile's digest-pinned uv
image, every `setup-uv` step and `.readthedocs.yaml`. Dependabot updates only
the Dockerfile image, so update the other pins in the same pull request; the
build workflow fails while they differ.

## Pre-commit hooks

Install hooks with `uv run --no-sync pre-commit install`, and check all
files with `uv run --no-sync pre-commit run --all-files`. The Ruff hooks run
the Ruff installed in the project environment, so the `ruff==` pin in the
`lint` dependency group (and `uv.lock`) is the only place its version is
set; Dependabot updates both together. Run `uv sync --locked` after pulling
a Ruff update.

## Releasing

ROOT Explorer uses CalVer `YYYY.M.N` and annotated Git tags. A version
change must also update `uv.lock` **before** committing and tagging, since
the lock records the local distribution version.

From an up-to-date clean `main` checkout with `uv sync --locked` already run:

```bash
uv run --no-sync python scripts/release.py patch
git push --follow-tags
```

Use `release` instead of `patch` for the first release of a new calendar
month; the helper refuses a `release` bump in a month that already has a
release. It invokes `bump-my-version` without committing or tagging itself,
regenerates the lock, verifies the matching versions, then creates a single
release commit and annotated tag. It does not push.
The publish workflow independently verifies `uv lock --check` and the tag.

The **Release** GitHub Actions workflow offers the same from the browser:
choose *Run workflow* on `main`, the version part, and a mode. `dry-run`
creates the release in a throwaway clone without push access and checks it
with `scripts/check_release.py`; `publish` does the same dry run first, then
creates the release on `main`, pushes the commit and tag atomically as the
user who started the run, and starts the Docker publish workflow for the tag.
Pull requests that change `pyproject.toml`, `uv.lock` or the release scripts
dry-run both parts automatically.

## Documentation and screenshots

Build docs with `uv run --no-sync python -m sphinx -n -W --keep-going -b html
docs/source docs/_build/html` after `uv sync --locked`.
For screenshots first install Chromium with
`uv run --no-sync playwright install --with-deps chromium` (Linux requires
system browser dependencies), then run
`uv run --no-sync python docs/scripts/generate_screenshots.py --theme both`.
Screenshots are reviewed before being published into the source tree.

The synthetic ROOT fixtures and histogram invariant tests require no PyROOT.
For headless tests, set `MPLBACKEND=Agg`. The app can be started locally
with `uv run --no-sync root-explorer`; ordinary end users may still use pip.
