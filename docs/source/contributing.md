# Development and contributing

ROOT Explorer uses [uv](https://docs.astral.sh/uv/) for reproducible development.
Python 3.12 is the supported interpreter. The committed `uv.lock` is the
source of resolved dependencies for CI and Docker, while `pyproject.toml`
maintains broad compatibility requirements for ordinary installations.

```bash
uv sync --locked
uv run --no-sync pytest
uv run --no-sync mypy
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

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

## Pre-commit hooks

Install hooks with `uv run --no-sync pre-commit install`, and check all
files with `uv run --no-sync pre-commit run --all-files`. The external
ruff-pre-commit hook is retained. When upgrading Ruff, keep its hook revision,
the `lint` dependency version and `tool.ruff.required-version` aligned.

## Releasing

ROOT Explorer uses CalVer `YYYY.M.N` and annotated Git tags. A version
change must also update `uv.lock` **before** committing and tagging, since
the lock records the local distribution version.

From an up-to-date clean `main` checkout with `uv sync --locked` already run:

```bash
uv run --no-sync python scripts/release.py patch
git push --follow-tags
```

Use `release` instead of `patch` for a new calendar release where
appropriate. The helper invokes `bump-my-version` without committing or
tagging itself, regenerates the lock, verifies the matching versions, then
creates a single release commit and annotated tag. It does not push.
The publish workflow independently verifies `uv lock --check` and the tag.

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
