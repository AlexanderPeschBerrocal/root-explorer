# Development and contributing

From the repository root, install the package in editable mode with the developer
tools, including Ruff and pre-commit:

```bash
pip install -e ".[dev,test]"
```

## Running the tests

Run the test suite from the repository root:

```bash
python -m pytest
```

The tests use small synthetic ROOT files, generated during the test run or
committed in `tests/data`, and require neither external data nor a CERN ROOT
installation. Set `MPLBACKEND=Agg` when running on
a machine without a display, as CI does.

To reproduce the CI test job, including the coverage report, run:

```bash
MPLBACKEND=Agg python -m pytest -q --cov=root_explorer --cov-report=term-missing
```

Type checking uses mypy with the file selection configured in `pyproject.toml`:

```bash
python -m mypy
```

The check covers the complete `src/root_explorer` package. The `dev` extra
installs the `pandas-stubs` and `scipy-stubs` type stubs. Plotly provides no type
information and is excluded by a narrow per-module override; prefer precise
annotations or explicit narrowing over `# type: ignore` comments.

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to run the Ruff checks
defined in `.pre-commit-config.yaml`. The configuration pins the Ruff version and
enables two hooks:

- `ruff-check` runs the configured lint checks and applies safe fixes.
- `ruff-format` checks and applies Ruff's formatting.

Install the hooks once for your local checkout:

```bash
pre-commit install
```

The hooks run automatically on every commit. To run them manually, for example
before opening a pull request, check the complete repository with:

```bash
pre-commit run --all-files
```

To run only one hook, use its hook ID:

```bash
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
```

The hooks may modify files because `ruff-check` is configured with `--fix`. Review
and stage those changes, then run the hooks again or create the commit again.

When changing `.pre-commit-config.yaml`, validate it from the repository root:

```bash
pre-commit validate-config
```

If the Ruff revision is updated, keep it synchronized with the Ruff version in
`pyproject.toml` and `.github/workflows/ruff.yaml` so local hooks and CI use the
same version. Use `pre-commit autoupdate` only when intentionally updating the
hook revisions, and review the resulting configuration before committing it.

## Versioning and releases

This project uses calendar versioning (CalVer) in the form `YYYY.M.N`, where
`YYYY.M` is the year and month of the release and `N` is the release sequence
within that month (for example, `2026.9.0` or `2026.9.1`). Version changes are
managed with [bump-my-version](https://callowayproject.github.io/bump-my-version/),
which updates the package version, creates the release commit, and creates its
annotated Git tag. The `[project].version` value in `pyproject.toml` is the
single authoritative version literal; `root_explorer.__version__` is derived
from the installed distribution metadata. The application must be installed to
run; there is no separate version literal for running from the source tree.

Installing the project copies the version into the distribution metadata. An
editable installation keeps the metadata of its last installation, so after
changing the version, for example with `bump-my-version`, reinstall it:

```bash
python -m pip install -e .
```

The **About** menu reads the installed version on every rerun, so a running app
shows the new version after the reinstall without a restart. `root-explorer
--version` and `root_explorer.__version__` in an already running process update
after a restart. If the installed and declared versions disagree, `tests/test_version.py`
fails with a reminder to reinstall. CI performs the same comparison of
`pyproject.toml`, the installed wheel and `root_explorer.__version__`, and checks
the version inside the Docker image.

Releases must be created from a clean, up-to-date `main` branch. Before creating
a release, inspect the configured version and preview the change without modifying
files:

```bash
git switch main
git pull --ff-only

bump-my-version show current_version
bump-my-version bump patch --dry-run -v
```

Create the release commit and annotated tag, then push both:

```bash
bump-my-version bump patch
git push --follow-tags
```

Pushing a `vYYYY.M.N` tag triggers the release workflow. Before publishing, the
workflow verifies that the tag follows the CalVer convention, that the tag version
matches the authoritative version in `pyproject.toml`, and that the tagged commit
is contained in `main`.

A successful release publishes the same container image to GHCR with two tags:

- `vYYYY.M.N`: an immutable version-specific release tag;
- `stable`: a moving tag pointing to the most recent release.

The workflow then creates the corresponding GitHub Release with automatically
generated release notes. Python wheel or source-distribution artefacts are not
published as part of this release process.


## Documentation development

With Python 3.12 and the repository checked out, install the optional
documentation dependencies and build HTML locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -n -W --keep-going -b html docs/source docs/_build/html
```

Review `docs/_build/html/index.html` and all changed pages in a browser.
The CI workflow in `.github/workflows/docs.yaml` runs the same strict
Sphinx build on pull requests to `main`. Screenshots are generated by a
separate workflow and are not required for every documentation build.
