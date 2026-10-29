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

ROOT Explorer uses CalVer `YYYY.M.N` and annotated Git tags. Each calendar
month's first release has patch number `0` and is created with the version
part `release`; any further release in the same month uses `patch`. For
example, after `2026.10.1` the next release is `2026.10.2` (`patch`) in
October and `2026.11.0` (`release`) in November. A `release` bump in a month
that already has a release is refused.

A release can be published from the browser with the **Release** workflow or
from a local checkout. Both use `scripts/release.py`, which runs
`bump-my-version`, regenerates `uv.lock` (it records the package version), and
creates one release commit and an annotated tag.

### What a release produces

- A commit `Release: YYYY.M.N` on `main` that changes only the version in
  `pyproject.toml` and `uv.lock`.
- An annotated tag `vYYYY.M.N` on that commit.
- From the **Publish Docker image** workflow, which runs for the tag: checks
  that the tag matches `pyproject.toml`, that `uv.lock` is up to date and that
  the commit is on `main`; builds the `linux/amd64` and `linux/arm64` images;
  pushes `ghcr.io/alexanderpeschberrocal/root-explorer:vYYYY.M.N`; moves the
  `stable` tag to it; and creates a GitHub Release with generated notes.

### Publishing with the Release button

Anyone with write access to the repository can publish a release:

1. Open the repository's **Actions** tab and select the **Release** workflow.
2. Choose **Run workflow**.
3. Under *Use workflow from*, select `main`. Publishing from any other branch
   fails before anything is changed.
4. Under *Version part*, select `patch` or `release` as described above.
5. Under *Mode*, select `publish`. The default, `dry-run`, only checks.
6. Choose **Run workflow** to start.

With `publish`, the workflow runs two jobs:

1. **Release dry run** creates the release in a throwaway clone without push
   access and checks it with `scripts/check_release.py`. If this fails, nothing
   is published.
2. **Publish release** creates the release on the current `main`, checks it again,
   pushes the commit and tag together (either both arrive or neither does),
   and then starts the **Publish Docker image** workflow on the new tag. It
   has to start that workflow itself, because pushes made by a workflow do not
   trigger other workflows. The release commit and tag are attributed to the
   user who started the run.

With `dry-run`, only the first job runs, and **Publish release** is shown as
skipped in that run.

Only one manual Release run executes at a time. A run started meanwhile waits
for it; GitHub keeps at most one waiting run and cancels older waiting ones.

After a successful run, check that:

- the **Publish Docker image** run for the new tag succeeded (Actions tab);
- the GitHub Release `vYYYY.M.N` exists;
- the image is available for both platforms, for example with
  `docker buildx imagetools inspect ghcr.io/alexanderpeschberrocal/root-explorer:vYYYY.M.N`.

Publishing pushes directly to `main`. This works while `main` allows pushes
from GitHub Actions; if a branch ruleset restricting updates to `main` is
enabled, add GitHub Actions as a bypass actor or release locally instead. The
release commit does not start the push-triggered CI workflows on `main`, as
it only changes the version.

### Dry runs

A dry run checks that a release could be created, without changing the
repository or publishing anything. `scripts/check_release.py` then verifies
that the release commit changes only `pyproject.toml` and `uv.lock`, that the
version increases and agrees with `uv.lock`, and that the annotated
`vYYYY.M.N` tag points to the release commit.

- **Automatically:** the separate **Release dry run** workflow runs on pull
  requests that change `pyproject.toml`, `uv.lock`, the release scripts or
  that workflow, and dry-runs both version parts. In a month that already has
  a release, the `release` part passes when the helper correctly refuses it.
  The **Release** workflow itself only runs from the button, so its publish
  job never appears in pull request checks. The button reuses the same
  dry-run workflow.
- **Manually:** run the **Release** workflow with mode `dry-run` on **any
  branch**, selected under *Use workflow from*. No tag is needed. The run uses
  that branch's workflow and scripts, so this tests release changes before
  they are merged. The *Run workflow* button appears only once the workflow
  exists on the default branch.

A Release dry run does not build Docker images. To test the image on a branch,
run the **Docker** workflow manually on that branch: it builds and tests both
platforms and compares the installed packages with `uv.lock`, without pushing
anything. The **Publish Docker image** workflow itself runs only for release
tags, because it moves `stable` and creates a GitHub Release.

### If a release fails

- *Use 'patch' for another release in the same month*: this month already has
  a release; run again with the version part `patch`.
- *Releases can only be published from main*: run again with `main` selected
  under *Use workflow from*.
- **The push is rejected**, for example because `main` changed during the run
  or the tag already exists: nothing was pushed. Run the workflow again.
- **Publish Docker image fails** after the push: the release commit and tag
  are already on `main`. Fix the cause, then use *Re-run failed jobs* on that
  run, or start **Publish Docker image** manually with the tag `vYYYY.M.N`
  selected under *Use workflow from*. Do not create a new release for the
  same version.

### Releasing locally

From an up-to-date, clean `main` checkout with `uv sync --locked` already run:

```bash
uv run --no-sync python scripts/release.py patch   # or: release
git push --follow-tags
```

The helper does not push, so the commit and tag can be reviewed first. Pushing
the tag starts the **Publish Docker image** workflow directly.

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
