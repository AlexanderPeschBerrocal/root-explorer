# root-explorer
A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees and RNTuples in the web browser.

## Run the app

From a checkout, users can launch the app directly:

```bash
streamlit run streamlit_app.py
```

### Docker

The application can also be built and run in Docker. This provides a reproducible
runtime environment and does not require installing Python or the root-explorer
dependencies on the host system.

Released images are published to the GitHub Container Registry (GHCR). To run the
most recent stable release:

```bash
docker pull ghcr.io/alexanderpeschberrocal/root-explorer:stable
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

For reproducible deployments, use a versioned image tag instead, for example:

```bash
docker pull ghcr.io/alexanderpeschberrocal/root-explorer:v2026.9.1
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:v2026.9.1
```

The `stable` tag moves to the most recent release, while versioned tags remain
fixed. Public GHCR images can be pulled without authentication; private packages
require authenticating Docker to GHCR first.

From the repository root, the image can also be built locally:

```bash
docker build -t root-explorer .
```

Run the locally built container and publish the Streamlit port:

```bash
docker run --rm -p 8501:8501 root-explorer
```

Then open [http://localhost:8501](http://localhost:8501) in a web browser.

To make local ROOT or GDML files available when using the application's `Path`
input mode, bind-mount a host directory into the container. For example:

```bash
docker run --rm \
    -p 8501:8501 \
    --mount type=bind,source="/absolute/path/to/data",target=/data,readonly \
    root-explorer
```

Use the path as seen inside the container, for example `/data/events.root` or
`/data/detector.gdml`, rather than the corresponding host path. Files uploaded
through the web interface do not require a bind mount.

The container runs the application as an unprivileged user and includes a Docker
health check for the Streamlit service. Stop the application with `Ctrl+C`; because
`--rm` is used, Docker removes the stopped container automatically.

## Development

From the repository root, install the package in editable mode with the developer
tools, including Ruff and pre-commit:

```bash
pip install -e ".[dev]"
```

### Pre-commit hooks

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

### Versioning and releases

This project uses calendar versioning (CalVer) in the form `YYYY.M.N`, where
`YYYY.M` is the year and month of the release and `N` is the release sequence
within that month (for example, `2026.9.0` or `2026.9.1`). Version changes are
managed with [bump-my-version](https://callowayproject.github.io/bump-my-version/),
which updates the package version, creates the release commit, and creates its
annotated Git tag.

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
matches both `pyproject.toml` and `root_explorer.__version__`, and that the tagged
commit is contained in `main`.

A successful release publishes the same container image to GHCR with two tags:

- `vYYYY.M.N`: an immutable version-specific release tag;
- `stable`: a moving tag pointing to the most recent release.

The workflow then creates the corresponding GitHub Release with automatically
generated release notes. Python wheel or source-distribution artefacts are not
published as part of this release process.

## License

ROOT Explorer is licensed under the [BSD 3-Clause License](LICENSE).
Third-party dependencies retain their own licence terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for an overview.
