<p align="center">
  <img src="src/root_explorer/assets/root-explorer-app-icon-transparent-edges.png"
       alt="ROOT Explorer icon" width="128">
</p>

# ROOT Explorer

**Explore HEP ROOT files directly in your browser.** ROOT Explorer is an interactive Streamlit app for browsing TTrees, RNTuples and persisted ROOT objects, making plots, fitting histograms and inspecting individual events.

**[Documentation](https://root-explorer.readthedocs.io/)** · **[Source](https://github.com/AlexanderPeschBerrocal/root-explorer)** · **[Issues](https://github.com/AlexanderPeschBerrocal/root-explorer/issues)**

![ROOT Explorer: real application interface generated with synthetic ROOT data](docs/source/_static/screenshots/overview.png)

## What can it do?

- Inspect TTrees, RNTuples, ROOT histograms, profiles and graphs.
- Plot scalar branches and jagged particle collections, with cuts, weights and derived observables.
- Create 1D and 2D histograms, scatter plots, density displays, KDEs and least-squares fits.
- Inspect and rank events, or overlay supported GDML geometry on 3D scatter plots.
- Download figures and standalone Python plotting scripts for reproducible follow-up analysis.

## Quick start

Run the published Docker image:

```bash
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

Open [http://localhost:8501](http://localhost:8501). Use **Upload** to select a ROOT file from your computer. For local paths and reproducible deployments, consult [installation and Docker](docs/source/installation.md). A versioned image tag is recommended in production.

With a local Python 3.12 installation, `python -m pip install .` provides the `root-explorer` command; `root-explorer path/to/file.root` starts the app with that file opened.

For Python 3.12 source installation, development, and scientific explanations of histogram statistics and fitting, use the [documentation on Read the Docs](https://root-explorer.readthedocs.io/). Its source is in [`docs/source`](docs/source/index.rst).

## Licence

ROOT Explorer is licensed under the [BSD 3-Clause Licence](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).

## Reproducible development using uv

ROOT Explorer uses a committed `uv.lock` for its development, CI and Docker
environments. Install [uv](https://docs.astral.sh/uv/) and use:

```bash
uv sync --locked
uv run --no-sync pytest
uv run --no-sync mypy
uv run --no-sync ruff check .
```

The lock is updated intentionally with `uv lock --upgrade`, and consistency is
verified with `uv lock --check`. Public `pip install .` installations still
resolve compatible versions independently; the lock is not part of public
package metadata.
