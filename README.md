<p align="center">
  <img
    src="src/root_explorer/assets/root-explorer-app-icon-transparent-edges.png"
    alt="ROOT Explorer app icon"
    width="160"
  >
</p>

<h1 align="left">ROOT Explorer</h1>

<p align="left">
  A general-purpose Streamlit app for interactive inspection and plotting of
  HEP ROOT TTrees and RNTuples in the web browser.
</p>

## Run the app

ROOT Explorer requires Python 3.12. From a checkout, create a virtual
environment, install the project and its runtime dependencies, and launch the
app:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run streamlit_app.py
```

### Unit-area density normalisation

For event-derived **1D histograms**, enable **Normalise to unit area** under
Histogram analysis. The displayed histogram is a bin-width-corrected density:

```text
density[i] = sum_weights[i] / (normalisation_sum * bin_width[i])
sum_i density[i] * bin_width[i] = 1
```

The operation is performed after variable evaluation, selections, non-finite
filtering and optional event weights. **Ignore** normalises using only nominal
in-range bins, producing a conditional in-range density. **Fold** first moves
underflow and overflow yields into the corresponding edge bins and then includes
them in normalisation; the edge-bin density is therefore a visualisation of
folded yields, not a local density estimate of the original out-of-range data.
**Show separate bins** cannot be combined with density mode because flow
categories do not have finite widths. Axis zoom does not affect the
normalisation domain.

Unweighted and weighted histograms use their visible bin sums as denominators.
For signed weights, a positive net integral produces a *signed unit-integral
density*, which need not be non-negative or interpretable as a probability
distribution. Empty, non-positive or numerically unstable denominators leave
the original yields visible with a warning. Original sample statistics and
flow tables are unchanged. The y-axis defaults to **Normalised density**;
for an observable measured in GeV, the physical density unit is GeV⁻¹.

For a fixed denominator `S`, stored variances are rescaled by
`1 / (S * bin_width[i])**2`. Since the denominator is estimated from the
same data, these diagonal variances do not capture the full covariance of a
self-normalised distribution. They must not be treated as independent
statistical uncertainties in shape-only fits or tests.

Histogram fits continue to operate on the original flow-adjusted bin yields;
their displayed predictions are divided by `S * bin_width[i]` in density mode.
Fit parameters and goodness-of-fit statistics therefore do not change when
the density display is toggled. KDE overlays use density units when enabled;
the smooth KDE may have tails outside the histogram range. KDE is unavailable
for normalised flow-folded histograms.

**Download plot code** reproduces the chosen density, flow, fit and KDE
semantics. Currently, the control applies to event-derived 1D histograms.
Persisted ROOT `TH1*` objects can contain counts, densities or other
quantities without reliable semantic metadata, and are not normalised
automatically. `TProfile*` bin means must not be treated as yields.

### Built-in histogram fitting

For a 1D histogram, enable **Histogram fit** and choose Gaussian,
Exponential, single-sided Crystal Ball, or double-sided Crystal Ball.
This is available for histograms built from TTree/RNTuple branches and
persisted `TH1*` ROOT objects. Fitting of `TProfile` bin means is
not yet supported.

ROOT Explorer uses **binned least squares**, not a Poisson likelihood.
Ordinary least squares assigns equal importance to every selected bin.
You may explicitly enable **Use stored bin variances** to minimise the
sum of squared residuals divided by each bin's variance (`sum(w²)`).
Bins with missing, non-finite, or non-positive variances are excluded and
reported; they do not affect starting parameter estimates. No Poisson
uncertainties are inferred from unweighted counts.

Predictions are **integrals of the fit model over the original bins**
rather than evaluations at bin centres. The custom fit interval selects
whole bins whose centres fall inside it. Underflow and overflow do not
contribute. Parameter guesses and physically meaningful default bounds
are provided automatically; both can be overridden. Fit failures display
a warning without removing the histogram.

The output table reports parameters and approximate local covariance-based
uncertainties where identifiable. Ordinary least squares reports the
sum of squared residuals, not a chi-squared p-value. Variance-weighted
least squares reports a chi-squared statistic whose interpretation depends
on the validity of the supplied variances. Fits are also reproduced by
**Download plot code** when enabled.

### Custom mathematical histogram fits

Choose **Custom expression** in the histogram fit model selector to enter a
restricted mathematical expression such as:

```text
A * exp(-0.5 * ((x - mu) / sigma)**2) + c
```

`x` is the independent variable. Other identifiers become fit parameters,
except for the constants `pi` and `e` and the functions `exp`, `log`,
`sqrt`, `abs`, `sin`, `cos`, `tan`, `sinh`, `cosh`, and `tanh`.
Arithmetic with `+`, `-`, `*`, `/`, and `**` is supported. The interface
discovers parameters in their first-appearance order and provides controls for
initial values and lower/upper bounds. By default, a custom parameter starts
at 1 and is unbounded; supply appropriate initial values and bounds for
non-linear or ill-conditioned fits.

Expressions are validated against a restricted syntax tree before evaluation.
Python statements, imports, attribute access, indexing, and arbitrary function
calls are forbidden. Custom models are integrated over the original bins using
fixed-order numerical quadrature, whereas built-in models retain their analytic
bin integrals. Extremely narrow peaks or rapidly oscillating expressions within
one bin can exceed the numerical accuracy of the quadrature. Fit failures leave
the underlying histogram visible. Exported standalone plot code includes the
same expression, parameter settings, and restricted expression evaluator.

### Standalone plot code

The **Download plot code** action creates a standalone, editable Python script
for the selected static plot. It contains only the analysis and rendering
operations selected at export time. Input paths, branch names, labels, ranges,
and non-structural styles can be edited directly; regenerate the script after
changing the plot type, variable modes, optional features, or logarithmic scales.

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
annotated Git tag. The `[project].version` value in `pyproject.toml` is the
single authoritative version literal; `root_explorer.__version__` is derived
from the installed distribution metadata.

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

## License

ROOT Explorer is licensed under the [BSD 3-Clause License](LICENSE).
Third-party dependencies retain their own licence terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for an overview.
