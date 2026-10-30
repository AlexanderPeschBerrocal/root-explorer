# Regenerating documentation screenshots

Screenshots are actual Chromium captures of the locally running Streamlit app
using deterministic, completely synthetic NanoAOD-like test data; no external
ROOT data, CMS collaboration files, PyROOT, or production server are required.

Run from the repository root with the locked `screenshots` dependency group:

```bash
uv sync --locked --no-default-groups --group screenshots
uv run --no-sync playwright install chromium
uv run --no-sync python docs/scripts/generate_screenshots.py
```

The command launches and stops its own Streamlit instance on port 8751 and
generates **ten PNGs**, five each in light and dark mode, in
`docs/source/_static/screenshots/`. Dark images retain their existing names
(e.g. `overview.png`); light images use `-light.png` (e.g.
`overview-light.png`). Both themes use a 1440 × 900 CSS-pixel viewport
captured at 2× device scale (2880 × 1800 PNG pixels) for sharp documentation
images. This increases pixel density, not nominal PNG DPI metadata.
Static plots keep Matplotlib's default white style in both themes, as in the
app; only the Streamlit interface follows the theme.

The app must not already be listening on this port. To use a different free
port, pass `--port` (e.g. `--port 8752`); the script then polls
`http://127.0.0.1:8752`. `--base-url` overrides the polled address if needed.
Run `uv run --no-sync python docs/scripts/generate_screenshots.py --theme light` to regenerate
only the light screenshots (or `--theme dark` for dark only).

The separate `Documentation screenshots` GitHub Actions workflow runs on
relevant pull requests and on `workflow_dispatch`, and uploads the resulting
PNGs for human review. It has read-only repository permissions: **the screenshot
generation workflow never commits or pushes generated images**, including on
trusted branches. It does
not perform brittle pixel equality checks. To refresh screenshots, run the
generator, review the resulting images for legibility and privacy, and commit
the changed PNGs in an ordinary pull request. The Sphinx/Read the Docs build
never installs Chromium.

To publish a successful screenshot run **without downloading the ZIP locally**,
use the separate **Publish reviewed documentation screenshots** workflow in
GitHub Actions: choose *Run workflow*, enter the numeric run ID from the
successful **Documentation screenshots** job, and enter the existing review
branch for that run. Replace the form's historical `docs/74-readthedocs` default
with your current review branch. The workflow verifies the run's origin,
success status, matching branch and ten images, then commits both sets of
PNGs *and* updates the Sphinx/MyST image references to use Furo's `only-light`
and `only-dark` classes atomically on the review branch. It never publishes
directly to `main`; the images enter `main` through normal PR review and merge.

Alternatively, download the reviewed screenshots from a successful CI run or
generate them locally and commit the PNG files. From the repository root on
the review branch, run:

```bash
gh run download <successful-run-id> --repo AlexanderPeschBerrocal/root-explorer --name root-explorer-screenshots --dir docs/source/_static/screenshots
```

Then review all ten screenshots and commit them. Existing documentation image
references already select light and dark variants. When adding or renaming
screenshots, update those references in `index.rst` and the affected MyST guides
as well, retaining the `only-light` and `only-dark` classes.

Only publish after visually reviewing the screenshots. If the source run
predates a change to the capture script or UI, regenerate first.

Generated PNGs are expected to be committed; if capture fails, check the CI
artifact `failure-dark.png` or `failure-light.png` and update selectors
in the deterministic browser scenario.
