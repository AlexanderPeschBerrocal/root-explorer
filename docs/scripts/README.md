# Regenerating documentation screenshots

Screenshots are actual Chromium captures of the locally running Streamlit app
using deterministic, completely synthetic NanoAOD-like test data; no external
ROOT data, CMS collaboration files, PyROOT, or production server are required.

Use Python 3.12 and run from the repository root:

```bash
python -m pip install -e ".[screenshots]"
python -m playwright install chromium
python docs/scripts/generate_screenshots.py
```

The command launches and stops its own Streamlit instance on port 8751 and
generates five PNGs in `docs/source/_static/screenshots/`. The app must not
already be listening on this port. To use a different free port, pass both
`--port` and `--base-url` (e.g. `--port 8752 --base-url
http://127.0.0.1:8752`).

The separate `Documentation screenshots` GitHub Actions workflow runs on
relevant pull requests and on `workflow_dispatch`, and uploads the resulting
PNGs for human review. It has read-only repository permissions: **CI never
commits or pushes generated images**, including on trusted branches. It does
not perform brittle pixel equality checks. To refresh screenshots, run the
generator, review the resulting images for legibility and privacy, and commit
the changed PNGs in an ordinary pull request. The Sphinx/Read the Docs build
never installs Chromium.

To publish a successful screenshot run **without downloading the ZIP locally**,
use the separate **Publish reviewed documentation screenshots** workflow in
GitHub Actions: choose *Run workflow*, enter the numeric run ID from the
successful **Documentation screenshots** job, and select the review branch
(currently `docs/74-readthedocs`). The workflow verifies the run's origin,
success status, matching branch and image files, then commits only the five
PNG screenshots to that review branch. It never publishes directly to
`main`; the images enter `main` through normal PR review and merge.

**Important:** GitHub only exposes the manual Run workflow button after the
publication workflow file exists on the repository's default branch. Until
this PR is merged, use the local regeneration-and-commit method above or
invoke the workflow via the GitHub CLI once it is registered on `main`.
Only invoke publication after visually reviewing the screenshots in the
source run's artifact. If the source run is older than the current UI change,
regenerate first.

The generated PNGs are expected to be committed; if the first capture fails,
check the CI artifact `failure.png` and update selectors in the deterministic
browser scenario. Run from a fresh local environment before publishing changes
that alter the photographed UI.
