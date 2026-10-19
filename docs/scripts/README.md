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

The separate `Documentation screenshots` GitHub Actions workflow also runs
automatically for relevant PRs and on `workflow_dispatch`, and uploads the
resulting PNGs for review. It does **not** automatically commit images on
untrusted PRs or perform fragile pixel equality checks. On the trusted
`docs/74-readthedocs` branch, the first baseline is committed automatically
by the push workflow; subsequent screenshot refreshes should use a reviewed PR. After regenerating,
review the images for legibility and privacy, commit the reviewed PNGs, and
submit them in the same documentation/application PR. The ordinary Sphinx/RTD
build never installs Chromium.

The generated PNGs are expected to be committed; if the first capture fails,
check the CI artifact `failure.png` and update selectors in the deterministic
browser scenario. Run from a fresh local environment before publishing changes
that alter the photographed UI.
