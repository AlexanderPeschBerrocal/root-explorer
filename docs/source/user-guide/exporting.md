# Exporting plots and reproducible code

ROOT Explorer provides two conceptually distinct mechanisms:

- **Download** transfers a plot through the browser to your local machine.
- **Save** writes a file on the **machine running the Streamlit server**,
  which may be a container or remote host rather than your laptop.

The export controls apply to the **last generated plot**, not to widget
changes that have not been followed by **Make plot** (or **Make event plot**).

## Supported formats

For static Matplotlib plots, select PNG, PDF or both, and a suitable
output resolution. The app can offer separate **Download PNG** and
**Download PDF** actions, as well as saving selected formats to a server
directory. For interactive Plotly 3D plots, export HTML.

PNG exports offer 150, 300 (default) or 600 DPI. The preview uses its own
resolution, so it need not match the exported PNG. If a PNG export is
estimated to exceed **40 million pixels**, **Download PNG** is disabled, and so
is **Save** while PNG is a selected format (no format is saved, including PDF);
reduce figure size or DPI. PDF keeps text and vector drawing
resolution-independent; the DPI setting only affects rasterised content inside
the PDF.

The HTML contains the figure data but loads Plotly JavaScript from a CDN.
It therefore needs network access to that CDN; it is not a fully self-contained
offline export. Interactive 3D plots do not offer standalone Python plot code.

Choose a writable directory for **Save**. Saving again with the same filename
and format overwrites the existing file; use distinct names for results you
want to preserve. For persistent container output, see
[Saving files from Docker](../installation.md#saving-files-from-docker).

## Standalone Python code

For supported static plots, **Download plot code** produces an editable
standalone Python script reproducing the selected source and plotting
configuration. It includes the chosen variables, labels, entry range,
applicable cuts, binning, histogram flow, normalisation, fitting and KDE
when those operations are enabled.

Input paths and non-structural styles can be edited in the generated
script. Regenerate it after changing plot type, variable modes, optional
features or logarithmic axis settings; otherwise the script may no
longer describe the current intended configuration. Inspect and execute
downloaded analysis code in an appropriate trusted environment.

Downloaded scripts may refer to the original ROOT file path. A path
inside a container or remote service is **not automatically valid**
on your own computer: update it to a local file accessible to Python.
Uploads may appear as temporary paths on the server; keep a copy of the
original ROOT file and replace that path in the script as well.

### Run a downloaded script

Use Python 3.12 and install the scientific packages in a virtual environment.
The script includes its selected helper functions and does not require
ROOT Explorer, Streamlit or PyROOT to be installed:

```bash
python3.12 -m venv .plot-venv
source .plot-venv/bin/activate
python -m pip install numpy awkward uproot hist matplotlib pandas scipy
python root_explorer_plot.py
```

Replace `root_explorer_plot.py` with the downloaded filename and use the
activation command appropriate to your shell. On a desktop, the final
`plt.show()` opens the Matplotlib figure. For a headless run, replace that
call with `plt.savefig("plot.png", dpi=300, bbox_inches="tight")` and run
with Matplotlib's non-interactive `Agg` backend, for example
`MPLBACKEND=Agg python root_explorer_plot.py` on macOS/Linux.

Keep package versions alongside the script for reproducibility; the command
above installs dependencies but does not lock their versions.

## Reproducible scientific analyses

Use ROOT Explorer for exploratory checks and plot prototyping; keep
final selections, calibrations and publication plots in version-controlled,
reproducible code and retain the input dataset provenance.
