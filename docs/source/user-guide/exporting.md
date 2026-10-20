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

## Reproducible scientific analyses

Use ROOT Explorer for exploratory checks and plot prototyping; keep
final selections, calibrations and publication plots in version-controlled,
reproducible code and retain the input dataset provenance.
