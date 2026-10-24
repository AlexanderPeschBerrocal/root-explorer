# Plotting ROOT data

After opening a ROOT file and selecting a TTree or RNTuple, use the **Plotter** tab to build a plot from one or more branches. For a first example, follow [Your first plot](../getting-started.md).

```{image} ../_static/screenshots/plot-2d-light.png
:alt: 2D histogram in the ROOT Explorer Plotter, generated from synthetic ROOT data
:class: only-light
```

```{image} ../_static/screenshots/plot-2d.png
:alt: 2D histogram in the ROOT Explorer Plotter, generated from synthetic ROOT data
:class: only-dark
```

## Main plot types

ROOT Explorer supports **1D histograms**, **2D histograms**, **2D scatter**, and
**3D scatter interactive**. Interactive three-dimensional scatter plots can be
combined with a GDML geometry overlay when a suitable geometry file and
coordinates are available. Their **Aspect mode** (`data`, `cube` or `auto`) sets
the proportions of the 3D axes. Some options depend on the selected plot type.

A 1D histogram counts observations in bins (or accumulates weights if weighting is selected). The x-axis represents the selected variable. For jagged particle values, the plotted population may be **objects** rather than **events**; see [Variables and collections](variables-and-collections.md).

A 2D histogram bins paired values on two axes. A scatter plot shows individual pairs, but display-oriented point limits/downsampling can apply to large samples. Do not interpret a downsampled scatter cloud as necessarily containing every selected observation. When a scatter plot draws only a sample, a caption below the plot states how many of the points are shown; the summary statistics still use all of them.

## Make a plot

1. Choose the plot type.
2. Select the x variable, plus y and z where required, choosing direct branches or supported derived/reduced modes.
3. Configure any applicable cuts, weights, bin settings and plot style.
4. Click **Make plot** (shortcut **P** while the Plotter tab is open) to generate the output.

The **Style** and **Labels** sections, with figure size, axis scales and limits, grid, marker and colour settings, and title and label overrides, are collapsed by default; click a section to open it. Collapsing only hides the controls: their values stay active and are included in downloaded plot code. Everything that changes the plotted data or statistics, such as variables, cuts, weights, binning, normalisation and fits, stays visible.

Settings such as the number of bins, figure size, marker size and opacity are sliders by default. To enter an exact value, click the keyboard button next to a slider; it turns into a number input with the same limits, and the button next to it switches back. The value is kept when switching, and each setting remembers its mode for the session.

The plot area continues displaying the **most recently generated plot** until you explicitly generate another. Changing widgets without clicking **Make plot** does not retroactively update the displayed result; instead, a notice above the plot says that the settings have changed since it was made. The notice disappears once you make a new plot or return to the settings of the displayed one. Download settings such as the filename do not count as plot settings.

## Titles and labels

The plot title and axis labels can be overridden in the **Labels** section of the plot controls. Labels of static plots support mathematical notation through Matplotlib's built-in [mathtext](https://matplotlib.org/stable/users/explain/text/mathtext.html), so no LaTeX installation is required. Write the mathematical part between dollar signs:

| Label text | Displayed as |
| --- | --- |
| `$p_T$ [GeV]` | $p_T$ [GeV] |
| `$\eta$` | $\eta$ |
| `$\Delta R(\mu, j)$` | $\Delta R(\mu, j)$ |
| `$\frac{1}{N}\,\mathrm{d}N/\mathrm{d}p_T$` | $\frac{1}{N}\,\mathrm{d}N/\mathrm{d}p_T$ |

The same syntax works for the colour-bar label and for labels in the persisted histogram and graph viewers. A label containing invalid math, for example an unknown command such as `$\pt$` or an unclosed brace, is reported when you click **Make plot**, and no plot is generated until it is corrected. Downloaded figures and generated plot code use the same labels.

Interactive 3D scatter plots currently display labels literally, without rendering math.

## Sample summaries and correlations

The summary's count, minimum, median, mean, standard deviation and maximum
describe the finite sample after cuts. They are **unweighted**, including for
weighted histograms. Histogram ranges, flow settings and axis zoom do not
change them.
A separate table reports weighted summaries when histogram weights are selected;
see [Counts, weights and statistical uncertainty](histograms-and-fits.md#counts-weights-and-statistical-uncertainty).

2D scatter plots report **Pearson** correlation (linear association) and
**Spearman** correlation (association of ranks, with ties assigned average
ranks). Both use the full paired, finite sample after cuts, before scatter
downsampling, and are unweighted. Too few observations or constant coordinates
make a coefficient undefined. A correlation alone does not establish a physical
relationship or capture every nonlinear dependence.

Scatter plots apply the configured point cap using deterministic sampling
(the default cap is 100,000). In **Event inspection**, the data preview shows at most 500 plotted
observations. Neither the cloud nor its preview necessarily contains every
observation represented in the summary and correlation calculations.

## Axis limits, colours and logarithmic scales

Axis limits change the displayed window; they do not act as cuts. For scatter
colouring, an event-level value, including a per-event reduction such as
`N(Muon_pt)` or `sum(Jet_pt)`, is repeated for the objects of its event, while an
object-level colour field must match the plotted collection. Non-finite colours
exclude their observations: colouring by a mean, max, min or nth object hides the
objects of events whose collection is empty or too short, and a caption below the
colour selector points this out. Prefer multiplicity or sum when every object
should stay visible. Colour limits clip the colour mapping rather than
remove points: values outside the limits use the end colours.
If only one colour limit is entered for a 2D or 3D scatter plot, the other
endpoint comes from the minimum or maximum of all selected colour values, also
when only a sample of the points is drawn. The
resulting minimum must be smaller than the maximum; otherwise the plot is
refused with a message that states the automatic range.

Static plots offer logarithmic scales where applicable. For scatter and
event-derived 2D histograms, logarithmic coordinate axes require positive
plotted coordinate values; zooming to positive limits does not repair
non-positive inputs. A 1D histogram instead requires positive nominal bin edges
on a logarithmic x-axis, so out-of-range flow values do not invalidate a
positive histogram range. A logarithmic histogram yield axis needs
positive bin contents to display and cannot faithfully show zero or negative
yields. Logarithmic 2D histogram colours mask non-positive bins; logarithmic
scatter colours require all plotted colour values to be positive. Check the
linear view when using signed weights. The interactive 3D plot does not offer
the static plots' logarithmic-axis controls.

## Interpretation

**Selected entry range, cuts, finite-value filtering, weights, flow-bin handling and normalisation** can change the statistical meaning of the displayed histogram. A histogram density, weighted yield, and simple count distribution are not interchangeable. For precise definitions of weighted yields, statistical uncertainties, density normalisation and fit objectives, see [Histogram statistics, normalisation and fitting](histograms-and-fits.md).

For threshold/quantile calculations, KDE bandwidth and sampling, and the
distinction between histogram-content contours and confidence regions, see
the histogram guide above. For fields with missing values, see
[Missing values](variables-and-collections.md#missing-values); for files replaced
at an existing path, see
[Caching and changed files](../performance.md#caching-and-changed-files).

Static plots can be downloaded and selected plotting configurations can be exported as standalone Python code. Retain a reproducible script for final physics results.

Next: [Variables and collections](variables-and-collections.md) or [ROOT objects](files-and-objects.md).
