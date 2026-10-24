# Histogram statistics, normalisation and fitting

This chapter covers histograms **computed from events**, and where explicitly
noted, persisted ROOT histograms. Histogram operations affect physical and
statistical interpretation: a displayed bin height is not always a count.

```{image} ../_static/screenshots/histogram-fit-light.png
:alt: Histogram fit displayed in ROOT Explorer, generated with synthetic ROOT data
:class: only-light
```

```{image} ../_static/screenshots/histogram-fit.png
:alt: Histogram fit displayed in ROOT Explorer, generated with synthetic ROOT data
:class: only-dark
```

## Counts, weights and statistical uncertainty

An unweighted bin yield counts selected observations; for a jagged branch,
these can be *objects*, rather than *events*. For weighted yields, the bin
content is the sum of weights, $\sum_i w_i$, and stored per-bin variance
is typically $\sum_i w_i^2$, not $\sum_i w_i$.
Negative generator weights allow negative bin sums. The resulting yield
need not be a probability distribution; never apply ordinary Poisson
count formulas indiscriminately to weighted histograms.

The summary table above the plot describes the selected sample after cuts and
finite-value filtering; it is not redefined by histogram range, zoom, or flow
display settings. Its `n`, `min`, `median`, `mean`, `std` and `max` columns are
**unweighted**. When a weight branch is selected (weights are available for 1D
and 2D histograms in the Plotter and in Event inspection), a second table below it
shows weighted statistics of the plotted values:

- **sum of weights** $\sum_i w_i$;
- **N_eff**, the effective number of entries $(\sum_i w_i)^2 / \sum_i w_i^2$,
  which equals the number of entries for uniform weights and decreases as the
  weights become more uneven;
- **weighted mean** $\sum_i w_i x_i / \sum_i w_i$;
- **weighted std** $\sqrt{\sum_i w_i (x_i - \bar{x}_w)^2 / \sum_i w_i}$, the
  same population convention as ROOT uses for weighted histograms.

For object-level values with an event-level weight, each object carries the
weight of its event. Negative weights are reported in a caption. The weighted
mean and standard deviation are shown as unavailable when the sum of weights is
not positive, and the weighted standard deviation is unavailable when negative
weights make the weighted variance negative. If your physics selection changes,
regenerate the plot.

## Histogram range and flow bins

For an event-derived 1D histogram, enable **Custom histogram range and flow
handling** to set an optional histogram minimum and maximum for the binning
range. A blank bound is inferred from the selected sample. The
**Underflow / overflow** choice and the flow summary apply only while this
option is enabled and at least one bound is entered; otherwise flow values are
ignored and no flow summary is shown.
Values below the minimum are underflow and values above the maximum are overflow;
the maximum itself belongs to the last nominal bin. Persisted ROOT histograms
retain their stored bin contents and flow classification.

- **Ignore (report separately):** draw and fit only nominal in-range bins.
- **Fold into edge bins:** add underflow to the first bin and overflow to the last, including
  their variances. These adjusted edge bins also participate in fitting.
- **Show separate bins:** display flow categories alongside nominal bins.
  The extra categories do not represent finite physical intervals and are
  excluded from fitting.

The flow summary reports the original out-of-range yields even when they are
ignored or folded in the plot. Axis limits only change the view; they neither
rebin nor filter the sample.

## Thresholds and quantiles

For event-derived 1D histograms, enable **Calculate threshold / quantile**.
The calculation uses the full finite sample after cuts, including observations
outside the histogram range. Changing flow handling, density mode or axis zoom
does not change the result.

A threshold divides observations into **x ≤ threshold** and **x > threshold**.
The table reports counts and, when weights are selected, weighted sums and
fractions. Signed-weight fractions can lie outside [0, 1]; a zero or numerically
unstable total weight makes weighted fractions undefined.

Unweighted quantiles use NumPy's linearly interpolated sample quantile.
Weighted quantiles use the inverse empirical cumulative distribution: the first
sorted value whose cumulative weight reaches the requested fraction, without
interpolation. For example, the median of `[0, 10]` is 5 unweighted, but 0 with
equal positive weights. Weighted quantiles are unavailable if any selected
weight is negative or the total weight is zero. Empty samples have no quantile.
Zero-weight observations do not influence a weighted quantile, including its
0% and 100% endpoints, which are the smallest and largest positively weighted
values. Quantile and threshold fractions use scaled weights internally to avoid
overflow from an otherwise harmless change of weight units. Raw sums can still
exceed floating-point range for extreme inputs.

## Unit-area density normalisation

For event-derived **1D histograms**, enable **Normalise to unit area** under
Histogram analysis. The displayed histogram is a bin-width-corrected density:

$$
d_i = \frac{\sum_{j \in B_i} w_j}{S\,\Delta x_i},
\qquad
\sum_i d_i\,\Delta x_i = 1.
$$

where $B_i$ is the set of observations in bin $i$, $\Delta x_i$ is its
width and $S$ is the sum of weights in the chosen normalisation domain.

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

For a fixed denominator $S$, stored variances are rescaled by
$1/(S\,\Delta x_i)^2$. Since the denominator is estimated from the
same data, these diagonal variances do not capture the full covariance of a
self-normalised distribution. They must not be treated as independent
statistical uncertainties in shape-only fits or tests.

Histogram fits continue to operate on the original flow-adjusted bin yields;
their displayed predictions are divided by `S * bin_width[i]` in density mode.
Fit parameters and goodness-of-fit statistics therefore do not change when
the density display is toggled. KDE overlays use density units when enabled;
their visible area need not be one because the underlying smooth distribution
can extend beyond the displayed range. KDE is unavailable for normalised
flow-folded histograms.

**Download plot code** reproduces the chosen density, flow, fit and KDE
semantics. Currently, the control applies to event-derived 1D histograms.
Persisted ROOT `TH1*` objects can contain counts, densities or other
quantities without reliable semantic metadata, and are not normalised
automatically. `TProfile` and `TProfile2D` bin means must not be treated as yields.

## Built-in histogram fitting

For a 1D histogram, select **Enable fit** in the histogram fit controls and choose a **Fit model**:
**Gaussian**, **Exponential**, **Crystal Ball** (single-sided), **Double-sided Crystal Ball**
or **Custom expression** (see below).
This is available for histograms built from TTree/RNTuple branches and
persisted `TH1*` ROOT objects. Fitting of `TProfile` bin means is
not yet supported.

ROOT Explorer uses **binned least squares**, not a Poisson likelihood.
Ordinary least squares assigns equal importance to every selected bin.
**Use stored bin variances (weighted least squares)** instead minimises the
sum of squared residuals divided by each bin's variance (`sum(w²)`). In the
Plotter and Event inspection it is enabled by default when a weight branch is
selected and disabled otherwise; for stored `TH1*` objects it is disabled by
default. Bins with non-finite or non-positive variances (for example empty
bins) are excluded and reported; they do not affect starting parameter
estimates.

For a stored `TH1*` object, the fit uses the same per-bin errors as the drawn
error bars: its stored `sum(w²)`, or for a histogram saved without `Sumw2`,
ROOT's default errors $\sqrt{N}$ of the bin contents. Unweighted histograms built
from event branches store no variances, and no Poisson uncertainties are inferred
from their counts: variance-weighted fitting then fails with a warning, so
disable **Use stored bin variances** to use ordinary least squares.

Predictions are **integrals of the fit model over the original bins**
rather than evaluations at bin centres. The custom fit interval selects
whole bins whose centres fall inside it. With **Fold**, flow yields contribute
through the adjusted edge bins; with **Ignore** or **Show separate bins**, they
do not contribute. Parameter guesses and physically meaningful default bounds
(for example `A ≥ 0`, `sigma > 0`, `alpha` in [0.001, 100] and `n` in
[1.0001, 10⁴]) are provided automatically; both can be overridden under
**Override initial parameters and bounds**. If you enter only a lower or only an
upper bound, the default bound on the other side is kept. An invalid entry in a
fit range or parameter field is marked with a warning next to the field, and the
fit then fails with a message while the histogram is still drawn; `inf` and
`-inf` are accepted as bounds. Fit failures display
a warning without removing the histogram.

The output table reports parameters and approximate local covariance-based
uncertainties where identifiable. Ordinary least squares reports the
sum of squared residuals, not a chi-squared p-value. Variance-weighted
least squares reports a chi-squared statistic whose interpretation depends
on the validity of the supplied variances. Fits are also reproduced by
**Download plot code** when enabled.

## Custom mathematical histogram fits

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
An expression may contain at most 512 characters and 12 fit parameters.
Python statements, imports, attribute access, indexing, and arbitrary function
calls are forbidden. Custom models are integrated over the original bins using
fixed-order numerical quadrature, whereas built-in models retain their analytic
bin integrals. Extremely narrow peaks or rapidly oscillating expressions within
one bin can exceed the numerical accuracy of the quadrature. Fit failures leave
the underlying histogram visible. Exported standalone plot code includes the
same expression, parameter settings, and restricted expression evaluator.

## KDE smoothing and 2D contours

Event-derived 1D and 2D histograms can show a Gaussian kernel density estimate
(KDE). **Scott** and **Silverman** choose bandwidth automatically; **Custom factor**
sets SciPy's covariance bandwidth factor directly, not a multiplier of Scott's
choice. A smaller factor produces less smoothing.

KDE uses at most **5,000 observations**, selected by deterministic sampling
when necessary, while the histogram uses the full selected sample. Selected
weights are retained for KDE, but negative weights or a zero total weight
prevent it. Too few observations, constant coordinates or a singular covariance
can also prevent KDE; the histogram remains available with a warning.

For a 1D histogram, KDE uses observations inside the histogram range, even when
flow yields are folded. It is scaled to yield per bin in count mode and to
density in unit-area mode. The curve is drawn over the histogram range without
renormalising its visible area. Normalised flow-folded histograms cannot show
KDE. A 2D KDE is drawn as dashed density contours; these are distinct from the
histogram-content contours below.

For a 2D histogram, **Show contours** accepts percentages such as 68, 95 and 99.
These enclose histogram content by accumulating bins from highest content to
lowest. They are **not parameter-confidence regions** or fixed numbers of
Gaussian standard deviations. They use the full binned sample, not the KDE
subsample. Negative bin contents prevent this calculation; coarse binning and
ties can merge requested levels or prevent a contour from being drawn. A level
that encloses only the peak bin or every bin cannot be drawn and is named in a
warning, for example "Contour levels 10%, 30% cannot be drawn: they enclose only
the peak bin."; the other levels are still drawn. Adjust the binning and check
warnings when interpreting the overlay.

## Standalone plot code

The **Download plot code** action creates a standalone, editable Python script
for the selected static plot. It contains only the analysis and rendering
operations selected at export time. Input paths, branch names, labels, ranges,
and non-structural styles can be edited directly; regenerate the script after
changing the plot type, variable modes, optional features, or logarithmic scales.
