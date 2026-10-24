# Variables and particle collections

Event-oriented ROOT data commonly mix **one value per event** and **a varying number of particle values per event**. Choosing the appropriate interpretation is essential for correct plots.

## Event-level scalars

Fields such as `MET_pt`, `nMuon`, and `genWeight` normally have one value per event. A histogram of `MET_pt` represents the selected events (subject to cuts and other transformations).

## Jagged collections

A field such as `Muon_pt` normally contains a list of muon transverse momenta *within each event*. The list length can vary across events, including zero.

If you plot direct jagged branch values in a histogram, ROOT Explorer **flattens** the lists and plots object-level observations. An event with three selected muons can contribute three histogram entries, while an event with none contributes no muon entries. This is not the same as a histogram with exactly one entry per event.

For event-level observables derived from a collection, the Plotter's axis, colour and cut selectors offer:

| Mode | Meaning | Empty collection |
| --- | --- | --- |
| Multiplicity per event | Number of objects | 0 |
| nth object per event | Object at the specified **zero-based** position, in stored order | Missing (`NaN`) |
| Sum per event | Sum of the collection's values | 0 |
| Mean per event | Arithmetic mean | `NaN` |
| Max / Min per event | Largest / smallest value | `-inf` / `+inf` |

Non-finite results are excluded from the plotted sample. Consequently, a sum
histogram can include an empty event at zero while a mean histogram omits it.
`nth` does not sort objects: index 0 is the first stored object, not necessarily
the highest-momentum object. An out-of-range index is missing. For an event-level
scalar, index 0 returns that scalar and later indices are missing.

**Reductions happen before cuts**, on the original collection in the selected
read range. For example, summing `Muon_pt` and then cutting on `MET_pt` selects
events with their original muon sums; it does not first remove low-momentum muons
and sum the survivors. The interface does not provide a chained object-filter,
then event-reduction workflow. Event reductions are not offered in the
single-event plot of **Event inspection**, where a reduction would be one constant
for the displayed event.

Sum, mean, extrema and nth-object reductions require scalars or a single list of
scalars per event. Multiply nested lists are rejected with an explanation;
choose a scalar leaf field instead. Multiplicity counts the outer collection's
objects. Direct numeric nested values can still be flattened for plotting when
their structures align.

## Derived variables

The selectors offer the following derived operations. Inputs must describe
matching observations: combining two object branches requires matching per-event
collection lengths, not merely equal total lengths. Operations pair corresponding
objects; they do not construct all pairs or match particles between collections.
Plot axes likewise need matching structures; a scalar axis is not automatically
repeated for a jagged axis.

| Operation (selector label) | Definition / convention |
| --- | --- |
| Absolute value (`\|A\|`) | $\lvert a\rvert$ |
| 3D magnitude (`sqrt(A² + B² + C²)`) | $\sqrt{x^2+y^2+z^2}$ |
| Transverse radius (`r = sqrt(x² + y²)`) | $\sqrt{x^2+y^2}$ |
| Azimuth from x/y (`phi = atan2(y, x)`) | $\operatorname{atan2}(y,x)$, in radians |
| Pseudorapidity from x/y/z (`eta = asinh(z / sqrt(x²+y²))`) | $\operatorname{asinh}(z/\sqrt{x^2+y^2})$; undefined at zero transverse radius |
| Massless energy (`Energy (massless)`) | $p_T\cosh(\eta)$, or $p_T\cosh(y)$ using rapidity |
| Massive energy (`Energy (with mass)`), longitudinal coordinate `eta (pseudorapidity)` | $\sqrt{m^2+(p_T\cosh(\eta))^2}$ |
| Massive energy (`Energy (with mass)`), longitudinal coordinate `y (rapidity)` | $\sqrt{m^2+p_T^2}\cosh(y)$ |
| Invariant mass from E/px/py/pz (`Invariant mass (single particle)`) | $\sqrt{\max(E^2-p_x^2-p_y^2-p_z^2,0)}$; negative mass squared is clipped to zero |
| Ratio / difference / sum / product (`A / B`, `A - B`, `A + B`, `A * B`) | $a/b$, $a-b$, $a+b$, $ab$; division by zero gives `NaN` |
| Delta phi (`delta phi`) | Wrapped $\phi_1-\phi_2$ in $(-\pi,\pi]$ |
| Delta eta (`delta eta`) | $\eta_1-\eta_2$ |
| Delta R (`delta R`) | $\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$ using wrapped delta phi |

Use consistent momentum, mass and energy units; no unit conversion is performed.
Angles are in radians. The invariant-mass mode operates on the supplied
four-vector components; it does not add two particles' four-vectors for you.
Variable selectors do not accept arbitrary Python expressions.

## Missing values

RNTuple fields can contain missing (optional) values. ROOT Explorer keeps every
missing value at its position until all participating values are aligned, so a
value is never paired with a value from a different event or object:

- A derived variable, such as `A - B`, is only computed where all its inputs are
  present. If `A` is missing in one event and `B` in another, neither event
  contributes a value.
- When plotting, an event or object is excluded if any plotted variable, weight,
  colour or cut value is missing or non-finite. The number of excluded
  observations is counted before range cuts, once per event or object even if
  several inputs are invalid. It is shown below the summary table; ordinary
  range-cut failures are not included in that count. A clear message appears if
  no values remain. Exported Python scripts apply the same selection and reject
  empty selections with the same explanation.
- Per-event reductions skip missing values inside a collection, so the mean of
  `[1, missing]` is 1. A missing whole collection gives a missing result, which
  is excluded, whereas an empty collection keeps its documented value, for
  example a multiplicity and sum of 0.
- Event rankings keep the original event number for every value; events whose
  metric is missing are not listed.

Missing values are never replaced by zero.

## Cuts and weights

Set **Number of cuts** (up to 20) to add range cuts. Range cuts are combined with logical **AND**, with inclusive lower and upper
bounds (`minimum <= value <= maximum`). Leave a bound blank for an open end.
An invalid or non-finite bound, or a minimum greater than the maximum, prevents
plot generation until corrected; it never silently becomes an open bound.
Non-finite cut values fail the cut. Event-level cuts, such as `MET_pt` or a
per-event reduction such as `max(Muon_pt)`, are repeated for all plotted objects
in each event, so they keep or reject the objects of an event together. For
example, plotting `Jet_pt` with the cut `max(Muon_pt) >= 50` keeps every jet of
the events whose highest muon `pt` is at least 50. A mean, max, min or nth object
has no value for an event with an empty or too short collection, so such events
fail the cut. Object-level cuts must match the plotted collection's per-event
structure. A cut on one particle collection does not
implicitly select or match objects in a different collection.

A common finite-value mask removes observations with non-finite coordinates,
weights, scatter colours or cut values before applying range cuts. Histogram ranges and axis limits are
separate controls, not additional physics selections.

For weighted histograms, an event-level `genWeight` can be broadcast to each selected object in that event; object-level weights can be used when they match the plotted observations. Thus a weighted object histogram sums **object contributions with their associated weights**, which need not equal the number of events.

Negative generator weights can produce signed bin yields. For the interpretation of uncertainties, flow bins and density normalisation, consult [Histogram statistics, normalisation and fitting](histograms-and-fits.md).

For selector structure badges, turn on **Show branch details in selectors**
for a TTree, or **Show field details in selectors** for an RNTuple. Inspect
whether inputs are scalar, fixed-size or jagged before combining them.

For fields with optional/missing values, see [Missing values](#missing-values).

Continue with [Plotting](plotting.md).
