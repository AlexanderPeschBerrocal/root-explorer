# Variables and particle collections

Event-oriented ROOT data commonly mix **one value per event** and **a varying number of particle values per event**. Choosing the appropriate interpretation is essential for correct plots.

## Event-level scalars

Fields such as `MET_pt`, `nMuon`, and `genWeight` normally have one value per event. A histogram of `MET_pt` represents the selected events (subject to cuts and other transformations).

## Jagged collections

A field such as `Muon_pt` normally contains a list of muon transverse momenta *within each event*. The list length can vary across events, including zero.

If you plot direct jagged branch values in a histogram, ROOT Explorer **flattens** the lists and plots object-level observations. An event with three selected muons can contribute three histogram entries, while an event with none contributes no muon entries. This is not the same as a histogram with exactly one entry per event.

For event-level observables derived from a collection, the Plotter's axis selectors offer:

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
then event-reduction workflow. Event reductions are not offered in the cut or
scatter-colour selectors, or in single-event plot axes.

## Derived variables

The selectors offer the following derived operations. Inputs must describe
matching observations: combining two object branches requires matching per-event
collection lengths, not merely equal total lengths. Operations pair corresponding
objects; they do not construct all pairs or match particles between collections.
Plot axes likewise need matching structures; a scalar axis is not automatically
repeated for a jagged axis.

| Operation | Definition / convention |
| --- | --- |
| Absolute value | $\lvert a\rvert$ |
| 3D magnitude | $\sqrt{x^2+y^2+z^2}$ |
| Transverse radius | $\sqrt{x^2+y^2}$ |
| Azimuth from x/y | $\operatorname{atan2}(y,x)$, in radians |
| Pseudorapidity from x/y/z | $\operatorname{asinh}(z/\sqrt{x^2+y^2})$; undefined at zero transverse radius |
| Massless energy | $p_T\cosh(\eta)$, or $p_T\cosh(y)$ using rapidity |
| Massive energy from pseudorapidity | $\sqrt{m^2+(p_T\cosh(\eta))^2}$ |
| Massive energy from rapidity | $\sqrt{m^2+p_T^2}\cosh(y)$ |
| Invariant mass from E/px/py/pz | $\sqrt{\max(E^2-p_x^2-p_y^2-p_z^2,0)}$; negative mass squared is clipped to zero |
| Ratio / difference / sum / product | $a/b$, $a-b$, $a+b$, $ab$; division by zero gives `NaN` |
| Delta phi | Wrapped $\phi_1-\phi_2$ in $(-\pi,\pi]$ |
| Delta eta | $\eta_1-\eta_2$ |
| Delta R | $\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$ using wrapped delta phi |

Use consistent momentum, mass and energy units; no unit conversion is performed.
Angles are in radians. The invariant-mass mode operates on the supplied
four-vector components; it does not add two particles' four-vectors for you.
Variable selectors do not accept arbitrary Python expressions.

## Cuts and weights

Range cuts are combined with logical **AND**, with inclusive lower and upper
bounds (`minimum <= value <= maximum`). Leave a bound blank for an open end.
Non-finite cut values fail the cut. Event-level scalar cuts can be broadcast
to all plotted objects in each event; object-level cuts must match the plotted
collection's per-event structure. A cut on one particle collection does not
implicitly select or match objects in a different collection.

After cuts, a common finite-value mask removes observations with non-finite
coordinates, weights or scatter colours. Histogram ranges and axis limits are
separate controls, not additional physics selections.

For weighted histograms, an event-level `genWeight` can be broadcast to each selected object in that event; object-level weights can be used when they match the plotted observations. Thus a weighted object histogram sums **object contributions with their associated weights**, which need not equal the number of events.

Negative generator weights can produce signed bin yields. For the interpretation of uncertainties, flow bins and density normalisation, consult [Histogram statistics, normalisation and fitting](histograms-and-fits.md).

For selector structure badges, turn on **Show branch details in selectors**
for a TTree, or **Show field details in selectors** for an RNTuple. Inspect
whether inputs are scalar, fixed-size or jagged before combining them.

Optional/missing source values currently have an alignment limitation; consult
[Known correctness limitations](../performance.md#known-correctness-limitations)
before combining or ranking fields containing missing values.

Continue with [Plotting](plotting.md).
