# Variables and particle collections

Event-oriented ROOT data commonly mix **one value per event** and **a varying number of particle values per event**. Choosing the appropriate interpretation is essential for correct plots.

## Event-level scalars

Fields such as `MET_pt`, `nMuon`, and `genWeight` normally have one value per event. A histogram of `MET_pt` represents the selected events (subject to cuts and other transformations).

## Jagged collections

A field such as `Muon_pt` normally contains a list of muon transverse momenta *within each event*. The list length can vary across events, including zero.

If you plot direct jagged branch values in a histogram, ROOT Explorer **flattens** the lists and plots object-level observations. An event with three selected muons can contribute three histogram entries, while an event with none contributes no muon entries. This is not the same as a histogram with exactly one entry per event.

For event-level observables derived from a collection, use the appropriate variable mode:

- **Multiplicity per event:** the number of objects in the event.
- **nth object per event:** a specified position in each event's collection, when available.
  For an event-level scalar, index 0 returns that scalar and later indices are missing.
- **Sum / Mean / Max / Min per event:** reduce selected object values to an event-level quantity.

These operations carry different missing/empty-collection semantics; check the app's explanatory labels and resulting statistics rather than silently treating them as interchangeable.

## Derived variables

The plot variable selectors offer direct field values and derived physical or mathematical quantities where the source branches permit them, including vector magnitudes, angle-related quantities, energy/invariant mass options, and combinations of coordinates. The available mode list and required inputs depend on the selected variable and plot type; ROOT Explorer does not evaluate arbitrary untrusted Python expressions as branch selectors.

## Cuts and weights

Range cuts restrict which values are selected. In the current interface, supported range cuts are combined with logical **AND**. Do not assume that a future event/object-specific chained filter engine is already present.

For weighted histograms, an event-level `genWeight` can be broadcast to each selected object in that event; object-level weights can be used when they match the plotted observations. Thus a weighted object histogram sums **object contributions with their associated weights**, which need not equal the number of events.

Negative generator weights can produce signed bin yields. For the interpretation of uncertainties, flow bins and density normalisation, consult [Histogram statistics, normalisation and fitting](histograms-and-fits.md).

For supported selector structure badges, turn on **Show entry details in selectors** (the word *entry* changes to the selected container's field type) and inspect whether values are tagged scalar, jagged or another structure.

Continue with [Plotting](plotting.md).
