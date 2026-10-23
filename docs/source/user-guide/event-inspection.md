# Event inspection and ranking

Event inspection is distinct from the Plotter's aggregation across an
entry range. The **Event inspection** tab shows the selected event and its
objects, allowing you to investigate individual outliers and event topology.

```{image} ../_static/screenshots/event-inspection-light.png
:alt: ROOT Explorer event inspection with synthetic data
:class: only-light
```

```{image} ../_static/screenshots/event-inspection.png
:alt: ROOT Explorer event inspection with synthetic data
:class: only-dark
```

## Select an event

1. Open a ROOT file containing a TTree or RNTuple and choose an event
   container and entry range.
2. Open **Event inspection**. Use the event-index control to select an event
   inside the chosen range.
3. Choose an appropriate plot type and coordinate variables (for example,
   muon $p_T$ and $\eta$) and click **Make event plot**.
4. Change the selected entry and regenerate the plot to inspect another
   event. Do not mistake a previous plot for a newly selected event.

## Rank events

Use the ranking controls in Event inspection to find unusually active
or interesting events without manually scanning every entry.

1. Select a numerical field, for example `Muon_pt`.
2. Choose a supported ranking measure, such as multiplicity, sum,
   extremum or count above a threshold.
3. Select **largest first** if you want the highest-scoring events.
4. Click **Compute ranking**, choose an event from the result, and select
   **Use selected event**.

Ranking evaluates the currently selected entry range and is intended
for exploratory investigation, not a substitute for a documented
physics event selection. A jagged field's ranking measure may be a
reduction over particles rather than a scalar per-event value.

Available ranking measures are multiplicity, sum, mean, maximum, minimum,
count above a threshold and count below a threshold. Threshold counts use
strict `>` and `<` comparisons. The absolute-value option applies before
calculating the measure (and has no effect on multiplicity).

An empty collection has multiplicity, sum and threshold counts of zero;
its mean and extrema are non-finite and excluded from ranking. For a scalar
field, multiplicity is one, threshold counts are zero or one, and other
measures use the scalar value, optionally made absolute.

Ranking uses the sidebar's read range but **does not apply Plotter cuts or
histogram weights**. Recompute it after changing the ranking settings.
Selecting a ranked event does not regenerate an event plot automatically:
click **Make event plot** after **Use selected event**.

## Practical considerations

The UI displays entry indices in the selected container. These are not
necessarily physics event identifiers (`run`/`lumi`/`event`). Keep the
distinction clear when transferring an interesting event into a real
analysis. Working with smaller read ranges can reduce computation time.
Rankings keep the original entry index for every value; entries whose metric
is missing are not listed (see
[Missing values](variables-and-collections.md#missing-values)).
