# ROOT files, event containers and persisted objects

A ROOT file may contain multiple kinds of objects. ROOT Explorer exposes two related but distinct workflows:

- **Event-data exploration:** Select a TTree or RNTuple containing entries/events, choose a read range and plot branches (fields) from those entries.
- **Persisted-object inspection:** Browse a ROOT file's stored objects and plot the supported 1D/2D histograms, profiles and graphs. These objects are not created afresh from event branches by ROOT Explorer.

```{image} ../_static/screenshots/object-browser-light.png
:alt: ROOT Explorer file-object and branch browser with synthetic data
:class: only-light
```

```{image} ../_static/screenshots/object-browser.png
:alt: ROOT Explorer file-object and branch browser with synthetic data
:class: only-dark
```

## Choose a TTree or RNTuple

Open a ROOT file using the sidebar's **Path** or **Upload** input. If the file contains event data, choose an **Event container**. Select **entry start** and **entry stop** to define the half-open interval of entries to inspect (the stop value is exclusive).

A typical NanoAOD-like `Events` tree has scalar event branches such as `MET_pt` or `nMuon`, as well as jagged particle fields such as `Muon_pt`. See [Variables and collections](variables-and-collections.md) for interpretation.

Use the tab containing event-container fields to inspect names, types and per-entry structure. ROOT Explorer distinguishes scalars, fixed-size fields, jagged fields and nested/record structures; not every source type is directly plottable.

## Browse stored ROOT objects

Use the **ROOT objects** browser to view objects and their directory paths without immediately loading all their contents. Search/filter the catalogue, select an inspectable object and click **Open selected object**. The fallback **Object to inspect** selector can be used when browser row selection is unavailable.

ROOT Explorer can plot the following stored ROOT classes:

- **1D histograms:** `TH1` classes such as `TH1F`, `TH1D` and `TH1I`.
- **2D histograms:** `TH2` classes such as `TH2F`, `TH2D` and `TH2I` (`TH2Poly` is not supported).
- **Profiles:** `TProfile` and `TProfile2D`.
- **Graphs:** `TGraph`, `TGraphErrors` and `TGraphAsymmErrors`.

Other classes are listed in the browser with their ROOT class name but cannot be plotted. Specific capabilities depend on the object's ROOT class and dimensionality: a TProfile contains per-bin means, **not** event-count histogram yields. Histogram-only operations should not be assumed valid for profiles or graphs.

A ROOT file that contains **no TTree or RNTuple** can still be opened in the
ROOT-object browser. The same applies when the selected TTree or RNTuple has
**zero entries** or cannot be read: ROOT Explorer shows a warning in the sidebar,
omits the event-plotting and event-inspection tabs, and keeps the ROOT-object
browser and the Geometry page available. If the file contains another, non-empty
event container, select it in **Event container** to continue with the normal
workflow. Plots and rankings created for a previously selected container are
cleared when you switch to an empty one.

Individual branches or fields can be unreadable even when the file opens, for
example because a data block is corrupted or uses an unsupported type. If a plot,
event plot or ranking needs such a branch, ROOT Explorer reports which branch or
field of which container and file failed, and for which entries or event. The
original error is available under **Technical details**. The previous plot or
ranking is removed rather than shown as if it were current, and other branches
can still be used.

Objects may reside in nested ROOT directories. The browser shows the latest
cycle of each object by default; enable **Show all ROOT cycles** to inspect
older stored versions too. Selecting a row does not itself open a viewer:
use **Open selected object**. Opening a supported object generates its initial
plot; after changing its controls, click **Make plot** to update it.

## What is loaded?

Browsing object metadata is different from opening the object or loading data for a plot. Use entry ranges for large event datasets and keep object inspection deliberate to reduce unnecessary memory use.

Next: [Plotting](plotting.md).
