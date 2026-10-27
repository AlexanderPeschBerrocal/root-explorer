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

Open a ROOT file using the sidebar's **Path** or **Upload** input (uploads accept `.root` files). If the file contains event data, choose an **Event container**. Select **entry start** and **entry stop** to define the half-open interval of entries to inspect (the stop value is exclusive). By default, **entry stop** is the number of entries or 500,000, whichever is smaller.

A typical NanoAOD-like `Events` tree has scalar event branches such as `MET_pt` or `nMuon`, as well as jagged particle fields such as `Muon_pt`. See [Variables and collections](variables-and-collections.md) for interpretation.

Use the **Branches and objects** tab (**Fields and objects** for an RNTuple) to inspect names, types and per-entry structure; the branch or field table follows the ROOT file contents. ROOT Explorer distinguishes scalars, fixed-size fields, jagged fields and nested/record structures; not every source type is directly plottable.

Variable selectors exclude fields known to contain strings or record objects.
If a non-empty container has no supported numeric fields, its metadata and ROOT
objects remain browseable; the Plotter and Event inspection explain why plotting
and ranking are unavailable.

## Browse stored ROOT objects

The **ROOT file contents** browser lists objects and their directory paths without loading their contents. It is at the top of the **Branches and objects** tab (**Fields and objects** for an RNTuple), or in the **ROOT objects** tab when the file has no usable event container. Use **Filter ROOT objects** to search the catalogue, select an inspectable object and click **Open selected object** (shortcut **O**). The fallback **Object to inspect** selector can be used when browser row selection is unavailable. **Close inspection** (shortcut **Esc**) returns to the full catalogue.

ROOT Explorer can plot the following stored ROOT classes:

- **1D histograms:** `TH1` classes such as `TH1F`, `TH1D` and `TH1I`.
- **2D histograms:** `TH2` classes such as `TH2F`, `TH2D` and `TH2I` (`TH2Poly` is not supported).
- **Profiles:** `TProfile` and `TProfile2D`.
- **Graphs:** `TGraph`, `TGraphErrors` and `TGraphAsymmErrors`.

Other classes are listed in the browser with their ROOT class name but cannot be plotted. Classes whose names start with `TH3`, `TH2Poly`, `TProfile3D` or `TGraph` but are not listed above, such as `TH3D`, `TGraph2D` or `TGraphMultiErrors`, can be selected and opened, but show an unsupported-class warning instead of a plot. Specific capabilities depend on the object's ROOT class and dimensionality: a TProfile contains per-bin means, **not** event-count histogram yields. Histogram-only operations should not be assumed valid for profiles or graphs.

A ROOT file that contains **no TTree or RNTuple** can still be opened in the
ROOT-object browser. The same applies when the selected TTree or RNTuple has
**zero entries** or cannot be read, or when **entry stop** is not larger than
**entry start**: ROOT Explorer shows a message in the sidebar (information for a
file without event containers, an error for an unreadable container, otherwise a
warning). Only the **ROOT objects** and
**Geometry** tabs remain; **Plotter**, **Event inspection** and **Help** are omitted. If the file contains another, non-empty
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

The stored-object viewers offer the following controls:

- **Histograms and profiles:** **Figure width** and **Figure height**; for 1D
  objects a **Drawing mode** of **Step**, **Error bars**, **Step + error bars** or
  **Filled** (profiles default to **Error bars**, other histograms to **Step**); for
  2D objects the colour-scale controls; **Log x-axis**, **Log y-axis**, **Grid**,
  **Apply manual axis limits** and the labels. 1D histograms (not profiles) also
  offer the fit controls described in
  [Histograms and fits](histograms-and-fits.md#built-in-histogram-fitting).
- **Graphs:** a **Drawing mode** of **Markers**, **Line** or **Line + markers**,
  **Show error bars** (for graphs with uncertainties), **Marker size**,
  **Line width**, the figure size, **Log x-axis**, **Log y-axis**, **Grid**,
  **Apply manual axis limits** and the labels. Points with non-finite
  coordinates are omitted with a warning. On a logarithmic axis, the plot is
  refused if a visible lower error bar reaches zero or below; turn off **Show
  error bars** to draw such a graph. Points with zero or negative values cannot be
  drawn on that axis.

**Make plot** (shortcut **P**) applies the changed settings.

## What is loaded?

Browsing object metadata is different from opening the object or loading data for a plot. Use entry ranges for large event datasets and keep object inspection deliberate to reduce unnecessary memory use.

Next: [Plotting](plotting.md).
