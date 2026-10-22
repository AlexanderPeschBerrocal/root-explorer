# Performance and limitations

ROOT Explorer is a single interactive application, not a distributed
event-processing platform. Runtime and memory use depend strongly on
selected entry ranges, nested/jagged field sizes, rendering mode and
server resources.

## Reduce the workload

- Start with a small **Read range** when choosing fields and plot options;
  increase it after the result is correct.
- Jagged collections can expand into many more object observations
  than events. Multiple cached large fields can be memory intensive.
- 2D/3D scatter rendering can cap or sample plotted points for responsiveness.
  Treat any scatter sample as a visualisation, not necessarily the full dataset.
- For GDML, reduce parsed and drawn vertices and omit unneeded volumes.

## Known correctness limitations

The current implementation has the following known limitations. These are
tracked bugs, not intended analysis conventions.

| Situation | Effect | Workaround / tracking |
| --- | --- | --- |
| Source fields contain optional/missing values | Flattening can drop missing values independently, misalign combined variables or shift ranked entry indices without a warning | Preprocess inputs while preserving event/object correspondence, using a common validity mask across related fields; do not drop missing values independently. [#137](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/137) |
| A ROOT or GDML file is replaced at the same server path | Cached metadata, event data, rankings or geometry can remain stale | Use a new, versioned filename or restart the application server. A browser refresh alone is insufficient. [#117](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/117) |
| The selected TTree/RNTuple has zero entries | Startup stops before the ROOT-object browser, even if the file contains valid histograms or graphs | Choose a non-empty event container if available, or use a separate file containing only persisted objects. [#138](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/138) |
| GDML uses referenced transforms or unsupported constructs | Geometry may be misplaced, simplified or omitted; not every omission produces a warning | Check the [GDML support table](user-guide/geometry.md#supported-gdml-subset) and compare with a trusted viewer. [#116](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/116) |

## Local and hosted constraints

The current Streamlit configuration sets `server.maxUploadSize = 4096`,
i.e. **4,096 MiB per uploaded file**; this is not a promise that every
host can store or process files of that size. CPU, memory, disk and
network constraints can be considerably tighter.

The planned three-file per-session cap, expiry policy, and public upload-only
profile belong to separate, unimplemented work (#38 and #110). Do not rely
on them as active limits or protections in the present application.

For detailed statistical restrictions of histogram density, flow and fits,
see [Histogram statistics and fitting](user-guide/histograms-and-fits.md).
