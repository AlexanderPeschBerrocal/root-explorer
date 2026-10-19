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
