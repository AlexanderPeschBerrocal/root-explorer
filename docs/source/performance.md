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

## Caching and changed files

ROOT Explorer caches file catalogues, field metadata, read data, event rankings
and parsed GDML geometry so that repeated interactions stay fast. Cache entries
for **local files** are tied to the file's path, modification time and size.
If a ROOT or GDML file is regenerated or replaced at the same path, the next
interaction reads it again automatically, and plots and rankings created from
the previous version are discarded; no restart is needed. If a file disappears
while you are working with it, ROOT Explorer reports that it is no longer
available instead of showing previously cached data.

An open stored histogram or graph is redrawn from the updated ROOT file. If
the inspected object was removed or changed class, its viewer closes. After a
GDML change, existing 3D plots using that file are cleared; click **Make plot**
or **Make event plot** to generate the updated overlay.

**Remote files** (`root://`, `http://`, `https://`) cannot be checked in the same
way. Their data stays cached until you click **Reload source** below the file
path, which reads the file again for all users of the application.
The `root://` protocol additionally requires an optional transport backend;
see [Installation](installation.md#the-root-explorer-command).

On filesystems with coarse timestamps, a file rewritten within the same
timestamp interval with exactly the same size may, rarely, not be detected.
Use a new filename if you rewrite files in very quick succession.

## Known correctness limitations

The current implementation has the following known limitations. These are
tracked bugs, not intended analysis conventions.

| Situation | Effect | Workaround / tracking |
| --- | --- | --- |
| GDML uses expressions, unsupported solids or placement constructs | Geometry may be simplified or omitted; parser warnings identify supported checks but do not establish complete fidelity | Check the [GDML support table](user-guide/geometry.md#supported-gdml-subset) and compare with a trusted viewer. [#116](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/116) |

## Local and hosted constraints

Uploads are stored as content-addressed temporary files on the server. The same
upload is reused across widget reruns, so changing controls does not invalidate
its plots or rankings. Different contents receive a different path, even when
their original filenames match. Uploaded copies currently have no automatic
expiry or per-session cleanup.

The current Streamlit configuration sets `server.maxUploadSize = 4096`,
i.e. **4,096 MiB per uploaded file**; this is not a promise that every
host can store or process files of that size. CPU, memory, disk and
network constraints can be considerably tighter.

The planned three-file per-session cap, expiry policy, and public upload-only
profile belong to separate, unimplemented work (#38 and #110). Do not rely
on them as active limits or protections in the present application.

For detailed statistical restrictions of histogram density, flow and fits,
see [Histogram statistics and fitting](user-guide/histograms-and-fits.md).
