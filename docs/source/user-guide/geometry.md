# Detector geometry overlays

ROOT Explorer can add a GDML detector geometry overlay to an interactive
**3D scatter** plot. It cannot overlay GDML onto a 1D histogram, 2D
histogram or ordinary 2D scatter plot.

## Add a GDML file

1. In the sidebar, enable **Enable 3D detector overlay**.
2. Choose **Geometry source**: **Path** to read an accessible server-side
   `.gdml` file, or **Upload** to select one in the browser.
3. Set **mesh quality**, vertex/object parsing limits, opacity, coordinate
   scale and optional offsets.
4. Select **3D scatter interactive** in Plotter; choose suitable x, y,
   z coordinates and click **Make plot**.

The **Geometry** tab shows a parsed-geometry summary and warnings.
The geometry is converted into triangle meshes rendered as Plotly
`Mesh3d` traces. Units are not automatically guaranteed to match
the event coordinates: e.g. GDML in mm and ROOT positions in cm require
an appropriate scale. The interface also supports coordinate offsets.

## Performance and correctness

Large GDML detectors may contain many volumes and triangles. Keep
**max geometry objects to parse**, **max geometry vertices to parse**,
and **max geometry vertices to draw** conservative while exploring.
Disable unnecessary world/container volumes. A truncated or simplified
mesh may omit structural detail; it should not be used as authoritative
detector geometry for quantitative reconstruction.

The current local application accepts Path and Upload; future restricted
deployment profiles are specified in
[deployment environments](../deployment.md), not yet available.
