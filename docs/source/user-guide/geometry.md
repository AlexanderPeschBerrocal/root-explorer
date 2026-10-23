# Detector geometry overlays

ROOT Explorer can add a GDML detector geometry overlay to an interactive
**3D scatter** plot. It cannot overlay GDML onto a 1D histogram, 2D
histogram or ordinary 2D scatter plot.

## Add a GDML file

1. Open a ROOT file first, then enable **Enable 3D detector overlay** in the sidebar.
2. Choose **Geometry source**: **Path** to read an accessible server-side
   `.gdml` file, or **Upload** to select one in the browser.
3. Set **mesh quality**, vertex/object parsing limits, opacity, coordinate
   scale and optional offsets.
4. Select **3D scatter interactive** in Plotter; choose suitable x, y,
   z coordinates and click **Make plot**.

The **Geometry** tab shows a parsed-geometry summary and warnings, rather than
a standalone geometry viewer. The overlay appears in the generated 3D scatter
plot as Plotly `Mesh3d` traces. Regenerate the plot after changing geometry
settings. Parsed counts can exceed displayed counts because of filtering and
drawing limits.

Recognised GDML length units are converted to **millimetres** internally.
An unrecognised unit is treated as millimetres without a warning.
Use a geometry scale of **0.1** if the ROOT coordinates are in centimetres,
or **0.001** for metres. Offsets are applied after scaling and must be in the
plot coordinates' units. ROOT coordinate units are not detected automatically.

## Supported GDML subset

The parser provides a simplified visualisation, not a complete GDML implementation.

| Construct | Current behaviour |
| --- | --- |
| `box`, `tube`, `cutTube`, `cone` | Converted to triangle meshes; curved surfaces depend on mesh quality |
| Units | Lengths without a unit are in mm and angles without a unit in radians, as in Geant4 |
| Nested `physvol` placements | Inline numeric `position` and `rotation` elements are applied. As in Geant4, a placement rotation turns the volume by the inverse of the given rotation |
| `union` | Combines operand meshes, including the second operand's inline transform, whose rotation is applied as given, as in Geant4; does not remove interior surfaces as a true Boolean union would |
| `subtraction`, `intersection` | Approximated by the first operand, with a warning |
| `positionref`, `rotationref` | Not resolved; referenced placements can silently appear at the origin or with the wrong rotation |
| Constants and expressions in numeric attributes | Not evaluated; unparseable values fall back to defaults |
| Assemblies, replicas and parameterised placements | Not expanded as ordinary nested volumes |
| Other solid types | Skipped with warnings |

An absence of warnings does **not** establish that a geometry is faithful.
For referenced transforms, a workaround is a simplified GDML file with numeric
positions and rotations written inline. Compare with a trusted detector viewer
before interpreting alignments. GDML fidelity improvements are tracked in
[issue #116](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/116).

The world volume, container volumes (any volume with daughter placements,
whatever its material) and volumes made of `Galactic` or `G4_Galactic` are
excluded by default. Other low-density materials, such as `G4_AIR` or a
custom vacuum material, are still drawn unless deselected. Only the first
**30 materials in sorted order** are selected initially; use **materials to
draw** to include others. Check the material selection and parsed summary when
expected structures are absent. Opacity and wireframe settings can help reveal
otherwise obscured objects.

## Performance and correctness

Large GDML detectors may contain many volumes and triangles. Keep
**max geometry objects to parse**, **max geometry vertices to parse**,
and **max geometry vertices to draw** conservative while exploring.
Disable unnecessary world/container volumes. A truncated or simplified
mesh may omit structural detail; it should not be used as authoritative
detector geometry for quantitative reconstruction.

If you replace a GDML file at the same server path, the geometry is parsed
again automatically on the next interaction; see
[Caching and changed files](../performance.md#caching-and-changed-files).

The current local application accepts Path and Upload; future restricted
deployment profiles are specified in
[deployment environments](../deployment.md), not yet available.
