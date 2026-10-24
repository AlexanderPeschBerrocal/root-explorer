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

Recognised GDML length units (`mm`, `cm`, `m`, `km`, `um`, `nm`, `angstrom`,
`fermi`, `pc` and their long names such as `millimeter`) are converted to
**millimetres** internally, and angle units (`rad`, `mrad`, `deg` and their long
names) to radians. An unrecognised length unit is treated as millimetres, and an
unrecognised angle unit as radians, with a parser warning.
Use a geometry scale of **0.1** if the ROOT coordinates are in centimetres,
or **0.001** for metres. Offsets are applied after scaling and must be in the
plot coordinates' units. ROOT coordinate units are not detected automatically.

## Supported GDML subset

The parser provides a simplified visualisation, not a complete GDML implementation.

| Construct | Current behaviour |
| --- | --- |
| `box`, `tube`, `cutTube`, `cone` | Converted to triangle meshes; curved surfaces depend on mesh quality |
| Units | Lengths without a unit are in mm and angles without a unit in radians, as in Geant4 |
| Nested `physvol` placements | Inline numeric `position`, `rotation` and `scale` elements and references to definitions are applied. As in Geant4, a placement rotation turns the volume by the inverse of the given rotation, and a scale (including a reflection such as `z="-1"`) is applied before the rotation |
| `union` | Combines operand meshes, including the first operand's `firstposition`/`firstrotation` and the second operand's `position`/`rotation` (inline or referenced), whose rotations are applied as given, as in Geant4; does not remove interior surfaces as a true Boolean union would |
| `subtraction`, `intersection` | Approximated by the first operand, including its `firstposition`/`firstrotation`, with a warning |
| `positionref`, `rotationref`, `scaleref` and the `first…ref` forms | Resolved from numeric `position`, `rotation` and `scale` entries in `define`; unresolved references stop loading with an error |
| Constants and expressions in numeric attributes | Not evaluated; unparseable attributes of supported solids and transforms fall back to defaults with warnings |
| `assembly` | Expanded: its placements are positioned relative to the assembly's placement, including nested assemblies |
| Replicas, divisions, parameterised placements (`replicavol`, `divisionvol`, `paramvol`) and physvol `file` references | Not expanded; a parser warning names the construct, and the mother volume is treated as a container |
| Files that include other files through external XML entities (`<!ENTITY … SYSTEM "…">`) | Not loaded; an error asks you to merge the included files into one GDML file |
| Other solid types | Skipped with warnings |

An absence of warnings does **not** establish that a geometry is faithful.
Use numeric dimensions and transforms for the supported subset. Compare with a trusted detector viewer
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

**include world volume** controls the world independently of **include container
volumes**. The latter also enables the otherwise hidden Galactic vacuum volumes.

## Performance and correctness

Large GDML detectors may contain many volumes and triangles. Keep
**max geometry objects to parse**, **max geometry vertices to parse**,
and **max geometry vertices to draw** conservative while exploring.
Disable unnecessary world/container volumes. A truncated or simplified
mesh may omit structural detail; it should not be used as authoritative
detector geometry for quantitative reconstruction.

The parse vertex limit is a hard cap rounded down to complete triangles. A warning
reports parsing truncation only when an object or triangle was actually left out;
a warning below a 3D plot reports display truncation.
Cyclic solid, volume or assembly references and volume nesting deeper than 256
levels stop loading with an error. Changing any geometry setting (enabling the overlay, mesh quality, inclusion
flags, parsing and drawing limits, materials, opacity, scale, offsets or
wireframe) marks an existing 3D plot as out of date; click **Make plot** (or
**Make event plot**) to regenerate it.

If you replace a GDML file at the same server path, the geometry is parsed
again automatically on the next interaction; see
[Caching and changed files](../performance.md#caching-and-changed-files).

The current local application accepts Path and Upload; future restricted
deployment profiles are specified in
[deployment environments](../deployment.md), not yet available.
