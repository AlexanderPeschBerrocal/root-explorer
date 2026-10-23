"""GDML primitives, units, placements and warnings with known results (#153).

Expected values follow Geant4's GDML reader (G4GDMLReadSolids, G4GDMLReadDefine
and G4GDMLReadStructure): box, tube and cone lengths are full lengths, lengths
default to mm and angles to radians, a physvol rotation is applied as the
inverse of the rotation matrix built from rotateX, rotateY and rotateZ, and a
Boolean's second-operand rotation is applied directly. Comparisons use vertex
coordinates and bounding boxes rather than triangle order.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from root_explorer.geometry.gdml import SceneBuilder
from root_explorer.geometry.overlay import decode_geometry_scene, make_geometry_plotly_traces

ATOL = 1e-4  # positions are stored as float32


def _gdml(solids: str, structure: str, world: str = "World") -> str:
    return f"""<?xml version="1.0"?>
<gdml>
  <solids>
    <box name="world_box" x="1000" y="1000" z="1000"/>
{solids}
  </solids>
  <structure>
{structure}
  </structure>
  <setup name="Default" version="1.0"><world ref="{world}"/></setup>
</gdml>
"""


def _single(solid: str, *, placement: str = "") -> str:
    """A world containing one placed volume made of the given solid named 'shape'."""

    return _gdml(
        solid,
        f"""    <volume name="Shape">
      <materialref ref="G4_Si"/>
      <solidref ref="shape"/>
    </volume>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol name="shape_pv">
        <volumeref ref="Shape"/>
{placement}
      </physvol>
    </volume>""",
    )


def _scene(tmp_path: Path, text: str, **options) -> dict:
    path = tmp_path / "geometry.gdml"
    path.write_text(text, encoding="utf-8")
    return SceneBuilder(path, quality=options.pop("quality", "high"), **options).build()


def _vertices(scene: dict, object_name: str | None = None) -> np.ndarray:
    positions, _, ids = decode_geometry_scene(scene)
    if object_name is None:
        return positions.astype(float)
    (obj,) = [obj for obj in scene["objects"] if obj["volume"] == object_name]
    return positions[ids == obj["id"]].astype(float)


def _bbox(vertices: np.ndarray) -> list[float]:
    return [*vertices.min(axis=0), *vertices.max(axis=0)]


@pytest.mark.parametrize(
    ("unit", "factor"), [("", 1.0), (' lunit="mm"', 1.0), (' lunit="cm"', 10.0)]
)
def test_box_dimensions_are_full_lengths_in_the_given_unit(tmp_path, unit, factor) -> None:
    scene = _scene(tmp_path, _single(f'<box name="shape" x="2" y="4" z="6"{unit}/>'))
    expected = [-1 * factor, -2 * factor, -3 * factor, 1 * factor, 2 * factor, 3 * factor]
    np.testing.assert_allclose(_bbox(_vertices(scene)), expected, atol=ATOL)


def _cylindrical(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = vertices.T
    return np.hypot(x, y), np.arctan2(y, x), z


def test_tube_has_its_radii_and_half_length(tmp_path) -> None:
    scene = _scene(
        tmp_path,
        _single('<tube name="shape" rmin="1" rmax="3" z="10" deltaphi="360" aunit="deg"/>'),
    )
    radius, _, z = _cylindrical(_vertices(scene))
    assert np.all(np.isclose(radius, 1, atol=ATOL) | np.isclose(radius, 3, atol=ATOL))
    assert np.all(np.isclose(np.abs(z), 5, atol=ATOL))
    np.testing.assert_allclose(_bbox(_vertices(scene)), [-3, -3, -5, 3, 3, 5], atol=1e-2)


@pytest.mark.parametrize(
    "angles",
    [
        'startphi="0" deltaphi="90" aunit="deg"',
        'startphi="0" deltaphi="1.5707963267948966" aunit="rad"',
        # Without aunit, Geant4 reads angles in radians.
        'startphi="0" deltaphi="1.5707963267948966"',
    ],
    ids=["degrees", "radians", "default-unit"],
)
def test_tube_sector_covers_the_first_quadrant(tmp_path, angles) -> None:
    scene = _scene(tmp_path, _single(f'<tube name="shape" rmin="1" rmax="2" z="2" {angles}/>'))
    _, phi, _ = _cylindrical(_vertices(scene))
    assert phi.min() >= -ATOL
    assert phi.max() <= math.pi / 2 + ATOL
    assert phi.max() == pytest.approx(math.pi / 2, abs=ATOL)


@pytest.mark.parametrize("deltaphi", ["", ' deltaphi="2*pi"'], ids=["missing", "expression"])
def test_missing_or_unparseable_tube_angles_fall_back_to_a_full_tube(tmp_path, deltaphi) -> None:
    scene = _scene(tmp_path, _single(f'<tube name="shape" rmin="1" rmax="2" z="2"{deltaphi}/>'))
    np.testing.assert_allclose(_bbox(_vertices(scene)), [-2, -2, -1, 2, 2, 1], atol=1e-2)


def test_cone_radii_change_between_the_two_ends(tmp_path) -> None:
    scene = _scene(
        tmp_path,
        _single(
            '<cone name="shape" rmin1="0" rmax1="2" rmin2="0" rmax2="4" z="8" '
            'deltaphi="360" aunit="deg"/>'
        ),
    )
    radius, _, z = _cylindrical(_vertices(scene))
    assert radius[np.isclose(z, -4, atol=ATOL)].max() == pytest.approx(2, abs=ATOL)
    assert radius[np.isclose(z, 4, atol=ATOL)].max() == pytest.approx(4, abs=ATOL)
    assert np.all(np.isclose(np.abs(z), 4, atol=ATOL))


def test_cut_tube_end_faces_lie_on_their_cut_planes(tmp_path) -> None:
    low = np.array([0.0, -0.5, -1.0])
    high = np.array([0.3, 0.0, 1.0])
    scene = _scene(
        tmp_path,
        _single(
            '<cutTube name="shape" rmin="0" rmax="2" z="6" deltaphi="360" aunit="deg" '
            f'lowX="{low[0]}" lowY="{low[1]}" lowZ="{low[2]}" '
            f'highX="{high[0]}" highY="{high[1]}" highZ="{high[2]}"/>'
        ),
    )
    vertices = _vertices(scene)
    radius, _, _ = _cylindrical(vertices)
    rim = np.isclose(radius, 2, atol=ATOL)
    # The cut planes pass through (0, 0, -z/2) and (0, 0, +z/2) with the given normals.
    on_low = np.abs((vertices - [0, 0, -3]) @ low) < 1e-3
    on_high = np.abs((vertices - [0, 0, 3]) @ high) < 1e-3
    assert np.all(on_low[rim] | on_high[rim])
    assert np.any(on_low[rim]) and np.any(on_high[rim])


def test_nested_translations_accumulate_with_units(tmp_path) -> None:
    text = _gdml(
        '<box name="parent_box" x="100" y="100" z="100"/>\n<box name="leaf_box" x="2" y="2" z="2"/>',
        """    <volume name="Leaf">
      <materialref ref="G4_Si"/>
      <solidref ref="leaf_box"/>
    </volume>
    <volume name="Parent">
      <materialref ref="G4_AIR"/>
      <solidref ref="parent_box"/>
      <physvol><volumeref ref="Leaf"/><position name="p2" x="0" y="0.5" z="0" unit="cm"/></physvol>
    </volume>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol><volumeref ref="Parent"/><position name="p1" x="10" y="0" z="-3"/></physvol>
    </volume>""",
    )
    scene = _scene(tmp_path, text)
    np.testing.assert_allclose(_bbox(_vertices(scene, "Leaf")), [9, 4, -4, 11, 6, -2], atol=ATOL)


QUARTER_TUBE = '<tube name="shape" rmin="1" rmax="3" z="2" startphi="0" deltaphi="90" aunit="deg"/>'


def test_placement_rotation_is_applied_like_geant4(tmp_path) -> None:
    # A first-quadrant sector placed with rotation z = 90 deg. Geant4 applies the
    # inverse rotation, -90 deg about z, so the sector ends up in the fourth
    # quadrant; the translation is applied after the rotation.
    scene = _scene(
        tmp_path,
        _single(
            QUARTER_TUBE,
            placement='        <position name="p" x="10" y="0" z="0"/>\n'
            '        <rotation name="r" x="0" y="0" z="90" unit="deg"/>',
        ),
    )
    bbox = _bbox(_vertices(scene))
    np.testing.assert_allclose(bbox, [10, -3, -1, 13, 0, 1], atol=1e-2)


def test_placement_rotation_without_unit_is_in_radians(tmp_path) -> None:
    scene = _scene(
        tmp_path,
        _single(QUARTER_TUBE, placement=f'        <rotation name="r" z="{math.pi / 2}"/>'),
    )
    np.testing.assert_allclose(_bbox(_vertices(scene)), [0, -3, -1, 3, 0, 1], atol=1e-2)


def test_nested_rotations_rotate_the_daughter_position(tmp_path) -> None:
    # The parent is placed with rotation z = 90 deg, i.e. turned by -90 deg, so a
    # daughter at local (5, 0, 0) ends up at (0, -5, 0) in the world.
    text = _gdml(
        '<box name="parent_box" x="100" y="100" z="100"/>\n<box name="leaf_box" x="2" y="2" z="2"/>',
        """    <volume name="Leaf">
      <materialref ref="G4_Si"/>
      <solidref ref="leaf_box"/>
    </volume>
    <volume name="Parent">
      <materialref ref="G4_AIR"/>
      <solidref ref="parent_box"/>
      <physvol><volumeref ref="Leaf"/><position name="p2" x="5" y="0" z="0"/></physvol>
    </volume>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol><volumeref ref="Parent"/><rotation name="r" z="90" unit="deg"/></physvol>
    </volume>""",
    )
    scene = _scene(tmp_path, text)
    np.testing.assert_allclose(_bbox(_vertices(scene, "Leaf")), [-1, -6, -1, 1, -4, 1], atol=ATOL)


def test_union_applies_the_second_operand_transform_directly(tmp_path) -> None:
    # Unlike a placement, Geant4 applies a Boolean's second-operand rotation as given:
    # the sector rotated by +90 deg covers the second quadrant, then moves by (0, 0, 10).
    solids = (
        '<box name="a" x="2" y="2" z="2"/>\n'
        '<tube name="b" rmin="1" rmax="3" z="2" startphi="0" deltaphi="90" aunit="deg"/>\n'
        '<union name="shape"><first ref="a"/><second ref="b"/>'
        '<position name="up" x="0" y="0" z="10"/><rotation name="turn" z="90" unit="deg"/></union>'
    )
    scene = _scene(tmp_path, _single(solids))
    vertices = _vertices(scene)
    sector = vertices[vertices[:, 2] > 5]
    np.testing.assert_allclose(_bbox(sector), [-3, 0, 9, 0, 3, 11], atol=1e-2)
    np.testing.assert_allclose(
        _bbox(vertices[vertices[:, 2] < 5]), [-1, -1, -1, 1, 1, 1], atol=ATOL
    )


@pytest.mark.parametrize("operation", ["subtraction", "intersection"])
def test_other_booleans_are_approximated_by_the_first_operand_with_a_warning(
    tmp_path, operation
) -> None:
    solids = (
        '<box name="a" x="4" y="4" z="4"/>\n<box name="b" x="1" y="1" z="1"/>\n'
        f'<{operation} name="shape"><first ref="a"/><second ref="b"/></{operation}>'
    )
    scene = _scene(tmp_path, _single(solids))
    np.testing.assert_allclose(_bbox(_vertices(scene)), [-2, -2, -2, 2, 2, 2], atol=ATOL)
    assert (
        f"Boolean {operation!r} for solid name 'shape' is approximated by its first operand."
        in scene["meta"]["warnings"]
    )


def test_unsupported_and_missing_references_are_reported(tmp_path) -> None:
    text = _gdml(
        '<sphere name="ball" rmax="3"/>',
        """    <volume name="Ball"><materialref ref="G4_Si"/><solidref ref="ball"/></volume>
    <volume name="Ghost"><materialref ref="G4_Si"/><solidref ref="nowhere"/></volume>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol><volumeref ref="Ball"/></physvol>
      <physvol><volumeref ref="Ghost"/></physvol>
      <physvol><volumeref ref="Absent"/></physvol>
    </volume>""",
    )
    scene = _scene(tmp_path, text)
    assert scene["meta"]["objects"] == 0
    assert set(scene["meta"]["warnings"]) == {
        "Unsupported solid type 'sphere' for ball#0",
        "Missing solid reference: nowhere",
        "Missing volume reference: Absent",
    }


CONTAINERS = _gdml(
    '<box name="hall_box" x="100" y="100" z="100"/>\n'
    '<box name="leaf_box" x="2" y="2" z="2"/>\n<box name="air_box" x="4" y="4" z="4"/>',
    """    <volume name="Leaf"><materialref ref="G4_Si"/><solidref ref="leaf_box"/></volume>
    <volume name="AirGap"><materialref ref="G4_AIR"/><solidref ref="air_box"/></volume>
    <volume name="VacuumGap"><materialref ref="G4_Galactic"/><solidref ref="air_box"/></volume>
    <volume name="Hall">
      <materialref ref="G4_Fe"/>
      <solidref ref="hall_box"/>
      <physvol><volumeref ref="Leaf"/></physvol>
      <physvol><volumeref ref="AirGap"/><position name="g" x="10" y="0" z="0"/></physvol>
      <physvol><volumeref ref="VacuumGap"/><position name="v" x="20" y="0" z="0"/></physvol>
    </volume>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol><volumeref ref="Hall"/></physvol>
    </volume>""",
)


def test_world_containers_and_vacuum_are_excluded_by_default(tmp_path) -> None:
    default = _scene(tmp_path, CONTAINERS)
    assert {obj["volume"] for obj in default["objects"]} == {"Leaf", "AirGap"}

    everything = _scene(tmp_path, CONTAINERS, include_containers=True, include_world=True)
    assert {obj["volume"] for obj in everything["objects"]} == {
        "World",
        "Hall",
        "Leaf",
        "AirGap",
        "VacuumGap",
    }


def test_material_selection_draws_only_the_selected_materials(tmp_path) -> None:
    scene = _scene(tmp_path, CONTAINERS)
    leaf_vertices = len(_vertices(scene, "Leaf"))
    traces, info = make_geometry_plotly_traces(
        scene, selected_materials={"G4_Si"}, max_vertices=10_000
    )
    assert sum(len(trace.x) for trace in traces) == leaf_vertices
    assert info["truncated"] is False


def test_parse_limits_stop_with_a_warning(tmp_path) -> None:
    scene = _scene(tmp_path, CONTAINERS, max_objects=1)
    assert scene["meta"]["objects"] == 1
    assert any(warning.startswith("Stopped at 1 objects") for warning in scene["meta"]["warnings"])
