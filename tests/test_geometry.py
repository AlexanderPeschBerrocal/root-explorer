from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from root_explorer.geometry.overlay import (
    decode_geometry_scene,
    geometry_materials,
    geometry_summary_dataframe,
    load_geometry_scene,
    make_geometry_plotly_traces,
    overlay_geometry_on_plotly,
    plotly_color_from_rgba,
)


def test_gdml_scene_decodes_and_overlays(tiny_gdml_path) -> None:
    scene = load_geometry_scene(str(tiny_gdml_path), "fast", False, False, 10, 1_000)
    positions, colors, ids = decode_geometry_scene(scene)

    assert scene["meta"]["objects"] == 1
    assert positions.shape[1] == 3
    assert len(colors) == len(positions) == len(ids)
    assert geometry_materials(scene) == ["G4_Si"]
    assert "bbox [xmin,ymin,zmin,xmax,ymax,zmax]" in set(geometry_summary_dataframe(scene)["item"])
    assert positions[:, 0].min() == pytest.approx(4.0)

    traces, info = make_geometry_plotly_traces(
        scene, selected_materials={"G4_Si"}, max_vertices=9, scale=2.0, offset=(1, 0, 0)
    )
    assert len(traces) == 1
    assert info["truncated"] is True
    assert np.min(traces[0].x) == pytest.approx(9.0)

    base = go.Figure(data=[go.Scatter3d(x=[0], y=[0], z=[0])])
    overlaid = overlay_geometry_on_plotly(base, scene, {"enabled": True, "max_vertices": 9})
    assert len(overlaid.data) == 2
    assert overlaid.layout.meta["geometry_overlay"]["shown_vertices"] == 9
    assert plotly_color_from_rgba((1, 2, 3, 128), 0.5) == ("rgb(1,2,3)", pytest.approx(128 / 510))


def test_geometry_loader_rejects_non_gdml(tmp_path) -> None:
    bad_path = tmp_path / "geometry.txt"
    bad_path.write_text("not geometry", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Only .gdml files"):
        load_geometry_scene(str(bad_path), "fast", False, False, 10, 1_000)
