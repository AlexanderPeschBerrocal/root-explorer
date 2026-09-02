from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st

from examples.generate_test_nanoaod import generate_test_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_streamlit_caches() -> None:
    """Prevent cached reads and scenes from leaking between temporary fixtures."""

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def root_path(tmp_path: Path) -> Path:
    return generate_test_root(tmp_path / "test_nanoaod.root", n_events=12, seed=20250831)


@pytest.fixture
def tiny_gdml_path(tmp_path: Path) -> Path:
    """Create the smallest useful world-plus-detector GDML scene."""

    path = tmp_path / "tiny_detector.gdml"
    path.write_text(
        """<?xml version="1.0"?>
<gdml>
  <solids>
    <box name="world_box" x="20" y="20" z="20" lunit="mm"/>
    <box name="sensor_box" x="2" y="4" z="6" lunit="mm"/>
  </solids>
  <structure>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol name="sensor_placement">
        <volumeref ref="Sensor"/>
        <position x="5" y="0" z="0" unit="mm"/>
      </physvol>
    </volume>
    <volume name="Sensor">
      <materialref ref="G4_Si"/>
      <solidref ref="sensor_box"/>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
""",
        encoding="utf-8",
    )
    return path
