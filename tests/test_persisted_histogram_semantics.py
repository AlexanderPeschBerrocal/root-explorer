"""Numerical ROOT histogram round-trip tests without a PyROOT test dependency."""

from __future__ import annotations

import os
import subprocess
import sys

import hist
import numpy as np
import uproot
from uproot.writing.identify import to_TAxis, to_TProfile, to_TProfile2D

from root_explorer.io.histograms import read_histogram
from root_explorer.plotting.histogram_codegen import generate_histogram_plot_code
from root_explorer.plotting.histograms import (
    HistogramPlotSpec,
    histogram_errors,
    render_histogram_data,
)


def test_weighted_histogram_preserves_sumw2_flow_and_variable_bins(tmp_path):
    path = tmp_path / "weights.root"
    original = hist.Hist(
        hist.axis.Variable([0.0, 1.0, 4.0], underflow=True, overflow=True),
        storage=hist.storage.Weight(),
    )
    original.fill(np.array([-1.0, 0.5, 0.7, 3.0, 8.0]), weight=[9.0, 2.0, -3.0, 4.0, -5.0])
    with uproot.recreate(path) as root_file:
        root_file["weighted"] = original

    data = read_histogram(str(path), "weighted;1")
    np.testing.assert_allclose(data.histogram.axes[0].edges, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(data.histogram.values(flow=True), [9.0, -1.0, 4.0, -5.0])
    np.testing.assert_allclose(data.histogram.variances(flow=True), [81.0, 13.0, 16.0, 25.0])
    np.testing.assert_allclose(histogram_errors(data), [np.sqrt(13.0), 4.0])


def _axis(name: str, title: str):
    return to_TAxis(fName=name, fTitle=title, fNbins=2, fXmin=0.0, fXmax=2.0)


def test_root_profile_mean_errors_and_axis_metadata(tmp_path):
    path = tmp_path / "profile.root"
    profile = to_TProfile(
        fName="profile",
        fTitle="Mean energy;Position;Energy [GeV]",
        data=np.array([0.0, 6.0, 9.0, 0.0], dtype=np.float64),
        fEntries=5.0,
        fTsumw=5.0,
        fTsumw2=5.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=15.0,
        fTsumwy2=55.0,
        fSumw2=np.array([0.0, 20.0, 35.0, 0.0]),
        fBinEntries=np.array([0.0, 2.0, 3.0, 0.0]),
        fBinSumw2=np.array([], dtype=np.float64),
        fXaxis=_axis("xaxis", "Position [cm]"),
    )
    with uproot.recreate(path) as root_file:
        root_file["profile"] = profile

    with uproot.open(path) as root_file:
        original = root_file["profile"]
        expected_errors = original.errors()
        expected_mean = original.values()

    data = read_histogram(str(path), "profile")
    assert data.classname == "TProfile"
    assert data.kind == "MEAN"
    assert data.is_profile
    np.testing.assert_allclose(data.histogram.values(), expected_mean)
    np.testing.assert_allclose(histogram_errors(data), expected_errors)
    assert data.title == "Mean energy"
    assert data.x_label == "Position [cm]"
    assert data.y_label == "Energy [GeV]"

    figure = render_histogram_data(data, {"fig_width": 7, "fig_height": 5}, {})
    try:
        assert figure.axes[0].get_ylabel() == "Energy [GeV]"
        assert figure.axes[0].containers  # Default profile presentation includes error bars.
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_root_profile2d_fallback_retains_mean_and_errors(tmp_path):
    path = tmp_path / "profile2d.root"
    raw = np.zeros(16, dtype=np.float64)
    entries = np.zeros(16, dtype=np.float64)
    second_moment = np.zeros(16, dtype=np.float64)
    raw[5] = 6.0
    entries[5] = 2.0
    second_moment[5] = 20.0

    profile = to_TProfile2D(
        fName="profile2d",
        fTitle="Mean response;eta;pt;response",
        data=raw,
        fEntries=2.0,
        fTsumw=2.0,
        fTsumw2=2.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=0.0,
        fTsumwy2=0.0,
        fTsumwxy=0.0,
        fTsumwz=6.0,
        fTsumwz2=20.0,
        fSumw2=second_moment,
        fBinEntries=entries,
        fBinSumw2=np.array([], dtype=np.float64),
        fXaxis=_axis("xaxis", "eta"),
        fYaxis=_axis("yaxis", "pt"),
    )
    with uproot.recreate(path) as root_file:
        root_file["nested/profile2d"] = profile

    with uproot.open(path) as root_file:
        source = root_file["nested/profile2d"]
        expected_values = source.values(flow=True)
        expected_variances = source.variances(flow=True)

    data = read_histogram(str(path), "nested/profile2d;1")
    assert data.histogram.ndim == 2
    assert data.kind == "MEAN"
    np.testing.assert_allclose(data.histogram.values(flow=True), expected_values)
    np.testing.assert_allclose(data.histogram.variances(flow=True), expected_variances)
    assert data.colorbar_label == "response"
    assert data.x_label == "eta"
    assert data.y_label == "pt"


def test_export_executes_without_root_explorer_import(tmp_path):
    path = tmp_path / "histogram.root"
    with uproot.recreate(path) as root_file:
        root_file["hist"] = np.histogram([1.5, 2.5, 2.6], bins=3, range=(1.0, 4.0))

    spec = HistogramPlotSpec(
        str(path),
        "hist;1",
        {"fig_width": 7.0, "fig_height": 5.0, "histogram_mode": "Step"},
        {"title": "Reproducible"},
    )
    source = generate_histogram_plot_code(spec)
    assert "from root_explorer" not in source
    assert "import streamlit" not in source
    export = tmp_path / "hist_export.py"
    export.write_text(source, encoding="utf-8")
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(export)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sparse_profile1d_distinguishes_empty_bins_from_measured_zero(tmp_path):
    """Occupancy must not be inferred from the value or error of a bin."""
    import matplotlib.pyplot as plt

    from root_explorer.plotting.histograms import visible_profile_occupancy

    path = tmp_path / "sparse_profile1d.root"
    # Three regular bins: empty, occupied with mean zero, occupied with mean 3.
    # One underflow observation proves occupancy retains flow semantics.
    root_file_data = to_TProfile(
        fName="sparse",
        fTitle="Sparse profile;x;mean",
        data=np.array([8.0, 0.0, 0.0, 6.0, 0.0]),
        fEntries=4.0,
        fTsumw=4.0,
        fTsumw2=4.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=14.0,
        fTsumwy2=52.0,
        fSumw2=np.array([64.0, 0.0, 0.0, 20.0, 0.0]),
        fBinEntries=np.array([1.0, 0.0, 1.0, 2.0, 0.0]),
        fBinSumw2=np.array([], dtype=float),
        fXaxis=to_TAxis(fName="xaxis", fTitle="x", fNbins=3, fXmin=0, fXmax=3),
    )
    with uproot.recreate(path) as root_file:
        root_file["sparse"] = root_file_data
    data = read_histogram(str(path), "sparse")
    np.testing.assert_array_equal(data.profile_entries, [1.0, 0.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(data.histogram.values(), [0.0, 0.0, 3.0])
    np.testing.assert_array_equal(visible_profile_occupancy(data), [False, True, True])

    # Markers must not include the empty bin; a genuine zero mean must remain.
    fig = render_histogram_data(
        data,
        {"fig_width": 6, "fig_height": 4, "histogram_mode": "Error bars"},
        {},
    )
    try:
        points = fig.axes[0].lines[0]
        np.testing.assert_allclose(points.get_xdata(), [1.5, 2.5])
        np.testing.assert_allclose(points.get_ydata(), [0.0, 3.0])
    finally:
        plt.close(fig)

    # Step and filled modes must break across missing bins, not invent zeros.
    for mode in ("Step", "Filled", "Step + error bars"):
        fig = render_histogram_data(
            data, {"fig_width": 6, "fig_height": 4, "histogram_mode": mode}, {}
        )
        try:
            vertices = fig.axes[0].patches[0].get_path().vertices
            assert np.isnan(vertices).any()
        finally:
            plt.close(fig)
    np.testing.assert_array_equal(data.histogram.values(), [0.0, 0.0, 3.0])


def test_sparse_profile2d_masks_empty_cells_but_not_measured_zero(tmp_path):
    """Profile occupancy is stored in ROOT's effective bin-entry array."""
    import matplotlib.pyplot as plt

    from root_explorer.plotting.histograms import visible_profile_occupancy

    path = tmp_path / "sparse_profile2d.root"
    raw = np.zeros(16)
    raw[5] = 0.0  # Bin (0, 0) is occupied, with mean exactly zero.
    raw[10] = 6.0  # Bin (1, 1) is occupied, with mean three.
    entries = np.zeros(16)
    entries[[5, 10]] = 2.0
    second_moment = np.zeros(16)
    second_moment[[5, 10]] = [2.0, 20.0]
    profile = to_TProfile2D(
        fName="sparse2d",
        fTitle="Sparse 2D;x;y;mean",
        data=raw,
        fEntries=4.0,
        fTsumw=4.0,
        fTsumw2=4.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=0.0,
        fTsumwy2=0.0,
        fTsumwxy=0.0,
        fTsumwz=6.0,
        fTsumwz2=22.0,
        fSumw2=second_moment,
        fBinEntries=entries,
        fBinSumw2=np.array([], dtype=float),
        fXaxis=_axis("xaxis", "x"),
        fYaxis=_axis("yaxis", "y"),
    )
    with uproot.recreate(path) as root_file:
        root_file["sparse2d"] = profile
    data = read_histogram(str(path), "sparse2d")
    expected_occupancy = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(visible_profile_occupancy(data), expected_occupancy)
    np.testing.assert_array_equal(data.histogram.values(), [[0.0, 0.0], [0.0, 3.0]])

    for style in (
        {"fig_width": 6, "fig_height": 4},
        {"fig_width": 6, "fig_height": 4, "log_z": True},
    ):
        fig = render_histogram_data(data, style, {})
        try:
            mesh = fig.axes[0].collections[0]
            masked = np.ma.getmaskarray(mesh.get_array()).reshape(2, 2)
            expected = ~expected_occupancy.T
            if style.get("log_z"):
                expected |= np.array([[True, False], [False, False]])
            np.testing.assert_array_equal(masked, expected)
            if not style.get("log_z"):
                np.testing.assert_allclose(np.ma.compressed(mesh.get_array()), [0, 3])
        finally:
            plt.close(fig)

    np.testing.assert_array_equal(data.histogram.values(), [[0.0, 0.0], [0.0, 3.0]])
