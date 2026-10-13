"""Numerical ROOT histogram round-trip tests without a PyROOT test dependency."""

from __future__ import annotations

import os
import subprocess
import sys

import hist
import numpy as np
import pytest
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
        str(path), "hist;1",
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
