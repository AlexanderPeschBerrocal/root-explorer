"""Generate the committed ROOT-object fixture used by graph and histogram tests.

Run with an environment that provides PyROOT, NumPy, and Uproot:

    python tests/data/generate_graph_fixture.py

PyROOT is intentionally only a fixture-generation dependency.  The test suite
reads the committed file exclusively through Uproot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT
import uproot

OUTPUT_PATH = Path(__file__).with_name("test_objects.root")


def _graph(
    graph_type: type,
    x: np.ndarray,
    y: np.ndarray,
    *errors: np.ndarray,
) -> object:
    """Create a ROOT graph while retaining NumPy's deterministic values."""
    return graph_type(len(x), x, y, *errors)


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    """Write graphs, histograms, profiles, and an event TTree into ``output_path``."""
    x = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    xerr = np.array([0.05, 0.10, 0.15], dtype=np.float64)
    yerr = np.array([0.20, 0.25, 0.30], dtype=np.float64)
    xerr_low = np.full(3, 0.1, dtype=np.float64)
    xerr_high = np.full(3, 0.2, dtype=np.float64)
    yerr_low = np.full(3, 0.15, dtype=np.float64)
    yerr_high = np.full(3, 0.3, dtype=np.float64)

    root_file = ROOT.TFile(str(output_path), "RECREATE")
    try:
        plain = _graph(ROOT.TGraph, x, y)
        plain.SetName("plain_graph")
        plain.SetTitle("Plain graph;Time [ns];Amplitude")
        plain.Write()

        symmetric = _graph(ROOT.TGraphErrors, x, y, xerr, yerr)
        symmetric.SetName("symmetric_graph")
        symmetric.SetTitle("Symmetric graph;Momentum [GeV];Efficiency")
        symmetric.Write()

        asymmetric = _graph(
            ROOT.TGraphAsymmErrors,
            x,
            y,
            xerr_low,
            xerr_high,
            yerr_low,
            yerr_high,
        )
        asymmetric.SetName("asymmetric_graph")
        asymmetric.SetTitle("Asymmetric graph;Position [cm];Rate [Hz]")
        asymmetric.Write()

        nested = root_file.mkdir("nested")
        nested.cd()
        nested_graph = _graph(ROOT.TGraph, x, y)
        nested_graph.SetName("nested_graph")
        nested_graph.SetTitle("Nested graph;Nested x;Nested y")
        nested_graph.Write()

        # Store a profile inside a TDirectory to test nested-object browsing.
        # ROOT profiles contain bin means and uncertainties, not event counts.
        profile_2d = ROOT.TProfile2D(
            "profile_2d",
            "Mean response;#eta;p_{T} [GeV];Mean response",
            2,
            0.0,
            2.0,
            2,
            0.0,
            2.0,
        )
        for eta, pt, response in (
            (0.5, 0.5, 2.0),
            (0.5, 0.5, 4.0),
            (0.5, 1.5, 1.0),
            (0.5, 1.5, 3.0),
            (1.5, 0.5, 5.0),
            (1.5, 0.5, 7.0),
            (1.5, 1.5, 9.0),
            (1.5, 1.5, 11.0),
        ):
            profile_2d.Fill(eta, pt, response)
        profile_2d.Write()
        root_file.cd()

        first_cycle = _graph(ROOT.TGraph, x, y)
        first_cycle.SetName("cycled_graph")
        first_cycle.SetTitle("First cycle;Cycle x;Cycle y")
        first_cycle.Write()
        second_cycle = _graph(ROOT.TGraph, x + 10.0, y + 10.0)
        second_cycle.SetName("cycled_graph")
        second_cycle.SetTitle("Second cycle;Cycle x;Cycle y")
        second_cycle.Write()

        histogram_1d = ROOT.TH1D("histogram_1d", "One-dimensional histogram", 3, 0.0, 3.0)
        histogram_1d.Fill(0.5)
        histogram_1d.Fill(1.5, 2.0)
        histogram_1d.Write()
        histogram_2d = ROOT.TH2D(
            "histogram_2d", "Two-dimensional histogram", 2, 0.0, 2.0, 2, 0.0, 2.0
        )
        histogram_2d.Fill(0.5, 1.5)
        histogram_2d.Write()

        # Three visible eta bins with known means and standard errors.
        # Flow entries are deliberate: their contents must be retained even
        # while the plotting interface does not expose flow bins.
        profile_1d = ROOT.TProfile(
            "profile_1d", "Mean deposited energy;#eta;Mean E [GeV]", 3, 0.0, 3.0
        )
        for eta, energy in (
            (-0.5, 11.0),
            (0.5, 2.0),
            (0.5, 4.0),
            (1.5, 1.0),
            (1.5, 3.0),
            (1.5, 5.0),
            (2.5, 5.0),
            (2.5, 7.0),
            (3.5, 12.0),
        ):
            profile_1d.Fill(eta, energy)
        profile_1d.Write()
    finally:
        root_file.Close()

    with uproot.update(output_path) as writable_file:
        writable_file.mktree(
            "Events",
            {
                "event": np.dtype("uint32"),
                "energy": np.dtype("float64"),
            },
        )
        writable_file["Events"].extend(
            {
                "event": np.array([1, 2, 3], dtype=np.uint32),
                "energy": np.array([10.0, 20.0, 30.0], dtype=np.float64),
            }
        )
    return output_path


if __name__ == "__main__":
    generate()
