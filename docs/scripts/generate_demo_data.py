"""Create deterministic public demonstration data for screenshots (no PyROOT)."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def generate(path: Path, *, events: int = 6000) -> Path:
    """Build a reproducible synthetic ROOT TTree and stored histogram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)
    counts = rng.poisson(1.6, events)
    pt = rng.gamma(2.5, 17.0, counts.sum()).astype(np.float32)
    eta = rng.normal(0.0, 1.25, counts.sum()).astype(np.float32)
    met = rng.exponential(52.0, events).astype(np.float32)
    fields = {
        "MET_pt": met,
        "nMuon": counts.astype(np.int32),
        "Muon_pt": ak.unflatten(pt, counts),
        "Muon_eta": ak.unflatten(eta, counts),
        "genWeight": np.ones(events, dtype=np.float32),
    }
    with uproot.recreate(path) as root:
        root.mktree("Events", fields)
        root["Overview/h_met"] = np.histogram(met, bins=45, range=(0, 250))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=6000)
    args = parser.parse_args()
    if args.events < 1:
        parser.error("--events must be positive")
    print(generate(args.output.resolve(), events=args.events))
