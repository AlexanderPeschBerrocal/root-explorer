#!/usr/bin/env python3
"""
ROOT Explorer
=============

A general-purpose Streamlit app for interactive plotting and event inspection 
of HEP ROOT TTrees in the web browser.

The app is intentionally analysis-agnostic. It works with generic TTrees from 
NanoAOD, DAOD, and small custom ROOT files, as long as uproot can read the selected branches.

Main features
-------------
- Open a local ROOT file from a path or via Streamlit upload.
- Browse TTrees and branches, including branch type and collection grouping.
- Plot direct branches, object multiplicities, leading/subleading elements, and
  common derived quantities such as r, phi, eta, delta-phi, delta-R, ratios,
  sums and differences.
- Make 1D histograms, 2D histograms, 2D scatter plots, and interactive 3D
  Plotly scatter plots.
- Apply multiple simultaneous range cuts using any supported variable expression.
- Inspect a single event and rank events by multiplicity, sum, mean, max, min,
  or threshold counts.
- Export static plots as PNG and/or PDF, and interactive 3D plots as HTML.
- Load GDML detector geometry from a path or upload and overlay transparent detector meshes on 3D scatter plots.

Install
-------
    micromamba install -c conda-forge streamlit uproot awkward numpy matplotlib pandas plotly

Run
---
    streamlit run root_explorer.py -- path/to/file.root

or simply:
    streamlit run root_explorer.py

Notes
-----
This app is designed as an exploratory viewer, not as a replacement for a final
analysis workflow. For very large files, start with a limited entry range and
increase it only after the desired plot configuration is stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Keep streamlit run root_explorer.py working from a checkout while the
# package is being developed. Installed users will resolve the package normally.
_SOURCE_DIR = Path(__file__).resolve().parent / "src"
if _SOURCE_DIR.is_dir():
    sys.path.insert(0, str(_SOURCE_DIR))

from root_explorer.app.main import main


if __name__ == "__main__":
    main()
