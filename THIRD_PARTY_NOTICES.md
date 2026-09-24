# Third-party licences

ROOT Explorer is distributed under the BSD 3-Clause License. It depends on
third-party software that remains subject to its own copyright and licence terms.

The direct runtime dependencies declared in `pyproject.toml` currently include:

| Package | Licence |
| --- | --- |
| Awkward Array | BSD-3-Clause |
| Matplotlib | Matplotlib licence (PSF-compatible) |
| NumPy | BSD-3-Clause and additional permissive licences for bundled components |
| pandas | BSD-3-Clause |
| Plotly | MIT |
| SciPy | BSD-3-Clause |
| Streamlit | Apache-2.0 |
| uproot | BSD-3-Clause |

This file is an inventory for convenience and is not a replacement for the
licence notices distributed with those packages. Docker images also contain
transitive Python dependencies and components from the Python base image; those
components remain subject to their respective licence terms.

For the authoritative terms for a particular installed version, consult the
licence metadata and licence files shipped with that package or component.
