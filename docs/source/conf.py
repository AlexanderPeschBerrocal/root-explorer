"""Sphinx configuration for the ROOT Explorer user documentation."""

from importlib.metadata import version as distribution_version

project = "ROOT Explorer"
author = "ROOT Explorer contributors"
copyright = "2026, ROOT Explorer contributors"
release = distribution_version("root-explorer")
version = ".".join(release.split(".")[:2])

# Keep this user-facing: no automatic Python API reference.
extensions = ["myst_parser"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} documentation"
