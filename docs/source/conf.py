"""Sphinx configuration for the ROOT Explorer user documentation."""

from importlib.metadata import version as distribution_version

project = "ROOT Explorer"
author = "ROOT Explorer contributors"
copyright = "2026, ROOT Explorer contributors"
release = distribution_version("root-explorer")
version = ".".join(release.split(".")[:2])

# Keep this user-facing: no automatic Python API reference.
extensions = ["myst_parser"]
# Dollar-delimited LaTeX is reliably parsed in the Markdown user guides.
myst_enable_extensions = ["dollarmath"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
# Furo offers an automatic light/dark setting and a visible theme switcher.
html_title = f"{project} {release} documentation"
