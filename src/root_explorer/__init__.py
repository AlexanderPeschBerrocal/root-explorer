"""Reusable components for the ROOT Explorer application."""

from importlib.metadata import version as _distribution_version


def installed_version() -> str:
    """Return the version recorded in the installed distribution metadata.

    ``[project].version`` in ``pyproject.toml`` is the single authoritative
    version literal; installing the project copies it into the distribution
    metadata read here. An editable installation keeps the metadata of its last
    installation, so it must be reinstalled after the version changes.
    """

    return _distribution_version("root-explorer")


__version__ = installed_version()
