"""Versioned identities for ROOT and GDML sources used in cache keys.

Streamlit caches are keyed by function arguments. A file replaced at the same
path keeps its path, so caches keyed only by path would return data derived from
the previous file. Cached readers therefore receive a :func:`source_version` that
changes whenever the source changes.

Local files are identified by their resolved path, nanosecond modification time
and size. This is cheap enough to compute on every Streamlit rerun; content
hashes of large ROOT files are deliberately avoided. Filesystems with coarse
timestamp resolution can, in rare cases, keep the same identity for a file that
is rewritten within one timestamp tick with an identical size.

Remote sources (``root://``, ``http://``, ``https://``) cannot be checked with
``stat``. Their identity only changes when :func:`reload_source` is called, for
example from an explicit *Reload source* action, so remote data stays cached
until it is reloaded.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

REMOTE_PREFIXES = ("root://", "http://", "https://")

SourceVersion = tuple[str, str, int, int]

_reload_tokens: dict[str, int] = {}
_reload_lock = threading.Lock()


class SourceUnavailableError(OSError):
    """Raised when a local source no longer exists or cannot be inspected."""


def is_remote_source(path: str) -> bool:
    """Return whether ``path`` refers to a remote ROOT source."""

    return path.startswith(REMOTE_PREFIXES)


def source_version(path: str) -> SourceVersion:
    """Return a cheap, hashable identity that changes when the source changes.

    A local source that cannot be inspected raises :class:`SourceUnavailableError`
    rather than returning a placeholder identity, which could otherwise be shared
    by unrelated missing files and return stale cached data.
    """

    if is_remote_source(path):
        return ("remote", path, _reload_tokens.get(path, 0), 0)
    try:
        status = os.stat(path)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise SourceUnavailableError(f"Source is not available: {path} ({reason}).") from exc
    return ("local", str(Path(path).resolve()), status.st_mtime_ns, status.st_size)


def reload_source(path: str) -> None:
    """Give a remote source a new identity so that its cached data is read again."""

    with _reload_lock:
        _reload_tokens[path] = _reload_tokens.get(path, 0) + 1
