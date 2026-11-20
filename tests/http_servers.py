"""Local HTTP servers that serve ROOT files with controlled misbehaviour.

Used by the remote-source tests (#168). Each server runs in a daemon thread on
127.0.0.1 and serves the bytes of one file under ``/<name>``.

Modes:
    range    Honours single HTTP byte ranges with status 206, like typical web servers.
             Multi-range requests receive the whole file with status 200, which uproot
             treats as missing multipart support.
    norange  Ignores Range headers and always returns the whole file with status 200,
             like ``python -m http.server``.
    hang     Accepts the connection and reads the request but never responds.
    drop     Like ``range``, but closes the connection in the middle of the body of
             every GET that touches ``drop_span`` (every GET if no span is given), as
             when a connection breaks while one object is being read.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SINGLE_RANGE = re.compile(r"^bytes=(\d+)-(\d*)$")


class _Handler(BaseHTTPRequestHandler):
    server: _FileServer

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep test output quiet

    def _send_head(self, status: int, length: int, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        if self.server.mode == "range":
            self.send_header("Accept-Ranges", "bytes")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _wait_until_shutdown(self) -> None:
        self.server.release.wait()

    def do_HEAD(self) -> None:
        if self.server.mode == "hang":
            self._wait_until_shutdown()
            return
        self._send_head(200, len(self.server.payload))

    def do_GET(self) -> None:
        payload = self.server.payload
        mode = self.server.mode
        if mode == "hang":
            self._wait_until_shutdown()
            return
        match = _SINGLE_RANGE.match(self.headers.get("Range", ""))
        start, stop = 0, len(payload)
        if mode in ("range", "drop") and match:
            start = int(match.group(1))
            stop = min(int(match.group(2)) + 1 if match.group(2) else len(payload), len(payload))
            body = payload[start:stop]
            self._send_head(
                206, len(body), {"Content-Range": f"bytes {start}-{stop - 1}/{len(payload)}"}
            )
        else:
            body = payload
            self._send_head(200, len(body))
        span = self.server.drop_span
        touches = span is None or (match and start < span[1] and stop > span[0])
        if mode == "drop" and touches:
            # Promise the full body, deliver part of it, then hang up.
            self.wfile.write(body[: max(1, len(body) // 2)])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


class _FileServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, payload: bytes, mode: str, drop_span: tuple[int, int] | None) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.payload = payload
        self.mode = mode
        self.drop_span = drop_span
        self.release = threading.Event()


@contextmanager
def serve_file(path: Path, mode: str, *, drop_span: tuple[int, int] | None = None) -> Iterator[str]:
    """Serve ``path`` in the given mode and yield its URL."""

    if mode not in ("range", "norange", "hang", "drop"):
        raise ValueError(f"Unknown server mode: {mode}")
    server = _FileServer(path.read_bytes(), mode, drop_span)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/{path.name}"
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
