"""Command-line launcher: ``root-explorer [ROOT_FILE] [options]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from . import __version__

APP_SCRIPT = files("root_explorer").joinpath("streamlit_app.py")

# Same prefixes as ``io.source.REMOTE_PREFIXES``. Importing ``root_explorer.io`` here
# would load Streamlit-cached readers and print warnings before Streamlit runs;
# a test keeps both definitions identical.
REMOTE_PREFIXES = ("root://", "http://", "https://")

# Settings from the repository's .streamlit/config.toml. Streamlit only reads that
# file from the current directory, so the launcher passes them explicitly; options
# given on the command line come later and take precedence.
DEFAULT_STREAMLIT_OPTIONS = (
    "--server.maxUploadSize=4096",
    "--browser.gatherUsageStats=false",
    "--client.toolbarMode=minimal",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="root-explorer",
        allow_abbrev=False,
        description="Launch ROOT Explorer, optionally with a ROOT file pre-selected.",
        epilog="Further options, for example --server.address=0.0.0.0, are passed to Streamlit.",
    )
    parser.add_argument(
        "root_file",
        nargs="?",
        help="ROOT file path or root:// / http(s):// URL to open on start-up.",
    )
    parser.add_argument("--port", type=int, help="Port to serve the application on.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def streamlit_arguments(argv: Sequence[str] | None = None) -> list[str]:
    """Translate launcher arguments into ``streamlit run`` arguments."""

    parser = _parser()
    tokens = iter(sys.argv[1:] if argv is None else argv)
    normalized = []
    for token in tokens:
        if token == "--":
            normalized.extend([token, *tokens])
            break
        # Streamlit's --section.setting options all take a value, including
        # boolean settings. Bind split values before argparse can mistake them
        # for the optional ROOT filename. Leave validation to Streamlit.
        if token.startswith("--") and "." in token and "=" not in token:
            value = next(tokens, None)
            if value is None or value.startswith("--"):
                parser.error(f"argument {token}: expected one argument")
            token = f"{token}={value}"
        normalized.append(token)
    args, streamlit_options = parser.parse_known_args(normalized)
    arguments = ["run", str(APP_SCRIPT), *DEFAULT_STREAMLIT_OPTIONS]
    if args.port is not None:
        arguments.append(f"--server.port={args.port}")
    if args.no_browser:
        arguments.append("--server.headless=true")
    arguments.extend(streamlit_options)
    if args.root_file:
        root_file = args.root_file
        if not root_file.startswith(REMOTE_PREFIXES):
            # The app may run with a different working directory, so pass an absolute path.
            root_file = str(Path(root_file).expanduser().resolve())
        arguments.extend(["--", root_file])
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    """Run ROOT Explorer with Streamlit."""

    arguments = streamlit_arguments(argv)
    from streamlit.web import cli as streamlit_cli

    streamlit_cli.main(arguments, prog_name="streamlit")
