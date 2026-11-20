"""Application-level constants shared by the Streamlit components."""

APP_TITLE = "ROOT Explorer"
DOCS_URL = "https://root-explorer.readthedocs.io/"
DEFAULT_MAX_ENTRIES = 500_000
DEFAULT_MAX_POINTS = 100_000
# Seconds a remote (http(s):// or root://) server may stay silent before a read fails,
# instead of blocking the app for the transport's default of several minutes.
REMOTE_TIMEOUT_SECONDS = 30
