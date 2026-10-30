# Build and runtime use the same immutable, multi-platform Python base.
# Keep .python-version at this Python release; the Docker workflow checks it.
FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS base

# The uv used to install the lock, pinned by digest like the base image. Keep
# setup-uv and .readthedocs.yaml at this version; the build workflow checks it.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv


# ----- Build stage -----------------------------------------------------------
FROM base AS builder

# uv is only needed to install the lock; it is not copied into the runtime.
COPY --from=uv /uv /uvx /usr/local/bin/

# Use the base image's Python, install into an isolated virtual environment,
# copy files instead of hard-linking from the uv cache, and precompile
# bytecode because the runtime user cannot write to /opt/venv.
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /build

COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src

# Install exactly the locked runtime dependencies (no development groups or
# optional extras) and the application itself as a regular, non-editable package.
RUN uv sync --locked --no-default-groups --no-editable


# ----- Runtime stage ---------------------------------------------------------
FROM base AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create an unprivileged user for running the application.
RUN useradd --create-home --uid 10001 appuser

# Copy only the installed application and its dependencies.
COPY --from=builder /opt/venv /opt/venv

# Copy only files actually needed at runtime.
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
