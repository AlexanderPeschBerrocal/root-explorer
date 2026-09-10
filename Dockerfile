# ----- Build stage -----------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install the application into an isolated virtual environment.
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install .


# ----- Runtime stage ---------------------------------------------------------
FROM python:3.12-slim AS runtime

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