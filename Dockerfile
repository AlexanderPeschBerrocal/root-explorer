FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create an unprivileged user for running the application.
RUN useradd --create-home --uid 10001 appuser

# Copy the files required to install root-explorer.
COPY pyproject.toml README.md ./
COPY src ./src

# Install root-explorer and all runtime dependencies declared in pyproject.toml.
RUN python -m pip install .

# Copy the Streamlit entry point and configuration.
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]