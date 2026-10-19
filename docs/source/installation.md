# Installation and launching

ROOT Explorer requires **Python 3.12**. It can run locally or in Docker. The browser is the user interface; with either option, the application process reads and processes the ROOT data.

## Install from source

Clone the repository and enter its directory:

```bash
git clone https://github.com/AlexanderPeschBerrocal/root-explorer.git
cd root-explorer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run streamlit_app.py
```

On Windows, activate the virtual environment with the appropriate shell-specific command instead. Streamlit normally serves the app at `http://localhost:8501`.

To work on the application itself, the developer extras can be installed separately with `python -m pip install -e ".[dev,test]"`.

## Run with Docker

Use the published stable image:

```bash
docker pull ghcr.io/alexanderpeschberrocal/root-explorer:stable
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

For reproducible deployments, select a **versioned** image tag rather than the moving `stable` tag. Alternatively, from the repository root:

```bash
docker build -t root-explorer .
docker run --rm -p 8501:8501 root-explorer
```

Browse to `http://localhost:8501`. Stop with `Ctrl+C`.

### Access host files from Docker

If you select **Path** input in the app, a local filesystem path must refer to a file **inside the container**, not merely a path on your laptop. Mount a directory read-only, for example:

```bash
docker run --rm -p 8501:8501 \
  --mount type=bind,source="/absolute/path/to/data",target=/data,readonly \
  ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

Then enter `/data/example.root` in the app. This also applies to GDML files. With **Upload**, you instead select a file from the browser and do not need a host mount.

**Security:** server-side Path inputs operate as the app's process identity. Do not expose the existing unrestricted local-style Path workflow to untrusted public users. More restrictive public/internal deployment profiles are planned but are not described here as available.

## Further reading

Continue with [Your first plot](getting-started.md), then [ROOT files and objects](user-guide/files-and-objects.md).
