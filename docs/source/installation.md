# Installation and launching

**Docker is the recommended way to run ROOT Explorer.** It bundles the application and its Python dependencies, so you do not need to install Python, ROOT or Uproot on your computer. The user interface opens in your browser; the container processes the data.

## Recommended: Run with Docker

Install [Docker](https://docs.docker.com/get-started/get-docker/) and start ROOT Explorer using the published stable image:

```bash
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. In the sidebar choose **Upload** to select a ROOT file from your computer. You can immediately follow [Your first plot](getting-started.md); no repository checkout is required.

Stop the container with `Ctrl+C` in the terminal. The `stable` tag moves with releases; pin a **versioned image tag** for reproducible deployments.

### Reading files from the host filesystem

Use a **read-only bind mount** if you want to load files by path instead of uploading them:

```bash
docker run --rm -p 8501:8501 \
  --mount type=bind,source="/absolute/path/to/data",target=/data,readonly \
  ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

For example, a host file at `/absolute/path/to/data/example.root` appears as `/data/example.root` **inside the container**. Select **Path** in ROOT Explorer and enter `/data/example.root`. A path on your laptop is not automatically visible inside Docker. The same distinction applies to GDML geometry paths.

With **Upload**, you select a file in the browser and do not need to mount anything. Uploaded data is processed by the running application.

**Security:** Path access runs with the server process's permissions. Do not expose the current unrestricted local-style Path workflow to untrusted public users. Restricted hosting profiles are planned but are not currently active; see [Deployment](deployment.md).

### Exporting plots from Docker

Plots are exported with the **Download** buttons, which send files through
the browser to your computer. No output directory or writable mount is needed.
See [Exporting plots](user-guide/exporting.md) for formats and standalone
Python scripts.

### Alternative: Build the image locally

To build a container from the repository instead of pulling the published image:

```bash
git clone https://github.com/AlexanderPeschBerrocal/root-explorer.git
cd root-explorer
docker build -t root-explorer .
docker run --rm -p 8501:8501 root-explorer
```

## Alternative: Install from source (Python 3.12)

Source installation is useful for developers and users who prefer a local Python environment. It requires **Python 3.12.4 or a later 3.12 release**, unlike the Docker method.

```bash
git clone https://github.com/AlexanderPeschBerrocal/root-explorer.git
cd root-explorer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
root-explorer
```

On Windows, use the activation command appropriate to your shell. Streamlit normally serves the application at `http://localhost:8501`. A local source installation can read files using their normal filesystem paths.

### The `root-explorer` command

Installing the package provides a `root-explorer` command, which starts the application from any directory. Pass a ROOT file to open it directly; relative paths are resolved against the directory in which you run the command, and `root://` or `http(s)://` URLs are passed unchanged:

```bash
root-explorer                         # start without a file
root-explorer path/to/file.root       # open a file on start-up
root-explorer file.root --port 8600   # use another port
root-explorer file.root --no-browser  # do not open a browser window
root-explorer --server.address 127.0.0.1 file.root  # forward a Streamlit setting
```

Only one ROOT file can be given; additional file arguments, also after `--`,
are rejected with an error.

`python -m root_explorer` is equivalent. Further options are passed to Streamlit, using either `--server.address=127.0.0.1` or `--server.address 127.0.0.1`, before or after the ROOT filename; run `root-explorer --help` for an overview. Boolean Streamlit settings also require a value, for example `--server.headless true`. The command applies the project's Streamlit defaults, such as the 4096 MiB upload limit, independently of the current directory. Starting the app with `streamlit run streamlit_app.py` from the repository root continues to work.

`http://` and `https://` sources work with the default installation and the
Docker image, which include the required `aiohttp` and `requests` packages.
`root://` (XRootD) access additionally requires the optional `fsspec-xrootd`
backend and the XRootD Python bindings in the **server's Python environment**:
install them with `python -m pip install "root-explorer[xrootd]"` (or `-e
".[xrootd]"` in a source checkout); the bindings may need to be compiled. They are
not included in the Docker image. Without the backend, opening a `root://` source
shows an error naming what to install; alternatively, download the file
separately and use a local Path or Upload. Remote servers may also require
credentials and network access configured on the server.

After changing the package version or dependencies in a source checkout, use `uv lock` and `uv sync --locked` in a uv-managed environment, or reinstall with `python -m pip install -e .` for pip-managed environments.

For development and testing, use `uv sync --locked` and consult [Contributing](contributing.md). Ordinary `pip install -e .` remains supported for users who do not need the locked development environment.

## Next step

Follow [Your first plot](getting-started.md) and then [ROOT files and objects](user-guide/files-and-objects.md).
