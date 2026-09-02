# root-explorer
A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees in the web browser.

## Run the app

From a checkout, users can launch the app directly:

```bash
streamlit run streamlit_app.py
```

## Development

From the repository root, install the package in editable mode with the developer
tools, including Ruff and pre-commit:

```bash
pip install -e ".[dev]"
```

### Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to run the Ruff checks
defined in `.pre-commit-config.yaml`. The configuration pins the Ruff version and
enables two hooks:

- `ruff-check` runs the configured lint checks and applies safe fixes.
- `ruff-format` checks and applies Ruff's formatting.

Install the hooks once for your local checkout:

```bash
pre-commit install
```

The hooks run automatically on every commit. To run them manually, for example
before opening a pull request, check the complete repository with:

```bash
pre-commit run --all-files
```

To run only one hook, use its hook ID:

```bash
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
```

The hooks may modify files because `ruff-check` is configured with `--fix`. Review
and stage those changes, then run the hooks again or create the commit again.

When changing `.pre-commit-config.yaml`, validate it from the repository root:

```bash
pre-commit validate-config
```

If the Ruff revision is updated, keep it synchronized with the Ruff version in
`pyproject.toml` and `.github/workflows/ruff.yaml` so local hooks and CI use the
same version. Use `pre-commit autoupdate` only when intentionally updating the
hook revisions, and review the resulting configuration before committing it.
