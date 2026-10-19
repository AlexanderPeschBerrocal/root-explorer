# Your first ROOT Explorer plot

This tutorial shows how to produce a histogram from a ROOT `TTree` named `Events`, with a numerical branch called `MET_pt`. You can use your own ROOT file, or create the entirely synthetic example below.

## Launch ROOT Explorer

The easiest route is **Docker**, without a local Python installation:

```bash
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

Open [http://localhost:8501](http://localhost:8501) in your browser. See [Installation](installation.md) for host-directory mounts, versioned image tags, and the optional Python source setup.

## Prepare a ROOT file

If you already have a ROOT file containing a TTree or RNTuple, you can use it directly with **Upload**. Otherwise, generate `demo.root` using Python with `numpy` and `uproot`. These are only needed for **generating this example**, not for running the Docker container.

```bash
python3 -m pip install numpy uproot
```

Run the following in Python in the directory where you want the example file:

```python
import numpy as np
import uproot

rng = np.random.default_rng(42)
with uproot.recreate("demo.root") as root_file:
    root_file.mktree("Events", {"MET_pt": "float32", "nMuon": "int32"})
    root_file["Events"].extend(
        {
            "MET_pt": rng.exponential(40.0, 5000).astype("float32"),
            "nMuon": rng.poisson(1.5, 5000).astype("int32"),
        }
    )
```

These are **synthetic values**, not experimental data. If you run ROOT Explorer from Python source, `numpy` and `uproot` are already part of its dependencies.

## Open the file and make a histogram

1. In the **Input** sidebar, choose **Upload** and select `demo.root` from your computer.
2. Choose the **Events** event container. Keep the default **Read range**, or select entries `0` through `5000` to use the complete example.
3. In the **Plotter** tab, select a **1D histogram**, choose the direct `MET_pt` branch for the x-axis and click **Make plot**.
4. Switch to `nMuon` to display the discrete event-by-event muon multiplicity, then click **Make plot** again.

The plot represents the **most recently generated configuration**. Changing a control does not regenerate the plot until you click **Make plot**.

If you prefer **Path** rather than Upload while running Docker, mount a read-only host directory and use the **container** path as described in [Installation](installation.md).

The ROOT object browser can inspect persisted ROOT objects independently of event plotting; see [ROOT files and objects](user-guide/files-and-objects.md).

## Continue exploring

Read [Plotting](user-guide/plotting.md) and [Variables and collections](user-guide/variables-and-collections.md) for more workflows. Start with a limited entry range when exploring a large dataset.

ROOT Explorer is intended for exploratory work. Preserve final physics selections and plots in reproducible analysis code.
