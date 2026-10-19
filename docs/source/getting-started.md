# Your first ROOT Explorer plot

ROOT Explorer lets you inspect a ROOT file without writing a plotting script. This walk-through uses a **local ROOT file** containing a TTree called `Events` with a numeric branch such as `MET_pt`. You can use your own ROOT file or generate the small synthetic example below.

## Create a demonstration ROOT file

Install ROOT Explorer as described in [Installation](installation.md), then, from the repository root, create `demo.root` with Uproot (already installed as a dependency):

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

These are **synthetic demonstration values**, not experimental data.

## Open and inspect the file

1. Launch the application with `streamlit run streamlit_app.py` and open the local URL shown in the terminal (normally `http://localhost:8501`).
2. In the **Input** sidebar, choose **Path** and enter the absolute path to `demo.root`. Alternatively choose **Upload** and select the file in your browser.
3. Choose the **Events** event container and keep the default **Read range**, or select entries `0` through `5000` to use the whole example.
4. In the **Plotter** tab, select a 1D histogram, choose the direct `MET_pt` branch for the x-axis and click **Make plot**.
5. Try `nMuon` to inspect the event-by-event muon multiplicity. The displayed plot represents the **last generated** settings until you click **Make plot** again.

The ROOT object browser can inspect file contents independently of your choice of plot variables. The [files and objects](user-guide/files-and-objects.md) guide explains the distinction.

## Continue exploring

Read [plotting](user-guide/plotting.md) for plot types, and [variables and collections](user-guide/variables-and-collections.md) for per-event scalars, jagged particle collections and reductions. For larger samples, first select a small read range while configuring the plot.

ROOT Explorer is intended for exploration. Preserve important final selections and plots in reproducible analysis code.
