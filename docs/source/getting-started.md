# Your first ROOT Explorer plot

This tutorial shows how to produce a histogram of synthetic muon transverse momenta (`Muon_pt`) from an `Events` ROOT TTree. The example uses the same synthetic distributions and branches as the documentation screenshots, so you can follow along with the illustrated interface. You can also use your own ROOT file.

## Launch ROOT Explorer

The easiest route is **Docker**, without a local Python installation:

```bash
docker run --rm -p 8501:8501 ghcr.io/alexanderpeschberrocal/root-explorer:stable
```

Open [http://localhost:8501](http://localhost:8501) in your browser. See [Installation](installation.md) for host-directory mounts, versioned image tags, and the optional Python source setup.

## Prepare a ROOT file

If you already have a ROOT file containing a TTree or RNTuple, you can use it directly with **Upload**. Otherwise, generate `demo.root` with the same reproducible, entirely synthetic data recipe used for the screenshots.

The example generator needs `numpy`, `awkward`, and `uproot`. These packages are required only to **create the example file**, not to run ROOT Explorer through Docker:

```bash
python3 -m pip install numpy awkward uproot
```

Run the following Python code in the directory where you want the example file:

```python
import awkward as ak
import numpy as np
import uproot

rng = np.random.default_rng(20261019)
events = 6000

# One muon count per event; transverse momentum and pseudorapidity
# are jagged collections containing that event's muons.
counts = rng.poisson(1.6, events)
pt = rng.gamma(2.5, 17.0, counts.sum()).astype(np.float32)
eta = rng.normal(0.0, 1.25, counts.sum()).astype(np.float32)
met = rng.exponential(52.0, events).astype(np.float32)

fields = {
    "MET_pt": met,
    "nMuon": counts.astype(np.int32),
    "Muon_pt": ak.unflatten(pt, counts),
    "Muon_eta": ak.unflatten(eta, counts),
    "genWeight": np.ones(events, dtype=np.float32),
}
with uproot.recreate("demo.root") as root:
    root.mktree("Events", fields)
    root["Overview/h_met"] = np.histogram(met, bins=45, range=(0, 250))
```

These values are **toy distributions, not experimental data**. The code mirrors the repository's [screenshot demo generator](https://github.com/AlexanderPeschBerrocal/root-explorer/blob/main/docs/scripts/generate_demo_data.py); in a source checkout, you can instead run `python docs/scripts/generate_demo_data.py --output demo.root`. The generator is also responsible for the persisted `Overview/h_met` histogram used in the ROOT object browser.

## Open the file and make a histogram

1. In the **Input** sidebar, choose **Upload** and select `demo.root` from your computer.
2. Select the **Events** event container and keep the full **Read range** of entries `0` to `6000`.
3. In the **Plotter** tab, select **1D histogram**, choose **Direct branch values: A** and `Muon_pt` for the x-axis, and click **Make plot**. This reproduces the main variable in the overview screenshot.
4. Change the x branch to `Muon_eta` to explore pseudorapidity. You can enable **Enable fit** to try the fitting interface.
5. Select **2D histogram**, with `Muon_pt` on the x-axis and `Muon_eta` on the y-axis, to reproduce the two-dimensional screenshot.
6. Optionally, return to a **1D histogram** and choose `nMuon`. Unlike `Muon_pt` (one entry **per muon**), `nMuon` has one integer value **per event**.

The plot represents the **most recently generated configuration**. Changing a control does not regenerate the plot until you click **Make plot**.

You can also use **Event inspection** to select an event and examine its individual muons.

If you prefer **Path** rather than Upload while running Docker, mount a read-only host directory and use the **container** path as described in [Installation](installation.md).

The ROOT object browser can inspect persisted ROOT objects independently of event plotting; see [ROOT files and objects](user-guide/files-and-objects.md).

## Continue exploring

Read [Plotting](user-guide/plotting.md) and [Variables and collections](user-guide/variables-and-collections.md) for more workflows. Start with a limited entry range when exploring a large dataset.

ROOT Explorer is intended for exploratory work. Preserve final physics selections and plots in reproducible analysis code.
