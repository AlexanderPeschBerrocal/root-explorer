"""Event-ranking calculations used by the Streamlit application."""

from __future__ import annotations

from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd
import streamlit as st

from ..io.root import read_branch
from .arrays import multiplicity_per_event, safe_float_array, to_numpy_flat


@st.cache_data(show_spinner=True, max_entries=32)
def compute_event_metric(path: str, tree_name: str, branch: str, mode: str, entry_start: int,
                         entry_stop: int, threshold: Optional[float], abs_value: bool):
    arr = read_branch(path, tree_name, branch, entry_start, entry_stop)
    if mode == "multiplicity":
        values = multiplicity_per_event(arr)
    else:
        vals = ak.values_astype(arr, np.float64)
        if abs_value:
            vals = abs(vals)
        try:
            ak.num(vals, axis=1)
            is_jagged = True
        except Exception:
            is_jagged = False

        if is_jagged:
            if mode == "sum":
                values = ak.to_numpy(ak.sum(vals, axis=1)).astype(float)
            elif mode == "mean":
                values = ak.to_numpy(ak.mean(vals, axis=1)).astype(float)
            elif mode == "max":
                values = ak.to_numpy(ak.max(vals, axis=1, mask_identity=False)).astype(float)
            elif mode == "min":
                values = ak.to_numpy(ak.min(vals, axis=1, mask_identity=False)).astype(float)
            elif mode == "count_above":
                values = ak.to_numpy(ak.sum(vals > (threshold or 0.0), axis=1)).astype(float)
            elif mode == "count_below":
                values = ak.to_numpy(ak.sum(vals < (threshold or 0.0), axis=1)).astype(float)
            else:
                raise ValueError(f"Unknown metric: {mode}")
        else:
            flat = safe_float_array(to_numpy_flat(vals))
            if mode in {"sum", "mean", "max", "min"}:
                values = flat
            elif mode == "count_above":
                values = (flat > (threshold or 0.0)).astype(float)
            elif mode == "count_below":
                values = (flat < (threshold or 0.0)).astype(float)
            else:
                raise ValueError(f"Unknown metric: {mode}")
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    return np.arange(entry_start, entry_start + len(values), dtype=int), values


def metric_dataframe(event_indices: np.ndarray, values: np.ndarray, descending: bool, max_rows: int) -> pd.DataFrame:
    mask = np.isfinite(values)
    df = pd.DataFrame({"event": event_indices[mask].astype(int), "metric": values[mask]})
    return df.sort_values("metric", ascending=not descending).head(max_rows).reset_index(drop=True)
