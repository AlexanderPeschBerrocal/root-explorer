"""Streamlit result display and small application file helpers."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..models import BranchInfo
from ..plotting.plots import (
    matplotlib_figure_bytes,
    plotly_figure_html,
    safe_filename,
    save_matplotlib_figure,
    save_plotly_figure,
)


def show_cached_plot_result(
    result_key: str, outdir: Path, *, save_button_key: str, empty_message: str
) -> None:
    """Display the most recently generated plot and provide a save button.

    Streamlit buttons are momentary: clicking a save button triggers a rerun where
    the original "Make plot" button is no longer pressed. Keeping the last
    figure in session state lets the save button exist on that rerun.
    """

    result = st.session_state.get(result_key)
    if result is None:
        st.info(empty_message)
        return

    st.caption(
        "Showing the most recently generated plot. Click the make-plot button again to update it after changing settings."
    )

    summary = result.get("summary")
    if summary is not None:
        st.dataframe(summary, width="stretch")

    preview = result.get("preview")
    if preview is not None and len(preview) > 0:
        st.markdown("#### Data preview")
        st.dataframe(preview, width="stretch")

    fig = result["figure"]
    save_name = result["save_name"]
    download_name = safe_filename(save_name)

    if result["kind"] == "plotly":
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### Export")
        st.caption("Save writes to the cluster; Download sends the file to this browser.")
        save_col, download_col = st.columns(2)
        with save_col:
            if st.button(
                result.get("save_button_label", "Save HTML"), key=save_button_key, width="stretch"
            ):
                saved_path = save_plotly_figure(fig, outdir, save_name)
                st.success(f"Saved: {saved_path}")
        with download_col:
            st.download_button(
                "Download HTML",
                data=lambda: plotly_figure_html(fig),
                file_name=f"{download_name}.html",
                mime="text/html",
                key=f"{save_button_key}_download_html",
                on_click="ignore",
                width="stretch",
            )
    elif result["kind"] == "matplotlib":
        st.pyplot(fig)
        st.markdown("#### Export")
        st.caption("Save writes to the cluster; Download sends the file to this browser.")
        formats = result.get("save_formats", ["png"])
        export_columns = st.columns(1 + max(len(formats), 1))
        with export_columns[0]:
            if st.button(
                result.get("save_button_label", "Save plot"), key=save_button_key, width="stretch"
            ):
                saved_paths = save_matplotlib_figure(fig, outdir, save_name, formats)
                if saved_paths:
                    st.success("Saved: " + ", ".join(str(path) for path in saved_paths))
                else:
                    st.warning(
                        "No static export format selected. Select PNG, PDF, or both before making the plot."
                    )
        mime_types = {"png": "image/png", "pdf": "application/pdf"}
        for column, file_format in zip(export_columns[1:], formats):
            with column:
                st.download_button(
                    f"Download {file_format.upper()}",
                    data=lambda fmt=file_format: matplotlib_figure_bytes(fig, fmt),
                    file_name=f"{download_name}.{file_format}",
                    mime=mime_types[file_format],
                    key=f"{save_button_key}_download_{file_format}",
                    on_click="ignore",
                    width="stretch",
                )
    else:
        st.error(f"Unknown cached plot kind: {result['kind']}")


def collection_summary(branch_info: list[BranchInfo]) -> pd.DataFrame:
    rows = []
    by_collection: dict[str, list[BranchInfo]] = {}
    for b in branch_info:
        by_collection.setdefault(b.collection, []).append(b)
    for collection, items in sorted(by_collection.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fields = ", ".join(item.field for item in items[:12])
        if len(items) > 12:
            fields += ", ..."
        rows.append({"collection": collection, "branches": len(items), "example fields": fields})
    return pd.DataFrame(rows)


def store_uploaded_file(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".root"
    data = bytes(uploaded.getbuffer())
    digest = hashlib.sha256(data).hexdigest()[:12]
    temp_dir = Path(tempfile.gettempdir()) / "root_explorer_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{safe_filename(Path(uploaded.name).stem)}_{digest}{suffix}"
    path.write_bytes(data)
    return str(path)
