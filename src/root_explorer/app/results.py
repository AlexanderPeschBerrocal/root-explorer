"""Streamlit result display and small application file helpers."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ..analysis.statistics import QuantileResult, ThresholdResult
from ..models import FieldInfo
from ..plotting.plots import (
    ColorClippingResult,
    CorrelationResult,
    InvalidPlotResolutionError,
    matplotlib_figure_bytes,
    plotly_figure_html,
    safe_filename,
    save_matplotlib_figure,
    save_plotly_figure,
    validate_raster_resolution,
)
from .controls import export_controls, selected_static_formats
from .tables import render_dataframe


def show_cached_plot_result(
    result_key: str,
    *,
    export_prefix: str,
    save_button_key: str,
    make_plot_label: str,
    make_plot_shortcut: str,
    empty_message: str,
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
        f"Showing the most recently generated plot. Click the {make_plot_label} button again "
        f"or press {make_plot_shortcut} to update it after changing settings."
    )

    summary = result.get("summary")
    if summary is not None:
        render_dataframe(summary)

    correlation_result = result.get("correlation_result")
    if isinstance(correlation_result, CorrelationResult):
        _show_correlation_result(correlation_result)

    threshold_result = result.get("threshold_result")
    if isinstance(threshold_result, ThresholdResult):
        _show_threshold_result(threshold_result)

    quantile_result = result.get("quantile_result")
    if isinstance(quantile_result, QuantileResult):
        _show_quantile_result(quantile_result)

    for warning in result.get("warnings", []):
        st.warning(warning)

    preview = result.get("preview")
    if preview is not None and len(preview) > 0:
        st.markdown("#### Data preview", anchors=False)
        render_dataframe(preview)

    fig = result["figure"]
    kind = result["kind"]

    if kind == "plotly":
        st.plotly_chart(fig, width="stretch")
    elif kind == "matplotlib":
        st.pyplot(fig, width="stretch")
    else:
        st.error(f"Unknown cached plot kind: {kind}")
        return

    color_clipping_result = result.get("color_clipping_result")
    if isinstance(color_clipping_result, ColorClippingResult):
        _show_color_clipping_result(color_clipping_result)

    export = export_controls(export_prefix, kind, default_save_name=result["save_name"])
    outdir = Path(export["save_directory"])
    save_name = export["save_name"]
    download_name = safe_filename(save_name)
    st.caption("Save writes to the cluster; Download sends the file to this browser.")

    if kind == "plotly":
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
    else:
        formats = selected_static_formats(export)
        export_dpi = int(export.get("export_dpi", 300))
        resolution_error = None
        if "png" in formats:
            try:
                validate_raster_resolution(fig, export_dpi)
            except InvalidPlotResolutionError as exc:
                resolution_error = str(exc)
                st.warning(resolution_error)
        code = result.get("code")
        export_columns = st.columns(1 + len(formats) + int(code is not None))
        with export_columns[0]:
            if st.button(
                result.get("save_button_label", "Save plot"),
                key=save_button_key,
                width="stretch",
                disabled=resolution_error is not None,
            ):
                saved_paths = save_matplotlib_figure(
                    fig, outdir, save_name, formats, dpi=export_dpi
                )
                if saved_paths:
                    st.success("Saved: " + ", ".join(str(path) for path in saved_paths))
                else:
                    st.warning("No static export format selected. Select PNG, PDF, or both.")
        image_columns = export_columns[1:]
        if code is not None:
            with export_columns[1]:
                st.download_button(
                    "Download plot code",
                    data=code,
                    file_name=f"{download_name}.py",
                    mime="text/x-python",
                    key=f"{save_button_key}_download_code",
                    on_click="ignore",
                    width="stretch",
                )
            image_columns = export_columns[2:]
        mime_types = {"png": "image/png", "pdf": "application/pdf"}
        for column, file_format in zip(image_columns, formats, strict=True):
            with column:
                st.download_button(
                    f"Download {file_format.upper()}",
                    data=lambda fmt=file_format: matplotlib_figure_bytes(fig, fmt, dpi=export_dpi),
                    file_name=f"{download_name}.{file_format}",
                    mime=mime_types[file_format],
                    key=f"{save_button_key}_download_{file_format}",
                    on_click="ignore",
                    width="stretch",
                    disabled=file_format == "png" and resolution_error is not None,
                )


def color_clipping_summary(result: ColorClippingResult) -> str | None:
    """Return a concise description of actually clipped scatter colour values."""

    parts: list[str] = []
    if result.below_count > 0 and result.color_min is not None:
        noun = "point" if result.below_count == 1 else "points"
        text = f"{result.below_count} {noun} below color min {result.color_min:g}"
        if result.below_weight is not None:
            text += f" (sum of weights: {result.below_weight:g})"
        parts.append(text)
    if result.above_count > 0 and result.color_max is not None:
        noun = "point" if result.above_count == 1 else "points"
        text = f"{result.above_count} {noun} above color max {result.color_max:g}"
        if result.above_weight is not None:
            text += f" (sum of weights: {result.above_weight:g})"
        parts.append(text)
    return "Color clipping: " + " · ".join(parts) if parts else None


def _show_color_clipping_result(result: ColorClippingResult) -> None:
    summary = color_clipping_summary(result)
    if summary is not None:
        st.caption(summary)


def _show_threshold_result(result: ThresholdResult) -> None:
    """Display one cached threshold calculation next to the plot summary."""

    st.markdown("#### Threshold calculator", anchors=False)
    total_text = f"Selected values: {result.total_count}"
    if result.weighted:
        assert result.total_weight is not None
        total_text += f" · Total sum of weights: {result.total_weight:g}"
    st.caption(total_text)
    rows: list[dict[str, object]] = [
        {
            "selection": f"x ≤ {result.threshold:g}",
            "count": result.below_count,
            "fraction": result.below_fraction,
        },
        {
            "selection": f"x > {result.threshold:g}",
            "count": result.above_count,
            "fraction": result.above_fraction,
        },
    ]
    if result.weighted:
        rows[0]["sum of weights"] = result.below_weight
        rows[1]["sum of weights"] = result.above_weight
        rows[0]["weighted fraction"] = rows[0].pop("fraction")
        rows[1]["weighted fraction"] = rows[1].pop("fraction")
    dataframe = pd.DataFrame(rows)
    render_dataframe(dataframe)

    if result.total_count == 0:
        st.warning("The selected sample is empty; threshold fractions are unavailable.")
    elif result.weighted and result.below_fraction is None:
        st.warning(
            "The total sum of weights is zero; weighted fractions are unavailable. "
            "Counts and signed sums of weights are still shown."
        )
    elif result.weighted and result.has_negative_weights:
        st.caption("Weighted fractions use signed sums of weights and may lie outside 0 to 1.")


def _show_quantile_result(result: QuantileResult) -> None:
    """Display one cached quantile calculation."""

    st.markdown("#### Quantile calculator", anchors=False)
    value_text = "unavailable" if result.value is None else f"{result.value:g}"
    dataframe = pd.DataFrame([{"percentile (%)": result.percentile, "quantile": value_text}])
    render_dataframe(dataframe)
    if result.total_count == 0:
        st.warning("The selected sample is empty; the quantile is unavailable.")
    elif result.weighted and result.has_negative_weights:
        st.warning(
            "Weighted quantiles are unavailable for signed weights because the signed "
            "cumulative weight is not a monotonic empirical distribution."
        )
    elif result.weighted and result.value is None:
        st.warning("The total non-negative weight is zero; the weighted quantile is unavailable.")


def _show_correlation_result(result: CorrelationResult) -> None:
    """Display pairwise statistics for the full filtered 2D-scatter sample."""

    st.markdown("#### Correlations", anchors=False)
    dataframe = pd.DataFrame(
        [
            {
                "Pearson r": result.pearson_r,
                "Spearman \u03c1": result.spearman_rho,
            }
        ]
    )
    render_dataframe(dataframe)


def collection_summary(field_info: list[FieldInfo], *, item_name: str = "field") -> pd.DataFrame:
    rows = []
    by_collection: dict[str, list[FieldInfo]] = {}
    for field in field_info:
        by_collection.setdefault(field.collection, []).append(field)
    for collection, items in sorted(by_collection.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fields = ", ".join(item.field for item in items[:12])
        if len(items) > 12:
            fields += ", ..."
        rows.append(
            {
                "collection": collection,
                f"{item_name}s": len(items),
                f"example {item_name}s": fields,
            }
        )
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
