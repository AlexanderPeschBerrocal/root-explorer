"""Read persisted ROOT histograms into the canonical Scikit-HEP representation."""

from __future__ import annotations

from dataclasses import dataclass

import hist
import numpy as np
import uproot


class UnsupportedHistogramError(ValueError):
    """The selected ROOT object cannot be displayed as a 1D or 2D histogram."""


@dataclass(frozen=True)
class PersistedHistogram:
    """A canonical histogram with ROOT-specific display metadata.

    `histogram` contains all bins, including flow bins. `profile_variances`
    contains visible *variances on the mean* for profiles, as calculated by
    Uproot/ROOT; they must not be interpreted as Poisson count variances.
    """

    histogram: hist.Hist
    classname: str
    kind: str
    title: str
    x_label: str
    y_label: str
    colorbar_label: str
    profile_variances: np.ndarray | None = None

    @property
    def is_profile(self) -> bool:
        return self.kind == "MEAN"


def _axis_title(obj: object, member: str, fallback: str) -> str:
    """Prefer the stored ROOT axis title over semicolon-delimited title text."""
    try:
        title = str(obj.member(member).member("fTitle") or "").strip()  # type: ignore[attr-defined]
    except (AttributeError, KeyError, ValueError):
        title = ""
    return title or fallback


def _profile2d_fallback(obj: object) -> hist.Hist:
    """Adapt a ROOT TProfile2D while Uproot's to_boost is unimplemented.

    This preserves the bin means and their ROOT-standard uncertainty on the
    mean, including flow bins. The Weight storage is used *only* to store these
    two arrays; it does not imply that profile means are sums of weights.
    """
    axes = [
        hist.axis.Variable(
            np.asarray(obj.axis(i).edges(), dtype=float),  # type: ignore[attr-defined]
            name=("x", "y")[i],
            label=str(obj.axis(i).member("fTitle") or ""),  # type: ignore[attr-defined]
            underflow=True,
            overflow=True,
        )
        for i in range(2)
    ]
    result = hist.Hist(*axes, storage=hist.storage.Weight())
    view = result.view(flow=True)
    view.value[...] = np.asarray(obj.values(flow=True), dtype=float)  # type: ignore[attr-defined]
    view.variance[...] = np.asarray(obj.variances(flow=True), dtype=float)  # type: ignore[attr-defined]
    return result


def histogram_from_object(obj: object) -> PersistedHistogram:
    """Adapt a deserialised Uproot histogram without rebinning its contents."""
    classname = str(getattr(obj, "classname", type(obj).__name__))
    if not classname.startswith(("TH1", "TH2", "TProfile")) or classname.startswith(
        ("TH2Poly", "TH3", "TProfile3D")
    ):
        raise UnsupportedHistogramError(
            f"ROOT class {classname} is not supported for persisted histogram plotting."
        )
    kind = str(getattr(obj, "kind", ""))
    if kind not in {"COUNT", "MEAN"}:
        raise UnsupportedHistogramError(f"Unrecognised histogram semantics for {classname}: {kind}")

    try:
        try:
            histogram = obj.to_hist()  # type: ignore[attr-defined]
        except NotImplementedError:
            if classname != "TProfile2D":
                raise
            histogram = _profile2d_fallback(obj)
    except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
        raise UnsupportedHistogramError(f"Cannot convert {classname} to hist.Hist: {exc}") from exc

    if not isinstance(histogram, hist.Hist) or histogram.ndim not in (1, 2):
        raise UnsupportedHistogramError(f"{classname} is not a supported 1D or 2D histogram.")

    parts = str(getattr(obj, "title", "") or "").split(";")
    title = parts[0].strip()
    x_label = _axis_title(obj, "fXaxis", parts[1].strip() if len(parts) > 1 else "")
    y_label = _axis_title(obj, "fYaxis", parts[2].strip() if len(parts) > 2 else "")
    z_label = _axis_title(obj, "fZaxis", parts[3].strip() if len(parts) > 3 else "")

    profile_variances = (
        np.asarray(obj.variances(flow=False), dtype=float).copy() if kind == "MEAN" else None
    )
    return PersistedHistogram(
        histogram=histogram,
        classname=classname,
        kind=kind,
        title=title,
        x_label=x_label or "x",
        y_label=y_label
        or (
            "Mean"
            if kind == "MEAN" and histogram.ndim == 1
            else "Entries"
            if histogram.ndim == 1
            else "y"
        ),
        colorbar_label=z_label or ("Mean" if kind == "MEAN" else "Bin content"),
        profile_variances=profile_variances,
    )


def read_histogram(root_path: str, object_path: str) -> PersistedHistogram:
    """Load only the selected histogram, honouring nested paths and ROOT cycles."""
    with uproot.open(root_path) as root_file:
        return histogram_from_object(root_file[object_path])
