"""Standalone Python source generation for static ROOT Explorer plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import VariableSpec

if TYPE_CHECKING:
    from ..app.plotting import PlotRequest


STATIC_PLOT_TYPES = frozenset({"1D histogram", "2D histogram", "2D scatter"})


def generate_static_plot_code(request: PlotRequest) -> str:
    """Return a self-contained, configuration-specialized Matplotlib script."""

    if request.plot_type not in STATIC_PLOT_TYPES:
        raise ValueError(f"Plot-code download is not available for {request.plot_type!r}.")

    context = request.context
    labels = request.labels
    style = request.style
    title = labels["title"] or (
        f"Event {request.event_index}" if request.event_index is not None else request.plot_type
    )
    selected_items = _request_branches(request)
    read_start = request.event_index if request.event_index is not None else context.entry_start
    read_stop = request.event_index + 1 if request.event_index is not None else context.entry_stop

    lines = [
        '"""Standalone plot exported by ROOT Explorer."""',
        "",
        "import awkward as ak",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import uproot",
    ]
    if request.plot_type == "2D histogram" and style.get("log_z", False):
        lines.append("from matplotlib.colors import LogNorm")
    lines.extend(
        [
            "",
            "",
            "def main() -> None:",
            f"    root_path = {context.root_path!r}  # Change this if the ROOT file is moved.",
            f"    tree_name = {context.event_container_name!r}",
            f"    entry_start = {read_start!r}",
            f"    entry_stop = {read_stop!r}",
            f"    selected_items = {selected_items!r}",
            "",
            "    with uproot.open(root_path) as root_file:",
            "        tree = root_file[tree_name]",
            "        data = tree.arrays(",
            '            selected_items, library="ak", entry_start=entry_start, entry_stop=entry_stop',
            "        )",
            "",
            "    def flatten(values):",
            "        return np.asarray(ak.to_numpy(ak.flatten(values, axis=None)), dtype=float)",
            "",
            "    def align_to_x(values, name):",
            "        if len(values) == len(x):",
            "            return values",
            "        if len(values) == len(x_counts):",
            "            return np.repeat(values, x_counts)",
            '        raise ValueError(f"{name} does not align with x values.")',
            "",
        ]
    )

    x_label = _emit_variable(lines, "x", request.x_spec)
    reference_branch = _reference_branch(request.x_spec)
    lines.extend(
        [
            "",
            "    try:",
            f"        x_counts = np.asarray(ak.to_numpy(ak.num(data[{reference_branch!r}], axis=1)), dtype=int)",
            "    except Exception:",
            "        x_counts = np.ones(len(x), dtype=int)",
            "    if x_counts.sum() != len(x):",
            "        x_counts = np.ones(len(x), dtype=int)",
        ]
    )

    y_label = "entries"
    if request.y_spec is not None:
        lines.append("")
        y_label = _emit_variable(lines, "y", request.y_spec)
        lines.extend(
            [
                "    if x.shape != y.shape:",
                '        raise ValueError("x and y values must have matching shapes.")',
            ]
        )
    else:
        lines.append("    y = None")

    if request.weight_branch is not None:
        lines.extend(
            [
                "",
                f"    weights = align_to_x(flatten(data[{request.weight_branch!r}]), {request.weight_branch!r})",
            ]
        )
    else:
        lines.append("    weights = None")

    color_label = None
    if request.color_spec is not None:
        lines.append("")
        color_label = _emit_variable(lines, "color", request.color_spec)
        lines.append(f"    color = align_to_x(color, {color_label!r})")
    else:
        lines.append("    color = None")

    cut_names: list[tuple[str, float | None, float | None]] = []
    for index, (spec, bounds) in enumerate(zip(request.cut_specs, request.cut_bounds, strict=True)):
        name = f"cut_{index}"
        lines.append("")
        label = _emit_variable(lines, name, spec)
        lines.append(f"    {name} = align_to_x({name}, {label!r})")
        cut_names.append((name, *bounds))

    lines.extend(["", "    mask = np.isfinite(x)"])
    if request.y_spec is not None:
        lines.append("    mask &= np.isfinite(y)")
    if request.weight_branch is not None:
        lines.append("    mask &= np.isfinite(weights)")
    if request.color_spec is not None:
        lines.append("    mask &= np.isfinite(color)")
    for name, lo, hi in cut_names:
        lines.append(f"    mask &= np.isfinite({name})")
        if lo is not None:
            lines.append(f"    mask &= {name} >= {lo!r}")
        if hi is not None:
            lines.append(f"    mask &= {name} <= {hi!r}")
    lines.append("    x = x[mask]")
    if request.y_spec is not None:
        lines.append("    y = y[mask]")
    if request.weight_branch is not None:
        lines.append("    weights = weights[mask]")
    if request.color_spec is not None:
        lines.append("    color = color[mask]")

    if request.plot_type == "2D scatter":
        lines.extend(_scatter_downsampling_lines(request))

    resolved_x_label = labels["custom_x"] or x_label
    resolved_y_label = labels["custom_y"] or y_label
    lines.extend(["", _figure_line(style), "    axes = figure.gca()"])
    lines.extend(
        _plot_lines(
            request,
            x_label=resolved_x_label,
            y_label=resolved_y_label,
            color_label=color_label,
            title=title,
        )
    )
    lines.extend(["", "    plt.show()", "", "", 'if __name__ == "__main__":', "    main()"])
    return "\n".join(lines) + "\n"


def _request_branches(request: PlotRequest) -> list[str]:
    specs = [request.x_spec, request.y_spec, request.color_spec, *request.cut_specs]
    branches = {branch for spec in specs if spec is not None for branch in _spec_branches(spec)}
    if request.weight_branch is not None:
        branches.add(request.weight_branch)
    return sorted(branches)


def _spec_branches(spec: VariableSpec) -> list[str]:
    if spec.mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        return [spec.branch] if spec.branch else []
    return [
        branch for branch in [spec.branch_a, spec.branch_b, spec.branch_c, spec.branch_d] if branch
    ]


def _reference_branch(spec: VariableSpec) -> str:
    branches = _spec_branches(spec)
    if not branches:
        raise ValueError("The x variable has no input branch.")
    return branches[0]


def _emit_variable(lines: list[str], target: str, spec: VariableSpec) -> str:
    """Append direct NumPy/Awkward code for one selected variable and return its label."""

    if spec.mode == "branch":
        branch = _required_branch(spec)
        lines.append(f"    {target} = flatten(data[{branch!r}])")
        return branch

    if spec.mode == "multiplicity":
        branch = _required_branch(spec)
        lines.extend(
            [
                "    try:",
                f"        {target} = np.asarray(ak.to_numpy(ak.num(data[{branch!r}], axis=1)), dtype=float)",
                "    except Exception:",
                f"        {target} = np.ones(len(flatten(data[{branch!r}])), dtype=float)",
            ]
        )
        return f"N({branch})"

    if spec.mode == "nth":
        branch = _required_branch(spec)
        lines.extend(
            [
                "    try:",
                f"        padded = ak.pad_none(data[{branch!r}], {spec.index + 1}, axis=1)",
                f"        {target} = np.asarray(ak.to_numpy(ak.fill_none(padded[:, {spec.index}], np.nan)), dtype=float)",
                "    except Exception:",
                f"        {target} = flatten(data[{branch!r}])",
            ]
        )
        return f"{branch}[{spec.index}]"

    if spec.mode in {"sum", "mean", "max", "min"}:
        branch = _required_branch(spec)
        function = {"sum": "sum", "mean": "mean", "max": "max", "min": "min"}[spec.mode]
        extra = ", mask_identity=False" if spec.mode in {"max", "min"} else ""
        lines.extend(
            [
                "    try:",
                f"        {target} = np.asarray(ak.to_numpy(ak.{function}(data[{branch!r}], axis=1{extra})), dtype=float)",
                "    except Exception:",
                f"        {target} = flatten(data[{branch!r}])",
            ]
        )
        return f"{spec.mode}({branch})"

    branches = _spec_branches(spec)
    if len(branches) < 2:
        raise ValueError(f"Variable mode {spec.mode!r} needs at least two branches.")
    for suffix, branch in zip(["a", "b", "c", "d"], branches, strict=False):
        lines.append(f"    {target}_{suffix} = flatten(data[{branch!r}])")
    lines.extend(
        [
            f"    if {target}_a.shape != {target}_b.shape:",
            f'        raise ValueError({spec.mode!r} + " inputs must have matching shapes.")',
        ]
    )
    if spec.mode == "r_xy":
        lines.append(f"    {target} = np.hypot({target}_a, {target}_b)")
        return f"r({branches[0]},{branches[1]})"
    if spec.mode == "phi_xy":
        lines.append(f"    {target} = np.arctan2({target}_b, {target}_a)")
        return f"phi({branches[0]},{branches[1]})"
    if spec.mode == "ratio":
        lines.extend(
            [
                f"    {target} = np.full(len({target}_a), np.nan)",
                f"    np.divide({target}_a, {target}_b, out={target}, where={target}_b != 0)",
            ]
        )
        return f"{branches[0]}/{branches[1]}"
    if spec.mode == "difference":
        lines.append(f"    {target} = {target}_a - {target}_b")
        return f"{branches[0]}-{branches[1]}"
    if spec.mode == "sum_ab":
        lines.append(f"    {target} = {target}_a + {target}_b")
        return f"{branches[0]}+{branches[1]}"
    if spec.mode == "product":
        lines.append(f"    {target} = {target}_a * {target}_b")
        return f"{branches[0]}*{branches[1]}"
    if spec.mode == "delta_phi":
        lines.append(
            f"    {target} = np.arctan2(np.sin({target}_a - {target}_b), np.cos({target}_a - {target}_b))"
        )
        return f"Δφ({branches[0]},{branches[1]})"
    if spec.mode == "eta_xyz":
        _require_branch_count(spec, branches, 3)
        lines.extend(
            [
                f"    {target}_r = np.hypot({target}_a, {target}_b)",
                f"    {target} = np.full(len({target}_r), np.nan)",
                f"    valid = np.isfinite({target}_r) & np.isfinite({target}_c) & ({target}_r > 0)",
                f"    {target}[valid] = np.arcsinh({target}_c[valid] / {target}_r[valid])",
            ]
        )
        return f"eta({branches[0]},{branches[1]},{branches[2]})"
    if spec.mode == "delta_r":
        _require_branch_count(spec, branches, 4)
        lines.append(
            f"    {target} = np.hypot({target}_a - {target}_c, "
            f"np.arctan2(np.sin({target}_b - {target}_d), np.cos({target}_b - {target}_d)))"
        )
        return f"ΔR({branches[0]},{branches[1]},{branches[2]},{branches[3]})"
    raise ValueError(f"Unsupported variable mode: {spec.mode}")


def _required_branch(spec: VariableSpec) -> str:
    if not spec.branch:
        raise ValueError(f"Variable mode {spec.mode!r} needs a branch.")
    return spec.branch


def _require_branch_count(spec: VariableSpec, branches: list[str], count: int) -> None:
    if len(branches) < count:
        raise ValueError(f"Variable mode {spec.mode!r} needs {count} branches.")


def _scatter_downsampling_lines(request: PlotRequest) -> list[str]:
    max_points = int(request.style["max_points"])
    return [
        "",
        f"    if len(x) > {max_points}:",
        "        rng = np.random.default_rng(12345)",
        f"        indices = rng.choice(len(x), size={max_points}, replace=False)",
        "        x = x[indices]",
        "        y = y[indices]",
        "        if color is not None:",
        "            color = color[indices]",
    ]


def _figure_line(style: dict) -> str:
    return f"    figure = plt.figure(figsize=({style['fig_width']!r}, {style['fig_height']!r}))"


def _plot_lines(
    request: PlotRequest, *, x_label: str, y_label: str, color_label: str | None, title: str
) -> list[str]:
    style = request.style
    lines: list[str] = []
    if request.plot_type == "1D histogram":
        lines.append(f"    axes.hist(x, bins={style['bins_x']!r}, weights=weights)")
        if style.get("log_y", False):
            lines.append('    axes.set_yscale("log")')
    elif request.plot_type == "2D histogram":
        lines.extend(
            [
                f"    histogram, xedges, yedges = np.histogram2d(x, y, bins=({style['bins_x']!r}, {style['bins_y']!r}), weights=weights)",
                "    norm = None",
            ]
        )
        if style.get("log_z", False):
            lines.extend(
                [
                    "    positive = histogram[np.isfinite(histogram) & (histogram > 0)]",
                    "    if len(positive) > 1 and positive.max() > positive.min():",
                    "        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))",
                ]
            )
        colorbar_label = (
            "entries" if request.weight_branch is None else f"sum weights ({request.weight_branch})"
        )
        lines.extend(
            [
                '    mesh = axes.pcolormesh(xedges, yedges, histogram.T, shading="auto", norm=norm)',
                f"    figure.colorbar(mesh, ax=axes, label={colorbar_label!r})",
            ]
        )
    elif request.plot_type == "2D scatter":
        if request.color_spec is None:
            lines.append(
                f"    axes.scatter(x, y, s={style['marker_size_2d']!r}, alpha={style['opacity_2d']!r})"
            )
        else:
            cmap = str(style.get("colorscale", "Viridis")).lower()
            if style.get("reverse_colorscale", False):
                cmap += "_r"
            color_min = _optional_float(style.get("color_min_text", ""))
            color_max = _optional_float(style.get("color_max_text", ""))
            lines.extend(
                [
                    "    scatter = axes.scatter(",
                    f"        x, y, s={style['marker_size_2d']!r}, alpha={style['opacity_2d']!r},",
                    f"        c=color, cmap={cmap!r}, vmin={color_min!r}, vmax={color_max!r},",
                    "    )",
                    f"    figure.colorbar(scatter, ax=axes, label={color_label!r})",
                ]
            )
    lines.extend(
        [
            f"    axes.set_xlabel({x_label!r})",
            f"    axes.set_ylabel({y_label!r})",
            f"    axes.set_title({title!r})",
        ]
    )
    if style.get("equal_aspect", False):
        lines.append('    axes.set_aspect("equal", adjustable="box")')
    if style.get("grid", False):
        lines.append("    axes.grid(alpha=0.3)")
    return lines


def _optional_float(value: object) -> float | None:
    try:
        return float(str(value).strip()) if str(value).strip() else None
    except ValueError:
        return None
