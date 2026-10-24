"""Capture reproducible, legible screenshots of actual Streamlit plot output.

Requires the optional `.[screenshots]` extras and Playwright Chromium.
Never overwrite a reviewed image from CI: GitHub Actions uploads only artefacts.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from generate_demo_data import generate
from PIL import Image
from playwright.sync_api import Locator, Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "overview.png",
    "histogram-fit.png",
    "plot-2d.png",
    "object-browser.png",
    "event-inspection.png",
)
THEMES = ("dark", "light")
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900
SCREENSHOT_SCALE = 2  # Retina-style captures (2880 x 1800 physical pixels)


def screenshot_name(name: str, theme: str) -> str:
    """Preserve the reviewed dark filenames; suffix additional light captures."""
    if theme not in THEMES:
        raise ValueError(f"Unsupported screenshot theme: {theme}")
    if name not in NAMES:
        raise ValueError(f"Unknown screenshot: {name}")
    return name if theme == "dark" else f"{Path(name).stem}-light.png"


def matplotlib_rc(theme: str) -> str:
    """Give figures the same theme as their Streamlit surroundings."""
    if theme not in THEMES:
        raise ValueError(f"Unsupported screenshot theme: {theme}")
    # matplotlibrc treats unquoted '#' as a comment, including in hex colours.
    background = '"#0e1117"' if theme == "dark" else '"#ffffff"'
    foreground = '"#fafafa"' if theme == "dark" else '"#242424"'
    grid = '"#606975"' if theme == "dark" else '"#d0d4da"'
    return (
        "\n".join(
            (
                f"figure.facecolor: {background}",
                f"axes.facecolor: {background}",
                f"savefig.facecolor: {background}",
                f"axes.edgecolor: {foreground}",
                f"axes.labelcolor: {foreground}",
                f"text.color: {foreground}",
                f"xtick.color: {foreground}",
                f"ytick.color: {foreground}",
                f"grid.color: {grid}",
            )
        )
        + "\n"
    )


def wait_for_server(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Streamlit exited before becoming healthy")
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Streamlit did not become healthy: {base_url}")


def plot_image(tab: Locator) -> Locator:
    """Identify the rendered Matplotlib plot, never a button/export control."""
    image = tab.locator('[data-testid="stImage"] img').first
    expect(image).to_be_visible(timeout=60000)
    expect(image).to_have_attribute("src", re.compile(r".+"), timeout=60000)
    return image


def wait_for_clean_render(page: Page, target: Locator) -> None:
    """Wait until the plotted result is in view and Streamlit has *unfaded* it.

    Streamlit marks old page content stale during widget-triggered reruns. A
    visible heading/button is insufficient: these can already exist while the
    app and its plot remain semi-transparent. Inspect effective opacity up the
    DOM ancestry instead of inserting a fixed sleep or changing application CSS.
    """
    expect(target).to_be_visible(timeout=60000)
    target.evaluate("(node) => node.scrollIntoView({block: 'center', inline: 'nearest'})")
    handle = target.element_handle()
    if handle is None:
        raise RuntimeError("The plot disappeared before screenshot capture")
    page.wait_for_function(
        """node => {
            if (!node.isConnected) return false;
            for (let parent = node; parent; parent = parent.parentElement) {
                const style = window.getComputedStyle(parent);
                if (Number(style.opacity) < 0.99 ||
                    style.visibility !== 'visible' ||
                    parent.getAttribute('data-stale') === 'true') {
                    return false;
                }
            }
            const bounds = node.getBoundingClientRect();
            return bounds.width > 150 && bounds.height > 150 &&
                bounds.top >= 0 && bounds.bottom <= window.innerHeight &&
                bounds.left >= 0 && bounds.right <= window.innerWidth;
        }""",
        arg=handle,
        timeout=60000,
    )
    expect(page.locator(".stException")).to_have_count(0)


def capture(page: Page, path: Path, focus: Locator) -> None:
    """Save the current viewport with the actual *visible* result centred."""
    wait_for_clean_render(page, focus)
    # Streamlit scrolls its central pane independently; full_page=True took
    # screenshots of scrolled-away configuration controls, not the output.
    page.screenshot(path=str(path), animations="disabled", full_page=False)
    with Image.open(path) as screenshot:
        expected_size = (VIEWPORT_WIDTH * SCREENSHOT_SCALE, VIEWPORT_HEIGHT * SCREENSHOT_SCALE)
        if screenshot.size != expected_size:
            raise RuntimeError(
                f"Screenshot has unexpected size: {path}, "
                f"expected {expected_size}, got {screenshot.size}"
            )
    if path.stat().st_size < 16000:
        raise RuntimeError(f"Suspiciously empty screenshot: {path}")


def plot_button(page: Page, *, event: bool = False) -> Locator:
    label = r"^Make event plot" if event else r"^Make plot"
    return page.get_by_role("button", name=re.compile(label))


def choose_combobox(page: Page, section: Locator, name: str, value: str) -> None:
    """Select a Streamlit combobox option with one committed UI action.

    Sending several ArrowDown presses is unreliable for React Aria selectboxes:
    each press can trigger a Streamlit rerun and replace the input before the
    next press. Selecting an exact menu option avoids those intermediate
    committed values. Search is needed only if the option is virtualised.
    """
    widget = section.get_by_role("combobox", name=name, exact=True)
    if widget.input_value() == value:
        return

    for attempt in range(3):
        widget = section.get_by_role("combobox", name=name, exact=True)
        if widget.input_value() == value:
            return
        widget.click()
        option = page.get_by_role("option", name=value, exact=True)
        try:
            expect(option).to_be_visible(timeout=2500)
        except AssertionError:
            # React Aria's searchable selectbox may virtualise long lists or
            # lose its open popover during a previous Streamlit rerun.
            widget = section.get_by_role("combobox", name=name, exact=True)
            widget.fill(value)
            try:
                expect(option).to_be_visible(timeout=5000)
            except AssertionError:
                if attempt == 2:
                    raise RuntimeError(
                        f"Cannot find option {value!r} in selectbox {name!r}"
                    ) from None
                continue
        option.click()
        widget = section.get_by_role("combobox", name=name, exact=True)
        try:
            expect(widget).to_have_value(value, timeout=8000)
        except AssertionError:
            if attempt == 2:
                raise RuntimeError(
                    f"Selectbox {name!r} did not retain {value!r}; "
                    f"actual value is {widget.input_value()!r}"
                ) from None
            continue
        return

    raise RuntimeError(f"Failed to select {value!r} for {name!r}")


def verified_event_plot(page: Page, section: Locator) -> Locator:
    """Recover if a pending Streamlit rerun resets an event-axis selector.

    Immediately after choosing a branch, the client-side combobox can show
    the requested value while a previously queued rerun is still updating
    Streamlit's widget state. That rerun may restore an older choice before
    the Make event plot request is processed. Verify the values *after* the
    plot request, and never publish a plot with unexpected axes.
    """
    axes = {"x: branch": "Muon_eta", "y: branch": "Muon_pt"}
    image = section.locator('[data-testid="stImage"] img').first

    for attempt in range(3):
        for name, value in axes.items():
            choose_combobox(page, section, name, value)

        # The immediate check helps diagnose a missed popup interaction, but
        # does not by itself guarantee a pending Streamlit rerun has finished.
        for name, value in axes.items():
            expect(section.get_by_role("combobox", name=name, exact=True)).to_have_value(value)
        previous_src = image.get_attribute("src") if image.count() else None
        plot_button(page, event=True).click()
        expect(section.get_by_text("Output: event 10")).to_be_visible(timeout=30000)

        try:
            expect(image).to_be_visible(timeout=15000)
            if previous_src is not None:
                expect(image).not_to_have_attribute("src", previous_src, timeout=15000)
        except AssertionError:
            current = {
                name: section.get_by_role("combobox", name=name, exact=True).input_value()
                for name in axes
            }
            if current == axes:
                errors = section.get_by_role("alert").all_text_contents()
                raise RuntimeError(
                    f"Event plot failed with the expected axes {axes}: {errors}"
                ) from None
        else:
            current = {
                name: section.get_by_role("combobox", name=name, exact=True).input_value()
                for name in axes
            }
            if current == axes:
                expect(section.get_by_role("alert")).to_have_count(0)
                return image

        print(
            f"Event-axis selectors reverted during rerun (attempt {attempt + 1}/3): "
            f"expected {axes}, got {current}; reselecting.",
            file=sys.stderr,
        )

    raise RuntimeError(f"Event-axis selectors repeatedly reverted: expected {axes}")


def select_demo(page: Page, data: Path) -> Locator:
    expect(page.get_by_text("ROOT Explorer", exact=True).first).to_be_visible(timeout=30000)
    page.get_by_label("ROOT file path").fill(str(data))
    page.get_by_label("ROOT file path").press("Tab")
    expect(page.get_by_text("Entries/events:")).to_be_visible(timeout=30000)
    page.get_by_role("tab", name="Plotter").click()
    tab = page.get_by_role("tabpanel", name="Plotter")
    expect(plot_button(page)).to_be_visible(timeout=30000)
    return tab


def generate_captures(base_url: str, out: Path, data: Path, *, theme: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            device_scale_factor=SCREENSHOT_SCALE,
            color_scheme=theme,
            locale="en-GB",
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.set_default_timeout(30000)
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            plotter = select_demo(page, data)

            choose_combobox(page, plotter, "x: branch", "Muon_pt")
            plot_button(page).click()
            overview = plot_image(plotter)
            capture(page, out / screenshot_name("overview.png", theme), overview)
            expect(
                plotter.get_by_role("group", name="Bins", exact=True).get_by_role("slider")
            ).to_have_value("100")

            # A Gaussian muon pseudorapidity distribution makes a meaningful fit.
            choose_combobox(page, plotter, "x: branch", "Muon_eta")
            fit_box = plotter.get_by_role("checkbox", name="Enable fit")
            fit_box.focus()
            fit_box.press("Space")
            expect(fit_box).to_be_checked()
            previous_src = overview.get_attribute("src")
            plot_button(page).click()
            expect(plotter.get_by_role("heading", name="Histogram fit")).to_be_visible(
                timeout=60000
            )
            fitted = plot_image(plotter)
            expect(fitted).not_to_have_attribute("src", previous_src, timeout=60000)
            capture(page, out / screenshot_name("histogram-fit.png", theme), fitted)
            expect(
                plotter.get_by_role("group", name="Bins", exact=True).get_by_role("slider")
            ).to_have_value("100")

            # A different plot type must yield a genuinely new plot rather than
            # capturing the previously cached 1D histogram during a rerun.
            choose_combobox(page, plotter, "Plot type", "2D histogram")
            choose_combobox(page, plotter, "x: branch", "Muon_pt")
            choose_combobox(page, plotter, "y: branch", "Muon_eta")
            previous_src = fitted.get_attribute("src")
            plot_button(page).click()
            hist2d = plot_image(plotter)
            expect(hist2d).not_to_have_attribute("src", previous_src, timeout=60000)
            capture(page, out / screenshot_name("plot-2d.png", theme), hist2d)

            page.get_by_role("tab", name="Event inspection").click()
            event_tab = page.get_by_role("tabpanel", name="Event inspection")
            entry = event_tab.get_by_role("spinbutton", name="event index")
            entry.fill("10")  # Seeded demo: entry 10 has four generated muons.
            entry.press("Tab")
            expect(entry).to_have_value("10")
            choose_combobox(page, event_tab, "Plot type", "2D scatter")
            event_plot = verified_event_plot(page, event_tab)
            capture(page, out / screenshot_name("event-inspection.png", theme), event_plot)

            page.get_by_role("tab", name="Branches and objects").click()
            objects = page.get_by_role("tabpanel", name="Branches and objects")
            expect(objects.get_by_text("ROOT file contents")).to_be_visible(timeout=30000)
            objects.get_by_role("button", name=re.compile(r"^Open selected object")).click()
            expect(objects.get_by_role("button", name="Close inspection")).to_be_visible(
                timeout=60000
            )
            stored_hist = plot_image(objects)
            capture(page, out / screenshot_name("object-browser.png", theme), stored_hist)
        except Exception:
            # A full-page capture is useful for debugging selectors/timeouts but
            # is never published as one of the documentation illustrations.
            page.screenshot(path=str(out / f"failure-{theme}.png"), full_page=False)
            raise
        finally:
            browser.close()

    missing = [
        screenshot_name(name, theme)
        for name in NAMES
        if not (out / screenshot_name(name, theme)).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing screenshots: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/source/_static/screenshots")
    parser.add_argument("--base-url", default="http://127.0.0.1:8751")
    parser.add_argument("--port", type=int, default=8751)
    parser.add_argument(
        "--theme",
        choices=(*THEMES, "both"),
        default="both",
        help="Capture dark, light, or both themes (default: both)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="root-explorer-docs-") as temp:
        data = generate(Path(temp) / "synthetic.root")
        themes = THEMES if args.theme == "both" else (args.theme,)
        for theme in themes:
            # Each theme gets its own Streamlit process and Matplotlib config,
            # so no settings can leak between the light and dark captures.
            mplconfig = Path(temp) / f"matplotlib-{theme}"
            mplconfig.mkdir()
            (mplconfig / "matplotlibrc").write_text(matplotlib_rc(theme))
            app = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "streamlit_app.py"),
                    "--server.headless=true",
                    "--server.address=127.0.0.1",
                    f"--server.port={args.port}",
                    "--browser.gatherUsageStats=false",
                    f"--theme.base={theme}",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
                    "MPLCONFIGDIR": str(mplconfig),
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_server(args.base_url, app)
                generate_captures(args.base_url, args.output_dir, data, theme=theme)
            finally:
                app.terminate()
                try:
                    app.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    app.kill()


if __name__ == "__main__":
    main()
