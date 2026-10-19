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
        if screenshot.width < 1200 or screenshot.height < 700:
            raise RuntimeError(f"Screenshot has unexpected size: {path}")
    if path.stat().st_size < 16000:
        raise RuntimeError(f"Suspiciously empty screenshot: {path}")


def plot_button(page: Page, *, event: bool = False) -> Locator:
    label = r"^Make event plot" if event else r"^Make plot"
    return page.get_by_role("button", name=re.compile(label))


def choose_combobox(page: Page, section: Locator, name: str, value: str) -> None:
    """Select a named Streamlit option rather than relying on default order."""
    widget = section.get_by_role("combobox", name=name, exact=True)
    current = widget.input_value()
    if current == value:
        return

    # Streamlit's Plot type selectbox does not reliably expose its dropdown
    # options to Playwright after a rerun, even when its adjacent Open button
    # is clicked. Its accessible keyboard interface does work; use the stable
    # source-defined ordering instead of trying to find an absent option.
    if name == "Plot type":
        types = ("1D histogram", "2D histogram", "2D scatter", "3D scatter interactive")
        if current not in types or value not in types:
            raise ValueError(f"Unexpected plot type transition: {current!r} -> {value!r}")
        delta = types.index(value) - types.index(current)
        widget.focus()
        for _ in range(abs(delta)):
            widget.press("ArrowDown" if delta > 0 else "ArrowUp")
        widget.press("Enter")
        expect(widget).to_have_value(value, timeout=20000)
        return

    # For branch selectors, the popup options are accessible; this route
    # selects a named branch independent of input ordering.
    widget.click()
    option = page.get_by_role("option", name=value, exact=True)
    expect(option).to_be_visible(timeout=20000)
    option.click()
    expect(widget).to_have_value(value, timeout=20000)


def select_demo(page: Page, data: Path) -> Locator:
    expect(page.get_by_text("ROOT Explorer", exact=True).first).to_be_visible(timeout=30000)
    page.get_by_label("ROOT file path").fill(str(data))
    page.get_by_label("ROOT file path").press("Tab")
    expect(page.get_by_text("Entries/events:")).to_be_visible(timeout=30000)
    page.get_by_role("tab", name="Plotter").click()
    tab = page.get_by_role("tabpanel", name="Plotter")
    expect(plot_button(page)).to_be_visible(timeout=30000)
    return tab


def generate_captures(base_url: str, out: Path, data: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            color_scheme="dark",
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
            capture(page, out / "overview.png", overview)

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
            capture(page, out / "histogram-fit.png", fitted)

            # A different plot type must yield a genuinely new plot rather than
            # capturing the previously cached 1D histogram during a rerun.
            choose_combobox(page, plotter, "Plot type", "2D histogram")
            choose_combobox(page, plotter, "x: branch", "Muon_pt")
            choose_combobox(page, plotter, "y: branch", "Muon_eta")
            previous_src = fitted.get_attribute("src")
            plot_button(page).click()
            hist2d = plot_image(plotter)
            expect(hist2d).not_to_have_attribute("src", previous_src, timeout=60000)
            capture(page, out / "plot-2d.png", hist2d)

            page.get_by_role("tab", name="Event inspection").click()
            event_tab = page.get_by_role("tabpanel", name="Event inspection")
            entry = event_tab.get_by_role("spinbutton", name="event index")
            entry.fill("10")  # Seeded demo: entry 10 has four generated muons.
            entry.press("Tab")
            expect(entry).to_have_value("10")
            choose_combobox(page, event_tab, "Plot type", "2D scatter")
            choose_combobox(page, event_tab, "event_x: branch", "Muon_eta")
            choose_combobox(page, event_tab, "event_y: branch", "Muon_pt")
            plot_button(page, event=True).click()
            expect(event_tab.get_by_text("Output: event 10")).to_be_visible(timeout=60000)
            event_plot = plot_image(event_tab)
            capture(page, out / "event-inspection.png", event_plot)

            page.get_by_role("tab", name="Branches and objects").click()
            objects = page.get_by_role("tabpanel", name="Branches and objects")
            expect(objects.get_by_text("ROOT file contents")).to_be_visible(timeout=30000)
            objects.get_by_role("button", name=re.compile(r"^Open selected object")).click()
            expect(objects.get_by_role("button", name="Close inspection")).to_be_visible(
                timeout=60000
            )
            stored_hist = plot_image(objects)
            capture(page, out / "object-browser.png", stored_hist)
        except Exception:
            # A full-page capture is useful for debugging selectors/timeouts but
            # is never published as one of the documentation illustrations.
            page.screenshot(path=str(out / "failure.png"), full_page=False)
            raise
        finally:
            browser.close()

    missing = [name for name in NAMES if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing screenshots: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs/source/_static/screenshots"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8751")
    parser.add_argument("--port", type=int, default=8751)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="root-explorer-docs-") as temp:
        data = generate(Path(temp) / "synthetic.root")
        # Screenshot-only Matplotlib configuration: keep the plot surface dark
        # without changing ROOT Explorer's defaults for normal users.
        mplconfig = Path(temp) / "matplotlib"
        mplconfig.mkdir()
        (mplconfig / "matplotlibrc").write_text(
            "figure.facecolor: #0e1117\n"
            "axes.facecolor: #0e1117\n"
            "savefig.facecolor: #0e1117\n"
            "axes.edgecolor: #fafafa\n"
            "axes.labelcolor: #fafafa\n"
            "text.color: #fafafa\n"
            "xtick.color: #fafafa\n"
            "ytick.color: #fafafa\n"
            "grid.color: #606975\n"
        )
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
                "--theme.base=dark",
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
            generate_captures(args.base_url, args.output_dir, data)
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait()


if __name__ == "__main__":
    main()
