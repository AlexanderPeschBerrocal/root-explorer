"""Capture actual Streamlit UI examples in deterministic Chromium sessions.

Run after installing `.[screenshots]` and `python -m playwright install chromium`.
This tool is intentionally independent of Sphinx/Read the Docs.
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
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "overview.png",
    "histogram-fit.png",
    "plot-2d.png",
    "object-browser.png",
    "event-inspection.png",
)


def wait_for_server(base_url: str, process: subprocess.Popen[str] | None) -> None:
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("Streamlit exited before becoming healthy")
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Streamlit did not become healthy: {base_url}")


def settle(page: Page) -> None:
    """Ensure Streamlit has completed the rerun after each browser action."""
    expect(page.get_by_text("ROOT Explorer", exact=True).first).to_be_visible(timeout=30000)
    expect(page.get_by_text("Error running app")).to_have_count(0)


def select_demo(page: Page, data: Path) -> None:
    page.goto(page.url, wait_until="domcontentloaded")
    settle(page)
    page.get_by_label("ROOT file path").fill(str(data))
    page.get_by_label("ROOT file path").press("Tab")
    expect(page.get_by_text("Entries/events:")).to_be_visible(timeout=30000)
    expect(page.get_by_role("tab", name="Plotter")).to_be_visible()
    page.get_by_role("tab", name="Plotter").click()
    expect(page.get_by_role("button", name=re.compile(r"^Make plot"))).to_be_visible()


def image(page: Page, path: Path) -> None:
    # Fixed device scale and viewport are configured in the browser context.
    expect(page.locator(".stException")).to_have_count(0)
    page.screenshot(path=str(path), animations="disabled", full_page=True)
    if path.stat().st_size < 12000:
        raise RuntimeError(f"Suspiciously small screenshot: {path}")


def make_plot(page: Page) -> None:
    page.get_by_role("button", name=re.compile(r"^Make plot")).click()
    expect(page.get_by_role("button", name="Download plot code")).to_be_visible(timeout=60000)


def run(base_url: str, out: Path, data: Path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            color_scheme="light",
            locale="en-GB",
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.set_default_timeout(20000)
        page.goto(base_url, wait_until="domcontentloaded")
        try:
            select_demo(page, data)
            make_plot(page)
            image(page, out / "overview.png")

            fit_box = page.get_by_role("tabpanel", name="Plotter").get_by_role("checkbox", name="Enable fit")
            fit_box.focus()
            fit_box.press("Space")
            expect(fit_box).to_be_checked()
            page.get_by_role("button", name=re.compile(r"^Make plot")).click()
            expect(page.get_by_role("tabpanel", name="Plotter").get_by_role("heading", name="Histogram fit")).to_be_visible(timeout=60000)
            image(page, out / "histogram-fit.png")

            plot_combo = page.get_by_role("tabpanel", name="Plotter").get_by_role("combobox", name="Plot type")
            plot_combo.focus()
            plot_combo.press("ArrowDown")
            plot_combo.press("Enter")
            expect(plot_combo).to_have_value("2D histogram")
            make_plot(page)
            image(page, out / "plot-2d.png")

            page.get_by_role("tab", name="Event inspection").click()
            page.get_by_role("button", name="Make event plot").click()
            expect(page.get_by_text("Output: event", exact=False)).to_be_visible(timeout=60000)
            image(page, out / "event-inspection.png")

            # A separate capture avoids accidental hidden-tab or stale-plot state.
            page.get_by_role("tab", name="Branches and objects").click()
            expect(page.get_by_text("ROOT file contents")).to_be_visible()
            image(page, out / "object-browser.png")
        except PlaywrightTimeoutError as exc:
            page.screenshot(path=str(out / "failure.png"), full_page=True)
            raise RuntimeError("Screenshot browser scenario failed; see failure.png") from exc
        finally:
            browser.close()
    missing = [name for name in NAMES if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing screenshots: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/source/_static/screenshots",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8751")
    parser.add_argument("--port", type=int, default=8751)
    args = parser.parse_args()

    # The local server is trusted and deliberately accepts a synthetic Path.
    with tempfile.TemporaryDirectory(prefix="root-explorer-docs-") as temp:
        data = generate(Path(temp) / "synthetic.root")
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
            ],
            cwd=ROOT,
            env={**os.environ, "STREAMLIT_SERVER_RUN_ON_SAVE": "false"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(args.base_url, app)
            run(args.base_url, args.output_dir, data)
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait()


if __name__ == "__main__":
    main()
