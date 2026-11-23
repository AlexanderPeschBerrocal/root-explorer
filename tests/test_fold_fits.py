"""Fits use only entries inside the histogram range, whatever the flow mode (#206).

With Fold, under- and overflow are added to the edge bins for display. The fit model
describes only the visible range, so fitting those inflated bins gave sigma wrong by up
to 15 standard deviations, and nonsense parameters reported as "converged".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLD_NOTE = "the underflow and overflow folded into the edge bins are drawn but not fitted"


@pytest.fixture
def gaussian_file(tmp_path) -> Path:
    rng = np.random.default_rng(206)
    path = tmp_path / "toy.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree(
            "Events",
            {"x": rng.normal(1.0, 0.5, 50_000), "w": rng.uniform(0.5, 1.5, 50_000)},
        )
    return path


def _fit(path: Path, flow_mode: str, *, weighted: bool) -> tuple[AppTest, object]:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=30).run()
    at.text_input(key="root_path").input(str(path)).run()
    at.selectbox(key="x_branch").select("x").run()
    if weighted:
        at.checkbox(key="plot_use_weights").check().run()
        at.selectbox(key="plot_weight_branch").select("w").run()
    at.checkbox(key="plot_manual_histogram_limits").check().run()
    # Only one sigma either side of the peak: about a third of the entries are outside the range.
    at.text_input(key="plot_hist_x_min").input("0.5").run()
    at.text_input(key="plot_hist_x_max").input("1.5").run()
    at.selectbox(key="plot_hist_flow_mode").select(flow_mode).run()
    at.checkbox(key="plot_fit_enabled").check().run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    return at, at.session_state["plot_result"]["fit_result"]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_fold_fits_equal_ignore_fits(gaussian_file, weighted) -> None:
    _, ignore = _fit(gaussian_file, "ignore", weighted=weighted)
    fold_app, fold = _fit(gaussian_file, "fold", weighted=weighted)
    _, separate = _fit(gaussian_file, "separate", weighted=weighted)
    assert ignore.success and fold.success and separate.success
    assert fold.parameters == ignore.parameters
    assert fold.uncertainties == ignore.uncertainties
    assert separate.parameters == ignore.parameters
    # And the shared result recovers the truth (mu = 1, sigma = 0.5) within its uncertainty.
    for name, truth in (("mu", 1.0), ("sigma", 0.5)):
        assert abs(fold.parameters[name] - truth) < 4 * fold.uncertainties[name]
    assert any(FOLD_NOTE in caption.value for caption in fold_app.caption)


def test_note_is_only_shown_for_fold(gaussian_file) -> None:
    at, _ = _fit(gaussian_file, "ignore", weighted=False)
    assert not any(FOLD_NOTE in caption.value for caption in at.caption)
