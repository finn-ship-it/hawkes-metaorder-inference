"""Tests for the prompt-07c regime-detectability sweep runner.

Per the 07c spec, at least 4 tests covering:
- cell construction is correct (30 cells, 5 x 3 x 2);
- the 07b calibration is loaded byte-for-byte unchanged from
  `d2_latent_book_calibration_summary.json::winning_candidate`;
- HMM configuration is the dissertation default (no D2-specific
  re-tuning);
- threshold-finding logic correctly identifies the lowest-intensity
  cell with `F1_latent >= 0.7` (or correctly returns None when none
  exists).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SRC_PATH = str(HERE.parent.parent)


def _ensure_src_on_path():
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


def _load_runner():
    _ensure_src_on_path()
    runner_dir = Path(__file__).resolve().parent.parent.parent.parent / "dissertation_artifacts"
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    import importlib
    if "build_d2_regime_detectability" in sys.modules:
        return importlib.reload(sys.modules["build_d2_regime_detectability"])
    return importlib.import_module("build_d2_regime_detectability")


def _konark_repo_available() -> bool:
    env = os.environ.get("KONARK_REPO_ROOT")
    if env and os.path.isdir(os.path.join(env, "HawkesRLTrading")):
        return True
    here = os.path.dirname(os.path.abspath(__file__))
    work_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(here)))
    )
    return os.path.isdir(
        os.path.join(work_root, "Konarks-github repo", "HawkesRLTrading")
    )


_KONARK_AVAILABLE = _konark_repo_available()
_skip_no_konark = pytest.mark.skipif(
    not _KONARK_AVAILABLE,
    reason="Konark repo (Konarks-github repo/) not on disk",
)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------


def test_sweep_grid_has_thirty_cells():
    runner = _load_runner()
    cells = runner.build_sweep_grid()
    assert len(cells) == 30, f"expected 30 cells (5*3*2), got {len(cells)}"


def test_sweep_grid_dimensions_unique():
    """Each cell has a unique (intensity, duty, channels) triple, and
    the grid spans 5 intensities * 3 duties * 2 channel-sets."""
    runner = _load_runner()
    cells = runner.build_sweep_grid()
    triples = {(c.intensity_multiplier, c.duty_cycle, c.channel_set) for c in cells}
    assert len(triples) == 30
    intensities = {c.intensity_multiplier for c in cells}
    duties = {c.duty_cycle for c in cells}
    channels = {c.channel_set for c in cells}
    assert intensities == set(runner.INTENSITY_MULTIPLIERS)
    assert duties == set(runner.DUTY_CYCLES)
    assert channels == set(runner.CHANNEL_SETS)


def test_sweep_grid_period_seconds_matches_duty_cycle():
    runner = _load_runner()
    cells = runner.build_sweep_grid()
    for c in cells:
        assert c.period_seconds == pytest.approx(
            runner.WINDOW_WIDTH_SECONDS / c.duty_cycle, abs=1e-9
        ), (
            f"cell {c.cell_id}: period_seconds={c.period_seconds} should equal "
            f"window_width / duty_cycle = {runner.WINDOW_WIDTH_SECONDS / c.duty_cycle}"
        )


def test_default_channel_set_is_two_types_broader_is_four():
    runner = _load_runner()
    cells = runner.build_sweep_grid()
    for c in cells:
        if c.channel_set == "DEFAULT":
            assert len(c.affected_channels_template) == 2
        else:
            assert c.channel_set == "BROADER"
            assert len(c.affected_channels_template) == 4


# ---------------------------------------------------------------------------
# 07b frozen-input loading
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parent.parent.parent.parent
        / "dissertation_artifacts"
        / "data"
        / "d2_latent_book_calibration_summary.json"
    ).exists(),
    reason="07b calibration artefact missing",
)
def test_frozen_inputs_match_07b_artefact_byte_for_byte():
    runner = _load_runner()
    frozen = runner.load_frozen_inputs()
    # Re-load artefact and compare exactly
    raw = json.loads(runner.INPUT_07B_JSON.read_text())
    wc = raw["winning_candidate"]["parameters"]
    assert frozen.spread0 == float(wc["spread0"])
    assert frozen.seed == int(wc["seed"])
    assert frozen.use_exp_approx == bool(wc["use_exp_approx"])
    assert frozen.inspread_bid_over_ask_mu_ratio == float(
        wc["inspread_bid_over_ask_mu_ratio"]
    )
    assert frozen.per_native_type_mu_multipliers == tuple(
        (str(name), float(mult))
        for name, mult in wc["per_native_type_mu_multipliers"]
    )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parent.parent.parent.parent
        / "dissertation_artifacts"
        / "data"
        / "d2_latent_book_calibration_summary.json"
    ).exists(),
    reason="07b calibration artefact missing",
)
def test_frozen_inputs_hmm_default_unchanged():
    runner = _load_runner()
    frozen = runner.load_frozen_inputs()
    assert frozen.hmm_default_recovery_method == "hmm"
    assert frozen.hmm_n_states == 2
    assert frozen.hmm_emission_features == (
        "recent_event_rate",
        "directional_imbalance",
    )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parent.parent.parent.parent
        / "dissertation_artifacts"
        / "data"
        / "d2_latent_book_calibration_summary.json"
    ).exists(),
    reason="07b calibration artefact missing",
)
def test_frozen_inputs_runtime_patches_present():
    runner = _load_runner()
    frozen = runner.load_frozen_inputs()
    targets_blob = " ".join(frozen.runtime_patch_targets).lower()
    assert "thinningogataIs2".lower() in targets_blob.lower() or "thinningogata" in targets_blob
    assert "left" in targets_blob or "truncation" in targets_blob


# ---------------------------------------------------------------------------
# Threshold-finding logic
# ---------------------------------------------------------------------------


def _mock_cell(cell_id, intensity, duty, ch, f1_lat, f1_obs=None):
    return {
        "cell_id": int(cell_id),
        "intensity_multiplier": float(intensity),
        "duty_cycle": float(duty),
        "channel_set": str(ch),
        "period_seconds": 30.0 / float(duty),
        "affected_channels_template": ("mo_<side>", "lo_inspread_<side>"),
        "f1_latent": (None if f1_lat is None else float(f1_lat)),
        "f1_observed": (None if f1_obs is None else float(f1_obs)),
        "censoring_cost_cell": (
            None
            if (f1_lat is None or f1_obs is None)
            else float(f1_obs - f1_lat)
        ),
    }


def test_threshold_finding_picks_lowest_intensity_cell_passing():
    runner = _load_runner()
    # Synthesise: cell A (intensity 10) passes with F1_latent=0.85;
    # cell B (intensity 5) passes with F1_latent=0.75; cell C
    # (intensity 3) fails with F1_latent=0.5. Threshold should be B.
    cells = [
        _mock_cell(1, 3.0, 0.20, "DEFAULT", 0.50, 0.40),
        _mock_cell(2, 5.0, 0.20, "DEFAULT", 0.75, 0.60),
        _mock_cell(3, 10.0, 0.20, "DEFAULT", 0.85, 0.70),
    ]
    threshold = runner.find_threshold_cell(cells)
    assert threshold is not None
    assert threshold["cell_id"] == 2
    assert threshold["f1_latent"] == pytest.approx(0.75)


def test_threshold_finding_returns_none_when_no_cell_qualifies():
    runner = _load_runner()
    cells = [
        _mock_cell(1, 3.0, 0.20, "DEFAULT", 0.50, 0.40),
        _mock_cell(2, 5.0, 0.20, "DEFAULT", 0.60, 0.55),
        _mock_cell(3, 10.0, 0.20, "DEFAULT", 0.69, 0.65),
    ]
    threshold = runner.find_threshold_cell(cells)
    assert threshold is None


def test_threshold_tie_breaks_by_duty_then_channels():
    runner = _load_runner()
    # Two cells at intensity 5: one with duty 0.20 DEFAULT, one with
    # duty 0.50 BROADER. Threshold should be (5, 0.20, DEFAULT) — lower
    # duty AND smaller channel set.
    cells = [
        _mock_cell(1, 5.0, 0.50, "BROADER", 0.80, 0.70),
        _mock_cell(2, 5.0, 0.20, "DEFAULT", 0.78, 0.68),
    ]
    threshold = runner.find_threshold_cell(cells)
    assert threshold is not None
    assert threshold["cell_id"] == 2


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def test_classification_complete_when_threshold_exists():
    runner = _load_runner()
    cells = [_mock_cell(1, 5.0, 0.20, "DEFAULT", 0.85, 0.75)]
    threshold = runner.find_threshold_cell(cells)
    cls, why = runner.classify(cells, threshold)
    assert cls == "07C_COMPLETE"


def test_classification_partial_when_no_threshold_but_variation_exists():
    runner = _load_runner()
    cells = [
        _mock_cell(1, 2.0, 0.20, "DEFAULT", 0.30, 0.25),
        _mock_cell(2, 10.0, 0.50, "BROADER", 0.55, 0.45),
    ]
    threshold = runner.find_threshold_cell(cells)
    cls, why = runner.classify(cells, threshold)
    assert cls == "07C_PARTIAL"


def test_classification_config_incomplete_when_no_variation():
    runner = _load_runner()
    cells = [
        _mock_cell(1, 2.0, 0.20, "DEFAULT", 0.30, 0.25),
        _mock_cell(2, 10.0, 0.50, "BROADER", 0.31, 0.28),
    ]
    threshold = runner.find_threshold_cell(cells)
    cls, why = runner.classify(cells, threshold)
    assert cls == "07C_CONFIGURATION_INCOMPLETE"


# ---------------------------------------------------------------------------
# Live cell construction (touches Konark; smoke-tier)
# ---------------------------------------------------------------------------


@_skip_no_konark
def test_cell_meta_order_windows_alternates_sides_and_uses_period():
    runner = _load_runner()
    cells = runner.build_sweep_grid()
    # Pick the duty=0.20 DEFAULT cell at intensity 2.0
    cell = next(c for c in cells
                if c.duty_cycle == 0.20 and c.channel_set == "DEFAULT" and c.intensity_multiplier == 2.0)
    windows = runner.cell_meta_order_windows(cell, runner.T_SIM_SECONDS)
    # Period 150s, T=1800 -> first start at 0, last <= 1770; expect 12 windows
    assert len(windows) >= 10
    assert windows[0].direction == "Bid"
    assert windows[1].direction == "Ask"
    assert windows[0].intensity_multiplier == 2.0
    assert windows[0].affected_native_types == ("mo_Bid", "lo_inspread_Bid")
    assert windows[1].affected_native_types == ("mo_Ask", "lo_inspread_Ask")
