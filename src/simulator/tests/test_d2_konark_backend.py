"""Tests for the D2 Konark backend adapter (`simulator.d2_konark_backend`).

The three smoke tests required by `restored_dissertation_pack/07_konark_sim_d2_integration.md`:

1. **smoke run** — 60 s of D2 simulation produces non-empty events.
2. **latent-vs-observed counts** — number of Konark-native LOB events
   strictly exceeds the number of FX-projected top-of-book events
   (the censoring is non-trivial).
3. **regime-channel consistency** — D2 with the inspread channel active
   (initial spread > 1 tick) produces strictly more FX-projected events
   than the default (spread = 1 tick) configuration that censors
   inspread events.

Plus deterministic-projection unit tests that exercise the Phi_beta map
without touching Konark's stochastic core.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from simulator.d2_konark_backend_legacy import (
    D2KonarkConfig,
    FX_TOPOFBOOK_TYPES,
    KONARK_NATIVE_TYPES,
    KONARK_RUNTIME_PATCHES,
    KONARK_TYPE_TO_INDEX,
    PHI_BETA_PROJECTION,
    project_konark_to_fx_topofbook,
    simulate_d2_konark,
)


def _konark_repo_available() -> bool:
    """Skip-marker for environments without the Konark repo on disk."""
    env = os.environ.get("KONARK_REPO_ROOT")
    if env and os.path.isdir(os.path.join(env, "HawkesRLTrading")):
        return True
    here = os.path.dirname(os.path.abspath(__file__))
    work_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(here)))
    )
    candidate = os.path.join(work_root, "Konarks-github repo")
    return os.path.isdir(os.path.join(candidate, "HawkesRLTrading"))


_KONARK_AVAILABLE = _konark_repo_available()
_skip_if_no_konark = pytest.mark.skipif(
    not _KONARK_AVAILABLE,
    reason="Konark repo (Konarks-github repo/) not on disk; D2 backend not testable",
)


# ---------------------------------------------------------------------------
# Deterministic-projection unit tests (no Konark dependency)
# ---------------------------------------------------------------------------


def test_konark_native_types_are_twelve():
    assert len(KONARK_NATIVE_TYPES) == 12
    assert len(set(KONARK_NATIVE_TYPES)) == 12


def test_konark_type_to_index_round_trip():
    for i, name in enumerate(KONARK_NATIVE_TYPES):
        assert KONARK_TYPE_TO_INDEX[name] == i


def test_fx_topofbook_types_match_simulator_enum():
    """The FX 4-type names match the FX-native simulator EventType enum order."""
    assert FX_TOPOFBOOK_TYPES == ("BID_UP", "BID_DOWN", "ASK_UP", "ASK_DOWN")


def test_phi_beta_projection_covers_all_konark_types():
    """Every Konark-native type has a Phi_beta entry (None for censored)."""
    for name in KONARK_NATIVE_TYPES:
        assert name in PHI_BETA_PROJECTION


def test_phi_beta_projection_active_types_are_inspread_and_market_orders():
    active = [n for n, v in PHI_BETA_PROJECTION.items() if v is not None]
    assert set(active) == {
        "mo_Bid",
        "mo_Ask",
        "lo_inspread_Bid",
        "lo_inspread_Ask",
    }


def test_phi_beta_projection_assigns_correct_fx_codes():
    """mo_Bid -> BID_DOWN, mo_Ask -> ASK_UP, lo_inspread_Bid -> BID_UP,
    lo_inspread_Ask -> ASK_DOWN. Each maps onto the FX-native simulator
    enum index."""
    assert PHI_BETA_PROJECTION["mo_Bid"] == 1  # BID_DOWN
    assert PHI_BETA_PROJECTION["mo_Ask"] == 2  # ASK_UP
    assert PHI_BETA_PROJECTION["lo_inspread_Bid"] == 0  # BID_UP
    assert PHI_BETA_PROJECTION["lo_inspread_Ask"] == 3  # ASK_DOWN


def test_project_konark_to_fx_topofbook_drops_censored_types():
    """A small hand-constructed sequence with one of each Konark-native
    type produces exactly four FX-projected events in the canonical
    order."""
    times = np.arange(12, dtype=np.float64)
    types = np.arange(12, dtype=np.int64)
    fx_t, fx_k = project_konark_to_fx_topofbook(times, types)
    assert fx_t.shape == fx_k.shape
    assert fx_t.shape == (4,)
    # Events at indices 4 (mo_Ask), 5 (lo_inspread_Ask), 6 (lo_inspread_Bid),
    # 7 (mo_Bid) are the only ones that pass the projection.
    assert sorted(fx_t.tolist()) == [4.0, 5.0, 6.0, 7.0]
    expected_fx = sorted(
        [
            PHI_BETA_PROJECTION["mo_Ask"],
            PHI_BETA_PROJECTION["lo_inspread_Ask"],
            PHI_BETA_PROJECTION["lo_inspread_Bid"],
            PHI_BETA_PROJECTION["mo_Bid"],
        ]
    )
    assert sorted(fx_k.tolist()) == expected_fx


def test_project_konark_to_fx_topofbook_validates_shapes():
    with pytest.raises(ValueError):
        project_konark_to_fx_topofbook(
            np.zeros(3, dtype=np.float64), np.zeros(2, dtype=np.int64)
        )


def test_runtime_patches_documented():
    """Both known Konark bugs (numpy-2.x scalar conversion + self.left
    truncation reset) are documented in `KONARK_RUNTIME_PATCHES`."""
    assert len(KONARK_RUNTIME_PATCHES) >= 2
    targets = " ".join(p["target"] for p in KONARK_RUNTIME_PATCHES)
    assert "thinningOgataIS2" in targets
    assert "truncation" in targets.lower() or "left" in targets.lower()
    for entry in KONARK_RUNTIME_PATCHES:
        assert "target" in entry
        assert "issue" in entry
        assert "patch" in entry


# ---------------------------------------------------------------------------
# Smoke tests against the live Konark backend
# ---------------------------------------------------------------------------


@_skip_if_no_konark
def test_smoke_60s_d2_produces_non_empty_events():
    """Spec smoke test #1: 60 s of D2 simulation produces non-empty events."""
    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=60.0)
    assert res.n_konark_latent > 0
    assert res.n_fx_observed >= 0  # may be zero under degenerate spread; not under spread0=0.03


@_skip_if_no_konark
def test_smoke_latent_strictly_exceeds_observed():
    """Spec smoke test #2: latent count > observed count (the censoring
    is non-trivial)."""
    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=60.0)
    assert res.n_konark_latent > res.n_fx_observed
    assert res.latent_gt_observed is True


@_skip_if_no_konark
def test_smoke_regime_channel_consistency():
    """Spec smoke test #3: D2 with the inspread channels active
    (spread0=0.03) produces strictly more FX-projected events than the
    censored configuration (spread0=0.01, where the in-spread baselines
    are gated off by Konark's `if 100*round(spread,2) < 2` check)."""
    cfg_off = D2KonarkConfig(spread0=0.01, seed=1, use_exp_approx=True)
    cfg_on = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res_off = simulate_d2_konark(cfg_off, horizon_seconds=60.0)
    res_on = simulate_d2_konark(cfg_on, horizon_seconds=60.0)
    assert res_on.n_fx_observed > res_off.n_fx_observed


@_skip_if_no_konark
def test_smoke_fx_event_types_in_valid_range():
    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=60.0)
    if res.n_fx_observed > 0:
        assert int(res.fx_topofbook_types.min()) >= 0
        assert int(res.fx_topofbook_types.max()) <= 3


@_skip_if_no_konark
def test_smoke_fx_times_are_subset_of_konark_times():
    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=60.0)
    # FX times are a subset of Konark times.
    if res.n_fx_observed > 0:
        konark_set = set(res.konark_native_times.tolist())
        for t in res.fx_topofbook_times.tolist():
            assert t in konark_set
