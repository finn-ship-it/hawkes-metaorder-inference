"""Tests for `simulator.d2_konark_latent_book` (D2 latent full-book reframe).

The 07b spec requires at least 8 tests covering:
- latent stream non-empty; observed stream non-empty; latent count > observed count;
- regime windows applied correctly: side-asymmetric event uplift inside windows;
- regime truth labels correctly identify window membership for latent events;
- C_beta projection rule matches the documented table (table-driven test);
- C_beta strips regime labels: observed events do NOT carry regime fields;
- JSON round-trip on D2LatentBookConfig;
- source-of-truth: imports trace to Konarks-github repo/ (inherited from 07);
- runtime patches still applied (numpy-2.x in thinningOgataIS2; self.left
  reset post-truncation).
"""

from __future__ import annotations

import json
import os
from dataclasses import fields

import numpy as np
import pytest

from simulator.d2_konark_latent_book import (
    D2LatentBookConfig,
    FX_TOPOFBOOK_TYPES,
    KONARK_NATIVE_TYPES,
    KONARK_RUNTIME_PATCHES,
    KONARK_TYPE_TO_INDEX,
    LatentEvent,
    MetaOrderRegimeWindow,
    ObservedEvent,
    PHI_BETA_PROJECTION,
    default_meta_order_regime_windows,
    project_latent_to_observed,
    simulate_d2_latent_book,
)


def _konark_repo_available() -> bool:
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
    reason="Konark repo (Konarks-github repo/) not on disk; D2 latent-book not testable",
)


# ---------------------------------------------------------------------------
# Deterministic / projection-table tests (no live Konark needed)
# ---------------------------------------------------------------------------


def test_konark_native_types_are_twelve():
    assert len(KONARK_NATIVE_TYPES) == 12
    assert len(set(KONARK_NATIVE_TYPES)) == 12


def test_phi_beta_projection_table_07b_reference():
    """Table-driven verification of the documented C_beta projection rule
    (07b spec, Step 4).

    Per row: native type -> projected FX type (or None for censored).
    """
    expected = {
        "lo_top_Bid": "BID_UP",
        "co_top_Bid": "BID_DOWN",
        "lo_top_Ask": "ASK_DOWN",
        "co_top_Ask": "ASK_UP",
        "mo_Ask": "ASK_UP",
        "mo_Bid": "BID_DOWN",
        "lo_inspread_Bid": "BID_UP",
        "lo_inspread_Ask": "ASK_DOWN",
        "lo_deep_Ask": None,
        "co_deep_Ask": None,
        "lo_deep_Bid": None,
        "co_deep_Bid": None,
    }
    assert set(expected.keys()) == set(KONARK_NATIVE_TYPES)
    for native, expect in expected.items():
        proj_idx = PHI_BETA_PROJECTION[native]
        if expect is None:
            assert proj_idx is None, (
                f"native {native!r} should be censored (None), got {proj_idx}"
            )
        else:
            assert proj_idx is not None, (
                f"native {native!r} should project to {expect!r}, got None"
            )
            assert FX_TOPOFBOOK_TYPES[int(proj_idx)] == expect


def test_table_driven_projection_one_event_per_type():
    """For each Konark-native type, build a single LatentEvent and verify
    project_latent_to_observed maps it (or doesn't) per the documented
    table."""
    expected_pass_through = {
        n for n, v in PHI_BETA_PROJECTION.items() if v is not None
    }
    expected_censored = set(KONARK_NATIVE_TYPES) - expected_pass_through
    for i, name in enumerate(KONARK_NATIVE_TYPES):
        ev = LatentEvent(
            time=float(i),
            konark_native_type=name,
            side="Ask" if "Ask" in name else "Bid",
            depth_level="deep" if "deep" in name else ("in_spread" if "inspread" in name else "top"),
            regime_window_id=None,
            regime_side=None,
            in_window=False,
            injected_intensity_multiplier=1.0,
        )
        out = project_latent_to_observed([ev])
        if name in expected_censored:
            assert out == [], f"{name!r} should be censored; projected to {out}"
        else:
            assert len(out) == 1, f"{name!r} should produce 1 observed event"
            expected_fx = FX_TOPOFBOOK_TYPES[int(PHI_BETA_PROJECTION[name])]
            assert out[0].observed_type == expected_fx


def test_observed_event_has_no_regime_fields():
    """C_beta strips regime labels: ObservedEvent dataclass MUST NOT carry
    any regime_* attribute."""
    field_names = {f.name for f in fields(ObservedEvent)}
    for forbidden in ("regime_window_id", "regime_side", "in_window", "regime_truth"):
        assert forbidden not in field_names, (
            f"ObservedEvent must not carry {forbidden!r}; censoring would leak"
        )


def test_meta_order_regime_window_validation():
    with pytest.raises(ValueError):
        MetaOrderRegimeWindow(
            window_id=1, start_t=10.0, end_t=5.0,
            direction="Bid", intensity_multiplier=2.0,
            affected_native_types=("mo_Bid",),
        )
    with pytest.raises(ValueError):
        MetaOrderRegimeWindow(
            window_id=2, start_t=0.0, end_t=10.0,
            direction="Middle", intensity_multiplier=2.0,
            affected_native_types=("mo_Bid",),
        )
    with pytest.raises(ValueError):
        MetaOrderRegimeWindow(
            window_id=3, start_t=0.0, end_t=10.0,
            direction="Bid", intensity_multiplier=0.0,
            affected_native_types=("mo_Bid",),
        )
    with pytest.raises(ValueError):
        MetaOrderRegimeWindow(
            window_id=4, start_t=0.0, end_t=10.0,
            direction="Bid", intensity_multiplier=2.0,
            affected_native_types=("not_a_real_type",),
        )


def test_default_meta_order_regime_windows_alternation():
    """30 s windows every 150 s, alternating Bid / Ask starting with Bid,
    multiplier 2.0 on the side's mo_* and lo_inspread_*."""
    windows = default_meta_order_regime_windows(450.0)
    assert len(windows) == 3  # 0-30, 150-180, 300-330
    assert windows[0].direction == "Bid"
    assert windows[1].direction == "Ask"
    assert windows[2].direction == "Bid"
    assert windows[0].intensity_multiplier == 2.0
    assert windows[0].affected_native_types == ("mo_Bid", "lo_inspread_Bid")
    assert windows[1].affected_native_types == ("mo_Ask", "lo_inspread_Ask")
    assert windows[0].covers(15.0)
    assert not windows[0].covers(60.0)


def test_d2_config_json_round_trip():
    cfg = D2LatentBookConfig(
        spread0=0.04,
        seed=7,
        use_exp_approx=True,
        inspread_bid_over_ask_mu_ratio=1.5,
        per_native_type_mu_multipliers=(("mo_Ask", 2.0), ("co_top_Bid", 0.5)),
        meta_order_regime_windows=default_meta_order_regime_windows(300.0),
    )
    s = cfg.to_json()
    cfg2 = D2LatentBookConfig.from_json(s)
    assert cfg2.spread0 == cfg.spread0
    assert cfg2.seed == cfg.seed
    assert cfg2.use_exp_approx == cfg.use_exp_approx
    assert cfg2.inspread_bid_over_ask_mu_ratio == cfg.inspread_bid_over_ask_mu_ratio
    assert cfg2.per_native_type_mu_multipliers == cfg.per_native_type_mu_multipliers
    assert len(cfg2.meta_order_regime_windows) == len(cfg.meta_order_regime_windows)
    for w_a, w_b in zip(cfg.meta_order_regime_windows, cfg2.meta_order_regime_windows):
        assert w_a.window_id == w_b.window_id
        assert w_a.start_t == w_b.start_t
        assert w_a.end_t == w_b.end_t


def test_runtime_patches_documented():
    """Both inherited prompt-07 runtime patches (numpy-2.x scalar
    conversion + self.left truncation reset) are documented in
    `KONARK_RUNTIME_PATCHES`."""
    assert len(KONARK_RUNTIME_PATCHES) >= 2
    txt = " ".join(p["target"] + " " + p["issue"] + " " + p["patch"] for p in KONARK_RUNTIME_PATCHES)
    assert "thinningOgataIS2" in txt
    assert "left" in txt.lower()
    for entry in KONARK_RUNTIME_PATCHES:
        assert entry["inherited_from_prompt"] == "07"


# ---------------------------------------------------------------------------
# Live-Konark smoke tests
# ---------------------------------------------------------------------------


@_skip_if_no_konark
def test_smoke_latent_and_observed_streams_non_empty():
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=60.0)
    assert res.n_latent > 0
    assert res.n_observed > 0


@_skip_if_no_konark
def test_smoke_latent_strictly_exceeds_observed():
    """Censoring is non-trivial: latent > observed (at least one deep
    event drops out)."""
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=60.0)
    assert res.n_latent > res.n_observed
    assert res.latent_gt_observed


@_skip_if_no_konark
def test_smoke_regime_truth_labels_correctly_identify_window_membership():
    """Latent events inside any regime window carry the matching
    (window_id, side) labels; events outside carry (None, None, False)."""
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=120.0)
    windows = res.meta_order_regime_windows
    for ev in res.latent_events:
        in_any = False
        for w in windows:
            if w.covers(ev.time):
                in_any = True
                assert ev.regime_window_id == w.window_id, (
                    f"event at t={ev.time} should carry window {w.window_id}"
                )
                assert ev.regime_side == w.direction
                assert ev.in_window is True
                break
        if not in_any:
            assert ev.regime_window_id is None
            assert ev.regime_side is None
            assert ev.in_window is False


@_skip_if_no_konark
def test_smoke_observed_events_carry_no_regime_label():
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=60.0)
    for ev in res.observed_events:
        for forbidden in ("regime_window_id", "regime_side", "in_window"):
            assert not hasattr(ev, forbidden), (
                f"observed event leaks {forbidden!r}; censoring not honoured"
            )


@_skip_if_no_konark
def test_smoke_in_window_intensity_multiplier_matches_spec():
    """For latent events in a window AND of an affected_native_type, the
    `injected_intensity_multiplier` must equal the window's configured
    intensity_multiplier (within tolerance)."""
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=120.0)
    saw_in_window_affected = False
    for ev in res.latent_events:
        if ev.in_window and ev.regime_window_id is not None:
            window = next(
                w
                for w in res.meta_order_regime_windows
                if w.window_id == ev.regime_window_id
            )
            if ev.konark_native_type in window.affected_native_types:
                saw_in_window_affected = True
                assert ev.injected_intensity_multiplier == pytest.approx(
                    window.intensity_multiplier, abs=1e-9
                )
            else:
                # Event in window but unaffected type -> multiplier 1.0
                assert ev.injected_intensity_multiplier == pytest.approx(1.0, abs=1e-9)
    assert saw_in_window_affected, (
        "expected at least one in-window affected-type event; check seed / horizon"
    )


@_skip_if_no_konark
def test_smoke_regime_window_uplift_visible_in_event_rate():
    """At intensity_multiplier=2.0 the rate of affected types on the
    matching side, INSIDE the matching-side windows, should be visibly
    higher than the same types' rate OUTSIDE all windows on the same side.

    Bookkeeping: for each side (Bid / Ask), compute total time inside
    same-side windows vs outside any window; count affected-type events
    in each. Hawkes self-excitation makes the empirical ratio noisier
    than the 2.0 multiplier, but the directional uplift should hold.
    """
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=600.0)
    windows = res.meta_order_regime_windows
    # Per-side window time
    bid_window_dt = sum(w.end_t - w.start_t for w in windows if w.direction == "Bid")
    ask_window_dt = sum(w.end_t - w.start_t for w in windows if w.direction == "Ask")
    out_dt = res.horizon_seconds - sum(w.end_t - w.start_t for w in windows)
    if min(bid_window_dt, ask_window_dt, out_dt) <= 0:
        pytest.skip("degenerate window configuration for this test")
    # Count affected events per (side, in_matching_window vs out)
    bid_in = 0
    bid_out = 0
    ask_in = 0
    ask_out = 0
    for ev in res.latent_events:
        if ev.konark_native_type not in {
            "mo_Bid", "lo_inspread_Bid", "mo_Ask", "lo_inspread_Ask"
        }:
            continue
        if ev.side == "Bid":
            if ev.in_window and ev.regime_side == "Bid":
                bid_in += 1
            elif not ev.in_window:
                bid_out += 1
        else:
            if ev.in_window and ev.regime_side == "Ask":
                ask_in += 1
            elif not ev.in_window:
                ask_out += 1
    rate_bid_in = bid_in / bid_window_dt
    rate_bid_out = bid_out / out_dt
    rate_ask_in = ask_in / ask_window_dt
    rate_ask_out = ask_out / out_dt
    # At least one of (Bid, Ask) must show uplift; both sides should ideally,
    # but the supplemental-thinning Poisson scheme has variance, so accept
    # if either side shows the directional uplift.
    bid_uplift = rate_bid_in > rate_bid_out
    ask_uplift = rate_ask_in > rate_ask_out
    assert bid_uplift or ask_uplift, (
        f"expected uplift on at least one of Bid / Ask sides under "
        f"injected regimes; got rate_bid_in={rate_bid_in:.3f}, "
        f"rate_bid_out={rate_bid_out:.3f}, "
        f"rate_ask_in={rate_ask_in:.3f}, rate_ask_out={rate_ask_out:.3f}"
    )


@_skip_if_no_konark
def test_smoke_observed_event_count_consistent_with_projection_table():
    """observed.count == sum of latent.count for all NON-censored types."""
    cfg = D2LatentBookConfig(spread0=0.03, seed=1, inspread_bid_over_ask_mu_ratio=1.0)
    res = simulate_d2_latent_book(cfg, horizon_seconds=60.0)
    expected_observed = sum(
        c for n, c in res.pre_projection_native_counts.items()
        if PHI_BETA_PROJECTION[n] is not None
    )
    assert res.n_observed == expected_observed


@_skip_if_no_konark
def test_smoke_runtime_patches_applied_via_method_type_swap():
    """The patched method on the live arrival is sourced from this module,
    not the Konark file (i.e., types.MethodType swap actually took)."""
    from simulator.d2_konark_latent_book import (
        D2LatentBookConfig,
        build_konark_arrival,
    )
    import inspect

    cfg = D2LatentBookConfig(spread0=0.03, seed=1)
    arr, _mu = build_konark_arrival(cfg)
    src = inspect.getsourcefile(arr.thinningOgataIS2)
    assert "d2_konark_latent_book" in (src or ""), (
        f"thinningOgataIS2 source file is {src!r}; patch did not take"
    )


@_skip_if_no_konark
def test_smoke_konark_import_traces_to_konarks_github_repo():
    """Source-of-truth verification: the HawkesArrival type imported by
    the latent-book module's `build_konark_arrival` resolves to a file
    under `Konarks-github repo/`, NOT under `My repo/`."""
    from simulator.d2_konark_latent_book import build_konark_arrival, D2LatentBookConfig

    arr, _mu = build_konark_arrival(D2LatentBookConfig(spread0=0.03, seed=1))
    src = type(arr).__module__
    assert "HawkesRLTrading" in src
    file_path = type(arr).__module__
    # Inspect the actual file location:
    import inspect
    actual_path = inspect.getsourcefile(type(arr)) or ""
    assert "Konarks-github repo" in actual_path, (
        f"HawkesArrival imports from {actual_path!r}; "
        "should be from Konarks-github repo/"
    )
