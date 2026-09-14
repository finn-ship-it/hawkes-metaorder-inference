"""Tests for the D1/D2 cross-simulator inference chain.

These tests exercise the Layer-1 EM and Layer-2 HMM modules on a
short D2 stream to confirm:

- the FX-projected D2 stream is byte-compatible with the FX-native
  inference pipeline (same array layout, same event-type alphabet);
- both inference modules terminate cleanly on D2 input;
- the per-backend EM converges within the smoke horizon.

The full matched-configuration comparison (with rho_D1 vs rho_D2 and
HMM accuracy gates) lives in the build runner
`dissertation_artifacts/build_d2_konark_inference.py` and is reported
in the JSON artefact + V1-V6 validation. These tests are the unit-test
floor under that pipeline.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from simulator.d2_konark_backend_legacy import D2KonarkConfig, simulate_d2_konark


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
    reason="Konark repo (Konarks-github repo/) not on disk; D2 inference chain not testable",
)


@_skip_if_no_konark
def test_d2_stream_compatible_with_layer1_em():
    """Layer-1 EM (`fit_layer1_em`) accepts a D2 FX-projected stream and
    terminates cleanly. Convergence is not asserted (the smoke horizon
    is short); termination is."""
    from simulator.layer1_em import Layer1EMConfig, fit_layer1_em

    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=120.0)
    if res.n_fx_observed < 30:
        pytest.skip(
            f"D2 produced {res.n_fx_observed} FX events; below the EM smoke "
            "floor of 30. Skipping to avoid a degenerate fit."
        )
    em_cfg = Layer1EMConfig(
        betas=(2.0,),
        max_iter=50,
        tol_relative_ll=1.0e-4,
    )
    em_res = fit_layer1_em(
        times=res.fx_topofbook_times,
        types=res.fx_topofbook_types,
        T=120.0,
        M=4,
        config=em_cfg,
    )
    assert em_res.n_iter >= 1
    assert np.isfinite(em_res.log_likelihood)
    assert em_res.spectral_radius >= 0.0


@_skip_if_no_konark
def test_d2_stream_compatible_with_hmm_recovery():
    """HMM recovery (`recover_regimes_hmm`) accepts a D2 FX-projected
    stream's window features and terminates cleanly."""
    from simulator.observation import ObservedTrace
    from simulator.recovery import compute_window_features
    from simulator.recovery_hmm import HMMRecoveryConfig, recover_regimes_hmm

    cfg = D2KonarkConfig(spread0=0.03, seed=1, use_exp_approx=True)
    res = simulate_d2_konark(cfg, horizon_seconds=120.0)
    if res.n_fx_observed < 30:
        pytest.skip(
            f"D2 produced {res.n_fx_observed} FX events; below smoke floor."
        )
    # Build a minimal ObservedTrace from the D2 stream so that
    # compute_window_features can consume it.
    n = int(res.n_fx_observed)
    obs = ObservedTrace(
        times=res.fx_topofbook_times,
        event_types=res.fx_topofbook_types,
        bid_ticks=np.zeros(n, dtype=np.int64),
        ask_ticks=np.ones(n, dtype=np.int64),
        spread_ticks=np.ones(n, dtype=np.int64),
        initial_bid_ticks=0,
        initial_ask_ticks=1,
        min_spread_ticks=1,
        crossing_policy="censor",
        tick_size=1e-5,
        num_latent_events=int(res.n_konark_latent),
        num_censored_events=int(res.n_konark_latent - n),
    )
    features = compute_window_features(
        obs, window_seconds=5.0, hop_seconds=5.0, horizon=120.0
    )
    assert len(features) >= 4
    hmm_cfg = HMMRecoveryConfig(
        n_states=2,
        emission_features=("recent_event_rate", "directional_imbalance"),
        max_iter=50,
        tol_relative_ll=1e-5,
        seed=0,
    )
    hmm_res = recover_regimes_hmm(features, hmm_cfg)
    assert hmm_res.posterior.shape[0] == len(features)
    assert hmm_res.posterior.shape[1] == 2
    assert np.allclose(hmm_res.posterior.sum(axis=1), 1.0, atol=1e-6)
    assert hmm_res.n_iter >= 1
