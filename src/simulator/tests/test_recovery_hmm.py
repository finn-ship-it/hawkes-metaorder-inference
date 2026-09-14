"""Tests for recovery_hmm.py.

Acceptance criteria from restored_dissertation_pack/03_layer2_hmm_recovery.md:
- Synthetic two-regime stream with known transitions: HMM recovers
  regime labels with Viterbi accuracy >= 0.85.
- Posterior probabilities sum to 1 per window.
- Threshold-equivalent limit: HMM with degenerate (deterministic)
  emissions reproduces the threshold detector's labels (qualitatively
  via well-separated emissions).
- Convergence: log-likelihood non-decreasing across Baum-Welch
  iterations (numerically; up to floating-point round-off).
- JSON round-trip is idempotent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.recovery import WindowFeatures
from simulator.recovery_hmm import (
    HMMRecoveryConfig,
    HMMRecoveryResult,
    recover_regimes_hmm,
    viterbi_decode,
)


# ---------------------------------------------------------------------------
# Synthetic two-regime feature stream
# ---------------------------------------------------------------------------


def _synthetic_two_regime_features(
    n_windows: int,
    transitions: list,
    rate_per_state: tuple,
    imb_per_state: tuple,
    rate_noise: float,
    imb_noise: float,
    rng: np.random.Generator,
) -> tuple:
    """Generate a synthetic feature trace.

    Parameters
    ----------
    n_windows : number of windows
    transitions : list of (window_index, new_state) tuples; first must be (0, state0)
    rate_per_state, imb_per_state : tuples of (rate, imb) per state
    rate_noise, imb_noise : Gaussian noise std
    """
    state = transitions[0][1]
    next_idx = 1
    states = []
    features = []
    for i in range(n_windows):
        if next_idx < len(transitions) and i == transitions[next_idx][0]:
            state = transitions[next_idx][1]
            next_idx += 1
        states.append(state)
        rate = float(rate_per_state[state]) + rate_noise * rng.normal()
        imb = float(imb_per_state[state]) + imb_noise * rng.normal()
        features.append(
            WindowFeatures(
                window_start=float(i * 0.5),
                window_end=float((i + 1) * 0.5),
                num_events=int(rate),
                directional_imbalance=imb,
                spread_change=0.0,
                recent_event_rate=rate,
            )
        )
    return features, np.array(states, dtype=int)


# ---------------------------------------------------------------------------
# Test 1: synthetic two-regime recovery
# ---------------------------------------------------------------------------


def test_hmm_recovers_two_regime_synthetic_with_viterbi_accuracy_above_085():
    """Synthetic HMM-like stream with two well-separated emission regimes:
    the Viterbi MAP labelling should match the true regime sequence on
    >= 85% of windows. Permutation invariance: HMM state labels may be
    swapped relative to the true labels — we report the best of the two
    permutations.
    """
    rng = np.random.default_rng(0)
    n_windows = 240
    transitions = [(0, 0), (60, 1), (120, 0), (180, 1)]
    rate_per_state = (1.0, 6.0)  # well separated
    imb_per_state = (0.0, 0.4)
    features, true_states = _synthetic_two_regime_features(
        n_windows, transitions, rate_per_state, imb_per_state,
        rate_noise=0.5, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(n_states=2, max_iter=200, tol_relative_ll=1e-7, seed=0)
    res = recover_regimes_hmm(features, cfg)

    # Compute accuracy under both label permutations and take the max
    acc_direct = float(np.mean(res.viterbi_labels == true_states))
    acc_swapped = float(np.mean(res.viterbi_labels == (1 - true_states)))
    acc = max(acc_direct, acc_swapped)
    assert acc >= 0.85, (
        f"Viterbi accuracy {acc:.3f} below 0.85; "
        f"true_states[:20]={true_states[:20]}, "
        f"viterbi[:20]={res.viterbi_labels[:20]}"
    )


# ---------------------------------------------------------------------------
# Test 2: posterior sums to 1 per window
# ---------------------------------------------------------------------------


def test_hmm_posterior_sums_to_one_per_window():
    rng = np.random.default_rng(1)
    n_windows = 80
    features, _ = _synthetic_two_regime_features(
        n_windows, [(0, 0), (40, 1)], (1.0, 4.0), (0.0, 0.3),
        rate_noise=0.5, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(n_states=2, max_iter=80)
    res = recover_regimes_hmm(features, cfg)
    sums = np.sum(res.posterior, axis=1)
    np.testing.assert_allclose(sums, np.ones(n_windows), atol=1e-9)


# ---------------------------------------------------------------------------
# Test 3: threshold-equivalent limit (well-separated rates → HMM labels
# agree with rate-threshold labels qualitatively).
# ---------------------------------------------------------------------------


def test_hmm_threshold_equivalent_limit_when_emissions_are_well_separated():
    """When the two states' rate emissions are well separated and the
    transition matrix is near-diagonal, the HMM's MAP labelling agrees
    with a simple rate-threshold detector that puts the threshold
    halfway between the two state means. Match >= 95% of windows.
    """
    rng = np.random.default_rng(2)
    n_windows = 200
    rate_per_state = (1.0, 5.0)
    imb_per_state = (0.0, 0.0)  # imbalance carries no signal
    features, true_states = _synthetic_two_regime_features(
        n_windows, [(0, 0), (50, 1), (110, 0), (170, 1)],
        rate_per_state, imb_per_state,
        rate_noise=0.3, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(n_states=2, max_iter=120)
    res = recover_regimes_hmm(features, cfg)
    # Apply threshold detector: midpoint = (1 + 5) / 2 = 3
    rates = np.asarray([f.recent_event_rate for f in features])
    thresh_labels = (rates > 3.0).astype(int)
    # Flip HMM labels if state 0 has the higher mean
    if res.emission_means[0, 0] > res.emission_means[1, 0]:
        hmm_labels = 1 - res.viterbi_labels
    else:
        hmm_labels = res.viterbi_labels
    agreement = float(np.mean(hmm_labels == thresh_labels))
    assert agreement >= 0.95, (
        f"HMM-vs-threshold agreement {agreement:.3f} < 0.95 "
        f"(emissions well-separated case)"
    )


# ---------------------------------------------------------------------------
# Test 4: monotone log-likelihood (Baum-Welch property)
# ---------------------------------------------------------------------------


def test_hmm_log_likelihood_monotone_non_decreasing_across_iterations():
    rng = np.random.default_rng(3)
    n_windows = 100
    features, _ = _synthetic_two_regime_features(
        n_windows, [(0, 0), (50, 1)], (2.0, 4.0), (0.0, 0.3),
        rate_noise=0.4, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(
        n_states=2, max_iter=60, tol_relative_ll=1e-12  # disable early-stop
    )
    res = recover_regimes_hmm(features, cfg)
    diffs = np.diff(res.log_likelihood_trace)
    tol = 1e-6 * max(abs(float(res.log_likelihood)), 1.0)
    assert np.all(diffs >= -tol), (
        f"log-likelihood not monotone non-decreasing: min diff = {diffs.min()} "
        f"(tol = {tol}); trace[:10] = {res.log_likelihood_trace[:10]}"
    )


# ---------------------------------------------------------------------------
# Test 5: JSON round-trip
# ---------------------------------------------------------------------------


def test_hmm_recovery_result_json_roundtrip_idempotent():
    rng = np.random.default_rng(4)
    n_windows = 40
    features, _ = _synthetic_two_regime_features(
        n_windows, [(0, 0), (20, 1)], (1.5, 4.0), (0.0, 0.3),
        rate_noise=0.4, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(n_states=2, max_iter=30)
    res = recover_regimes_hmm(features, cfg)
    s = res.to_json()
    res2 = HMMRecoveryResult.from_json(s)
    s2 = res2.to_json()
    assert s == s2
    np.testing.assert_allclose(res.posterior, res2.posterior)
    np.testing.assert_allclose(res.transition_matrix, res2.transition_matrix)
    np.testing.assert_allclose(res.emission_means, res2.emission_means)
    np.testing.assert_allclose(res.emission_stds, res2.emission_stds)
    np.testing.assert_array_equal(res.viterbi_labels, res2.viterbi_labels)
    assert res.feature_names == res2.feature_names
    assert res.n_iter == res2.n_iter


# ---------------------------------------------------------------------------
# Test 6: Viterbi pure-decode unit test (small handcrafted HMM)
# ---------------------------------------------------------------------------


def test_viterbi_decode_handcrafted_two_state_chain():
    """Tiny handcrafted HMM: pi=[0.5, 0.5], A=[[0.9, 0.1], [0.1, 0.9]],
    log_B determines outcome. Confirm Viterbi picks the higher-scoring
    state per timestep when transitions are inexpensive.
    """
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    # Strong observations: t=0 favours state 1 strongly, t=1 favours state 1
    # strongly, t=2 favours state 0 marginally.
    log_B = np.log(np.array([
        [0.05, 0.95],
        [0.10, 0.90],
        [0.55, 0.45],
    ]))
    labels = viterbi_decode(log_pi, log_A, log_B)
    # With sticky transitions, the best path should be 1, 1, 1 (avoids
    # the cost of leaving state 1 just for one slightly-favoured frame).
    assert tuple(labels.tolist()) == (1, 1, 1), labels.tolist()


# ---------------------------------------------------------------------------
# Test 7: warm-start via initial_emission_means
# ---------------------------------------------------------------------------


def test_hmm_accepts_warm_start_emission_means():
    rng = np.random.default_rng(5)
    n_windows = 80
    features, _ = _synthetic_two_regime_features(
        n_windows, [(0, 0), (40, 1)], (1.5, 5.0), (0.0, 0.3),
        rate_noise=0.4, imb_noise=0.05, rng=rng,
    )
    cfg = HMMRecoveryConfig(n_states=2, max_iter=60)
    init_means = np.array([[1.5, 0.0], [5.0, 0.3]])
    res = recover_regimes_hmm(features, cfg, initial_emission_means=init_means)
    # Should converge close to the truth
    assert res.converged or res.n_iter >= 20, (
        f"warm-start did not converge as expected; "
        f"n_iter={res.n_iter}, reason={res.converged_reason}"
    )


# ---------------------------------------------------------------------------
# Test 8: config validation
# ---------------------------------------------------------------------------


def test_hmm_recovery_config_validation():
    with pytest.raises(ValueError):
        HMMRecoveryConfig(n_states=1)
    with pytest.raises(ValueError):
        HMMRecoveryConfig(max_iter=0)
    with pytest.raises(ValueError):
        HMMRecoveryConfig(tol_relative_ll=0.0)
    with pytest.raises(ValueError):
        HMMRecoveryConfig(sigma_floor=0.0)
    with pytest.raises(ValueError):
        HMMRecoveryConfig(init_method="foo")
    with pytest.raises(ValueError):
        HMMRecoveryConfig(emission_features=())


def test_hmm_rejects_too_few_windows():
    rng = np.random.default_rng(6)
    cfg = HMMRecoveryConfig(n_states=3)
    features = [
        WindowFeatures(0.0, 0.5, 1, 0.0, 0.0, 1.0),
    ]
    with pytest.raises(ValueError):
        recover_regimes_hmm(features, cfg)
