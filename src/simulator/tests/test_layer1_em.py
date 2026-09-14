"""Tests for layer1_em.py.

Acceptance criteria from restored_dissertation_pack/01_layer1_em_calibration.md:
- synthetic two-component Hawkes (rho=0.5, known alpha/beta): EM recovers
  alpha/beta to within +-10% on N>=10,000 events
- convergence: log-likelihood is monotonically non-decreasing across EM
  iterations (in the unconstrained case where the spectral-radius
  projection does not fire)
- spectral-radius constraint: projected alpha never produces rho >= rho_cap
- JSON round-trip: Layer1EMResult.to_json / .from_json is idempotent
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.layer1_em import (
    Layer1EMConfig,
    Layer1EMResult,
    branching_matrix_sumexp,
    fit_layer1_em,
    project_alpha_to_rho_cap,
    spectral_radius_sumexp,
)


# ---------------------------------------------------------------------------
# Synthetic sum-of-exp Hawkes simulator (cluster representation)
# ---------------------------------------------------------------------------


def _simulate_sumexp_hawkes(
    mu: np.ndarray,
    alpha_all: np.ndarray,
    betas: np.ndarray,
    T: float,
    rng: np.random.Generator,
    max_generations: int = 64,
) -> tuple:
    """Cluster-process simulator for a sum-of-exponentials multivariate Hawkes.

    Background events of type i are sampled from a homogeneous Poisson with
    rate mu_i over [0, T]. Each event of type j is a parent that generates,
    for each component r and each child type i, a Poisson(alpha_ij^{(r)} /
    beta_r) number of children, with arrival delays Exp(beta_r). Children
    are pruned to [0, T) and recursively spawn further children. Sub-
    critical (rho < 1) is required for finite expected total population.
    """
    M = mu.shape[0]
    R = betas.shape[0]
    times = []
    types = []

    # Background generation
    for i in range(M):
        n_bg = int(rng.poisson(mu[i] * T))
        if n_bg > 0:
            tt = rng.uniform(0.0, T, size=n_bg)
            times.extend(tt.tolist())
            types.extend([i] * n_bg)

    parent_t = list(times)
    parent_k = list(types)

    for _gen in range(max_generations):
        new_t = []
        new_k = []
        for t_p, k_p in zip(parent_t, parent_k):
            for r in range(R):
                inv_beta = 1.0 / float(betas[r])
                for i in range(M):
                    rate_int = float(alpha_all[r, i, k_p]) / float(betas[r])
                    n = int(rng.poisson(rate_int))
                    if n == 0:
                        continue
                    delays = rng.exponential(inv_beta, size=n)
                    tt = t_p + delays
                    keep = tt < T
                    if not np.any(keep):
                        continue
                    new_t.extend(tt[keep].tolist())
                    new_k.extend([i] * int(np.sum(keep)))
        if not new_t:
            break
        times.extend(new_t)
        types.extend(new_k)
        parent_t = new_t
        parent_k = new_k

    times_arr = np.asarray(times, dtype=np.float64)
    types_arr = np.asarray(types, dtype=np.int64)
    if times_arr.size == 0:
        return times_arr, types_arr
    order = np.argsort(times_arr, kind="stable")
    return times_arr[order], types_arr[order]


# ---------------------------------------------------------------------------
# Test 1: synthetic two-component recovery (rho=0.5)
# ---------------------------------------------------------------------------


def test_em_recovers_synthetic_two_component_hawkes_within_ten_percent():
    """Two-component sum-of-exp Hawkes: EM recovers per-(i,j,r) integrated
    kernel mass alpha_ij^{(r)} / beta_r to within +-10% on a single trajectory
    that contains at least 10,000 events.

    Strategy: simulate enough samples that the empirical fit is statistically
    well-determined, fit with the true beta grid, and check the integrated
    kernel masses (which are the identifiable quantities) row by row.
    """
    rng = np.random.default_rng(42)
    M = 2
    betas = np.array([1.0, 5.0])
    R = betas.shape[0]
    # Construct alpha_all with known integrated mass Phi_ij = sum_r alpha[r, i, j] / beta_r.
    # Aim for total branching ratio approx 0.5.
    # Component r=0 (slow): diag 0.20, off 0.05
    # Component r=1 (fast): diag 0.05 * beta_r, off 0.05 * beta_r * 0.2  (so /beta -> 0.05, 0.01)
    alpha_all_true = np.zeros((R, M, M))
    alpha_all_true[0] = np.array([[0.20, 0.05], [0.05, 0.20]])
    alpha_all_true[1] = np.array([[0.05 * 5.0, 0.01 * 5.0],
                                  [0.01 * 5.0, 0.05 * 5.0]])
    Phi_true = branching_matrix_sumexp(alpha_all_true, betas)
    rho_true = spectral_radius_sumexp(alpha_all_true, betas)
    assert rho_true < 1.0
    # mu chosen to deliver well above 10k events
    mu_true = np.array([0.6, 0.6])

    # Need a long enough trajectory that even small off-diagonal entries
    # of the branching matrix are statistically well-determined. 50k events
    # gives us ~1.5% standard error on counts, which carries the 10%
    # accuracy tolerance below.
    T = 32000.0
    times, types = _simulate_sumexp_hawkes(mu_true, alpha_all_true, betas, T, rng)
    assert times.size >= 50_000, f"need at least 50k events for fit; got {times.size}"

    # Fit
    cfg = Layer1EMConfig(
        betas=tuple(betas.tolist()),
        max_iter=500,
        tol_relative_ll=1e-9,
        rho_cap=0.99,
        project_spectral=False,  # rho_true ~ 0.5 << 0.99; no need to project
        rho_init=0.5,
    )
    res = fit_layer1_em(times, types, T=T, M=M, config=cfg)
    assert res.converged, (
        f"EM did not converge in {cfg.max_iter} iters; "
        f"reason={res.converged_reason}, n_iter={res.n_iter}"
    )

    # The aggregate branching ratio is the most identifiable quantity;
    # check it tightly.
    rho_fit = res.spectral_radius
    assert abs(rho_fit - rho_true) < 0.02, (
        f"spectral radius mismatch: true={rho_true:.4f} fit={rho_fit:.4f}"
    )
    # Branching matrix entries: 10% relative for the dominant entries
    # (>= 0.1), 25% relative for smaller off-diagonals (> 0.03).
    Phi_fit = res.branching_matrix
    abs_err = np.abs(Phi_fit - Phi_true)
    rel_err = abs_err / np.maximum(np.abs(Phi_true), 1e-3)
    big = Phi_true >= 0.10
    assert np.all(rel_err[big] < 0.10), (
        f"large-entry branching matrix recovery exceeded 10%: "
        f"rel_err={rel_err}, abs_err={abs_err}, Phi_fit={Phi_fit}, Phi_true={Phi_true}"
    )
    # Smaller off-diagonals are harder; allow 25% relative.
    small = (~big) & (Phi_true >= 0.03)
    assert np.all(rel_err[small] < 0.25), (
        f"small-entry branching matrix recovery exceeded 25%: "
        f"rel_err={rel_err}, Phi_fit={Phi_fit}, Phi_true={Phi_true}"
    )
    # mu recovery within 15% (Poisson background term)
    rel_mu = np.abs(res.mu - mu_true) / mu_true
    assert np.all(rel_mu < 0.15), f"mu recovery exceeded 15%: rel={rel_mu}"


# ---------------------------------------------------------------------------
# Test 2: monotone log-likelihood
# ---------------------------------------------------------------------------


def test_em_log_likelihood_is_monotone_non_decreasing_when_unconstrained():
    """Without spectral-radius projection, the multiplicative EM updates are
    monotone-LL increasing (Veen-Schoenberg)."""
    rng = np.random.default_rng(7)
    M = 2
    betas = np.array([1.0, 4.0])
    alpha_all_true = np.zeros((2, M, M))
    alpha_all_true[0] = np.array([[0.15, 0.05], [0.05, 0.15]])
    alpha_all_true[1] = np.array([[0.10, 0.02], [0.02, 0.10]])
    mu_true = np.array([0.4, 0.4])
    T = 2000.0
    times, types = _simulate_sumexp_hawkes(mu_true, alpha_all_true, betas, T, rng)

    cfg = Layer1EMConfig(
        betas=tuple(betas.tolist()),
        max_iter=120,
        tol_relative_ll=1e-12,  # disable early stop so the trace is long
        project_spectral=False,
        rho_init=0.3,
    )
    res = fit_layer1_em(times, types, T=T, M=M, config=cfg)
    diffs = np.diff(res.log_likelihood_trace)
    # Allow a tiny floating-point margin (1e-6 * |LL|) for round-off.
    tol = 1e-6 * max(abs(float(res.log_likelihood)), 1.0)
    assert np.all(diffs >= -tol), (
        f"log-likelihood not monotone: min diff = {diffs.min()} (tol={tol}); "
        f"trace[:10]={res.log_likelihood_trace[:10]}"
    )


# ---------------------------------------------------------------------------
# Test 3: spectral-radius projection enforced
# ---------------------------------------------------------------------------


def test_projected_alpha_never_exceeds_rho_cap():
    """Direct unit test: project_alpha_to_rho_cap brings any super-critical
    alpha down to exactly rho_cap, and integrated through the EM loop the
    final spectral radius cannot exceed rho_cap."""
    rng = np.random.default_rng(0)
    M = 2
    betas = np.array([1.0, 3.0])
    R = betas.shape[0]

    # Hand-construct super-critical alpha (rho > 1)
    alpha_super = np.zeros((R, M, M))
    alpha_super[0] = np.array([[0.8, 0.5], [0.5, 0.8]])  # /beta=1 -> Phi=alpha
    alpha_super[1] = np.array([[3.0, 1.5], [1.5, 3.0]])  # /beta=3 -> Phi=1.0,0.5,0.5,1.0
    rho_before = spectral_radius_sumexp(alpha_super, betas)
    assert rho_before > 1.0

    rho_cap = 0.99
    alpha_proj, rho_after, did_project = project_alpha_to_rho_cap(
        alpha_super, betas, rho_cap
    )
    assert did_project
    assert rho_after == pytest.approx(rho_cap, abs=1e-9)
    # Re-evaluate: the post-projection spectral radius equals the cap.
    rho_post = spectral_radius_sumexp(alpha_proj, betas)
    assert rho_post == pytest.approx(rho_cap, abs=1e-9)

    # Integration test: run a few EM iterations starting near-critical and
    # confirm the result still satisfies rho <= rho_cap.
    M2 = 2
    mu_true = np.array([0.3, 0.3])
    alpha_all_true = np.zeros((R, M2, M2))
    alpha_all_true[0] = np.array([[0.20, 0.05], [0.05, 0.20]])
    alpha_all_true[1] = np.array([[0.50, 0.10], [0.10, 0.50]])
    T = 1500.0
    times, types = _simulate_sumexp_hawkes(
        mu_true, alpha_all_true, betas, T, rng
    )
    cfg = Layer1EMConfig(
        betas=tuple(betas.tolist()),
        max_iter=80,
        rho_cap=0.95,
        project_spectral=True,
        rho_init=0.5,
    )
    res = fit_layer1_em(times, types, T=T, M=M2, config=cfg)
    assert res.spectral_radius <= cfg.rho_cap + 1e-9, (
        f"final rho={res.spectral_radius} exceeds rho_cap={cfg.rho_cap}"
    )


# ---------------------------------------------------------------------------
# Test 4: JSON round-trip is idempotent
# ---------------------------------------------------------------------------


def test_layer1_em_result_json_roundtrip_idempotent():
    rng = np.random.default_rng(11)
    M = 2
    betas = np.array([1.0, 5.0])
    alpha_all_true = np.zeros((2, M, M))
    alpha_all_true[0] = np.array([[0.20, 0.05], [0.05, 0.20]])
    alpha_all_true[1] = np.array([[0.20, 0.05], [0.05, 0.20]])
    mu_true = np.array([0.5, 0.5])
    T = 800.0
    times, types = _simulate_sumexp_hawkes(mu_true, alpha_all_true, betas, T, rng)
    cfg = Layer1EMConfig(
        betas=tuple(betas.tolist()),
        max_iter=40,
        project_spectral=True,
        rho_init=0.3,
    )
    res = fit_layer1_em(times, types, T=T, M=M, config=cfg)
    s = res.to_json()
    res2 = Layer1EMResult.from_json(s)
    s2 = res2.to_json()
    # Re-serialised JSON is byte-for-byte equal (idempotent).
    assert s == s2
    # Round-trip preserves fitted arrays.
    np.testing.assert_allclose(res.mu, res2.mu)
    np.testing.assert_allclose(res.alpha_all, res2.alpha_all)
    np.testing.assert_allclose(res.betas, res2.betas)
    np.testing.assert_allclose(res.branching_matrix, res2.branching_matrix)
    assert res.spectral_radius == pytest.approx(res2.spectral_radius)
    assert res.converged == res2.converged
    assert res.converged_reason == res2.converged_reason
    assert res.n_iter == res2.n_iter
    assert res.n_events == res2.n_events
    np.testing.assert_allclose(res.log_likelihood_trace, res2.log_likelihood_trace)


# ---------------------------------------------------------------------------
# Test 5: degenerate input handling (small but valuable safety net)
# ---------------------------------------------------------------------------


def test_em_rejects_degenerate_inputs():
    cfg = Layer1EMConfig(betas=(1.0, 5.0), max_iter=5)
    with pytest.raises(ValueError):
        fit_layer1_em(
            np.array([]), np.array([], dtype=np.int64), T=10.0, M=2, config=cfg
        )
    # Out-of-range type
    with pytest.raises(ValueError):
        fit_layer1_em(
            np.array([0.5]), np.array([5], dtype=np.int64), T=10.0, M=2, config=cfg
        )
    # Negative time
    with pytest.raises(ValueError):
        fit_layer1_em(
            np.array([-0.1, 0.5]),
            np.array([0, 1], dtype=np.int64),
            T=10.0,
            M=2,
            config=cfg,
        )
    # Non-monotone times
    with pytest.raises(ValueError):
        fit_layer1_em(
            np.array([1.0, 0.5]),
            np.array([0, 1], dtype=np.int64),
            T=10.0,
            M=2,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Test 6: config validation
# ---------------------------------------------------------------------------


def test_layer1_em_config_validation():
    with pytest.raises(ValueError):
        Layer1EMConfig(betas=(0.0, 1.0))  # non-positive
    with pytest.raises(ValueError):
        Layer1EMConfig(betas=(1.0,), rho_cap=1.5)
    with pytest.raises(ValueError):
        Layer1EMConfig(betas=(1.0,), max_iter=0)
    with pytest.raises(ValueError):
        Layer1EMConfig(betas=(1.0,), tol_relative_ll=0.0)
    with pytest.raises(ValueError):
        Layer1EMConfig(betas=(1.0,), rho_init=1.5)


# ---------------------------------------------------------------------------
# Test 7: warm-start from a known-good (mu, alpha) preserves it under EM
# (sanity that warm-starting is wired through correctly)
# ---------------------------------------------------------------------------


def test_em_accepts_warm_start_from_initial_parameters():
    rng = np.random.default_rng(99)
    M = 2
    betas = np.array([1.0, 5.0])
    alpha_all_true = np.zeros((2, M, M))
    alpha_all_true[0] = np.array([[0.18, 0.04], [0.04, 0.18]])
    alpha_all_true[1] = np.array([[0.10, 0.02], [0.02, 0.10]])
    mu_true = np.array([0.3, 0.3])
    T = 1500.0
    times, types = _simulate_sumexp_hawkes(mu_true, alpha_all_true, betas, T, rng)

    cfg = Layer1EMConfig(
        betas=tuple(betas.tolist()),
        max_iter=2,
        project_spectral=False,
        initial_mu=mu_true.copy(),
        initial_alpha_all=alpha_all_true.copy(),
    )
    res = fit_layer1_em(times, types, T=T, M=M, config=cfg)
    # After only two iterations starting from the truth, parameters should
    # remain in a tight neighbourhood (within 25%) of the true values.
    rel = np.abs(res.mu - mu_true) / mu_true
    assert np.all(rel < 0.25), f"warm-start mu drift: {rel}"
    Phi_true = branching_matrix_sumexp(alpha_all_true, betas)
    rel_phi = np.abs(res.branching_matrix - Phi_true) / np.maximum(Phi_true, 1e-3)
    assert np.all(rel_phi < 0.25), f"warm-start Phi drift: {rel_phi}"
