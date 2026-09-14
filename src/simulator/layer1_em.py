"""
layer1_em.py — Layer-1 EM calibration for sum-of-exponentials Hawkes.

Implements the Veen-Schoenberg cluster-process EM with a fixed beta grid.
The fixed-beta-grid case is the parametric expansion-on-basis path. The
Wiener-Hopf non-parametric path lives in layer1_nonparametric.py.

Mathematical model
------------------
Multivariate Hawkes with kernel

    phi_ij(u) = sum_{r=1}^{R} alpha_ij^{(r)} exp(-beta_r u)

and conditional intensity

    lambda_i(t) = mu_i + sum_j sum_r alpha_ij^{(r)} sum_{t_m < t, type_m = j}
                                                    exp(-beta_r (t - t_m)).

The R beta values are fixed inputs; only mu (M,) and alpha_all (R, M, M)
are estimated. This matches the offline L-BFGS-B sum-of-exp anchor at
betas = (0.1, 1, 10, 100) s^-1 used by the EURUSD primary-day fit.

EM updates (multiplicative, monotone-LL increasing in the unconstrained
case; see Veen and Schoenberg, JASA 2008)

    mu_i_new        = mu_i_old   * sum_{k: type=i}    1 / lambda_i(t_k)   / T

    alpha_ij^{(r)}_new = alpha_ij^{(r)}_old * B_ij^{(r)} / G_j^{(r)}

with

    B_ij^{(r)} = sum_{k: type=i} A_j^{(r)}(t_k^-) / lambda_i(t_k)
    A_j^{(r)}(t) = sum_{t_m < t, type=j} exp(-beta_r (t - t_m))
    G_j^{(r)}    = sum_{t_m: type=j} (1 - exp(-beta_r (T - t_m))) / beta_r

(A_j^{(r)} maintained recursively across the event loop in O(N R M).)

Spectral-radius projection
--------------------------
The branching matrix Phi_ij = sum_r alpha_ij^{(r)} / beta_r must have
spectral radius rho < 1 for stationarity. After each M-step, if
rho(Phi_new) > rho_cap, alpha_all is scaled by rho_cap / rho(Phi_new).
This breaks strict EM monotonicity but is required for stability when
warm-starting near the boundary of the stationary region.

Reference for L-BFGS-B comparison: Data/layer1_hawkes_fit.py
(`fit_sumexp_hawkes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import json

import numpy as np


__all__ = [
    "Layer1EMConfig",
    "Layer1EMResult",
    "fit_layer1_em",
    "branching_matrix_sumexp",
    "spectral_radius_sumexp",
    "project_alpha_to_rho_cap",
    "log_likelihood_sumexp",
]


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------


@dataclass
class Layer1EMConfig:
    """Configuration for sum-of-exponentials Hawkes EM.

    Parameters
    ----------
    betas : sequence of strictly positive floats, length R.
        Fixed decay-rate grid shared across (i, j) pairs.
    max_iter : int.
        Maximum EM iterations (default 200).
    tol_relative_ll : float.
        Convergence threshold on relative log-likelihood improvement.
        Stops when |LL_new - LL_old| / max(|LL_new|, 1.0) < tol.
    rho_cap : float in (0, 1).
        Spectral-radius cap; alpha_all is scaled down at the end of any
        M-step where rho(Phi) > rho_cap.
    project_spectral : bool.
        If False, the rho_cap projection is disabled (used in synthetic
        recovery tests where the true rho is well below the cap).
    initial_mu : optional ndarray (M,).
        Warm-start mu. If None, base rate N_i / T is used (Poisson-like).
    initial_alpha_all : optional ndarray (R, M, M).
        Warm-start alpha. If None, a structured initialisation is used:
        diagonal entries set to give an initial branching ratio of
        rho_init split across components inversely proportional to
        beta_r; off-diagonals at one tenth of diagonal.
    rho_init : float.
        Initial branching ratio used when initial_alpha_all is None
        (default 0.3).
    """

    betas: Tuple[float, ...]
    max_iter: int = 200
    tol_relative_ll: float = 1.0e-7
    rho_cap: float = 0.99
    project_spectral: bool = True
    initial_mu: Optional[np.ndarray] = None
    initial_alpha_all: Optional[np.ndarray] = None
    rho_init: float = 0.3
    record_trace: bool = True

    def __post_init__(self) -> None:
        b = np.asarray(self.betas, dtype=float)
        if b.ndim != 1 or b.size == 0:
            raise ValueError(f"betas must be a 1-D sequence; got shape {b.shape}")
        if np.any(b <= 0.0):
            raise ValueError("betas must be strictly positive")
        self.betas = tuple(b.tolist())
        if not (0.0 < self.rho_cap < 1.0):
            raise ValueError(f"rho_cap must lie in (0, 1); got {self.rho_cap}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if self.tol_relative_ll <= 0.0:
            raise ValueError("tol_relative_ll must be > 0")
        if not (0.0 < self.rho_init < 1.0):
            raise ValueError(f"rho_init must lie in (0, 1); got {self.rho_init}")


@dataclass
class Layer1EMResult:
    """Fitted parameters and diagnostics from a sum-of-exp EM run."""

    mu: np.ndarray                       # (M,)
    alpha_all: np.ndarray                # (R, M, M)
    betas: np.ndarray                    # (R,)
    branching_matrix: np.ndarray         # (M, M)
    spectral_radius: float
    log_likelihood: float
    log_likelihood_trace: np.ndarray     # (n_iter,)
    n_iter: int
    converged: bool
    converged_reason: str                # "tol", "maxiter", "rho_cap_saturated"
    fit_window_seconds: float
    n_events: int
    rho_cap: float
    projection_count: int                # number of M-steps where projection fired

    # ---------------- JSON round-trip ----------------

    def to_json(self) -> str:
        payload = {
            "mu": self.mu.tolist(),
            "alpha_all": self.alpha_all.tolist(),
            "betas": self.betas.tolist(),
            "branching_matrix": self.branching_matrix.tolist(),
            "spectral_radius": float(self.spectral_radius),
            "log_likelihood": float(self.log_likelihood),
            "log_likelihood_trace": self.log_likelihood_trace.tolist(),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "converged_reason": self.converged_reason,
            "fit_window_seconds": float(self.fit_window_seconds),
            "n_events": int(self.n_events),
            "rho_cap": float(self.rho_cap),
            "projection_count": int(self.projection_count),
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Layer1EMResult":
        d = json.loads(s)
        return cls(
            mu=np.asarray(d["mu"], dtype=float),
            alpha_all=np.asarray(d["alpha_all"], dtype=float),
            betas=np.asarray(d["betas"], dtype=float),
            branching_matrix=np.asarray(d["branching_matrix"], dtype=float),
            spectral_radius=float(d["spectral_radius"]),
            log_likelihood=float(d["log_likelihood"]),
            log_likelihood_trace=np.asarray(d["log_likelihood_trace"], dtype=float),
            n_iter=int(d["n_iter"]),
            converged=bool(d["converged"]),
            converged_reason=str(d["converged_reason"]),
            fit_window_seconds=float(d["fit_window_seconds"]),
            n_events=int(d["n_events"]),
            rho_cap=float(d["rho_cap"]),
            projection_count=int(d["projection_count"]),
        )


# ---------------------------------------------------------------------------
# Branching matrix and spectral-radius helpers
# ---------------------------------------------------------------------------


def branching_matrix_sumexp(alpha_all: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Return the (M, M) branching matrix Phi_ij = sum_r alpha_ij^{(r)} / beta_r."""
    return np.sum(alpha_all / betas[:, None, None], axis=0)


def spectral_radius_sumexp(alpha_all: np.ndarray, betas: np.ndarray) -> float:
    Phi = branching_matrix_sumexp(alpha_all, betas)
    return float(np.max(np.abs(np.linalg.eigvals(Phi))))


def project_alpha_to_rho_cap(
    alpha_all: np.ndarray, betas: np.ndarray, rho_cap: float
) -> Tuple[np.ndarray, float, bool]:
    """If rho(Phi(alpha_all)) > rho_cap, scale alpha_all uniformly so rho == rho_cap.

    Returns (alpha_projected, rho_after, projected_flag).
    """
    rho = spectral_radius_sumexp(alpha_all, betas)
    if rho <= rho_cap or rho == 0.0:
        return alpha_all, float(rho), False
    scale = rho_cap / rho
    return alpha_all * scale, float(rho_cap), True


def log_likelihood_sumexp(
    times: np.ndarray,
    types: np.ndarray,
    T: float,
    M: int,
    mu: np.ndarray,
    alpha_all: np.ndarray,
    betas: np.ndarray,
) -> float:
    """Compute log-likelihood at (mu, alpha_all) for a sum-of-exp Hawkes.

    Uses the same recursive accumulator structure as _em_one_iteration.
    No M-step is taken; this is a pure score function suitable for
    cross-checking parameters fitted by a different optimiser.
    """
    times = np.ascontiguousarray(times, dtype=np.float64)
    types = np.ascontiguousarray(types, dtype=np.int64)
    mu = np.asarray(mu, dtype=np.float64)
    alpha_all = np.asarray(alpha_all, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    R = betas.shape[0]
    N = times.shape[0]
    A = np.zeros((R, M), dtype=np.float64)
    log_lik = 0.0
    t_prev = 0.0
    for k in range(N):
        tk = float(times[k])
        ki = int(types[k])
        dt = tk - t_prev
        if dt > 0.0:
            A *= np.exp(-betas[:, None] * dt)
        contrib = float(np.einsum("rj,rj->", alpha_all[:, ki, :], A))
        lam = float(mu[ki]) + contrib
        if lam <= 0.0 or not np.isfinite(lam):
            return float("-inf")
        log_lik += float(np.log(lam))
        A[:, ki] += 1.0
        t_prev = tk
    G = _compensator_G(times, types, T, M, betas)
    comp = float(np.sum(mu) * T) + float(np.einsum("rij,rj->", alpha_all, G))
    return float(log_lik - comp)


# ---------------------------------------------------------------------------
# Inner E/M-step computations
# ---------------------------------------------------------------------------


def _compensator_G(
    times: np.ndarray, types: np.ndarray, T: float, M: int, betas: np.ndarray
) -> np.ndarray:
    """Compute G[r, j] = sum_{m: type=j} (1 - exp(-beta_r (T - t_m))) / beta_r."""
    R = betas.shape[0]
    G = np.zeros((R, M), dtype=np.float64)
    for j in range(M):
        tm = times[types == j]
        if tm.size == 0:
            continue
        # (R, n_tm)
        decay = np.exp(-betas[:, None] * (T - tm)[None, :])
        G[:, j] = np.sum(1.0 - decay, axis=1) / betas
    return G


def _em_one_iteration(
    times: np.ndarray,
    types: np.ndarray,
    T: float,
    M: int,
    betas: np.ndarray,
    mu: np.ndarray,
    alpha_all: np.ndarray,
    G: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """One Veen-Schoenberg EM iteration. Returns (mu_new, alpha_new, log_likelihood).

    Cost: O(N * R * M) using a recursive update of A_j^{(r)}.
    """
    R = betas.shape[0]
    N = times.shape[0]
    A = np.zeros((R, M), dtype=np.float64)         # active mass at t_k^-
    C = np.zeros(M, dtype=np.float64)              # background numerator per i
    B = np.zeros((R, M, M), dtype=np.float64)      # B[r, i, j]
    log_lik = 0.0
    t_prev = 0.0
    for k in range(N):
        tk = float(times[k])
        ki = int(types[k])
        dt = tk - t_prev
        if dt > 0.0:
            A *= np.exp(-betas[:, None] * dt)
        # Intensity for event k of type ki: lambda = mu[ki] + sum_r sum_j alpha[r, ki, j] * A[r, j]
        contrib = np.einsum("rj,rj->", alpha_all[:, ki, :], A)
        lam = mu[ki] + contrib
        if lam <= 0.0 or not np.isfinite(lam):
            return mu, alpha_all, float("-inf")
        log_lik += float(np.log(lam))
        inv_lam = 1.0 / lam
        # Accumulate
        C[ki] += inv_lam
        # B[r, ki, j] += inv_lam * A[r, j]
        B[:, ki, :] += inv_lam * A
        # Add this event to A (jump only on its own type)
        A[:, ki] += 1.0
        t_prev = tk

    # Compensator: integral of total intensity over [0, T]
    comp = float(np.sum(mu) * T) + float(np.einsum("rij,rj->", alpha_all, G))
    log_lik -= comp

    # M-step (multiplicative)
    mu_new = mu * C / T
    G_safe = np.maximum(G, 1.0e-30)
    alpha_new = alpha_all * B / G_safe[:, None, :]
    return mu_new, alpha_new, log_lik


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------


def fit_layer1_em(
    times: np.ndarray,
    types: np.ndarray,
    T: float,
    M: int,
    config: Layer1EMConfig,
) -> Layer1EMResult:
    """Fit a sum-of-exp multivariate Hawkes by Veen-Schoenberg EM.

    Parameters
    ----------
    times : ndarray (N,), strictly increasing event times in seconds.
    types : ndarray (N,) of int in {0, ..., M-1}.
    T : float, fit window length in seconds (events assumed inside [0, T]).
    M : int, dimension (number of event types).
    config : Layer1EMConfig.

    Returns
    -------
    Layer1EMResult.
    """
    times = np.ascontiguousarray(times, dtype=np.float64)
    types = np.ascontiguousarray(types, dtype=np.int64)
    if times.ndim != 1 or types.ndim != 1 or times.shape != types.shape:
        raise ValueError("times and types must be 1-D and the same shape")
    N = int(times.shape[0])
    if N == 0:
        raise ValueError("at least one event is required")
    if T <= 0.0:
        raise ValueError(f"T must be strictly positive; got {T}")
    if np.any(times < 0.0) or np.any(times >= T):
        raise ValueError("all event times must lie in [0, T)")
    if np.any(types < 0) or np.any(types >= M):
        raise ValueError("all event types must lie in {0, ..., M-1}")
    if not np.all(np.diff(times) >= 0.0):
        raise ValueError("times must be non-decreasing")

    betas = np.asarray(config.betas, dtype=float)
    R = betas.shape[0]

    # Initialise mu, alpha_all
    if config.initial_mu is not None:
        mu = np.asarray(config.initial_mu, dtype=float).copy()
        if mu.shape != (M,):
            raise ValueError(
                f"initial_mu shape {mu.shape} must equal ({M},)"
            )
        if np.any(mu <= 0.0):
            raise ValueError("initial_mu must be strictly positive")
    else:
        N_i = np.bincount(types, minlength=M).astype(np.float64)
        base_rate = np.maximum(N_i / T, 1.0e-12)
        mu = base_rate * (1.0 - config.rho_init)

    if config.initial_alpha_all is not None:
        alpha_all = np.asarray(config.initial_alpha_all, dtype=float).copy()
        if alpha_all.shape != (R, M, M):
            raise ValueError(
                f"initial_alpha_all shape {alpha_all.shape} must equal ({R}, {M}, {M})"
            )
        if np.any(alpha_all < 0.0):
            raise ValueError("initial_alpha_all must be non-negative")
    else:
        # Distribute the initial branching across R components inversely
        # proportional to beta (so slower kernels carry more amplitude).
        weights = (1.0 / betas) / np.sum(1.0 / betas)
        alpha_all = np.zeros((R, M, M), dtype=np.float64)
        for r in range(R):
            diag_val = config.rho_init * betas[r] * weights[r]
            off_val = 0.1 * diag_val
            alpha_all[r] = np.full((M, M), off_val)
            np.fill_diagonal(alpha_all[r], diag_val)

    # If projection is enabled, ensure starting point respects rho_cap.
    projection_count = 0
    if config.project_spectral:
        alpha_all, _, projected = project_alpha_to_rho_cap(
            alpha_all, betas, config.rho_cap
        )
        if projected:
            projection_count += 1

    # Pre-compute compensator G (depends only on data and betas)
    G = _compensator_G(times, types, T, M, betas)

    # EM loop
    ll_trace_list = []
    log_lik = float("-inf")
    converged = False
    converged_reason = "maxiter"
    n_iter_done = 0
    for it in range(int(config.max_iter)):
        mu_new, alpha_new, ll_new = _em_one_iteration(
            times, types, T, M, betas, mu, alpha_all, G
        )
        if not np.isfinite(ll_new):
            converged_reason = "non_finite_intensity"
            break
        # Spectral-radius projection on alpha
        if config.project_spectral:
            alpha_new, _, projected = project_alpha_to_rho_cap(
                alpha_new, betas, config.rho_cap
            )
            if projected:
                projection_count += 1
        # Floor mu to a tiny positive value for numerical safety.
        mu_new = np.maximum(mu_new, 1.0e-15)

        if config.record_trace:
            ll_trace_list.append(float(ll_new))

        # Convergence test on log-likelihood (compared to previous iteration's LL)
        if it > 0:
            denom = max(abs(ll_new), 1.0)
            rel = abs(ll_new - log_lik) / denom
            if rel < config.tol_relative_ll:
                mu = mu_new
                alpha_all = alpha_new
                log_lik = ll_new
                n_iter_done = it + 1
                converged = True
                converged_reason = "tol"
                break

        mu = mu_new
        alpha_all = alpha_new
        log_lik = ll_new
        n_iter_done = it + 1

    # Final diagnostics
    Phi = branching_matrix_sumexp(alpha_all, betas)
    rho_spec = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    return Layer1EMResult(
        mu=mu,
        alpha_all=alpha_all,
        betas=betas,
        branching_matrix=Phi,
        spectral_radius=rho_spec,
        log_likelihood=float(log_lik),
        log_likelihood_trace=np.asarray(ll_trace_list, dtype=float),
        n_iter=int(n_iter_done),
        converged=bool(converged),
        converged_reason=str(converged_reason),
        fit_window_seconds=float(T),
        n_events=int(N),
        rho_cap=float(config.rho_cap),
        projection_count=int(projection_count),
    )
