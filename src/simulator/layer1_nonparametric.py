"""
layer1_nonparametric.py — legacy one-sided kernel-recovery experiment.

This historical implementation is excluded from the final dissertation evidence.
It estimates a stationary conditional response g_ij(t), then applies a one-sided
Volterra recurrence. That combination is not the stationary Bacry-Muzy
Wiener-Hopf estimator: the latter requires negative-lag conditional responses
in a coupled integral system. The historical API names remain for compatibility.
The saved result must not be interpreted as a validated Bacry-Muzy estimate or
corrected solely by a claimed histogram-bias factor.

The legacy recurrence is based on

    g(t) = Phi(t) + ∫_0^t Phi(t - s) g(s) ds,    t > 0

discretised on a uniform grid of bin width Delta over [0, max_lag).

The forward iterative solve in matrix form is

    Phi_k = g_k - Delta * sum_{l=1}^{k-1} Phi_l g_{k-l},   k = 1, ..., K-1

with Phi_k, g_k both d x d matrices. For univariate histories this reduces
to the scalar Volterra recurrence. Non-negativity is restored at the end
by clipping to zero (post-projection); the projected kernel's L1 mass is
the branching matrix entry, the spectral radius of which is reported.

Architectural inspiration comes from Konark `src/fit/ConditionalLaw.py`
(audit class B). No external Hawkes library dependency (`tick` not used).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import json

import numpy as np


__all__ = [
    "BacryMuzyConfig",
    "BacryMuzyResult",
    "fit_bacry_muzy",
    "empirical_g",
]


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------


@dataclass
class BacryMuzyConfig:
    """Configuration for the legacy one-sided kernel-recovery experiment.

    Parameters
    ----------
    n_grid : int.
        Number of grid bins in [0, max_lag). Default 200.
    max_lag : float.
        Maximum lag in seconds at which the kernel is estimated. Default 10.
    tikhonov_lambda : float.
        Tikhonov regularisation strength on the kernel grid. Adds
        `tikhonov_lambda * Phi_k` to the right-hand side at each k.
        Default 0 (off).
    enforce_non_negative : bool.
        If True, clip negative kernel entries to zero after the solve.
        Default True.
    basis : str.
        One of "histogram", "bspline", "raised_cosine". Only "histogram"
        is implemented in this revision; the other names are accepted
        but raise NotImplementedError at solve time.
    """

    n_grid: int = 200
    max_lag: float = 10.0
    tikhonov_lambda: float = 0.0
    enforce_non_negative: bool = True
    basis: str = "histogram"

    def __post_init__(self) -> None:
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2; got {self.n_grid}")
        if self.max_lag <= 0.0:
            raise ValueError(f"max_lag must be > 0; got {self.max_lag}")
        if self.tikhonov_lambda < 0.0:
            raise ValueError(
                f"tikhonov_lambda must be >= 0; got {self.tikhonov_lambda}"
            )
        if self.basis not in {"histogram", "bspline", "raised_cosine"}:
            raise ValueError(
                f"basis must be 'histogram'/'bspline'/'raised_cosine'; got {self.basis}"
            )


@dataclass
class BacryMuzyResult:
    """Fitted non-parametric kernel and diagnostics."""

    grid_seconds: np.ndarray         # (K,) grid midpoints (s)
    bin_width: float
    phi_grid: np.ndarray             # (d, d, K) kernel values
    branching_matrix: np.ndarray     # (d, d) Phi_ij = bin_width * sum_k phi[i,j,k]
    spectral_radius: float
    asymptotic_intensities: np.ndarray  # (d,) lambda_i = N_i / T
    fit_window_seconds: float
    n_events: int
    n_events_per_type: np.ndarray    # (d,) counts
    fit_quality: dict                # diagnostics

    def to_json(self) -> str:
        payload = {
            "grid_seconds": self.grid_seconds.tolist(),
            "bin_width": float(self.bin_width),
            "phi_grid": self.phi_grid.tolist(),
            "branching_matrix": self.branching_matrix.tolist(),
            "spectral_radius": float(self.spectral_radius),
            "asymptotic_intensities": self.asymptotic_intensities.tolist(),
            "fit_window_seconds": float(self.fit_window_seconds),
            "n_events": int(self.n_events),
            "n_events_per_type": self.n_events_per_type.tolist(),
            "fit_quality": self.fit_quality,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "BacryMuzyResult":
        d = json.loads(s)
        return cls(
            grid_seconds=np.asarray(d["grid_seconds"], dtype=float),
            bin_width=float(d["bin_width"]),
            phi_grid=np.asarray(d["phi_grid"], dtype=float),
            branching_matrix=np.asarray(d["branching_matrix"], dtype=float),
            spectral_radius=float(d["spectral_radius"]),
            asymptotic_intensities=np.asarray(
                d["asymptotic_intensities"], dtype=float
            ),
            fit_window_seconds=float(d["fit_window_seconds"]),
            n_events=int(d["n_events"]),
            n_events_per_type=np.asarray(d["n_events_per_type"], dtype=int),
            fit_quality=dict(d["fit_quality"]),
        )


# ---------------------------------------------------------------------------
# Empirical conditional intensity g_ij(t)
# ---------------------------------------------------------------------------


def empirical_g(
    times: np.ndarray,
    types: np.ndarray,
    T: float,
    M: int,
    n_grid: int,
    max_lag: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the empirical second-order conditional intensity.

    g[i, j, k] = E[(dN_i/dt)(t_j + k*Delta) | event j at t_j] - lambda_i

    where Delta = max_lag / n_grid and the average is taken over all
    j-events in the window. Self-pairs (i = j) exclude the j-event itself
    at lag 0 (we use a strict `t > t_j` cut via `side='right'`).

    Returns
    -------
    g : ndarray (M, M, n_grid).
    lambdas : ndarray (M,) of asymptotic intensities lambda_i = N_i / T.
    bin_width : float, equal to max_lag / n_grid.
    """
    delta = max_lag / float(n_grid)
    times = np.asarray(times, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    N = times.shape[0]

    # Per-type sorted event times (assumes input `times` is sorted; we
    # then split per type maintaining relative order).
    by_type = [times[types == j] for j in range(M)]
    counts_per_type = np.array([t.size for t in by_type], dtype=np.int64)
    lambdas = counts_per_type.astype(np.float64) / float(T)

    g = np.zeros((M, M, n_grid), dtype=np.float64)
    for j in range(M):
        j_events = by_type[j]
        if j_events.size == 0:
            g[:, j, :] = -lambdas[:, None]
            continue
        for i in range(M):
            i_events = by_type[i]
            if i_events.size == 0:
                g[i, j, :] = -lambdas[i]
                continue
            # For each j-event find the half-open window of i-events at
            # strictly larger times. side='right' makes the lo index point
            # past any tie at t_j (correct for self-pairs).
            los = np.searchsorted(i_events, j_events, side="right")
            his = np.searchsorted(i_events, j_events + max_lag, side="right")
            counts = np.zeros(n_grid, dtype=np.float64)
            for k in range(j_events.size):
                if his[k] > los[k]:
                    lags = i_events[los[k]:his[k]] - j_events[k]
                    bin_idx = (lags / delta).astype(np.int64)
                    bin_idx = bin_idx[bin_idx < n_grid]
                    if bin_idx.size > 0:
                        np.add.at(counts, bin_idx, 1.0)
            # Convert counts to rate, subtract asymptotic mean.
            g[i, j, :] = counts / (j_events.size * delta) - lambdas[i]
    return g, lambdas, delta


# ---------------------------------------------------------------------------
# Legacy one-sided Volterra recurrence (historical function name retained)
# ---------------------------------------------------------------------------


def _wiener_hopf_solve(
    g: np.ndarray, delta: float, tikhonov_lambda: float = 0.0
) -> np.ndarray:
    """Evaluate the legacy one-sided Volterra recurrence for Phi_k.

    From the continuous equation g(t) = phi(t) + integral_0^t phi(t-s) g(s) ds,
    the left-endpoint discretisation on uniform grid {kDelta} gives

        g_k = phi_k + Delta * sum_{l=0}^{k-1} phi_{k-l} g_l    for k >= 1,
        g_0 = phi_0                                              for k = 0.

    Pulling the l=0 term (which involves phi_k itself) out of the sum,

        g_k = phi_k * (I + Delta * g_0) + Delta * sum_{l=1}^{k-1} phi_{k-l} g_l,

    so the forward update is

        phi_k = (g_k - Delta * sum_{l=1}^{k-1} phi_{k-l} g_l) (I + Delta * g_0)^{-1}.

    The factor (I + Delta * g_0)^{-1} belongs to this discrete recurrence;
    it does not supply the missing negative-lag stationary terms.
    Tikhonov regularisation lambda >= 0 is applied as an extra
    factor 1 / (1 + lambda) at each step.
    """
    M, M2, K = g.shape
    assert M == M2, f"g must be (M, M, K); got {g.shape}"
    g0 = g[..., 0]
    I = np.eye(M, dtype=np.float64)
    A = I + delta * g0  # (M, M); inversion below requires a nonsingular matrix
    A_inv = np.linalg.inv(A)
    Phi = np.zeros_like(g)
    # Boundary: phi_0 = g_0 (no integral term in continuous Volterra at t=0).
    Phi[..., 0] = g[..., 0]
    if tikhonov_lambda > 0.0:
        Phi[..., 0] = Phi[..., 0] / (1.0 + tikhonov_lambda)
    for k in range(1, K):
        # Sum_{l=1}^{k-1} Phi_{k-l} @ g_l   (matrix product over the inner d)
        update = np.zeros((M, M), dtype=np.float64)
        for l in range(1, k):
            update += Phi[..., k - l] @ g[..., l]
        rhs = g[..., k] - delta * update
        Phi_k = rhs @ A_inv
        if tikhonov_lambda > 0.0:
            Phi_k = Phi_k / (1.0 + tikhonov_lambda)
        Phi[..., k] = Phi_k
    return Phi


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------


def fit_bacry_muzy(
    times: np.ndarray,
    types: np.ndarray,
    T: float,
    M: int,
    config: BacryMuzyConfig,
) -> BacryMuzyResult:
    """Run the legacy one-sided experiment; excluded from submission evidence.

    Parameters
    ----------
    times : ndarray (N,) of event times in [0, T), assumed sorted.
    types : ndarray (N,) of event types in {0, ..., M-1}.
    T : float, fit window length in seconds.
    M : int, dimension.
    config : BacryMuzyConfig.

    Returns
    -------
    BacryMuzyResult.
    """
    if config.basis != "histogram":
        raise NotImplementedError(
            f"basis='{config.basis}' is reserved for a future revision; "
            "only 'histogram' is implemented in this milestone"
        )
    times = np.ascontiguousarray(times, dtype=np.float64)
    types = np.ascontiguousarray(types, dtype=np.int64)
    if times.shape != types.shape or times.ndim != 1:
        raise ValueError("times and types must be 1-D arrays of equal length")
    if times.size == 0:
        raise ValueError("at least one event is required")
    if T <= 0.0:
        raise ValueError(f"T must be > 0; got {T}")
    if np.any(types < 0) or np.any(types >= M):
        raise ValueError("all event types must lie in {0, ..., M-1}")
    if not np.all(np.diff(times) >= 0.0):
        raise ValueError("times must be non-decreasing")

    g, lambdas, bin_width = empirical_g(
        times, types, T, M, config.n_grid, config.max_lag
    )

    Phi_grid = _wiener_hopf_solve(g, bin_width, config.tikhonov_lambda)

    # Diagnostic counts BEFORE non-negative projection
    n_neg_before = int(np.sum(Phi_grid < 0.0))
    min_value_before = float(np.min(Phi_grid))

    if config.enforce_non_negative:
        Phi_grid = np.clip(Phi_grid, 0.0, None)

    # Branching matrix: integral over lag of phi[i, j, t] approx bin_width * sum_k
    branching = bin_width * np.sum(Phi_grid, axis=2)
    rho_spec = float(np.max(np.abs(np.linalg.eigvals(branching))))

    # Diagnostics
    grid_midpoints = (np.arange(config.n_grid) + 0.5) * bin_width
    fit_quality = {
        "n_grid": int(config.n_grid),
        "max_lag_seconds": float(config.max_lag),
        "bin_width_seconds": float(bin_width),
        "n_negative_grid_values_before_clip": n_neg_before,
        "min_grid_value_before_clip": min_value_before,
        "tikhonov_lambda": float(config.tikhonov_lambda),
        "kernel_basis": config.basis,
    }

    counts_per_type = np.bincount(types, minlength=M).astype(np.int64)
    return BacryMuzyResult(
        grid_seconds=grid_midpoints,
        bin_width=float(bin_width),
        phi_grid=Phi_grid,
        branching_matrix=branching,
        spectral_radius=rho_spec,
        asymptotic_intensities=lambdas,
        fit_window_seconds=float(T),
        n_events=int(times.size),
        n_events_per_type=counts_per_type,
        fit_quality=fit_quality,
    )
