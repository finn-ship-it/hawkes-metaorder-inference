"""
layer1_konark_cls.py — Layer-1 Konark CLS-LogLin long-memory non-parametric
Hawkes estimator (Kirchner-2015 conditional least squares on a log-linear
lag grid, OSQP-constrained quadratic program).

This module implements the long-memory non-parametric Layer-1 estimator
described in restored_dissertation_pack/02b_long_memory_nonparametric_konark_cls.md
and modelled on Konark Jain's
`Konarks-github repo/src/fit/ConditionalLeastSquaresLogLin.py`. It is a
sibling (not a replacement) of:

- the parametric fixed-beta-grid EM in `layer1_em.py` (prompt 01);
- the legacy one-sided experiment in `layer1_nonparametric.py`, excluded
  from the final dissertation evidence;

and complements the parametric `POWERLAW_CUTOFF` simulator kernel in
`kernels.py` (prompt 04).

Mathematical model
------------------
For a stationary multivariate Hawkes process N_i(t) with kernel matrix
phi_ij(u), the conditional intensity is

    lambda_i(t) = mu_i + sum_j integral phi_ij(u) dN_j(t - u).

We discretise the kernel on a log-linear lag grid

    lag_grid = concat(linspace(0, min_lag, K)[:-1],  exp(linspace(log(min_lag), log(max_lag), K)))

with K = num_datapoints. Default num_datapoints = 10 yields 9 + 10 = 19
lag points and therefore **18 non-overlapping lag bins** (the convention
used by KonarkCLSResult.lag_grid: lag_grid[b] is the *left edge* of bin b
and lag_grid[b+1] is the right edge; bin width = diff).

A piecewise-constant kernel approximation gives

    phi_ij(u) = theta_{ij, b}    for u in [lag_grid[b], lag_grid[b+1]),

so the integrated lagged-event count over bin b for source type j is

    X_{t, j, b} = N_j([t - lag_grid[b+1], t - lag_grid[b]))

at each fine bin t (width Delta_fine). The expected event count
Y_{t, i} of type i in fine bin [t, t + Delta_fine) is

    E[Y_{t, i}]  =  Delta_fine * mu_i  +  sum_{j, b} (Delta_fine * theta_{ij, b}) * X_{t, j, b}.

Regressing Y on (1, X) via OLS over **all** fine bins yields intercept
alpha_i = Delta_fine * mu_i and coefficients
beta_{ij, b} = Delta_fine * theta_{ij, b}. The integrated kernel mass per
pair is

    Phi_ij = sum_b theta_{ij, b} * bin_width_b
           = sum_b beta_{ij, b} * (bin_width_b / Delta_fine).

This module iterates over **all** fine bins (active and empty) and
streams the (X^T X, X^T Y) accumulators in chunks. This is a deviation
from Konark's `transformData` which iterates over active bins only;
Konark's choice introduces an active-bins-only conditioning bias that
inflates the intercept and biases the kernel coefficients upward.
For FX top-of-book streams the active-bins fraction is small, so the
bias is large; the all-fine-bins iteration here is the unbiased
formulation.

The OSQP problem (per destination type i) is

    minimize   (1/2) ||X beta_i - Y_i||^2
    subject to sum_b beta_{i, j, b} * (bin_width_b / Delta_fine) <= subcritical_cap   for each j
               beta_{i, j, b} >= 0,  alpha_i >= 0
               (optional NPHC sign bound: beta_{i, j, b} <= 0 forces 0)

Per-pair subcriticality is the per-row analogue of the spectral-radius
constraint and matches Konark's per-(i, j) row-sum constraint structure.

Out of scope
------------
- Special-date calendars (`MEXP`, `FOMC`, `MEND`, etc.).
- 12-event LOB schema (this module is FX 4-type).
- Multi-day NPHC pre-fit. The `nphc_graph_signs` field accepts a
  precomputed (M, M, B) array of {-1, 0, +1}; the estimator does not
  compute NPHC itself.
- Equity-style time-of-day / spread-power-law corrections.

Reference: Kirchner 2015, "An estimation procedure for the Hawkes process".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import json

import numpy as np
from scipy import sparse


__all__ = [
    "KonarkCLSConfig",
    "KonarkCLSResult",
    "fit_konark_cls",
    "build_lag_grid",
]


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------


@dataclass
class KonarkCLSConfig:
    """Configuration for the Konark CLS-LogLin long-memory estimator.

    Defaults are FX-tuned single-day defaults; equity defaults from
    Konark's pipeline (`fitTODParams`, `fitConditionalInSpread`, etc.)
    are out of scope for this module.

    Parameters
    ----------
    min_lag : float
        Lower bound of the log-spaced section of the lag grid (s).
        Default 1e-3 s.
    max_lag : float
        Upper bound of the log-spaced section of the lag grid (s).
        Default 500.0 s. A 500 s upper bound covers FX top-of-book
        events with the long-memory regime that fixed-beta-grid kernels
        cannot represent.
    num_datapoints : int
        Number of points in each of the linear and log-spaced segments
        of the grid. Default 10. With this default the concatenated
        grid has 9 + 10 = 19 points and **18 non-overlapping lag bins**.
        See module docstring for the bin convention used downstream.
    fine_bin_seconds : float
        The fine-bin width Delta_fine used as the regression discretisation
        unit (NOT the same as min_lag; this module decouples the two so
        FX-low-rate event streams can use a coarser regression grid).
        Default 0.05 s. The regression iterates over T / fine_bin_seconds
        bins, streamed in chunks for memory.
    solver : str
        One of "osqp" (constrained QP, primary path), "ridge"
        (unconstrained Tikhonov-regularised LS, fallback when OSQP is
        infeasible), "lstsq" (vanilla unconstrained LS, used by the
        parity test).
    subcritical_cap : float
        Per-pair subcriticality cap on the integrated kernel mass
        Phi_ij = sum_b theta_{ij, b} * bin_width_b. Default 0.999. The
        OSQP problem enforces sum_b beta_{i, j, b} * bin_width_b
        / Delta_fine <= subcritical_cap.
    ridge_alpha : float
        Tikhonov regularisation strength on the kernel coefficients.
        Default 1e-3. Only consulted when solver == "ridge".
    nphc_graph_signs : Optional ndarray of shape (M, M, n_bins)
        Optional sign hints from a precomputed NPHC pre-fit. Values
        in {-1, 0, +1}: -1 forces beta_{ij, b} <= 0 (effectively
        zero, since the default constraint is beta >= 0); +1 leaves
        the default beta >= 0 constraint; 0 (default) is also
        unconstrained. None means no sign hints are applied.
    osqp_eps_abs, osqp_eps_rel : float
        OSQP tolerance settings. Default 1e-6 each.
    osqp_polish : bool
        Enable the OSQP polish step (default True). Polish-status is
        reported but is not gated on (the residual-based acceptance
        criteria in the spec test for primal/dual feasibility).
    osqp_max_iter : int
        OSQP iteration cap. Default 100000.
    chunk_size : int
        Number of fine bins processed per chunk in the streaming X^T X
        accumulation. Default 50000.
    """

    min_lag: float = 1.0e-3
    max_lag: float = 500.0
    num_datapoints: int = 10
    fine_bin_seconds: float = 0.05
    solver: str = "osqp"
    subcritical_cap: float = 0.999
    ridge_alpha: float = 1.0e-3
    nphc_graph_signs: Optional[np.ndarray] = None
    osqp_eps_abs: float = 1.0e-6
    osqp_eps_rel: float = 1.0e-6
    osqp_polish: bool = True
    osqp_max_iter: int = 100_000
    chunk_size: int = 50_000

    def __post_init__(self) -> None:
        if self.min_lag <= 0.0:
            raise ValueError(f"min_lag must be > 0; got {self.min_lag}")
        if self.max_lag <= self.min_lag:
            raise ValueError(
                f"max_lag must be > min_lag; got {self.max_lag} <= {self.min_lag}"
            )
        if self.num_datapoints < 3:
            raise ValueError(
                f"num_datapoints must be >= 3; got {self.num_datapoints}"
            )
        if self.fine_bin_seconds <= 0.0:
            raise ValueError(
                f"fine_bin_seconds must be > 0; got {self.fine_bin_seconds}"
            )
        if self.solver not in {"osqp", "ridge", "lstsq"}:
            raise ValueError(
                f"solver must be 'osqp', 'ridge' or 'lstsq'; got {self.solver!r}"
            )
        if not (0.0 < self.subcritical_cap < 1.0):
            raise ValueError(
                f"subcritical_cap must lie in (0, 1); got {self.subcritical_cap}"
            )
        if self.ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must be >= 0; got {self.ridge_alpha}")
        if self.osqp_eps_abs <= 0.0 or self.osqp_eps_rel <= 0.0:
            raise ValueError("osqp_eps_abs and osqp_eps_rel must be > 0")
        if self.osqp_max_iter < 100:
            raise ValueError(f"osqp_max_iter must be >= 100; got {self.osqp_max_iter}")
        if self.chunk_size < 1000:
            raise ValueError(f"chunk_size must be >= 1000; got {self.chunk_size}")


@dataclass
class KonarkCLSResult:
    """Output of the Konark CLS-LogLin estimator."""

    theta: np.ndarray                 # (M, M, n_bins) kernel values per (i, j, b)
    intercept: np.ndarray             # (M,) baseline rates (events / second)
    lag_grid: np.ndarray              # (n_bins + 1,) lag-bin edges (s)
    bin_widths: np.ndarray            # (n_bins,) bin widths (s)
    delta_fine: float                 # fine-bin width used for binning (s)
    branching_matrix: np.ndarray      # (M, M) Phi_ij = sum_b theta * bin_width
    spectral_radius: float
    solver: str
    osqp_status: str
    osqp_iter: int
    osqp_pri_res: float
    osqp_dua_res: float
    osqp_polish_status: Optional[str]
    n_events: int
    n_events_per_type: np.ndarray
    fit_window_seconds: float
    n_bins: int
    n_fine_bins: int
    fit_quality: dict

    def to_json(self) -> str:
        payload = {
            "theta": self.theta.tolist(),
            "intercept": self.intercept.tolist(),
            "lag_grid": self.lag_grid.tolist(),
            "bin_widths": self.bin_widths.tolist(),
            "delta_fine": float(self.delta_fine),
            "branching_matrix": self.branching_matrix.tolist(),
            "spectral_radius": float(self.spectral_radius),
            "solver": self.solver,
            "osqp_status": self.osqp_status,
            "osqp_iter": int(self.osqp_iter),
            "osqp_pri_res": float(self.osqp_pri_res),
            "osqp_dua_res": float(self.osqp_dua_res),
            "osqp_polish_status": self.osqp_polish_status,
            "n_events": int(self.n_events),
            "n_events_per_type": self.n_events_per_type.tolist(),
            "fit_window_seconds": float(self.fit_window_seconds),
            "n_bins": int(self.n_bins),
            "n_fine_bins": int(self.n_fine_bins),
            "fit_quality": self.fit_quality,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "KonarkCLSResult":
        d = json.loads(s)
        return cls(
            theta=np.asarray(d["theta"], dtype=np.float64),
            intercept=np.asarray(d["intercept"], dtype=np.float64),
            lag_grid=np.asarray(d["lag_grid"], dtype=np.float64),
            bin_widths=np.asarray(d["bin_widths"], dtype=np.float64),
            delta_fine=float(d["delta_fine"]),
            branching_matrix=np.asarray(d["branching_matrix"], dtype=np.float64),
            spectral_radius=float(d["spectral_radius"]),
            solver=str(d["solver"]),
            osqp_status=str(d["osqp_status"]),
            osqp_iter=int(d["osqp_iter"]),
            osqp_pri_res=float(d["osqp_pri_res"]),
            osqp_dua_res=float(d["osqp_dua_res"]),
            osqp_polish_status=d["osqp_polish_status"],
            n_events=int(d["n_events"]),
            n_events_per_type=np.asarray(d["n_events_per_type"], dtype=np.int64),
            fit_window_seconds=float(d["fit_window_seconds"]),
            n_bins=int(d["n_bins"]),
            n_fine_bins=int(d.get("n_fine_bins", 0)),
            fit_quality=dict(d["fit_quality"]),
        )


# ---------------------------------------------------------------------------
# Lag-grid construction
# ---------------------------------------------------------------------------


def build_lag_grid(
    min_lag: float, max_lag: float, num_datapoints: int
) -> np.ndarray:
    """Construct the log-linear lag grid as Konark's
    `np.append(timegridLin[:-1], timegridLog)`.

    With num_datapoints = K, the resulting grid has 2 K - 1 points
    (K - 1 from the linear segment + K from the log segment) and
    therefore 2 K - 2 non-overlapping lag bins.
    """
    timegrid_lin = np.linspace(0.0, min_lag, num_datapoints)
    timegrid_log = np.exp(
        np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints)
    )
    grid = np.append(timegrid_lin[:-1], timegrid_log)
    # Guarantee strict monotonicity (rare floating-point ties at the seam).
    grid = np.maximum.accumulate(grid)
    return grid


# ---------------------------------------------------------------------------
# Streaming P, q accumulation over fine bins
# ---------------------------------------------------------------------------


def _accumulate_pq(
    times_per_type, T: float, M: int,
    lag_grid: np.ndarray, delta_fine: float, chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Stream X^T X (= P_acc) and X^T Y (= q_acc) over all fine bins of
    width delta_fine in [0, T).

    Returns (P_acc, q_acc, n_fine_bins).

    X has shape (n_fine, 1 + M * n_bins) with X[t, 0] = 1 (intercept
    column) and X[t, 1 + j * n_bins + b] = number of type-j events with
    timestamp in [t * delta_fine - lag_grid[b+1], t * delta_fine - lag_grid[b]).

    Y has shape (n_fine, M) with Y[t, i] = number of type-i events in
    [t * delta_fine, (t + 1) * delta_fine).

    Boundary: lag windows extending below 0 are simply truncated at 0
    (no past events exist for t < 0). This matches Konark's transformData
    implicit boundary by virtue of np.searchsorted on a sorted array.
    """
    n_bins = lag_grid.size - 1
    n_fine_bins = int(np.ceil(T / delta_fine))
    n_params = 1 + M * n_bins
    P_acc = np.zeros((n_params, n_params), dtype=np.float64)
    q_acc = np.zeros((n_params, M), dtype=np.float64)

    # Pre-sort per-type times once
    tt_per_type = [np.ascontiguousarray(times_per_type[j]) for j in range(M)]

    for chunk_lo in range(0, n_fine_bins, chunk_size):
        chunk_hi = min(chunk_lo + chunk_size, n_fine_bins)
        n_chunk = chunk_hi - chunk_lo
        # left edges of fine bins in the chunk
        t_chunk = (np.arange(chunk_lo, chunk_hi, dtype=np.float64)) * delta_fine

        # Build X_chunk
        X_chunk = np.empty((n_chunk, n_params), dtype=np.float64)
        X_chunk[:, 0] = 1.0
        for j in range(M):
            tt = tt_per_type[j]
            for b in range(n_bins):
                lag_low = lag_grid[b]
                lag_high = lag_grid[b + 1]
                # X[t, j, b] = #{m : type_m = j, t_m in [t - lag_high, t - lag_low)}
                hi = np.searchsorted(tt, t_chunk - lag_low, side="left")
                lo = np.searchsorted(tt, t_chunk - lag_high, side="left")
                X_chunk[:, 1 + j * n_bins + b] = (hi - lo).astype(np.float64)

        # Build Y_chunk: Y[t, i] = count of type-i events in [t, t + delta_fine)
        Y_chunk = np.empty((n_chunk, M), dtype=np.float64)
        for i in range(M):
            tt_i = tt_per_type[i]
            hi_e = np.searchsorted(tt_i, t_chunk + delta_fine, side="left")
            lo_e = np.searchsorted(tt_i, t_chunk, side="left")
            Y_chunk[:, i] = (hi_e - lo_e).astype(np.float64)

        P_acc += X_chunk.T @ X_chunk
        q_acc += X_chunk.T @ Y_chunk

    return P_acc, q_acc, n_fine_bins


# ---------------------------------------------------------------------------
# OSQP solve (per destination type i)
# ---------------------------------------------------------------------------


def _solve_osqp_per_target(
    P: sparse.csc_matrix, q: np.ndarray, A: sparse.csc_matrix,
    l: np.ndarray, u: np.ndarray, cfg: KonarkCLSConfig,
) -> dict:
    """Solve the OSQP QP for one destination type."""
    import osqp

    prob = osqp.OSQP()
    prob.setup(
        P, q, A, l, u,
        eps_abs=cfg.osqp_eps_abs,
        eps_rel=cfg.osqp_eps_rel,
        polish=cfg.osqp_polish,
        max_iter=cfg.osqp_max_iter,
        verbose=False,
    )
    res = prob.solve()
    info = res.info
    polish_status = getattr(info, "status_polish", None)
    return {
        "x": np.asarray(res.x, dtype=np.float64),
        "status": str(getattr(info, "status", "unknown")),
        "n_iter": int(getattr(info, "iter", 0)),
        "pri_res": float(getattr(info, "pri_res", float("nan"))),
        "dua_res": float(getattr(info, "dua_res", float("nan"))),
        "polish_status": str(polish_status) if polish_status is not None else None,
    }


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------


def fit_konark_cls(
    times: np.ndarray, types: np.ndarray, T: float, M: int,
    config: KonarkCLSConfig,
) -> KonarkCLSResult:
    """Fit the Konark CLS-LogLin estimator.

    Parameters
    ----------
    times : ndarray (N,) of strictly non-decreasing event times in [0, T).
    types : ndarray (N,) of int in {0, ..., M-1}.
    T : float, fit window length in seconds.
    M : int, dimension (number of event types).
    config : KonarkCLSConfig.
    """
    times = np.ascontiguousarray(times, dtype=np.float64)
    types = np.ascontiguousarray(types, dtype=np.int64)
    if times.ndim != 1 or types.ndim != 1 or times.shape != types.shape:
        raise ValueError("times and types must be 1-D arrays of equal length")
    N = int(times.shape[0])
    if N < M + 1:
        raise ValueError(
            f"need at least M + 1 events; got N = {N} for M = {M}"
        )
    if T <= 0.0:
        raise ValueError(f"T must be > 0; got {T}")
    if np.any(times < 0.0) or np.any(times >= T):
        raise ValueError("all event times must lie in [0, T)")
    if np.any(types < 0) or np.any(types >= M):
        raise ValueError("all event types must lie in {0, ..., M-1}")
    if not np.all(np.diff(times) >= 0.0):
        raise ValueError("times must be non-decreasing")
    if config.nphc_graph_signs is not None:
        nphc = np.asarray(config.nphc_graph_signs)
        if nphc.dtype not in (np.int8, np.int16, np.int32, np.int64,
                              np.float32, np.float64):
            raise ValueError("nphc_graph_signs must be numeric")

    lag_grid = build_lag_grid(config.min_lag, config.max_lag, config.num_datapoints)
    n_bins = lag_grid.size - 1
    bin_widths = np.diff(lag_grid)
    delta_fine = float(config.fine_bin_seconds)

    if config.nphc_graph_signs is not None:
        nphc = np.asarray(config.nphc_graph_signs)
        if nphc.shape != (M, M, n_bins):
            raise ValueError(
                f"nphc_graph_signs shape {nphc.shape} != ({M}, {M}, {n_bins})"
            )

    # Stream P, q over all fine bins (all-fine-bins iteration; see module docstring)
    times_per_type = [np.sort(times[types == j]) for j in range(M)]
    P_dense, q_full, n_fine_bins = _accumulate_pq(
        times_per_type, T, M, lag_grid, delta_fine, config.chunk_size
    )
    P_csc = sparse.csc_matrix(P_dense)
    n_params = 1 + M * n_bins  # = P_dense.shape[0]

    # Per-(j) subcriticality constraint vector (per destination type i).
    # Coefficient on beta_{i, j, b}: bin_widths[b] / delta_fine.
    constr_subcrit = np.zeros((M, n_params), dtype=np.float64)
    for j in range(M):
        idx = 1 + j * n_bins + np.arange(n_bins)
        constr_subcrit[j, idx] = bin_widths / delta_fine

    inf_val = np.inf
    theta_out = np.zeros((M, M, n_bins), dtype=np.float64)
    alpha_out = np.zeros(M, dtype=np.float64)
    osqp_per_target = []

    for i_target in range(M):
        q_i = -q_full[:, i_target]

        # Constraint matrix: [subcrit (M rows); identity on params]
        A_subcrit = constr_subcrit
        A_identity = np.eye(n_params, dtype=np.float64)
        A_dense = np.vstack([A_subcrit, A_identity])
        A_csc = sparse.csc_matrix(A_dense)

        l_subcrit = np.full(M, -inf_val)
        u_subcrit = np.full(M, config.subcritical_cap)
        l_box = np.zeros(n_params)
        u_box = np.full(n_params, inf_val)
        if config.nphc_graph_signs is not None:
            nphc = np.asarray(config.nphc_graph_signs)
            for j in range(M):
                for b in range(n_bins):
                    if nphc[i_target, j, b] == -1:
                        u_box[1 + j * n_bins + b] = 0.0

        l = np.concatenate([l_subcrit, l_box])
        u = np.concatenate([u_subcrit, u_box])

        if config.solver == "osqp":
            sol = _solve_osqp_per_target(P_csc, q_i, A_csc, l, u, config)
            beta_full = sol["x"]
        elif config.solver == "ridge":
            P_aug = P_dense + config.ridge_alpha * np.eye(n_params)
            beta_full = np.linalg.solve(P_aug, q_full[:, i_target])
            sol = {
                "x": beta_full, "status": "ridge", "n_iter": 0,
                "pri_res": 0.0, "dua_res": 0.0, "polish_status": None,
            }
        else:  # lstsq
            beta_full = np.linalg.solve(P_dense, q_full[:, i_target])
            sol = {
                "x": beta_full, "status": "lstsq", "n_iter": 0,
                "pri_res": 0.0, "dua_res": 0.0, "polish_status": None,
            }

        alpha_i = float(beta_full[0])
        beta_kernel_i = beta_full[1:].reshape(M, n_bins)
        # Numerical safety: clip tiny negative values from OSQP polish
        alpha_i = max(alpha_i, 0.0)
        beta_kernel_i = np.maximum(beta_kernel_i, 0.0)

        # Convert: beta = Delta_fine * theta, alpha = Delta_fine * mu
        theta_i = beta_kernel_i / delta_fine
        alpha_out[i_target] = alpha_i / delta_fine  # mu_i (events per second)
        theta_out[i_target] = theta_i
        osqp_per_target.append(sol)

    branching = np.sum(theta_out * bin_widths[None, None, :], axis=2)
    rho_spec = float(np.max(np.abs(np.linalg.eigvals(branching))))
    margins = config.subcritical_cap - branching

    fit_quality = {
        "n_bins": int(n_bins),
        "n_fine_bins": int(n_fine_bins),
        "min_lag_seconds": float(config.min_lag),
        "max_lag_seconds": float(config.max_lag),
        "delta_fine_seconds": float(delta_fine),
        "subcriticality_margin_min": float(np.min(margins)),
        "subcriticality_margin_max": float(np.max(margins)),
        "subcriticality_margin_per_ij": margins.tolist(),
        "ridge_alpha": float(config.ridge_alpha) if config.solver == "ridge" else None,
        "nphc_sign_bounds_active": int(
            np.sum(np.asarray(config.nphc_graph_signs) == -1)
            if config.nphc_graph_signs is not None
            else 0
        ),
    }

    if config.solver == "osqp":
        statuses = [s["status"] for s in osqp_per_target]
        n_iter_max = max(s["n_iter"] for s in osqp_per_target)
        # Use float-safe max (NaNs from OSQP polish handled)
        pri_vals = [s["pri_res"] if np.isfinite(s["pri_res"]) else 0.0
                    for s in osqp_per_target]
        dua_vals = [s["dua_res"] if np.isfinite(s["dua_res"]) else 0.0
                    for s in osqp_per_target]
        pri_res_max = max(pri_vals)
        dua_res_max = max(dua_vals)
        polish_states = [s["polish_status"] for s in osqp_per_target]
        if all(s == "solved" for s in statuses):
            agg_status = "solved"
        elif all(s in ("solved", "solved_inaccurate") for s in statuses):
            agg_status = "solved_inaccurate"
        else:
            agg_status = ";".join(statuses)
        agg_polish = (
            polish_states[0]
            if all(p == polish_states[0] for p in polish_states)
            else ";".join(str(p) for p in polish_states)
        )
        fit_quality["osqp_per_target_pri_res"] = [s["pri_res"] for s in osqp_per_target]
        fit_quality["osqp_per_target_dua_res"] = [s["dua_res"] for s in osqp_per_target]
    else:
        agg_status = osqp_per_target[0]["status"]
        n_iter_max = 0
        pri_res_max = 0.0
        dua_res_max = 0.0
        agg_polish = None

    counts_per_type = np.bincount(types, minlength=M).astype(np.int64)
    return KonarkCLSResult(
        theta=theta_out,
        intercept=alpha_out,
        lag_grid=lag_grid,
        bin_widths=bin_widths,
        delta_fine=delta_fine,
        branching_matrix=branching,
        spectral_radius=rho_spec,
        solver=config.solver,
        osqp_status=agg_status,
        osqp_iter=int(n_iter_max),
        osqp_pri_res=float(pri_res_max),
        osqp_dua_res=float(dua_res_max),
        osqp_polish_status=agg_polish,
        n_events=N,
        n_events_per_type=counts_per_type,
        fit_window_seconds=float(T),
        n_bins=int(n_bins),
        n_fine_bins=int(n_fine_bins),
        fit_quality=fit_quality,
    )
