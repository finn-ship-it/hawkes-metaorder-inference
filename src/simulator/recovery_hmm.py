"""
recovery_hmm.py — Layer-2 HMM regime / meta-order recovery with posterior.

Implements a Hidden Markov Model with Baum-Welch training on
window-feature observations of the FX top-of-book event stream. The
HMM produces (i) a smoothed posterior p(z_t | O_{0:T}) per window,
(ii) Viterbi MAP labels, (iii) the learned transition matrix and
emission parameters, and (iv) the marginal log-likelihood and
convergence diagnostics.

The threshold detector in `recovery.py` provides the labelled baseline
for comparison.

Mathematical model
------------------
States  z_t in {0, ..., S-1} (S=2 by default: 0 = calm, 1 = active).
Observations O_t = (rate_t, imbalance_t), one feature vector per window.
Emissions: independent Gaussian on each component, with state-specific
mean and variance. Transition matrix A is row-stochastic.

Forward-backward and Baum-Welch updates run in log-space for numerical
stability. Convergence is declared when the relative log-likelihood
improvement falls below `tol_relative_ll`.

Reference: Rabiner, "A tutorial on hidden Markov models and selected
applications in speech recognition", Proc. IEEE 77(2):257-286, 1989.
The implementation here uses the standard scaled / log-space
forward-backward recursion (no scaling-factor approach is needed when
working entirely in log-space).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import json

import numpy as np

from .recovery import WindowFeatures


__all__ = [
    "HMMRecoveryConfig",
    "HMMRecoveryResult",
    "recover_regimes_hmm",
    "viterbi_decode",
]


_NEG_INF = -1.0e300


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------


@dataclass
class HMMRecoveryConfig:
    """Configuration for the Baum-Welch HMM regime recovery.

    Parameters
    ----------
    n_states : int, default 2.
        Number of latent states.
    emission_features : tuple of str.
        Names of the WindowFeatures fields used as emission components.
        Default: ("recent_event_rate", "directional_imbalance"). Each
        component is modelled as an independent Gaussian.
    max_iter : int, default 100.
        Maximum Baum-Welch iterations.
    tol_relative_ll : float, default 1e-6.
        Convergence tolerance on the relative log-likelihood improvement.
    sigma_floor : float, default 1e-3.
        Lower floor on the per-state, per-feature variance during the
        M-step. Prevents the EM from collapsing a state's variance to
        zero on a single observation.
    init_method : str, default "kmeans1d".
        "kmeans1d" splits observations by quantile on the first feature
        to initialise per-state means; "uniform" initialises everything
        to the global mean / variance.
    seed : int, default 0.
        Optional random seed used by the kmeans1d-style initialiser
        (nudges initial means by a tiny amount per state to break ties).
    """

    n_states: int = 2
    emission_features: Tuple[str, ...] = (
        "recent_event_rate",
        "directional_imbalance",
    )
    max_iter: int = 100
    tol_relative_ll: float = 1.0e-6
    sigma_floor: float = 1.0e-3
    init_method: str = "kmeans1d"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError(f"n_states must be >= 2; got {self.n_states}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if self.tol_relative_ll <= 0.0:
            raise ValueError(
                f"tol_relative_ll must be > 0; got {self.tol_relative_ll}"
            )
        if self.sigma_floor <= 0.0:
            raise ValueError(f"sigma_floor must be > 0; got {self.sigma_floor}")
        if self.init_method not in {"kmeans1d", "uniform"}:
            raise ValueError(
                f"init_method must be 'kmeans1d' or 'uniform'; got {self.init_method!r}"
            )
        if not self.emission_features:
            raise ValueError("emission_features must be non-empty")


@dataclass
class HMMRecoveryResult:
    """Output of `recover_regimes_hmm`."""

    posterior: np.ndarray        # (T, S) p(z_t = s | O_{0:T})
    viterbi_labels: np.ndarray   # (T,) MAP labels
    transition_matrix: np.ndarray  # (S, S)
    initial_state_distribution: np.ndarray  # (S,)
    emission_means: np.ndarray   # (S, F) per-state, per-feature mean
    emission_stds: np.ndarray    # (S, F) per-state, per-feature std
    feature_names: Tuple[str, ...]
    log_likelihood: float
    log_likelihood_trace: np.ndarray  # (n_iter,)
    n_iter: int
    converged: bool
    converged_reason: str
    window_starts: np.ndarray    # (T,) seconds; left edges of windows
    n_windows: int

    def to_json(self) -> str:
        payload = {
            "posterior": self.posterior.tolist(),
            "viterbi_labels": self.viterbi_labels.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "initial_state_distribution": self.initial_state_distribution.tolist(),
            "emission_means": self.emission_means.tolist(),
            "emission_stds": self.emission_stds.tolist(),
            "feature_names": list(self.feature_names),
            "log_likelihood": float(self.log_likelihood),
            "log_likelihood_trace": self.log_likelihood_trace.tolist(),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "converged_reason": str(self.converged_reason),
            "window_starts": self.window_starts.tolist(),
            "n_windows": int(self.n_windows),
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "HMMRecoveryResult":
        d = json.loads(s)
        return cls(
            posterior=np.asarray(d["posterior"], dtype=float),
            viterbi_labels=np.asarray(d["viterbi_labels"], dtype=int),
            transition_matrix=np.asarray(d["transition_matrix"], dtype=float),
            initial_state_distribution=np.asarray(
                d["initial_state_distribution"], dtype=float
            ),
            emission_means=np.asarray(d["emission_means"], dtype=float),
            emission_stds=np.asarray(d["emission_stds"], dtype=float),
            feature_names=tuple(d["feature_names"]),
            log_likelihood=float(d["log_likelihood"]),
            log_likelihood_trace=np.asarray(d["log_likelihood_trace"], dtype=float),
            n_iter=int(d["n_iter"]),
            converged=bool(d["converged"]),
            converged_reason=str(d["converged_reason"]),
            window_starts=np.asarray(d["window_starts"], dtype=float),
            n_windows=int(d["n_windows"]),
        )


# ---------------------------------------------------------------------------
# Log-space Gaussian emission likelihoods
# ---------------------------------------------------------------------------


_LOG_2PI = float(np.log(2.0 * np.pi))


def _log_gaussian_emission(
    O: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """log p(O_t | z_t = s) for independent Gaussian features.

    Parameters
    ----------
    O : (T, F)
    means : (S, F)
    stds : (S, F)

    Returns
    -------
    log_B : (T, S)
    """
    var = stds**2
    # diff: (T, S, F)
    diff = O[:, None, :] - means[None, :, :]
    # log-density: -0.5 * sum_f [log(2 pi var_sf) + (diff_tsf)^2 / var_sf]
    log_term = -0.5 * (np.log(var)[None, :, :] + _LOG_2PI + diff**2 / var[None, :, :])
    return np.sum(log_term, axis=2)


# ---------------------------------------------------------------------------
# Forward, backward, and posterior in log-space
# ---------------------------------------------------------------------------


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is None:
        return out.squeeze()
    return np.squeeze(out, axis=axis)


def _forward_log(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray):
    """Forward recursion in log-space.

    Returns
    -------
    log_alpha : (T, S)
    log_likelihood : float = log P(O_{0:T-1})
    """
    T, S = log_B.shape
    log_alpha = np.full((T, S), _NEG_INF, dtype=np.float64)
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        # log_alpha[t, s] = logsumexp_{s'} (log_alpha[t-1, s'] + log_A[s', s]) + log_B[t, s]
        log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]
    return log_alpha, float(_logsumexp(log_alpha[-1]))


def _backward_log(log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """Backward recursion in log-space.

    Returns
    -------
    log_beta : (T, S) with log_beta[T-1] = 0.
    """
    T, S = log_B.shape
    log_beta = np.full((T, S), _NEG_INF, dtype=np.float64)
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, s] = logsumexp_{s'} (log_A[s, s'] + log_B[t+1, s'] + log_beta[t+1, s'])
        log_beta[t] = _logsumexp(log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    return log_beta


def _posterior_log(
    log_alpha: np.ndarray, log_beta: np.ndarray
) -> np.ndarray:
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    return log_gamma


def viterbi_decode(
    log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
) -> np.ndarray:
    """Standard Viterbi MAP decoding. Returns a (T,) integer label sequence."""
    T, S = log_B.shape
    delta = np.full((T, S), _NEG_INF, dtype=np.float64)
    psi = np.zeros((T, S), dtype=np.int64)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        # delta[t, s] = max_s' (delta[t-1, s'] + log_A[s', s]) + log_B[t, s]
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_B[t]
    labels = np.zeros(T, dtype=np.int64)
    labels[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        labels[t] = psi[t + 1, labels[t + 1]]
    return labels


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _initialise_params(
    O: np.ndarray, n_states: int, init_method: str, seed: int
):
    """Return (log_pi, log_A, means, stds) initial parameters."""
    T, F = O.shape
    if init_method == "kmeans1d":
        # Per-state means via quantile split on the first feature
        primary = O[:, 0]
        order = np.argsort(primary)
        chunk = max(1, T // n_states)
        means = np.zeros((n_states, F), dtype=np.float64)
        stds = np.zeros((n_states, F), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for s in range(n_states):
            lo = s * chunk
            hi = (s + 1) * chunk if s < n_states - 1 else T
            sel = order[lo:hi]
            if sel.size == 0:
                sel = order
            means[s] = np.mean(O[sel], axis=0) + 1e-6 * rng.normal(size=F)
            stds[s] = np.std(O[sel], axis=0) + 1e-3
    else:
        global_mean = np.mean(O, axis=0)
        global_std = np.std(O, axis=0) + 1e-3
        means = np.tile(global_mean, (n_states, 1))
        stds = np.tile(global_std, (n_states, 1))

    pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1), dtype=np.float64)
    np.fill_diagonal(A, 0.95)
    log_pi = np.log(pi)
    log_A = np.log(A)
    return log_pi, log_A, means, stds


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------


def _features_to_array(
    features: Sequence[WindowFeatures], names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack named WindowFeatures fields into (T, F) array; return window
    start times alongside."""
    T = len(features)
    F = len(names)
    O = np.zeros((T, F), dtype=np.float64)
    starts = np.zeros(T, dtype=np.float64)
    for t, f in enumerate(features):
        starts[t] = f.window_start
        for k, name in enumerate(names):
            O[t, k] = float(getattr(f, name))
    return O, starts


def recover_regimes_hmm(
    features: Sequence[WindowFeatures],
    cfg: HMMRecoveryConfig,
    *,
    initial_emission_means: Optional[np.ndarray] = None,
) -> HMMRecoveryResult:
    """Fit a Baum-Welch HMM and return posterior + Viterbi labels.

    Parameters
    ----------
    features : sequence of WindowFeatures.
        Output of `recovery.compute_window_features`.
    cfg : HMMRecoveryConfig.
    initial_emission_means : optional (S, F) array.
        Optional warm-start for the per-state emission means. If not
        provided, the kmeans1d-style initialiser is used. The
        Layer1EMResult mu vector (per-event-type mean rate) can be
        passed in as a partial warm-start for the rate component.

    Returns
    -------
    HMMRecoveryResult.
    """
    if len(features) == 0:
        raise ValueError("at least one window is required")
    O, starts = _features_to_array(features, cfg.emission_features)
    T, F = O.shape
    S = cfg.n_states
    if T < S:
        raise ValueError(
            f"need at least n_states={S} windows; got {T}"
        )

    log_pi, log_A, means, stds = _initialise_params(
        O, S, cfg.init_method, cfg.seed
    )
    if initial_emission_means is not None:
        warm = np.asarray(initial_emission_means, dtype=np.float64)
        if warm.shape != (S, F):
            raise ValueError(
                f"initial_emission_means shape {warm.shape} != ({S}, {F})"
            )
        means = warm.copy()

    ll_trace: List[float] = []
    last_ll = -np.inf
    converged = False
    converged_reason = "maxiter"
    n_iter_done = 0
    for it in range(int(cfg.max_iter)):
        log_B = _log_gaussian_emission(O, means, stds)
        log_alpha, ll = _forward_log(log_pi, log_A, log_B)
        log_beta = _backward_log(log_A, log_B)
        log_gamma = _posterior_log(log_alpha, log_beta)
        gamma = np.exp(log_gamma)  # (T, S)

        # Pair posterior (xi)
        # log_xi[t, s, s'] = log_alpha[t, s] + log_A[s, s'] + log_B[t+1, s'] + log_beta[t+1, s'] - log P(O)
        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + log_B[1:, None, :]
            + log_beta[1:, None, :]
            - ll
        )
        xi = np.exp(log_xi)  # (T-1, S, S)

        # M-step
        gamma_sum = np.sum(gamma, axis=0)  # (S,)
        # Initial distribution
        log_pi = np.log(gamma[0] + 1.0e-300)
        # Transition matrix
        xi_sum = np.sum(xi, axis=0)  # (S, S)
        # gamma without the final row, summed
        gamma_no_last = np.sum(gamma[:-1], axis=0)  # (S,)
        A_new = xi_sum / np.maximum(gamma_no_last[:, None], 1.0e-300)
        # Renormalise rows to be safe against floating drift
        A_new = A_new / np.maximum(np.sum(A_new, axis=1, keepdims=True), 1.0e-300)
        log_A = np.log(np.maximum(A_new, 1.0e-300))
        # Emission updates: weighted MLE for Gaussian
        means_new = (gamma.T @ O) / np.maximum(gamma_sum[:, None], 1.0e-300)
        diff = O[None, :, :] - means_new[:, None, :]  # (S, T, F)
        weighted_var = np.sum(gamma.T[:, :, None] * diff**2, axis=1) / np.maximum(
            gamma_sum[:, None], 1.0e-300
        )
        stds_new = np.sqrt(np.maximum(weighted_var, cfg.sigma_floor**2))

        means = means_new
        stds = stds_new

        ll_trace.append(float(ll))
        n_iter_done = it + 1

        if it > 0:
            denom = max(abs(ll), 1.0)
            rel = abs(ll - last_ll) / denom
            if rel < cfg.tol_relative_ll:
                converged = True
                converged_reason = "tol"
                break
        last_ll = ll

    # Final smoothing pass with the converged params
    log_B = _log_gaussian_emission(O, means, stds)
    log_alpha, final_ll = _forward_log(log_pi, log_A, log_B)
    log_beta = _backward_log(log_A, log_B)
    log_gamma = _posterior_log(log_alpha, log_beta)
    posterior = np.exp(log_gamma)
    viterbi_labels = viterbi_decode(log_pi, log_A, log_B)

    return HMMRecoveryResult(
        posterior=posterior,
        viterbi_labels=viterbi_labels,
        transition_matrix=np.exp(log_A),
        initial_state_distribution=np.exp(log_pi),
        emission_means=means,
        emission_stds=stds,
        feature_names=tuple(cfg.emission_features),
        log_likelihood=float(final_ll),
        log_likelihood_trace=np.asarray(ll_trace, dtype=np.float64),
        n_iter=int(n_iter_done),
        converged=bool(converged),
        converged_reason=str(converged_reason),
        window_starts=starts,
        n_windows=int(T),
    )
