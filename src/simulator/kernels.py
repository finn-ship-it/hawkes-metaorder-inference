"""
kernels.py — kernel evaluation, kernel integrals, and upper bounds.

This module implements the exponential kernel family required by the first
coding milestone:

    phi_ij(t) = alpha_ij * exp(-beta_ij * t),   t >= 0.

The protocol `Kernel` is written so that additional kernel families (power-law
with cutoff, exponential sum, non-parametric) can be added later without
changes to hawkes_core.py.

Mathematical reference: Konark's `src/backup/hawkes/simulate_optimized.py`
(functions `expKernel`, `powerLawKernel`, `powerLawCutoff`, and
`powerLawKernelIntegral`). Only the exponential expressions are reimplemented
here; the power-law forms are deferred to a later milestone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np

from .config import (
    SimulatorConfig,
    KernelFamily,
    ExponentialKernelParams,
    ExponentialSumKernelParams,
    PowerLawCutoffKernelParams,
)


# ---------------------------------------------------------------------------
# Kernel protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Kernel(Protocol):
    """Abstract interface for a d x d kernel matrix phi_ij(t)."""

    dimension: int

    def evaluate(self, t: float) -> np.ndarray:
        """Return the (d, d) matrix phi_ij(t) for a scalar lag t >= 0."""
        ...

    def integral(self, t: float) -> np.ndarray:
        """Return the (d, d) matrix of integrals int_0^t phi_ij(s) ds."""
        ...

    def upper_bound(self, lookback: float) -> np.ndarray:
        """Return a (d, d) matrix of per-pair upper bounds on phi_ij(t) for
        t in [0, lookback]. The current simulator instead uses the right-limit
        total intensity to bound its supported non-increasing kernels."""
        ...


# ---------------------------------------------------------------------------
# Exponential kernel
# ---------------------------------------------------------------------------


class ExponentialKernel:
    """Exponential kernel phi_ij(t) = alpha_ij * exp(-beta_ij * t).

    Parameters
    ----------
    alpha : ndarray of shape (d, d), non-negative.
    beta  : ndarray of shape (d, d), strictly positive.
    """

    def __init__(self, alpha: np.ndarray, beta: np.ndarray) -> None:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if alpha.shape != beta.shape:
            raise ValueError(
                f"alpha shape {alpha.shape} must equal beta shape {beta.shape}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
            raise ValueError(f"alpha must be square, got shape {alpha.shape}")
        if np.any(alpha < 0.0):
            raise ValueError("alpha must be non-negative")
        if np.any(beta <= 0.0):
            raise ValueError("beta must be strictly positive")

        self._alpha = alpha
        self._beta = beta
        self.dimension = alpha.shape[0]

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha

    @property
    def beta(self) -> np.ndarray:
        return self._beta

    def evaluate(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        return self._alpha * np.exp(-self._beta * float(t))

    def integral(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        # int_0^t alpha * exp(-beta * s) ds = (alpha / beta) * (1 - exp(-beta * t))
        return (self._alpha / self._beta) * (1.0 - np.exp(-self._beta * float(t)))

    def upper_bound(self, lookback: float) -> np.ndarray:
        """For a non-negative monotonically decreasing kernel the supremum on
        [0, lookback] is attained at t = 0 and equals alpha. The `lookback`
        argument is accepted for interface uniformity with future kernel
        families that are not monotone on [0, lookback]."""
        if lookback < 0.0:
            raise ValueError(f"lookback must be non-negative, got {lookback}")
        return self._alpha.copy()


# ---------------------------------------------------------------------------
# Sum-of-exponentials kernel
# ---------------------------------------------------------------------------


class ExponentialSumKernel:
    """Sum-of-exponentials kernel:

        phi_ij(t) = sum_{r=0}^{R-1} alphas[i, j, r] * exp(-betas[i, j, r] * t).

    The R = 1 case is exactly the ExponentialKernel up to indexing.
    """

    def __init__(self, alphas: np.ndarray, betas: np.ndarray) -> None:
        alphas = np.asarray(alphas, dtype=float)
        betas = np.asarray(betas, dtype=float)
        if alphas.shape != betas.shape:
            raise ValueError(
                f"alphas shape {alphas.shape} must equal betas shape {betas.shape}"
            )
        if alphas.ndim != 3 or alphas.shape[0] != alphas.shape[1]:
            raise ValueError(
                f"alphas must have shape (d, d, R); got {alphas.shape}"
            )
        if np.any(alphas < 0.0):
            raise ValueError("alphas must be non-negative")
        if np.any(betas <= 0.0):
            raise ValueError("betas must be strictly positive")
        self._alphas = alphas
        self._betas = betas
        self.dimension = int(alphas.shape[0])
        self._R = int(alphas.shape[2])

    @property
    def alphas(self) -> np.ndarray:
        return self._alphas

    @property
    def betas(self) -> np.ndarray:
        return self._betas

    def evaluate(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        # sum_r alphas[i,j,r] * exp(-betas[i,j,r]*t)
        return np.sum(self._alphas * np.exp(-self._betas * float(t)), axis=2)

    def integral(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        # sum_r alphas[i,j,r]/betas[i,j,r] * (1 - exp(-betas[i,j,r] * t))
        return np.sum(
            (self._alphas / self._betas) * (1.0 - np.exp(-self._betas * float(t))),
            axis=2,
        )

    def upper_bound(self, lookback: float) -> np.ndarray:
        if lookback < 0.0:
            raise ValueError(f"lookback must be non-negative, got {lookback}")
        # Each component is monotone decreasing on [0, lookback], so the sup
        # over the window is attained at t = 0 and equals sum_r alphas[i,j,r].
        return np.sum(self._alphas, axis=2)


# ---------------------------------------------------------------------------
# Power-law-with-cutoff kernel
# ---------------------------------------------------------------------------


class PowerLawCutoffKernel:
    """Power-law-with-cutoff kernel:

        phi_ij(t) = alpha_ij * (1 + gamma_ij * t)^(-beta_ij)   for t in [0, t_max],
                  = 0                                            otherwise.

    Stationarity check: requires beta_ij > 1 so the kernel is L^1.
    Closed-form integrated mass per pair:

        Phi_ij = (alpha_ij / (gamma_ij * (beta_ij - 1)))
                 * (1 - (1 + gamma_ij * t_max)^(1 - beta_ij)).

    Mathematical reference: `Konarks-github repo/src/simulation/functions.py`,
    `powerLawCutoff`. Code reimplemented here, not vendored.
    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        t_max: float,
    ) -> None:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        if alpha.shape != beta.shape or alpha.shape != gamma.shape:
            raise ValueError(
                f"alpha/beta/gamma shape mismatch: "
                f"{alpha.shape} {beta.shape} {gamma.shape}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
            raise ValueError(f"alpha must be square (d, d); got {alpha.shape}")
        if np.any(alpha < 0.0):
            raise ValueError("alpha must be non-negative")
        if np.any(beta <= 1.0):
            raise ValueError(
                "beta must be strictly > 1 for the kernel to be integrable"
            )
        if np.any(gamma <= 0.0):
            raise ValueError("gamma must be strictly positive")
        if t_max <= 0.0:
            raise ValueError(f"t_max must be > 0; got {t_max}")
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._t_max = float(t_max)
        self.dimension = int(alpha.shape[0])

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha

    @property
    def beta(self) -> np.ndarray:
        return self._beta

    @property
    def gamma(self) -> np.ndarray:
        return self._gamma

    @property
    def t_max(self) -> float:
        return self._t_max

    def evaluate(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        if t > self._t_max:
            return np.zeros_like(self._alpha)
        return self._alpha * (1.0 + self._gamma * float(t)) ** (-self._beta)

    def integral(self, t: float) -> np.ndarray:
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        u = min(float(t), self._t_max)
        # antideriv of alpha (1 + gamma t)^(-beta) is
        #   alpha / (gamma (1 - beta)) * (1 + gamma t)^(1 - beta)
        # for beta > 1; sign convention gives positive integral.
        coeff = self._alpha / (self._gamma * (self._beta - 1.0))
        return coeff * (1.0 - (1.0 + self._gamma * u) ** (1.0 - self._beta))

    def upper_bound(self, lookback: float) -> np.ndarray:
        if lookback < 0.0:
            raise ValueError(f"lookback must be non-negative, got {lookback}")
        # Monotone decreasing on [0, t_max]; sup over [0, lookback] is alpha.
        return self._alpha.copy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_kernel(cfg: SimulatorConfig) -> Kernel:
    """Instantiate a concrete Kernel from a validated SimulatorConfig."""
    if cfg.kernel_family is KernelFamily.EXPONENTIAL:
        params: ExponentialKernelParams = cfg.kernel_params
        return ExponentialKernel(alpha=params.alpha, beta=params.beta)
    if cfg.kernel_family is KernelFamily.EXPONENTIAL_SUM:
        params_sum: ExponentialSumKernelParams = cfg.kernel_params
        return ExponentialSumKernel(alphas=params_sum.alphas, betas=params_sum.betas)
    if cfg.kernel_family is KernelFamily.POWERLAW_CUTOFF:
        params_pl: PowerLawCutoffKernelParams = cfg.kernel_params
        return PowerLawCutoffKernel(
            alpha=params_pl.alpha,
            beta=params_pl.beta,
            gamma=params_pl.gamma,
            t_max=params_pl.t_max,
        )
    raise NotImplementedError(
        f"kernel family {cfg.kernel_family.name} is not implemented"
    )


# ---------------------------------------------------------------------------
# Branching matrix and spectral radius
# ---------------------------------------------------------------------------


def branching_matrix(kernel: Kernel) -> np.ndarray:
    """Return the (d, d) branching matrix Phi_ij = int_0^inf phi_ij(s) ds.

    Closed-form for each supported kernel family:
        ExponentialKernel:        Phi = alpha / beta.
        ExponentialSumKernel:     Phi_ij = sum_r alphas_ij^{(r)} / betas_ij^{(r)}.
        PowerLawCutoffKernel:     Phi_ij = alpha_ij / (gamma_ij * (beta_ij - 1))
                                            * (1 - (1 + gamma_ij * t_max)^(1 - beta_ij)).
    """
    if isinstance(kernel, ExponentialKernel):
        return kernel.alpha / kernel.beta
    if isinstance(kernel, ExponentialSumKernel):
        return np.sum(kernel.alphas / kernel.betas, axis=2)
    if isinstance(kernel, PowerLawCutoffKernel):
        coeff = kernel.alpha / (kernel.gamma * (kernel.beta - 1.0))
        return coeff * (1.0 - (1.0 + kernel.gamma * kernel.t_max) ** (1.0 - kernel.beta))
    raise NotImplementedError(
        "branching_matrix not implemented for kernel type "
        f"{type(kernel).__name__}"
    )


def spectral_radius(kernel: Kernel) -> float:
    """Return the spectral radius of the branching matrix."""
    phi = branching_matrix(kernel)
    eigenvalues = np.linalg.eigvals(phi)
    return float(np.max(np.abs(eigenvalues)))
