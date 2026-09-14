"""
config.py — typed configuration surface for the simulator core.

Defines the event-type enumeration, the kernel-family tag, the parameter
dataclasses consumed by the kernel and simulator modules, and the associated
validation and JSON serialization utilities.

The first coding milestone restricts the configuration to:
    dimension = 4
    event types = {BID_UP, BID_DOWN, ASK_UP, ASK_DOWN}
    kernel family = EXPONENTIAL
    latent_regime.num_regimes = 1
    meta_order.enabled = False
    short horizon, fixed seed

The dataclass surface is deliberately general so that later milestones can
extend kernel family, latent regime, and meta-order specifications without
altering the simulator core signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from enum import IntEnum
from typing import Optional, Tuple, Any, Dict
import json
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EventType(IntEnum):
    """FX top-of-book event types used by the first coding milestone.

    The integer values are the canonical component indices used throughout the
    simulator core and the output schema. They must remain stable across
    milestones because they are persisted in the run artefact.
    """

    BID_UP = 0
    BID_DOWN = 1
    ASK_UP = 2
    ASK_DOWN = 3


class KernelFamily(IntEnum):
    """Supported kernel families.

    Only EXPONENTIAL is implemented in the first coding milestone. The other
    entries are reserved so that configuration files produced now will still
    parse once additional families are added.
    """

    EXPONENTIAL = 0
    POWERLAW_CUTOFF = 1  # reserved; not implemented in the first coding milestone
    EXPONENTIAL_SUM = 2  # reserved; not implemented in the first coding milestone


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExponentialKernelParams:
    """Parameters for the exponential kernel family phi_ij(t) = alpha_ij * exp(-beta_ij * t).

    alpha : ndarray of shape (d, d), non-negative excitation magnitudes.
    beta  : ndarray of shape (d, d), strictly positive decay rates.
    """

    alpha: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class ExponentialSumKernelParams:
    """Parameters for the sum-of-exponentials kernel family.

        phi_ij(t) = sum_{r=0}^{R-1} alpha_ij^{(r)} * exp(-beta_ij^{(r)} * t).

    alphas : ndarray of shape (d, d, R), non-negative.
    betas  : ndarray of shape (d, d, R), strictly positive.
    Branching matrix: Phi_ij = sum_r alpha_ij^{(r)} / beta_ij^{(r)}.
    """

    alphas: np.ndarray
    betas: np.ndarray


@dataclass(frozen=True)
class PowerLawCutoffKernelParams:
    """Parameters for the power-law-with-cutoff kernel family.

        phi_ij(t) = alpha_ij * (1 + gamma_ij * t)^(-beta_ij)   for t in [0, t_max],
                  = 0                                            otherwise.

    alpha : ndarray of shape (d, d), non-negative.
    beta  : ndarray of shape (d, d), strictly > 1.
            (beta <= 1 gives a non-integrable kernel; rejected at validate-time.)
    gamma : ndarray of shape (d, d), strictly > 0.
    t_max : float, strictly > 0. Truncation cutoff in seconds.

    Closed-form integrated mass per (i, j) (when beta > 1):
        Phi_ij = (alpha_ij / (gamma_ij * (beta_ij - 1)))
                 * (1 - (1 + gamma_ij * t_max)^(1 - beta_ij)).

    Mathematical reference: Bacry-Hardiman-Bouchaud microstructure kernel
    (cf. `Konarks-github repo/src/simulation/functions.py`, `powerLawCutoff`).
    Code reimplemented here; not vendored.
    """

    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    t_max: float


@dataclass(frozen=True)
class BaselineParams:
    """Per-component baseline intensities.

    mu : ndarray of shape (d,), strictly positive baseline intensities.
    """

    mu: np.ndarray


@dataclass(frozen=True)
class LatentRegimeSpec:
    """Latent regime channel specification.

    For K == 1 the specification is degenerate: transition_rates is None and
    initial_regime is 0.

    For K > 1 the latent regime follows a continuous-time Markov chain (CTMC)
    on the state set {0, 1, ..., K - 1} with generator matrix
    transition_rates of shape (K, K). The generator satisfies the standard
    CTMC conditions: off-diagonal entries are non-negative and each row sums
    to zero.
    """

    num_regimes: int = 1
    transition_rates: Optional[np.ndarray] = None  # shape (K, K); None when K == 1
    initial_regime: int = 0


@dataclass(frozen=True)
class MetaOrderWindow:
    """A single meta-order window applied to the baseline intensity.

    During [start_time, end_time) the baseline intensity mu_i(z_t) for each
    event type i in target_event_types is multiplied by (1 + alpha).
    Non-target event types are unaffected. Kernels are unchanged.

    Fields
    ------
    start_time : float
        Absolute window start, in simulator-native seconds. Required to be
        non-negative and strictly less than end_time.
    end_time : float
        Absolute window end (exclusive).
    alpha : float
        Multiplicative uplift magnitude. The effective factor applied to
        mu_i during the window equals (1 + alpha). Required to be finite
        and non-negative.
    target_event_types : Tuple[int, ...]
        Event-type indices to which the window applies. Required to be
        non-empty, unique, and within [0, dimension).
    """

    start_time: float
    end_time: float
    alpha: float
    target_event_types: Tuple[int, ...]


@dataclass(frozen=True)
class MetaOrderSpec:
    """Meta-order specification.

    Holds zero or more non-overlapping windows. The boolean `enabled`
    attribute is derived from whether any windows are configured and is
    exposed as a property so that callers can query a single flag without
    inspecting the windows tuple.
    """

    windows: Tuple[MetaOrderWindow, ...] = ()

    @property
    def enabled(self) -> bool:
        return len(self.windows) > 0


@dataclass(frozen=True)
class NumericalSettings:
    """Numerical controls for the thinning loop."""

    truncation_window: float = 500.0
    spectral_radius_cap: float = 0.95
    upper_bound_safety: float = 1.10
    use_exp_approximation: bool = True


@dataclass(frozen=True)
class SimulatorConfig:
    """Top-level configuration object consumed by the simulator core."""

    dimension: int
    horizon: float
    seed: int
    kernel_family: KernelFamily
    kernel_params: ExponentialKernelParams
    baseline: BaselineParams
    latent_regime: LatentRegimeSpec = field(default_factory=LatentRegimeSpec)
    meta_order: MetaOrderSpec = field(default_factory=MetaOrderSpec)
    numerics: NumericalSettings = field(default_factory=NumericalSettings)
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ConfigurationError(ValueError):
    """Raised when a SimulatorConfig fails validation."""


def validate_shapes(cfg: SimulatorConfig) -> None:
    d = cfg.dimension
    if d < 1:
        raise ConfigurationError(f"dimension must be >= 1, got {d}")

    k = cfg.latent_regime.num_regimes
    if k < 1:
        raise ConfigurationError(f"latent_regime.num_regimes must be >= 1, got {k}")

    mu = np.asarray(cfg.baseline.mu)
    if k == 1:
        if mu.shape != (d,) and mu.shape != (1, d):
            raise ConfigurationError(
                f"baseline.mu has shape {mu.shape}, expected ({d},) or (1, {d}) "
                f"for num_regimes == 1"
            )
    else:
        if mu.shape != (k, d):
            raise ConfigurationError(
                f"baseline.mu has shape {mu.shape}, expected ({k}, {d}) "
                f"for num_regimes == {k}"
            )

    if cfg.kernel_family is KernelFamily.EXPONENTIAL:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        if alpha.shape != (d, d):
            raise ConfigurationError(
                f"kernel_params.alpha has shape {alpha.shape}, expected ({d}, {d})"
            )
        if beta.shape != (d, d):
            raise ConfigurationError(
                f"kernel_params.beta has shape {beta.shape}, expected ({d}, {d})"
            )
    elif cfg.kernel_family is KernelFamily.EXPONENTIAL_SUM:
        alphas = np.asarray(cfg.kernel_params.alphas)
        betas = np.asarray(cfg.kernel_params.betas)
        if alphas.ndim != 3 or alphas.shape[:2] != (d, d):
            raise ConfigurationError(
                f"kernel_params.alphas shape {alphas.shape}, expected ({d}, {d}, R)"
            )
        if betas.shape != alphas.shape:
            raise ConfigurationError(
                f"kernel_params.betas shape {betas.shape} != alphas shape {alphas.shape}"
            )
    elif cfg.kernel_family is KernelFamily.POWERLAW_CUTOFF:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        gamma = np.asarray(cfg.kernel_params.gamma)
        if alpha.shape != (d, d) or beta.shape != (d, d) or gamma.shape != (d, d):
            raise ConfigurationError(
                f"kernel_params.alpha/beta/gamma must each have shape ({d}, {d}); got "
                f"{alpha.shape}/{beta.shape}/{gamma.shape}"
            )
    else:
        raise ConfigurationError(
            f"unsupported kernel_family {cfg.kernel_family}"
        )

    q = cfg.latent_regime.transition_rates
    if k == 1:
        if q is not None:
            raise ConfigurationError(
                "latent_regime.transition_rates must be None when num_regimes == 1"
            )
    else:
        if q is None:
            raise ConfigurationError(
                f"latent_regime.transition_rates is required for num_regimes == {k}"
            )
        q_arr = np.asarray(q)
        if q_arr.shape != (k, k):
            raise ConfigurationError(
                f"latent_regime.transition_rates has shape {q_arr.shape}, "
                f"expected ({k}, {k})"
            )

    ir = int(cfg.latent_regime.initial_regime)
    if not (0 <= ir < k):
        raise ConfigurationError(
            f"latent_regime.initial_regime {ir} out of range [0, {k})"
        )


def validate_nonnegative(cfg: SimulatorConfig) -> None:
    mu = np.asarray(cfg.baseline.mu)
    if not np.all(np.isfinite(mu)):
        raise ConfigurationError("baseline.mu contains non-finite values")
    if np.any(mu <= 0.0):
        raise ConfigurationError("baseline.mu must be strictly positive")

    if cfg.kernel_family is KernelFamily.EXPONENTIAL:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        if not np.all(np.isfinite(alpha)):
            raise ConfigurationError("kernel_params.alpha contains non-finite values")
        if not np.all(np.isfinite(beta)):
            raise ConfigurationError("kernel_params.beta contains non-finite values")
        if np.any(alpha < 0.0):
            raise ConfigurationError("kernel_params.alpha must be non-negative")
        if np.any(beta <= 0.0):
            raise ConfigurationError("kernel_params.beta must be strictly positive")
    elif cfg.kernel_family is KernelFamily.EXPONENTIAL_SUM:
        alphas = np.asarray(cfg.kernel_params.alphas)
        betas = np.asarray(cfg.kernel_params.betas)
        if not np.all(np.isfinite(alphas)) or not np.all(np.isfinite(betas)):
            raise ConfigurationError("kernel_params.alphas/betas contain non-finite values")
        if np.any(alphas < 0.0):
            raise ConfigurationError("kernel_params.alphas must be non-negative")
        if np.any(betas <= 0.0):
            raise ConfigurationError("kernel_params.betas must be strictly positive")
    elif cfg.kernel_family is KernelFamily.POWERLAW_CUTOFF:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        gamma = np.asarray(cfg.kernel_params.gamma)
        if not (np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(gamma))):
            raise ConfigurationError("powerlaw kernel params contain non-finite values")
        if np.any(alpha < 0.0):
            raise ConfigurationError("kernel_params.alpha must be non-negative")
        if np.any(beta <= 1.0):
            raise ConfigurationError(
                "kernel_params.beta must be strictly > 1 for an integrable power-law kernel"
            )
        if np.any(gamma <= 0.0):
            raise ConfigurationError("kernel_params.gamma must be strictly positive")
        if cfg.kernel_params.t_max <= 0.0:
            raise ConfigurationError("kernel_params.t_max must be > 0")

    q = cfg.latent_regime.transition_rates
    if q is not None:
        q_arr = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q_arr)):
            raise ConfigurationError(
                "latent_regime.transition_rates contains non-finite values"
            )
        k = cfg.latent_regime.num_regimes
        off_diag_mask = ~np.eye(k, dtype=bool)
        if np.any(q_arr[off_diag_mask] < 0.0):
            raise ConfigurationError(
                "latent_regime.transition_rates off-diagonal entries must be "
                "non-negative"
            )
        row_sums = q_arr.sum(axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            raise ConfigurationError(
                f"latent_regime.transition_rates row sums must be zero "
                f"(got {row_sums})"
            )
        diag = np.diag(q_arr)
        if np.any(diag > 0.0):
            raise ConfigurationError(
                "latent_regime.transition_rates diagonal entries must be "
                "non-positive"
            )

    if cfg.horizon <= 0.0:
        raise ConfigurationError("horizon must be strictly positive")
    if cfg.numerics.truncation_window <= 0.0:
        raise ConfigurationError("numerics.truncation_window must be strictly positive")
    if not (0.0 < cfg.numerics.spectral_radius_cap < 1.0):
        raise ConfigurationError(
            "numerics.spectral_radius_cap must lie in (0, 1)"
        )
    if cfg.numerics.upper_bound_safety < 1.0:
        raise ConfigurationError(
            "numerics.upper_bound_safety must be >= 1.0"
        )


def validate_spectral_radius(cfg: SimulatorConfig) -> float:
    """Return the spectral radius of the branching matrix and raise if it
    violates the configured cap.

    Closed-form branching matrix per family:
        EXPONENTIAL:        Phi_ij = alpha_ij / beta_ij.
        EXPONENTIAL_SUM:    Phi_ij = sum_r alphas_ij^{(r)} / betas_ij^{(r)}.
        POWERLAW_CUTOFF:    Phi_ij = alpha_ij / (gamma_ij * (beta_ij - 1))
                                     * (1 - (1 + gamma_ij * t_max)^(1 - beta_ij)).
    """
    if cfg.kernel_family is KernelFamily.EXPONENTIAL:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        phi = alpha / beta
    elif cfg.kernel_family is KernelFamily.EXPONENTIAL_SUM:
        alphas = np.asarray(cfg.kernel_params.alphas)
        betas = np.asarray(cfg.kernel_params.betas)
        phi = np.sum(alphas / betas, axis=2)
    elif cfg.kernel_family is KernelFamily.POWERLAW_CUTOFF:
        alpha = np.asarray(cfg.kernel_params.alpha)
        beta = np.asarray(cfg.kernel_params.beta)
        gamma = np.asarray(cfg.kernel_params.gamma)
        t_max = float(cfg.kernel_params.t_max)
        coeff = alpha / (gamma * (beta - 1.0))
        phi = coeff * (1.0 - (1.0 + gamma * t_max) ** (1.0 - beta))
    else:
        raise ConfigurationError(
            f"spectral-radius validation not implemented for "
            f"kernel family {cfg.kernel_family.name}"
        )

    eigenvalues = np.linalg.eigvals(phi)
    rho = float(np.max(np.abs(eigenvalues)))

    if rho >= cfg.numerics.spectral_radius_cap:
        raise ConfigurationError(
            f"spectral radius {rho:.6f} exceeds cap "
            f"{cfg.numerics.spectral_radius_cap:.6f}"
        )
    return rho


def validate_meta_order(cfg: SimulatorConfig) -> None:
    """Validate the meta-order specification.

    Each window must satisfy: 0 <= start_time < end_time (both finite),
    alpha finite and non-negative, target_event_types non-empty, unique,
    and within [0, dimension). Windows must not overlap.
    """
    d = cfg.dimension
    windows = cfg.meta_order.windows
    if not windows:
        return

    for idx, w in enumerate(windows):
        if not (np.isfinite(w.start_time) and np.isfinite(w.end_time)):
            raise ConfigurationError(
                f"meta_order.windows[{idx}] has non-finite time bounds"
            )
        if w.start_time < 0.0:
            raise ConfigurationError(
                f"meta_order.windows[{idx}].start_time must be >= 0, "
                f"got {w.start_time}"
            )
        if w.end_time <= w.start_time:
            raise ConfigurationError(
                f"meta_order.windows[{idx}]: end_time ({w.end_time}) must be "
                f"strictly greater than start_time ({w.start_time})"
            )
        if not np.isfinite(w.alpha) or w.alpha < 0.0:
            raise ConfigurationError(
                f"meta_order.windows[{idx}].alpha must be finite and >= 0, "
                f"got {w.alpha}"
            )
        if len(w.target_event_types) == 0:
            raise ConfigurationError(
                f"meta_order.windows[{idx}].target_event_types must be non-empty"
            )
        if len(set(w.target_event_types)) != len(w.target_event_types):
            raise ConfigurationError(
                f"meta_order.windows[{idx}].target_event_types contains "
                f"duplicates"
            )
        for i in w.target_event_types:
            if not (0 <= int(i) < d):
                raise ConfigurationError(
                    f"meta_order.windows[{idx}] target event type {i} out of "
                    f"range [0, {d})"
                )

    # Non-overlap check. Sort a copy by start_time; require each end_time to be
    # at most the next start_time.
    sorted_windows = sorted(windows, key=lambda w: w.start_time)
    for i in range(len(sorted_windows) - 1):
        if sorted_windows[i].end_time > sorted_windows[i + 1].start_time:
            raise ConfigurationError(
                f"meta_order windows overlap: "
                f"[{sorted_windows[i].start_time}, {sorted_windows[i].end_time}) "
                f"and "
                f"[{sorted_windows[i + 1].start_time}, "
                f"{sorted_windows[i + 1].end_time})"
            )


def validate(cfg: SimulatorConfig) -> None:
    """Run all validators in dependency order."""
    validate_shapes(cfg)
    validate_nonnegative(cfg)
    validate_spectral_radius(cfg)
    validate_meta_order(cfg)


def validate_first_milestone(cfg: SimulatorConfig) -> None:
    """Enforce the first coding milestone constraints in addition to validate().

    dimension == 4, kernel_family == EXPONENTIAL,
    latent_regime.num_regimes == 1, meta_order.enabled is False.
    """
    validate(cfg)
    if cfg.dimension != 4:
        raise ConfigurationError(
            f"first coding milestone requires dimension == 4, got {cfg.dimension}"
        )
    if cfg.kernel_family is not KernelFamily.EXPONENTIAL:
        raise ConfigurationError(
            "first coding milestone requires kernel_family == EXPONENTIAL"
        )
    if cfg.latent_regime.num_regimes != 1:
        raise ConfigurationError(
            "first coding milestone requires latent_regime.num_regimes == 1"
        )
    if len(cfg.meta_order.windows) > 0:
        raise ConfigurationError(
            "first coding milestone requires meta_order.windows to be empty"
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "dtype": str(obj.dtype), "data": obj.tolist()}
    if isinstance(obj, IntEnum):
        return {"__enum__": type(obj).__name__, "value": int(obj)}
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [_to_serializable(x) for x in obj]}
    if isinstance(obj, (list,)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


_ENUM_REGISTRY = {
    "EventType": EventType,
    "KernelFamily": KernelFamily,
}


def _from_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.asarray(obj["data"], dtype=obj["dtype"])
        if "__enum__" in obj:
            return _ENUM_REGISTRY[obj["__enum__"]](obj["value"])
        if obj.get("__tuple__"):
            return tuple(_from_serializable(x) for x in obj["items"])
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(x) for x in obj]
    return obj


def config_to_json(cfg: SimulatorConfig, path: str) -> None:
    """Serialize cfg to a JSON file at path.

    The serialization round-trips through config_from_json without loss for
    all supported field types in the first coding milestone.
    """
    payload: Dict[str, Any] = {
        "__simulator_config_version__": cfg.version,
        "dimension": cfg.dimension,
        "horizon": cfg.horizon,
        "seed": cfg.seed,
        "kernel_family": _to_serializable(cfg.kernel_family),
        "kernel_params": _to_serializable(cfg.kernel_params),
        "baseline": _to_serializable(cfg.baseline),
        "latent_regime": _to_serializable(cfg.latent_regime),
        "meta_order": _to_serializable(cfg.meta_order),
        "numerics": _to_serializable(cfg.numerics),
        "version": cfg.version,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _reconstruct_meta_order_spec(meta_order_raw: Any) -> MetaOrderSpec:
    """Reconstruct a MetaOrderSpec from a deserialised JSON payload.

    Accepted forms:
        {"windows": (...)} or {"windows": [...]}  - canonical new format
        {"enabled": False, ...}                    - legacy format with no
                                                     windows recorded; treated
                                                     as an empty specification.
    """
    if not isinstance(meta_order_raw, dict):
        return MetaOrderSpec()
    windows_raw = meta_order_raw.get("windows", ())
    # After _from_serializable, the "windows" payload is either a list or a
    # tuple (if it went through the __tuple__ wrapper).
    if windows_raw is None:
        windows_raw = ()
    reconstructed = []
    for w in windows_raw:
        reconstructed.append(
            MetaOrderWindow(
                start_time=float(w["start_time"]),
                end_time=float(w["end_time"]),
                alpha=float(w["alpha"]),
                target_event_types=tuple(int(x) for x in w["target_event_types"]),
            )
        )
    return MetaOrderSpec(windows=tuple(reconstructed))


def config_from_json(path: str) -> SimulatorConfig:
    """Deserialize a SimulatorConfig from a JSON file at path."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    kernel_family = _from_serializable(raw["kernel_family"])
    kernel_params_raw = _from_serializable(raw["kernel_params"])
    baseline_raw = _from_serializable(raw["baseline"])
    latent_regime_raw = _from_serializable(raw["latent_regime"])
    meta_order_raw = _from_serializable(raw["meta_order"])
    numerics_raw = _from_serializable(raw["numerics"])

    return SimulatorConfig(
        dimension=int(raw["dimension"]),
        horizon=float(raw["horizon"]),
        seed=int(raw["seed"]),
        kernel_family=kernel_family,
        kernel_params=ExponentialKernelParams(
            alpha=np.asarray(kernel_params_raw["alpha"], dtype=float),
            beta=np.asarray(kernel_params_raw["beta"], dtype=float),
        ),
        baseline=BaselineParams(
            mu=np.asarray(baseline_raw["mu"], dtype=float),
        ),
        latent_regime=LatentRegimeSpec(
            num_regimes=int(latent_regime_raw["num_regimes"]),
            transition_rates=(
                np.asarray(latent_regime_raw["transition_rates"], dtype=float)
                if latent_regime_raw.get("transition_rates") is not None
                else None
            ),
            initial_regime=int(latent_regime_raw.get("initial_regime", 0)),
        ),
        meta_order=_reconstruct_meta_order_spec(meta_order_raw),
        numerics=NumericalSettings(
            truncation_window=float(numerics_raw["truncation_window"]),
            spectral_radius_cap=float(numerics_raw["spectral_radius_cap"]),
            upper_bound_safety=float(numerics_raw["upper_bound_safety"]),
            use_exp_approximation=bool(numerics_raw["use_exp_approximation"]),
        ),
        version=str(raw.get("version", "0.1.0")),
    )
