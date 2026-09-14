"""
d2_konark_backend.py — D2 cross-simulator adapter (Konark HawkesRLTrading).

Wraps Konark's `HawkesRLTrading.src.Stochastic_Processes.Arrival_Models.HawkesArrival`
as a second generative backend (D2) behind the same Phi_beta observation
operator the FX-native simulator (D1) uses. The Konark source is consumed
**read-only**; this adapter does not edit `Konarks-github repo/`. Any
runtime patches required to make Konark's source compatible with the
current numpy version are applied at the instance level and documented
explicitly in the `KONARK_RUNTIME_PATCHES` constant.

Konark source-of-truth verification (per prompt-07 spec, items (a)-(d)):

(a) Imports trace to `Konarks-github repo/HawkesRLTrading/`:
        from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import HawkesArrival
    `My repo/` is NOT used as the event-generation source; the D2
    backend's only role is to wrap Konark's generator and project its
    12-event LOB stream onto the FX 4-type top-of-book alphabet.

(b) Event-generation call-site (Konark API):
        time, side, order_type, level, size = arrival.get_nextarrival(timelimit=T)
    which internally calls `HawkesArrival.thinningOgataIS2(T)` (the
    multivariate Ogata thinning loop documented in
    Bacry-Mastromatteo-Muzy 2015). The `noRL_runner.py` script in the
    Konark repo wires this same `HawkesArrival` instance into a
    `tradingEnv`; this adapter cuts the trading-env layer (which depends
    on equity-style data loaders and a hardcoded INTC kernel pickle, both
    out of scope per `KONARK_REUSE_AUDIT.md` class C and the spec's
    "equity LOBSTER taxonomy is out of scope" exclusion) and uses
    `HawkesArrival` directly with Konark's own `generatefakeparams()`
    defaults.

(c) Pre-projection events carry the 12-type Konark-native alphabet:
        ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask",
         "mo_Ask", "lo_inspread_Ask", "lo_inspread_Bid", "mo_Bid",
         "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
    Post-projection events carry the FX 4-type alphabet:
        {0: BID_UP, 1: BID_DOWN, 2: ASK_UP, 3: ASK_DOWN}.

(d) Latent count > observed count holds by construction: only four of
    the twelve Konark types map onto top-of-book changes (mo_Bid,
    mo_Ask, lo_inspread_Bid, lo_inspread_Ask); the remaining eight
    (lo_top, co_top, lo_deep, co_deep on each side) are queue-depth or
    deeper-level events that do NOT change the best bid or ask in the
    Phi_beta projection adopted here. Latent / observed counts are
    surfaced in `D2RunResult` for the spec's verification gate.

The Phi_beta projection rule in this adapter is the **deterministic
top-of-book projection** that censors all events not changing the best
bid or best ask:

    mo_Bid          -> BID_DOWN  (market sell consumes top bid)
    mo_Ask          -> ASK_UP    (market buy consumes top ask)
    lo_inspread_Bid -> BID_UP    (improving bid posted inside spread)
    lo_inspread_Ask -> ASK_DOWN  (improving ask posted inside spread)
    all other types -> CENSORED

This deterministic rule is documented; alternative probabilistic
projections (for example, co_top_Bid clearing the top with probability
1 / queue_depth) are explicitly out of scope for the MSc-bounded D2
backend and would be a methodological extension.
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    "D2KonarkConfig",
    "D2RunResult",
    "ensure_konark_on_path",
    "build_konark_arrival",
    "simulate_d2_konark",
    "project_konark_to_fx_topofbook",
    "KONARK_NATIVE_TYPES",
    "KONARK_TYPE_TO_INDEX",
    "FX_TOPOFBOOK_TYPES",
    "PHI_BETA_PROJECTION",
    "KONARK_RUNTIME_PATCHES",
]


# ---------------------------------------------------------------------------
# Konark-native and FX-projected alphabets
# ---------------------------------------------------------------------------


KONARK_NATIVE_TYPES: Tuple[str, ...] = (
    "lo_deep_Ask",
    "co_deep_Ask",
    "lo_top_Ask",
    "co_top_Ask",
    "mo_Ask",
    "lo_inspread_Ask",
    "lo_inspread_Bid",
    "mo_Bid",
    "co_top_Bid",
    "lo_top_Bid",
    "co_deep_Bid",
    "lo_deep_Bid",
)

KONARK_TYPE_TO_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(KONARK_NATIVE_TYPES)
}

# FX 4-type top-of-book alphabet, matching the FX-native simulator's
# EventType enum (BID_UP=0, BID_DOWN=1, ASK_UP=2, ASK_DOWN=3).
FX_TOPOFBOOK_TYPES: Tuple[str, ...] = (
    "BID_UP",
    "BID_DOWN",
    "ASK_UP",
    "ASK_DOWN",
)

# Deterministic Phi_beta projection: maps a Konark-native type name to a
# FX top-of-book index, or to None for censored types. Documented in the
# module docstring.
PHI_BETA_PROJECTION: Dict[str, Optional[int]] = {
    "mo_Bid": 1,            # BID_DOWN
    "mo_Ask": 2,            # ASK_UP
    "lo_inspread_Bid": 0,   # BID_UP
    "lo_inspread_Ask": 3,   # ASK_DOWN
    # Censored (no top-of-book change under the deterministic projection)
    "lo_deep_Ask": None,
    "co_deep_Ask": None,
    "lo_top_Ask": None,
    "co_top_Ask": None,
    "lo_deep_Bid": None,
    "co_deep_Bid": None,
    "lo_top_Bid": None,
    "co_top_Bid": None,
}


# ---------------------------------------------------------------------------
# Konark runtime patches
# ---------------------------------------------------------------------------


KONARK_RUNTIME_PATCHES: Tuple[Dict[str, str], ...] = (
    {
        "target": "HawkesArrival.thinningOgataIS2 line ~335",
        "issue": (
            "Original line `self.lamb = float(sum(decays))` fails on "
            "numpy 2.x because `sum(decays)` (Python builtin over a "
            "(12, 1) array) returns a (1,)-shaped numpy array, and "
            "`float(...)` on a non-scalar array now raises TypeError."
        ),
        "patch": (
            "Replace the offending line with "
            "`self.lamb = float(np.sum(decays))`. Applied at instance "
            "level via types.MethodType in `_apply_runtime_patches`. "
            "Konark source is NOT modified on disk (read-only)."
        ),
    },
    {
        "target": "HawkesArrival.thinningOgataIS2 timeseries-truncation block (post-append)",
        "issue": (
            "Konark advances `self.left` past stale points but never "
            "resets it to 0 after the timeseries is truncated. Once "
            "self.left grows to the post-truncation length, the next "
            "truncation wipes the entire timeseries, and the next "
            "`get_nextarrival` call hits IndexError on "
            "`self.timeseries[-1]`. Surfaces after ~30 events under the "
            "default expApprox=True / TAU=10 configuration."
        ),
        "patch": (
            "Append `self.left = 0` immediately after the truncation "
            "slice. Applied at instance level in the same patched "
            "method. Konark source is NOT modified on disk (read-only)."
        ),
    },
)


# ---------------------------------------------------------------------------
# Path resolution: locate Konark repo
# ---------------------------------------------------------------------------


def _resolve_konark_repo_root() -> str:
    """Return absolute path to the Konark repo root (the directory whose
    immediate child is the `HawkesRLTrading/` package). Resolves first via
    the `KONARK_REPO_ROOT` env var, then via the canonical layout
    `.../Work on Hawkes/Konarks-github repo/`."""
    env = os.environ.get("KONARK_REPO_ROOT")
    if env and os.path.isdir(os.path.join(env, "HawkesRLTrading")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    repo_root = os.path.dirname(src_dir)
    work_root = os.path.dirname(repo_root)
    candidate = os.path.join(work_root, "Konarks-github repo")
    if os.path.isdir(os.path.join(candidate, "HawkesRLTrading")):
        return candidate
    raise FileNotFoundError(
        "Could not locate Konark repo root. Set the KONARK_REPO_ROOT "
        "environment variable or place the repo at "
        "'<work_root>/Konarks-github repo/'."
    )


def ensure_konark_on_path() -> str:
    """Idempotently add the Konark repo root to `sys.path` and return it.

    Re-importable: subsequent calls return the same path without
    duplicating the entry."""
    root = _resolve_konark_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


# ---------------------------------------------------------------------------
# Runtime patch — fix numpy-2.x incompatibility in Konark's thinning loop
# ---------------------------------------------------------------------------


def _patched_thinning_ogata_is2(self, T=None):
    """Drop-in replacement for `HawkesArrival.thinningOgataIS2` with the
    numpy-2.x scalar-conversion fix (line ~335) applied. Otherwise the
    body is byte-equivalent to Konark's original implementation; cf.
    `KONARK_RUNTIME_PATCHES`."""
    if T is None:
        raise ValueError(
            "Expected 'float' for Thinning Ogata time limit, received 'NoneType'"
        )
    if self.n is None:
        self.n = self.num_nodes * [0]
    if self.baselines is None:
        self.baselines = self.num_nodes * [0]
    if self.Ts is None:
        self.Ts = self.num_nodes * [()]
    if self.spread is None:
        self.spread = 1
    self.baselines = self.kernelparams[1].copy()
    mat = np.zeros((self.num_nodes, self.num_nodes))
    if self.s is None:
        self.s = 0
    hourIndex = min(12, int(self.s // 1800))
    self.todmult = self.tod[:, hourIndex].reshape((12, 1))
    if not np.sum(self.kernelparams[0][3]) == 0:
        mat = (
            self.todmult
            * self.kernelparams[0][0]
            * self.kernelparams[0][1]
            / (
                (self.kernelparams[0][2] - (1 * int(not self.expApprox)))
                * self.kernelparams[0][3]
            )
        )
    mat = np.nan_to_num(mat)
    self.baselines[5] = (
        (self.spread / self.avgSpread) ** self.beta
    ) * self.baselines[5]
    self.baselines[6] = (
        (self.spread / self.avgSpread) ** self.beta
    ) * self.baselines[6]
    specRad = np.max(np.linalg.eig(mat)[0]).real
    if specRad < 1:
        specRad = 0.99
    if self.lamb is None:
        decays = (0.99 / specRad) * self.todmult * self.baselines
        self.lamb = float(np.sum(decays))  # patched
    if self.left is None:
        self.left = 0
    if self.timeseries is None:
        self.timeseries = []

    pointcount = 0
    while self.s < T:
        lamb_bar = self.lamb
        u = np.random.uniform(0, 1)
        if lamb_bar == 0:
            self.s += 0.1
        else:
            w = max(1e-7, -1 * np.log(u) / lamb_bar)
            self.s += w
        if self.s > T:
            return pointcount
        hourIndex = min(12, int(self.s // 1800))
        self.todmult = self.tod[:, hourIndex].reshape((12, 1)) * (0.99 / specRad)
        decays = self.todmult * self.baselines
        if self.timeseries == []:
            pass
        else:
            while self.left < len(self.timeseries):
                if self.s - self.timeseries[self.left][0] >= self.TAU:
                    self.left += 1
                else:
                    break
            for point in self.timeseries[self.left :]:
                if self.expApprox:
                    from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import (
                        expApprox,
                    )

                    kern = expApprox(
                        time=self.s - point[0],
                        alpha=self.kernelparams[0][0][point[1]]
                        * self.kernelparams[0][1][point[1]],
                        beta=self.kernelparams[0][2][point[1]],
                        gamma=self.kernelparams[0][3][point[1]],
                    )
                else:
                    from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import (
                        powerLawCutoff,
                    )

                    kern = powerLawCutoff(
                        time=self.s - point[0],
                        alpha=self.kernelparams[0][0][point[1]]
                        * self.kernelparams[0][1][point[1]],
                        beta=self.kernelparams[0][2][point[1]],
                        gamma=self.kernelparams[0][3][point[1]],
                    )
                kern = kern.reshape((12, 1))
                decays += self.todmult * kern
        decays = np.maximum(decays, 0)
        decays[5] = (
            (self.spread / self.avgSpread) ** self.beta
        ) * decays[5]
        decays[6] = (
            (self.spread / self.avgSpread) ** self.beta
        ) * decays[6]
        if 100 * np.round(self.spread, 2) < 2:
            decays[5] = decays[6] = 0
        self.lamb = float(np.sum(decays))  # patched
        D = np.random.uniform(0, 1)
        if D * lamb_bar <= self.lamb:
            pointcount += 1
            self.current_intensity = decays.copy()
            k = 0
            total = decays[k]
            while D * lamb_bar >= total:
                k += 1
                total += decays[k]
            newdecays = (
                self.todmult
                * self.kernelparams[0][0][k].reshape(12, 1)
                * self.kernelparams[0][1][k].reshape(12, 1)
            )
            newdecays = np.maximum(newdecays, 0)
            newdecays[5] = (
                (self.spread / self.avgSpread) ** self.beta
            ) * newdecays[5]
            newdecays[6] = (
                (self.spread / self.avgSpread) ** self.beta
            ) * newdecays[6]
            if 100 * np.round(self.spread, 2) < 2:
                newdecays[5] = newdecays[6] = 0
            self.lamb += float(np.sum(newdecays))  # patched
            if len(self.Ts[k]) > 0:
                T_Minus1 = self.Ts[k][-1]
            else:
                T_Minus1 = 0
            decays = np.array(self.baselines.copy())
            decays[5] = (
                (self.spread / self.avgSpread) ** self.beta
            ) * decays[5]
            decays[6] = (
                (self.spread / self.avgSpread) ** self.beta
            ) * decays[6]
            decays = decays * (self.s - T_Minus1)
            self.Ts[k] += (self.s,)
            self.n[k] += 1
            self.timeseries.append((self.s, k))
            if self.timeseries[-1][0] - self.timeseries[0][0] > self.TAU:
                self.timeseries = self.timeseries[self.left :]
                self.left = 0  # patched: reset stale offset post-truncation
            self.pointcount += pointcount
            return pointcount
    return pointcount


def _apply_runtime_patches(arrival) -> None:
    """Bind the patched `thinningOgataIS2` to the given HawkesArrival
    instance. Does not modify the class definition or Konark source on
    disk."""
    arrival.thinningOgataIS2 = types.MethodType(
        _patched_thinning_ogata_is2, arrival
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D2KonarkConfig:
    """Configuration for the D2 Konark backend.

    Defaults: Konark's `generatefakeparams()` produces a 12x12 power-law
    kernel matrix with random sign mask. The default seed is 1; user
    code can override.
    """

    spread0: float = 0.01
    seed: int = 1
    avg_spread_override: Optional[float] = None
    use_exp_approx: bool = True

    def __post_init__(self) -> None:
        if self.spread0 <= 0.0:
            raise ValueError(f"spread0 must be > 0; got {self.spread0}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0; got {self.seed}")


# ---------------------------------------------------------------------------
# D2 run output
# ---------------------------------------------------------------------------


@dataclass
class D2RunResult:
    """Result of one D2 simulation + Phi_beta projection.

    Fields
    ------
    horizon_seconds : float
    konark_native_times : np.ndarray
        Times of all 12-event Konark-native events.
    konark_native_types : np.ndarray
        Konark-native type indices [0, 12).
    konark_native_type_counts : Dict[str, int]
        Per-name count over the 12 Konark-native event types.
    fx_topofbook_times : np.ndarray
        Times of FX-projected top-of-book events (subset of
        `konark_native_times`).
    fx_topofbook_types : np.ndarray
        FX 4-type indices [0, 4).
    fx_topofbook_type_counts : Dict[str, int]
        Per-name count over the FX 4-type alphabet.
    n_konark_latent : int
    n_fx_observed : int
    censoring_fraction : float
    wall_seconds : float
    """

    horizon_seconds: float
    konark_native_times: np.ndarray
    konark_native_types: np.ndarray
    konark_native_type_counts: Dict[str, int]
    fx_topofbook_times: np.ndarray
    fx_topofbook_types: np.ndarray
    fx_topofbook_type_counts: Dict[str, int]
    n_konark_latent: int
    n_fx_observed: int
    censoring_fraction: float
    wall_seconds: float

    @property
    def latent_gt_observed(self) -> bool:
        return self.n_konark_latent > self.n_fx_observed


# ---------------------------------------------------------------------------
# Public API: build, simulate, project
# ---------------------------------------------------------------------------


def build_konark_arrival(cfg: D2KonarkConfig):
    """Construct a Konark `HawkesArrival` with `generatefakeparams()`
    defaults and the runtime patch applied.

    Returns the live `HawkesArrival` instance.
    """
    ensure_konark_on_path()
    from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import (
        HawkesArrival,
    )

    np.random.seed(int(cfg.seed))
    arrival = HawkesArrival(spread0=float(cfg.spread0), seed=int(cfg.seed))
    if cfg.avg_spread_override is not None:
        arrival.avgSpread = float(cfg.avg_spread_override)
    arrival.expApprox = bool(cfg.use_exp_approx)
    if cfg.use_exp_approx:
        arrival.TAU = 10
    _apply_runtime_patches(arrival)
    return arrival


def project_konark_to_fx_topofbook(
    times: np.ndarray, type_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the deterministic Phi_beta projection.

    Parameters
    ----------
    times : np.ndarray (n,)
        Konark-native event times.
    type_indices : np.ndarray (n,)
        Konark-native type indices in [0, 12).

    Returns
    -------
    fx_times : np.ndarray
    fx_types : np.ndarray
        Subset of `times` and projected FX 4-type indices for the events
        that map onto a top-of-book change.
    """
    times = np.asarray(times, dtype=np.float64)
    type_indices = np.asarray(type_indices, dtype=np.int64)
    if times.shape != type_indices.shape:
        raise ValueError(
            f"times shape {times.shape} != type_indices shape {type_indices.shape}"
        )
    name_table = list(KONARK_NATIVE_TYPES)
    fx_t = []
    fx_k = []
    for t, k in zip(times, type_indices):
        proj = PHI_BETA_PROJECTION[name_table[int(k)]]
        if proj is None:
            continue
        fx_t.append(float(t))
        fx_k.append(int(proj))
    return (
        np.asarray(fx_t, dtype=np.float64),
        np.asarray(fx_k, dtype=np.int64),
    )


def simulate_d2_konark(cfg: D2KonarkConfig, horizon_seconds: float) -> D2RunResult:
    """Run a Konark Hawkes simulation up to `horizon_seconds` and project
    the latent stream onto the FX 4-type top-of-book alphabet.

    Returns a `D2RunResult` with both the latent (Konark-native) and the
    observed (FX-projected) streams.
    """
    if horizon_seconds <= 0.0:
        raise ValueError(
            f"horizon_seconds must be > 0; got {horizon_seconds}"
        )
    arrival = build_konark_arrival(cfg)
    t0 = time.time()
    latent_times: List[float] = []
    latent_types: List[int] = []
    while True:
        result = arrival.get_nextarrival(timelimit=float(horizon_seconds))
        if result is None:
            break
        ev_time, side, order_type, level, _size = result
        if "Ask" in level:
            side_label = "Ask"
        else:
            side_label = "Bid"
        if order_type == "lo":
            if level == "Ask_L1":
                k = KONARK_TYPE_TO_INDEX["lo_top_Ask"]
            elif level == "Ask_L2":
                k = KONARK_TYPE_TO_INDEX["lo_deep_Ask"]
            elif level == "Ask_inspread":
                k = KONARK_TYPE_TO_INDEX["lo_inspread_Ask"]
            elif level == "Bid_L1":
                k = KONARK_TYPE_TO_INDEX["lo_top_Bid"]
            elif level == "Bid_L2":
                k = KONARK_TYPE_TO_INDEX["lo_deep_Bid"]
            elif level == "Bid_inspread":
                k = KONARK_TYPE_TO_INDEX["lo_inspread_Bid"]
            else:
                continue
        elif order_type == "mo":
            if level == "Ask_MO":
                k = KONARK_TYPE_TO_INDEX["mo_Ask"]
            elif level == "Bid_MO":
                k = KONARK_TYPE_TO_INDEX["mo_Bid"]
            else:
                continue
        elif order_type == "co":
            if level == "Ask_L1":
                k = KONARK_TYPE_TO_INDEX["co_top_Ask"]
            elif level == "Ask_L2":
                k = KONARK_TYPE_TO_INDEX["co_deep_Ask"]
            elif level == "Bid_L1":
                k = KONARK_TYPE_TO_INDEX["co_top_Bid"]
            elif level == "Bid_L2":
                k = KONARK_TYPE_TO_INDEX["co_deep_Bid"]
            else:
                continue
        else:
            continue
        latent_times.append(float(ev_time))
        latent_types.append(int(k))
        if ev_time >= horizon_seconds:
            break
    wall = time.time() - t0

    konark_times = np.asarray(latent_times, dtype=np.float64)
    konark_types = np.asarray(latent_types, dtype=np.int64)
    fx_times, fx_types = project_konark_to_fx_topofbook(konark_times, konark_types)

    konark_counts: Dict[str, int] = {name: 0 for name in KONARK_NATIVE_TYPES}
    for k in konark_types:
        konark_counts[KONARK_NATIVE_TYPES[int(k)]] += 1
    fx_counts: Dict[str, int] = {name: 0 for name in FX_TOPOFBOOK_TYPES}
    for k in fx_types:
        fx_counts[FX_TOPOFBOOK_TYPES[int(k)]] += 1

    n_latent = int(konark_times.shape[0])
    n_observed = int(fx_times.shape[0])
    cens = (
        float(n_latent - n_observed) / float(n_latent) if n_latent > 0 else 0.0
    )

    return D2RunResult(
        horizon_seconds=float(horizon_seconds),
        konark_native_times=konark_times,
        konark_native_types=konark_types,
        konark_native_type_counts=konark_counts,
        fx_topofbook_times=fx_times,
        fx_topofbook_types=fx_types,
        fx_topofbook_type_counts=fx_counts,
        n_konark_latent=n_latent,
        n_fx_observed=n_observed,
        censoring_fraction=cens,
        wall_seconds=float(wall),
    )
