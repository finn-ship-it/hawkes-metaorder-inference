"""
d2_konark_latent_book.py — D2 latent full-book Hawkes process under explicit C_beta.

Reframes D2 from prompt 07's "Konark-projected stream calibrated to D1
composition" (which was a category error: D1 is itself a stylised
simulator, not the right observable target) to **Konark as a latent
full-book Hawkes process under an explicit observation operator C_beta**.
The 4-type FX top-of-book stream is the *observable*, NOT the calibration
target.

The dissertation studies meta-order / informed-flow inference under
censored observation. D2 is the only one of the three backends
(D1 / D2 / ABIDES) that exposes a non-trivial censoring object AND
simulator-side latent truth.

This module supersedes the user-facing role of `d2_konark_backend.py`
(preserved at `d2_konark_backend_legacy.py` for audit). The Konark wrap,
the source-of-truth verification, and the two runtime patches from
prompt 07 are inherited unchanged.

Inherited from prompt 07
------------------------
- Konark source path: `Konarks-github repo/HawkesRLTrading/src/Stochastic_Processes/Arrival_Models.HawkesArrival`. Read-only; no Konark source edits.
- Two runtime patches monkey-patched on the live `HawkesArrival` instance via `types.MethodType` and re-applied here:
    - **Patch 1 (numpy-2.x scalar conversion)** at `Arrival_Models.py:~335`: original `self.lamb = float(sum(decays))` fails on numpy 2.x because `sum(decays)` (Python builtin over a (12,1) array) returns a (1,)-shaped array; patched to `self.lamb = float(np.sum(decays))`.
    - **Patch 2 (`self.left` reset post-truncation)**: original missing `self.left = 0` after the in-loop `self.timeseries = self.timeseries[self.left:]` truncation; `self.left` grows past the truncated length and the next truncation wipes events. Patched with `self.left = 0` after the slice.

Reference C_beta projection rule
--------------------------------

| Konark-native type | Projects to    | Censored? |
|--------------------|----------------|-----------|
| `lo_top_Bid`       | BID_UP         | no        |
| `co_top_Bid`       | BID_DOWN       | no        |
| `lo_top_Ask`       | ASK_DOWN       | no        |
| `co_top_Ask`       | ASK_UP         | no        |
| `mo_Ask`           | ASK_UP         | no        |
| `mo_Bid`           | BID_DOWN       | no        |
| `lo_inspread_Bid`  | BID_UP         | no        |
| `lo_inspread_Ask`  | ASK_DOWN       | no        |
| `lo_deep_Ask`      | (not projected)| YES       |
| `co_deep_Ask`      | (not projected)| YES       |
| `lo_deep_Bid`      | (not projected)| YES       |
| `co_deep_Bid`      | (not projected)| YES       |

Note: this projection differs from the prompt-07 `d2_konark_backend.py`
which only projected the four `mo_*` and `lo_inspread_*` types. The 07b
reframe **un-censors** `lo_top_*` and `co_top_*` events, treating each
as a directional bid/ask side movement at the top, because real-FX
top-of-book observation does see top-of-book limit additions and
cancellations as price changes (or as queue-depth pressure that
correlates with subsequent price changes). Deep events remain censored.

Censoring inventory (hidden by C_beta)
--------------------------------------

- order IDs;
- depth-level (deep events not projected; top vs in-spread collapse into the same bid/ask side movement);
- parent / regime labels (the meta-order `regime_truth` is exposed only on `latent_events`, NOT on `observed_events`);
- in-spread vs cancellation distinction (collapsed into a directional bid/ask side movement);
- volume / size (dimension absent from `observed_events`);
- queue position.

Surviving C_beta
----------------

- event time;
- projected event type (BID_UP / BID_DOWN / ASK_UP / ASK_DOWN).

Volume is hidden ENTIRELY. The observable is event-time top-of-book
direction without volume. Delayed-volume / fill-time / toxicity at
dealer-side delayed information is named as PhD continuation and is
NOT in scope for 07b.

Meta-order regime injection
---------------------------

`MetaOrderRegimeWindow` is the controlled experimental factor. During
a window:
- the `affected_native_types` mu rates are scaled by `intensity_multiplier`;
- every event in the window carries `regime_truth = (window_id, side, in_window=True)`;
- the regime label is exposed on `latent_events` only; `observed_events`
  has labels stripped (the operator C_beta destroys them, by design).

Default configuration:
- 30-second windows starting every 150 seconds (20% duty cycle);
- direction alternates: window 1 = Bid, window 2 = Ask, ...;
- `intensity_multiplier = 2.0`;
- `affected_native_types = [mo_<side>, lo_inspread_<side>]`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    "D2LatentBookConfig",
    "D2LatentBookResult",
    "MetaOrderRegimeWindow",
    "LatentEvent",
    "ObservedEvent",
    "default_meta_order_regime_windows",
    "simulate_d2_latent_book",
    "project_latent_to_observed",
    "ensure_konark_on_path",
    "build_konark_arrival",
    "KONARK_NATIVE_TYPES",
    "KONARK_TYPE_TO_INDEX",
    "FX_TOPOFBOOK_TYPES",
    "PHI_BETA_PROJECTION",
    "KONARK_RUNTIME_PATCHES",
    "INTENSITY_MULTIPLIER_TOLERANCE",
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

# FX 4-type top-of-book alphabet (matches the FX-native simulator's
# EventType enum BID_UP=0, BID_DOWN=1, ASK_UP=2, ASK_DOWN=3).
FX_TOPOFBOOK_TYPES: Tuple[str, ...] = (
    "BID_UP",
    "BID_DOWN",
    "ASK_UP",
    "ASK_DOWN",
)

# Phi_beta projection (07b reframe; supersedes prompt 07's narrower
# projection). See module docstring for full table.
PHI_BETA_PROJECTION: Dict[str, Optional[int]] = {
    "lo_top_Bid": 0,        # BID_UP
    "co_top_Bid": 1,        # BID_DOWN
    "lo_top_Ask": 3,        # ASK_DOWN
    "co_top_Ask": 2,        # ASK_UP
    "mo_Ask": 2,            # ASK_UP
    "mo_Bid": 1,            # BID_DOWN
    "lo_inspread_Bid": 0,   # BID_UP
    "lo_inspread_Ask": 3,   # ASK_DOWN
    "lo_deep_Ask": None,
    "co_deep_Ask": None,
    "lo_deep_Bid": None,
    "co_deep_Bid": None,
}


# ---------------------------------------------------------------------------
# Konark runtime patches (inherited from prompt 07)
# ---------------------------------------------------------------------------


KONARK_RUNTIME_PATCHES: Tuple[Dict[str, str], ...] = (
    {
        "target": "HawkesArrival.thinningOgataIS2 line ~335",
        "issue": (
            "Original `self.lamb = float(sum(decays))` fails on numpy 2.x "
            "because `sum(decays)` (Python builtin over a (12, 1) array) "
            "returns a (1,)-shaped numpy array, and `float(...)` on a "
            "non-scalar array raises TypeError."
        ),
        "patch": (
            "Replace with `self.lamb = float(np.sum(decays))`. Applied at "
            "instance level via types.MethodType. Konark source NOT "
            "modified on disk (read-only)."
        ),
        "inherited_from_prompt": "07",
    },
    {
        "target": "HawkesArrival.thinningOgataIS2 timeseries-truncation block (post-append)",
        "issue": (
            "Konark advances `self.left` past stale points but never "
            "resets it after the timeseries is truncated. Once self.left "
            "grows to the post-truncation length, the next truncation "
            "wipes the entire timeseries; the next `get_nextarrival` call "
            "hits IndexError on `self.timeseries[-1]`."
        ),
        "patch": (
            "Append `self.left = 0` immediately after the truncation slice. "
            "Applied at instance level. Konark source NOT modified on disk "
            "(read-only)."
        ),
        "inherited_from_prompt": "07",
    },
)


INTENSITY_MULTIPLIER_TOLERANCE: float = 1e-9


# ---------------------------------------------------------------------------
# Path resolution + Konark loading
# ---------------------------------------------------------------------------


def _resolve_konark_repo_root() -> str:
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
    root = _resolve_konark_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


# ---------------------------------------------------------------------------
# Patched thinning method (inherited from prompt 07)
# ---------------------------------------------------------------------------


def _patched_thinning_ogata_is2(self, T=None):
    """Drop-in replacement for `HawkesArrival.thinningOgataIS2` with both
    inherited prompt-07 runtime patches applied. Otherwise byte-equivalent
    to Konark's original; see KONARK_RUNTIME_PATCHES for details."""
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
    arrival.thinningOgataIS2 = types.MethodType(
        _patched_thinning_ogata_is2, arrival
    )


# ---------------------------------------------------------------------------
# MetaOrderRegimeWindow + default factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaOrderRegimeWindow:
    """One injected meta-order regime window on the latent Konark stream.

    During [start_t, end_t), the baseline mu rates of `affected_native_types`
    are scaled by `intensity_multiplier`. Every latent event in the window
    carries `regime_truth = (window_id, direction, True)`; events outside
    any window carry `(None, None, False)`. The C_beta projection strips
    these labels from `observed_events`.
    """

    window_id: int
    start_t: float
    end_t: float
    direction: str  # "Bid" or "Ask"
    intensity_multiplier: float
    affected_native_types: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.end_t <= self.start_t:
            raise ValueError(
                f"end_t ({self.end_t}) must be > start_t ({self.start_t}) "
                f"for window {self.window_id}"
            )
        if self.direction not in {"Bid", "Ask"}:
            raise ValueError(
                f"direction must be 'Bid' or 'Ask'; got {self.direction!r}"
            )
        if self.intensity_multiplier <= 0.0:
            raise ValueError(
                f"intensity_multiplier must be > 0; "
                f"got {self.intensity_multiplier}"
            )
        for t in self.affected_native_types:
            if t not in KONARK_TYPE_TO_INDEX:
                raise ValueError(
                    f"affected_native_types contains unknown type {t!r} "
                    f"for window {self.window_id}"
                )

    def covers(self, t: float) -> bool:
        return self.start_t <= float(t) < self.end_t


def default_meta_order_regime_windows(
    horizon_seconds: float,
    *,
    window_seconds: float = 30.0,
    period_seconds: float = 150.0,
    intensity_multiplier: float = 2.0,
) -> Tuple[MetaOrderRegimeWindow, ...]:
    """Build the spec-default meta-order regime windows: 30-second windows
    starting every 150 seconds (20% duty cycle), alternating Bid/Ask
    direction starting with Bid, intensity_multiplier=2.0 on the side's
    `mo_*` and `lo_inspread_*` channels."""
    if horizon_seconds <= 0.0:
        raise ValueError(f"horizon_seconds must be > 0; got {horizon_seconds}")
    windows: List[MetaOrderRegimeWindow] = []
    wid = 0
    start = 0.0
    while start + window_seconds <= horizon_seconds:
        wid += 1
        side = "Bid" if (wid % 2 == 1) else "Ask"
        affected = (f"mo_{side}", f"lo_inspread_{side}")
        windows.append(
            MetaOrderRegimeWindow(
                window_id=wid,
                start_t=float(start),
                end_t=float(start + window_seconds),
                direction=side,
                intensity_multiplier=float(intensity_multiplier),
                affected_native_types=affected,
            )
        )
        start += period_seconds
    return tuple(windows)


# ---------------------------------------------------------------------------
# LatentEvent / ObservedEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentEvent:
    """One pre-projection Konark-native event with full latent metadata."""

    time: float
    konark_native_type: str
    side: str
    depth_level: str  # "deep" / "top" / "in_spread"
    regime_window_id: Optional[int]
    regime_side: Optional[str]
    in_window: bool
    injected_intensity_multiplier: float


@dataclass(frozen=True)
class ObservedEvent:
    """One post-projection FX 4-type event. Regime labels stripped."""

    time: float
    observed_type: str  # one of FX_TOPOFBOOK_TYPES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D2LatentBookConfig:
    """Configuration for the 07b D2 latent full-book simulator.

    Fields
    ------
    spread0 : float
        Initial spread (in tick-size units) passed to Konark's HawkesArrival.
        Konark gates the in-spread channels off when `100 * round(spread, 2) < 2`,
        so spread0 must be >= 0.02 for in-spread events to fire.
    seed : int
        RNG seed for both Konark's internal np.random and meta-order
        regime placement.
    use_exp_approx : bool
        Konark's expApprox flag (default True). True triggers the
        exponential-approximation kernel path with TAU=10.
    inspread_bid_over_ask_mu_ratio : float
        Ratio applied to Konark's mu vector to rebalance
        `mu[lo_inspread_Bid] / mu[lo_inspread_Ask]`. Konark's default is
        ~3:1 (asymmetric); 1.0 makes it symmetric (the spec default for
        the highest-leverage knob).
    per_native_type_mu_multipliers : tuple[(str, float), ...]
        Per-name multiplier overlay applied to Konark's mu vector
        AFTER the inspread rebalance. Empty tuple = no overlay. Used by
        the calibration search as a secondary knob.
    meta_order_regime_windows : tuple[MetaOrderRegimeWindow, ...] or None
        Injected experimental factor on the latent stream. If None,
        `default_meta_order_regime_windows(horizon_seconds)` is used at
        simulation time.
    """

    spread0: float = 0.03
    seed: int = 1
    use_exp_approx: bool = True
    inspread_bid_over_ask_mu_ratio: float = 1.0
    per_native_type_mu_multipliers: Tuple[Tuple[str, float], ...] = ()
    meta_order_regime_windows: Optional[Tuple[MetaOrderRegimeWindow, ...]] = None

    def __post_init__(self) -> None:
        if self.spread0 <= 0.0:
            raise ValueError(f"spread0 must be > 0; got {self.spread0}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0; got {self.seed}")
        if self.inspread_bid_over_ask_mu_ratio <= 0.0:
            raise ValueError(
                f"inspread_bid_over_ask_mu_ratio must be > 0; "
                f"got {self.inspread_bid_over_ask_mu_ratio}"
            )
        for name, mult in self.per_native_type_mu_multipliers:
            if name not in KONARK_TYPE_TO_INDEX:
                raise ValueError(
                    f"per_native_type_mu_multipliers contains unknown "
                    f"Konark-native type {name!r}"
                )
            if mult <= 0.0:
                raise ValueError(
                    f"per_native_type_mu_multipliers entry for {name!r} "
                    f"has non-positive multiplier {mult}"
                )

    def to_json(self) -> str:
        windows = (
            None
            if self.meta_order_regime_windows is None
            else [
                {
                    "window_id": w.window_id,
                    "start_t": float(w.start_t),
                    "end_t": float(w.end_t),
                    "direction": w.direction,
                    "intensity_multiplier": float(w.intensity_multiplier),
                    "affected_native_types": list(w.affected_native_types),
                }
                for w in self.meta_order_regime_windows
            ]
        )
        return json.dumps(
            {
                "spread0": float(self.spread0),
                "seed": int(self.seed),
                "use_exp_approx": bool(self.use_exp_approx),
                "inspread_bid_over_ask_mu_ratio": float(
                    self.inspread_bid_over_ask_mu_ratio
                ),
                "per_native_type_mu_multipliers": [
                    [n, float(m)] for n, m in self.per_native_type_mu_multipliers
                ],
                "meta_order_regime_windows": windows,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "D2LatentBookConfig":
        d = json.loads(s)
        windows = d.get("meta_order_regime_windows")
        if windows is not None:
            windows = tuple(
                MetaOrderRegimeWindow(
                    window_id=int(w["window_id"]),
                    start_t=float(w["start_t"]),
                    end_t=float(w["end_t"]),
                    direction=str(w["direction"]),
                    intensity_multiplier=float(w["intensity_multiplier"]),
                    affected_native_types=tuple(w["affected_native_types"]),
                )
                for w in windows
            )
        return cls(
            spread0=float(d["spread0"]),
            seed=int(d["seed"]),
            use_exp_approx=bool(d.get("use_exp_approx", True)),
            inspread_bid_over_ask_mu_ratio=float(
                d.get("inspread_bid_over_ask_mu_ratio", 1.0)
            ),
            per_native_type_mu_multipliers=tuple(
                (str(n), float(m))
                for n, m in d.get("per_native_type_mu_multipliers", [])
            ),
            meta_order_regime_windows=windows,
        )


# ---------------------------------------------------------------------------
# D2 latent-book result
# ---------------------------------------------------------------------------


@dataclass
class D2LatentBookResult:
    """Result of one D2 latent-book simulation under explicit C_beta.

    Fields
    ------
    horizon_seconds : float
    latent_events : List[LatentEvent]
        Full Konark-native event stream with per-event regime label.
        Source of truth for latent state.
    pre_projection_native_counts : Dict[str, int]
        Per-Konark-type count over `latent_events`.
    observed_events : List[ObservedEvent]
        Post-projection FX 4-type stream. Regime labels stripped.
    post_projection_fx_counts : Dict[str, int]
        Per-FX-type count over `observed_events`.
    n_latent : int
    n_observed : int
    censoring_fraction : float
    regime_truth_aggregate : Dict[str, ...]
        Aggregate statistics about the regime truth (number of windows,
        per-window event counts, mean intensity multiplier inside windows).
        The full per-event regime labels live on `latent_events` only.
    meta_order_regime_windows : Tuple[MetaOrderRegimeWindow, ...]
    config_summary : Dict
    runtime_patches : List[Dict]
    wall_seconds : float
    """

    horizon_seconds: float
    latent_events: List[LatentEvent]
    pre_projection_native_counts: Dict[str, int]
    observed_events: List[ObservedEvent]
    post_projection_fx_counts: Dict[str, int]
    n_latent: int
    n_observed: int
    censoring_fraction: float
    regime_truth_aggregate: Dict
    meta_order_regime_windows: Tuple[MetaOrderRegimeWindow, ...]
    config_summary: Dict
    runtime_patches: List[Dict]
    wall_seconds: float

    @property
    def latent_gt_observed(self) -> bool:
        return self.n_latent > self.n_observed

    @property
    def censoring_pct(self) -> float:
        return 100.0 * self.censoring_fraction

    @property
    def latent_minus_observed_pct(self) -> float:
        if self.n_observed == 0:
            return float("inf")
        return 100.0 * (self.n_latent - self.n_observed) / self.n_observed


# ---------------------------------------------------------------------------
# build_konark_arrival
# ---------------------------------------------------------------------------


def _apply_mu_overrides(arrival, cfg: D2LatentBookConfig) -> Dict[str, float]:
    """Apply Konark mu rebalance + per-native-type multipliers in-place on
    the live HawkesArrival instance. Returns the resulting per-name mu
    dict (for audit logging in the artefact).

    Konark's `kernelparams[1]` holds a (12, 1)-shaped baseline mu vector;
    we operate on it via 0-d scalar extraction (mu[i, 0]) and write back
    to the same slot to preserve shape."""
    mu = arrival.kernelparams[1].astype(np.float64)
    inspread_bid_idx = KONARK_TYPE_TO_INDEX["lo_inspread_Bid"]
    inspread_ask_idx = KONARK_TYPE_TO_INDEX["lo_inspread_Ask"]
    cur_ask = float(mu[inspread_ask_idx, 0])
    if cur_ask > 0.0:
        new_bid = cfg.inspread_bid_over_ask_mu_ratio * cur_ask
        mu[inspread_bid_idx, 0] = new_bid
    for name, mult in cfg.per_native_type_mu_multipliers:
        idx = KONARK_TYPE_TO_INDEX[name]
        mu[idx, 0] = float(mu[idx, 0]) * float(mult)
    arrival.kernelparams = (arrival.kernelparams[0], mu)
    return {
        name: float(mu[KONARK_TYPE_TO_INDEX[name], 0])
        for name in KONARK_NATIVE_TYPES
    }


def build_konark_arrival(cfg: D2LatentBookConfig):
    """Construct a Konark HawkesArrival with `generatefakeparams()` defaults,
    apply the runtime patches, and apply the mu-vector overrides."""
    ensure_konark_on_path()
    from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import (
        HawkesArrival,
    )

    np.random.seed(int(cfg.seed))
    arrival = HawkesArrival(spread0=float(cfg.spread0), seed=int(cfg.seed))
    arrival.expApprox = bool(cfg.use_exp_approx)
    if cfg.use_exp_approx:
        arrival.TAU = 10
    _apply_runtime_patches(arrival)
    mu_after = _apply_mu_overrides(arrival, cfg)
    return arrival, mu_after


# ---------------------------------------------------------------------------
# Per-event metadata helpers
# ---------------------------------------------------------------------------


def _depth_level_of(name: str) -> str:
    if "deep" in name:
        return "deep"
    if "inspread" in name:
        return "in_spread"
    return "top"


def _side_of(name: str) -> str:
    return "Ask" if "Ask" in name else "Bid"


def _regime_window_at(
    t: float,
    windows: Tuple[MetaOrderRegimeWindow, ...],
) -> Optional[MetaOrderRegimeWindow]:
    for w in windows:
        if w.covers(t):
            return w
    return None


def _intensity_multiplier_for_event(
    konark_native_type: str,
    t: float,
    windows: Tuple[MetaOrderRegimeWindow, ...],
) -> Tuple[float, Optional[MetaOrderRegimeWindow]]:
    """Return (multiplier_in_force, covering_window) for the given event.
    The multiplier is 1.0 outside any window OR for a type that the active
    window does not affect; otherwise it is `window.intensity_multiplier`."""
    w = _regime_window_at(t, windows)
    if w is None:
        return 1.0, None
    if konark_native_type in w.affected_native_types:
        return float(w.intensity_multiplier), w
    return 1.0, w


# ---------------------------------------------------------------------------
# Konark side-and-level → Konark-native type
# ---------------------------------------------------------------------------


def _classify_konark_event(side: str, order_type: str, level: str) -> Optional[str]:
    """Map (side, order_type, level) from Konark's `get_nextarrival` output
    to one of the 12 Konark-native type names. Returns None for events
    that do not fit the standard taxonomy."""
    if order_type == "lo":
        if level == "Ask_L1":
            return "lo_top_Ask"
        if level == "Ask_L2":
            return "lo_deep_Ask"
        if level == "Ask_inspread":
            return "lo_inspread_Ask"
        if level == "Bid_L1":
            return "lo_top_Bid"
        if level == "Bid_L2":
            return "lo_deep_Bid"
        if level == "Bid_inspread":
            return "lo_inspread_Bid"
    elif order_type == "mo":
        if level == "Ask_MO":
            return "mo_Ask"
        if level == "Bid_MO":
            return "mo_Bid"
    elif order_type == "co":
        if level == "Ask_L1":
            return "co_top_Ask"
        if level == "Ask_L2":
            return "co_deep_Ask"
        if level == "Bid_L1":
            return "co_top_Bid"
        if level == "Bid_L2":
            return "co_deep_Bid"
    return None


# ---------------------------------------------------------------------------
# Public API: simulate, project
# ---------------------------------------------------------------------------


def project_latent_to_observed(
    latent_events: List[LatentEvent],
) -> List[ObservedEvent]:
    """Apply the deterministic Phi_beta projection to a latent-event list.

    Censored events (deep events) are dropped. Regime labels are stripped
    from the output by construction (`ObservedEvent` has no regime field).
    """
    out: List[ObservedEvent] = []
    for ev in latent_events:
        proj = PHI_BETA_PROJECTION.get(ev.konark_native_type)
        if proj is None:
            continue
        out.append(
            ObservedEvent(
                time=float(ev.time),
                observed_type=FX_TOPOFBOOK_TYPES[int(proj)],
            )
        )
    return out


def _meta_order_acceptance_oracle(
    konark_native_type: str,
    t: float,
    windows: Tuple[MetaOrderRegimeWindow, ...],
    rng: np.random.Generator,
) -> bool:
    """Bernoulli acceptance step that converts Konark's baseline-only
    thinning into an event stream consistent with the regime-window
    intensity uplift. Returns True if the event should be retained, False
    if it should be re-drawn (i.e., dropped here and the next Konark
    candidate consulted).

    The acceptance probability is `1 / multiplier` if the event is in a
    window AND outside the affected types (no uplift, but the underlying
    Konark stream over-represents this type relative to the window's
    intent — so accept with probability 1.0 here, never down-thin); and
    `1.0` if multiplier > 1 on the affected types (uplift from baseline,
    already represented by augmenting the rate via thinning candidates
    drawn separately). In practice we apply uplift via post-hoc REJECTION
    SAMPLING on a baseline Konark stream: in-window candidates of an
    affected type are accepted with probability 1.0; out-of-window
    candidates are accepted with probability 1/multiplier when the
    underlying stream is re-cast as the uplifted process.

    NOTE: this oracle is NOT used in the current implementation; we
    instead emulate the uplift by SUPPLEMENTAL THINNING (extra in-window
    candidates drawn from a homogeneous Poisson process at the
    incremental rate). Kept here as documentation of the alternative
    rejection-sampling path.
    """
    return True


def simulate_d2_latent_book(
    config: D2LatentBookConfig,
    horizon_seconds: float,
) -> D2LatentBookResult:
    """Run a Konark Hawkes simulation up to `horizon_seconds` and return
    the latent (Konark-native) + observed (FX-projected) streams.

    Meta-order regime injection is applied as **supplemental thinning**:
    Konark's baseline simulator runs to produce the unaugmented latent
    stream; for each window, additional events of the affected types are
    drawn from a homogeneous Poisson process at the incremental rate
    `(intensity_multiplier - 1) * baseline_rate` and merged into the
    latent stream. This preserves the Hawkes self-excitation structure
    of the baseline while giving the regime an experimentally controlled
    intensity uplift on the affected channels during the window.

    Returns a `D2LatentBookResult` with both latent and observed streams.
    """
    if horizon_seconds <= 0.0:
        raise ValueError(
            f"horizon_seconds must be > 0; got {horizon_seconds}"
        )

    # 1. Resolve regime windows.
    windows = config.meta_order_regime_windows
    if windows is None:
        windows = default_meta_order_regime_windows(horizon_seconds)

    # 2. Construct Konark arrival with patches and mu overrides.
    arrival, mu_after = build_konark_arrival(config)

    t0 = time.time()

    # 3. Run Konark's baseline thinning loop to produce the unaugmented
    #    latent stream.
    baseline_latent: List[Tuple[float, str]] = []
    while True:
        result = arrival.get_nextarrival(timelimit=float(horizon_seconds))
        if result is None:
            break
        ev_time, side, order_type, level, _size = result
        ktype = _classify_konark_event(side, order_type, level)
        if ktype is None:
            continue
        baseline_latent.append((float(ev_time), ktype))
        if ev_time >= horizon_seconds:
            break

    # 4. Supplemental thinning for the regime windows: at each window,
    #    add Poisson-distributed extra events of the affected types at
    #    the incremental rate. Use the baseline empirical rate per type
    #    over the FULL horizon as an estimator for `baseline_rate`.
    rng = np.random.default_rng(int(config.seed) + 7919)
    per_type_baseline_count: Dict[str, int] = {n: 0 for n in KONARK_NATIVE_TYPES}
    for _t, ktype in baseline_latent:
        per_type_baseline_count[ktype] += 1
    baseline_rate_per_type: Dict[str, float] = {
        n: per_type_baseline_count[n] / horizon_seconds for n in KONARK_NATIVE_TYPES
    }
    suppl: List[Tuple[float, str]] = []
    for w in windows:
        delta = w.intensity_multiplier - 1.0
        if delta <= 0.0:
            continue
        for ktype in w.affected_native_types:
            base_rate = baseline_rate_per_type.get(ktype, 0.0)
            extra_rate = delta * base_rate
            window_dt = w.end_t - w.start_t
            expected = extra_rate * window_dt
            if expected <= 0.0:
                continue
            n_extra = int(rng.poisson(expected))
            if n_extra <= 0:
                continue
            extra_times = rng.uniform(w.start_t, w.end_t, size=n_extra)
            for t in extra_times:
                suppl.append((float(t), ktype))

    # 5. Merge baseline + supplemental, sort by time.
    full = baseline_latent + suppl
    full.sort(key=lambda x: x[0])

    # 6. Build LatentEvent list with per-event regime metadata.
    latent_events: List[LatentEvent] = []
    pre_counts: Dict[str, int] = {n: 0 for n in KONARK_NATIVE_TYPES}
    in_window_count = 0
    multiplier_records: Dict[int, List[float]] = {w.window_id: [] for w in windows}
    per_window_event_count: Dict[int, int] = {w.window_id: 0 for w in windows}
    for t, ktype in full:
        mult, w = _intensity_multiplier_for_event(ktype, t, windows)
        ev = LatentEvent(
            time=float(t),
            konark_native_type=ktype,
            side=_side_of(ktype),
            depth_level=_depth_level_of(ktype),
            regime_window_id=(w.window_id if w is not None else None),
            regime_side=(w.direction if w is not None else None),
            in_window=bool(w is not None),
            injected_intensity_multiplier=float(mult),
        )
        latent_events.append(ev)
        pre_counts[ktype] += 1
        if w is not None:
            in_window_count += 1
            per_window_event_count[w.window_id] += 1
            multiplier_records[w.window_id].append(float(mult))

    wall = time.time() - t0

    # 7. Project to observed (FX 4-type), regime labels stripped.
    observed_events = project_latent_to_observed(latent_events)
    post_counts: Dict[str, int] = {n: 0 for n in FX_TOPOFBOOK_TYPES}
    for ev in observed_events:
        post_counts[ev.observed_type] += 1

    n_latent = len(latent_events)
    n_observed = len(observed_events)
    cens = (n_latent - n_observed) / n_latent if n_latent > 0 else 0.0

    # 8. Aggregate regime truth.
    regime_truth_aggregate = {
        "n_windows_total": len(windows),
        "n_events_in_any_window": int(in_window_count),
        "n_events_outside_windows": int(n_latent - in_window_count),
        "in_window_event_fraction": (
            float(in_window_count) / float(n_latent) if n_latent > 0 else 0.0
        ),
        "per_window_event_count": {
            int(wid): int(c) for wid, c in per_window_event_count.items()
        },
        "mean_intensity_multiplier_in_windows": {
            int(wid): (
                float(np.mean(rec)) if rec else 0.0
            )
            for wid, rec in multiplier_records.items()
        },
    }

    config_summary = {
        "spread0": float(config.spread0),
        "seed": int(config.seed),
        "use_exp_approx": bool(config.use_exp_approx),
        "inspread_bid_over_ask_mu_ratio": float(
            config.inspread_bid_over_ask_mu_ratio
        ),
        "per_native_type_mu_multipliers": [
            [n, float(m)] for n, m in config.per_native_type_mu_multipliers
        ],
        "mu_vector_after_overrides": mu_after,
    }
    runtime_patches = [{k: v for k, v in p.items()} for p in KONARK_RUNTIME_PATCHES]

    return D2LatentBookResult(
        horizon_seconds=float(horizon_seconds),
        latent_events=latent_events,
        pre_projection_native_counts=pre_counts,
        observed_events=observed_events,
        post_projection_fx_counts=post_counts,
        n_latent=int(n_latent),
        n_observed=int(n_observed),
        censoring_fraction=float(cens),
        regime_truth_aggregate=regime_truth_aggregate,
        meta_order_regime_windows=tuple(windows),
        config_summary=config_summary,
        runtime_patches=runtime_patches,
        wall_seconds=float(wall),
    )
