"""Build artefact: dissertation_artifacts/data/eurusd_observed_composition.json.

Caches the real-EURUSD primary-day (2021-07-20) observed composition as
the calibration target for prompt 07b's D2 latent-book reframe. The
cache is consumed by `build_d2_latent_book_calibration.py`.

Per the prompt-07b spec:
- N = 97,952 events (must reproduce the existing primary-day count)
- fit_window_seconds = 86,400 s
- four-type composition fractions sum to 1.0
- bid/ask in-spread ratio: empirical estimate if reconstructable from
  the source data, else null with a documented reason
- spread / mid diagnostics: same conditional rule
- data provenance: source CSV path, jitter scheme (random_ms, seed
  20210720), column convention.

This is a one-shot diagnostic cache; the file is produced once per
prompt 07b run and is not re-emitted by the inference runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PARQUET = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "eurusd_events.parquet"
OUT_PATH = HERE / "data" / "eurusd_observed_composition.json"

PRIMARY_DAY = "2021-07-20"
PRIMARY_4TYPE = ("bid_up", "bid_down", "ask_up", "ask_down")
JITTER_SEED = 20210720
FIT_WINDOW_SECONDS = 86_400.0


def _load_day_events(date_str: str):
    type_to_idx = {t: i for i, t in enumerate(PRIMARY_4TYPE)}
    df = pd.read_parquet(
        PARQUET,
        columns=["timestamp", "event_type"],
        filters=[("event_type", "in", list(PRIMARY_4TYPE))],
    )
    day = pd.Timestamp(date_str)
    mask = (df["timestamp"] >= day) & (df["timestamp"] < day + pd.Timedelta(days=1))
    df = df.loc[mask].copy()
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    t = (df["timestamp"] - day).dt.total_seconds().to_numpy(dtype=np.float64)
    typ = df["event_type"].map(type_to_idx).to_numpy(dtype=np.int64)
    return t, typ


def _apply_random_ms_jitter(t: np.ndarray, typ: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    t_j = t + rng.uniform(0.0, 1.0e-3, size=t.shape[0])
    order = np.argsort(t_j, kind="stable")
    return t_j[order], typ[order]


def main():
    print(f"=== EURUSD observed composition cache (primary day {PRIMARY_DAY}) ===")
    t_raw, typ_raw = _load_day_events(PRIMARY_DAY)
    n_events = int(t_raw.size)
    print(f"  loaded N = {n_events:,}")

    t, typ = _apply_random_ms_jitter(t_raw, typ_raw, JITTER_SEED)
    counts = np.bincount(typ, minlength=4)
    fractions = counts / max(1, counts.sum())

    payload = {
        "schema_version": "1.0",
        "source": "EURUSD primary-day events (2021-07-20)",
        "n_events": n_events,
        "fit_window_seconds": float(FIT_WINDOW_SECONDS),
        "event_rate_per_second": float(n_events) / FIT_WINDOW_SECONDS,
        "composition_fractions": {
            "bid_up_fraction": float(fractions[0]),
            "bid_down_fraction": float(fractions[1]),
            "ask_up_fraction": float(fractions[2]),
            "ask_down_fraction": float(fractions[3]),
        },
        "per_type_event_count": {
            "bid_up": int(counts[0]),
            "bid_down": int(counts[1]),
            "ask_up": int(counts[2]),
            "ask_down": int(counts[3]),
        },
        # The source parquet exposes only the four-type FX top-of-book
        # alphabet. It does NOT distinguish in-spread limit improvements
        # from top-of-book limit moves (both collapse into the same FX
        # event-type code). Therefore the in-spread bid/ask ratio cannot
        # be estimated from the cached events; it falls back to symmetric
        # (1:1) as the mu rebalance target. This is documented in the
        # prompt-07b spec under the calibration-target step.
        "bid_ask_in_spread_ratio": None,
        "bid_ask_in_spread_ratio_unavailability_reason": (
            "EURUSD events parquet exposes the four-type FX top-of-book "
            "alphabet only; in-spread vs top-of-book limit moves collapse "
            "into the same FX event-type code and cannot be disentangled "
            "from the cached events. Estimator falls back to symmetric "
            "(1:1) as the mu rebalance target."
        ),
        # Spread / mid diagnostics: the source parquet does not carry a
        # per-event bid / ask price column in the columns we load
        # (`timestamp`, `event_type`); spread and mid would require
        # joining against a full LOB-state column. For the prompt-07b
        # plausibility-band purpose, the FX-event composition is
        # sufficient and the spread / mid diagnostics are recorded as
        # null with this reason.
        "spread_diagnostics": None,
        "spread_diagnostics_unavailability_reason": (
            "Source parquet columns (`timestamp`, `event_type`) do not "
            "carry a per-event bid / ask price; spread / mid reconstruction "
            "would require a join against full LOB-state columns. Not "
            "needed for the 07b plausibility band; recorded as null."
        ),
        "mid_diagnostics": None,
        "mid_diagnostics_unavailability_reason": (
            "Same reason as spread_diagnostics: source parquet does not "
            "carry per-event mid in the loaded columns."
        ),
        "data_provenance": {
            "source_parquet": str(PARQUET),
            "primary_day_utc": PRIMARY_DAY,
            "jitter_scheme": "random_ms",
            "jitter_seed": int(JITTER_SEED),
            "column_convention": list(PRIMARY_4TYPE),
            "filter": "event_type in {bid_up, bid_down, ask_up, ask_down}",
            "sort_kind": "stable",
        },
        "cache_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(
        f"  composition: bid_up={fractions[0]:.4f} "
        f"bid_down={fractions[1]:.4f} ask_up={fractions[2]:.4f} "
        f"ask_down={fractions[3]:.4f}"
    )
    print(f"  rate: {payload['event_rate_per_second']:.4f} events/s")


if __name__ == "__main__":
    main()
