"""Prompt 06 — refresh targets/empirical_envelope_draft.json (EURUSD-only).

Adds an `eurusd` block and a `meta` block alongside the original
`fields` dict. This driver uses the dissertation's EURUSD audit (§3.3, §4.6).

Idempotent: re-running overwrites the eurusd / meta blocks with the
latest values; the fields dict is untouched.

After refresh, runs first_milestone (seed 42) against the merged envelope
and writes a short status file under
`runs/empirical_envelope_refresh_2026-05-06/`.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

DRAFT_PATH = REPO / "targets" / "empirical_envelope_draft.json"
EUR_AUDIT = REPO.parent / "Data" / "Data output" / "EURUSD output" / "eurusd_event_audit.csv"
EUR_LAYER1_FULL = (
    REPO.parent / "Data" / "Data output" / "EURUSD output" / "layer1" /
    "layer1_params_sumexp_4type.json"
)
EUR_LAYER1_DAY = (
    REPO.parent / "Data" / "Data output" / "EURUSD output" / "layer1" /
    "layer1_params_sumexp_4type_day_2021-07-19.json"
)
SPREAD_FIT = REPO / "runs" / "spread_scaling_2026-05-06" / "spread_scaling_fit.json"
ENS_DIR = REPO / "runs" / "ensembles_2026-05-06"
ENVELOPE_REFRESH_REPORT = REPO / "runs" / "empirical_envelope_refresh_2026-05-06"


def _audit_pair(audit_csv: Path) -> dict:
    if not audit_csv.exists():
        return {}
    with open(audit_csv) as f:
        rows = list(csv.DictReader(f))
    out = {}
    rate_by_type = {}
    for r in rows:
        et = r["event_type"]
        if et in ("bid_up", "bid_down", "ask_up", "ask_down"):
            rate_by_type[et] = float(r["rate_per_s_global"])
        if et == "_spread_summary_":
            for k in ("spread_pips_mean", "spread_pips_median", "spread_pips_p90",
                      "spread_pips_p99", "spread_pips_max"):
                if k in r and r[k]:
                    out[k] = float(r[k])
    if rate_by_type:
        out["rates_by_event_type_per_s"] = [
            rate_by_type.get(et) for et in ("bid_up", "bid_down", "ask_up", "ask_down")
        ]
        out["total_rate_per_s"] = sum(v for v in rate_by_type.values() if v is not None)
    # Convert pips to ticks at simulator tick_size = 1e-5 (1 pip = 10 ticks).
    if "spread_pips_median" in out:
        out["spread_p50_ticks"] = out["spread_pips_median"] * 10
    if "spread_pips_p90" in out:
        out["spread_p90_ticks"] = out["spread_pips_p90"] * 10
    if "spread_pips_p99" in out:
        out["spread_p99_ticks"] = out["spread_pips_p99"] * 10
    return out


def _layer1(full_path: Path, day_path: Path) -> dict:
    """Read EURUSD Tier-3 sum-exp 4-type fit JSONs.

    The "primary" fit is a single UTC day at PRIMARY_DATE = 2021-07-20
    (T = 86,400 s). The "secondary" fit is the robustness check at
    SECONDARY_DATE = 2021-07-19. Both are single-day fits — no multi-day
    pooling is performed by the EURUSD layer1 ladder.
    """
    out = {}
    if full_path.exists():
        f = json.load(open(full_path))
        out["branching_ratio_primary_day"] = float(f["spectral_radius"])
        out["n_events_primary_day"] = int(f["N"])
        out["window_seconds_primary_day"] = float(f["T"])
        out["primary_day_label"] = "2021-07-20"
    if day_path.exists():
        d = json.load(open(day_path))
        out["branching_ratio_secondary_day"] = float(d["spectral_radius"])
        out["n_events_secondary_day"] = int(d["N"])
        out["secondary_day_label"] = "2021-07-19"
    return out


def _spread_exponents() -> dict:
    if not SPREAD_FIT.exists():
        return {}
    f = json.load(open(SPREAD_FIT))
    fits = f.get("fits", {})
    out = {}
    for k, v in fits.items():
        short = k.split(".", 1)[1] if "." in k else k
        out[short] = {"b": v.get("b"), "a": v.get("a"), "r2": v.get("r2")}
    return out


def _ensemble_stats() -> dict:
    out = {}
    for cfg in ("first_milestone", "two_regime_smoke", "meta_order_smoke"):
        es_path = (
            ENS_DIR / f"{cfg}_50seeds" / f"ensemble_{cfg}_50seeds" / "ensemble_summary.json"
        )
        if not es_path.exists():
            continue
        es = json.load(open(es_path))
        agg = es["aggregate"]
        out[cfg] = {
            "rho_mean": agg["endogeneity.spectral_radius"]["mean"],
            "rho_std": agg["endogeneity.spectral_radius"]["std"],
            "total_rate_mean": agg["latent.total_rate"]["mean"],
            "total_rate_std": agg["latent.total_rate"]["std"],
            "spread_mean_mean": agg["observation.spread_mean_ticks"]["mean"],
            "num_crossed_obs_max": agg["observation.num_crossed_observations"]["max"],
        }
    return out


def main():
    ENVELOPE_REFRESH_REPORT.mkdir(parents=True, exist_ok=True)
    draft = json.load(open(DRAFT_PATH))

    eur = _audit_pair(EUR_AUDIT)
    eur.update(_layer1(EUR_LAYER1_FULL, EUR_LAYER1_DAY))
    eur["spread_scaling_exponents"] = _spread_exponents()
    eur["window"] = "2021-07 -- 2022-07 (HistData EURUSD 13 monthly CSVs)"

    ensemble_stats = _ensemble_stats()
    crossed_holds = (
        all(v["num_crossed_obs_max"] == 0 for v in ensemble_stats.values())
        if ensemble_stats else None
    )

    meta = {
        "last_refresh_date": "2026-05-06",
        "eurusd_window": eur["window"],
        "n_seeds_per_ensemble": 50,
        "crossed_obs_invariant_holds": crossed_holds,
        "fields_block_unchanged": True,
        "ensemble_summary_inputs": ensemble_stats,
    }

    draft["eurusd"] = eur
    draft["meta"] = meta

    with open(DRAFT_PATH, "w") as f:
        json.dump(draft, f, indent=2)
    print(f"wrote {DRAFT_PATH}")

    # Regression check on first_milestone seed 42 against the refreshed draft.
    from simulator.config import config_from_json
    from simulator.runner import run_once
    from simulator.observation import default_observation_config
    import dataclasses

    cfg = config_from_json(str(REPO / "configs" / "first_milestone.json"))
    cfg = dataclasses.replace(cfg, seed=42)
    run_dir = run_once(
        cfg=cfg,
        runs_root=str(ENVELOPE_REFRESH_REPORT),
        observation_config=default_observation_config(),
        write_summary=True,
        compare_envelope_path=str(REPO / "targets" / "empirical_envelope_draft.json"),
    )
    print(f"regression run: {run_dir}")
    comp = json.load(open(os.path.join(run_dir, "comparison.json")))
    summary_status = {
        "overall_status": comp.get("overall_status"),
        "status_counts": comp.get("status_counts"),
    }
    with open(ENVELOPE_REFRESH_REPORT / "regression_status.json", "w") as f:
        json.dump(summary_status, f, indent=2)
    print(json.dumps(summary_status, indent=2))


if __name__ == "__main__":
    main()
