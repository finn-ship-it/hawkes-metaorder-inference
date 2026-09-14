"""Internal dissertation artefact rebuild driver.

This script requires external empirical inputs and an adjacent dissertation
source tree. It is retained as provenance for the frozen outputs under
``dissertation_artifacts/`` and is not part of the public ``make verify`` path.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

DATE = "2026-05-06"
RUNS = REPO / "runs"
ARTIFACTS = REPO / "dissertation_artifacts"
DATA_DIR = ARTIFACTS / "data"
OVERLEAF = REPO.parent / "overleaf" / "my overleaf" / "Work_on_Hawkes"

ENVELOPE = REPO / "targets" / "real_fx_envelope.json"
DRAFT = REPO / "targets" / "empirical_envelope_draft.json"

DEFAULT_SEEDS = 50
QUICK_SEEDS = 5
DEFAULT_SPREAD_HORIZONS = (30.0, 60.0, 120.0, 240.0, 480.0, 960.0)
QUICK_SPREAD_HORIZONS = (30.0, 60.0, 120.0)


# ---------------------------------------------------------------------------
# Step 01 — ensembles (prompt 02 logic, parametrised by seed count)
# ---------------------------------------------------------------------------


def step_ensembles(seeds: tuple[int, ...]) -> dict:
    from simulator.experiment import ExperimentConfig, run_experiment
    from simulator.artifact import load_run

    configs = [
        ("first_milestone", "first_milestone.json"),
        ("two_regime_smoke", "two_regime_smoke.json"),
        ("meta_order_smoke", "meta_order_smoke.json"),
    ]
    root = RUNS / f"ensembles_{DATE}"
    root.mkdir(parents=True, exist_ok=True)

    summary = {"seeds": list(seeds), "n_seeds": len(seeds), "configs": {}}
    t_step = time.time()
    for name, basename in configs:
        final_dir = root / f"{name}_{len(seeds)}seeds"
        if not final_dir.is_dir():
            cfg = ExperimentConfig(
                config_path=str(REPO / "configs" / basename),
                seeds=seeds,
                output_root=str(root),
                envelope_path=str(ENVELOPE),
                experiment_label=f"{name}_{len(seeds)}seeds",
            )
            t0 = time.time()
            experiment_dir = run_experiment(cfg)
            print(f"  [{name}] ran in {time.time()-t0:.1f}s")
            os.rename(experiment_dir, final_dir)
        else:
            print(f"  [{name}] reused {final_dir}")

        inner = next(
            d for d in final_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"ensemble_{name}_")
        )
        es = json.load(open(inner / "ensemble_summary.json"))
        agg = es["aggregate"]
        summary["configs"][name] = {
            "rho_mean": agg["endogeneity.spectral_radius"]["mean"],
            "rho_std": agg["endogeneity.spectral_radius"]["std"],
            "total_rate_mean": agg["latent.total_rate"]["mean"],
            "spread_mean": agg["observation.spread_mean_ticks"]["mean"],
            "crossed_total": int(sum(
                json.load(open(d / "summary.json"))["observation"]["num_crossed_observations"]
                for d in inner.iterdir() if d.is_dir() and d.name.startswith(name)
            )),
            "n_members": es["num_members"],
            "envelope_overall": json.load(open(inner / "ensemble_comparison.json")).get("overall_status"),
        }
    summary["wall_seconds"] = time.time() - t_step
    return summary


# ---------------------------------------------------------------------------
# Step 02 — spread-scaling sweep (prompt 03)
# ---------------------------------------------------------------------------


def step_spread_scaling(seeds: tuple[int, ...], horizons: tuple[float, ...]) -> dict:
    from simulator.config import config_from_json
    from simulator.ensemble import run_ensemble

    cfg_path = REPO / "configs" / "first_milestone.json"
    base_cfg = config_from_json(str(cfg_path))
    sweep_root = RUNS / f"spread_scaling_{DATE}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary = {"seeds": list(seeds), "horizons": list(horizons), "rows": []}
    t_step = time.time()
    for T in horizons:
        final_dir = sweep_root / f"T{int(T):03d}"
        if not final_dir.is_dir():
            cfg = dataclasses.replace(base_cfg, horizon=float(T))
            final_dir.mkdir(parents=True)
            t0 = time.time()
            run_ensemble(
                cfg, seeds=seeds, runs_root=str(final_dir), observe=True,
                envelope_path=None, timestamp_prefix=f"spread_T{int(T):03d}",
            )
            print(f"  [T={T}] ran in {time.time()-t0:.1f}s")
        else:
            print(f"  [T={T}] reused {final_dir}")
        inner = next(
            d for d in final_dir.iterdir()
            if d.is_dir() and d.name.startswith("ensemble_")
        )
        es = json.load(open(inner / "ensemble_summary.json"))
        agg = es["aggregate"]
        row = {"T": T, "N_obs_mean": agg["observation.num_observed_events"]["mean"]}
        for f in ("spread_mean_ticks", "spread_median_ticks", "spread_p90_ticks",
                  "spread_p99_ticks", "spread_max_ticks", "final_spread_ticks"):
            full = "observation." + f
            row[f] = agg[full]["mean"]
        summary["rows"].append(row)

    # Write summary CSV + fit JSON.
    import numpy as np
    csv_path = sweep_root / "spread_scaling_summary.csv"
    fields = ("spread_mean_ticks", "spread_median_ticks", "spread_p90_ticks",
              "spread_p99_ticks", "spread_max_ticks", "final_spread_ticks")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        headers = ["T", "n_members", "N_obs_mean", "N_obs_std"] + [
            f"observation.{f}.{stat}" for f in fields for stat in ("mean", "std")
        ]
        w.writerow(headers)
        for row, T in zip(summary["rows"], horizons):
            inner = next(
                d for d in (sweep_root / f"T{int(T):03d}").iterdir()
                if d.is_dir() and d.name.startswith("ensemble_")
            )
            es = json.load(open(inner / "ensemble_summary.json"))
            agg = es["aggregate"]
            line = [int(T), es["num_members"],
                    agg["observation.num_observed_events"]["mean"],
                    agg["observation.num_observed_events"]["std"]]
            for f in fields:
                line.extend([agg["observation." + f]["mean"],
                             agg["observation." + f]["std"]])
            w.writerow(line)

    fits = {}
    Ns = np.array([r["N_obs_mean"] for r in summary["rows"]])
    for f in fields:
        ys = np.array([r[f] for r in summary["rows"]])
        if (Ns > 0).all() and (ys > 0).all() and len(Ns) >= 2:
            slope, intercept = np.polyfit(np.log(Ns), np.log(ys), 1)
            yhat = slope * np.log(Ns) + intercept
            ss_res = float(np.sum((np.log(ys) - yhat) ** 2))
            ss_tot = float(np.sum((np.log(ys) - np.log(ys).mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
            fits["observation." + f] = {"a": float(math.exp(intercept)),
                                         "b": float(slope), "r2": r2,
                                         "n": int(len(Ns))}
        else:
            fits["observation." + f] = {"a": None, "b": None, "r2": None, "n": int(len(Ns))}
    with open(sweep_root / "spread_scaling_fit.json", "w") as fh:
        json.dump({"model": "field_mean = a * N_obs^b",
                   "horizons": list(horizons),
                   "n_seeds_per_horizon": len(seeds),
                   "fits": fits, "rows": summary["rows"]}, fh, indent=2)
    summary["fits"] = fits
    summary["wall_seconds"] = time.time() - t_step
    return summary


# ---------------------------------------------------------------------------
# Step 03 — recovery (prompt 04)
# ---------------------------------------------------------------------------


def step_recovery(seeds: tuple[int, ...]) -> dict:
    from simulator.experiment import ExperimentConfig, run_experiment
    from simulator.recovery import MetaOrderBaselineConfig

    label = f"recovery_meta_order_smoke_{len(seeds)}seeds"
    root = RUNS / f"recovery_meta_order_smoke_{DATE}"
    root.mkdir(parents=True, exist_ok=True)
    existing = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(label)]
    t_step = time.time()
    if existing:
        experiment_dir = existing[0]
        print(f"  [recovery] reused {experiment_dir}")
    else:
        cfg = ExperimentConfig(
            config_path=str(REPO / "configs" / "meta_order_smoke.json"),
            seeds=seeds,
            output_root=str(root),
            envelope_path=str(ENVELOPE),
            meta_order_baseline_cfg=MetaOrderBaselineConfig(
                window_seconds=5.0, score_threshold=0.5,
                direction="up", min_active_seconds=5.0,
            ),
            experiment_label=label,
        )
        t0 = time.time()
        experiment_dir = Path(run_experiment(cfg))
        print(f"  [recovery] ran in {time.time()-t0:.1f}s")

    eval_summary = json.load(open(experiment_dir / "evaluation_summary.json"))
    rows = []
    for member in eval_summary.get("members", []):
        ev = member.get("meta_order_evaluation")
        if ev is None:
            continue
        rows.append({
            "seed": member["seed"],
            "precision": ev.get("precision"),
            "recall": ev.get("recall"),
            "f1": ev.get("f1"),
            "mean_detection_delay_seconds": ev.get("mean_detection_delay_seconds"),
            "fp_intervals": len(ev.get("false_positive_intervals", [])),
            "fn_intervals": len(ev.get("false_negative_intervals", [])),
        })

    csv_path = root / "recovery_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _stat(field):
        xs = [r[field] for r in rows if r[field] is not None]
        return {
            "n": len(xs),
            "mean": statistics.mean(xs) if xs else None,
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
        }

    aggregate = {
        "n_seeds": len(rows),
        "precision": _stat("precision"),
        "recall": _stat("recall"),
        "f1": _stat("f1"),
        "mean_detection_delay_seconds": _stat("mean_detection_delay_seconds"),
        "fp_intervals": _stat("fp_intervals"),
        "fn_intervals": _stat("fn_intervals"),
    }
    with open(root / "recovery_aggregate.json", "w") as f:
        json.dump({"experiment_dir": str(experiment_dir), "aggregate": aggregate}, f, indent=2)
    summary = {"seeds": list(seeds), "n_seeds": len(seeds),
               "aggregate": aggregate, "wall_seconds": time.time() - t_step}
    return summary


# ---------------------------------------------------------------------------
# Step 04 — envelope refresh (prompt 06)
# ---------------------------------------------------------------------------


def step_envelope_refresh() -> dict:
    """Re-execute prompt-06 driver in-process."""
    import importlib.util
    p06 = importlib.util.spec_from_file_location("run_prompt_06", str(RUNS / "run_prompt_06.py"))
    mod = importlib.util.module_from_spec(p06)
    sys.modules["run_prompt_06"] = mod
    p06.loader.exec_module(mod)
    t0 = time.time()
    mod.main()
    return {"wall_seconds": time.time() - t0,
            "envelope_path": str(DRAFT)}


# ---------------------------------------------------------------------------
# Step 05 — stage data files (prompt 09)
# ---------------------------------------------------------------------------


def step_stage_data(seeds_count: int) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        (RUNS / f"ensembles_{DATE}/first_milestone_{seeds_count}seeds/ensemble_first_milestone_{seeds_count}seeds/ensemble_summary.json",
         DATA_DIR / "ensemble_summary_first_milestone.json"),
        (RUNS / f"ensembles_{DATE}/two_regime_smoke_{seeds_count}seeds/ensemble_two_regime_smoke_{seeds_count}seeds/ensemble_summary.json",
         DATA_DIR / "ensemble_summary_two_regime_smoke.json"),
        (RUNS / f"ensembles_{DATE}/meta_order_smoke_{seeds_count}seeds/ensemble_meta_order_smoke_{seeds_count}seeds/ensemble_summary.json",
         DATA_DIR / "ensemble_summary_meta_order_smoke.json"),
        (RUNS / f"spread_scaling_{DATE}/spread_scaling_summary.csv",
         DATA_DIR / "spread_scaling_summary.csv"),
        (RUNS / f"spread_scaling_{DATE}/spread_scaling_fit.json",
         DATA_DIR / "spread_scaling_fit.json"),
        (RUNS / f"recovery_meta_order_smoke_{DATE}/recovery_summary.csv",
         DATA_DIR / "recovery_summary.csv"),
        (RUNS / f"recovery_meta_order_smoke_{DATE}/recovery_aggregate.json",
         DATA_DIR / "recovery_aggregate.json"),
    ]
    copied = []
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(REPO)))
        else:
            print(f"  [stage_data] WARN: missing {src}")
    return {"copied": copied}


# ---------------------------------------------------------------------------
# Step 06 — figures (prompt 07)
# ---------------------------------------------------------------------------


def step_figures() -> dict:
    import importlib.util
    p = ARTIFACTS / "figures" / "regenerate.py"
    spec = importlib.util.spec_from_file_location("figures_regenerate", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["figures_regenerate"] = mod
    t0 = time.time()
    spec.loader.exec_module(mod)
    mod.main()
    return {"wall_seconds": time.time() - t0,
            "figures": [p.name for p in (ARTIFACTS / "figures").glob("*.pdf")]}


# ---------------------------------------------------------------------------
# Step 07 — tables (prompt 08)
# ---------------------------------------------------------------------------


def step_tables() -> dict:
    import importlib.util
    p = ARTIFACTS / "tables" / "regenerate.py"
    spec = importlib.util.spec_from_file_location("tables_regenerate", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tables_regenerate"] = mod
    t0 = time.time()
    spec.loader.exec_module(mod)
    mod.main()
    return {"wall_seconds": time.time() - t0,
            "tables": [p.name for p in (ARTIFACTS / "tables").glob("*.tex")]}


# ---------------------------------------------------------------------------
# Step 08 — sync to overleaf folder (path-handling decision per prompt 10/11)
# ---------------------------------------------------------------------------


def step_copy_to_overleaf() -> dict:
    if not OVERLEAF.is_dir():
        return {"skipped": True, "reason": f"{OVERLEAF} not present"}
    (OVERLEAF / "tables").mkdir(parents=True, exist_ok=True)
    (OVERLEAF / "figures").mkdir(parents=True, exist_ok=True)
    n_tables = 0
    for src in (ARTIFACTS / "tables").glob("*.tex"):
        shutil.copy2(src, OVERLEAF / "tables" / src.name)
        n_tables += 1
    n_figs = 0
    for src in (ARTIFACTS / "figures").glob("*.pdf"):
        shutil.copy2(src, OVERLEAF / "figures" / src.name)
        n_figs += 1
    return {"tables_copied": n_tables, "figures_copied": n_figs,
            "overleaf_dir": str(OVERLEAF)}


# ---------------------------------------------------------------------------
# Step 09 — manifest data (eurusd_layer1_summary.json)
# ---------------------------------------------------------------------------


def step_manifest_refresh() -> dict:
    """Build dissertation_artifacts/data/eurusd_layer1_summary.json."""
    csv_path = REPO.parent / "Data" / "Data output" / "EURUSD output" / "layer1" / "layer1_models_summary.csv"
    if not csv_path.exists():
        return {"skipped": True, "reason": f"{csv_path} missing"}
    rows = list(csv.DictReader(open(csv_path)))
    sumexp_primary = next(r for r in rows if r["model"] == "SumExpHawkes_4type_R4")
    sumexp_secondary = next(r for r in rows if r["model"] == "SumExpHawkes_4type_R4_day_2021-07-19")
    # Both rows are provider-time-day fits on HistData's fixed EST clock:
    # SumExpHawkes_4type_R4 is the PRIMARY_DATE = 2021-07-20 fit (one provider day,
    # T = 86,400 s); SumExpHawkes_4type_R4_day_2021-07-19 is the SECONDARY_DATE
    # robustness fit (also one provider day). No multi-day pooling is performed.
    payload = {
        "preferred_model": "SumExpHawkes_4type_R4",
        "primary_day_2021_07_20": {
            "rho_spec": float(sumexp_primary["rho_spec"]),
            "n_events": int(sumexp_primary["N"]),
            "log_likelihood": float(sumexp_primary["ll"]),
            "AIC": float(sumexp_primary["AIC"]),
            "BIC": float(sumexp_primary["BIC"]),
            "fit_window_seconds": 86400.0,
            "fit_window_description": "single provider-time day (fixed EST), 2021-07-20, 86,400 s",
        },
        "day_2021_07_19": {
            "rho_spec": float(sumexp_secondary["rho_spec"]),
            "n_events": int(sumexp_secondary["N"]),
            "log_likelihood": float(sumexp_secondary["ll"]),
            "fit_window_seconds": 86400.0,
            "fit_window_description": "single provider-time day (fixed EST), 2021-07-19, 86,400 s (robustness)",
        },
        "betas_grid_per_s": [0.1, 1.0, 10.0, 100.0],
        "data_window": "2021-07 -- 2022-07 EURUSD HistData (13 monthly CSVs); fits restricted to one provider-time day (fixed EST) each",
        "primary_date": "2021-07-20",
        "secondary_date": "2021-07-19",
        "source_csv": str(csv_path),
    }
    out = DATA_DIR / "eurusd_layer1_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    return {"path": str(out.relative_to(REPO))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="seed count for ensembles + recovery (default 50)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test mode: 5 seeds and 3 horizons")
    args = parser.parse_args()

    seed_count = QUICK_SEEDS if args.quick else args.seeds
    seeds = tuple(range(1, seed_count + 1))
    horizons = QUICK_SPREAD_HORIZONS if args.quick else DEFAULT_SPREAD_HORIZONS

    print(f"=== rebuild.py — seeds={seed_count}, horizons={horizons} ===")
    log: dict = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seeds": list(seeds),
        "n_seeds": seed_count,
        "horizons": list(horizons),
        "quick": bool(args.quick),
        "steps": {},
    }

    t_total = time.time()

    print("\n[1/9] ensembles…")
    log["steps"]["ensembles"] = step_ensembles(seeds)

    print("\n[2/9] spread_scaling…")
    log["steps"]["spread_scaling"] = step_spread_scaling(seeds=tuple(range(1, 31)),
                                                          horizons=horizons) \
        if not args.quick else step_spread_scaling(seeds=seeds, horizons=horizons)

    print("\n[3/9] recovery…")
    log["steps"]["recovery"] = step_recovery(seeds)

    print("\n[4/9] envelope_refresh…")
    log["steps"]["envelope_refresh"] = step_envelope_refresh()

    if args.quick:
        # --quick is a smoke test of the simulator-side pipeline only.
        # Skip steps that would overwrite production artefacts in
        # dissertation_artifacts/data/, figures/, tables/, and the Overleaf folder.
        # The 5-seed ensemble + recovery dirs created in steps 1 and 3 sit
        # alongside the 50-seed production dirs; they do not replace them.
        print("\n[5/9 .. 9/9] skipped under --quick (artefact regeneration would "
              "overwrite production 50-seed numbers; rerun without --quick to refresh "
              "data/, figures/, tables/, and Overleaf copies).")
        log["steps"]["stage_data"] = {"skipped": True, "reason": "quick"}
        log["steps"]["manifest_refresh"] = {"skipped": True, "reason": "quick"}
        log["steps"]["figures"] = {"skipped": True, "reason": "quick"}
        log["steps"]["tables"] = {"skipped": True, "reason": "quick"}
        log["steps"]["copy_to_overleaf"] = {"skipped": True, "reason": "quick"}
    else:
        print("\n[5/9] stage_data…")
        log["steps"]["stage_data"] = step_stage_data(seed_count)

        print("\n[6/9] manifest_refresh…")
        log["steps"]["manifest_refresh"] = step_manifest_refresh()

        print("\n[7/9] figures…")
        log["steps"]["figures"] = step_figures()

        print("\n[8/9] tables…")
        log["steps"]["tables"] = step_tables()

        print("\n[9/9] copy_to_overleaf…")
        log["steps"]["copy_to_overleaf"] = step_copy_to_overleaf()

    log["wall_seconds_total"] = time.time() - t_total
    log["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    log_path = ARTIFACTS / "build_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"\n=== rebuild complete in {log['wall_seconds_total']:.1f} s; build_log → {log_path} ===")


if __name__ == "__main__":
    main()
