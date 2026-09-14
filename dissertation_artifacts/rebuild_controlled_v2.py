"""Recompute controlled-dissertation results after the thinning correction.

Usage: python dissertation_artifacts/rebuild_controlled_v2.py --output-root NEW_DIR

The output directory must not exist. Existing research outputs are never replaced.
Numerical protocols, seeds, detector settings, and fill-RNG seeding and within-run
pairing are retained.
Plot rendering is deliberately separate: the final submission renderers consume the
resulting JSON/CSV files after review. This builder creates no PDFs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values):
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {"n": len(xs), "mean": statistics.mean(xs) if xs else None,
            "std": statistics.pstdev(xs) if xs else None,
            "min": min(xs) if xs else None, "max": max(xs) if xs else None}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def recovery_table(root, data):
    """Table 5.1 retains the submitted table's sample-standard-deviation convention."""
    with (data / "recovery_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ["precision", "recall", "f1", "mean_detection_delay_seconds"]
    lines = [r"\begin{tabular}{ccccc}", r"\hline",
             r"Seed & Precision & Recall & F1 & Start offset (s) \\", r"\hline"]
    for row in rows[:5]:
        vals = [float(row[key]) for key in fields]
        lines.append(f"{row['seed']} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.1f}" + r" \\")
    first = [statistics.mean(float(row[key]) for row in rows[:5]) for key in fields]
    lines += [r"\hline", "mean (seeds 1--5) & " + " & ".join(
        f"{v:.3f}" if i < 3 else f"{v:.1f}" for i, v in enumerate(first)) + r" \\"]
    cells = []
    for i, key in enumerate(fields):
        values = [float(row[key]) for row in rows]
        digits = 3 if i < 3 else 2
        cells.append(f"{statistics.mean(values):.{digits}f}" + r" $\pm$ " +
                     f"{statistics.stdev(values):.{digits}f}")
    lines += [r"mean (seeds 1--50) $\pm$ std & " + " & ".join(cells) + r" \\",
              r"\hline", r"\end{tabular}"]
    path = root / "artifacts" / "tables" / "tab_recovery_example.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    from collections import Counter
    hmm = read(data / "recovery_hmm_meta_order_smoke.json")
    counts = {"threshold": dict(sorted(Counter(float(row[fields[-1]]) for row in rows).items())),
              "hmm": dict(sorted(Counter(float(row[fields[-1]]) for row in hmm["per_seed"]).items()))}
    dump(data / "recovery_onset_counts.json", counts)
    return counts


def ensemble(name, root, seeds):
    from simulator.config import config_from_json
    from simulator.ensemble import run_ensemble
    cfg = config_from_json(str(REPO / "configs" / f"{name}.json"))
    return Path(run_ensemble(cfg, seeds=seeds, runs_root=str(root), observe=True,
                            envelope_path=str(REPO / "targets" / "real_fx_envelope.json"),
                            timestamp_prefix=f"{name}_50seeds"))


def recovery(root, data):
    from simulator.artifact import load_run
    from simulator.recovery import MetaOrderBaselineConfig, score_meta_order_activity
    from simulator.recovery_eval import evaluate_meta_order_recovery
    import build_recovery_hmm_meta_order_smoke as hmm

    inner = ensemble("meta_order_smoke", root / "raw" / "ensembles", range(1, 51))
    baseline = MetaOrderBaselineConfig(window_seconds=5.0, score_threshold=0.5,
                                      direction="up", min_active_seconds=5.0)
    rows = []
    for seed in range(1, 51):
        run = load_run(str(inner / f"meta_order_smoke_50seeds_{seed}"))
        intervals = score_meta_order_activity(run.observed, baseline, horizon=run.config.horizon)
        result = evaluate_meta_order_recovery(run.config.meta_order.windows, intervals,
                                             horizon=run.config.horizon, time_resolution=0.1)
        rows.append({"seed": seed, "precision": result.precision, "recall": result.recall,
                     "f1": result.f1, "mean_detection_delay_seconds": result.mean_detection_delay_seconds,
                     "fp_intervals": len(result.false_positive_intervals),
                     "fn_intervals": len(result.false_negative_intervals)})
    write_csv(data / "recovery_summary.csv", rows)
    aggregate = {"n_seeds": 50}
    aggregate.update({key: stat(row[key] for row in rows) for key in rows[0] if key != "seed"})
    dump(data / "recovery_aggregate.json", {
        "experiment_dir": str(inner.relative_to(root)), "recovery_method": "threshold",
        "aggregate": aggregate})
    hmm.ENSEMBLE_DIR = inner
    hmm.THRESHOLD_CSV = data / "recovery_summary.csv"
    hmm.OUT_JSON = data / "recovery_hmm_meta_order_smoke.json"
    hmm._make_posterior_figure = lambda per_seed: None
    # The original builder initializes at zero: LoadedRun does not expose .seed.
    # Retain that historical setting to isolate the simulator correction.
    hmm.main()
    recovery_table(root, data)
    dump(data / "ensemble_summary_meta_order_smoke.json", read(inner / "ensemble_summary.json"))
    return {"threshold": aggregate, "hmm": read(hmm.OUT_JSON)["hmm_aggregate"]}


def layer3(root, data):
    import build_layer3_d1_skew_markout as base
    import build_layer3_d1_skew_frontier as frontier
    base.OUT_JSON = data / "layer3_d1_skew_markout_comparison.json"
    base._make_figure = lambda per_seed: None
    base.main()
    frontier.OUT_JSON = data / "layer3_d1_skew_frontier.json"
    frontier.OUT_TEX = root / "artifacts" / "tables" / "tab_layer3_d1_skew_frontier.tex"
    frontier.COMMITTED_10MSC = base.OUT_JSON
    frontier._make_figure = lambda rows, feasible: None
    frontier.main()
    return {"comparison": read(base.OUT_JSON)["gate_summary"],
            "frontier": read(frontier.OUT_JSON)["frontier"],
            "feasible_region": read(frontier.OUT_JSON)["feasible_region"]}


def spread(root, data):
    from simulator.config import config_from_json
    from simulator.ensemble import run_ensemble
    cfg = config_from_json(str(REPO / "configs" / "first_milestone.json"))
    fields = ["spread_mean_ticks", "spread_median_ticks", "spread_p90_ticks",
              "spread_p99_ticks", "spread_max_ticks", "final_spread_ticks"]
    horizons = [30, 60, 120, 240, 480, 960]
    rows = []
    for horizon in horizons:
        started = time.perf_counter()
        inner = Path(run_ensemble(dataclasses.replace(cfg, horizon=float(horizon)),
                     seeds=range(1, 31), runs_root=str(root / "raw" / "spread"),
                     observe=True, envelope_path=None, timestamp_prefix=f"spread_T{horizon:03d}"))
        summary = read(inner / "ensemble_summary.json")
        ag = summary["aggregate"]
        row = {"T": horizon, "n_members": 30,
               "N_obs_mean": ag["observation.num_observed_events"]["mean"],
               "N_obs_std": ag["observation.num_observed_events"]["std"]}
        for field in fields:
            for measure in ("mean", "std"):
                row[f"observation.{field}.{measure}"] = ag[f"observation.{field}"][measure]
        rows.append(row)
        print(f"Spread T={horizon}: {time.perf_counter()-started:.1f}s", flush=True)
    fits = {}
    xs = np.log([row["N_obs_mean"] for row in rows])
    for field in fields:
        ys = np.log([row[f"observation.{field}.mean"] for row in rows])
        slope, intercept = np.polyfit(xs, ys, 1)
        residual = float(np.sum((ys - (slope * xs + intercept)) ** 2))
        total = float(np.sum((ys - ys.mean()) ** 2))
        fits[f"observation.{field}"] = {"a": float(np.exp(intercept)), "b": float(slope),
                                       "r2": 1 - residual / total, "n": len(rows)}
    write_csv(data / "spread_scaling_summary.csv", rows)
    dump(data / "spread_scaling_fit.json", {"model": "field_mean = a * N_obs^b",
         "horizons": horizons, "n_seeds_per_horizon": 30, "fits": fits, "rows": rows})
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r"$T$ (s) & $N_{\mathrm{obs}}$ & mean & median & p90 & p99 & max & final \\", r"\hline"]
    for row in rows:
        vals = [f"{row[f'observation.{field}.mean']:.2f}" for field in fields]
        lines.append(f"{row['T']} & {row['N_obs_mean']:.1f} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    table = root / "artifacts" / "tables" / "tab_spread_scaling.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n")
    return {"fits": fits, "rows": rows}


def baselines(root, data):
    summaries = {}
    for name in ("first_milestone", "two_regime_smoke"):
        inner = ensemble(name, root / "raw" / "ensembles", range(1, 51))
        summary = read(inner / "ensemble_summary.json")
        dump(data / f"ensemble_summary_{name}.json", summary)
        summaries[name] = summary["aggregate"]
    return summaries


def smoke(root, data):
    """A seed-42 mechanical check per configuration, not a 50-run baseline rerun."""
    from simulator.config import config_from_json
    from simulator.observation import default_observation_config
    from simulator.runner import run_once
    rows = []
    for name in ("first_milestone", "two_regime_smoke", "meta_order_smoke"):
        cfg = dataclasses.replace(config_from_json(str(REPO / "configs" / f"{name}.json")), seed=42)
        previous = root / "raw" / "ensembles" / f"ensemble_{name}_50seeds" / f"{name}_50seeds_42"
        if (previous / "summary.json").is_file():
            location = previous
        else:
            location = Path(run_once(cfg, runs_root=str(root / "raw" / "smoke"),
                            observation_config=default_observation_config(), write_summary=True,
                            timestamp=f"{name}_seed42_smoke"))
        summary = read(location / "summary.json")
        rows.append({"config": f"configs/{name}.json", "seed": 42, "horizon_seconds": cfg.horizon,
                     "raw_run": str(location.relative_to(root)),
                     "spectral_radius": summary["endogeneity"]["spectral_radius"],
                     "num_crossed_observations": summary["observation"]["num_crossed_observations"],
                     "num_observed_events": summary["observation"]["num_observed_events"]})
    payload = {"scope": "One seed-42 mechanical check for each of the three controlled configurations",
               "n_runs": 3, "n_seeds_per_configuration": 1,
               "all_spectral_radii_equal_0_2": all(np.isclose(r["spectral_radius"], 0.2) for r in rows),
               "total_crossed_observations": sum(r["num_crossed_observations"] for r in rows), "runs": rows}
    assert payload["all_spectral_radii_equal_0_2"]
    assert payload["total_crossed_observations"] == 0
    dump(data / "controlled_baseline_smoke_v2.json", payload)
    return payload


def benchmark(root, data):
    from simulator.tests.benchmarks import bench_kernels as bench
    rows = []
    for name, builder in (("EXPONENTIAL", bench._exp_cfg),
                          ("EXPONENTIAL_SUM", bench._sumexp_cfg),
                          ("POWERLAW_CUTOFF", bench._powerlaw_cfg)):
        results = [bench._bench_one(builder(seed)) for seed in range(42, 62)]
        timings, counts = np.asarray(results).T
        rows.append({"kernel_family": name, "horizon_seconds": 10.0, "dimension": 4,
                     "n_runs": 20, "median_wall_clock_seconds": float(np.median(timings)),
                     "mean_wall_clock_seconds": float(np.mean(timings)),
                     "min_wall_clock_seconds": float(np.min(timings)),
                     "max_wall_clock_seconds": float(np.max(timings)),
                     "median_event_count": int(np.median(counts)),
                     "mean_event_count": float(np.mean(counts))})
    write_csv(data / "kernel_benchmark.csv", rows)
    return rows


def portable(value, root):
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root), ".").replace(str(REPO), "repo")
    return value


BASELINE_NAMES = ["data/recovery_summary.csv", "data/recovery_aggregate.json",
                  "data/recovery_hmm_meta_order_smoke.json",
                  "data/layer3_d1_skew_markout_comparison.json", "data/layer3_d1_skew_frontier.json",
                  "data/spread_scaling_summary.csv", "data/spread_scaling_fit.json",
                  "data/kernel_benchmark.csv", "data/ensemble_summary_first_milestone.json",
                  "data/ensemble_summary_two_regime_smoke.json", "data/ensemble_summary_meta_order_smoke.json",
                  "tables/tab_recovery_example.tex", "tables/tab_layer3_d1_skew_frontier.tex",
                  "tables/tab_spread_scaling.tex"]


def snapshot_baseline(root, git_ref=None):
    """Freeze comparisons before any publication step can replace canonical files."""
    for name in BASELINE_NAMES:
        dest = root / "before_artifacts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if git_ref is None:
            if (HERE / name).is_file():
                shutil.copy2(HERE / name, dest)
        else:
            relative = str((HERE / name).relative_to(REPO))
            content = subprocess.run(["git", "show", f"{git_ref}:{relative}"], cwd=REPO,
                                     check=False, capture_output=True)
            if content.returncode == 0:
                dest.write_bytes(content.stdout)


def finish_report(root, report):
    report["file_mapping"] = []
    report["before_after"] = {}
    selectors = {"recovery_hmm_meta_order_smoke.json": ["hmm_aggregate", "threshold_aggregate"],
                 "recovery_aggregate.json": ["aggregate"],
                 "layer3_d1_skew_markout_comparison.json": ["gate_summary"],
                 "layer3_d1_skew_frontier.json": ["frontier", "feasible_region"],
                 "spread_scaling_fit.json": ["fits", "rows"]}
    selectors.update({f"ensemble_summary_{name}.json": ["aggregate"]
                      for name in ("first_milestone", "two_regime_smoke", "meta_order_smoke")})
    # Normalize generated CSV line endings before recording their final digests.
    for path in (root / "artifacts").rglob("*.csv"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    # Sanitize new outputs, but preserve byte-identical historical snapshots.
    for path in root.rglob("*.json"):
        if not path.is_relative_to(root / "before_artifacts"):
            dump(path, portable(read(path), root))
    for corrected in sorted((root / "artifacts").rglob("*")):
        if not corrected.is_file():
            continue
        relative = corrected.relative_to(root / "artifacts")
        old = root / "before_artifacts" / relative
        report["file_mapping"].append({"corrected": str(corrected.relative_to(root)),
            "canonical": str((HERE / relative).relative_to(REPO)),
            "before_sha256": sha(old) if old.is_file() else None, "after_sha256": sha(corrected)})
        if not old.is_file():
            continue
        if corrected.name in selectors:
            before, after = read(old), read(corrected)
            report["before_after"][corrected.name] = {
                key: {"before": before.get(key), "after": after.get(key)}
                for key in selectors[corrected.name]}
        elif corrected.suffix == ".csv":
            with old.open(newline="") as a, corrected.open(newline="") as b:
                report["before_after"][corrected.name] = {
                    "before": list(csv.DictReader(a)), "after": list(csv.DictReader(b))}
    report["final_builder_sha256"] = sha(Path(__file__))
    if "baselines" not in report["stages"]:
        report["historical_not_rerun"] = {
            "scope": "Historical baseline ensembles and envelope regressions are not current submission results",
            "artifacts": ["dissertation_artifacts/data/ensemble_summary_first_milestone.json",
                          "dissertation_artifacts/data/ensemble_summary_two_regime_smoke.json",
                          "dissertation_artifacts/data/ensembles_aggregate.json",
                          "dissertation_artifacts/tables/tab_single_run_regression.tex",
                          "runs/empirical_envelope_refresh_2026-05-06/"],
            "current_mechanical_check": ("artifacts/data/controlled_baseline_smoke_v2.json"
                                         if (root / "artifacts/data/controlled_baseline_smoke_v2.json").is_file()
                                         else None)}
    dump(root / "correction_report.json", portable(report, root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-git-ref", help="Optional explicit commit containing historical comparison artifacts")
    parser.add_argument("--stages", nargs="+", choices=["recovery", "layer3", "spread", "smoke", "baselines", "benchmark"],
                        default=["recovery", "layer3", "spread", "smoke", "benchmark"],
                        help="Default: submission results and timings; baselines is an optional historical-ensemble rerun")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / "artifacts" / "data"
    data.mkdir(parents=True)
    snapshot_baseline(root, args.baseline_git_ref)
    source_files = [Path(__file__), REPO / "src/simulator/hawkes_core.py",
                    HERE / "build_recovery_hmm_meta_order_smoke.py",
                    HERE / "build_layer3_d1_skew_markout.py", HERE / "build_layer3_d1_skew_frontier.py"]
    source_files += list((REPO / "src" / "simulator").glob("*.py"))
    source_files += [REPO / "configs" / f"{n}.json" for n in ("first_milestone", "two_regime_smoke", "meta_order_smoke")]
    report = {"purpose": "Recompute controlled results after proposal-bound correction",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in source_files},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "platform": platform.platform(), "machine": platform.machine()},
              "protocol_notes": ["Recovery HMM initialization seed remains zero, matching the historical builder.",
                                 "git_head records the checkout base; source_sha256 pins the actual working-tree implementation used.",
                                 "Threshold detector is explicitly selected, not the current experiment-runner default.",
                                 "Layer3 endpoint checks compare two corrected runs, not historical numbers.",
                                 "Existing raw and published results are untouched; no PDF plots are generated.",
                                 "Empirical fits, fitted-model forward validation and twelve-type results are unchanged."],
              "baseline_source": args.baseline_git_ref or "working-tree snapshot taken before computation",
              "stages": {}, "before_after": {}, "file_mapping": []}
    for name in args.stages:
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        result = globals()[name](root, data)
        report["stages"][name] = {"elapsed_seconds": time.perf_counter()-started, "summary": result}
        dump(root / "correction_report.json", portable(report, root))
        print(f"DONE {name}: {report['stages'][name]['elapsed_seconds']:.1f}s", flush=True)
    finish_report(root, report)
    print(f"COMPLETE: {root / 'correction_report.json'}", flush=True)


if __name__ == "__main__":
    main()
