"""Regenerate all dissertation tables under dissertation_artifacts/tables/.

Each fragment is a bare `\\begin{tabular}` ... `\\end{tabular}` environment;
the chapter wraps it in `\\begin{table}[H] \\caption{...} \\label{...}`.

Idempotent: running twice produces byte-identical output.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = Path(__file__).resolve().parent
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SRC = REPO / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_table(name: str, body: str) -> Path:
    """Write a tabular fragment, ensuring trailing newline. Returns the path."""
    path = TABLES_DIR / name
    if not body.endswith("\n"):
        body = body + "\n"
    path.write_text(body)
    return path


def fmt(x, digits=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


# ---------------------------------------------------------------------------
# 1. tab_sim_params.tex — shipped configurations summary
# ---------------------------------------------------------------------------


def gen_sim_params() -> Path:
    """4-column table matching the shipped tab:sim-params.

    Columns: Configuration | Horizon | Regimes | Meta-order.
    """
    def regime_str(cfg: dict) -> str:
        K = cfg["latent_regime"]["num_regimes"]
        baseline = cfg["baseline"]["mu"]
        arr = baseline.get("data", baseline) if isinstance(baseline, dict) else baseline
        if K == 1:
            mu_val = arr[0] if isinstance(arr[0], (int, float)) else arr[0][0]
            return f"one, $\\mu=[{mu_val}]^4$"
        # Multi-regime: list each μ row
        rows = arr if isinstance(arr[0], list) else [arr]
        return ", ".join(f"$[{r[0]}]^4$" for r in rows)

    def meta_str(cfg: dict) -> str:
        windows = cfg.get("meta_order", {}).get("windows", {}).get("items", [])
        if not windows:
            return "none"
        w = windows[0]
        targets = ", ".join(str(t) for t in w["target_event_types"]["items"]) \
            if isinstance(w["target_event_types"], dict) else str(w["target_event_types"])
        return f"$[{int(w['start_time'])},{int(w['end_time'])})$, $\\alpha={w['alpha']}$, types {targets}"

    rows = []
    for cfg_name, label in (
        ("first_milestone", "first\\_milestone.json"),
        ("two_regime_smoke", "two\\_regime\\_smoke.json"),
        ("meta_order_smoke", "meta\\_order\\_smoke.json"),
    ):
        cfg = json.load(open(REPO / "configs" / f"{cfg_name}.json"))
        T = cfg["horizon"]
        rows.append({
            "name": label,
            "horizon": f"{int(T) if T == int(T) else T} s",
            "regimes": regime_str(cfg),
            "meta": meta_str(cfg),
        })
    body = (
        "\\begin{tabular}{llll}\n"
        "  \\hline\n"
        "  Configuration & Horizon & Regimes & Meta-order \\\\\n"
        "  \\hline\n"
    )
    for r in rows:
        body += (
            f"  \\texttt{{{r['name']}}} & {r['horizon']} & {r['regimes']} & {r['meta']} \\\\\n"
        )
    body += "  \\hline\n\\end{tabular}"
    return write_table("tab_sim_params.tex", body)


# ---------------------------------------------------------------------------
# 2. tab_module_layout.tex — simulator package modules
# ---------------------------------------------------------------------------


def gen_module_layout() -> Path:
    """2-column table matching the shipped tab:module-layout.

    Module ordering and one-line descriptions match the dissertation prose.
    Includes __init__.py to total 16 rows.
    """
    rows = [
        ("config.py", "Typed configuration dataclasses, validators, JSON round trip."),
        ("kernels.py", "Exponential kernel, branching matrix, spectral radius."),
        ("latent.py", "CTMC regimes and meta-order schedules."),
        ("hawkes_core.py", "Ogata thinning loop with regime, meta-order, and event streams."),
        ("observation.py", "Top-of-book projection and minimum-spread censor policy."),
        ("artifact.py", "Run persistence and loading."),
        ("runner.py", "Single-run CLI and \\texttt{--experiment} dispatch."),
        ("evaluation.py", "\\texttt{summary.json} generation."),
        ("comparison.py", "Single-run envelope comparison."),
        ("ensemble.py", "Multi-seed ensemble harness."),
        ("ensemble_comparison.py", "Optional ensemble-level comparison."),
        ("envelope_bridge.py", "Non-destructive empirical-envelope bridge."),
        ("recovery.py", "Threshold regime and meta-order baseline."),
        ("recovery_eval.py", "Precision, recall, F1, and delay evaluation."),
        ("experiment.py", "End-to-end dissertation experiment runner."),
        ("__init__.py", "Public re-exports."),
    ]
    sim_dir = REPO / "src" / "simulator"
    actual = {p.name for p in sim_dir.glob("*.py")}
    missing = [m for m, _ in rows if m not in actual]
    extra = sorted(actual - {m for m, _ in rows})
    if missing or extra:
        # Fall back to a strict enumeration if the package layout drifts.
        rows = [(m, "(no description)") for m in sorted(actual)]
    body = (
        "\\begin{tabular}{ll}\n"
        "  \\hline\n"
        "  Module & Role \\\\\n"
        "  \\hline\n"
    )
    for m, desc in rows:
        body += f"  \\texttt{{{m.replace('_', '\\_')}}} & {desc} \\\\\n"
    body += "  \\hline\n\\end{tabular}"
    return write_table("tab_module_layout.tex", body)


# ---------------------------------------------------------------------------
# 3. tab_single_run_regression.tex — seed 42 from prompt 02 ensembles
# ---------------------------------------------------------------------------


def gen_single_run_regression() -> Path:
    base = REPO / "runs" / "ensembles_2026-05-06"
    rows = []
    for cfg_name, label in (
        ("first_milestone", "first\\_milestone"),
        ("meta_order_smoke", "meta\\_order\\_smoke"),
        ("two_regime_smoke", "two\\_regime\\_smoke"),
    ):
        member = base / f"{cfg_name}_50seeds" / f"ensemble_{cfg_name}_50seeds" / f"{cfg_name}_50seeds_42"
        s = json.load(open(member / "summary.json"))
        rows.append({
            "name": label,
            "total_rate": s["latent"]["total_rate"],
            "mean_spread": s["observation"]["spread_mean_ticks"],
            "rho": s["endogeneity"]["spectral_radius"],
            "crossed": s["observation"]["num_crossed_observations"],
        })
    body = (
        "\\begin{tabular}{lcccc}\n"
        "  \\hline\n"
        "  Configuration & Total rate (events/s) & Mean spread (ticks) & $\\rho(\\Phi)$ & Crossed obs. \\\\\n"
        "  \\hline\n"
    )
    for r in rows:
        body += (
            f"  \\texttt{{{r['name']}}} & {fmt(r['total_rate'], 2)} & "
            f"{fmt(r['mean_spread'], 2)} & {fmt(r['rho'], 4)} & {int(r['crossed'])} \\\\\n"
        )
    body += "  \\hline\n\\end{tabular}"
    return write_table("tab_single_run_regression.tex", body)


# ---------------------------------------------------------------------------
# 4. tab_spread_scaling.tex — from prompt 03 summary CSV
# ---------------------------------------------------------------------------


def gen_spread_scaling() -> Path:
    csv_path = REPO / "runs" / "spread_scaling_2026-05-06" / "spread_scaling_summary.csv"
    if not csv_path.exists():
        body = (
            "\\begin{tabular}{rrrrrrrr}\n"
            "  \\hline\n"
            "  $T$ & $N_{\\mathrm{obs}}$ & mean & median & p90 & p99 & max & final \\\\\n"
            "  \\hline\n"
            "  \\multicolumn{8}{c}{(prompt 03 not yet run; placeholder)} \\\\\n"
            "  \\hline\n"
            "\\end{tabular}"
        )
        return write_table("tab_spread_scaling.tex", body)
    rows = list(csv.DictReader(open(csv_path)))
    body = (
        "\\begin{tabular}{rrrrrrrr}\n"
        "  \\hline\n"
        "  $T$ (s) & $N_{\\mathrm{obs}}$ & mean & median & p90 & p99 & max & final \\\\\n"
        "  \\hline\n"
    )
    fields = (
        "observation.spread_mean_ticks",
        "observation.spread_median_ticks",
        "observation.spread_p90_ticks",
        "observation.spread_p99_ticks",
        "observation.spread_max_ticks",
        "observation.final_spread_ticks",
    )
    for r in rows:
        T = int(float(r["T"]))
        N = float(r["N_obs_mean"])
        vals = [float(r[f + ".mean"]) for f in fields]
        body += (
            f"  {T} & {N:.1f} & {vals[0]:.2f} & {vals[1]:.2f} & "
            f"{vals[2]:.2f} & {vals[3]:.2f} & {vals[4]:.2f} & {vals[5]:.2f} \\\\\n"
        )
    body += "  \\hline\n\\end{tabular}"
    return write_table("tab_spread_scaling.tex", body)


# ---------------------------------------------------------------------------
# 5. tab_recovery_example.tex — seeds 1–5 + 50-seed mean from prompt 04
# ---------------------------------------------------------------------------


def gen_recovery_example() -> Path:
    csv_path = REPO / "runs" / "recovery_meta_order_smoke_2026-05-06" / "recovery_summary.csv"
    if not csv_path.exists():
        body = (
            "\\begin{tabular}{ccccc}\n"
            "  \\hline\n"
            "  Seed & Precision & Recall & F1 & Mean delay (s) \\\\\n"
            "  \\hline\n"
            "  \\multicolumn{5}{c}{(prompt 04 not yet run; placeholder)} \\\\\n"
            "  \\hline\n"
            "\\end{tabular}"
        )
        return write_table("tab_recovery_example.tex", body)
    rows = list(csv.DictReader(open(csv_path)))
    rows_sorted = sorted(rows, key=lambda r: int(r["seed"]))
    seeds_1_5 = rows_sorted[:5]
    p_all = [float(r["precision"]) for r in rows_sorted]
    r_all = [float(r["recall"]) for r in rows_sorted]
    f_all = [float(r["f1"]) for r in rows_sorted]
    d_all = [float(r["mean_detection_delay_seconds"]) for r in rows_sorted]
    body = (
        "\\begin{tabular}{ccccc}\n"
        "  \\hline\n"
        "  Seed & Precision & Recall & F1 & Mean delay (s) \\\\\n"
        "  \\hline\n"
    )
    for r in seeds_1_5:
        body += (
            f"  {int(r['seed'])} & {float(r['precision']):.3f} & "
            f"{float(r['recall']):.3f} & {float(r['f1']):.3f} & "
            f"{float(r['mean_detection_delay_seconds']):.1f} \\\\\n"
        )
    body += (
        "  \\hline\n"
        f"  mean (seeds 1--5) & {statistics.mean(float(r['precision']) for r in seeds_1_5):.3f} & "
        f"{statistics.mean(float(r['recall']) for r in seeds_1_5):.3f} & "
        f"{statistics.mean(float(r['f1']) for r in seeds_1_5):.3f} & "
        f"{statistics.mean(float(r['mean_detection_delay_seconds']) for r in seeds_1_5):.1f} \\\\\n"
        f"  mean (seeds 1--{len(rows_sorted)}) $\\pm$ std & "
        f"{statistics.mean(p_all):.3f} $\\pm$ {statistics.pstdev(p_all):.3f} & "
        f"{statistics.mean(r_all):.3f} $\\pm$ {statistics.pstdev(r_all):.3f} & "
        f"{statistics.mean(f_all):.3f} $\\pm$ {statistics.pstdev(f_all):.3f} & "
        f"{statistics.mean(d_all):.2f} $\\pm$ {statistics.pstdev(d_all):.2f} \\\\\n"
        "  \\hline\n"
        "\\end{tabular}"
    )
    return write_table("tab_recovery_example.tex", body)


# ---------------------------------------------------------------------------
# 6. tab_baseline_parameters.tex — Appendix B baseline parameters
# ---------------------------------------------------------------------------


def gen_baseline_parameters() -> Path:
    """3-column table matching the shipped tab:baseline_parameters.

    Columns: Parameter | Description | Typical value.
    Six rows: $d$, $\\mu_i$, $\\alpha_{ij}$, $\\beta_{ij}$, $\\rho(\\Phi)$, $T$.
    Observation tick_size and min_spread_ticks live in the surrounding prose.
    """
    cfg = json.load(open(REPO / "configs" / "first_milestone.json"))
    d = cfg["dimension"]
    T = cfg["horizon"]
    a = cfg["kernel_params"]["alpha"]["data"][0][0]
    b = cfg["kernel_params"]["beta"]["data"][0][0]
    mu = cfg["baseline"]["mu"]["data"][0]
    rho = a / b * d
    body = (
        "\\begin{tabular}{lll}\n"
        "  \\hline\n"
        "  Parameter & Description & Typical value \\\\\n"
        "  \\hline\n"
        f"  $d$ & Number of Hawkes components & ${d}$ \\\\\n"
        f"  $\\mu_i$ & First-milestone baseline intensity & ${mu}$ \\\\\n"
        f"  $\\alpha_{{ij}}$ & Exponential excitation amplitude & ${a}$ \\\\\n"
        f"  $\\beta_{{ij}}$ & Exponential decay rate & ${b}$ \\\\\n"
        f"  $\\rho(\\Phi)$ & Spectral radius & ${rho:.4f}$ \\\\\n"
        f"  $T$ & First-milestone horizon & ${int(T) if T == int(T) else T}$ seconds \\\\\n"
        "  \\hline\n"
        "\\end{tabular}"
    )
    return write_table("tab_baseline_parameters.tex", body)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    paths = []
    paths.append(gen_sim_params())
    paths.append(gen_module_layout())
    paths.append(gen_single_run_regression())
    paths.append(gen_spread_scaling())
    paths.append(gen_recovery_example())
    paths.append(gen_baseline_parameters())
    for p in paths:
        rel = p.relative_to(REPO)
        print(f"  wrote {rel} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
