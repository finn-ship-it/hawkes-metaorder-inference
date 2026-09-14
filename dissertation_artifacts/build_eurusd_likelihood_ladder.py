"""Build artefact: EUR/USD Layer-1 likelihood / AIC / BIC model ladder.

Assembles a model-comparison ladder for the 2021-07-20 EUR/USD four-type
event stream from EXISTING fits only (no new model fitting):

  Data/Data output/EURUSD output/layer1/layer1_models_summary.csv
  dissertation_artifacts/data/eurusd_layer1_em_summary.json

Rows (all at N = 97,952, the four-type primary day):
  - Poisson_4type                              (independence baseline)
  - ExpHawkes_4type_jitter_random_ms           (single-exp Hawkes)
  - SumExpHawkes_4type_R4  (L-BFGS-B)          historical, NON-CONVERGED/superseded
  - SumExpHawkes_4type_R4  (EM-converged)      the dissertation anchor

The non-EM rows are taken verbatim from the CSV artefact. For the EM row,
AIC and BIC are computed from the pinned EM log-likelihood, N, and parameter
count k = 68 (4 baselines + R*M*M = 4*4*4 = 64 kernel coefficients), using the
SAME convention as the CSV:  AIC = 2k - 2 ll,  BIC = k ln N - 2 ll.
The convention is verified by reproducing the CSV's L-BFGS-B AIC/BIC to a
tight tolerance; the script STOPs on mismatch (artifact wins over memory).

Outputs:
  dissertation_artifacts/data/eurusd_layer1_likelihood_ladder.json
  dissertation_artifacts/tables/tab_eurusd_likelihood_ladder.tex
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
CSV_PATH = WORKSPACE / "Data" / "Data output" / "EURUSD output" / "layer1" / "layer1_models_summary.csv"
EM_JSON = HERE / "data" / "eurusd_layer1_em_summary.json"
OUT_JSON = HERE / "data" / "eurusd_layer1_likelihood_ladder.json"
OUT_TEX = HERE / "tables" / "tab_eurusd_likelihood_ladder.tex"

N_PRIMARY = 97952
K_SUMEXP_R4 = 68  # 4 mu + 4*4*4 alpha
AIC_BIC_TOL = 1.0e-3


def aic(k, ll):
    return 2.0 * k - 2.0 * ll


def bic(k, ll, n):
    return k * math.log(n) - 2.0 * ll


def load_csv_rows():
    rows = {}
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


def main():
    csv_rows = load_csv_rows()
    em = json.loads(EM_JSON.read_text())["primary_day_2021_07_20"]
    em_ll = em["em_result"]["log_likelihood"]
    em_rho = em["em_result"]["rho_spec"]
    em_n = em["n_events"]
    em_niter = em["em_result"]["n_iter"]
    assert em_n == N_PRIMARY, f"EM N {em_n} != {N_PRIMARY}"

    # --- verify the AIC/BIC convention against the CSV L-BFGS-B SumExp row ---
    lb = csv_rows["SumExpHawkes_4type_R4"]
    lb_ll = float(lb["ll"])
    lb_k = int(lb["n_params"])
    lb_n = int(lb["N"])
    aic_check = abs(aic(lb_k, lb_ll) - float(lb["AIC"]))
    bic_check = abs(bic(lb_k, lb_ll, lb_n) - float(lb["BIC"]))
    print(f"  convention self-check vs CSV L-BFGS-B: dAIC={aic_check:.2e} dBIC={bic_check:.2e}")
    if aic_check > AIC_BIC_TOL or bic_check > AIC_BIC_TOL:
        raise SystemExit(
            "STOP: AIC/BIC convention does not reproduce the CSV L-BFGS-B row; "
            f"dAIC={aic_check}, dBIC={bic_check}."
        )
    if lb_k != K_SUMEXP_R4 or lb_n != N_PRIMARY:
        raise SystemExit(f"STOP: unexpected L-BFGS-B k/N: k={lb_k}, N={lb_n}.")

    poisson = csv_rows["Poisson_4type"]
    poisson_ll = float(poisson["ll"])

    def row_from_csv(model, label, status):
        r = csv_rows[model]
        ll = float(r["ll"])
        return {
            "model": model,
            "label": label,
            "status": status,
            "N": int(r["N"]),
            "k_params": int(r["n_params"]),
            "log_likelihood": ll,
            "rho_spec": float(r["rho_spec"]),
            "kernel": r["kernel"],
            "AIC": float(r["AIC"]),
            "BIC": float(r["BIC"]),
            "delta_ll_vs_poisson_4type": ll - poisson_ll,
            "source": "layer1_models_summary.csv",
        }

    em_row = {
        "model": "SumExpHawkes_4type_R4_EM",
        "label": "Sum-exp Hawkes (R=4), EM-converged",
        "status": "converged (tol); dissertation empirical anchor",
        "N": int(em_n),
        "k_params": K_SUMEXP_R4,
        "log_likelihood": em_ll,
        "rho_spec": em_rho,
        "kernel": "sumexp_R=4",
        "n_iter": int(em_niter),
        "AIC": aic(K_SUMEXP_R4, em_ll),
        "BIC": bic(K_SUMEXP_R4, em_ll, em_n),
        "delta_ll_vs_poisson_4type": em_ll - poisson_ll,
        "ll_improvement_over_lbfgsb": em_ll - lb_ll,
        "source": "eurusd_layer1_em_summary.json (AIC/BIC computed here, same convention as CSV)",
    }

    ladder = [
        row_from_csv("Poisson_4type", "Poisson (4-type, independent)", "baseline"),
        row_from_csv(
            "ExpHawkes_4type_jitter_random_ms",
            "Single-exp Hawkes (4-type, random\\_ms jitter)",
            "converged",
        ),
        row_from_csv(
            "SumExpHawkes_4type_R4",
            "Sum-exp Hawkes (R=4), L-BFGS-B",
            "NON-CONVERGED; historical, superseded by the EM fit",
        ),
        em_row,
    ]

    payload = {
        "artifact": "eurusd_layer1_likelihood_ladder",
        "description": (
            "Layer-1 model-comparison ladder for the 2021-07-20 EUR/USD four-type "
            "event stream (N=97,952). Non-EM rows verbatim from "
            "layer1_models_summary.csv; the EM row's AIC/BIC computed from the "
            "pinned EM log-likelihood with k=68 under the CSV's convention "
            "(AIC=2k-2ll, BIC=k ln N - 2ll), self-checked against the CSV L-BFGS-B "
            "row. No new model fitting."
        ),
        "N_primary": N_PRIMARY,
        "aic_bic_convention": "AIC = 2k - 2*loglik ; BIC = k*ln(N) - 2*loglik",
        "convention_self_check_vs_csv_lbfgsb": {
            "aic_abs_diff": aic_check,
            "bic_abs_diff": bic_check,
            "tolerance": AIC_BIC_TOL,
            "passed": True,
        },
        "ladder": ladder,
        "notes": (
            "Lower AIC/BIC is better. The EM-converged sum-exp fit attains the "
            "lowest AIC and BIC of the ladder and is also the legitimate optimiser "
            "fixed point; the L-BFGS-B sum-exp row is retained only as the "
            "historical, non-converged reading it supersedes."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {OUT_JSON}")

    # --- LaTeX table ---
    def fmt(x):
        return f"{x:,.0f}"

    lines = [
        "% Auto-generated by build_eurusd_likelihood_ladder.py (P6b). Do not edit by hand.",
        "\\begin{tabular}{lrrrrr}",
        "\\hline\\hline",
        "Model & $k$ & $\\log L$ & $\\rho(\\Phi)$ & AIC & BIC \\\\",
        "\\hline",
    ]
    disp = [
        ("Poisson (4-type)", ladder[0]),
        ("Single-exp Hawkes", ladder[1]),
        ("Sum-exp Hawkes $R{=}4$ (L-BFGS-B)\\textsuperscript{$\\dagger$}", ladder[2]),
        ("Sum-exp Hawkes $R{=}4$ (EM)\\textsuperscript{$\\ast$}", ladder[3]),
    ]
    for name, r in disp:
        lines.append(
            f"{name} & {r['k_params']} & {fmt(r['log_likelihood'])} & "
            f"{r['rho_spec']:.4f} & {fmt(r['AIC'])} & {fmt(r['BIC'])} \\\\"
        )
    lines += [
        "\\hline\\hline",
        "\\end{tabular}",
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_TEX}")

    print("\n  Ladder (N=97,952):")
    for r in ladder:
        print(
            f"    {r['label']:42s} k={r['k_params']:>2} "
            f"ll={r['log_likelihood']:>13.2f} rho={r['rho_spec']:.4f} "
            f"AIC={r['AIC']:>13.2f} BIC={r['BIC']:>13.2f}"
        )


if __name__ == "__main__":
    main()
