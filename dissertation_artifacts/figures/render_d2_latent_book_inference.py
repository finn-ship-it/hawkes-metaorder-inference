"""Plot-only renderer for dissertation Figure 4.5.

The renderer reads the unchanged historical summary and a plot-ready posterior
artifact recovered by the isolated, deterministic Stage 2 replay.  It imports no
simulation, calibration, EM, HMM, or evaluation code and writes only the figure.

The left panel shows latent native-event counts and the subset retained by the
projection before those retained types collapse into four projected directions.
The upper-right panel compares latent- and projected-stream recovery at the
default configuration.  The lower-right panel summarises the stored
full-sequence probability of the HMM's higher-rate state inside and outside the
known imposed periods.  The inferred state is not a direction-of-pressure
estimate and the posterior is not real-time information.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
SUMMARY_PATH = ARTIFACTS / "data" / "d2_latent_book_calibration_summary.json"
POSTERIOR_PATH = ARTIFACTS / "data" / "d2_latent_book_posterior_plot_data.json"
OUTPUT = HERE / "d2_latent_book_inference.pdf"

EXPECTED_SUMMARY_SHA256 = "5fe7b2c0355e9d6c89dffe06058e602323f38a35b7369ef16ac977c69d744d2e"
# The public copy changes only two external filesystem paths in provenance.
PUBLIC_SUMMARY_SHA256 = "1152ad3b68ff81e4d9485a3ede13ba3c65065f4f0c709399ccd6d2a157740e88"
EXPECTED_POSTERIOR_SHA256 = "63fbcb444b58427039412b91f39cacb873694f7d9a2773f93e53fb2c1eee53a2"
LOCKED_MATPLOTLIB = "3.10.9"
BLUE = "#0072B2"
GOLD = "#E69F00"
LIGHT_GREY = "#D9D9D9"

NATIVE_TYPES = (
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
NATIVE_LABELS = {
    "lo_deep_Ask": "Ask: deep limit order",
    "co_deep_Ask": "Ask: deep cancellation",
    "lo_top_Ask": "Ask: top limit order",
    "co_top_Ask": "Ask: top cancellation",
    "mo_Ask": "Ask: market order",
    "lo_inspread_Ask": "Ask: in-spread limit order",
    "lo_inspread_Bid": "Bid: in-spread limit order",
    "mo_Bid": "Bid: market order",
    "co_top_Bid": "Bid: top cancellation",
    "lo_top_Bid": "Bid: top limit order",
    "co_deep_Bid": "Bid: deep cancellation",
    "lo_deep_Bid": "Bid: deep limit order",
}
DEEP_TYPES = {"lo_deep_Ask", "co_deep_Ask", "co_deep_Bid", "lo_deep_Bid"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate():
    if matplotlib.__version__ != LOCKED_MATPLOTLIB:
        raise RuntimeError(
            f"matplotlib {LOCKED_MATPLOTLIB} is required; found {matplotlib.__version__}"
        )
    actual_summary = sha256(SUMMARY_PATH)
    actual_posterior = sha256(POSTERIOR_PATH)
    if actual_summary not in (EXPECTED_SUMMARY_SHA256, PUBLIC_SUMMARY_SHA256):
        raise ValueError(f"Unexpected historical-summary hash: {actual_summary}")
    if actual_posterior != EXPECTED_POSTERIOR_SHA256:
        raise ValueError(f"Unexpected posterior-artifact hash: {actual_posterior}")

    summary = json.loads(SUMMARY_PATH.read_text())
    posterior = json.loads(POSTERIOR_PATH.read_text())
    if posterior["source_files"]["published_summary"]["sha256"] != EXPECTED_SUMMARY_SHA256:
        raise ValueError("Posterior artifact does not identify the historical summary")
    if posterior["authoritative_repository_head"] != "cad0912932bde3ac70df4161b1e4d959f07bfc73":
        raise ValueError("Unexpected posterior-recovery code revision")
    if posterior["replay_scope"] != {
        "production_simulation": True,
        "projected_stream_hmm": True,
        "calibration_search": False,
        "eurusd_calibration": False,
        "projected_stream_em": False,
        "latent_stream_hmm": False,
    }:
        raise ValueError("Unexpected posterior-recovery scope")

    production = summary["production_run"]
    replay_production = posterior["production_run"]
    if int(production["n_latent"]) != 14_619 or int(production["n_observed"]) != 11_962:
        raise ValueError("Unexpected historical production counts")
    if int(replay_production["n_latent"]) != int(production["n_latent"]):
        raise ValueError("Replay latent count does not match")
    if int(replay_production["n_projected"]) != int(production["n_observed"]):
        raise ValueError("Replay projected count does not match")
    if replay_production["pre_projection_native_counts"] != production["pre_projection_native_counts"]:
        raise ValueError("Replay native counts do not match")
    if replay_production["post_projection_fx_counts"] != production["post_projection_fx_counts"]:
        raise ValueError("Replay projected counts do not match")
    if not np.isclose(
        float(replay_production["censoring_fraction"]),
        float(production["censoring_fraction"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Replay censoring fraction does not match")

    stored_hmm = summary["hmm_layer2_observed_d2"]
    replay_hmm = posterior["projected_stream_hmm"]
    if replay_hmm != stored_hmm:
        raise ValueError("Replay HMM summary does not match the historical summary")
    if not replay_hmm["hmm_converged"] or int(replay_hmm["n_windows_features"]) != 360:
        raise ValueError("Unexpected HMM identity")
    if not np.isclose(float(replay_hmm["f1"]), 0.29069767441860467, rtol=0.0, atol=1e-15):
        raise ValueError("Unexpected projected-stream F1")
    if float(summary["config"]["hmm_f1_threshold"]) != 0.7:
        raise ValueError("Unexpected experiment acceptance threshold")

    starts = np.asarray(posterior["window_starts_seconds"], dtype=float)
    probabilities = np.asarray(posterior["posterior_probability_imposed_period"], dtype=float)
    if starts.shape != (360,) or probabilities.shape != (360,):
        raise ValueError("Unexpected posterior dimensions")
    if not np.array_equal(starts, np.arange(0.0, 1800.0, 5.0)):
        raise ValueError("Unexpected posterior time grid")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Posterior probability lies outside [0,1]")
    if not np.isclose(
        float(np.mean(probabilities > 0.5)),
        float(replay_hmm["active_window_rate_posterior_gt_half"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Posterior array does not reproduce its stored summary")

    periods = replay_production["known_imposed_periods"]
    if len(periods) != 12:
        raise ValueError("Unexpected number of imposed periods")
    for index, period in enumerate(periods):
        if period["start_seconds"] != 150.0 * index or period["end_seconds"] != 150.0 * index + 30.0:
            raise ValueError("Unexpected imposed-period coordinates")

    rho_projected = float(summary["layer1_em_observed_d2"]["rho_spec"])
    rho_eurusd = float(summary["rho_d1_reference"])
    if not np.isclose(rho_projected, 0.7080400383306511, rtol=0.0, atol=1e-15):
        raise ValueError("Unexpected projected-stream branching ratio")
    if not np.isclose(rho_eurusd, 0.937308200001031, rtol=0.0, atol=1e-15):
        raise ValueError("Unexpected EUR/USD branching ratio")
    if not np.isclose(
        float(summary["three_way_rho_table"]["rho_eurusd_real_anchor"]),
        rho_eurusd,
        rtol=0.0,
        atol=3e-7,
    ):
        raise ValueError("Historical EUR/USD anchor is not the duplicated reference")
    return summary, starts, probabilities, periods


def main() -> None:
    summary, starts, probabilities, periods = load_and_validate()
    counts = summary["production_run"]["pre_projection_native_counts"]
    latent = np.asarray([counts[name] for name in NATIVE_TYPES], dtype=int)
    retained = np.asarray([0 if name in DEEP_TYPES else counts[name] for name in NATIVE_TYPES], dtype=int)
    n_latent = int(summary["production_run"]["n_latent"])
    n_projected = int(summary["production_run"]["n_observed"])
    removed_fraction = float(summary["production_run"]["censoring_fraction"])
    f1_latent = float(summary["hmm_layer2_latent_d2"]["f1"])
    f1_projected = float(summary["hmm_layer2_observed_d2"]["f1"])
    evaluation_threshold = float(summary["config"]["hmm_f1_threshold"])

    inside_period = np.asarray(
        [
            any(
                float(period["start_seconds"]) <= start < float(period["end_seconds"])
                for period in periods
            )
            for start in starts
        ],
        dtype=bool,
    )
    if int(np.sum(inside_period)) != 72:
        raise ValueError("Unexpected number of five-second windows inside imposed periods")
    posterior_groups = (probabilities[~inside_period], probabilities[inside_period])
    posterior_medians = tuple(float(np.median(group)) for group in posterior_groups)

    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    fig.suptitle("Latent-book projection and recovery under weak pressure", fontsize=9.5)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(0.76, 1.0),
        hspace=0.16,
        wspace=0.12,
    )
    count_ax = fig.add_subplot(grid[:, 0])
    recovery_ax = fig.add_subplot(grid[0, 1])
    posterior_ax = fig.add_subplot(grid[1, 1])

    y = np.arange(len(NATIVE_TYPES))
    count_ax.barh(
        y,
        latent,
        color=LIGHT_GREY,
        edgecolor="#777777",
        linewidth=0.45,
        label="Latent events",
    )
    count_ax.barh(
        y,
        retained,
        color=BLUE,
        edgecolor="#303030",
        linewidth=0.35,
        hatch="////",
        label="Events retained by the projection",
    )
    count_ax.set_yticks(y, [NATIVE_LABELS[name] for name in NATIVE_TYPES])
    count_ax.invert_yaxis()
    count_ax.set_xlabel("Number of events")
    count_ax.set_title("(a) Events before and after projection")
    count_ax.grid(True, axis="x", alpha=0.22)
    count_ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        handlelength=2.2,
        labelspacing=0.25,
    )
    count_ax.text(
        0.98,
        0.985,
        (
            rf"$n_{{\rm latent}}={n_latent:,}$"
            "\n"
            rf"$n_{{\rm projected}}={n_projected:,}$"
            "\n"
            rf"Removed fraction $={removed_fraction:.3f}$"
        ),
        transform=count_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    annotation_x = 1_050
    count_ax.text(
        annotation_x,
        1.45,
        "Removed by projection",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#555555",
    )
    count_ax.text(
        annotation_x,
        10.0,
        "Removed by projection",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#555555",
    )

    recovery_x = np.asarray([0.0, 1.0])
    recovery_f1 = np.asarray([f1_latent, f1_projected])
    recovery_ax.plot(recovery_x, recovery_f1, color="#777777", linewidth=1.0, zorder=1)
    recovery_ax.scatter(
        recovery_x,
        recovery_f1,
        s=42,
        c=(LIGHT_GREY, BLUE),
        edgecolors="#303030",
        linewidths=0.7,
        zorder=2,
    )
    recovery_ax.axhline(
        evaluation_threshold,
        color="#C44E52",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
    )
    recovery_ax.text(
        0.98,
        evaluation_threshold + 0.025,
        "Evaluation threshold",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#9C3035",
    )
    for x_value, f1_value in zip(recovery_x, recovery_f1, strict=True):
        recovery_ax.text(
            x_value,
            f1_value + 0.045,
            f"{f1_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    recovery_ax.set_xlim(-0.45, 1.45)
    recovery_ax.set_ylim(0.0, 1.0)
    recovery_ax.set_xticks(
        recovery_x,
        ("Latent simulated\nstream", "Projected simulated\nstream"),
    )
    recovery_ax.set_ylabel(r"$F_1$ score")
    recovery_ax.set_title("(b) Recovery remains weak\nbefore and after projection")
    recovery_ax.grid(True, axis="y", alpha=0.20)

    posterior_positions = np.asarray([0.0, 1.0])
    posterior_colours = ("#777777", GOLD)
    for position, values, colour in zip(
        posterior_positions,
        posterior_groups,
        posterior_colours,
        strict=True,
    ):
        sequence = np.arange(values.size, dtype=float)
        deterministic_jitter = (((sequence * 0.6180339887498949) % 1.0) - 0.5) * 0.34
        posterior_ax.scatter(
            position + deterministic_jitter,
            values,
            s=8,
            color=colour,
            alpha=0.30,
            edgecolors="none",
            zorder=1,
        )
    box = posterior_ax.boxplot(
        posterior_groups,
        positions=posterior_positions,
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"facecolor": "white", "edgecolor": "#303030", "linewidth": 0.9},
        whiskerprops={"color": "#303030", "linewidth": 0.8},
        capprops={"color": "#303030", "linewidth": 0.8},
    )
    for patch, colour in zip(box["boxes"], posterior_colours, strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.18)
    for position, median in zip(posterior_positions, posterior_medians, strict=True):
        posterior_ax.text(
            position,
            1.015,
            f"median {median:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#333333",
        )
    posterior_ax.set_xlim(-0.55, 1.55)
    posterior_ax.set_ylim(0.0, 1.09)
    posterior_ax.set_xticks(
        posterior_positions,
        ("Outside known\npressure periods", "Inside known\npressure periods"),
    )
    posterior_ax.set_ylabel("Full-sequence probability\nof the higher-rate state")
    posterior_ax.set_title("(c) Limited separation of\nknown pressure periods")
    posterior_ax.grid(True, axis="y", alpha=0.20)

    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={
            "Title": "Latent-book projection and recovery under weak pressure",
            "Author": "Dissertation artifact renderer",
            "Creator": "render_d2_latent_book_inference.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT}")
    print(f"SHA-256 {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
