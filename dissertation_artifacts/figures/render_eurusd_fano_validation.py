#!/usr/bin/env python3
"""Render the EUR/USD Fano-factor forward-validation figure.

This is a plot-only path. It reads two stored summaries, validates their
hashes and the four displayed values, and writes one PDF. It does not import
or call event construction, fitting, simulation, or validation code.

The empirical value is compared with the median and minimum-maximum range
across 30 independently generated fitted-model days. The stored summaries do
not contain the two quartile endpoints, so no box or quartile interval is
reconstructed from the stored interquartile-range width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EMPIRICAL_PATH = (
    REPO
    / "dissertation_artifacts"
    / "data"
    / "plot_inputs"
    / "eurusd_empirical_fano.json"
)
SIMULATION_PATH = (
    REPO
    / "dissertation_artifacts"
    / "data"
    / "plot_inputs"
    / "eurusd_simulated_fano.json"
)
DEFAULT_OUTPUT = HERE / "eurusd_fano_forward_validation.pdf"

EXPECTED_SHA256 = {
    EMPIRICAL_PATH: "7546b7c2c0009abc4af868e918f7fc73cc1f24b94307bacf91f99521afdbe208",
    SIMULATION_PATH: "7cb4720844029b63529c2c30d5a81f4bb77ecc4139b36979c961f26b332685f9",
}
LOCKED_MATPLOTLIB = "3.10.9"
LOCKED_NUMPY = "2.4.6"
WINDOWS = np.asarray([1.0, 10.0, 60.0, 300.0])
EXPECTED_EMPIRICAL = np.asarray(
    [3.6779183152443524, 11.337803701283775, 46.95833968565101, 198.49671721296605]
)
EXPECTED_MEDIAN = np.asarray(
    [6.0782462954308905, 24.356611140255026, 92.18250503012922, 194.17142912836954]
)
EXPECTED_MINIMUM = np.asarray(
    [5.788154643165645, 21.679499149208393, 77.80304663930467, 154.8489447291807]
)
EXPECTED_MAXIMUM = np.asarray(
    [6.6503765827309085, 28.987167081079384, 119.35930698435361, 297.3456031038011]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matplotlib.__version__ != LOCKED_MATPLOTLIB:
        raise RuntimeError(
            f"matplotlib {LOCKED_MATPLOTLIB} is required; found {matplotlib.__version__}"
        )
    if np.__version__ != LOCKED_NUMPY:
        raise RuntimeError(f"numpy {LOCKED_NUMPY} is required; found {np.__version__}")
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Stored input hash mismatch for {path}: expected {expected}, found {actual}"
            )

    empirical = json.loads(EMPIRICAL_PATH.read_text(encoding="utf-8"))
    simulation = json.loads(SIMULATION_PATH.read_text(encoding="utf-8"))
    if int(empirical["n_events"]) != 97_952:
        raise ValueError("Unexpected empirical event count")
    if int(simulation["n_members"]) != 30 or simulation["seeds"] != list(range(1, 31)):
        raise ValueError("Expected exactly the stored simulations for seeds 1 through 30")
    if float(simulation["T_seconds"]) != 86_400.0:
        raise ValueError("Expected one-day fitted-model simulations")

    empirical_values = np.asarray(
        [float(empirical["fano"][str(int(window))]) for window in WINDOWS]
    )
    fano_table = simulation["scalar_table"]["fano"]
    median = np.asarray(
        [float(fano_table[f"fano_{int(window)}"]["median"]) for window in WINDOWS]
    )
    minimum = np.asarray(
        [float(fano_table[f"fano_{int(window)}"]["min"]) for window in WINDOWS]
    )
    maximum = np.asarray(
        [float(fano_table[f"fano_{int(window)}"]["max"]) for window in WINDOWS]
    )
    inside = np.asarray(
        [bool(fano_table[f"fano_{int(window)}"]["inside_range"]) for window in WINDOWS]
    )

    checks = (
        (empirical_values, EXPECTED_EMPIRICAL, "empirical values"),
        (median, EXPECTED_MEDIAN, "simulation medians"),
        (minimum, EXPECTED_MINIMUM, "simulation minima"),
        (maximum, EXPECTED_MAXIMUM, "simulation maxima"),
    )
    for actual, expected, label in checks:
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-13):
            raise ValueError(f"Stored {label} do not match the validated values")
    if not np.array_equal(inside, np.asarray([False, False, False, True])):
        raise ValueError("Unexpected empirical inclusion pattern")
    if not np.all(empirical_values[:3] < minimum[:3]):
        raise ValueError("Empirical values are not below all shorter-window simulations")
    if not minimum[3] <= empirical_values[3] <= maximum[3]:
        raise ValueError("The 300-second empirical value is not within the simulation range")
    return empirical_values, median, minimum, maximum


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def render(output: Path, layout: str) -> None:
    empirical, median, minimum, maximum = load_and_validate()
    set_style()

    if layout == "categorical":
        x = np.arange(len(WINDOWS), dtype=float)
        offset = 0.105
        empirical_x = x - offset
        simulation_x = x + offset
    elif layout == "log-window":
        x = WINDOWS
        empirical_x = x / 1.075
        simulation_x = x * 1.075
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    fig, ax = plt.subplots(figsize=(6.25, 3.65))
    lower = median - minimum
    upper = maximum - median
    simulation = ax.errorbar(
        simulation_x,
        median,
        yerr=np.vstack([lower, upper]),
        fmt="s",
        markersize=5.4,
        markerfacecolor="white",
        markeredgecolor="#202020",
        markeredgewidth=1.0,
        color="#4D4D4D",
        ecolor="#4D4D4D",
        elinewidth=1.6,
        capsize=4.0,
        capthick=1.1,
        linestyle="none",
        label="Fitted-model simulation median and minimum-maximum range (30 days)",
        zorder=2,
    )
    empirical_points = ax.scatter(
        empirical_x,
        empirical,
        marker="D",
        s=34,
        facecolor="black",
        edgecolor="white",
        linewidth=0.45,
        label="Empirical EUR/USD day",
        zorder=3,
    )

    if layout == "categorical":
        ax.set_xticks(x, ["1", "10", "60", "300"])
        ax.set_xlim(-0.5, 3.5)
    else:
        ax.set_xscale("log")
        ax.set_xticks(WINDOWS, ["1", "10", "60", "300"])
        ax.set_xlim(0.65, 465.0)
    ax.set_yscale("log")
    ax.set_ylim(2.6, 380.0)
    ax.set_yticks([3, 5, 10, 20, 50, 100, 200, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Counting-window length (seconds)")
    ax.set_ylabel("Fano factor (dimensionless)")
    ax.set_title("Empirical and simulated count dispersion", pad=7)
    ax.grid(axis="y", which="both", color="#D5D5D5", linewidth=0.5, alpha=0.9)
    ax.text(
        0.02,
        0.97,
        "Empirical value below all 30 simulations at 1, 10 and 60 seconds",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
    )
    handles = [empirical_points, simulation]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.8)
    fig.tight_layout(pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Empirical and simulated count dispersion",
            "Author": "Dissertation artifact renderer",
            "Creator": "render_eurusd_fano_validation.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=("categorical", "log-window"),
        default="categorical",
        help="Horizontal representation used for the four stored windows.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.output.resolve(), args.layout)
