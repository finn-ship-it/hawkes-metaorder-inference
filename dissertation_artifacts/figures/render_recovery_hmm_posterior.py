#!/usr/bin/env python3
"""Render the Chapter 5 illustrative HMM posterior from frozen results only.

This plot-only path reads the stored fifty-seed HMM summary and the locked
configuration, validates both inputs, and writes only
``recovery_hmm_posterior.pdf``.  It imports no simulator, HMM, recovery, or
evaluation code and performs no research computation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCKED_MATPLOTLIB = "3.10.9"
LOCKED_NUMPY = "2.4.6"

REPO = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO / "dissertation_artifacts" / "data" / (
    "recovery_hmm_meta_order_smoke.json"
)
CONFIG_PATH = REPO / "configs" / "meta_order_smoke.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "recovery_hmm_posterior.pdf"

EXPECTED_SHA256 = {
    SUMMARY_PATH: "81266b130ca3e5b6f2c6dc2b4e7edd0ede4abfe326549673bf2c8571da26a49b",
    CONFIG_PATH: "97ac34661d225e15d5b66c635ec2bb7f2dd9c4ca41d8e550c3226e411efa58bf",
}
SEED_PATTERN = re.compile(r"^meta_order_smoke_50seeds_(\d+)$")

BLUE = "#0072B2"
RED = "#D55E00"
DARK = "#222222"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_locked_inputs() -> None:
    if matplotlib.__version__ != LOCKED_MATPLOTLIB:
        raise RuntimeError(
            f"matplotlib {LOCKED_MATPLOTLIB} is required; "
            f"found {matplotlib.__version__}"
        )
    if np.__version__ != LOCKED_NUMPY:
        raise RuntimeError(f"numpy {LOCKED_NUMPY} is required; found {np.__version__}")
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Stored input hash mismatch for {path}: "
                f"expected {expected}, found {actual}"
            )


def _load_representative_trace() -> tuple[int, float, np.ndarray, np.ndarray, float, float]:
    with SUMMARY_PATH.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    per_seed = summary.get("per_seed", [])
    if len(per_seed) != 50:
        raise ValueError("Stored HMM summary must contain exactly fifty seed records")
    by_seed = {}
    for record in per_seed:
        match = SEED_PATTERN.fullmatch(str(record.get("seed_dir", "")))
        if match is None:
            raise ValueError("Malformed seed identifier")
        seed = int(match.group(1))
        if seed in by_seed:
            raise ValueError("Duplicated seed identifier")
        by_seed[seed] = record
    if set(by_seed) != set(range(1, 51)):
        raise ValueError("Stored HMM summary must contain seeds 1 through 50")
    per_seed = [by_seed[seed] for seed in sorted(by_seed)]

    f1_values = np.asarray([float(record["f1"]) for record in per_seed], dtype=float)
    if not np.all(np.isfinite(f1_values)) or np.any((f1_values < 0.0) | (f1_values > 1.0)):
        raise ValueError("Stored HMM F1 values must be finite and lie in [0, 1]")
    if not np.isclose(
        f1_values.mean(), float(summary["hmm_aggregate"]["f1_mean"]),
        atol=1e-14, rtol=0.0,
    ):
        raise ValueError("Per-seed F1 values do not match the stored mean")
    median_f1 = float(np.median(f1_values))
    representative_index = int(np.argmin(np.abs(f1_values - median_f1)))
    record = per_seed[representative_index]

    match = SEED_PATTERN.fullmatch(str(record.get("seed_dir", "")))
    if match is None:
        raise ValueError("Malformed representative-seed identifier")
    seed = int(match.group(1))

    active_state = int(record["active_state_index"])
    posterior = np.asarray(record["posterior"], dtype=float)
    starts = np.asarray(record["window_starts"], dtype=float)
    horizon = float(record["horizon"])
    if posterior.shape != (24, 2) or starts.shape != (24,):
        raise ValueError("Unexpected posterior-trace dimensions")
    if active_state not in (0, 1):
        raise ValueError("Unexpected active-state index")
    if not np.allclose(starts, np.arange(0.0, 120.0, 5.0), atol=0.0, rtol=0.0):
        raise ValueError("Unexpected five-second time grid")
    if horizon != 120.0:
        raise ValueError("Unexpected representative-seed horizon")
    if not np.all(np.isfinite(posterior)) or np.any((posterior < 0.0) | (posterior > 1.0)):
        raise ValueError("Stored posterior probabilities must lie in [0, 1]")
    if not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Stored posterior rows do not sum to one")

    window_items = config["meta_order"]["windows"]["items"]
    if len(window_items) != 1:
        raise ValueError("Expected one known directional-pressure period")
    pressure_start = float(window_items[0]["start_time"])
    pressure_end = float(window_items[0]["end_time"])
    if (pressure_start, pressure_end) != (20.0, 80.0):
        raise ValueError("Unexpected known directional-pressure period")

    f1 = float(record["f1"])
    return (
        seed,
        f1,
        starts,
        posterior[:, active_state],
        pressure_start,
        pressure_end,
    )


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def render() -> Path:
    """Validate the frozen inputs and write the illustrative posterior PDF."""
    _assert_locked_inputs()
    seed, f1, starts, probability, pressure_start, pressure_end = (
        _load_representative_trace()
    )
    _set_style()

    fig, axis = plt.subplots(figsize=(7.20, 3.08))
    axis.plot(
        starts,
        probability,
        color=BLUE,
        linewidth=1.65,
        label="Recovered probability (full sequence)",
        zorder=3,
    )
    axis.fill_between(
        starts,
        0.0,
        probability,
        color=BLUE,
        alpha=0.16,
        linewidth=0.0,
        zorder=1,
    )
    axis.axvspan(
        pressure_start,
        pressure_end,
        color=RED,
        alpha=0.16,
        label="Known directional-pressure period",
        zorder=0,
    )

    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Probability assigned to the\nhigher-rate activity state")
    fig.suptitle(
        "Recovering a known directional-pressure period\n"
        f"(seed {seed}, $F_1 = {f1:.3f}$)",
        y=0.985,
        fontsize=9.5,
    )
    axis.set_xlim(0.0, 120.0)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks(np.arange(0.0, 121.0, 20.0))
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(True, color="#B8B8B8", alpha=0.32, linewidth=0.55)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.835),
        frameon=False,
        ncol=2,
        columnspacing=1.4,
        handlelength=2.5,
    )

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.17, top=0.69)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        metadata={
            "Title": "Recovering a known directional-pressure period",
            "Subject": "Plot-only rendering from frozen fifty-seed HMM results",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    render()
