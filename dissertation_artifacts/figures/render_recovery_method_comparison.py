#!/usr/bin/env python3
"""Render the Chapter 5 recovery-method figure from stored results only.

This is a confined plotting path. It reads the stored HMM JSON and threshold
CSV, validates the one-to-one seed correspondence, and writes only
``recovery_method_comparison.pdf``. It does not import any simulation or
recovery code.

The plotting versions are pinned in ``requirements_lock.txt``. Source hashes
are asserted so that the figure cannot silently be rebuilt from different
research results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCKED_MATPLOTLIB = "3.10.9"
LOCKED_NUMPY = "2.4.6"

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "dissertation_artifacts" / "data"
HMM_JSON = DATA_DIR / "recovery_hmm_meta_order_smoke.json"
THRESHOLD_CSV = DATA_DIR / "recovery_summary.csv"
OUTPUT_PDF = Path(__file__).resolve().parent / "recovery_method_comparison.pdf"

EXPECTED_SHA256 = {
    HMM_JSON: "81266b130ca3e5b6f2c6dc2b4e7edd0ede4abfe326549673bf2c8571da26a49b",
    THRESHOLD_CSV: "b233eb3943e86f1ea46804f5a65ee5847c506fa7a6e9dd45d7ab549a15626b10",
}

SEED_NAME = re.compile(r"^meta_order_smoke_50seeds_(\d+)$")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#8A8A8A"
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
            f"matplotlib {LOCKED_MATPLOTLIB} is required; found {matplotlib.__version__}"
        )
    if np.__version__ != LOCKED_NUMPY:
        raise RuntimeError(f"numpy {LOCKED_NUMPY} is required; found {np.__version__}")
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Stored input hash mismatch for {path}: expected {expected}, found {actual}"
            )


def _load_results() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with HMM_JSON.open(encoding="utf-8") as handle:
        hmm = json.load(handle)
    with THRESHOLD_CSV.open(newline="", encoding="utf-8") as handle:
        threshold_rows = list(csv.DictReader(handle))

    hmm_by_seed: dict[int, dict] = {}
    for record in hmm.get("per_seed", []):
        match = SEED_NAME.fullmatch(str(record.get("seed_dir", "")))
        if match is None:
            raise ValueError(f"Malformed HMM seed identifier: {record.get('seed_dir')!r}")
        seed = int(match.group(1))
        if seed in hmm_by_seed:
            raise ValueError(f"Duplicated HMM seed identifier: {seed}")
        hmm_by_seed[seed] = record

    threshold_by_seed: dict[int, dict] = {}
    for record in threshold_rows:
        raw_seed = str(record.get("seed", ""))
        if not raw_seed.isdigit():
            raise ValueError(f"Malformed threshold-method seed identifier: {raw_seed!r}")
        seed = int(raw_seed)
        if seed in threshold_by_seed:
            raise ValueError(f"Duplicated threshold-method seed identifier: {seed}")
        threshold_by_seed[seed] = record

    expected_seeds = set(range(1, 51))
    if set(hmm_by_seed) != expected_seeds or set(threshold_by_seed) != expected_seeds:
        raise ValueError(
            "The HMM JSON and threshold CSV must each contain exactly seeds 1 through 50"
        )
    if set(hmm_by_seed) != set(threshold_by_seed):
        raise ValueError("The two stored inputs do not contain the same seed identifiers")

    seeds = np.asarray(sorted(expected_seeds), dtype=int)
    threshold_f1 = np.asarray(
        [float(threshold_by_seed[seed]["f1"]) for seed in seeds], dtype=float
    )
    hmm_f1 = np.asarray([float(hmm_by_seed[seed]["f1"]) for seed in seeds], dtype=float)
    threshold_delay = np.asarray(
        [float(threshold_by_seed[seed]["mean_detection_delay_seconds"]) for seed in seeds],
        dtype=float,
    )
    hmm_delay = np.asarray(
        [float(hmm_by_seed[seed]["mean_detection_delay_seconds"]) for seed in seeds],
        dtype=float,
    )

    if not np.all(np.isfinite(np.r_[threshold_f1, hmm_f1])):
        raise ValueError("Stored F1 scores contain a non-finite value")
    if not np.all((0.0 <= np.r_[threshold_f1, hmm_f1]) & (np.r_[threshold_f1, hmm_f1] <= 1.0)):
        raise ValueError("Stored F1 values must lie in [0, 1]")

    stored_hmm_mean = float(hmm["hmm_aggregate"]["f1_mean"])
    stored_threshold_mean = float(hmm["threshold_aggregate"]["f1_mean"])
    if not np.isclose(hmm_f1.mean(), stored_hmm_mean, atol=1e-14, rtol=0.0):
        raise ValueError("Recomputed HMM mean does not match the stored aggregate")
    if not np.isclose(threshold_f1.mean(), stored_threshold_mean, atol=1e-14, rtol=0.0):
        raise ValueError("Recomputed threshold-method mean does not match the stored aggregate")

    for delays, scores in ((threshold_delay, threshold_f1), (hmm_delay, hmm_f1)):
        if np.any(np.isinf(delays)) or np.any(delays < 0.0):
            raise ValueError("Stored detection delays must be non-negative or missing")
        if np.any(np.isnan(delays) & (scores > 0.0)):
            raise ValueError("A recovered true period must have a finite detection delay")

    return seeds, threshold_f1, hmm_f1, threshold_delay, hmm_delay


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def render() -> Path:
    """Validate the stored results and write the comparison PDF only."""
    _assert_locked_inputs()
    seeds, threshold_f1, hmm_f1, threshold_delay, hmm_delay = _load_results()
    _set_style()

    fig, ax = plt.subplots(figsize=(6.20, 4.05))
    jitter = (((seeds * 37) % 50) - 24.5) / 24.5 * 0.035
    # Algebraically equal F1 scores can differ at floating-point roundoff.
    hmm_equal = np.isclose(hmm_f1, threshold_f1, atol=1e-12, rtol=0.0)
    hmm_lower = (hmm_f1 < threshold_f1) & ~hmm_equal
    n_higher = int(np.sum((hmm_f1 > threshold_f1) & ~hmm_equal))
    n_lower = int(np.sum(hmm_lower))
    n_equal = len(seeds) - n_higher - n_lower

    for index in range(len(seeds)):
        ax.plot(
            [jitter[index], 1.0 + jitter[index]],
            [threshold_f1[index], hmm_f1[index]],
            color=DARK if hmm_lower[index] else GREY,
            linewidth=0.95 if hmm_lower[index] else 0.55,
            linestyle="--" if hmm_lower[index] else "-",
            alpha=0.92 if hmm_lower[index] else 0.48,
            zorder=1,
        )

    ax.scatter(
        jitter,
        threshold_f1,
        s=19,
        marker="o",
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    ax.scatter(
        1.0 + jitter,
        hmm_f1,
        s=19,
        marker="s",
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    means = [float(threshold_f1.mean()), float(hmm_f1.mean())]
    ax.scatter(
        [0.0, 1.0],
        means,
        s=75,
        marker="D",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        f"Mean {means[0]:.3f}",
        (0.0, means[0]),
        xytext=(-8, -15),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=7.4,
    )
    ax.annotate(
        f"Mean {means[1]:.3f}",
        (1.0, means[1]),
        xytext=(8, 13),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.4,
    )

    ax.set_xlim(-0.24, 1.24)
    ax.set_ylim(max(0.0, float(min(threshold_f1.min(), hmm_f1.min())) - 0.15), 1.025)
    ax.set_xticks(
        [0.0, 1.0],
        ["Rate-and-imbalance\nthreshold method", "Hidden Markov\nmodel"],
    )
    ax.set_xlabel("Recovery method")
    ax.set_ylabel(r"$F_1$ score")
    ax.set_title("Recovery accuracy across 50 controlled simulations", pad=7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.text(
        0.02,
        0.04,
        f"HMM higher / lower / equal: {n_higher} / {n_lower} / {n_equal} seeds.\n"
        "Dashed paths mark lower HMM scores.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.3,
    )
    ax.text(
        0.02,
        0.96,
        "Diamond: overall mean",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "Recovery accuracy across 50 controlled simulations",
            "Author": "Finn Smith",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    print(f"HMM mean F1: {means[1]:.15f}")
    print(f"Threshold-method mean F1: {means[0]:.15f}")
    print(f"HMM higher/lower: {n_higher}/{n_lower} seeds")
    print(
        "Start-time differences, threshold method: "
        f"{dict(sorted(Counter(threshold_delay.tolist()).items()))}"
    )
    print(
        "Start-time differences, HMM: "
        f"{dict(sorted(Counter(hmm_delay.tolist()).items()))}"
    )
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Output SHA-256: {_sha256(OUTPUT_PDF)}")
    return OUTPUT_PDF


if __name__ == "__main__":
    render()
