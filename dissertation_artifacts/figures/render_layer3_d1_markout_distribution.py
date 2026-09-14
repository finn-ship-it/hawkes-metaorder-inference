#!/usr/bin/env python3
"""Render the revised Chapter 6 Figure 6.1 from stored results only.

This confined plotting path reads only
``layer3_d1_skew_markout_comparison.json`` and writes only
``layer3_d1_markout_distribution.pdf``. It does not import the simulator,
the HMM, the fill harness, or the markout calculation.

Episode A in the hash-locked stored file is the setting without the
posterior-based adjustment (s=0); Episode B is the setting with it (s=1).
The stored experiment used a native quote tick of 1e-5. Since the JSON stores
markouts in price units but does not repeat that tick-size field, the renderer
uses the fixed experiment value and validates the resulting half-tick grid.

The left panel pools valid five-second post-fill records. Consequently, fills
rather than seeds receive equal weight there: seeds producing more valid fills
contribute more observations. The right-panel fill proportion is the number of
fills divided by the two side-specific quote opportunities evaluated at every
projected simulated market event. It is not a five-second measure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCKED_MATPLOTLIB = "3.10.9"
LOCKED_NUMPY = "2.4.6"
NATIVE_TICK_SIZE = 1e-5

REPO = Path(__file__).resolve().parents[2]
SOURCE_JSON = (
    REPO
    / "dissertation_artifacts"
    / "data"
    / "layer3_d1_skew_markout_comparison.json"
)
OUTPUT_PDF = Path(__file__).resolve().parent / "layer3_d1_markout_distribution.pdf"
EXPECTED_SOURCE_SHA256 = "74d7b94594bae38f747b749f2d2a2b8d6b16d4a52b287478dd42fbe266379e88"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#8A8A8A"


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
    actual = _sha256(SOURCE_JSON)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Stored input hash mismatch for {SOURCE_JSON}: "
            f"expected {EXPECTED_SOURCE_SHA256}, found {actual}"
        )


def _load_results() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    with SOURCE_JSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != "1.0":
        raise ValueError("Unexpected stored markout schema version")
    if float(payload.get("horizon_seconds", -1.0)) != 600.0:
        raise ValueError("The stored simulation horizon must be 600 seconds")
    if float(payload.get("headline_horizon_seconds", -1.0)) != 5.0:
        raise ValueError("The stored headline horizon must be five seconds")
    if int(payload.get("n_seeds_evaluated", -1)) != 30:
        raise ValueError("The stored result must contain 30 evaluated seeds")

    per_seed = payload.get("per_seed", [])
    if len(per_seed) != 30:
        raise ValueError("The stored per-seed array must contain 30 records")
    seed_ids = [int(record["seed"]) for record in per_seed]
    if len(set(seed_ids)) != len(seed_ids) or set(seed_ids) != set(range(1, 31)):
        raise ValueError("The stored result must contain unique seeds 1 through 30")

    absolute_a_ticks: list[float] = []
    absolute_b_ticks: list[float] = []
    ordered = sorted(per_seed, key=lambda record: int(record["seed"]))
    fill_rate_a: list[float] = []
    fill_rate_b: list[float] = []
    seed_mean_a: list[float] = []
    seed_mean_b: list[float] = []
    ratios: list[float] = []

    for record in ordered:
        n_events = int(record["n_events"])
        n_fills_a = int(record["n_fills_episode_a"])
        n_fills_b = int(record["n_fills_episode_b"])
        if float(record.get("horizon", -1.0)) != 600.0:
            raise ValueError("Every stored seed must use the 600-second simulation horizon")
        if n_events <= 0 or not (0 < n_fills_a <= 2 * n_events) or not (0 <= n_fills_b <= 2 * n_events):
            raise ValueError("Stored fill counts must fit the available quote opportunities")

        rate_a = float(record["fill_rate_episode_a"])
        rate_b = float(record["fill_rate_episode_b"])
        if not np.isclose(rate_a, n_fills_a / (2.0 * n_events), atol=1e-15, rtol=0.0):
            raise ValueError("Stored s=0 fill rate is inconsistent with its counts")
        if not np.isclose(rate_b, n_fills_b / (2.0 * n_events), atol=1e-15, rtol=0.0):
            raise ValueError("Stored s=1 fill rate is inconsistent with its counts")
        fill_rate_a.append(100.0 * rate_a)
        fill_rate_b.append(100.0 * rate_b)
        ratios.append(rate_b / rate_a)

        records_a = record["per_fill_signed_markout_5s_episode_a"]
        records_b = record["per_fill_signed_markout_5s_episode_b"]
        aggregate_a = record["aggregates_episode_a"]["5.0"]
        aggregate_b = record["aggregates_episode_b"]["5.0"]
        if len(records_a) != int(aggregate_a["n_fills"]):
            raise ValueError("The stored s=0 per-fill array does not match its five-second count")
        if len(records_b) != int(aggregate_b["n_fills"]):
            raise ValueError("The stored s=1 per-fill array does not match its five-second count")

        for per_fill, destination, aggregate in (
            (records_a, absolute_a_ticks, aggregate_a),
            (records_b, absolute_b_ticks, aggregate_b),
        ):
            if not per_fill:
                raise ValueError("Each plotted arm must contain valid five-second fills")
            seed_absolute = []
            for item in per_fill:
                if item.get("side") not in {"bid", "ask"}:
                    raise ValueError("Unexpected fill side in the stored per-fill array")
                value = float(item["signed_markout"])
                if not np.isfinite(value):
                    raise ValueError("Non-finite value in the stored per-fill array")
                destination.append(abs(value) / NATIVE_TICK_SIZE)
                seed_absolute.append(abs(value))
            if not np.isclose(
                np.mean(seed_absolute), float(aggregate["mean_toxicity"]),
                atol=1e-16, rtol=0.0,
            ):
                raise ValueError("Per-fill movements do not reproduce the seed-level mean")

        seed_mean_a.append(float(aggregate_a["mean_toxicity"]))
        seed_mean_b.append(float(aggregate_b["mean_toxicity"]))

    values_a = np.asarray(absolute_a_ticks, dtype=float)
    values_b = np.asarray(absolute_b_ticks, dtype=float)
    if np.max(np.abs(values_a * 2.0 - np.rint(values_a * 2.0))) >= 1e-9:
        raise ValueError("Stored s=0 movements do not lie on the expected half-tick grid")
    if np.max(np.abs(values_b * 2.0 - np.rint(values_b * 2.0))) >= 1e-9:
        raise ValueError("Stored s=1 movements do not lie on the expected half-tick grid")

    across = payload["across_seeds"]["5.0"]
    mean_a = float(np.mean(seed_mean_a))
    mean_b = float(np.mean(seed_mean_b))
    if not np.isclose(
        mean_a, float(across["mean_toxicity_episode_a"]), atol=1e-16, rtol=0.0
    ):
        raise ValueError("Recomputed s=0 mean does not match the stored equal-seed aggregate")
    if not np.isclose(
        mean_b, float(across["mean_toxicity_episode_b"]), atol=1e-16, rtol=0.0
    ):
        raise ValueError("Recomputed s=1 mean does not match the stored equal-seed aggregate")
    equal_seed_means_ticks = (mean_a / NATIVE_TICK_SIZE, mean_b / NATIVE_TICK_SIZE)

    stored_ratio = float(payload["gate_summary"]["median_fill_rate_ratio_b_over_a"])
    if not np.isclose(float(np.median(ratios)), stored_ratio, atol=1e-15, rtol=0.0):
        raise ValueError("Recomputed fill-retention statistic does not match the stored result")

    return (
        np.asarray([int(record["seed"]) for record in ordered], dtype=int),
        values_a,
        values_b,
        np.asarray(fill_rate_a, dtype=float),
        np.asarray(fill_rate_b, dtype=float),
        float(equal_seed_means_ticks[0]),
        float(equal_seed_means_ticks[1]),
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
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_cumulative_percentage(
    ax: plt.Axes, values: np.ndarray, *, color: str, linestyle: str, label: str
) -> None:
    unique, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts) / len(values) * 100.0
    ax.step(
        np.r_[0.0, unique],
        np.r_[0.0, cumulative],
        where="post",
        color=color,
        linestyle=linestyle,
        linewidth=1.65,
        label=label,
    )


def render() -> Path:
    """Validate the stored result and write the revised Figure 6.1 PDF only."""
    _assert_locked_inputs()
    (
        seeds,
        values_a,
        values_b,
        fill_rate_a,
        fill_rate_b,
        equal_seed_mean_a,
        equal_seed_mean_b,
    ) = _load_results()
    _set_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.25, 4.15),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    _plot_cumulative_percentage(
        axes[0],
        values_a,
        color=BLUE,
        linestyle="-",
        label=(
            "Posterior adjustment off ($s=0$)\n"
            f"n={len(values_a):,} valid records"
        ),
    )
    _plot_cumulative_percentage(
        axes[0],
        values_b,
        color=ORANGE,
        linestyle="--",
        label=(
            "Posterior adjustment on ($s=1$)\n"
            f"n={len(values_b):,} valid records"
        ),
    )
    axes[0].set_xlim(0.0, max(1.0, np.ceil(max(values_a.max(), values_b.max()) + 0.5)))
    axes[0].set_ylim(0.0, 101.0)
    axes[0].set_xlabel(
        "Absolute five-second movement (ticks)"
    )
    axes[0].set_ylabel(
        "Fills at or below this movement (%)"
    )
    axes[0].set_title("(a) Absolute post-fill movement", pad=6)
    axes[0].grid(color="#D9D9D9", linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False, fontsize=6.8)

    jitter = (((seeds * 17) % 30) - 14.5) / 14.5 * 0.025
    for index in range(len(seeds)):
        axes[1].plot(
            [jitter[index], 1.0 + jitter[index]],
            [fill_rate_a[index], fill_rate_b[index]],
            color=GREY,
            linewidth=0.6,
            alpha=0.56,
            zorder=1,
        )
    axes[1].scatter(
        jitter,
        fill_rate_a,
        s=17,
        marker="o",
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    axes[1].scatter(
        1.0 + jitter,
        fill_rate_b,
        s=17,
        marker="s",
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    axes[1].set_xlim(-0.18, 1.18)
    axes[1].set_ylim(0.0, float(max(fill_rate_a.max(), fill_rate_b.max())) * 1.12)
    axes[1].set_xticks(
        [0.0, 1.0],
        [
            "Adjustment off\n($s=0$)",
            "Adjustment on\n($s=1$)",
        ],
    )
    axes[1].set_xlabel("Quote setting")
    axes[1].set_ylabel(
        "Quote opportunities filled (%)"
    )
    axes[1].set_title("(b) Fill proportion", pad=6)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[1].text(
        0.98,
        0.96,
        f"Lower in {int(np.sum(fill_rate_b < fill_rate_a))} of {len(seeds)} simulations",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )

    fig.suptitle(
        "Post-fill movement and fill proportions",
        y=0.995,
        fontsize=9.8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), pad=0.65, w_pad=1.0)
    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "Post-fill movement and fill proportions",
            "Author": "Finn Smith",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    print(f"Valid five-second fills at s=0/s=1: {len(values_a)}/{len(values_b)}")
    print(
        "Pooled mean absolute movement at s=0/s=1 (ticks): "
        f"{values_a.mean():.8f}/{values_b.mean():.8f}"
    )
    print(
        "Equal-seed mean absolute movement at s=0/s=1 (ticks): "
        f"{equal_seed_mean_a:.8f}/{equal_seed_mean_b:.8f}"
    )
    print(f"Seeds with lower fill rate at s=1: {int(np.sum(fill_rate_b < fill_rate_a))}/30")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Output SHA-256: {_sha256(OUTPUT_PDF)}")
    return OUTPUT_PDF


if __name__ == "__main__":
    render()
