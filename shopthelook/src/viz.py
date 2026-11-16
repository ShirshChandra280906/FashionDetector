"""Shared matplotlib style so every chart in results/ looks consistent.

Colours come from a validated colour-blind-safe categorical palette
(fixed order, never cycled). Each chart's numbers are also saved as CSV
next to the PNG, so a table view always exists.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#a8a7a1"  # for baselines / de-emphasised bars


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False, "figure.dpi": 150, "axes.axisbelow": True,
    })


def hbar(ax, labels, values, color=SERIES[0], fmt="{:,.0f}", highlight=None):
    """Horizontal bar chart with value labels at the bar ends (largest on top)."""
    y = range(len(labels))[::-1]
    colors = [color if highlight is None or l in highlight else NEUTRAL for l in labels]
    ax.barh(list(y), values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    xmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, xmax * 1.15)
