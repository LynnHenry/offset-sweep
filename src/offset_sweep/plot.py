"""Offset curve figure: AUC against offset, one line per model (mean over surfaces, min-max band when several)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

# Categorical slots 1-3 of the validated default palette (light surface); text stays in neutral ink.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def curves(rows_by_file: dict[str, list[dict]]):
    """{model: {x: [auc per surface]}}, x = outward voxels when known, else layer offset; plus whether x is outward."""
    out, outward = defaultdict(lambda: defaultdict(list)), True
    for rows in rows_by_file.values():
        for r in rows:
            if r["outward_voxels"] == "":
                outward = False
    for rows in rows_by_file.values():
        for r in rows:
            x = int(r["outward_voxels"]) if outward else int(r["offset"])
            out[r["model"]][x].append(float(r["auc"]))
    return out, outward


def plot(rows_by_file: dict[str, list[dict]], path: Path, title: str | None = None, xlim: tuple[int, int] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, outward = curves(rows_by_file)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for i, (model, d) in enumerate(data.items()):
        c = SERIES[i % len(SERIES)] if i < len(SERIES) else INK2
        xs = sorted(x for x in d if xlim is None or xlim[0] <= x <= xlim[1])
        mean = np.array([np.mean(d[x]) for x in xs])
        n = max(len(d[x]) for x in xs)
        if n > 1:
            ax.fill_between(xs, [min(d[x]) for x in xs], [max(d[x]) for x in xs], color=c, alpha=0.12, lw=0)
        ax.plot(xs, mean, color=c, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1,
                label=model + (f" (mean of {n}, band = min–max)" if n > 1 else ""))
        ax.annotate(model, (xs[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    if outward:
        ax.set_xlabel("window offset along the surface normal (voxels)\n← inside (toward scroll centre)"
                      "          surface          outside →", color=INK2, fontsize=9)
    else:
        ax.set_xlabel("window offset (layers; inside/outside unknown, see --outward)", color=INK2, fontsize=9)
    ax.set_ylabel("ink AUC", color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
