"""
2×2 composite of feature / SOI selection-frequency figures.

Layout:
  a. Racial Inequality — Main Effects Selection
  b. Gender Inequality — Main Effects Selection
  c. Racial Inequality — Interactions Selection
  d. Gender Inequality — Interactions Selection

Bar labels: n selected (sign-alignment % vs LR among those selectors).
Sign % is only defined for ML top features (have an LR ground-truth sign).

Output: forecasts/outputs/fig_feature_selection_frequency.{svg,png}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
TEXTUAL_DIR = ROOT / "textual_analysis"
for p in (ROOT, TEXTUAL_DIR, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_style import apply_plot_style  # noqa: E402

import me_importance as me  # noqa: E402
import soi_importance as soi  # noqa: E402

OUT_DIR = FORECASTS / "outputs"

PANEL_SPECS = [
    ("a", "Racial Inequality — Main Effects Selection", "me", "race"),
    ("b", "Gender Inequality — Main Effects Selection", "me", "gender"),
    ("c", "Racial Inequality — Interactions Selection", "soi", "race"),
    ("d", "Gender Inequality — Interactions Selection", "soi", "gender"),
]

# Tilt y-tick labels to shrink left margin.
Y_LABEL_ROTATION = 30

# Display remaps for raw feature keys on y-ticks (internal key unchanged).
FEATURE_TICK_LABELS = {
    "hispanic_and_other": "hispanic",
}


def _feat_tick(name: str) -> str:
    return FEATURE_TICK_LABELS.get(name, name)


def _set_rotated_yticklabels(ax, labels: list[str], *, fontsize: float) -> None:
    ax.set_yticklabels(
        labels,
        fontsize=fontsize,
        rotation=Y_LABEL_ROTATION,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )


def _set_wrapped_yticklabels(ax, labels: list[str], *, fontsize: float) -> None:
    """Multiline y-tick labels with the same tilt as single-line feature labels."""
    ax.set_yticklabels(
        labels,
        fontsize=fontsize,
        rotation=Y_LABEL_ROTATION,
        ha="right",
        va="center",
        rotation_mode="anchor",
        linespacing=1.15,
    )


def _panel_title_artist(ax):
    """Return the axes title Text that actually holds the panel title string.

    Matplotlib routes ``set_title(..., loc='left')`` to ``ax._left_title``, while
    ``ax.title`` stays empty — aligning the wrong artist leaves the visible title
    stuck at the y-spine (x=0).
    """
    for artist in (ax._left_title, ax.title, ax._right_title):
        if artist.get_text():
            return artist
    return ax.title


def _align_titles_to_ylabels(fig, axes) -> None:
    """Left-align each panel title to a shared column left edge (y-label extents).

    Within each column (a/c and b/d), use the leftmost y-tick label across both
    panels so titles share one flush left edge.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_arr = np.atleast_2d(axes)
    n_rows, n_cols = axes_arr.shape

    # Display-space left edge per column (min over rows).
    col_x0: list[float] = []
    for j in range(n_cols):
        xs: list[float] = []
        for i in range(n_rows):
            ticks = [
                t for t in axes_arr[i, j].get_yticklabels()
                if t.get_visible() and t.get_text()
            ]
            if ticks:
                xs.append(min(t.get_window_extent(renderer).x0 for t in ticks))
        col_x0.append(min(xs) if xs else 0.0)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes_arr[i, j]
            x_ax = ax.transAxes.inverted().transform((col_x0[j], 0.0))[0]
            title = _panel_title_artist(ax)
            ax._autotitlepos = False
            title.set_position((x_ax, 1.02))
            title.set_ha("left")
            title.set_va("bottom")


def _soi_tick_label(a: str, b: str) -> str:
    return f"{_feat_tick(a)} *\n{_feat_tick(b)}"


def _me_ml_signs(task: str) -> dict[str, str]:
    with open(me.ML_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {e["feature"]: e["sign"] for e in raw[task]}


def _me_q3_cols(task: str) -> dict[str, int]:
    prefix = "Q Race.3" if task == "race" else "Q Gender.3"
    return {
        re.sub(rf"^{re.escape(prefix)} \(sign\) - ", "", h): i
        for i, h in enumerate(me.headers)
        if re.match(rf"^{re.escape(prefix)} \(sign\) - ", h)
    }


def _me_sign_align(task: str) -> dict[str, dict[str, int]]:
    """Among selectors of each ML feature, count correct LR direction."""
    ml_signs = _me_ml_signs(task)
    q1 = me.r1_col if task == "race" else me.g1_col
    q3 = _me_q3_cols(task)
    out = {f: {"n_selected": 0, "n_aligned": 0} for f in ml_signs}
    for row in me.data:
        cell = row[q1].strip()
        if not cell:
            continue
        selected = {x.strip() for x in cell.split(",") if x.strip()}
        for feat, ml_sign in ml_signs.items():
            if feat not in selected:
                continue
            out[feat]["n_selected"] += 1
            human = row[q3[feat]].strip() if feat in q3 else ""
            if human == ml_sign:
                out[feat]["n_aligned"] += 1
    return out


def _soi_ml_signs(task: str) -> dict[tuple[str, str], str]:
    with open(soi.ML_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        soi.canon_pair(e["feature_1"], e["feature_2"]): e["sign"]
        for e in raw[task]
    }


def _soi_sign_align(task: str) -> dict[tuple[str, str], dict[str, int]]:
    """Among selectors of each ML interaction, count correct LR direction."""
    ml_signs = _soi_ml_signs(task)
    pair_cols = soi.r_cols if task == "race" else soi.g_cols
    sign_prefix = "Q Race.9" if task == "race" else "Q Gender.9"
    sign_cols = [
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 1st)"),
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(soi.headers) if h.strip() == f"{sign_prefix} (SOI, sign, 3rd)"),
    ]
    out = {p: {"n_selected": 0, "n_aligned": 0} for p in ml_signs}
    for row in soi.data:
        chosen: dict[tuple[str, str], str] = {}
        for pc, sc in zip(pair_cols, sign_cols):
            p = soi.parse_pair(row[pc], soi.feature_set)
            if p is None:
                continue
            chosen[p] = row[sc].strip()
        for p, ml_sign in ml_signs.items():
            if p not in chosen:
                continue
            out[p]["n_selected"] += 1
            if chosen[p] == ml_sign:
                out[p]["n_aligned"] += 1
    return out


def _format_bar_label(n_sel: int, align: dict[str, int] | None) -> str:
    if align is None or align["n_selected"] <= 0:
        return f"{n_sel}"
    pct = align["n_aligned"] / align["n_selected"] * 100
    # Explicit tag so the trailing % is not read as a selection rate.
    return f"{n_sel}  ({pct:.0f}% sign-aligned)"


def _draw_me_panel(ax, *, task: str, title: str, letter: str) -> None:
    counts = me.race_counts if task == "race" else me.gender_counts
    ml_set = me.ml_top5[task]
    align = _me_sign_align(task)
    ranked = sorted(me.FEATURES, key=lambda f: counts[f], reverse=True)
    labels = [_feat_tick(f) for f in ranked]  # display labels for raw feature keys
    vals = [counts[f] for f in ranked]
    colors = [me.COLOR_ML if f in ml_set else me.COLOR_DEFAULT for f in ranked]

    y = np.arange(len(ranked))
    bars = ax.barh(y, vals, color=colors, height=0.65, edgecolor="white", linewidth=0.6)
    for b, v, feat in zip(bars, vals, ranked):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            _format_bar_label(v, align.get(feat)),
            va="center",
            ha="left",
            fontsize=29.0,
        )

    ax.set_yticks(y)
    _set_rotated_yticklabels(ax, labels, fontsize=40.0)
    ax.tick_params(axis="y", pad=12)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of contributors selecting feature\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=26.0,
    )
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)

    ax.legend(
        handles=[
            mpatches.Patch(color=me.COLOR_ML, label="In ML top-5"),
            mpatches.Patch(color=me.COLOR_DEFAULT, label="Not in ML top-5"),
        ],
        loc="lower right",
        fontsize=23.0,
        frameon=True,
        edgecolor="0.6",
        fancybox=False,
        framealpha=1.0,
    )
    ax.set_title(
        f"{letter}.  {title}",
        fontsize=31,
        fontweight="bold",
        pad=8,
        loc="left",
    )


def _draw_soi_panel(ax, *, task: str, title: str, letter: str, top_n: int = 8) -> None:
    counts = soi.race_counts if task == "race" else soi.gender_counts
    ml_set = soi.ml_pairs[task]
    align = _soi_sign_align(task)
    ranked = sorted(soi.pairs, key=lambda p: counts[p], reverse=True)[:top_n]
    labels = [_soi_tick_label(a, b) for a, b in ranked]
    vals = [counts[p] for p in ranked]
    colors = [soi.COLOR_ML if p in ml_set else soi.COLOR_DEFAULT for p in ranked]

    y = np.arange(len(ranked))
    # Taller bars leave room for two-line y labels.
    bars = ax.barh(y, vals, color=colors, height=0.72, edgecolor="white", linewidth=0.6)
    for b, v, pair in zip(bars, vals, ranked):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            _format_bar_label(v, align.get(pair)),
            va="center",
            ha="left",
            fontsize=31.0,
        )

    ax.set_yticks(y)
    _set_wrapped_yticklabels(ax, labels, fontsize=40.0)
    ax.tick_params(axis="y", pad=18)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of contributors selecting interaction\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=26.0,
    )
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    # Extra vertical padding between interaction rows.
    ax.set_ylim(len(ranked) - 0.5, -0.5)
    ax.margins(y=0.04)

    ax.legend(
        handles=[
            mpatches.Patch(color=soi.COLOR_ML, label="In ML top-3"),
            mpatches.Patch(color=soi.COLOR_DEFAULT, label="Not in ML top-3"),
        ],
        loc="lower right",
        fontsize=23.0,
        frameon=True,
        edgecolor="0.6",
        fancybox=False,
        framealpha=1.0,
    )
    ax.set_title(
        f"{letter}.  {title}",
        fontsize=31,
        fontweight="bold",
        pad=8,
        loc="left",
    )


def plot_feature_selection_frequency(
    out_stem: Path | None = None,
) -> list[Path]:
    out_stem = out_stem or (OUT_DIR / "fig_feature_selection_frequency")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # me/soi modules set Times; re-apply Nature-style Helvetica/Arial.
    apply_plot_style()

    # Bottom row (c–d) taller for wrapped interaction labels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(30.0, 33.5),
        gridspec_kw={"height_ratios": [1.45, 1.85], "hspace": 0.34, "wspace": 2.55},
    )
    for ax, (letter, title, kind, task) in zip(axes.ravel(), PANEL_SPECS):
        if kind == "me":
            _draw_me_panel(ax, task=task, title=title, letter=letter)
        else:
            _draw_soi_panel(ax, task=task, title=title, letter=letter)

    fig.subplots_adjust(left=0.18, right=0.99, top=0.96, bottom=0.06, hspace=0.40, wspace=2.55)
    _align_titles_to_ylabels(fig, axes)

    paths: list[Path] = []
    for fmt in ("svg", "png"):
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt, dpi=400, bbox_inches="tight", pad_inches=0.08)
        paths.append(p)
        print(f"Figure saved → {p}")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    plot_feature_selection_frequency()
