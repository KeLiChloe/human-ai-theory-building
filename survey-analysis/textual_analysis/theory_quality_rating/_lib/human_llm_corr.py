#!/usr/bin/env python3
"""Human vs LLM overall-quality correlation helpers for the combined quality-gap figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from scipy import stats

LIB = Path(__file__).resolve().parent
ASSESSMENT_DIR = LIB.parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent
ROOT = TEXTUAL_DIR.parent
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from viz_style import (  # noqa: E402
    BAR_ALPHA,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    GROUP_ORDER_COLLAPSED,
    PHASE_HATCH_COLOR,
    apply_plot_style,
    display_label,
    format_p_value_label,
    set_axis_labels,
    style_axes,
)

apply_plot_style()

COMBINED_CSV = ASSESSMENT_DIR / "theory_ratings_human_evaluators.csv"
GENAI_CSV = ASSESSMENT_DIR / "theory_ratings_genai_evaluator.csv"
NAME_COL = "Participant ID"

DIMS = [
    "clarity_coherence",
    "causal_reasoning",
    "theoretical_depth",
    "creativity",
    "persuasiveness",
]


def _fnum(x) -> float:
    try:
        v = float(str(x).strip())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _load_matched(model_tag: str) -> pd.DataFrame:
    gen = pd.read_csv(GENAI_CSV)
    llm = gen[gen["rater_identifier"].astype(str).str.strip() == model_tag].copy()
    if llm.empty:
        raise RuntimeError(f"No LLM overall scores found for model_tag={model_tag!r}")
    llm["llm"] = pd.to_numeric(llm["overall_quality"], errors="coerce")
    llm = llm.dropna(subset=["llm"])
    llm["task"] = llm["task"].astype(str).str.lower().str.strip()
    llm["effect"] = llm["effect"].astype(str).str.lower().str.strip()
    llm["phase"] = llm["phase"].astype(str).str.lower().str.strip()
    llm["name_key"] = llm["participant_ID"].astype(str).str.strip()
    llm = llm[["name_key", "task", "effect", "phase", "llm"]]

    hum = pd.read_csv(COMBINED_CSV)
    hum["cohort"] = hum["rater_identifier"].astype(str).str.strip()
    for c in DIMS:
        hum[c] = pd.to_numeric(hum[c], errors="coerce")
    hum["human"] = hum[DIMS].mean(axis=1)
    hum["task"] = hum["task"].astype(str).str.lower().str.strip()
    hum["effect"] = hum["effect"].astype(str).str.lower().str.strip()
    hum["phase"] = hum["phase"].astype(str).str.lower().str.strip()
    hum["name_key"] = hum["participant_ID"].astype(str).str.strip()
    hum["group"] = pd.to_numeric(hum["group"], errors="coerce").map(
        {0: "PhD Students", 1: "Senior Scientists", 2: "GenAI"}
    )

    m = hum.merge(llm, on=["name_key", "task", "effect", "phase"], how="inner")
    keys = ["name_key", "task", "effect", "phase", "cohort", "group"]
    return m.groupby(keys, as_index=False).agg(
        human=("human", "mean"),
        llm=("llm", "mean"),
        n_ratings=("human", "size"),
    )


def _corr_stats(df: pd.DataFrame) -> dict:
    x = df["human"].to_numpy(float)
    y = df["llm"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(
            n=n,
            pearson=np.nan,
            pearson_p=np.nan,
            spearman=np.nan,
            spearman_p=np.nan,
            slope=np.nan,
            intercept=np.nan,
        )
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    return dict(
        n=n,
        pearson=pr,
        pearson_p=pp,
        spearman=sr,
        spearman_p=sp,
        slope=slope,
        intercept=intercept,
    )


def _fmt_r(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "NA"


def _fmt_corr(r: float, p: float) -> str:
    return f"{_fmt_r(r)} ({format_p_value_label(p)})"


def _fmt_ols_equation(intercept: float, slope: float) -> str:
    """OLS fit of LLM score (ŷ) on human score (x)."""
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        return "ŷ = NA"
    a = f"{intercept:.2f}"
    b = abs(slope)
    if slope >= 0:
        return f"ŷ = {a} + {b:.2f} x"
    return f"ŷ = {a} − {b:.2f} x"


def _corr_stats_box(
    s: dict,
    *,
    fontsize: float,
) -> AnchoredOffsetbox:
    """Bold labels + left-aligned value column (Pearson / Spearman / OLS)."""
    props = {
        "size": fontsize,
        "color": "#333333",
        "family": "Helvetica",
    }
    bold = {**props, "weight": "bold"}
    labels = VPacker(
        children=[
            TextArea("Pearson", textprops=bold),
            TextArea("Spearman", textprops=bold),
            TextArea("OLS", textprops=bold),
        ],
        align="left",
        pad=0,
        sep=3,
    )
    values = VPacker(
        children=[
            TextArea(
                f"r = {_fmt_corr(s['pearson'], s['pearson_p'])}",
                textprops=props,
            ),
            TextArea(
                f"ρ = {_fmt_corr(s['spearman'], s['spearman_p'])}",
                textprops=props,
            ),
            TextArea(
                _fmt_ols_equation(s["intercept"], s["slope"]),
                textprops=props,
            ),
        ],
        align="left",
        pad=0,
        sep=3,
    )
    body = HPacker(children=[labels, values], align="baseline", pad=0, sep=10)
    box = AnchoredOffsetbox(
        loc="lower right",
        child=body,
        pad=0.4,
        borderpad=1.15,
        frameon=True,
    )
    box.patch.set_boxstyle("square,pad=0.35")
    box.patch.set_facecolor("white")
    box.patch.set_edgecolor(PHASE_HATCH_COLOR)
    box.patch.set_linewidth(0.8)
    return box


LEGEND_EDGE = PHASE_HATCH_COLOR


def _merge_cohort_means(m: pd.DataFrame) -> pd.DataFrame:
    """One point per theory: average human (and llm) across human evaluators."""
    return m.groupby(
        ["name_key", "task", "effect", "phase", "group"],
        as_index=False,
    ).agg(
        human=("human", "mean"),
        llm=("llm", "mean"),
        n_cohorts=("cohort", "nunique"),
        n_ratings=("n_ratings", "sum"),
    )


SCATTER_COLORS = {
    **GROUP_COLORS_COLLAPSED,
    "GenAI": GROUP_COLORS_TEXT["GenAI"],
}


def _scatter_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "Human":
        return df["group"].isin(["PhD Students", "Senior Scientists"])
    return df["group"] == group


def _draw_scatter_panel(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    llm_display: str | None = None,
    show_ylabel: bool = True,
    marker_size: float = 32,
    corr_fontsize: float = 10,
    axis_label_fontsize: float = 12,
    square_aspect: bool = True,
) -> None:
    for g in GROUP_ORDER_COLLAPSED:
        sub = df.loc[_scatter_mask(df, g)]
        ax.scatter(
            sub["human"],
            sub["llm"],
            s=marker_size,
            alpha=0.85,
            color=SCATTER_COLORS[g],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    s = _corr_stats(df)
    xs = np.linspace(1.0, 10.0, 50)
    ax.plot([1, 10], [1, 10], ls="--", color="#BDBDBD", lw=1.0, zorder=1)
    if np.isfinite(s["slope"]):
        ax.plot(
            xs,
            s["intercept"] + s["slope"] * xs,
            color="#333333",
            lw=1.25,
            zorder=2,
        )

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xticks(list(range(2, 11, 2)))
    ax.set_yticks(list(range(2, 11, 2)))
    if square_aspect:
        ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=14, pad=8, fontweight="bold")

    ax.add_artist(_corr_stats_box(s, fontsize=corr_fontsize))
    set_axis_labels(ax, "Score by human evaluators", None, xlabel_pad=8)
    if show_ylabel:
        ax.set_ylabel(
            f"Score by LLM ({llm_display})" if llm_display else "Score by LLM",
            fontsize=axis_label_fontsize,
        )
    style_axes(ax)
    ax.tick_params(axis="y", labelleft=True)


def _legend_handles(contributor_label=display_label) -> list:
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCATTER_COLORS[g],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=8.5,
            alpha=BAR_ALPHA,
            label=contributor_label(g),
        )
        for g in GROUP_ORDER_COLLAPSED
    ]
    line_handles = [
        Line2D([0], [0], color="#333333", lw=1.25, label="OLS fit"),
        Line2D(
            [0],
            [0],
            color="#BDBDBD",
            lw=1.0,
            ls="--",
            label="Identity (y = x)",
        ),
    ]
    return group_handles + line_handles
