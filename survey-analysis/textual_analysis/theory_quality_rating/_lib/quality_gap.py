#!/usr/bin/env python3
"""Combined figure: correlation + bar comparison + score distributions (3 rows).

Panel A (row 1): Human vs LLM scatter (corr), full width.
Panel B (row 2): Pre/Post-ML bar charts (PhD / Senior / Topic / GenAI; LLM | Human).
Panel C (row 3): Human vs GenAI score distributions (LLM | Human).

Human side: evaluator-1∩evaluator-2 texts averaged on overlap.
Annotated n = both evaluators summed.
Writes to theory_quality_rating/outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LIB = Path(__file__).resolve().parent
ASSESSMENT_DIR = LIB.parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent
ROOT = TEXTUAL_DIR.parent
OUT_DIR = ASSESSMENT_DIR / "outputs"
SURVEY_CSV = ROOT / "anonymous_survey_data.csv"
GENAI_CSV = ASSESSMENT_DIR / "theory_ratings_genai_evaluator.csv"

for p in (LIB, ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import human_llm_corr as corr_plot  # noqa: E402
import human_ratings as he  # noqa: E402
import llm_quality as llm  # noqa: E402
from common.viz_config import GROUP_COLORS  # noqa: E402
from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    PHASE_HATCH_COLOR,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    apply_plot_style,
    display_label,
)

apply_plot_style()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.sf": "Helvetica",
    }
)

# Panel B bars / significance (Topic Experts overlaps PhD/Senior).
BAR_GROUP_ORDER = ["PhD Students", "Senior Scientists", "Topic Experts", "GenAI"]
BAR_PAIRWISE = (
    ("PhD Students", "GenAI"),
    ("Senior Scientists", "GenAI"),
    ("Topic Experts", "GenAI"),
    ("Senior Scientists", "PhD Students"),
    ("Topic Experts", "Non Topic Experts"),
)
# Panel C distributions: collapsed Human vs GenAI only.
DIST_GROUPS = ["Human", "GenAI"]
DIST_COLORS = {**GROUP_COLORS_COLLAPSED, "GenAI": GROUP_COLORS_TEXT["GenAI"]}


def _theories_by_label(group: str) -> str:
    return f"Theories by {display_label(group)}"


CONTRIBUTOR_LABEL = _theories_by_label
NOTE_FONTSIZE = 10.0
MODEL = "gpt-5.5"
PANEL_LR_NOTE = "(left: LLM evaluator; right: human evaluators)"
SECTION_LABEL_GAP = 0.024
PANEL_TITLE_FONTSIZE = 19
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


def _group_label(gid: float) -> str | None:
    return {0.0: "PhD Students", 1.0: "Senior Scientists", 2.0: "GenAI"}.get(gid)


def _attach_topic_flag(df: pd.DataFrame, name_col: str = "participant_ID") -> pd.DataFrame:
    out = df.copy()
    keys = out[name_col].astype(str).str.strip()
    survey = pd.read_csv(SURVEY_CSV, dtype=str, keep_default_na=False)
    exact = (
        survey[[NAME_COL, he.TOPIC_EXPERT_COLUMN]]
        .assign(_id=lambda d: d[NAME_COL].astype(str).str.strip())
        .drop_duplicates(subset=["_id"], keep="first")
        .set_index("_id")[he.TOPIC_EXPERT_COLUMN]
        .astype(str)
        .str.strip()
    )
    out[he.TOPIC_EXPERT_COLUMN] = keys.map(exact)
    return out


def _load_llm_scores() -> pd.DataFrame:
    gen = pd.read_csv(GENAI_CSV)
    gen = gen[gen["rater_identifier"].astype(str).str.strip() == MODEL].copy()
    rows: list[dict] = []
    for _, r in gen.iterrows():
        group = _group_label(_fnum(r.get("group", np.nan)))
        score = _fnum(r.get("overall_quality", np.nan))
        if group is None or not np.isfinite(score):
            continue
        name = str(r["participant_ID"]).strip()
        topic = str(r.get(he.TOPIC_EXPERT_COLUMN, "")).strip()
        rows.append(
            dict(
                participant_ID=name,
                name_key=name,
                group=group,
                topic_expert=topic,
                llm_quality=score,
            )
        )
    return pd.DataFrame(rows)


def _load_human_by_cohort() -> pd.DataFrame:
    hum = pd.read_csv(he.COMBINED_CSV)
    for c in DIMS:
        hum[c] = pd.to_numeric(hum[c], errors="coerce")
    hum["human_quality"] = hum[DIMS].mean(axis=1)
    hum["cohort"] = hum["rater_identifier"].astype(str).str.strip()
    hum["group"] = pd.to_numeric(hum["group"], errors="coerce").map(
        {0: "PhD Students", 1: "Senior Scientists", 2: "GenAI"}
    )
    hum = _attach_topic_flag(hum)
    return hum.groupby(
        [
            "participant_ID",
            "task",
            "effect",
            "phase",
            "cohort",
            "group",
            he.TOPIC_EXPERT_COLUMN,
        ],
        as_index=False,
    ).agg(human_quality=("human_quality", "mean"))


def _merge_human_cohorts(by_cohort: pd.DataFrame) -> pd.DataFrame:
    return by_cohort.groupby(
        ["participant_ID", "task", "effect", "phase", "group", he.TOPIC_EXPERT_COLUMN],
        as_index=False,
    ).agg(human_quality=("human_quality", "mean"))


def _scores_by_group(
    df: pd.DataFrame,
    score_col: str,
    groups: list[str] | tuple[str, ...],
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for g in groups:
        if g == "Human":
            mask = df["group"].isin(["PhD Students", "Senior Scientists"])
        elif g == "Topic Experts":
            mask = (df["group"].isin(["PhD Students", "Senior Scientists"])) & (
                df[he.TOPIC_EXPERT_COLUMN].astype(str).str.strip() == "1"
            )
        else:
            mask = df["group"] == g
        out[g] = df.loc[mask, score_col].astype(float)
    return out


def _side_score_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (llm_cells, human_by_cohort, human_merged)."""
    cells = _load_llm_scores().dropna(subset=["llm_quality", "group"]).copy()
    by_cohort = _load_human_by_cohort().dropna(subset=["human_quality", "group"]).copy()
    merged = _merge_human_cohorts(by_cohort)
    return cells, by_cohort, merged


def _n_by_group(
    llm_cells: pd.DataFrame,
    human_by_cohort: pd.DataFrame,
    groups: list[str] | tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    llm_n = {
        g: int(_scores_by_group(llm_cells, "llm_quality", groups)[g].notna().sum())
        for g in groups
    }
    hum_n = {
        g: int(_scores_by_group(human_by_cohort, "human_quality", groups)[g].notna().sum())
        for g in groups
    }
    return llm_n, hum_n


def _dist_panel_data() -> list[tuple[str, dict[str, pd.Series], dict[str, int]]]:
    cells, by_cohort, merged = _side_score_frames()
    llm_scores = _scores_by_group(cells, "llm_quality", DIST_GROUPS)
    hum_scores = _scores_by_group(merged, "human_quality", DIST_GROUPS)
    llm_n, hum_n = _n_by_group(cells, by_cohort, DIST_GROUPS)
    return [
        ("llm", llm_scores, llm_n),
        ("human", hum_scores, hum_n),
    ]


def _bar_legend_ns() -> tuple[dict[str, int], dict[str, int]]:
    cells, by_cohort, _merged = _side_score_frames()
    return _n_by_group(cells, by_cohort, BAR_GROUP_ORDER)


def _load_human_merged_for_bars():
    ev1 = he.load_combined_ratings(next(c for c in he.COHORTS if c.key == he.EVALUATOR_1))
    ev2 = he.load_combined_ratings(next(c for c in he.COHORTS if c.key == he.EVALUATOR_2))
    return he.merge_evaluator_ratings(ev1, ev2)


def _load_llm_pooled_for_bars():
    return llm.load_pooled_from_long_csv(GENAI_CSV, rater=MODEL)


def _draw_dist(ax, scores: dict[str, pd.Series]) -> None:
    bins = np.linspace(1, 10, 19)
    for group in DIST_GROUPS:
        vals = scores[group].dropna().to_numpy()
        if len(vals) == 0:
            continue
        color = DIST_COLORS[group]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            color=color,
            alpha=0.50,
            edgecolor="white",
            linewidth=0.6,
        )
        if len(vals) >= 8:
            try:
                from scipy.stats import gaussian_kde

                xs = np.linspace(1, 10, 200)
                ax.plot(xs, gaussian_kde(vals)(xs), color=color, lw=2.0)
            except Exception:
                pass
        ax.axvline(float(np.median(vals)), color=color, ls="--", lw=1.4, alpha=0.9)


def _legend_dist(n_by_group: dict[str, int]):
    return [
        Patch(
            facecolor=DIST_COLORS[g],
            edgecolor="white",
            alpha=BAR_ALPHA,
            label=f"{CONTRIBUTOR_LABEL(g)} (n={n_by_group[g]})",
        )
        for g in DIST_GROUPS
    ] + [Line2D([0], [0], color="#555555", ls="--", lw=1.6, label="Median")]


def _legend_bars(n_by_group: dict[str, int]):
    group_handles = [
        Patch(
            facecolor=GROUP_COLORS_TEXT[g],
            edgecolor="white",
            alpha=BAR_ALPHA,
            label=f"{CONTRIBUTOR_LABEL(g)} (n={n_by_group[g]})",
        )
        for g in BAR_GROUP_ORDER
    ]
    phase_handles = [
        Patch(
            facecolor=PHASE_HATCH_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            alpha=BAR_ALPHA,
            label="Pre-ML",
        ),
        Patch(
            facecolor=PHASE_HATCH_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            alpha=BAR_ALPHA,
            hatch="///",
            label="Post-ML",
        ),
    ]
    return group_handles + phase_handles


def _style_legend_frame(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_linewidth(0.8)
    frame.set_edgecolor(PHASE_HATCH_COLOR)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def _legend_kw(**extra):
    return dict(
        frameon=True,
        fancybox=False,
        edgecolor=PHASE_HATCH_COLOR,
        facecolor="white",
        framealpha=1.0,
        **extra,
    )


def _place_corr_panel(ax_corr, axes_row, fig) -> tuple[float, float, float, float]:
    """Largest square that fits row A, horizontally aligned with the row below."""
    slot = ax_corr.get_position()
    pos_l = axes_row[0].get_position()
    pos_r = axes_row[1].get_position()
    w_span = pos_r.x1 - pos_l.x0
    fig_w, fig_h = fig.get_size_inches()
    s_w = min(w_span, slot.height * fig_h / fig_w)
    s_h = s_w * fig_w / fig_h
    x0 = pos_l.x0 + (w_span - s_w) / 2
    y0 = slot.y0 + (slot.height - s_h) / 2
    ax_corr.set_position([x0, y0, s_w, s_h])
    return x0, y0, s_w, s_h


def main() -> None:
    dist_panels = _dist_panel_data()
    human_vals = he.pooled_values(_load_human_merged_for_bars())
    llm_vals, _llm_paired = _load_llm_pooled_for_bars()
    corr_df = corr_plot._merge_cohort_means(corr_plot._load_matched(MODEL))

    fig = plt.figure(figsize=(14.5, 18.6))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.096, 1.14, 0.95],
        hspace=0.32,
        wspace=0.38,
    )
    ax_corr = fig.add_subplot(gs[0, :])
    axes_bars = np.array([fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])])
    axes_dist = np.array([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])])

    legend_kw_bars = _legend_kw(fontsize=9.0, borderpad=0.35, handletextpad=0.3)
    legend_kw_dist = _legend_kw(fontsize=12.0, borderpad=0.45, handletextpad=0.4)

    corr_plot._draw_scatter_panel(
        ax_corr,
        corr_df,
        title="",
        llm_display=None,
        show_ylabel=True,
        marker_size=26,
        corr_fontsize=16.0,
        axis_label_fontsize=12,
        square_aspect=True,
    )
    corr_n = {
        g: int(corr_plot._scatter_mask(corr_df, g).sum())
        for g in corr_plot.GROUP_ORDER_COLLAPSED
    }
    corr_handles = corr_plot._legend_handles(
        contributor_label=CONTRIBUTOR_LABEL,
        n_by_group=corr_n,
    )
    corr_legend = ax_corr.legend(
        corr_handles,
        [h.get_label() for h in corr_handles],
        loc="upper left",
        fontsize=12.0,
        markerscale=1.25,
        borderpad=0.45,
        handletextpad=0.45,
        labelspacing=0.45,
        **_legend_kw(),
    )
    _style_legend_frame(corr_legend)

    llm_bar_n, hum_bar_n = _bar_legend_ns()

    llm._draw_panel(
        axes_bars[0],
        llm_vals,
        "",
        group_order=BAR_GROUP_ORDER,
        group_colors=GROUP_COLORS_TEXT,
        pairwise=BAR_PAIRWISE,
        note_fontsize=NOTE_FONTSIZE,
        notes_layout="sig_color_pvals",
        sig_text_color="black",
        sig_legend_fontsize=9.0,
    )
    he._draw_panel(
        axes_bars[1],
        human_vals,
        "",
        group_order=BAR_GROUP_ORDER,
        group_colors=GROUP_COLORS_TEXT,
        pairwise=BAR_PAIRWISE,
        note_fontsize=NOTE_FONTSIZE,
        notes_layout="sig_color_pvals",
        sig_text_color="black",
        sig_legend_fontsize=9.0,
    )
    axes_bars[0].set_ylabel("Quality score (mean ± 95% CI)", fontsize=12)
    axes_bars[1].set_ylabel("")
    # Keep legends inside each panel; tiny right + up nudge.
    axes_bars[0].legend(
        handles=_legend_bars(llm_bar_n),
        loc="upper left",
        bbox_to_anchor=(0.03, 1.01),
        borderaxespad=0.0,
        **legend_kw_bars,
    )
    axes_bars[1].legend(
        handles=_legend_bars(hum_bar_n),
        loc="upper left",
        bbox_to_anchor=(0.03, 1.01),
        borderaxespad=0.0,
        **legend_kw_bars,
    )
    for ax in axes_bars:
        _style_legend_frame(ax.get_legend())

    for ax, (_key, scores, n_by_group) in zip(axes_dist, dist_panels):
        _draw_dist(ax, scores)
        ax.set_xlim(1, 10)
        ax.set_xticks([1, 3, 5, 7, 9, 10])
        ax.set_xlabel("Quality score (1–10)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=_legend_dist(n_by_group), loc="upper left", **legend_kw_dist)
    for ax in axes_dist:
        _style_legend_frame(ax.get_legend())
    axes_dist[0].set_ylabel("Density")
    ymax = max(axes_dist[0].get_ylim()[1], axes_dist[1].get_ylim()[1])
    for ax in axes_dist:
        ax.set_ylim(0, ymax)

    for ax in [ax_corr, *axes_bars, *axes_dist]:
        ax.xaxis.get_label().set_fontweight("normal")
        ax.yaxis.get_label().set_fontweight("normal")
        # +2 relative to pre-bump sizes used above / set_axis_labels default.
        if ax.get_xlabel():
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() + 2)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() + 2)
        for label in ax.get_xticklabels():
            label.set_fontweight("normal")
        for label in ax.get_yticklabels():
            label.set_fontweight("normal")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.035)
    fig.canvas.draw()
    corr_x0, corr_y0, _, corr_h = _place_corr_panel(ax_corr, axes_bars, fig)
    fig.canvas.draw()

    pos_b = axes_bars[0].get_position()
    pos_c = axes_dist[0].get_position()
    label_x = pos_b.x0 - 0.03
    fig.text(
        label_x,
        corr_y0 + corr_h + SECTION_LABEL_GAP,
        "a. Do human and LLM evaluators agree on theory quality?",
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        label_x,
        pos_b.y1 + SECTION_LABEL_GAP,
        "b. Independent evaluations of theory quality " + PANEL_LR_NOTE,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        label_x,
        pos_c.y1 + SECTION_LABEL_GAP,
        "c. Score distributions of theory quality " + PANEL_LR_NOTE,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "theory_quality_ratings"
    for ext in ("png", "svg"):
        fig.savefig(
            f"{stem}.{ext}",
            dpi=SAVE_DPI,
            pad_inches=0.04,
        )
    plt.close(fig)
    print(f"Wrote {stem}.{{png,svg}}")


if __name__ == "__main__":
    main()
