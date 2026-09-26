"""
Assess theory quality scores by group (Senior Scientists, PhD Students, GenAI).

Reads overall quality scores (mean of 5 dimensions) and plots Pre/Post bars:

1. 2×2 by task × effect (Race/Gender × Main/Interactions)
   - PhD / Senior / Topic Experts / GenAI
   - Human / Topic Experts / GenAI
2. Same group splits with all four panels pooled
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
ASSESSMENT_DIR = LIB.parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent
ROOT = TEXTUAL_DIR.parent
for p in (ASSESSMENT_DIR, TEXTUAL_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from common.stats_utils import (  # noqa: E402
    bootstrap_mean_ci,
    p_value_paired_ttest_pairs,
    p_value_welch_ttest,
)
from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_COLORS_TEXT,
    HUMAN_COMPOSITION_NOTE,
    PHASE_HATCH_COLOR,
    add_legend,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_pre_post_bracket,
    draw_pairwise_group_brackets,
    draw_pairwise_sig_legend,
    draw_pairwise_sig_color_legend,
    draw_pre_post_sig_columns,
    format_comparison_line,
    format_p_value_with_trailing_stars,
    save_figure,
    set_axis_labels,
    significance_label,
    style_axes,
    SAVE_DPI,
    SAVE_PAD_INCHES,
)

ASSESSMENT_SCORE_YMAX = 10
ASSESSMENT_PLOT_YMAX = 12.2

# Long-format LLM ratings (moved out of anonymous_survey_data.csv).
CSV_PATH = ASSESSMENT_DIR / "theory_ratings_genai_evaluator.csv"
DEFAULT_RATER = "gpt-5.5"
# Kept for callers that still build wide survey-style phase maps.
PANEL_SPECS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "Racial inequality — Main effects",
        {
            "Pre-ML": "Q Race.4 Overall Quality Score",
            "Post-ML": "Q Race.12 Overall Quality Score",
        },
    ),
    (
        "Racial inequality — Interactions",
        {
            "Pre-ML": "Q Race.10 Overall Quality Score",
            "Post-ML": "Q Race.15 Overall Quality Score",
        },
    ),
    (
        "Gender inequality — Main effects",
        {
            "Pre-ML": "Q Gender.4 Overall Quality Score",
            "Post-ML": "Q Gender.12 Overall Quality Score",
        },
    ),
    (
        "Gender inequality — Interactions",
        {
            "Pre-ML": "Q Gender.10 Overall Quality Score",
            "Post-ML": "Q Gender.15 Overall Quality Score",
        },
    ),
)

POOLED_TASK_LABEL = "Race & Gender × Main & Interactions (pooled)"

GROUP_ORDER = ["PhD Students", "Senior Scientists", "Topic Experts", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "Topic Experts", "GenAI"]
NON_TOPIC_GROUP = "Non Topic Experts"
VALUE_GROUPS = (*GROUP_ORDER, NON_TOPIC_GROUP)

GROUP_MAP = {
    "0": "PhD Students",
    "1": "Senior Scientists",
    "2": "GenAI",
}

PAIRWISE_THREE = (
    ("PhD Students", "GenAI"),
    ("Senior Scientists", "GenAI"),
    ("Senior Scientists", "PhD Students"),
    ("Topic Experts", NON_TOPIC_GROUP),
)
PAIRWISE_COLLAPSED = (
    ("Human", "GenAI"),
    ("Topic Experts", NON_TOPIC_GROUP),
)

PHASE_ORDER = ("Pre-ML", "Post-ML")

apply_plot_style()


def to_float(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_col_idx(headers: list[str], prefix: str) -> int:
    matches = [i for i, h in enumerate(headers) if h.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one column for prefix '{prefix}', got {matches}")
    return matches[0]


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    ci_low, ci_high = bootstrap_mean_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def merge_phase_values(
    parts: list[dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, list[float]]]:
    merged = {phase: {g: [] for g in VALUE_GROUPS} for phase in PHASE_ORDER}
    for part in parts:
        for phase, by_group in part.items():
            for group, vals in by_group.items():
                merged[phase][group].extend(vals)
    return merged


def merge_paired_by_group(
    parts: list[dict[str, list[tuple[float, float]]]],
) -> dict[str, list[tuple[float, float]]]:
    merged = {g: [] for g in VALUE_GROUPS}
    for part in parts:
        for group, pairs in part.items():
            merged[group].extend(pairs)
    return merged


def load_phase_and_paired_values(
    headers: list[str],
    data: list[list[str]],
    group_col: int,
    phase_map: dict[str, str],
    topic_expert_col: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[tuple[float, float]]]]:
    pre_col = find_col_idx(headers, phase_map["Pre-ML"])
    post_col = find_col_idx(headers, phase_map["Post-ML"])
    phase_to_grouped_values = {
        "Pre-ML": {g: [] for g in VALUE_GROUPS},
        "Post-ML": {g: [] for g in VALUE_GROUPS},
    }
    paired_by_group: dict[str, list[tuple[float, float]]] = {
        g: [] for g in VALUE_GROUPS
    }

    for row in data:
        gid = row[group_col].strip() if len(row) > group_col else ""
        gname = GROUP_MAP.get(gid)
        if gname is None:
            continue
        pre = to_float(row[pre_col]) if len(row) > pre_col else None
        post = to_float(row[post_col]) if len(row) > post_col else None

        targets = [gname]
        if gname in ("PhD Students", "Senior Scientists") and len(row) > topic_expert_col:
            flag = row[topic_expert_col].strip()
            if flag == "1":
                targets.append("Topic Experts")
            elif flag == "0":
                targets.append(NON_TOPIC_GROUP)

        for target in targets:
            if pre is not None:
                phase_to_grouped_values["Pre-ML"][target].append(pre)
            if post is not None:
                phase_to_grouped_values["Post-ML"][target].append(post)
            if pre is not None and post is not None:
                paired_by_group[target].append((pre, post))

    return phase_to_grouped_values, paired_by_group


def load_pooled_from_long_csv(
    path: Path = CSV_PATH,
    *,
    rater: str = DEFAULT_RATER,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[tuple[float, float]]]]:
    """Pool Pre/Post overall scores from long-format GenAI ratings CSV."""
    df = pd.read_csv(path)
    df = df[df["rater_identifier"].astype(str).str.strip() == rater].copy()
    if df.empty:
        raise RuntimeError(f"No rows for rater={rater!r} in {path}")

    phase_to_grouped_values = {
        "Pre-ML": {g: [] for g in VALUE_GROUPS},
        "Post-ML": {g: [] for g in VALUE_GROUPS},
    }
    paired_by_group: dict[str, list[tuple[float, float]]] = {
        g: [] for g in VALUE_GROUPS
    }

    def _targets(gid: object, topic: object) -> list[str]:
        gname = GROUP_MAP.get(str(gid).strip())
        if gname is None:
            return []
        out = [gname]
        if gname in ("PhD Students", "Senior Scientists"):
            flag = str(topic).strip()
            if flag == "1":
                out.append("Topic Experts")
            elif flag == "0":
                out.append(NON_TOPIC_GROUP)
        return out

    for _, row in df.iterrows():
        score = to_float(row.get("overall_quality"))
        if score is None:
            continue
        phase = str(row.get("phase", "")).strip().lower()
        phase_label = "Pre-ML" if phase == "pre" else "Post-ML" if phase == "post" else ""
        if not phase_label:
            continue
        for target in _targets(row.get("group"), row.get("topic_expert")):
            phase_to_grouped_values[phase_label][target].append(score)

    keys = ["participant_ID", "task", "effect", "group", "topic_expert"]
    for _, sub in df.groupby(keys, sort=False):
        by_phase = {
            str(p).strip().lower(): to_float(v)
            for p, v in zip(sub["phase"], sub["overall_quality"])
        }
        pre, post = by_phase.get("pre"), by_phase.get("post")
        if pre is None or post is None:
            continue
        row0 = sub.iloc[0]
        for target in _targets(row0.get("group"), row0.get("topic_expert")):
            paired_by_group[target].append((pre, post))

    return phase_to_grouped_values, paired_by_group


def _draw_panel(
    ax,
    values: dict[str, dict[str, list[float]]],
    title: str,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    note_fontsize: float,
    notes_layout: str = "stacked",
    sig_text_color: str | None = None,
    sig_legend_fontsize: float | None = None,
) -> None:
    x = np.arange(len(group_order))
    width = 0.34
    offsets = {"Pre-ML": -width / 2, "Post-ML": width / 2}
    bar_tops: dict[str, float] = {}
    phase_bar_tops: dict[str, float] = {phase: 0.0 for phase in PHASE_ORDER}

    for gi, group in enumerate(group_order):
        means, yerr_lo, yerr_hi = [], [], []
        for phase in PHASE_ORDER:
            stats = summarize(values[phase][group])
            mean = float(stats["mean"])
            lo = float(stats["ci_low"])
            hi = float(stats["ci_high"])
            means.append(mean)
            yerr_lo.append(
                max(0.0, mean - lo) if np.isfinite(mean) and np.isfinite(lo) else 0.0
            )
            yerr_hi.append(
                max(0.0, hi - mean) if np.isfinite(mean) and np.isfinite(hi) else 0.0
            )
        finite = [
            means[i] + yerr_hi[i]
            for i in range(len(PHASE_ORDER))
            if np.isfinite(means[i])
        ]
        bar_tops[group] = max(finite) if finite else 0.0

        for i, phase in enumerate(PHASE_ORDER):
            xpos = float(x[gi] + offsets[phase])
            bar = ax.bar(
                [xpos],
                [means[i]],
                width=width,
                color=group_colors[group],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            if phase == "Post-ML":
                bar[0].set_hatch("///")
            ax.errorbar(
                [xpos],
                [means[i]],
                yerr=[[yerr_lo[i]], [yerr_hi[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )
            phase_bar_tops[phase] = max(
                phase_bar_tops[phase],
                means[i] + yerr_hi[i] if np.isfinite(means[i]) else 0.0,
            )

        p_pre_post = p_value_welch_ttest(
            np.asarray(values["Pre-ML"][group], dtype=float),
            np.asarray(values["Post-ML"][group], dtype=float),
        )
        if notes_layout == "sig_color_pvals":
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["Pre-ML"]),
                float(x[gi] + offsets["Post-ML"]),
                bar_tops[group],
                p_pre_post,
                label=format_p_value_with_trailing_stars(p_pre_post),
                fontsize=note_fontsize + 2,
                color=sig_text_color if sig_text_color is not None else "black",
                fontweight="normal",
            )
        else:
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["Pre-ML"]),
                float(x[gi] + offsets["Post-ML"]),
                bar_tops[group],
                p_pre_post,
                color=sig_text_color,
            )

    phase_comp_sigs: dict[str, list[str]] = {phase: [] for phase in PHASE_ORDER}
    phase_comp_pvals: dict[str, list[float]] = {phase: [] for phase in PHASE_ORDER}
    pair_labels: list[str] = []
    for left, right in pairwise:
        pair_labels.append(comparison_pair_label(left, right))
        for phase in PHASE_ORDER:
            p_val = p_value_welch_ttest(
                np.asarray(values[phase][left], dtype=float),
                np.asarray(values[phase][right], dtype=float),
            )
            phase_comp_sigs[phase].append(significance_label(p_val))
            phase_comp_pvals[phase].append(p_val)

    ax.set_xticks(x)
    ax.set_xticklabels([display_label(g) for g in group_order], fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    else:
        ax.set_title("")
    set_axis_labels(ax, None, None, bold_xticks=True)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.tick_params(axis="x", labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")
    ax.set_ylim(0, ASSESSMENT_PLOT_YMAX)
    ax.set_yticks(list(range(0, ASSESSMENT_SCORE_YMAX + 1, 2)))

    if notes_layout == "pairwise_brackets":
        draw_pairwise_group_brackets(
            ax,
            x,
            group_order,
            pairwise,
            PHASE_ORDER,
            offsets,
            phase_comp_sigs,
            phase_bar_tops,
            fontsize=note_fontsize,
        )
    elif notes_layout == "sig_legend":
        draw_pairwise_sig_legend(
            ax,
            [
                ("Pre-ML", list(zip(phase_comp_sigs["Pre-ML"], pair_labels))),
                ("Post-ML", list(zip(phase_comp_sigs["Post-ML"], pair_labels))),
            ],
            loc="upper right",
            fontsize=note_fontsize,
        )
    elif notes_layout in ("sig_color_legend", "sig_color_pvals"):
        use_p = notes_layout == "sig_color_pvals"
        draw_pairwise_sig_color_legend(
            ax,
            [
                (
                    "Pre-ML",
                    list(
                        zip(
                            phase_comp_pvals["Pre-ML"]
                            if use_p
                            else phase_comp_sigs["Pre-ML"],
                            pairwise,
                        )
                    ),
                ),
                (
                    "Post-ML",
                    list(
                        zip(
                            phase_comp_pvals["Post-ML"]
                            if use_p
                            else phase_comp_sigs["Post-ML"],
                            pairwise,
                        )
                    ),
                ),
            ],
            group_colors=group_colors,
            loc="upper right",
            fontsize=note_fontsize if sig_legend_fontsize is None else sig_legend_fontsize,
            label_pvalues=use_p,
            sig_text_color=sig_text_color,
        )
    elif notes_layout == "pre_post_columns":
        # Bold Pre/Post headers; only significant comparison rows in red.
        draw_pre_post_sig_columns(
            ax,
            [
                ("Pre-ML", list(zip(phase_comp_sigs["Pre-ML"], pair_labels))),
                ("Post-ML", list(zip(phase_comp_sigs["Post-ML"], pair_labels))),
            ],
            y0=-0.14,
            fontsize=note_fontsize,
        )
    else:
        phase_notes = []
        for phase in PHASE_ORDER:
            comps = [
                f"{lab}: {sig}"
                for lab, sig in zip(pair_labels, phase_comp_sigs[phase])
            ]
            phase_notes.append(f"{phase}: " + "; ".join(comps))
        ax.text(
            0.5,
            -0.22,
            "\n".join(phase_notes),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=note_fontsize,
            color="#333333",
            clip_on=False,
        )
