"""Human rating data helpers for the combined quality-gap figure.

Reads theory_quality_rating/theory_ratings_human_evaluators.csv.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
ASSESSMENT_DIR = LIB.parent
TEXTUAL_DIR = ASSESSMENT_DIR.parent
ROOT = TEXTUAL_DIR.parent
for p in (TEXTUAL_DIR, ROOT, ASSESSMENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import bootstrap_mean_ci, p_value_welch_ttest  # noqa: E402
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
    SAVE_DPI,
    SAVE_PAD_INCHES,
    apply_plot_style,
    comparison_pair_label,
    display_label,
    draw_pre_post_bracket,
    draw_pairwise_group_brackets,
    draw_pairwise_sig_legend,
    draw_pairwise_sig_color_legend,
    draw_pre_post_sig_columns,
    format_p_value_with_trailing_stars,
    set_axis_labels,
    significance_label,
    style_axes,
)


@dataclass(frozen=True)
class CohortSpec:
    key: str  # display label, e.g. 'human evaluator 1'


EVALUATOR_1 = "human evaluator 1"
EVALUATOR_2 = "human evaluator 2"
COMBINED_CSV = ASSESSMENT_DIR / "theory_ratings_human_evaluators.csv"
PARTICIPANT_ID_COL = "participant_ID"

COHORTS: tuple[CohortSpec, ...] = (
    CohortSpec(key=EVALUATOR_1),
    CohortSpec(key=EVALUATOR_2),
)

SCORE_DIMS = [
    "clarity_coherence",
    "causal_reasoning",
    "theoretical_depth",
    "creativity",
    "persuasiveness",
]
OVERALL_COL = "overall_quality"
THEORY_KEYS = (PARTICIPANT_ID_COL, "task", "effect", "phase")
SURVEY_CSV = ROOT / "anonymous_survey_data.csv"
NAME_COLUMN = "Participant ID"
TOPIC_EXPERT_COLUMN = "topic_expert"

NON_TOPIC_GROUP = "Non Topic Experts"
GROUP_ORDER = ["PhD Students", "Senior Scientists", "Topic Experts", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "GenAI"]
# Bars + Non Topic (tests only) + Human (collapsed bars).
VALUE_GROUPS = (
    "PhD Students",
    "Senior Scientists",
    "Topic Experts",
    NON_TOPIC_GROUP,
    "Human",
    "GenAI",
)

GROUP_MAP = {
    0: "PhD Students",
    1: "Senior Scientists",
    2: "GenAI",
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

PHASE_ORDER = ("pre", "post")
PHASE_LABELS = {"pre": "Pre-ML", "post": "Post-ML"}

PANEL_SPECS = (
    ("race", "main", "Racial inequality — Main effects"),
    ("race", "interactions", "Racial inequality — Interactions"),
    ("gender", "main", "Gender inequality — Main effects"),
    ("gender", "interactions", "Gender inequality — Interactions"),
)

SCORE_YMAX = 10
PLOT_YMAX = 12.2

apply_plot_style()


def compute_overall_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scores = out[SCORE_DIMS].apply(pd.to_numeric, errors="coerce")
    out[OVERALL_COL] = scores.mean(axis=1)
    return out


def attach_topic_expert(df: pd.DataFrame) -> pd.DataFrame:
    """Merge survey ``topic_expert`` onto ratings via anonymous participant ID."""
    survey = pd.read_csv(SURVEY_CSV, dtype=str, keep_default_na=False)
    if NAME_COLUMN not in survey.columns or TOPIC_EXPERT_COLUMN not in survey.columns:
        raise KeyError(
            f"Survey CSV missing {NAME_COLUMN!r} or {TOPIC_EXPERT_COLUMN!r}"
        )
    lookup = (
        survey[[NAME_COLUMN, TOPIC_EXPERT_COLUMN]]
        .assign(_id_key=lambda d: d[NAME_COLUMN].astype(str).str.strip())
        .drop_duplicates(subset=["_id_key"], keep="first")
        .set_index("_id_key")[TOPIC_EXPERT_COLUMN]
    )
    out = df.copy()
    keys = out[PARTICIPANT_ID_COL].astype(str).str.strip()
    out[TOPIC_EXPERT_COLUMN] = keys.map(lookup)
    missing = out[TOPIC_EXPERT_COLUMN].isna() | (
        out[TOPIC_EXPERT_COLUMN].astype(str).str.strip() == ""
    )
    if missing.any():
        ids = sorted(out.loc[missing, PARTICIPANT_ID_COL].astype(str).unique())
        raise ValueError(
            f"Could not match topic_expert for {missing.sum()} rating rows; "
            f"unmatched ids: {ids[:10]}"
        )
    out[TOPIC_EXPERT_COLUMN] = out[TOPIC_EXPERT_COLUMN].astype(str).str.strip()
    return out


def audience_targets(group: object, topic_flag: object) -> list[str]:
    """Overlapping labels for bars + Topic/Non Topic test buckets."""
    try:
        gid = int(group)
    except (TypeError, ValueError):
        return []
    gname = GROUP_MAP.get(gid)
    if gname is None:
        return []
    targets = [gname]
    if gname in ("PhD Students", "Senior Scientists"):
        targets.append("Human")
        flag = str(topic_flag).strip()
        if flag == "1":
            targets.append("Topic Experts")
        elif flag == "0":
            targets.append(NON_TOPIC_GROUP)
    return targets


def _normalize_combined(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if PARTICIPANT_ID_COL not in out.columns and "participant_name" in out.columns:
        out = out.rename(columns={"participant_name": PARTICIPANT_ID_COL})
    out[PARTICIPANT_ID_COL] = out[PARTICIPANT_ID_COL].astype(str).str.strip()
    for col in ("task", "effect", "phase"):
        out[col] = out[col].astype(str).str.strip().str.lower()
    out["group"] = pd.to_numeric(out["group"], errors="coerce").astype("Int64")
    for col in SCORE_DIMS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if OVERALL_COL in out.columns:
        out[OVERALL_COL] = pd.to_numeric(out[OVERALL_COL], errors="coerce")
    else:
        out = compute_overall_quality(out)
    if TOPIC_EXPERT_COLUMN in out.columns:
        out[TOPIC_EXPERT_COLUMN] = out[TOPIC_EXPERT_COLUMN].astype(str).str.strip()
    if "rater_identifier" in out.columns:
        out["rater_identifier"] = out["rater_identifier"].astype(str).str.strip()
    return out


def load_all_ratings(path: Path = COMBINED_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing combined ratings: {path}")
    df = _normalize_combined(pd.read_csv(path))
    if TOPIC_EXPERT_COLUMN not in df.columns:
        df = attach_topic_expert(df)
    return df


def load_combined_ratings(
    cohort: CohortSpec, path: Path = COMBINED_CSV
) -> pd.DataFrame:
    df = load_all_ratings(path)
    out = df.loc[df["rater_identifier"] == cohort.key].copy()
    if out.empty:
        raise ValueError(f"No rows for {cohort.key!r} in {path.name}")
    n_unique = out.groupby(list(THEORY_KEYS), sort=False).ngroups
    n_overlap = len(out) - n_unique
    print(
        f"Loaded {cohort.key}: n={len(out)} "
        f"(unique theories={n_unique}, overlap rows kept as independent={n_overlap})"
    )
    return out


def merge_evaluator_ratings(ev1_df: pd.DataFrame, ev2_df: pd.DataFrame) -> pd.DataFrame:
    """Merge two human-evaluator ratings: overlap theories averaged."""
    keys = list(THEORY_KEYS)
    parts = []
    for cohort, df in [(EVALUATOR_1, ev1_df), (EVALUATOR_2, ev2_df)]:
        d = df.copy()
        d["cohort"] = cohort
        parts.append(d)
    both = pd.concat(parts, ignore_index=True)
    by_cohort = both.groupby(
        keys + ["cohort", "group", TOPIC_EXPERT_COLUMN],
        as_index=False,
    ).agg({OVERALL_COL: "mean"})
    return by_cohort.groupby(
        keys + ["group", TOPIC_EXPERT_COLUMN],
        as_index=False,
    ).agg({OVERALL_COL: "mean"})


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = bootstrap_mean_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def panel_values(
    df: pd.DataFrame,
    task: str,
    effect: str,
) -> dict[str, dict[str, list[float]]]:
    sub = df.loc[(df["task"] == task) & (df["effect"] == effect)].copy()
    return _values_from_subset(sub)


def pooled_values(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    return _values_from_subset(df.copy())


def _values_from_subset(sub: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {
        phase: {g: [] for g in VALUE_GROUPS} for phase in PHASE_ORDER
    }
    for _, row in sub.iterrows():
        phase = str(row["phase"]).strip().lower()
        if phase not in out:
            continue
        score = row[OVERALL_COL]
        if pd.isna(score):
            continue
        val = float(score)
        for audience in audience_targets(row["group"], row[TOPIC_EXPERT_COLUMN]):
            out[phase][audience].append(val)
    return out


def _draw_panel(
    ax,
    values: dict[str, dict[str, list[float]]],
    title: str,
    *,
    group_order: list[str] | tuple[str, ...],
    group_colors: dict[str, str],
    pairwise: tuple[tuple[str, str], ...],
    note_fontsize: float = 10,
    notes_layout: str = "stacked",
    sig_text_color: str | None = None,
    sig_legend_fontsize: float | None = None,
) -> None:
    x = np.arange(len(group_order))
    width = 0.34
    offsets = {"pre": -width / 2, "post": width / 2}
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
            if phase == "post":
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
            np.asarray(values["pre"][group], dtype=float),
            np.asarray(values["post"][group], dtype=float),
        )
        if notes_layout == "sig_color_pvals":
            draw_pre_post_bracket(
                ax,
                float(x[gi] + offsets["pre"]),
                float(x[gi] + offsets["post"]),
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
                float(x[gi] + offsets["pre"]),
                float(x[gi] + offsets["post"]),
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
    ax.set_xticklabels([display_label(g) for g in group_order], fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    else:
        ax.set_title("")
    set_axis_labels(ax, None, None, bold_xticks=True)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.tick_params(axis="x", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontweight("bold")
    ax.set_ylim(0, PLOT_YMAX)
    ax.set_yticks(list(range(0, SCORE_YMAX + 1, 2)))

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
                (PHASE_LABELS["pre"], list(zip(phase_comp_sigs["pre"], pair_labels))),
                (PHASE_LABELS["post"], list(zip(phase_comp_sigs["post"], pair_labels))),
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
                    PHASE_LABELS["pre"],
                    list(
                        zip(
                            phase_comp_pvals["pre"] if use_p else phase_comp_sigs["pre"],
                            pairwise,
                        )
                    ),
                ),
                (
                    PHASE_LABELS["post"],
                    list(
                        zip(
                            phase_comp_pvals["post"] if use_p else phase_comp_sigs["post"],
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
        draw_pre_post_sig_columns(
            ax,
            [
                (PHASE_LABELS["pre"], list(zip(phase_comp_sigs["pre"], pair_labels))),
                (PHASE_LABELS["post"], list(zip(phase_comp_sigs["post"], pair_labels))),
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
            phase_notes.append(f"{PHASE_LABELS[phase]}: " + "; ".join(comps))
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
