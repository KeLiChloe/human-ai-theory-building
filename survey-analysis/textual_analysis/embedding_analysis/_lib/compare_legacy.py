"""
Q5 pre vs post comparisons (apple-to-apple by task).

Each task pairs pre-ML and post-ML under the same topic and theory block, e.g.:
    race/main-effects/pre-ML  vs  race/main-effects/post-ML
    gender/soi/pre-ML         vs  gender/soi/post-ML

Prediction 1 reads plot-06 CSV outputs from analysis.py:
    embeddings_openai_text-embedding-3-large/<topic>/<task>/<phase>/semantic_clustering_*_collapsed.csv

Run run_semantic_clustering.py on each pre-ML / post-ML embedding set
before fig_core_tail_structure.py.

Outputs (batch root):
    outputs/core_tail_structure/
        — core_tail_structure.{svg,png}
    outputs/theory_revision/
        — theory_revision.{svg,png} + MixedLM CSVs (panel d only)

Example:
    python fig_core_tail_structure.py
    python fig_theory_revision.py --embeddings-root textual_analysis/.../embeddings_openai_text-embedding-3-large
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.transforms import Bbox
from scipy.stats import kruskal, linregress, pearsonr, spearmanr
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

# Package root is embedding_analysis/ (parent of _lib/), not this file's directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PACKAGE_ROOT
LIB_DIR = Path(__file__).resolve().parent
TEXTUAL_ANALYSIS_DIR = Path(__file__).resolve().parents[2]
SURVEY_ROOT = Path(__file__).resolve().parents[3]
for p in (LIB_DIR, SCRIPT_DIR, TEXTUAL_ANALYSIS_DIR, SURVEY_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import (
    bootstrap_mean_ci,
    p_value_welch_ttest,
)
from viz_style import (
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    BOX_EDGE_NEUTRAL,
    BOX_STYLE_PAD,
    COMPARE_PAD_PX,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    FONT_COMPARISON,
    FOOTNOTE_COLOR,
    GROUP_COLORS_COLLAPSED,
    GROUP_ORDER_COLLAPSED,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    save_figure_pdf_svg,
    VIZ_AXIS_LABEL_FONTSIZE,
    VIZ_BRACKET_FONTSIZE,
    VIZ_FOOTNOTE_FONTSIZE,
    VIZ_FOOTNOTE_LINE_STEP,
    VIZ_HEADER_VERTICAL_SHIFT,
    VIZ_LEGEND_FONTSIZE,
    VIZ_PANEL_TITLE_FONTSIZE,
    VIZ_SUPTITLE_FONTSIZE,
    VIZ_SUPTITLE_LINE_SPACING,
    VIZ_SUPYLABEL_X,
    VIZ_TICK_FONTSIZE,
    apply_plot_style,
    display_label,
    draw_paired_pre_post_bracket,
    format_comparison_line,
    format_p_value_label,
    format_p_value_with_trailing_stars,
    fmt_p,
    is_significant,
    layout_title_and_metric,
    SIG_LEVEL_LEGEND,
    SIG_TEXT_COLOR,
    significance_label,
)

from analysis_legacy import (
    ANALYSIS_SEED,
    COLLAPSED_PARTICIPANT_TYPE_COL,
    COMPARISONS_CORE_TAIL_SUBDIR,
    COMPARISONS_SELF_SUBDIR,
    COMPARISON_FOOTNOTE_LINE_HEIGHT,
    COMPARISON_FOOTNOTE_XLABEL_GAP,
    COMPARISON_FOOTNOTE_XLABEL_HEIGHT,
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    DIVERSITY_PRED_BOX_ASPECT,
    DIVERSITY_PRED_COL_GAP,
    DIVERSITY_PRED_FIGSIZE,
    DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
    DIVERSITY_PRED_ROW_GAP,
    DIVERSITY_PRED_XTICK_FONTSIZE,
    DIVERSITY_PRED_YLABEL_FONTSIZE,
    DIVERSITY_PRED_YLABEL_X,
    DIVERSITY_PRED_YTICK_FONTSIZE,
    DIVERSITY_TASK_PANEL_ORDER,
    EMBEDDING_SET_PART_LABELS,
    GROUP_COLORS_BY_PARTICIPANT_TYPE,
    PARTICIPANT_NAME_COL,
    PARTICIPANT_TYPE_COL,
    PARTICIPANT_TYPE_TO_LEGEND,
    PHASE_GRID_AXIS_FONTSIZE,
    PHASE_GRID_LEGEND_FONTSIZE,
    PHASE_GRID_PANEL_TITLE_FONTSIZE,
    PHASE_GRID_SEMANTIC_MAP_BOTTOM_EXTRA,
    PHASE_GRID_SEMANTIC_MAP_BOX_ASPECT,
    PHASE_GRID_SEMANTIC_MAP_SUPXLABEL_Y,
    PHASE_GRID_SUPYLABEL_X,
    PHASE_GRID_TICK_FONTSIZE,
    SEMANTIC_CLUSTERING_PARTICIPANT_CSV,
    SEMANTIC_CLUSTERING_SUMMARY_CSV,
    add_phase_grid_figure_legend,
    apply_semantic_map_2d_bounds,
    available_embedding_columns,
    ci_errorbar_offsets,
    clean_participant_display_name,
    comparison_pairs_for_groups,
    comparisons_pre_post_dir,
    diversity_comparison_bottom_layout,
    hdbscan_cluster_selection_epsilon,
    infer_embeddings_root,
    make_phase_grid_axes,
    ordered_groups,
    phase_grid_layout_adjust,
    phase_grid_semantic_map_axis_bounds,
    respondent_name_map_for_embeddings_root,
    read_anonymized_embeddings_wide,
    resolve_task_data_dir,
    core_pct_parameter_sensitivity,
    core_pct_with_wilson_ci,
    run_hdbscan_within_group,
    safe_name,
    stack_embeddings,
    summarize_core_tail,
    with_collapsed_group,
)

GENAI_TYPE = "GenAI"

TASK_PANEL_ORDER = [
    "race/main-effects",
    "race/soi",
    "gender/main-effects",
    "gender/soi",
]

CORE_TAIL_LATEX_STEM = "core_tail_structure"
CORE_TAIL_SENSITIVITY_FIG_STEM = "core_tail_hdbscan_sensitivity"
CORE_TAIL_BASE_EPSILON = 0.1365
SELF_PRE_POST_METRIC_SUBTITLE = (
    "Metric: cosine distance between each respondent's pre-ML and post-ML embedding "
    "(self-comparison).",
)
SELF_PRE_POST_BAR_WIDTH = 0.55
SELF_PRE_POST_YLIM_PAD = 1.22
SELF_PRE_POST_COMPARE_ABOVE_PAD_FRAC = 0.055
SELF_PRE_POST_COMPARE_ABOVE_LINE_FRAC = 0.075
SELF_PRE_POST_COLLAPSED_LEGEND_Y_SHIFT = 0.018
SELF_PRE_POST_FOOTNOTE_Y = 0.048
SELF_PRE_POST_WELCH_COLLAPSED_FOOTNOTE = (
    "Two-sided Welch t-test on mean self pre–post cosine distance (Humans vs GenAI).",
    SIG_LEVEL_LEGEND,
)
SELF_PRE_POST_DISTANCE_GROUPS = GROUP_ORDER_COLLAPSED
SELF_PRE_POST_DISTANCE_GROUP_LABELS = {
    g: display_label(g) for g in GROUP_ORDER_COLLAPSED
}
SELF_PRE_POST_DISTANCE_GROUP_COLORS = GROUP_COLORS_COLLAPSED
SELF_PRE_POST_DISTANCE_SUPTITLE = (
    "Pre–post embedding shift within respondent (self-comparison)\n"
    "Humans and GenAI"
)

SURVEY_CSV_PATH = SURVEY_ROOT / "anonymous_survey_data.csv"
ME_ML_PATH = SURVEY_ROOT / "forecasts" / "ML_results_main_effects.json"
SOI_ML_PATH = SURVEY_ROOT / "forecasts" / "ML_results_second_order_interactions.json"
FORECAST_SIGN_MAP = {"+": 1, "-": -1}
HUMAN_GROUP_IDS = frozenset({"0", "1"})
GENAI_GROUP_ID = "2"

TASK_PRE_ACCURACY_COLUMN = "pre_ml_accuracy"

SELF_PRE_POST_SHIFT_ACCURACY_KW_VARIANTS = {
    "humans": {
        "group_key": "Human",
        "scope": "Humans",
        "fig": "shift_by_pre_ml_accuracy_high_low_kruskal_wallis_humans.png",
        "suptitle": (
            "Theory-embedding shift by pre-ML accuracy (high vs. low)\n"
            "Humans"
        ),
    },
    "genai": {
        "group_key": "GenAI",
        "scope": "GenAI",
        "fig": "shift_by_pre_ml_accuracy_high_low_kruskal_wallis_genai.png",
        "suptitle": (
            "Theory-embedding shift by pre-ML accuracy (high vs. low)\n"
            "GenAI"
        ),
    },
}
SHIFT_ACCURACY_BINARY_ORDER = ("low", "high")
SHIFT_ACCURACY_BINARY_LABELS = {
    "low": "Low",
    "high": "High",
}
SHIFT_ACCURACY_BINARY_BAR_WIDTH = 0.58
SHIFT_ACCURACY_BINARY_BAR_ALPHAS = (0.55, 1.0)
SELF_PRE_POST_SHIFT_ACCURACY_KW_METRIC = (
    "Within each task: sort unique pre-ML accuracy values; lower half of levels → Low, upper half → High.",
    "Bars: mean theory-embedding shift with bootstrap 95% CI.",
)
SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE = (SIG_LEVEL_LEGEND,)
SELF_PRE_POST_SHIFT_ACCURACY_KW_SAVE_PAD = 0.18
SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE_Y = 0.0
SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE_GAP = 0.016
SELF_PRE_POST_SHIFT_ACCURACY_KW_XTICK_LABEL_HEIGHT = 0.040


def shift_accuracy_high_low_bottom_layout(
    footnote: tuple[str, ...],
) -> tuple[float, float]:
    """Return (footnote_y, subplot_bottom) from figure bottom up."""
    footnote_y = SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE_Y
    footnote_top = footnote_y + COMPARISON_FOOTNOTE_LINE_HEIGHT
    subplot_bottom = (
        footnote_top
        + SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE_GAP
        + SELF_PRE_POST_SHIFT_ACCURACY_KW_XTICK_LABEL_HEIGHT
    )
    return footnote_y, subplot_bottom
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_VARIANTS = {
    "humans": {
        "group_key": "Human",
        "scope": "Humans",
        "fig": "shift_vs_pre_ml_accuracy_scatter_humans.png",
        "suptitle": (
            "Theory-embedding shift vs. pre-ML prediction accuracy\n"
            "Humans"
        ),
        "marker_size": 34,
        "marker_alpha": 0.62,
    },
    "genai": {
        "group_key": "GenAI",
        "scope": "GenAI",
        "fig": "shift_vs_pre_ml_accuracy_scatter_genai.png",
        "suptitle": (
            "Theory-embedding shift vs. pre-ML prediction accuracy\n"
            "GenAI"
        ),
        "marker_size": 42,
        "marker_alpha": 0.78,
    },
}
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_METRIC = (
    "Pre-ML accuracy = cosine similarity between each respondent's pre-ML prediction",
    "and the ML benchmark (task-matched).",
    "Shift = cosine distance between pre-ML and post-ML theory embeddings.",
    "Dashed line: OLS fit.",
)
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_SAVE_PAD = 0.14
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_FOOTNOTE = (SIG_LEVEL_LEGEND,)
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_SUPXLABEL_GAP = 0.010
SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_TICK_LABEL_HEIGHT = 0.030

SELF_PRE_POST_POOLED_THREE_PANEL_FIG = "theory_revision.png"
SELF_PRE_POST_POOLED_THREE_PANEL_SUPTITLE = (
    "Extent of theory revision after ML and its link to pre-ML accuracy"
)
# Nature-style star key (no NS); placed below the figure in black, no box.
SELF_PRE_POST_POOLED_THREE_PANEL_SIG_LEGEND = (
    "*p < 0.05, **p < 0.01, ***p < 0.001"
)
SELF_PRE_POST_POOLED_THREE_PANEL_FOOTNOTE = (
    SELF_PRE_POST_POOLED_THREE_PANEL_SIG_LEGEND,
)
SELF_PRE_POST_POOLED_THREE_PANEL_SAVE_PAD = 0.12
SELF_PRE_POST_POOLED_THREE_PANEL_XLABEL = "Pre-ML prediction accuracy"
SELF_PRE_POST_POOLED_THREE_PANEL_YLABEL = "Extent of theory revision"
SELF_PRE_POST_POOLED_THREE_PANEL_AXIS_FONTSIZE = 20
# Nudge panel-(a) significance key slightly downward.
SELF_PRE_POST_POOLED_THREE_PANEL_SIG_Y_SHIFT = -0.018
SELF_PRE_POST_POOLED_THREE_PANEL_TITLES = (
    "a. Mean pre–post cosine distance",
    "b. Humans: revision vs pre-ML accuracy",
    "c. GenAI: revision vs pre-ML accuracy",
    r"d\enspace Mixed-effects model of theory revision",
)

SELF_PRE_POST_POOLED_ABC_STEM = "_theory_revision_abc_panels"
SELF_PRE_POST_MIXEDLM_CSV = "revision_vs_prior_error_mixedlm.csv"
SELF_PRE_POST_MIXEDLM_SLOPES_CSV = "revision_vs_prior_error_mixedlm_group_slopes.csv"
SELF_PRE_POST_MIXEDLM_TABLE_STEM = "revision_vs_prior_error_mixedlm_table"
# PriorError = 1 − pre-ML accuracy; Revision = self pre–post cosine distance.
SELF_PRE_POST_POOLED_TABLE_TERM_LABELS = {
    "Intercept": r"Intercept ($\beta_0$)",
    "prior_error": r"Prior error ($\beta_1$)",
    "C(group, Treatment(reference='Human'))[T.GenAI]": r"GenAI ($\beta_2$)",
    "prior_error:C(group, Treatment(reference='Human'))[T.GenAI]": (
        r"Prior error $\times$ GenAI ($\beta_3$)"
    ),
    "C(task)[T.gender/soi]": r"Gender $\cdot$ Interactions",
    "C(task)[T.race/main-effects]": r"Race $\cdot$ Main effects",
    "C(task)[T.race/soi]": r"Race $\cdot$ Interactions",
}
# Column order: β0, β1, β2, β3 (then implied slopes appended in the builder).
SELF_PRE_POST_POOLED_MIXEDLM_MAIN_TERMS = (
    "Intercept",
    "prior_error",
    "C(group, Treatment(reference='Human'))[T.GenAI]",
    "prior_error:C(group, Treatment(reference='Human'))[T.GenAI]",
)
SELF_PRE_POST_POOLED_TABLE_COL_HEADERS = {
    "Intercept": r"\makecell[c]{Intercept\\($\beta_0$)}",
    "prior_error": r"\makecell[c]{Prior error\\($\beta_1$)}",
    "C(group, Treatment(reference='Human'))[T.GenAI]": (
        r"\makecell[c]{GenAI\\($\beta_2$)}"
    ),
    "prior_error:C(group, Treatment(reference='Human'))[T.GenAI]": (
        r"\makecell[c]{Prior$\times$GenAI\\($\beta_3$)}"
    ),
    "C(task)[T.gender/soi]": r"\makecell[c]{Gender\\Interactions}",
    "C(task)[T.race/main-effects]": r"\makecell[c]{Race\\Main effects}",
    "C(task)[T.race/soi]": r"\makecell[c]{Race\\Interactions}",
}
SELF_PRE_POST_POOLED_MIXEDLM_TASK_FE_TERMS = (
    "C(task)[T.race/main-effects]",
    "C(task)[T.race/soi]",
    "C(task)[T.gender/soi]",
)
# Keep legacy alias used elsewhere.
SELF_PRE_POST_POOLED_MIXEDLM_TERM_ORDER = (
    *SELF_PRE_POST_POOLED_MIXEDLM_MAIN_TERMS,
    *SELF_PRE_POST_POOLED_MIXEDLM_TASK_FE_TERMS,
)


def shift_accuracy_scatter_bottom_layout(
    footnote: tuple[str, ...],
) -> tuple[float, float, float]:
    """Return (footnote_y, subplot_bottom, shared_xlabel_y) from figure bottom up."""
    footnote_y, _ = diversity_comparison_bottom_layout(footnote)
    footnote_top = footnote_y + COMPARISON_FOOTNOTE_LINE_HEIGHT
    shared_xlabel_y = (
        footnote_top
        + COMPARISON_FOOTNOTE_XLABEL_GAP
        + COMPARISON_FOOTNOTE_XLABEL_HEIGHT * 0.5
    )
    subplot_bottom = (
        shared_xlabel_y
        + COMPARISON_FOOTNOTE_XLABEL_HEIGHT * 0.5
        + SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_SUPXLABEL_GAP
        + SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_TICK_LABEL_HEIGHT
    )
    return footnote_y, subplot_bottom, shared_xlabel_y


def pooled_three_panel_bottom_layout(
    footnote: tuple[str, ...],
) -> tuple[float, float]:
    """Return (footnote_y, subplot_bottom) for per-panel x-labels (no shared xlabel)."""
    footnote_y, _ = diversity_comparison_bottom_layout(footnote)
    footnote_top = footnote_y + COMPARISON_FOOTNOTE_LINE_HEIGHT
    subplot_bottom = (
        footnote_top
        + COMPARISON_FOOTNOTE_XLABEL_GAP
        + SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_TICK_LABEL_HEIGHT
        + COMPARISON_FOOTNOTE_XLABEL_HEIGHT
        + SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_SUPXLABEL_GAP
    )
    return footnote_y, subplot_bottom

SELF_PRE_POST_PCA_PRE_SIZE = 30
SELF_PRE_POST_PCA_POST_SIZE = 42
SELF_PRE_POST_PCA_PRE_EDGEWIDTH = 1.15
SELF_PRE_POST_PCA_POST_EDGEWIDTH = 0.45
SELF_PRE_POST_PCA_LINE_ALPHA = 0.42
SELF_PRE_POST_PCA_LINEWIDTH = 0.85
SELF_PRE_POST_PCA_LEGEND_NCOL = 5
SELF_PRE_POST_PCA_COMBINED_FIG = "pre_post_trajectories_human_genai.png"
# Deprecated trajectory stems (combined + single-audience); deleted on rerun.
SELF_PRE_POST_PCA_REMOVED_FIGS = (
    "pre_post_trajectories_human_genai",
    "pre_post_trajectories_humans",
    "pre_post_trajectories_phd_students",
    "pre_post_trajectories_seniors",
    "pre_post_trajectories_genai",
)
SELF_PRE_POST_PCA_COMBINED_SUPTITLE = (
    "Pre–post ML updating trajectories in PCA\n"
    "Humans and GenAI"
)
SELF_PRE_POST_PCA_COMBINED_METRIC = (
    "2D PCA fit on pooled Pre+Post embeddings per task "
    "(same basis as semantic_space_map).",
)
# Deprecated single-audience trajectory stems (kept for older call sites).
SELF_PRE_POST_PCA_LEGACY_FIGS = SELF_PRE_POST_PCA_REMOVED_FIGS[1:]

SELF_PRE_POST_PCA_AUDIENCES = (
    {
        "key": "Human",
        "panel_title": "(a) Humans",
        "collapsed": True,
        "filter_group": "Human",
        "group_label": display_label("Human"),
        "group_color": GROUP_COLORS_COLLAPSED["Human"],
    },
    {
        "key": "GenAI",
        "panel_title": "(b) GenAI",
        "collapsed": True,
        "filter_group": "GenAI",
        "group_label": display_label("GenAI"),
        "group_color": GROUP_COLORS_COLLAPSED["GenAI"],
    },
)

AUDIENCE_CONFIGS = {
    "human": {
        "csv": "human_core_tail_pre_post_by_task.csv",
        "audience_group": "Human",
        "count_label": "human",
    },
    "genai": {
        "csv": "genai_core_tail_pre_post_by_task.csv",
        "audience_group": GENAI_TYPE,
        "count_label": "genai",
    },
}

apply_plot_style()


def format_task_part(part: str) -> str:
    key = part.lower()
    if key in EMBEDDING_SET_PART_LABELS:
        return EMBEDDING_SET_PART_LABELS[key]
    return part.replace("-", " ").title().replace("Ml", "ML")


def task_label_from_key(task_key: str) -> str:
    return " · ".join(format_task_part(part) for part in task_key.split("/"))


def discover_task_pairs(embeddings_root: Path) -> List[Tuple[Path, Path, str]]:
    """Return (pre_dir, post_dir, task_key) for each topic/task with both phases."""
    pairs: List[Tuple[Path, Path, str]] = []
    for pre_dir in sorted(embeddings_root.rglob("pre-ML")):
        if not (pre_dir / "embeddings_wide.parquet").exists():
            continue
        post_dir = pre_dir.parent / "post-ML"
        if not (post_dir / "embeddings_wide.parquet").exists():
            continue
        task_key = str(pre_dir.relative_to(embeddings_root).parent)
        pairs.append((pre_dir, post_dir, task_key))
    if not pairs:
        raise FileNotFoundError(
            f"No pre-ML/post-ML pairs found under {embeddings_root}."
        )
    return pairs


def clustering_analysis_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path,
) -> Path:
    return resolve_task_data_dir(
        embeddings_root, embedding_set_dir, embedding_col
    )


def clustering_csv_paths(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path,
) -> dict[str, Path]:
    space_dir = clustering_analysis_dir(
        embedding_set_dir, embedding_col, embeddings_root
    )
    collapsed = "_collapsed"
    return {
        "summary": space_dir / SEMANTIC_CLUSTERING_SUMMARY_CSV.format(suffix=collapsed),
        "participant": space_dir / SEMANTIC_CLUSTERING_PARTICIPANT_CSV.format(
            suffix=collapsed
        ),
    }


def require_analysis_csvs(paths: dict[str, Path], embedding_set_dir: Path) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing analysis clustering CSV(s). Run analysis.py on this embedding set first:\n"
            f"  {embedding_set_dir}\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def load_group_summary_row(csv_path: Path, audience_group: str) -> dict:
    summary_df = pd.read_csv(csv_path)
    match = summary_df.loc[summary_df["participant_type"] == audience_group]
    if match.empty:
        raise ValueError(
            f"No row for {audience_group!r} in {csv_path.name} "
            f"(available: {summary_df['participant_type'].tolist()})"
        )
    return match.iloc[0].to_dict()


def load_participant_clustering(csv_path: Path, audience_group: str) -> pd.DataFrame:
    participant_df = pd.read_csv(csv_path)
    group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    if group_col not in participant_df.columns:
        raise ValueError(f"Expected column {group_col!r} in {csv_path}")
    subset = participant_df.loc[
        participant_df[group_col] == audience_group
    ].copy()
    if subset.empty:
        raise ValueError(f"No {audience_group} rows in {csv_path}")
    return subset


def load_paired_audience_embeddings(
    pre_dir: Path,
    post_dir: Path,
    embedding_col: str,
    *,
    audience_group: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Aligned pre/post unit-norm embeddings for one collapsed audience group."""
    pre_df = with_collapsed_group(
        read_anonymized_embeddings_wide(pre_dir / "embeddings_wide.parquet")
    )
    post_df = with_collapsed_group(
        read_anonymized_embeddings_wide(post_dir / "embeddings_wide.parquet")
    )
    pre_sub = pre_df.loc[
        pre_df[COLLAPSED_PARTICIPANT_TYPE_COL] == audience_group,
        [PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL, embedding_col],
    ].rename(columns={embedding_col: f"{embedding_col}_pre"})
    post_sub = post_df.loc[
        post_df[COLLAPSED_PARTICIPANT_TYPE_COL] == audience_group,
        [PARTICIPANT_NAME_COL, embedding_col],
    ].rename(columns={embedding_col: f"{embedding_col}_post"})
    merged = pre_sub.merge(post_sub, on=PARTICIPANT_NAME_COL, how="inner")
    if merged.empty:
        raise ValueError(
            f"No matched {audience_group} participants under {pre_dir.parent}"
        )
    X_pre = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_pre": embedding_col}),
            embedding_col,
        )
    )
    X_post = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_post": embedding_col}),
            embedding_col,
        )
    )
    meta = merged[[PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL]].copy()
    return X_pre, X_post, meta


def analyze_prediction1_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
    embeddings_root: Path,
    *,
    audience_group: str,
    count_label: str,
) -> dict:
    pre_paths = clustering_csv_paths(pre_dir, embedding_col, embeddings_root)
    post_paths = clustering_csv_paths(post_dir, embedding_col, embeddings_root)
    require_analysis_csvs(pre_paths, pre_dir)
    require_analysis_csvs(post_paths, post_dir)

    epsilon = hdbscan_cluster_selection_epsilon(embedding_col)
    X_pre, X_post, meta = load_paired_audience_embeddings(
        pre_dir,
        post_dir,
        embedding_col,
        audience_group=audience_group,
    )
    pre_labels, _, _, _ = run_hdbscan_within_group(
        X_pre, cluster_selection_epsilon=epsilon
    )
    post_labels, _, _, _ = run_hdbscan_within_group(
        X_post, cluster_selection_epsilon=epsilon
    )
    pre_sum = summarize_core_tail(pre_labels)
    post_sum = summarize_core_tail(post_labels)

    pre_core, pre_lo, pre_hi = core_pct_with_wilson_ci(pre_labels)
    post_core, post_lo, post_hi = core_pct_with_wilson_ci(post_labels)
    pre_sens = core_pct_parameter_sensitivity(X_pre, base_epsilon=epsilon)
    post_sens = core_pct_parameter_sensitivity(X_post, base_epsilon=epsilon)
    pre_sens_vals = [float(r["core_pct"]) for r in pre_sens]
    post_sens_vals = [float(r["core_pct"]) for r in post_sens]

    return {
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "audience_group": audience_group,
        "hdbscan_epsilon": float(epsilon),
        f"n_{count_label}_paired": int(meta.shape[0]),
        "n_student": int((meta[PARTICIPANT_TYPE_COL] == "student").sum()),
        "n_senior": int((meta[PARTICIPANT_TYPE_COL] == "senior").sum()),
        "n_genai": int((meta[PARTICIPANT_TYPE_COL] == GENAI_TYPE).sum()),
        "pre_core_pct": float(pre_sum["core_pct"]),
        "pre_core_pct_ci_method": "wilson",
        "pre_core_ci_lo": float(pre_lo),
        "pre_core_ci_hi": float(pre_hi),
        "pre_tail_pct": float(pre_sum["tail_pct"]),
        "post_core_pct": float(post_sum["core_pct"]),
        "post_core_pct_ci_method": "wilson",
        "post_core_ci_lo": float(post_lo),
        "post_core_ci_hi": float(post_hi),
        "post_tail_pct": float(post_sum["tail_pct"]),
        "pre_core_sens_min": float(np.min(pre_sens_vals)) if pre_sens_vals else np.nan,
        "pre_core_sens_max": float(np.max(pre_sens_vals)) if pre_sens_vals else np.nan,
        "post_core_sens_min": float(np.min(post_sens_vals)) if post_sens_vals else np.nan,
        "post_core_sens_max": float(np.max(post_sens_vals)) if post_sens_vals else np.nan,
        "tail_pct_delta_post_minus_pre": float(
            post_sum["tail_pct"] - pre_sum["tail_pct"]
        ),
        "core_pct_delta_post_minus_pre": float(
            post_sum["core_pct"] - pre_sum["core_pct"]
        ),
        "post_still_has_core_and_tail": bool(
            post_sum["core_pct"] > 0 and post_sum["tail_pct"] > 0
        ),
        "post_tail_shorter_than_pre": bool(
            post_sum["tail_pct"] < pre_sum["tail_pct"]
        ),
        "pre_n_clusters": int(pre_sum["n_clusters"]),
        "post_n_clusters": int(post_sum["n_clusters"]),
        "_pre_sensitivity_rows": pre_sens,
        "_post_sensitivity_rows": post_sens,
    }


def _core_tail_fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "---"
    return f"{x:.1f}"


def _core_tail_fmt_pct_ci(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point):
        return "---"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _core_tail_fmt_pct(point)
    return (
        r"\begin{tabular}{@{}c@{}}"
        rf"{point:.1f}\\"
        rf"$[{lo:.1f},\ {hi:.1f}]$"
        r"\end{tabular}"
    )


def _core_tail_fmt_post_pct_ci(
    post: float,
    pre: float,
    lo: float,
    hi: float,
) -> str:
    """Post-ML core % with CI and ↑/↓ relative to Pre-ML."""
    base = _core_tail_fmt_pct_ci(post, lo, hi)
    if base == "---" or not np.isfinite(pre):
        return base
    arrow = ""
    if post > pre:
        arrow = r"{\(\uparrow\)}"
    elif post < pre:
        arrow = r"{\(\downarrow\)}"
    if not arrow:
        return base
    # Inject arrow after the point estimate line.
    return base.replace(
        rf"{post:.1f}\\",
        rf"{post:.1f}{arrow}\\",
        1,
    )


def _core_tail_fmt_n_cores(x: float) -> str:
    if not np.isfinite(x):
        return "---"
    return str(int(x))


def _core_tail_n(row: pd.Series) -> int:
    if "n_human_paired" in row.index and pd.notna(row.get("n_human_paired")):
        return int(row["n_human_paired"])
    if "n_genai_paired" in row.index and pd.notna(row.get("n_genai_paired")):
        return int(row["n_genai_paired"])
    return int(row.get("n_genai", 0) or 0)


def _core_tail_task_parts(task_key: str) -> tuple[str, str]:
    """Split ``race/main-effects`` → (``Race``, ``Main effects``)."""
    parts = str(task_key).split("/")
    if len(parts) != 2:
        label = task_label_from_key(task_key)
        return label, ""
    topic = format_task_part(parts[0])
    design_key = parts[1].lower()
    if design_key == "soi":
        design = "Interactions"
    elif design_key in {"main-effects", "main_effects"}:
        design = "Main effects"
    else:
        design = format_task_part(parts[1])
    return topic, design


def build_core_tail_pre_post_latex(
    human_df: pd.DataFrame,
    genai_df: pd.DataFrame,
) -> str:
    """Nature/Science-style booktabs + siunitx core-fraction Pre/Post table."""
    order = {key: i for i, key in enumerate(TASK_PANEL_ORDER)}
    blocks = [
        ("Humans", human_df),
        ("GenAI", genai_df),
    ]
    body_lines: list[str] = []
    for bi, (block_label, df) in enumerate(blocks):
        plot_df = df.sort_values(
            "task_key", key=lambda s: s.map(order)
        ).reset_index(drop=True)
        n_rows = len(plot_df)
        prev_topic: str | None = None
        topic_keys_ordered: list[str] = []
        for key in plot_df["task_key"]:
            topic_key = str(key).split("/")[0]
            if topic_key not in topic_keys_ordered:
                topic_keys_ordered.append(topic_key)
        topic_span = {
            topic: int(
                sum(
                    1
                    for key in plot_df["task_key"]
                    if str(key).split("/")[0] == topic
                )
            )
            for topic in topic_keys_ordered
        }
        for j, row in plot_df.iterrows():
            group_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{{block_label}}}" if j == 0 else ""
            )
            topic, design = _core_tail_task_parts(str(row["task_key"]))
            topic_key = str(row["task_key"]).split("/")[0]
            if topic_key != prev_topic:
                topic_cell = rf"\multirow{{{topic_span[topic_key]}}}{{*}}{{{topic}}}"
                prev_topic = topic_key
            else:
                topic_cell = ""
            pre_core = float(row["pre_core_pct"])
            post_core = float(row["post_core_pct"])
            cells = [
                group_cell,
                topic_cell,
                design,
                _core_tail_fmt_pct_ci(
                    pre_core,
                    float(row.get("pre_core_ci_lo", np.nan)),
                    float(row.get("pre_core_ci_hi", np.nan)),
                ),
                _core_tail_fmt_post_pct_ci(
                    post_core,
                    pre_core,
                    float(row.get("post_core_ci_lo", np.nan)),
                    float(row.get("post_core_ci_hi", np.nan)),
                ),
                _core_tail_fmt_n_cores(float(row["pre_n_clusters"])),
                _core_tail_fmt_n_cores(float(row["post_n_clusters"])),
            ]
            body_lines.append(" & ".join(cells) + r" \\")
            body_lines.append("")
        if bi < len(blocks) - 1:
            body_lines.append(r"\midrule")
            body_lines.append("")

    return "\n".join(
        [
            "% Auto-generated by fig_core_tail_structure / fig_theory_revision",
            "% Required packages:",
            "% \\usepackage{booktabs}",
            "% \\usepackage{multirow}",
            "% \\usepackage{siunitx}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            "",
            r"\caption{\mbox{\textbf{Core fraction of theoretical explanations "
            r"before and after ML exposure}}}",
            r"\label{tab:core_tail_pre_post}",
            "",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4.5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            "",
            r"\sisetup{",
            r"  table-number-alignment = center,",
            r"  detect-weight = true,",
            r"  detect-family = true",
            r"}",
            "",
            r"\begin{tabular}{",
            r"  @{}",
            r"  l",
            r"  l",
            r"  l",
            r"  c",
            r"  c",
            r"  S[table-format=1.0]",
            r"  S[table-format=1.0]",
            r"  @{}",
            r"}",
            r"\toprule",
            r"Group",
            r"& Topic",
            r"& Effect",
            r"& \multicolumn{2}{c}{Core fraction (\%)}",
            r"& \multicolumn{2}{c}{No.\ of cores} \\",
            r"\cmidrule(lr){4-5} \cmidrule(l){6-7}",
            r"&",
            r"&",
            r"& {Pre-ML}",
            r"& {Post-ML}",
            r"& {Pre-ML}",
            r"& {Post-ML} \\",
            r"\midrule",
            "",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\end{table}",
            "",
        ]
    )


def _core_tail_panel_title(task_key: str) -> str:
    topic, design = _core_tail_task_parts(task_key)
    return f"{topic}: {design}" if design else topic


def plot_core_tail_hdbscan_sensitivity(
    human_sens: pd.DataFrame,
    genai_sens: pd.DataFrame,
    out_stem: Path,
    *,
    base_epsilon: float = CORE_TAIL_BASE_EPSILON,
) -> Path:
    """2×2 line charts: core % vs ``min_cluster_size`` by task.

    X-axis is HDBSCAN ``min_cluster_size``. Lines are Humans/GenAI × Pre/Post
    at the tuned ``cluster_selection_epsilon``.
    """
    apply_plot_style()
    sens = pd.concat(
        [
            human_sens.assign(_group="Human"),
            genai_sens.assign(_group="GenAI"),
        ],
        ignore_index=True,
    )
    eps = sens["cluster_selection_epsilon"].astype(float)
    sens = sens.loc[np.isclose(eps, float(base_epsilon))].copy()
    if sens.empty:
        raise ValueError(
            f"No sensitivity rows for cluster_selection_epsilon={base_epsilon}."
        )

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.6), sharex=True, sharey=True)
    # Color = group; linestyle + marker = phase (Pre vs Post).
    line_specs = (
        ("Human", "pre", "-", "o"),
        ("Human", "post", "--", "s"),
        ("GenAI", "pre", "-", "o"),
        ("GenAI", "post", "--", "s"),
    )

    for ax, task_key in zip(axes.ravel(), TASK_PANEL_ORDER):
        cell = sens.loc[sens["task_key"].astype(str) == task_key]
        for group, phase, ls, marker in line_specs:
            sub = cell.loc[
                (cell["_group"] == group) & (cell["phase"].astype(str) == phase)
            ].sort_values("min_cluster_size")
            if sub.empty:
                continue
            ax.plot(
                sub["min_cluster_size"].to_numpy(dtype=float),
                sub["core_pct"].to_numpy(dtype=float),
                color=GROUP_COLORS_COLLAPSED[group],
                ls=ls,
                lw=1.6,
                marker=marker,
                ms=5.0,
                markerfacecolor=GROUP_COLORS_COLLAPSED[group],
                markeredgecolor=GROUP_COLORS_COLLAPSED[group],
            )
        ax.axvline(
            2.0,
            color="#888888",
            lw=0.8,
            ls=":",
            zorder=1,
        )
        ax.set_title(_core_tail_panel_title(task_key), fontsize=9, pad=2)
        ax.set_ylim(-3.0, 105.0)
        ax.set_xticks(list(range(2, 11)))
        ax.tick_params(labelsize=7.5)

    for ax in axes[1, :]:
        ax.set_xlabel(r"min_cluster_size", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Core fraction (%)", fontsize=9)

    legend_handles = [
        Line2D(
            [0], [0],
            color=GROUP_COLORS_COLLAPSED["Human"],
            lw=1.6,
            label="Humans",
        ),
        Line2D(
            [0], [0],
            color=GROUP_COLORS_COLLAPSED["GenAI"],
            lw=1.6,
            label="GenAI",
        ),
        Line2D(
            [0], [0],
            color="#444444",
            lw=1.6,
            ls="-",
            marker="o",
            ms=5.0,
            label="Pre-ML",
        ),
        Line2D(
            [0], [0],
            color="#444444",
            lw=1.6,
            # Short dash cycle so dashes appear on both sides of the marker.
            ls=(0, (2.2, 1.4)),
            marker="s",
            ms=5.0,
            label="Post-ML",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=3.6,
        handletextpad=0.55,
        columnspacing=1.2,
        markerscale=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    saved = save_figure_pdf_svg(fig, out_stem, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return next(p for p in saved if p.suffix == ".png")


def _core_tail_table_body_only(full_table_tex: str) -> str:
    """Strip float wrapper from the manuscript table fragment for standalone compile."""
    return (
        full_table_tex.replace(r"\begin{table}[t]", "")
        .replace(r"\centering", "")
        .replace(r"\end{table}", "")
    )


def build_core_tail_combined_tex(
    table_body_tex: str,
    *,
    sensitivity_pdf_name: str,
) -> str:
    """Panel a = core-fraction table; panel b = HDBSCAN sensitivity 2×2."""
    # Drop caption/label from the embedded table (outer figure has panel titles).
    lines = []
    skip_prefixes = (r"\caption{", r"\label{")
    for line in table_body_tex.splitlines():
        if any(line.strip().startswith(p) for p in skip_prefixes):
            continue
        lines.append(line)
    table_inner = "\n".join(lines).strip()
    return "\n".join([
        "% Auto-generated by fig_core_tail_structure / fig_theory_revision",
        "",
        r"\begin{tabular}{@{}l@{}}",
        r"{\small\textbf{a.\enspace Core fraction of theoretical explanations "
        r"before and after ML exposure}}\\[0.35em]",
        table_inner,
        r" \\",
        r"\rule{0pt}{1.2em} \\",
        r"{\small\textbf{b.\enspace Sensitivity of HDBSCAN core fraction to minimum cluster size (2--10)}}\\[0.35em]",
        rf"\includegraphics[width=5.8in]{{{sensitivity_pdf_name}}}",
        r"\end{tabular}",
        "",
    ])


def write_core_tail_pre_post_latex_table(
    human_df: pd.DataFrame,
    genai_df: pd.DataFrame,
    outdir: Path,
    *,
    human_sens: pd.DataFrame | None = None,
    genai_sens: pd.DataFrame | None = None,
) -> Path:
    """Compile cropped combined preview ``.svg`` + ``.png`` (no manuscript ``.tex`` kept).

    The figure stacks the table with a 2×2 HDBSCAN sensitivity line chart when
    sensitivity dataframes are provided.
    """
    from common.latex_table_pdf import compile_standalone_table

    body = build_core_tail_pre_post_latex(human_df, genai_df)
    outdir.mkdir(parents=True, exist_ok=True)

    if human_sens is not None and genai_sens is not None:
        sens_pdf = plot_core_tail_hdbscan_sensitivity(
            human_sens,
            genai_sens,
            outdir / CORE_TAIL_SENSITIVITY_FIG_STEM,
        )
        compile_body = build_core_tail_combined_tex(
            _core_tail_table_body_only(body),
            sensitivity_pdf_name=sens_pdf.name,
        )
        extra_packages = [
            r"\usepackage{multirow}",
            r"\usepackage{siunitx}",
            r"\usepackage{graphicx}",
        ]
    else:
        sens_pdf = None
        compile_body = _core_tail_table_body_only(body)
        extra_packages = [
            r"\usepackage{multirow}",
            r"\usepackage{siunitx}",
        ]

    compile_standalone_table(
        outdir,
        CORE_TAIL_LATEX_STEM,
        compile_body,
        output_format="svg+png",
        png_dpi=400,
        crop="standalone",
        delete_intermediate_tex=True,
        extra_packages=extra_packages,
    )
    final_svg = outdir / f"{CORE_TAIL_LATEX_STEM}.svg"
    final_png = outdir / f"{CORE_TAIL_LATEX_STEM}.png"
    standalone_svg = outdir / f"{CORE_TAIL_LATEX_STEM}_standalone.svg"
    standalone_png = outdir / f"{CORE_TAIL_LATEX_STEM}_standalone.png"
    standalone_pdf = outdir / f"{CORE_TAIL_LATEX_STEM}_standalone.pdf"
    if standalone_svg.is_file():
        standalone_svg.replace(final_svg)
    if standalone_png.is_file():
        standalone_png.replace(final_png)
    standalone_pdf.unlink(missing_ok=True)
    for stale in outdir.glob(f"{CORE_TAIL_LATEX_STEM}_preview*"):
        stale.unlink(missing_ok=True)
    (outdir / f"{CORE_TAIL_LATEX_STEM}.tex").unlink(missing_ok=True)
    (outdir / f"{CORE_TAIL_LATEX_STEM}_standalone.tex").unlink(missing_ok=True)
    (outdir / f"{CORE_TAIL_LATEX_STEM}.pdf").unlink(missing_ok=True)
    # Panel b is embedded in the combined figure; drop the standalone sensitivity chart.
    if sens_pdf is not None:
        for suffix in (".pdf", ".svg", ".png"):
            (outdir / f"{CORE_TAIL_SENSITIVITY_FIG_STEM}{suffix}").unlink(missing_ok=True)
    return final_svg if final_svg.is_file() else final_png


def run_prediction1(
    embeddings_root: Path,
    outdir: Path,
    embedding_col: str,
    audience_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del outdir  # figure written by run_all_prediction1; no per-audience CSV exports
    count_label = audience_cfg["count_label"]
    rows = []
    sensitivity_rows: list[dict] = []
    for pre_dir, post_dir, task_key in discover_task_pairs(embeddings_root):
        print(f"\n=== {task_label_from_key(task_key)} ({count_label}) ===")
        row = analyze_prediction1_task(
            pre_dir,
            post_dir,
            task_key,
            embedding_col,
            embeddings_root,
            audience_group=audience_cfg["audience_group"],
            count_label=count_label,
        )
        for phase, key in (("pre", "_pre_sensitivity_rows"), ("post", "_post_sensitivity_rows")):
            for sens in row.pop(key, []):
                sensitivity_rows.append({
                    "audience_group": audience_cfg["audience_group"],
                    "task_key": task_key,
                    "phase": phase,
                    **sens,
                })
        rows.append(row)
        print(
            f"  Core%: pre {row['pre_core_pct']:.1f}% "
            f"[{row['pre_core_ci_lo']:.1f}, {row['pre_core_ci_hi']:.1f}] "
            f"→ post {row['post_core_pct']:.1f}% "
            f"[{row['post_core_ci_lo']:.1f}, {row['post_core_ci_hi']:.1f}]"
        )
        print(
            f"  Sensitivity range: pre "
            f"[{row['pre_core_sens_min']:.1f}, {row['pre_core_sens_max']:.1f}] | "
            f"post [{row['post_core_sens_min']:.1f}, {row['post_core_sens_max']:.1f}]"
        )
        print(
            f"  Tail%: pre {row['pre_tail_pct']:.1f}% "
            f"→ post {row['post_tail_pct']:.1f}% "
            f"(Δ {row['tail_pct_delta_post_minus_pre']:+.1f} pp)"
        )
        print(
            f"  Post still core+tail: {row['post_still_has_core_and_tail']} | "
            f"Post tail shorter: {row['post_tail_shorter_than_pre']} | "
            f"Δ core {row['core_pct_delta_post_minus_pre']:+.1f} pp"
        )

    summary_df = pd.DataFrame(rows)
    sens_df = pd.DataFrame(sensitivity_rows)
    return summary_df, sens_df


def participant_self_pre_post_distance_table(
    pre_dir: Path,
    post_dir: Path,
    embedding_col: str,
    *,
    collapsed: bool,
) -> pd.DataFrame:
    """Per-respondent cosine distance between aligned pre-ML and post-ML embeddings."""
    pre_df = read_anonymized_embeddings_wide(pre_dir / "embeddings_wide.parquet")
    post_df = read_anonymized_embeddings_wide(post_dir / "embeddings_wide.parquet")
    if collapsed:
        pre_df = with_collapsed_group(pre_df)
        post_df = with_collapsed_group(post_df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    else:
        group_col = PARTICIPANT_TYPE_COL

    pre_sub = pre_df[
        [PARTICIPANT_NAME_COL, group_col, embedding_col]
    ].rename(columns={embedding_col: f"{embedding_col}_pre"})
    post_sub = post_df[[PARTICIPANT_NAME_COL, embedding_col]].rename(
        columns={embedding_col: f"{embedding_col}_post"}
    )
    merged = pre_sub.merge(post_sub, on=PARTICIPANT_NAME_COL, how="inner")
    if merged.empty:
        raise ValueError(f"No paired respondents under {pre_dir.parent}")

    X_pre = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_pre": embedding_col}),
            embedding_col,
        )
    )
    X_post = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_post": embedding_col}),
            embedding_col,
        )
    )
    merged["self_pre_post_cosine_distance"] = np.diag(
        cosine_distances(X_pre, X_post)
    )
    merged = merged.rename(columns={group_col: "participant_group"})
    return merged[
        [
            PARTICIPANT_NAME_COL,
            "participant_group",
            "self_pre_post_cosine_distance",
        ]
    ]


def analyze_self_pre_post_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
    *,
    collapsed: bool,
    groups: list[str],
    group_labels: dict[str, str],
) -> tuple[list[dict], pd.DataFrame]:
    table = participant_self_pre_post_distance_table(
        pre_dir,
        post_dir,
        embedding_col,
        collapsed=collapsed,
    )
    rows: list[dict] = []
    for group in groups:
        vals = table.loc[
            table["participant_group"] == group, "self_pre_post_cosine_distance"
        ].to_numpy(dtype=float)
        if len(vals) == 0:
            raise ValueError(
                f"No paired respondents for group {group!r} in task {task_key}"
            )
        mean_val = float(np.mean(vals))
        ci_lo, ci_hi = bootstrap_mean_ci(vals, seed=ANALYSIS_SEED)
        rows.append(
            {
                "task_key": task_key,
                "task_label": task_label_from_key(task_key),
                "embedding_col": embedding_col,
                "collapsed": collapsed,
                "participant_group": group,
                "group_label": group_labels[group],
                "n_paired": len(vals),
                "mean_self_pre_post_cosine_distance": mean_val,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }
        )
    out_table = table.copy()
    out_table["task_key"] = task_key
    out_table["task_label"] = task_label_from_key(task_key)
    out_table["embedding_col"] = embedding_col
    out_table["collapsed"] = collapsed
    return rows, out_table


def paired_task_embedding_bundle(
    pre_dir: Path,
    post_dir: Path,
    embedding_col: str,
    *,
    collapsed: bool,
) -> dict:
    """Aligned pre/post embedding matrices and group labels for one task."""
    pre_df = read_anonymized_embeddings_wide(pre_dir / "embeddings_wide.parquet")
    post_df = read_anonymized_embeddings_wide(post_dir / "embeddings_wide.parquet")
    if collapsed:
        pre_df = with_collapsed_group(pre_df)
        post_df = with_collapsed_group(post_df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    else:
        group_col = PARTICIPANT_TYPE_COL

    pre_sub = pre_df[
        [PARTICIPANT_NAME_COL, group_col, embedding_col]
    ].rename(columns={embedding_col: f"{embedding_col}_pre"})
    post_sub = post_df[[PARTICIPANT_NAME_COL, embedding_col]].rename(
        columns={embedding_col: f"{embedding_col}_post"}
    )
    merged = pre_sub.merge(post_sub, on=PARTICIPANT_NAME_COL, how="inner")
    if merged.empty:
        raise ValueError(f"No paired respondents under {pre_dir.parent}")

    X_pre = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_pre": embedding_col}),
            embedding_col,
        )
    )
    X_post = normalize(
        stack_embeddings(
            merged.rename(columns={f"{embedding_col}_post": embedding_col}),
            embedding_col,
        )
    )
    return {
        PARTICIPANT_NAME_COL: merged[PARTICIPANT_NAME_COL].to_numpy(),
        "participant_group": merged[group_col].to_numpy(),
        "X_pre": X_pre,
        "X_post": X_post,
    }


def filter_embedding_bundle(bundle: dict, group: str) -> dict:
    mask = bundle["participant_group"] == group
    if not np.any(mask):
        raise ValueError(f"No paired respondents for group {group!r}")
    return {
        PARTICIPANT_NAME_COL: bundle[PARTICIPANT_NAME_COL][mask],
        "participant_group": bundle["participant_group"][mask],
        "X_pre": bundle["X_pre"][mask],
        "X_post": bundle["X_post"][mask],
    }


def semantic_map_pre_panel(
    pre_dir: Path,
    embedding_col: str,
    *,
    seed: int = ANALYSIS_SEED,
) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    """All pre-ML rows with 2D PCA coords (pre-only fit; legacy helper)."""
    df = read_anonymized_embeddings_wide(pre_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return df, coords, pca


def fit_pooled_pre_post_semantic_map_pca(
    pre_dir: Path,
    post_dir: Path,
    embedding_col: str,
    *,
    seed: int = ANALYSIS_SEED,
) -> tuple[PCA, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Fit one PCA on pooled Pre+Post embeddings (same basis as semantic_space_map).

    Returns ``(pca, df_pre, coords_pre, df_post, coords_post)`` for the full
    phase samples (not listwise-paired only).
    """
    df_pre = read_anonymized_embeddings_wide(pre_dir / "embeddings_wide.parquet")
    df_post = read_anonymized_embeddings_wide(post_dir / "embeddings_wide.parquet")
    X_pre = normalize(stack_embeddings(df_pre, embedding_col))
    X_post = normalize(stack_embeddings(df_post, embedding_col))
    pca = PCA(n_components=2, random_state=seed).fit(np.vstack([X_pre, X_post]))
    return (
        pca,
        df_pre,
        pca.transform(X_pre),
        df_post,
        pca.transform(X_post),
    )


def fit_pre_ml_semantic_map_pca(
    pre_dir: Path,
    embedding_col: str,
    *,
    seed: int = ANALYSIS_SEED,
) -> PCA:
    """Fit PCA on all pre-ML respondents in one task (legacy pre-only basis)."""
    _, _, pca = semantic_map_pre_panel(pre_dir, embedding_col, seed=seed)
    return pca


def project_paired_embeddings_semantic_pca(
    pca: PCA,
    X_pre: np.ndarray,
    X_post: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project pre/post rows into a pre-fit semantic-map PCA."""
    return pca.transform(X_pre), pca.transform(X_post)


def draw_self_pre_post_pca_panel(
    ax,
    bundle: dict,
    coords_pre: np.ndarray,
    coords_post: np.ndarray,
    *,
    groups: list[str],
    group_colors: dict[str, str],
    panel_title: str,
    axis_bounds: dict[str, float],
) -> None:
    participant_groups = bundle["participant_group"]
    for i, group in enumerate(participant_groups):
        color = group_colors.get(group, "#888888")
        ax.plot(
            [coords_pre[i, 0], coords_post[i, 0]],
            [coords_pre[i, 1], coords_post[i, 1]],
            color=color,
            alpha=SELF_PRE_POST_PCA_LINE_ALPHA,
            linewidth=SELF_PRE_POST_PCA_LINEWIDTH,
            zorder=1,
        )

    for group in groups:
        mask = participant_groups == group
        if not np.any(mask):
            continue
        color = group_colors[group]
        ax.scatter(
            coords_pre[mask, 0],
            coords_pre[mask, 1],
            facecolors="none",
            edgecolors=color,
            s=SELF_PRE_POST_PCA_PRE_SIZE,
            linewidths=SELF_PRE_POST_PCA_PRE_EDGEWIDTH,
            zorder=2,
        )
        ax.scatter(
            coords_post[mask, 0],
            coords_post[mask, 1],
            c=color,
            s=SELF_PRE_POST_PCA_POST_SIZE,
            edgecolors=BAR_EDGE_COLOR,
            linewidths=SELF_PRE_POST_PCA_POST_EDGEWIDTH,
            zorder=3,
        )

    apply_semantic_map_2d_bounds(
        ax,
        axis_bounds,
        box_aspect=PHASE_GRID_SEMANTIC_MAP_BOX_ASPECT,
    )
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=PHASE_GRID_TICK_FONTSIZE)
    ax.grid(alpha=0.2, zorder=0)


def self_pre_post_pca_legend_handles(
    audiences: tuple[dict, ...] = SELF_PRE_POST_PCA_AUDIENCES,
) -> tuple[list, list[str]]:
    handles: list = []
    labels: list[str] = []
    for audience in audiences:
        handles.append(
            Patch(
                facecolor=audience["group_color"],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label=audience["group_label"],
            )
        )
        labels.append(audience["group_label"])
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#555555",
                markeredgewidth=SELF_PRE_POST_PCA_PRE_EDGEWIDTH,
                linestyle="None",
                markersize=7,
                label="pre-ML",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="#555555",
                markeredgecolor=BAR_EDGE_COLOR,
                markeredgewidth=SELF_PRE_POST_PCA_POST_EDGEWIDTH,
                linestyle="None",
                markersize=7,
                label="post-ML",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                alpha=SELF_PRE_POST_PCA_LINE_ALPHA,
                linewidth=1.4,
                label="Respondent trajectory",
            ),
        ]
    )
    labels.extend(["pre-ML", "post-ML", "Respondent trajectory"])
    return handles, labels


def _trajectory_task_panel_title(task_key: str) -> str:
    return task_label_from_key(task_key).replace(
        "Second-Order Interactions", "Interactions"
    )


def collect_self_pre_post_pca_task_bundles(
    embeddings_root: Path,
    embedding_col: str,
    *,
    collapsed: bool,
    filter_group: str,
) -> list[tuple[str, dict, np.ndarray, np.ndarray]]:
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    task_bundles: list[tuple[str, dict, np.ndarray, np.ndarray]] = []
    for pre_dir, post_dir, task_key in discover_task_pairs(embeddings_root):
        bundle = paired_task_embedding_bundle(
            pre_dir,
            post_dir,
            embedding_col,
            collapsed=collapsed,
        )
        bundle = filter_embedding_bundle(bundle, filter_group)
        pca, _, _, _, _ = fit_pooled_pre_post_semantic_map_pca(
            pre_dir, post_dir, embedding_col
        )
        coords_pre, coords_post = project_paired_embeddings_semantic_pca(
            pca,
            bundle["X_pre"],
            bundle["X_post"],
        )
        task_bundles.append((task_key, bundle, coords_pre, coords_post))
    task_bundles.sort(key=lambda item: order.get(item[0], 999))
    return task_bundles


def plot_self_pre_post_pca_trajectory_human_genai(
    audience_bundles: list[
        tuple[dict, list[tuple[str, dict, np.ndarray, np.ndarray]]]
    ],
    outpath: Path,
) -> None:
    """One figure, two audience panels (Humans | GenAI), each a 2×2 task grid."""
    if len(audience_bundles) != 2:
        raise ValueError("Expected exactly two audiences (Human, GenAI).")
    axis_bounds = phase_grid_semantic_map_axis_bounds()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18.8, 9.4),
        gridspec_kw={"hspace": 0.36, "wspace": 0.16},
    )

    for audience_i, (audience, task_bundles) in enumerate(audience_bundles):
        col0 = audience_i * 2
        filter_group = audience["filter_group"]
        groups = [filter_group]
        group_colors = {filter_group: audience["group_color"]}
        for task_i, (task_key, bundle, coords_pre, coords_post) in enumerate(
            task_bundles
        ):
            r, c = divmod(task_i, 2)
            ax = axes[r, col0 + c]
            draw_self_pre_post_pca_panel(
                ax,
                bundle,
                coords_pre,
                coords_post,
                groups=groups,
                group_colors=group_colors,
                panel_title=_trajectory_task_panel_title(task_key),
                axis_bounds=axis_bounds,
            )

    handles, labels = self_pre_post_pca_legend_handles()
    header = layout_title_and_metric(
        fig,
        suptitle=SELF_PRE_POST_PCA_COMBINED_SUPTITLE,
        metric_lines=SELF_PRE_POST_PCA_COMBINED_METRIC,
        suptitle_fontsize=VIZ_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
        gap_title_metric=0.008,
        gap_metric_legend=0.018,
    )
    add_phase_grid_figure_legend(
        fig,
        handles,
        labels,
        ncol=SELF_PRE_POST_PCA_LEGEND_NCOL,
        bbox_y=header.legend_y,
    )
    fig.supxlabel(
        "PCA dimension 1",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=PHASE_GRID_SEMANTIC_MAP_SUPXLABEL_Y,
    )
    fig.supylabel(
        "PCA dimension 2",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    phase_grid_layout_adjust(
        fig,
        top=header.panel_top - 0.035,
        bottom_extra=PHASE_GRID_SEMANTIC_MAP_BOTTOM_EXTRA,
    )
    fig.canvas.draw()
    for audience_i, (audience, _) in enumerate(audience_bundles):
        left_ax = axes[0, audience_i * 2]
        right_ax = axes[0, audience_i * 2 + 1]
        x0 = left_ax.get_position().x0
        x1 = right_ax.get_position().x1
        fig.text(
            (x0 + x1) / 2.0,
            header.panel_top - 0.005,
            audience["panel_title"],
            ha="center",
            va="top",
            fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE + 2,
            fontweight="bold",
            transform=fig.transFigure,
        )
    save_figure_pdf_svg(fig, outpath)


def run_all_self_pre_post_pca_trajectory(
    embeddings_root: Path,
    embedding_col: str,
) -> None:
    """Remove deprecated PCA trajectory figures (no longer generated)."""
    del embedding_col  # unused; kept for call-site compatibility
    outdir = comparisons_pre_post_dir(embeddings_root, COMPARISONS_SELF_SUBDIR)
    for stem in SELF_PRE_POST_PCA_REMOVED_FIGS:
        for suffix in (".png", ".pdf", ".svg"):
            stale = outdir / f"{stem}{suffix}"
            if stale.exists():
                stale.unlink()
                print(f"Removed: {stale.name}")


def self_pre_post_welch_comparisons(
    task_df: pd.DataFrame,
    groups: list[str],
) -> list[tuple[str, float]]:
    comparisons: list[tuple[str, float]] = []
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if set(groups) <= set(GROUP_ORDER_COLLAPSED)
        else PARTICIPANT_TYPE_COL
    )
    for left, right, label in comparison_pairs_for_groups(groups, group_col=group_col):
        vals_left = task_df.loc[
            task_df["participant_group"] == left, "self_pre_post_cosine_distance"
        ].to_numpy(dtype=float)
        vals_right = task_df.loc[
            task_df["participant_group"] == right, "self_pre_post_cosine_distance"
        ].to_numpy(dtype=float)
        comparisons.append((label, p_value_welch_ttest(vals_left, vals_right)))
    return comparisons


def self_pre_post_figure_ylim_top(summary_df: pd.DataFrame) -> float:
    panel_max = 0.0
    for _, row in summary_df.iterrows():
        mean_val = float(row["mean_self_pre_post_cosine_distance"])
        _, err_hi = ci_errorbar_offsets(
            mean_val, float(row["ci_low"]), float(row["ci_high"])
        )
        panel_max = max(panel_max, mean_val + err_hi)
    if panel_max <= 0:
        return 0.5
    step = 0.05 if panel_max < 0.5 else 0.1
    return float(np.ceil(panel_max * SELF_PRE_POST_YLIM_PAD / step) * step)


def draw_self_pre_post_welch_footnote(
    fig,
    footnote: tuple[str, ...],
    *,
    y: float = SELF_PRE_POST_FOOTNOTE_Y,
) -> None:
    for i, line in enumerate(footnote):
        fig.text(
            0.5,
            y - i * VIZ_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=VIZ_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def panel_bar_top(
    task_summary: pd.DataFrame,
    groups: list[str],
) -> float:
    top = 0.0
    for group in groups:
        row = task_summary.loc[task_summary["participant_group"] == group].iloc[0]
        mean_val = float(row["mean_self_pre_post_cosine_distance"])
        _, err_hi = ci_errorbar_offsets(
            mean_val, float(row["ci_low"]), float(row["ci_high"])
        )
        top = max(top, mean_val + err_hi)
    return top


def _draw_data_comparison_patch(
    ax,
    text_objs: list,
    has_sig: bool,
) -> None:
    if not text_objs:
        return
    renderer = ax.figure.canvas.get_renderer()
    bb = Bbox.union([t.get_window_extent(renderer=renderer) for t in text_objs])
    bb = Bbox.from_extents(
        bb.x0 - COMPARE_PAD_PX,
        bb.y0 - COMPARE_PAD_PX,
        bb.x1 + COMPARE_PAD_PX,
        bb.y1 + COMPARE_PAD_PX,
    )
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=ax.transData,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            linewidth=1.2 if has_sig else 0.9,
            clip_on=False,
            zorder=3,
        )
    )


def self_pre_post_comparison_box_top(
    panel_top: float,
    n_lines: int,
    *,
    ylim_top: float,
) -> float:
    line_step = max(0.012, SELF_PRE_POST_COMPARE_ABOVE_LINE_FRAC * ylim_top)
    pad = max(0.012, SELF_PRE_POST_COMPARE_ABOVE_PAD_FRAC * ylim_top)
    if n_lines == 0:
        return panel_top
    y_bottom = panel_top + pad
    return y_bottom + (n_lines - 1) * line_step + line_step * 1.15


def draw_self_pre_post_comparison_box_above_bars(
    ax,
    x_idx: float,
    welch_lines: list[tuple[str, float]],
    panel_top: float,
    *,
    ylim_top: float,
) -> float:
    line_step = max(0.012, SELF_PRE_POST_COMPARE_ABOVE_LINE_FRAC * ylim_top)
    pad = max(0.012, SELF_PRE_POST_COMPARE_ABOVE_PAD_FRAC * ylim_top)
    y_bottom = panel_top + pad
    text_objs: list = []
    has_sig = False
    for i, (label, pval) in enumerate(welch_lines):
        sig = is_significant(pval)
        has_sig = has_sig or sig
        text_objs.append(
            ax.text(
                x_idx,
                y_bottom + i * line_step,
                format_comparison_line(label, pval),
                transform=ax.transData,
                ha="center",
                va="bottom",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )
    ax.figure.canvas.draw()
    _draw_data_comparison_patch(ax, text_objs, has_sig)


def plot_self_pre_post_embedding_distance(
    summary_df: pd.DataFrame,
    participant_df: pd.DataFrame,
    outpath: Path,
    *,
    groups: list[str],
    group_labels: dict[str, str],
    group_colors: dict[str, str],
    suptitle: str,
    collapsed: bool,
    footnote: tuple[str, ...],
) -> None:
    plot_df = participant_df.loc[participant_df["collapsed"] == collapsed].copy()
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    task_keys = sorted(
        summary_df["task_key"].unique(),
        key=lambda key: order.get(key, 999),
    )
    ylim_top = self_pre_post_figure_ylim_top(summary_df)
    group_x = {group: float(i) for i, group in enumerate(groups)}
    compare_center_x = (len(groups) - 1) / 2.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DIVERSITY_PRED_FIGSIZE,
        gridspec_kw={
            "hspace": DIVERSITY_PRED_ROW_GAP,
            "wspace": DIVERSITY_PRED_COL_GAP,
        },
    )
    axes_flat = axes.ravel()

    for ax, task_key in zip(axes_flat, task_keys):
        task_summary = summary_df.loc[summary_df["task_key"] == task_key]
        task_participants = plot_df.loc[plot_df["task_key"] == task_key]
        for group in groups:
            row = task_summary.loc[task_summary["participant_group"] == group].iloc[0]
            mean_val = float(row["mean_self_pre_post_cosine_distance"])
            ci_lo = float(row["ci_low"])
            ci_hi = float(row["ci_high"])
            err_lo, err_hi = ci_errorbar_offsets(mean_val, ci_lo, ci_hi)
            x = group_x[group]
            ax.bar(
                x,
                mean_val,
                SELF_PRE_POST_BAR_WIDTH,
                color=group_colors[group],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label=group_labels[group],
                zorder=2,
            )
            ax.errorbar(
                x,
                mean_val,
                yerr=[[err_lo], [err_hi]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )

        welch_lines = self_pre_post_welch_comparisons(task_participants, groups)
        bar_top = panel_bar_top(task_summary, groups)
        panel_ylim = max(
            ylim_top,
            self_pre_post_comparison_box_top(
                bar_top,
                len(welch_lines),
                ylim_top=ylim_top,
            ),
        )

        ax.set_xticks([group_x[group] for group in groups])
        ax.set_xticklabels(
            [group_labels[group] for group in groups],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
        ax.set_title(
            task_label_from_key(task_key),
            fontweight="bold",
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            pad=10,
        )
        ax.tick_params(axis="y", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0.0, panel_ylim)
        ytick_step = 0.05 if panel_ylim <= 0.5 else 0.1
        ax.set_yticks(np.arange(0.0, panel_ylim + 0.001, ytick_step))
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

        draw_self_pre_post_comparison_box_above_bars(
            ax,
            compare_center_x,
            welch_lines,
            bar_top,
            ylim_top=ylim_top,
        )

    fig.supylabel(
        "Mean self pre–post cosine distance",
        fontweight="bold",
        x=DIVERSITY_PRED_YLABEL_X,
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    footnote_lines = len(footnote)
    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=SELF_PRE_POST_METRIC_SUBTITLE,
        suptitle_fontsize=VIZ_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=header.panel_top,
        bottom=0.08 + footnote_lines * VIZ_FOOTNOTE_LINE_STEP,
        hspace=DIVERSITY_PRED_ROW_GAP,
    )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_y_shift = (
        SELF_PRE_POST_COLLAPSED_LEGEND_Y_SHIFT if collapsed else 0.0
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(groups),
        frameon=True,
        fontsize=VIZ_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, header.legend_y + legend_y_shift),
        borderaxespad=0.0,
    )
    draw_self_pre_post_welch_footnote(fig, footnote)
    save_figure_pdf_svg(fig, outpath)


def collect_self_pre_post_welch_rows(
    participant_df: pd.DataFrame,
    *,
    groups: list[str],
    collapsed: bool,
    embedding_col: str,
) -> list[dict]:
    plot_df = participant_df.loc[participant_df["collapsed"] == collapsed].copy()
    rows: list[dict] = []
    for task_key in sorted(plot_df["task_key"].unique()):
        task_df = plot_df.loc[plot_df["task_key"] == task_key]
        for label, pval in self_pre_post_welch_comparisons(task_df, groups):
            rows.append(
                {
                    "task_key": task_key,
                    "task_label": task_label_from_key(task_key),
                    "embedding_col": embedding_col,
                    "collapsed": collapsed,
                    "comparison": label,
                    "welch_pvalue": pval,
                    "significance": significance_label(pval),
                }
            )
    return rows


def _forecast_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def _canon_forecast_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip(), b.strip())))


def _parse_forecast_pair(cell: str, valid_features: set[str]) -> tuple[str, str] | None:
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2 or parts[0] == parts[1]:
        return None
    if parts[0] not in valid_features or parts[1] not in valid_features:
        return None
    return _canon_forecast_pair(parts[0], parts[1])


def load_participant_pre_ml_forecasting_accuracy() -> pd.DataFrame:
    """Per-participant pre-ML forecasting accuracy for all four task panels."""
    with open(SURVEY_CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers, data = rows[0], rows[1:]

    name_col = next(
        i for i, h in enumerate(headers) if h.strip() == "Participant ID"
    )
    group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)

    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers
        if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feat_idx = {f: i for i, f in enumerate(features)}
    feature_set = set(features)
    pairs = list(combinations(sorted(features), 2))
    pair_idx = {p: i for i, p in enumerate(pairs)}

    with open(ME_ML_PATH) as f:
        ml_me = json.load(f)
    me_signs = {
        task: {
            e["feature"]: e["sign"]
            for e in sorted(entries, key=lambda x: x["rank"])
        }
        for task, entries in ml_me.items()
    }
    ml_race_me = np.zeros(len(feat_idx))
    ml_gender_me = np.zeros(len(feat_idx))
    for feat, sign_str in me_signs["race"].items():
        if feat in feat_idx:
            ml_race_me[feat_idx[feat]] = FORECAST_SIGN_MAP.get(sign_str, 0)
    for feat, sign_str in me_signs["gender"].items():
        if feat in feat_idx:
            ml_gender_me[feat_idx[feat]] = FORECAST_SIGN_MAP.get(sign_str, 0)

    with open(SOI_ML_PATH) as f:
        ml_soi = json.load(f)

    def _build_ml_soi(entries: list[dict]) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for e in entries:
            p = _canon_forecast_pair(e["feature_1"], e["feature_2"])
            if p in pair_idx:
                vec[pair_idx[p]] = FORECAST_SIGN_MAP.get(e["sign"], 0)
        return vec

    ml_race_soi = _build_ml_soi(ml_soi["race"])
    ml_gender_soi = _build_ml_soi(ml_soi["gender"])

    r1_col = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
    g1_col = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")
    r3_cols = {
        re.sub(r"^Q Race\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers)
        if re.match(r"^Q Race\.3 \(sign\) - ", h)
    }
    g3_cols = {
        re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers)
        if re.match(r"^Q Gender\.3 \(sign\) - ", h)
    }
    r_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
    ]
    r_sign_cols = [
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Race.9 (SOI, sign, 1st)"
        ),
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Race.9 (SOI, sign, 2nd)"
        ),
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Race.9 (SOI, sign, 3rd)"
        ),
    ]
    g_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
    ]
    g_sign_cols = [
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Gender.9 (SOI, sign, 1st)"
        ),
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Gender.9 (SOI, sign, 2nd)"
        ),
        next(
            i for i, h in enumerate(headers)
            if h.strip() == "Q Gender.9 (SOI, sign, 3rd)"
        ),
    ]

    def _build_me_vec(q1_col: int, q3_col_map: dict[str, int], row: list[str]):
        vec = np.zeros(len(feat_idx))
        cell = row[q1_col].strip()
        if not cell:
            return None
        for feat in cell.split(","):
            feat = feat.strip()
            if feat not in feat_idx:
                continue
            sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            vec[feat_idx[feat]] = FORECAST_SIGN_MAP.get(sign_str, 0)
        return vec

    def _build_soi_vec(
        pair_cols: list[int],
        sign_cols: list[int],
        row: list[str],
    ) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for pc, sc in zip(pair_cols, sign_cols):
            p = _parse_forecast_pair(row[pc], feature_set)
            if p is None:
                continue
            vec[pair_idx[p]] = FORECAST_SIGN_MAP.get(row[sc].strip(), 0)
        return vec

    records: list[dict] = []
    for row in data:
        gid = row[group_col].strip()
        if gid not in HUMAN_GROUP_IDS | {GENAI_GROUP_ID}:
            continue
        group = "GenAI" if gid == GENAI_GROUP_ID else "Human"
        name = row[name_col].strip()
        vr = _build_me_vec(r1_col, r3_cols, row)
        vg = _build_me_vec(g1_col, g3_cols, row)
        hr = _build_soi_vec(r_pair_cols, r_sign_cols, row)
        hg = _build_soi_vec(g_pair_cols, g_sign_cols, row)
        records.append(
            {
                PARTICIPANT_NAME_COL: name,
                "participant_group": group,
                "pre_ml_accuracy_race_me": (
                    _forecast_cosine_sim(vr, ml_race_me) if vr is not None else np.nan
                ),
                "pre_ml_accuracy_gender_me": (
                    _forecast_cosine_sim(vg, ml_gender_me) if vg is not None else np.nan
                ),
                "pre_ml_accuracy_race_soi": _forecast_cosine_sim(hr, ml_race_soi),
                "pre_ml_accuracy_gender_soi": _forecast_cosine_sim(hg, ml_gender_soi),
            }
        )
    return pd.DataFrame(records)


def _task_pre_accuracy_column(task_key: str) -> str:
    topic, task_type = task_key.split("/")
    return f"pre_ml_accuracy_{topic}_{'me' if task_type == 'main-effects' else 'soi'}"


def merge_shift_with_pre_ml_accuracy(participant_df: pd.DataFrame) -> pd.DataFrame:
    """Attach task-matched pre-ML forecasting accuracy to shift table."""
    acc = load_participant_pre_ml_forecasting_accuracy()
    accuracy_cols = [
        "pre_ml_accuracy_race_me",
        "pre_ml_accuracy_gender_me",
        "pre_ml_accuracy_race_soi",
        "pre_ml_accuracy_gender_soi",
    ]
    merged = participant_df.copy()
    merged["_merge_name"] = merged[PARTICIPANT_NAME_COL].map(
        clean_participant_display_name
    )
    acc_merge = acc[[PARTICIPANT_NAME_COL, *accuracy_cols]].copy()
    acc_merge["_merge_name"] = acc_merge[PARTICIPANT_NAME_COL].map(
        clean_participant_display_name
    )
    acc_merge = acc_merge.drop(columns=[PARTICIPANT_NAME_COL])
    merged = merged.merge(acc_merge, on="_merge_name", how="left")
    merged = merged.drop(columns=["_merge_name"])
    merged["pre_ml_accuracy"] = merged.apply(
        lambda row: row[_task_pre_accuracy_column(row["task_key"])],
        axis=1,
    )
    return merged


def _format_accuracy_tick(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _accuracy_group_xlabel(group_key: str, accuracy_values: list[float]) -> str:
    return SHIFT_ACCURACY_BINARY_LABELS[group_key]


def _shift_accuracy_group_defs(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, list[dict]] | None:
    """Assign rows to high/low groups by splitting unique accuracy levels in half.

    Not rank-based: respondents are grouped by their pre-ML accuracy value.
    Unique values are sorted; the lower half of distinct levels → Low, upper → High.
    """
    if len(x) < 4:
        return None
    frame = pd.DataFrame({"accuracy": x, "shift": y})
    unique_vals = np.sort(frame["accuracy"].unique())
    if len(unique_vals) < 2:
        return None

    group_defs: list[dict] = []
    if len(unique_vals) == 2:
        tier_values = {
            "low": [float(unique_vals[0])],
            "high": [float(unique_vals[1])],
        }
    else:
        tier_values = {key: [] for key in SHIFT_ACCURACY_BINARY_ORDER}
        n_levels = len(unique_vals)
        for idx, value in enumerate(unique_vals):
            tier_idx = min(int(idx * 2 / n_levels), 1)
            tier_key = SHIFT_ACCURACY_BINARY_ORDER[tier_idx]
            tier_values[tier_key].append(float(value))

    value_to_key: dict[float, str] = {}
    for tier_key, vals in tier_values.items():
        for value in vals:
            value_to_key[value] = tier_key

    for tier_idx, tier_key in enumerate(SHIFT_ACCURACY_BINARY_ORDER):
        vals = tier_values[tier_key]
        if not vals:
            continue
        group_defs.append(
            {
                "group_key": tier_key,
                "group_label": _accuracy_group_xlabel(tier_key, vals),
                "group_order": tier_idx,
                "accuracy_values": vals,
            }
        )

    frame["group_key"] = frame["accuracy"].map(value_to_key)
    if frame["group_key"].isna().any():
        return None
    meta = {row["group_key"]: row for row in group_defs}
    frame["group_label"] = frame["group_key"].map(
        lambda key: meta[str(key)]["group_label"]
    )
    frame["group_order"] = frame["group_key"].map(
        lambda key: meta[str(key)]["group_order"]
    )
    group_defs = sorted(group_defs, key=lambda row: row["group_order"])
    return frame, group_defs


def _shift_accuracy_group_panel_stats(
    x: np.ndarray,
    y: np.ndarray,
) -> dict | None:
    grouped = _shift_accuracy_group_defs(x, y)
    if grouped is None:
        return None
    frame, group_defs = grouped
    shift_groups = [
        frame.loc[frame["group_key"] == row["group_key"], "shift"].to_numpy(dtype=float)
        for row in group_defs
    ]
    if len(shift_groups) < 2 or any(len(group) < 1 for group in shift_groups):
        return None

    h_stat, pval = kruskal(*shift_groups)
    welch_pval = p_value_welch_ttest(shift_groups[0], shift_groups[1])
    groups: list[dict] = []
    for row, shifts in zip(group_defs, shift_groups):
        mean_shift = float(np.mean(shifts))
        ci_lo, ci_hi = bootstrap_mean_ci(shifts)
        groups.append(
            {
                "group_key": row["group_key"],
                "group_label": row["group_label"],
                "accuracy_values": row["accuracy_values"],
                "n": int(len(shifts)),
                "mean_shift": mean_shift,
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
            }
        )
    return {
        "groups": groups,
        "kruskal_h": float(h_stat),
        "kruskal_pvalue": float(pval),
        "kruskal_significance": significance_label(pval),
        "welch_pvalue": float(welch_pval) if np.isfinite(welch_pval) else np.nan,
        "welch_significance": significance_label(welch_pval),
        # Backward-compatible alias for CSV consumers.
        "significance": significance_label(pval),
    }


def _accuracy_group_bar_colors(
    group_color: str,
    n_groups: int,
) -> list[tuple[float, float, float, float]]:
    if n_groups <= 1:
        return [to_rgba(group_color, 1.0)]
    alphas = np.linspace(
        SHIFT_ACCURACY_BINARY_BAR_ALPHAS[0],
        SHIFT_ACCURACY_BINARY_BAR_ALPHAS[-1],
        n_groups,
    )
    return [to_rgba(group_color, alpha=float(alpha)) for alpha in alphas]


def collect_shift_accuracy_kruskal_wallis(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal–Wallis and Welch on shift across pre-ML accuracy groups (within task/scope)."""
    rows: list[dict] = []
    scopes = [("Human", "Humans"), ("GenAI", "GenAI"), ("All", "All")]
    task_keys = list(TASK_PANEL_ORDER) + ["pooled"]

    for task_key in task_keys:
        if task_key == "pooled":
            task_df = merged_df.copy()
            task_label = "Pooled (all tasks)"
        else:
            task_df = merged_df.loc[merged_df["task_key"] == task_key].copy()
            task_label = task_label_from_key(task_key)

        for group_key, group_label in scopes:
            if group_key == "All":
                sub = task_df
            else:
                sub = task_df.loc[task_df["participant_group"] == group_key]
            x = sub["pre_ml_accuracy"].to_numpy(dtype=float)
            y = sub["self_pre_post_cosine_distance"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            n = int(len(x))

            panel_stats = _shift_accuracy_group_panel_stats(x, y)
            if panel_stats is None:
                h_stat, pval = np.nan, np.nan
                welch_pval = np.nan
                group_stats: list[dict[str, float | int | str]] = []
            else:
                h_stat = panel_stats["kruskal_h"]
                pval = panel_stats["kruskal_pvalue"]
                welch_pval = panel_stats["welch_pvalue"]
                group_stats = [
                    {
                        "label": row["group_key"],
                        "group_label": row["group_label"],
                        "accuracy_values": row["accuracy_values"],
                        "n": row["n"],
                        "mean_shift": row["mean_shift"],
                    }
                    for row in panel_stats["groups"]
                ]

            row: dict = {
                "task_key": task_key,
                "task_label": task_label,
                "scope": group_label,
                "n": n,
                "n_accuracy_groups": len(group_stats),
                "kruskal_h": float(h_stat) if np.isfinite(h_stat) else np.nan,
                "kruskal_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "kruskal_significance": significance_label(pval),
                "welch_pvalue": float(welch_pval) if np.isfinite(welch_pval) else np.nan,
                "welch_significance": significance_label(welch_pval),
                "significance": significance_label(pval),
            }
            for stat in group_stats:
                label = stat["label"]
                row[f"group_label_{label}"] = stat["group_label"]
                row[f"accuracy_values_{label}"] = ";".join(
                    _format_accuracy_tick(float(v))
                    for v in stat["accuracy_values"]
                )
                row[f"n_{label}"] = int(stat["n"])
                row[f"mean_shift_{label}"] = float(stat["mean_shift"])
            rows.append(row)
    return pd.DataFrame(rows)


def collect_shift_accuracy_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("Human", "Humans"), ("GenAI", "GenAI"), ("All", "All")]
    task_keys = list(TASK_PANEL_ORDER) + ["pooled"]

    for task_key in task_keys:
        if task_key == "pooled":
            task_df = merged_df.copy()
            task_label = "Pooled (all tasks)"
        else:
            task_df = merged_df.loc[merged_df["task_key"] == task_key].copy()
            task_label = task_label_from_key(task_key)

        for group_key, group_label in scopes:
            if group_key == "All":
                sub = task_df
            else:
                sub = task_df.loc[task_df["participant_group"] == group_key]
            x = sub["pre_ml_accuracy"].to_numpy(dtype=float)
            y = sub["self_pre_post_cosine_distance"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 4:
                rho, pval, r, r_pval, slope, ols_pval = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            else:
                rho, pval = spearmanr(x[mask], y[mask])
                r, r_pval = pearsonr(x[mask], y[mask])
                if np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    slope, _, _, ols_pval, _ = linregress(x[mask], y[mask])
                else:
                    slope = np.nan
                    ols_pval = np.nan
            rows.append(
                {
                    "task_key": task_key,
                    "task_label": task_label,
                    "scope": group_label,
                    "n": n,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                    "pearson_r": float(r) if np.isfinite(r) else np.nan,
                    "pearson_pvalue": float(r_pval) if np.isfinite(r_pval) else np.nan,
                    "ols_slope": float(slope) if np.isfinite(slope) else np.nan,
                    "ols_pvalue": float(ols_pval) if np.isfinite(ols_pval) else np.nan,
                    "significance": significance_label(pval),
                    "pearson_significance": significance_label(r_pval),
                    "ols_significance": significance_label(ols_pval),
                }
            )
    return pd.DataFrame(rows)


def prepare_revision_prior_error_frame(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Rows for MixedLM: revision, prior_error, group, task, contributor."""
    df = merged_df.copy()
    df["revision"] = df["self_pre_post_cosine_distance"].astype(float)
    df["prior_error"] = 1.0 - df["pre_ml_accuracy"].astype(float)
    df["contributor"] = df[PARTICIPANT_NAME_COL].astype(str)
    df["group"] = df["participant_group"].astype(str)
    df["task"] = df["task_key"].astype(str)
    keep = (
        np.isfinite(df["revision"].to_numpy(dtype=float))
        & np.isfinite(df["prior_error"].to_numpy(dtype=float))
        & df["group"].isin(["Human", "GenAI"])
        & df["contributor"].ne("")
        & df["task"].ne("")
    )
    return df.loc[keep].copy()


def fit_revision_prior_error_mixedlm(
    merged_df: pd.DataFrame,
):
    """
    Multilevel model for revision as a function of prior error:

        revision ~ prior_error * group + task FE + (1 | contributor)

    Human is the reference group. Returns (result, coef_table, slopes_table).
    """
    df = prepare_revision_prior_error_frame(merged_df)
    if df.empty or df["contributor"].nunique() < 3:
        raise ValueError("Too few rows/contributors for MixedLM.")

    formula = (
        "revision ~ prior_error * C(group, Treatment(reference='Human')) "
        "+ C(task)"
    )
    model = smf.mixedlm(formula, data=df, groups=df["contributor"])
    result = model.fit(method="lbfgs", reml=True, maxiter=200)

    coef_rows: list[dict] = []
    for name in result.fe_params.index:
        coef = float(result.fe_params[name])
        se = float(result.bse_fe[name])
        pval = float(result.pvalues[name])
        coef_rows.append({
            "term": str(name),
            "coef": coef,
            "se": se,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "z": float(result.tvalues[name]),
            "pvalue": pval,
            "significance": significance_label(pval),
        })
    coef_df = pd.DataFrame(coef_rows)

    # Group-specific prior_error slopes via linear contrasts on fixed effects.
    fe_names = list(result.fe_params.index)
    pe_idx = fe_names.index("prior_error")
    pe_g_name = (
        "prior_error:C(group, Treatment(reference='Human'))[T.GenAI]"
    )
    pe_g_idx = fe_names.index(pe_g_name)
    k_fe = int(result.k_fe)

    def _slope_contrast(indices: list[int]) -> tuple[float, float, float]:
        r_matrix = np.zeros((1, k_fe), dtype=float)
        for idx in indices:
            r_matrix[0, idx] = 1.0
        tt = result.t_test(r_matrix)
        est = float(np.asarray(tt.effect).reshape(-1)[0])
        se = float(np.asarray(tt.sd).reshape(-1)[0])
        pval = float(np.asarray(tt.pvalue).reshape(-1)[0])
        return est, se, pval

    slopes: list[dict] = []
    for group, indices in (
        ("Human", [pe_idx]),
        ("GenAI", [pe_idx, pe_g_idx]),
    ):
        est, se, pval = _slope_contrast(indices)
        slopes.append({
            "group": group,
            "scope": "Humans" if group == "Human" else "GenAI",
            "slope": est,
            "se": se,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "z": float(est / se) if se > 0 else np.nan,
            "pvalue": pval,
            "significance": significance_label(pval),
            "n_obs": int((df["group"] == group).sum()),
            "n_contributors": int(
                df.loc[df["group"] == group, "contributor"].nunique()
            ),
        })
    slopes_df = pd.DataFrame(slopes)

    # Population-averaged intercept (mean of task FE, including reference = 0).
    params = result.fe_params
    task_keys = [k for k in params.index if str(k).startswith("C(task)")]
    mean_task_fe = float(np.mean([0.0] + [float(params[k]) for k in task_keys]))
    intercept = float(params["Intercept"]) + mean_task_fe
    genai_main = float(
        params.get("C(group, Treatment(reference='Human'))[T.GenAI]", 0.0)
    )
    slopes_df = slopes_df.assign(
        intercept=np.where(
            slopes_df["group"].eq("Human"),
            intercept,
            intercept + genai_main,
        )
    )
    return result, coef_df, slopes_df, df


def _draw_mixedlm_prior_error_line(
    ax,
    slopes_df: pd.DataFrame,
    *,
    scope: str,
    color: str,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> None:
    row = slopes_df.loc[slopes_df["scope"] == scope]
    if row.empty:
        return
    slope = float(row.iloc[0]["slope"])
    intercept = float(row.iloc[0]["intercept"])
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return
    xline = np.linspace(x_min, x_max, 50)
    yline = intercept + slope * xline
    ax.plot(
        xline,
        yline,
        color=color,
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )


def _sig_suffix(p: float) -> str:
    """Stars in parentheses when significant; bare NS otherwise."""
    label = significance_label(p)
    if label == "NS" or label == "n/a":
        return label
    return f"({label})"


def _annotate_mixedlm_prior_error_slope(
    ax,
    slopes_df: pd.DataFrame,
    *,
    scope: str,
    color: str,
) -> None:
    row = slopes_df.loc[slopes_df["scope"] == scope]
    if row.empty:
        return
    r = row.iloc[0]
    slope = float(r["slope"])
    pval = float(r["pvalue"])
    if np.isfinite(slope) and np.isfinite(pval):
        annot = (
            rf"MixedLM $\beta$ = {slope:+.3f} "
            rf"{_sig_suffix(pval)}"
        )
    elif np.isfinite(slope):
        annot = rf"MixedLM $\beta$ = {slope:+.3f}"
    else:
        annot = "MixedLM n/a"
    ax.text(
        0.03,
        0.97,
        annot,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=VIZ_BRACKET_FONTSIZE,
        color=color,
        fontweight="bold",
        linespacing=1.25,
    )


def _shift_accuracy_panel_ylim(
    y_vals: np.ndarray,
    *,
    fit_y: np.ndarray | None = None,
) -> float:
    ymax = float(np.nanmax(y_vals)) if len(y_vals) else 0.1
    if fit_y is not None and len(fit_y):
        ymax = max(ymax, float(np.nanmax(fit_y)))
    return max(ymax * 1.18, 0.12)


def _draw_shift_accuracy_regression(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linestyle: str = "--",
) -> None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return
    slope, intercept, _, _, _ = linregress(x, y)
    xline = np.linspace(float(np.min(x)), float(np.max(x)), 50)
    yline = slope * xline + intercept
    ax.plot(
        xline,
        yline,
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
        zorder=2,
    )


def _annotate_shift_accuracy_corr(
    ax,
    corr_df: pd.DataFrame,
    *,
    task_key: str,
    scope: str,
) -> None:
    corr_row = corr_df.loc[
        (corr_df["task_key"] == task_key) & (corr_df["scope"] == scope)
    ]
    if corr_row.empty:
        return
    rho = float(corr_row.iloc[0]["spearman_rho"])
    pval = float(corr_row.iloc[0]["spearman_pvalue"])
    sig = significance_label(pval)
    if np.isfinite(rho):
        annot = f"Spearman ρ = {rho:+.2f} {_sig_suffix(pval)}"
    else:
        annot = f"n = {int(corr_row.iloc[0]['n'])}\n(too few for Spearman ρ)"
    color = SIG_TEXT_COLOR if is_significant(pval) else FOOTNOTE_COLOR
    weight = "bold" if is_significant(pval) else "normal"
    ax.text(
        0.03,
        0.97,
        annot,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=VIZ_BRACKET_FONTSIZE,
        color=color,
        fontweight=weight,
    )


def _annotate_shift_accuracy_ols(
    ax,
    corr_df: pd.DataFrame,
    *,
    task_key: str,
    scope: str,
) -> None:
    """OLS slope annotation only."""
    corr_row = corr_df.loc[
        (corr_df["task_key"] == task_key) & (corr_df["scope"] == scope)
    ]
    if corr_row.empty:
        return
    slope = float(corr_row.iloc[0]["ols_slope"])
    if np.isfinite(slope):
        annot = rf"OLS $\beta$ = {slope:+.3f}"
    else:
        annot = f"n = {int(corr_row.iloc[0]['n'])}\n(too few for OLS)"
    ax.text(
        0.03,
        0.97,
        annot,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=VIZ_BRACKET_FONTSIZE,
        color="black",
    )


def _format_p_with_trailing_stars(p: float) -> str:
    """Italic capital P with trailing stars when significant, e.g. P < 0.001***."""
    return format_p_value_with_trailing_stars(p)


def _annotate_shift_accuracy_ols_and_corr(
    ax,
    corr_df: pd.DataFrame,
    *,
    task_key: str,
    scope: str,
    color: str = "black",
) -> None:
    """OLS β and Spearman ρ, each with parenthesized italic P (+ trailing stars)."""
    corr_row = corr_df.loc[
        (corr_df["task_key"] == task_key) & (corr_df["scope"] == scope)
    ]
    if corr_row.empty:
        return
    row = corr_row.iloc[0]
    slope = float(row["ols_slope"])
    rho = float(row["spearman_rho"])
    spearman_p = float(row["spearman_pvalue"])
    if "ols_pvalue" in row.index and np.isfinite(row["ols_pvalue"]):
        ols_p = float(row["ols_pvalue"])
    elif "pearson_pvalue" in row.index:
        # Simple OLS slope test ≡ Pearson r test.
        ols_p = float(row["pearson_pvalue"])
    else:
        ols_p = np.nan
    n = int(row["n"])
    lines: list[str] = []
    if np.isfinite(slope):
        lines.append(
            rf"OLS $\beta$ = {slope:+.3f}  ({_format_p_with_trailing_stars(ols_p)})"
        )
    if np.isfinite(rho):
        lines.append(
            f"Spearman ρ = {rho:+.2f}  ({_format_p_with_trailing_stars(spearman_p)})"
        )
    if not lines:
        lines.append(f"n = {n}\n(too few for OLS / Spearman ρ)")
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=VIZ_BRACKET_FONTSIZE,
        color=color,
        fontweight="bold",
        linespacing=1.25,
    )


def _draw_shift_accuracy_scatter_panel(
    ax,
    task_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    *,
    task_key: str,
    scope: str,
    group_color: str,
    marker_size: int,
    marker_alpha: float,
    panel_title: str | None = None,
    annotate: str = "spearman",
    xlabel: str | None = None,
    ylabel: str | None = "Theory-embedding shift",
    axis_label_fontsize: float | None = None,
    left_align_title: bool = False,
) -> None:
    x = task_df["pre_ml_accuracy"].to_numpy(dtype=float)
    y = task_df["self_pre_post_cosine_distance"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(
        x,
        y,
        s=marker_size,
        alpha=marker_alpha,
        color=group_color,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    if len(x) >= 2:
        _draw_shift_accuracy_regression(ax, x, y, color=group_color)
    if annotate == "spearman":
        _annotate_shift_accuracy_corr(
            ax,
            corr_df,
            task_key=task_key,
            scope=scope,
        )
    elif annotate == "ols":
        _annotate_shift_accuracy_ols(
            ax,
            corr_df,
            task_key=task_key,
            scope=scope,
        )
    elif annotate == "ols_spearman":
        _annotate_shift_accuracy_ols_and_corr(
            ax,
            corr_df,
            task_key=task_key,
            scope=scope,
            color=group_color,
        )

    y_top = _shift_accuracy_panel_ylim(y, fit_y=y if len(y) >= 2 else None)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, y_top)
    if panel_title:
        if left_align_title:
            ax.set_title("")
            ax._pooled_panel_title = ax.text(
                0.0,
                1.02,
                panel_title,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                clip_on=False,
            )
        else:
            ax.set_title(
                panel_title,
                fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                pad=8,
            )
    axis_fs = (
        DIVERSITY_PRED_YLABEL_FONTSIZE
        if axis_label_fontsize is None
        else axis_label_fontsize
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_fs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_fs)
    ax.tick_params(axis="both", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
    ax.grid(True, alpha=0.22, linewidth=0.6)


def _accuracy_group_panel_ylim(groups: list[dict]) -> float:
    tops = []
    for row in groups:
        err_hi = max(0.0, float(row["ci_high"]) - float(row["mean_shift"]))
        tops.append(float(row["mean_shift"]) + err_hi)
    ymax = max(tops) if tops else 0.1
    step = 0.05 if ymax < 0.5 else 0.1
    return max(float(np.ceil(ymax * 1.18 / step) * step), 0.12)


def _shift_accuracy_test_annot_line(
    test_name: str,
    pval: float,
) -> tuple[str, str, str]:
    if not np.isfinite(pval):
        return f"{test_name} (n/a)", FOOTNOTE_COLOR, "normal"
    sig = significance_label(pval)
    color = SIG_TEXT_COLOR if is_significant(pval) else FOOTNOTE_COLOR
    weight = "bold" if is_significant(pval) else "normal"
    return f"{test_name} ({sig})", color, weight


def _annotate_shift_accuracy_group_tests_panel(ax, panel_stats: dict) -> None:
    y = 0.97
    line_step = 0.055
    for test_name, pval_key in (
        ("Kruskal–Wallis", "kruskal_pvalue"),
        ("Welch t", "welch_pvalue"),
    ):
        annot, color, weight = _shift_accuracy_test_annot_line(
            test_name,
            float(panel_stats[pval_key]),
        )
        ax.text(
            0.03,
            y,
            annot,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=VIZ_BRACKET_FONTSIZE,
            color=color,
            fontweight=weight,
        )
        y -= line_step


def _draw_shift_accuracy_high_low_kw_panel(
    ax,
    task_df: pd.DataFrame,
    *,
    group_color: str,
    panel_title: str | None = None,
    show_xticklabels: bool = True,
) -> None:
    x = task_df["pre_ml_accuracy"].to_numpy(dtype=float)
    y = task_df["self_pre_post_cosine_distance"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    panel_stats = _shift_accuracy_group_panel_stats(x[mask], y[mask])
    if panel_stats is None:
        ax.text(
            0.5,
            0.5,
            "Too few observations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
            color=FOOTNOTE_COLOR,
        )
        if panel_title:
            ax.set_title(
                panel_title,
                fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                pad=8,
            )
        return

    groups = panel_stats["groups"]
    colors = _accuracy_group_bar_colors(group_color, len(groups))
    xpos = np.arange(len(groups), dtype=float)
    for idx, (row, color) in enumerate(zip(groups, colors)):
        mean_shift = float(row["mean_shift"])
        err_lo, err_hi = ci_errorbar_offsets(
            mean_shift,
            float(row["ci_low"]),
            float(row["ci_high"]),
        )
        ax.bar(
            xpos[idx],
            mean_shift,
            SHIFT_ACCURACY_BINARY_BAR_WIDTH,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
        )
        ax.errorbar(
            xpos[idx],
            mean_shift,
            yerr=[[err_lo], [err_hi]],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )

    _annotate_shift_accuracy_group_tests_panel(ax, panel_stats)
    y_top = _accuracy_group_panel_ylim(groups)
    ax.set_xticks(xpos)
    if show_xticklabels:
        ax.set_xticklabels(
            [row["group_label"] for row in groups],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0.0, y_top)
    ytick_step = 0.05 if y_top <= 0.5 else 0.1
    ax.set_yticks(np.arange(0.0, y_top + 0.001, ytick_step))
    if panel_title:
        ax.set_title(
            panel_title,
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            fontweight="bold",
            pad=8,
        )
    ax.set_ylabel(
        "Theory-embedding shift",
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
    ax.tick_params(axis="x", pad=6)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def plot_shift_by_pre_ml_accuracy_high_low_kw(
    merged_df: pd.DataFrame,
    outpath: Path,
    *,
    group_key: str,
    suptitle: str,
    group_color: str,
) -> None:
    order = {key: i for i, key in enumerate(TASK_PANEL_ORDER)}
    group_df = merged_df.loc[merged_df["participant_group"] == group_key]
    task_keys = sorted(
        group_df["task_key"].unique(),
        key=lambda key: order.get(key, 999),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DIVERSITY_PRED_FIGSIZE,
        gridspec_kw={
            "hspace": DIVERSITY_PRED_ROW_GAP,
            "wspace": DIVERSITY_PRED_COL_GAP,
        },
    )

    for ax, task_key in zip(axes.ravel(), task_keys):
        task_df = group_df.loc[group_df["task_key"] == task_key]
        _draw_shift_accuracy_high_low_kw_panel(
            ax,
            task_df,
            group_color=group_color,
            panel_title=task_label_from_key(task_key),
            show_xticklabels=True,
        )

    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=SELF_PRE_POST_SHIFT_ACCURACY_KW_METRIC,
        suptitle_fontsize=VIZ_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    footnote_y, subplot_bottom = shift_accuracy_high_low_bottom_layout(
        SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE,
    )
    plt.subplots_adjust(
        left=0.11,
        right=0.98,
        top=header.panel_top,
        bottom=subplot_bottom,
        hspace=DIVERSITY_PRED_ROW_GAP,
        wspace=DIVERSITY_PRED_COL_GAP,
    )
    draw_self_pre_post_welch_footnote(
        fig,
        SELF_PRE_POST_SHIFT_ACCURACY_KW_FOOTNOTE,
        y=footnote_y,
    )
    save_figure_pdf_svg(
        fig,
        outpath,
        bbox_inches="tight",
        pad_inches=SELF_PRE_POST_SHIFT_ACCURACY_KW_SAVE_PAD,
    )


def plot_shift_vs_pre_ml_accuracy_scatter(
    merged_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    outpath: Path,
    *,
    group_key: str,
    scope: str,
    suptitle: str,
    group_color: str,
    marker_size: int,
    marker_alpha: float,
) -> None:
    order = {key: i for i, key in enumerate(TASK_PANEL_ORDER)}
    group_df = merged_df.loc[merged_df["participant_group"] == group_key]
    task_keys = sorted(
        group_df["task_key"].unique(),
        key=lambda key: order.get(key, 999),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DIVERSITY_PRED_FIGSIZE,
        gridspec_kw={
            "hspace": DIVERSITY_PRED_ROW_GAP,
            "wspace": DIVERSITY_PRED_COL_GAP,
        },
    )

    for ax, task_key in zip(axes.ravel(), task_keys):
        task_df = group_df.loc[group_df["task_key"] == task_key]
        _draw_shift_accuracy_scatter_panel(
            ax,
            task_df,
            corr_df,
            task_key=task_key,
            scope=scope,
            group_color=group_color,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            panel_title=task_label_from_key(task_key),
        )

    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_METRIC,
        suptitle_fontsize=VIZ_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    footnote_y, subplot_bottom, shared_xlabel_y = shift_accuracy_scatter_bottom_layout(
        SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_FOOTNOTE,
    )
    plt.subplots_adjust(
        left=0.11,
        right=0.98,
        top=header.panel_top,
        bottom=subplot_bottom,
        hspace=DIVERSITY_PRED_ROW_GAP,
        wspace=DIVERSITY_PRED_COL_GAP,
    )
    fig.supxlabel(
        "Pre-ML prediction accuracy",
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
        y=shared_xlabel_y,
    )
    draw_self_pre_post_welch_footnote(
        fig,
        SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_FOOTNOTE,
        y=footnote_y,
    )
    save_figure_pdf_svg(
        fig,
        outpath,
        bbox_inches="tight",
        pad_inches=SELF_PRE_POST_SHIFT_ACCURACY_SCATTER_SAVE_PAD,
    )



def _pooled_group_mean_shift_rows(
    merged_df: pd.DataFrame,
    *,
    groups: list[str] | None = None,
) -> list[dict]:
    if groups is None:
        groups = list(SELF_PRE_POST_DISTANCE_GROUPS)
    rows: list[dict] = []
    for group in groups:
        vals = merged_df.loc[
            merged_df["participant_group"] == group,
            "self_pre_post_cosine_distance",
        ].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        ci_lo, ci_hi = bootstrap_mean_ci(vals, seed=ANALYSIS_SEED)
        rows.append(
            {
                "participant_group": group,
                "group_label": SELF_PRE_POST_DISTANCE_GROUP_LABELS[group],
                "n": len(vals),
                "mean_shift": mean_val,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }
        )
    return rows


def _draw_pooled_mean_shift_bar_panel(
    ax,
    merged_df: pd.DataFrame,
    *,
    panel_title: str,
    ylim_top: float | None = None,
    comparison_style: str = "box",
) -> float:
    """Human vs GenAI mean shift bars on pooled task×effect observations.

    comparison_style:
      - "box": legacy boxed Welch text above bars
      - "bracket": Nature-style black significance bracket (no box; NS omitted)
      - "none": bars only
    """
    groups = list(SELF_PRE_POST_DISTANCE_GROUPS)
    group_colors = SELF_PRE_POST_DISTANCE_GROUP_COLORS
    rows = _pooled_group_mean_shift_rows(merged_df, groups=groups)
    if not rows:
        ax.text(
            0.5,
            0.5,
            "No observations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=FOOTNOTE_COLOR,
        )
        return 0.12

    xpos = np.arange(len(rows), dtype=float)
    bar_top = 0.0
    for i, row in enumerate(rows):
        mean_val = float(row["mean_shift"])
        err_lo, err_hi = ci_errorbar_offsets(
            mean_val, float(row["ci_low"]), float(row["ci_high"])
        )
        bar_top = max(bar_top, mean_val + err_hi)
        group = row["participant_group"]
        ax.bar(
            xpos[i],
            mean_val,
            SELF_PRE_POST_BAR_WIDTH,
            color=group_colors[group],
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
        )
        ax.errorbar(
            xpos[i],
            mean_val,
            yerr=[[err_lo], [err_hi]],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )

    welch_lines = self_pre_post_welch_comparisons(merged_df, groups)
    if ylim_top is None:
        if comparison_style == "box":
            ylim_top = self_pre_post_comparison_box_top(
                bar_top,
                len(welch_lines),
                ylim_top=max(bar_top * SELF_PRE_POST_YLIM_PAD, 0.12),
            )
        else:
            # Leave headroom for a Nature bracket when present.
            pad = 1.28 if comparison_style == "bracket" else SELF_PRE_POST_YLIM_PAD
            step = 0.05 if bar_top < 0.5 else 0.1
            ylim_top = max(float(np.ceil(bar_top * pad / step) * step), 0.12)
    ax.set_xticks(xpos)
    ax.set_xticklabels([row["group_label"] for row in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, ylim_top)
    ytick_step = 0.05 if ylim_top <= 0.5 else 0.1
    yticks = np.arange(0.0, ylim_top + 1e-9, ytick_step)
    # Do not label the ceiling tick (e.g. drop 0.20 when ylim=0.20).
    if len(yticks) > 1 and np.isclose(yticks[-1], float(ylim_top)):
        yticks = yticks[:-1]
    yticks = np.asarray(
        [t for t in yticks if float(t) < float(ylim_top) - 1e-9],
        dtype=float,
    )
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_major_locator(plt.FixedLocator(yticks.tolist()))
    # Custom title so it can be left-aligned with the y-axis label.
    ax.set_title("")
    ax._pooled_panel_title = ax.text(
        0.0,
        1.02,
        panel_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
        fontweight="bold",
        clip_on=False,
    )
    ax.tick_params(axis="both", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    if comparison_style == "box":
        draw_self_pre_post_comparison_box_above_bars(
            ax,
            (len(rows) - 1) / 2.0,
            welch_lines,
            bar_top,
            ylim_top=ylim_top,
        )
    elif comparison_style == "bracket" and len(rows) >= 2 and welch_lines:
        _, pval = welch_lines[0]
        if is_significant(pval):
            draw_paired_pre_post_bracket(
                ax,
                float(xpos[0]),
                float(xpos[-1]),
                bar_top,
                pval,
                fontsize=VIZ_BRACKET_FONTSIZE,
                label=_format_p_with_trailing_stars(pval),
                fontweight="normal",
                color="black",
            )
    return float(ylim_top)


def _align_panel_title_to_ylabel_left(ax, *, y_axes: float = 1.02) -> None:
    """Shift custom panel title so its left edge matches the y-axis label."""
    text = getattr(ax, "_pooled_panel_title", None)
    ylab = ax.yaxis.label
    if text is None or ylab is None or not str(ylab.get_text()).strip():
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_px = float(ylab.get_window_extent(renderer).x0)
    x_axes = float(ax.transAxes.inverted().transform((left_px, 0.0))[0])
    text.set_position((x_axes, y_axes))


def _latex_signed_num(value: float, *, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        return f"$-{text[1:]}$"
    return text


def _latex_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def _latex_p_with_stars(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    stars = _latex_stars(p)
    if p < 0.001:
        return f"$<$0.001{stars}"
    # Attach stars inside math when present.
    if stars:
        return f"${p:.3f}{stars[1:]}"
    return f"{p:.3f}"


def _latex_coef_with_stars(coef: float, p: float, *, decimals: int = 3) -> str:
    num = _latex_signed_num(coef, decimals=decimals)
    stars = _latex_stars(p)
    if num.startswith("$") and num.endswith("$") and stars.startswith("$"):
        return f"${num[1:-1]}{stars[1:]}"
    return f"{num}{stars}"


def _latex_plain_num(value: float, *, decimals: int = 3) -> str:
    """Plain decimal for use inside math-mode CI brackets."""
    if not np.isfinite(value):
        return r"\mathrm{---}"
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        return f"-{text[1:]}"
    return text


def _latex_ci(lo: float, hi: float, *, decimals: int = 3) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return (
        rf"$[{_latex_plain_num(lo, decimals=decimals)},\ "
        rf"{_latex_plain_num(hi, decimals=decimals)}]$"
    )


def _latex_coef_ci_cell(
    coef: float,
    lo: float,
    hi: float,
    p: float,
    *,
    decimals: int = 3,
) -> str:
    """Point estimate with stars + 95% CI underneath (diversity-table style)."""
    point = _latex_coef_with_stars(coef, p, decimals=decimals)
    return (
        r"\begin{tabular}{@{}c@{}}"
        rf"{point}\\"
        rf"{_latex_ci(lo, hi, decimals=decimals)}"
        r"\end{tabular}"
    )


def build_revision_prior_error_mixedlm_tex(
    coef_df: pd.DataFrame,
    slopes_df: pd.DataFrame,
) -> str:
    """Transposed booktabs MixedLM table: terms + implied slopes as columns."""
    by_term = {str(r["term"]): r for _, r in coef_df.iterrows()}

    def _stats(coef: float, lo: float, hi: float, p: float) -> tuple[str, str, str]:
        return (
            _latex_coef_with_stars(coef, p),
            _latex_ci(lo, hi),
            _latex_p_with_stars(p),
        )

    headers: list[str] = []
    stats_cols: list[tuple[str, str, str]] = []

    for term in SELF_PRE_POST_POOLED_MIXEDLM_MAIN_TERMS:
        if term not in by_term:
            continue
        r = by_term[term]
        headers.append(SELF_PRE_POST_POOLED_TABLE_COL_HEADERS.get(term, term))
        stats_cols.append(
            _stats(
                float(r["coef"]),
                float(r["ci_low"]),
                float(r["ci_high"]),
                float(r["pvalue"]),
            )
        )

    for _, r in slopes_df.iterrows():
        scope = str(r["scope"])
        headers.append(rf"\makecell[c]{{Slope\\({scope})}}")
        stats_cols.append(
            _stats(
                float(r["slope"]),
                float(r["ci_low"]),
                float(r["ci_high"]),
                float(r["pvalue"]),
            )
        )

    n = len(headers)
    coef_cells = [c for c, _, _ in stats_cols]
    ci_cells = [ci for _, ci, _ in stats_cols]
    return "\n".join([
        "% Auto-generated by fig_core_tail_structure / fig_theory_revision",
        r"% Transposed MixedLM (β0–β3 + implied slopes; no P row; task FE omitted).",
        "",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4.6pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\begin{{tabular}}{{@{{}}l*{{{n}}}{{c}}@{{}}}}",
        r"\toprule",
        " & " + " & ".join(headers) + r" \\",
        rf"\cmidrule(lr){{2-{n + 1}}}",
        r"Coef. & " + " & ".join(coef_cells) + r" \\",
        r"95\% CI & " + " & ".join(ci_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
    ])


def build_pooled_three_panel_combined_tex(
    *,
    abc_pdf_name: str,
    mixedlm_table_tex: str,
) -> str:
    """a--c on top (horizontal); transposed MixedLM table below as panel d."""
    return "\n".join([
        "% Auto-generated by fig_core_tail_structure / fig_theory_revision",
        "",
        r"\begin{tabular}{@{}l@{}}",
        rf"\includegraphics[width=7.2in]{{{abc_pdf_name}}} \\",
        r"\rule{0pt}{1.0em} \\",
        rf"{{\small\textbf{{{SELF_PRE_POST_POOLED_THREE_PANEL_TITLES[3]}}}}}\\[0.45em]",
        mixedlm_table_tex.strip(),
        r"\end{tabular}",
        "",
    ])


def _plot_pooled_abc_panels_pdf(
    merged_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_pdf: Path,
) -> Path:
    """Render panels a--c in a horizontal 1×3 row (PNG for LaTeX include)."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.8, 5.6),
        gridspec_kw={"wspace": 0.28},
    )
    ax_bar, ax_human, ax_genai = axes

    y_all = merged_df["self_pre_post_cosine_distance"].to_numpy(dtype=float)
    y_all = y_all[np.isfinite(y_all)]
    scatter_top = _shift_accuracy_panel_ylim(y_all)

    _draw_pooled_mean_shift_bar_panel(
        ax_bar,
        merged_df,
        panel_title=SELF_PRE_POST_POOLED_THREE_PANEL_TITLES[0],
        ylim_top=None,
        comparison_style="bracket",
    )
    human_df = merged_df.loc[merged_df["participant_group"] == "Human"]
    genai_df = merged_df.loc[merged_df["participant_group"] == "GenAI"]
    xlabel = SELF_PRE_POST_POOLED_THREE_PANEL_XLABEL
    axis_fs = SELF_PRE_POST_POOLED_THREE_PANEL_AXIS_FONTSIZE
    _draw_shift_accuracy_scatter_panel(
        ax_human,
        human_df,
        corr_df,
        task_key="pooled",
        scope="Humans",
        group_color=SELF_PRE_POST_DISTANCE_GROUP_COLORS["Human"],
        marker_size=28,
        marker_alpha=0.55,
        panel_title=SELF_PRE_POST_POOLED_THREE_PANEL_TITLES[1],
        annotate="ols_spearman",
        xlabel=xlabel,
        ylabel=SELF_PRE_POST_POOLED_THREE_PANEL_YLABEL,
        axis_label_fontsize=axis_fs,
        left_align_title=True,
    )
    _draw_shift_accuracy_scatter_panel(
        ax_genai,
        genai_df,
        corr_df,
        task_key="pooled",
        scope="GenAI",
        group_color=SELF_PRE_POST_DISTANCE_GROUP_COLORS["GenAI"],
        marker_size=36,
        marker_alpha=0.72,
        panel_title=SELF_PRE_POST_POOLED_THREE_PANEL_TITLES[2],
        annotate="ols_spearman",
        xlabel=xlabel,
        ylabel=SELF_PRE_POST_POOLED_THREE_PANEL_YLABEL,
        axis_label_fontsize=axis_fs,
        left_align_title=True,
    )
    ax_human.set_ylim(0.0, scatter_top)
    ax_genai.set_ylim(0.0, scatter_top)
    for ax in (ax_bar, ax_human, ax_genai):
        ax.set_ylabel(
            SELF_PRE_POST_POOLED_THREE_PANEL_YLABEL,
            fontsize=axis_fs,
        )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.88,
        bottom=0.18,
        wspace=0.32,
    )
    bar_pos = ax_bar.get_position()
    for ax in (ax_human, ax_genai):
        pos = ax.get_position()
        ax.set_position([pos.x0, bar_pos.y0, pos.width, bar_pos.height])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _content_y0(ax) -> float:
        bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        return float(bbox.y0)

    a_y0 = _content_y0(ax_bar)
    bc_y0 = min(_content_y0(ax_human), _content_y0(ax_genai))
    delta = a_y0 - bc_y0
    if delta > 0.005:
        bar_pos = ax_bar.get_position()
        ax_bar.set_position([
            bar_pos.x0,
            bar_pos.y0 - delta,
            bar_pos.width,
            bar_pos.height + delta,
        ])

    for ax in (ax_bar, ax_human, ax_genai):
        _align_panel_title_to_ylabel_left(ax)

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_pdf,
        format="png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=SELF_PRE_POST_POOLED_THREE_PANEL_SAVE_PAD,
    )
    plt.close(fig)
    return out_pdf



def plot_self_pre_post_shift_pooled_three_panel(
    merged_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    slopes_df: pd.DataFrame,
    outpath: Path,
) -> Path:
    """Four-panel figure: mean bars | Humans OLS | GenAI OLS | LaTeX MixedLM table.

    Panels (a)--(c) are matplotlib; panel (d) is a booktabs table compiled via
    tectonic (same pipeline as ``forecast_diversity_combined_table_standalone``).
    """
    from common.latex_table_pdf import compile_standalone_table

    outpath = Path(outpath)
    outdir = outpath.parent
    stem = outpath.with_suffix("").name
    outdir.mkdir(parents=True, exist_ok=True)

    abc_pdf = _plot_pooled_abc_panels_pdf(
        merged_df,
        corr_df,
        outdir / f"{SELF_PRE_POST_POOLED_ABC_STEM}.png",
    )
    mixedlm_tex = build_revision_prior_error_mixedlm_tex(coef_df, slopes_df)

    combined_tex = build_pooled_three_panel_combined_tex(
        abc_pdf_name=abc_pdf.name,
        mixedlm_table_tex=mixedlm_tex,
    )
    compiled = compile_standalone_table(
        outdir,
        stem,
        combined_tex,
        output_format="svg+png",
        png_dpi=400,
        crop="standalone",
        delete_intermediate_tex=True,
        extra_packages=[r"\usepackage{graphicx}", r"\usepackage{amsmath}", r"\usepackage{makecell}"],
    )

    final_png = outdir / f"{stem}.png"
    final_svg = outdir / f"{stem}.svg"
    standalone_png = outdir / f"{stem}_standalone.png"
    standalone_svg = outdir / f"{stem}_standalone.svg"
    standalone_pdf = outdir / f"{stem}_standalone.pdf"
    if standalone_png.is_file():
        standalone_png.replace(final_png)
    if standalone_svg.is_file():
        standalone_svg.replace(final_svg)
    standalone_pdf.unlink(missing_ok=True)
    (outdir / f"{stem}.pdf").unlink(missing_ok=True)
    (outdir / f"{stem}_standalone.tex").unlink(missing_ok=True)
    (outdir / f"{SELF_PRE_POST_MIXEDLM_TABLE_STEM}.tex").unlink(missing_ok=True)
    # Drop temporary a--c include PDF and any preview leftovers.
    abc_pdf.unlink(missing_ok=True)
    for stale in outdir.glob(f"{SELF_PRE_POST_POOLED_ABC_STEM}*"):
        stale.unlink(missing_ok=True)
    for stale in outdir.glob(f"{stem}_preview*"):
        stale.unlink(missing_ok=True)
    return final_png if final_png.is_file() else compiled


def run_shift_conditioned_on_pre_ml_accuracy(
    participant_df: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """Visualize theory-embedding shift as a function of pre-ML accuracy."""
    collapsed_df = participant_df.loc[participant_df["collapsed"]].copy()
    merged = merge_shift_with_pre_ml_accuracy(collapsed_df)
    corr_df = collect_shift_accuracy_correlations(merged)
    kw_df = collect_shift_accuracy_kruskal_wallis(merged)
    _, mixedlm_coef_df, mixedlm_slopes_df, _ = fit_revision_prior_error_mixedlm(merged)

    outdir.mkdir(parents=True, exist_ok=True)
    # Public CSV: MixedLM only (panel d); a–c stats stay in-figure.
    mixedlm_path = outdir / SELF_PRE_POST_MIXEDLM_CSV
    slopes_path = outdir / SELF_PRE_POST_MIXEDLM_SLOPES_CSV

    mixedlm_coef_df.to_csv(mixedlm_path, index=False, encoding="utf-8-sig")
    mixedlm_slopes_df.to_csv(slopes_path, index=False, encoding="utf-8-sig")

    pooled_path = outdir / SELF_PRE_POST_POOLED_THREE_PANEL_FIG
    plot_self_pre_post_shift_pooled_three_panel(
        merged, corr_df, mixedlm_coef_df, mixedlm_slopes_df, pooled_path
    )
    print(f"Saved: {pooled_path}")
    print(f"Saved: {mixedlm_path}")
    print(f"Saved: {slopes_path}")
    print("\n--- MixedLM: revision ~ prior_error * group + task FE + (1|contributor) ---")
    for _, row in mixedlm_slopes_df.iterrows():
        print(
            f"  {row['scope']:7} β_prior_error={row['slope']:+.4f} "
            f"(SE={row['se']:.4f}, p={row['pvalue']:.4g}) {row['significance']} "
            f"[n_obs={int(row['n_obs'])}, n_contrib={int(row['n_contributors'])}]"
        )

    for scope in ("Humans", "GenAI"):
        print(f"\n--- Shift vs pre-ML accuracy ({scope}) ---")
        scope_corr = corr_df.loc[corr_df["scope"] == scope]
        for _, row in scope_corr.iterrows():
            rho = row["spearman_rho"]
            rho_str = f"{rho:+.3f}" if np.isfinite(rho) else "n/a"
            pval = row["spearman_pvalue"]
            pval_str = f"{pval:.4f}" if np.isfinite(pval) else "n/a"
            r = row["pearson_r"]
            r_str = f"{r:+.3f}" if np.isfinite(r) else "n/a"
            r_pval = row["pearson_pvalue"]
            r_pval_str = f"{r_pval:.4f}" if np.isfinite(r_pval) else "n/a"
            print(
                f"  {row['task_label']:28} n={int(row['n']):3d} "
                f"ρ={rho_str} p={pval_str} {row['significance']} | "
                f"r={r_str} p={r_pval_str} {row['pearson_significance']}"
            )

        print(f"\n--- High/low shift tests ({scope}) ---")
        scope_kw = kw_df.loc[kw_df["scope"] == scope]
        for _, row in scope_kw.iterrows():
            h = row["kruskal_h"]
            h_str = f"{h:.2f}" if np.isfinite(h) else "n/a"
            kw_pval = row["kruskal_pvalue"]
            kw_pval_str = f"{kw_pval:.4f}" if np.isfinite(kw_pval) else "n/a"
            welch_pval = row["welch_pvalue"]
            welch_pval_str = f"{welch_pval:.4f}" if np.isfinite(welch_pval) else "n/a"
            group_bits = []
            for col in sorted(row.index):
                if not col.startswith("n_") or col == "n":
                    continue
                group_key = col[2:]
                mean_col = f"mean_shift_{group_key}"
                vals_col = f"accuracy_values_{group_key}"
                if mean_col not in row.index or not np.isfinite(row[mean_col]):
                    continue
                acc_text = row[vals_col] if vals_col in row.index else group_key
                group_bits.append(
                    f"{acc_text}={row[mean_col]:.3f} (n={int(row[col])})"
                )
            mean_str = ", ".join(group_bits) if group_bits else "n/a"
            print(
                f"  {row['task_label']:28} "
                f"KW H={h_str} p={kw_pval_str} {row['kruskal_significance']} | "
                f"Welch t p={welch_pval_str} {row['welch_significance']} | "
                f"{mean_str}"
            )
    return merged


def run_self_pre_post_embedding_distance(
    embeddings_root: Path,
    embedding_col: str,
    outdir: Path,
) -> pd.DataFrame:
    """Compute self pre–post distances in memory (no panel-a CSV exports).

    Returns the participant table for the pooled theory-revision figure.
    """
    collapsed = True
    groups = list(SELF_PRE_POST_DISTANCE_GROUPS)
    group_labels = dict(SELF_PRE_POST_DISTANCE_GROUP_LABELS)
    summary_rows: list[dict] = []
    participant_tables: list[pd.DataFrame] = []

    for pre_dir, post_dir, task_key in discover_task_pairs(embeddings_root):
        print(f"\n=== Self pre–post distance: {task_label_from_key(task_key)} ===")
        rows, table = analyze_self_pre_post_task(
            pre_dir,
            post_dir,
            task_key,
            embedding_col,
            collapsed=collapsed,
            groups=groups,
            group_labels=group_labels,
        )
        summary_rows.extend(rows)
        participant_tables.append(table)
        for row in rows:
            print(
                f"  {row['group_label']}: "
                f"mean={row['mean_self_pre_post_cosine_distance']:.4f} "
                f"(n={row['n_paired']})"
            )

    outdir.mkdir(parents=True, exist_ok=True)
    participant_df = pd.concat(participant_tables, ignore_index=True)
    participant_df = participant_df.loc[participant_df["collapsed"] == collapsed].copy()

    welch_rows = collect_self_pre_post_welch_rows(
        participant_df,
        groups=groups,
        collapsed=collapsed,
        embedding_col=embedding_col,
    )
    for row in welch_rows:
        print(f"  Welch {row['comparison']}: p={row['welch_pvalue']:.4f} {row['significance']}")

    return participant_df


def run_all_self_pre_post_embedding_distance(
    embeddings_root: Path,
    embedding_col: str,
) -> None:
    outdir = comparisons_pre_post_dir(embeddings_root, COMPARISONS_SELF_SUBDIR)
    print(f"\n{'=' * 72}")
    print("Self pre–post embedding distance → theory_revision")
    print(f"Output: {outdir}")
    participant_df = run_self_pre_post_embedding_distance(
        embeddings_root, embedding_col, outdir
    )
    run_shift_conditioned_on_pre_ml_accuracy(participant_df, outdir)


def run_all_prediction1(
    embeddings_root: Path,
    embedding_col: str,
) -> None:
    outdir = comparisons_pre_post_dir(embeddings_root, COMPARISONS_CORE_TAIL_SUBDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, pd.DataFrame] = {}
    sensitivities: dict[str, pd.DataFrame] = {}
    for audience_key, audience_cfg in AUDIENCE_CONFIGS.items():
        print(f"\n{'=' * 72}")
        print(f"Audience: {audience_cfg['count_label']}")
        print(f"Output: {outdir}")
        summary_df, sens_df = run_prediction1(
            embeddings_root, outdir, embedding_col, audience_cfg
        )
        summaries[audience_key] = summary_df
        sensitivities[audience_key] = sens_df

    svg_path = write_core_tail_pre_post_latex_table(
        summaries["human"],
        summaries["genai"],
        outdir,
        human_sens=sensitivities["human"],
        genai_sens=sensitivities["genai"],
    )
    print(f"\nSaved core--tail figure: {svg_path}")


# Prefer fig_core_tail_structure.py / fig_theory_revision.py entrypoints.
