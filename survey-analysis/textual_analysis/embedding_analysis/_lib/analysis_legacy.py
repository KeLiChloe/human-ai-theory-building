"""
Visualize Q4: structure of the theoretical explanation space (library module).

Prefer the thin entrypoints under embedding_analysis/:
    build_network.py
    run_semantic_clustering.py
    fig_semantic_space_map_PCA.py
    fig_within_group_dispersion.py

Research question:
    Do junior scholars, senior scientists, and GenAI differ in the distribution
    or structure of their theoretical explanations?

Input parquet expected columns:
    participant_name
    participant_type
    text_word_count
    raw_embedding_dimension_3072
"""

import argparse
import ast
import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import skew, skewtest
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Package root is embedding_analysis/ (parent of _lib/), not this file's directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEXTUAL_ANALYSIS_DIR = Path(__file__).resolve().parents[2]
SURVEY_ROOT = Path(__file__).resolve().parents[3]
for p in (PACKAGE_ROOT, TEXTUAL_ANALYSIS_DIR, SURVEY_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.viz_config import GROUP_COLORS
from common.stats_utils import (
    bootstrap_mean_ci,
    p_value_paired_one_sided_post_lt_pre,
    p_value_welch_ttest,
    p_value_welch_ttest_one_sided,
)
from viz_style import (
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    GROUP_COLORS_COLLAPSED,
    GROUP_ORDER_COLLAPSED,
    METRIC_SUBTITLE_FONTSIZE,
    PHASE_HATCH_COLOR,
    SAVE_DPI,
    SAVE_PAD_INCHES,
    save_figure_pdf_svg,
    VIZ_AXIS_LABEL_FONTSIZE,
    VIZ_BRACKET_FONTSIZE,
    VIZ_FOOTNOTE_FONTSIZE,
    VIZ_FOOTNOTE_LINE_STEP,
    VIZ_FOOTNOTE_Y,
    VIZ_LEGEND_FONTSIZE,
    VIZ_LEGEND_Y_SHIFT,
    VIZ_PANEL_TITLE_FONTSIZE,
    VIZ_SUPTITLE_FONTSIZE,
    VIZ_SUPTITLE_LINE_SPACING,
    VIZ_SUPYLABEL_X,
    VIZ_TICK_FONTSIZE,
    apply_plot_style,
    collapsed_legend_labels,
    comparison_box_height,
    comparison_pair_label,
    display_label,
    draw_centered_comparison_box,
    draw_paired_pre_post_bracket,
    draw_sig_footnote,
    format_p_value_label,
    format_p_value_with_trailing_stars,
    DISPLAY_LABELS,
    figure_legend_panel_top,
    format_comparison_line,
    fmt_p,
    is_significant,
    ERROR_CAPSIZE,
    ERROR_LINEWIDTH,
    FOOTNOTE_COLOR,
    layout_title_and_metric,
    legend_entry,
    significance_label,
    SIG_LEVEL_LEGEND,
    SIG_TEXT_COLOR,
    BOX_EDGE_NEUTRAL,
    style_axes,
)


try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


try:
    import umap
    HAS_UMAP = True
except Exception:
    # ImportError, or RuntimeError from numba cache when env is restricted.
    HAS_UMAP = False


# ---------------------------------------------------------------------
# Paths & I/O
# ---------------------------------------------------------------------
SCRIPT_DIR = PACKAGE_ROOT
DEFAULT_EMBEDDINGS_ROOT = SCRIPT_DIR / "embeddings_openai_text-embedding-3-large"
DEFAULT_EMBEDDING_DIR = DEFAULT_EMBEDDINGS_ROOT / "race/main-effects/pre-ML"
VISUALIZATIONS_DIRNAME = ""  # legacy; figures live at package root (network/, outputs/)

# ---------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------
PARTICIPANT_NAME_COL = "participant_name"
PARTICIPANT_TYPE_COL = "participant_type"


def clean_participant_display_name(name: str) -> str:
    """Strip GenAI duplicate suffix, e.g. ``Claude_opus4.7(1)`` → ``Claude_opus4.7``."""
    s = str(name).strip()
    if s.endswith("(1)"):
        return s[:-3]
    return s

DEFAULT_EMBEDDING_COLUMNS = [
    "raw_embedding_dimension_3072",
]

# ---------------------------------------------------------------------
# Shared: reproducibility (random seed for PCA, bootstrap, MDS, tests)
# ---------------------------------------------------------------------
ANALYSIS_SEED = 12345
# MPWD inference: row-resample bootstrap / label permutation (fast Gram path below).
MPWD_N_BOOT = 2000
MPWD_N_PERM = 5000

RESPONDENT_ID_ALPHABET = string.ascii_lowercase + string.digits
RESPONDENT_ID_LENGTH = 5
HUMAN_NETWORK_PANEL_IDS = frozenset({"student", "senior", "expert", "Human"})
HUMAN_PARTICIPANT_TYPES = frozenset({"student", "senior", "expert"})
CONTRIBUTOR_ID_PREFIX_BY_PANEL = {
    "student": "phd_contributor",
    "senior": "senior_contributor",
    "expert": "senior_contributor",  # legacy embedding label for seniors
    "Human": "human_contributor",
}
_ANON_CONTRIBUTOR_ID_RE = re.compile(
    r"^(?:phd|senior|human)_contributor_[a-z0-9]+$",
    re.IGNORECASE,
)


def looks_like_anonymized_contributor_id(name: str) -> bool:
    return bool(_ANON_CONTRIBUTOR_ID_RE.fullmatch(clean_participant_display_name(name)))


def collect_participant_names_from_embeddings_root(
    embeddings_root: Path,
    *,
    participant_types: frozenset[str] | set[str] | None = None,
) -> list[str]:
    names: list[str] = []
    for set_dir in discover_embedding_set_dirs(str(embeddings_root)):
        parquet = set_dir / "embeddings_wide.parquet"
        if not parquet.exists():
            continue
        cols = [PARTICIPANT_NAME_COL]
        if participant_types is not None:
            cols.append(PARTICIPANT_TYPE_COL)
        df = pd.read_parquet(parquet, columns=cols)
        if participant_types is not None:
            mask = df[PARTICIPANT_TYPE_COL].isin(participant_types)
            names.extend(df.loc[mask, PARTICIPANT_NAME_COL].astype(str).tolist())
        else:
            names.extend(df[PARTICIPANT_NAME_COL].astype(str).tolist())
    return names


def build_contributor_code_map(
    names: Iterable[str],
    *,
    seed: int = ANALYSIS_SEED,
) -> dict[str, str]:
    """Stable map from cleaned human name → random code (prefix added per panel)."""
    unique = sorted(
        {
            clean_participant_display_name(n)
            for n in names
            if n
            and str(n).strip()
            and str(n).lower() != "nan"
            and not looks_like_anonymized_contributor_id(str(n))
        }
    )
    rng = np.random.default_rng(seed)
    used_codes: set[str] = set()
    mapping: dict[str, str] = {}
    alphabet = np.array(list(RESPONDENT_ID_ALPHABET))
    for name in unique:
        while True:
            code = "".join(rng.choice(alphabet) for _ in range(RESPONDENT_ID_LENGTH))
            if code not in used_codes:
                used_codes.add(code)
                mapping[name] = code
                break
    return mapping


def format_contributor_id(
    name: str,
    panel_group: str,
    code_map: dict[str, str],
) -> str:
    """``phd_contributor_`` / ``senior_contributor_`` / ``human_contributor_`` + code."""
    cleaned = clean_participant_display_name(name)
    if looks_like_anonymized_contributor_id(cleaned):
        return cleaned
    code = code_map.get(cleaned)
    if not code:
        return cleaned
    prefix = CONTRIBUTOR_ID_PREFIX_BY_PANEL.get(panel_group, "human_contributor")
    return f"{prefix}_{code}"


def anonymize_participant_name(
    name: str,
    participant_type: str,
    code_map: dict[str, str],
    *,
    type_lookup: dict[str, str] | None = None,
) -> str:
    """Map human display names to survey-style contributor IDs; keep GenAI labels."""
    cleaned = clean_participant_display_name(name)
    if not cleaned or cleaned.lower() == "nan":
        return cleaned
    if looks_like_anonymized_contributor_id(cleaned):
        return cleaned
    ptype = str(participant_type).strip()
    if ptype == "Human" and type_lookup:
        ptype = type_lookup.get(cleaned, ptype)
    if ptype == "expert":
        ptype = "senior"
    if ptype in HUMAN_PARTICIPANT_TYPES or ptype == "Human" or ptype == "senior":
        panel = ptype if ptype in CONTRIBUTOR_ID_PREFIX_BY_PANEL else "Human"
        return format_contributor_id(cleaned, panel, code_map)
    return cleaned


def anonymize_participant_names_df(
    df: pd.DataFrame,
    code_map: dict[str, str],
    *,
    type_col: str | None = PARTICIPANT_TYPE_COL,
    type_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy with human ``participant_name`` values anonymized."""
    if PARTICIPANT_NAME_COL not in df.columns or not code_map:
        return df
    out = df.copy()
    if type_col and type_col in out.columns:
        types = out[type_col].astype(str)
    else:
        types = pd.Series(["Human"] * len(out), index=out.index)
    out[PARTICIPANT_NAME_COL] = [
        anonymize_participant_name(n, t, code_map, type_lookup=type_lookup)
        for n, t in zip(out[PARTICIPANT_NAME_COL].astype(str), types)
    ]
    return out


def build_respondent_name_map(
    names: Iterable[str],
    *,
    seed: int = ANALYSIS_SEED,
) -> dict[str, str]:
    """Backward-compatible alias: returns name → random code (no prefix)."""
    return build_contributor_code_map(names, seed=seed)


def respondent_name_map_for_embeddings_root(
    embeddings_root: Path,
    *,
    seed: int = ANALYSIS_SEED,
) -> dict[str, str]:
    return build_contributor_code_map(
        collect_participant_names_from_embeddings_root(
            embeddings_root,
            participant_types=HUMAN_PARTICIPANT_TYPES,
        ),
        seed=seed,
    )


def read_anonymized_embeddings_wide(
    path: Path,
    *,
    code_map: dict[str, str] | None = None,
    embeddings_root: Path | None = None,
) -> pd.DataFrame:
    """Load ``embeddings_wide.parquet`` with human names mapped to contributor IDs."""
    path = Path(path)
    df = pd.read_parquet(path)
    if PARTICIPANT_NAME_COL not in df.columns or PARTICIPANT_TYPE_COL not in df.columns:
        return df
    cmap = code_map
    if cmap is None:
        root = embeddings_root
        if root is None:
            try:
                root = infer_embeddings_root(path.parent)
            except ValueError:
                root = path.parent.parent.parent.parent
        cmap = respondent_name_map_for_embeddings_root(root)
    return anonymize_participant_names_df(df, cmap)

# ---------------------------------------------------------------------
# Shared: participant groups, colors, and legend labels (all plots)
# ---------------------------------------------------------------------
GROUP_ORDER = ["student", "senior", "GenAI"]

PARTICIPANT_TYPE_TO_LEGEND = {
    "student": "PhD Students",
    "senior": "Senior Scientists",
    "GenAI": "GenAI",
}

GROUP_COLORS_BY_PARTICIPANT_TYPE = {
    "student": GROUP_COLORS["phd"],
    "senior": GROUP_COLORS["senior"],
    "GenAI": GROUP_COLORS["genai"],
}

COLLAPSED_PARTICIPANT_TYPE_COL = "collapsed_participant_type"
PARTICIPANT_TYPE_TO_COLLAPSED = {
    "student": "Human",
    "senior": "Human",
    "GenAI": "GenAI",
}

EMBEDDING_SET_PART_LABELS = {
    "soi": "Interactions",
}

# ---------------------------------------------------------------------
# Shared: typography and subplot layout (plots 02–06)
# ---------------------------------------------------------------------
DIAG_FONT_AXIS = VIZ_AXIS_LABEL_FONTSIZE
DIAG_FONT_TICK = VIZ_TICK_FONTSIZE
BOX_FONT_XTICK = VIZ_TICK_FONTSIZE
BOX_FONT_YLABEL = VIZ_AXIS_LABEL_FONTSIZE
BOX_SUBPLOT_LEFT = 0.20
BOX_SUBPLOT_TOP = 0.74
BOX_SUPYLABEL_X = 0.06

# ---------------------------------------------------------------------
# Shared: significance bracket layout (plots 02 & 04)
# ---------------------------------------------------------------------
BOX_COMPARISON_BOTTOM = 0.040
BOX_COMPARISON_AXIS_GAP = 0.085
BOX_FOOTNOTE_Y = 0.001
BOX_COMP_MAX_WIDTH_FRAC = 0.90
CENTROID_AXIS_GAP = 0.11
CENTROID_FOOTNOTE_BOTTOM_Y = 0.012
CENTROID_FOOTNOTE_LINE_STEP = 0.024
CENTROID_FOOTNOTE_TEXT_HEIGHT = 0.018
CENTROID_FOOTNOTE_TO_COMP_GAP = 0.016
DIVERSITY_SIG_FOOTNOTE = (
    "Two-sided Welch t-test on pairwise group mean differences.",
    SIG_LEVEL_LEGEND,
)

# ---------------------------------------------------------------------
# Plot 01 — Semantic space map (PCA 2D/3D)
# ---------------------------------------------------------------------
SEMANTIC_MAP_METHODS = ("pca",)
PROJECTION_DIMS = 2
LABEL_POINTS = False
UMAP_NEIGHBORS = 12  # only if UMAP is enabled in SEMANTIC_MAP_METHODS
UMAP_MIN_DIST = 0.10
SCATTER_SIZE = 58
SCATTER_ALPHA = 1.0
MAP_FONT_TITLE = VIZ_SUPTITLE_FONTSIZE
MAP_FONT_SUBTITLE = METRIC_SUBTITLE_FONTSIZE
MAP_FONT_AXIS = VIZ_AXIS_LABEL_FONTSIZE
MAP_FONT_TICK = VIZ_TICK_FONTSIZE
MAP_FIGSIZE_2D = (10.5, 6.8)
MAP_FIGSIZE_3D = (11.0, 6.2)
MAP_PAD_FRAC = 0.12
MAP_PAD_FRAC_RIGHT = 0.26
MAP_PAD_FRAC_RIGHT_PCA = 0.50
MAP_UMAP_PAD_MIN = 0.15  # UMAP fallback padding
MAP_UMAP_PAD_RIGHT_MIN = 0.9

# ---------------------------------------------------------------------
# Plot 02 — Within-group centroid-distance box plot
# ---------------------------------------------------------------------
CENTROID_BOXPLOT_FILENAME = "02_distance_to_group_centroid_boxplot.png"
CENTROID_BOXPLOT_COLLAPSED_FILENAME = (
    "02_distance_to_group_centroid_boxplot_collapsed.png"
)
CENTROID_DIST_FIGSIZE = (8.8, 7.2)  # shared with plot 03
CENTROID_BOX_WIDTH = 0.55
CENTROID_BOX_FACE_ALPHA = 0.62
BOXPLOT_EDGE_COLOR = "#333333"
BOXPLOT_WHISKER_WIDTH = 1.4
BOXPLOT_STAT_COLOR = "#D62728"
CENTROID_MEAN_SIG_FOOTNOTE = (
    "Two-sided Welch t-test on mean within-group centroid distance "
    "(pairwise group comparisons).",
    SIG_LEVEL_LEGEND,
)

# ---------------------------------------------------------------------
# Plot 03 — Within-group centroid-distance distribution
# ---------------------------------------------------------------------
CENTROID_DISTRIBUTION_FILENAME = "03_pairwise_cosine_distance_distribution.png"
CENTROID_DISTRIBUTION_COLLAPSED_FILENAME = (
    "03_pairwise_cosine_distance_distribution_collapsed.png"
)

# ---------------------------------------------------------------------
# Plot 05 — Semantic threshold network
# ---------------------------------------------------------------------
THRESHOLD_QUANTILE = 0.85  # global cosine-similarity quantile for within-group edges

THRESHOLD_NETWORK_FILENAME = "05_semantic_threshold_network.png"
THRESHOLD_NETWORK_COLLAPSED_FILENAME = "05_semantic_threshold_network_collapsed.png"
NETWORK_INTERACTIVE_HTML = "05_semantic_threshold_network.html"
NETWORK_INTERACTIVE_COLLAPSED_HTML = "05_semantic_threshold_network_collapsed.html"
NETWORK_INTERACTIVE_INDEX = "index.html"
NETWORK_LAYOUT_PAD = 0.08
NETWORK_NODE_SIZE = 148
NETWORK_EDGE_COLOR = "#6e6e6e"
NETWORK_EDGE_WIDTH = 0.7
NETWORK_EDGE_ALPHA = 0.32

# ---------------------------------------------------------------------
# Semantic clustering (HDBSCAN) — tables for diversity / core–tail analyses
# ---------------------------------------------------------------------
# Tuned on gender·3072d: epsilon merges persistent clusters so GenAI stays
# near-fully core while the more dispersed PhD group yields a larger tail.
HDBSCAN_CLUSTER_SELECTION_EPSILON_BY_EMBEDDING = {
    "raw_embedding_dimension_3072": 0.1365,
}

# ---------------------------------------------------------------------
# Statistical tables (CSV outputs saved beside figures)
# ---------------------------------------------------------------------
Q4_DIVERSITY_SUMMARY_CSV = "q4_group_diversity_summary{suffix}.csv"
Q4_DIVERSITY_PAIRWISE_CSV = "q4_group_diversity_pairwise{suffix}.csv"
SEMANTIC_CLUSTERING_SUMMARY_CSV = "semantic_clustering_summary_by_group{suffix}.csv"
SEMANTIC_CLUSTERING_PARTICIPANT_CSV = "semantic_clustering_by_participant{suffix}.csv"

# ---------------------------------------------------------------------
# Plot 07 — Cross-phase diversity predictions (batch parent folder)
# ---------------------------------------------------------------------
BATCH_VISUALIZATIONS_DIRNAME = ""  # figures live at embedding_analysis/ (network/, outputs/)
COMPARISONS_PRE_POST_SUBDIR = "outputs"
COMPARISONS_DIVERSITY_SUBDIR = "diversity"  # legacy; outputs now under within_group_variability/
COMPARISONS_WITHIN_GROUP_VAR_SUBDIR = "within_group_variability"
COMPARISONS_CORE_TAIL_SUBDIR = "core_tail_structure"
COMPARISONS_SELF_SUBDIR = "theory_revision"
PHASE_NAMES = ("pre-ML", "post-ML")

PHASE_GRID_FIGSIZE = (11.2, 12.4)
# Pre|Post 2×4 semantic maps (outputs/semantic_map_PCA).
PRE_POST_SEMANTIC_MAP_FIGSIZE = (29.5, 16.5)
PRE_POST_SEMANTIC_MAP_HSPACE = 0.30
PRE_POST_SEMANTIC_MAP_WSPACE = 0.10
PRE_POST_SEMANTIC_MAP_LEFT = 0.048
PRE_POST_SEMANTIC_MAP_RIGHT = 0.988
PRE_POST_SEMANTIC_MAP_BOTTOM = 0.068
PRE_POST_SEMANTIC_MAP_PAD_INCHES = 0.10
PRE_POST_SEMANTIC_MAP_LEGEND_TITLE_GAP = 0.018
PRE_POST_SEMANTIC_MAP_SCATTER_SIZE = 160
PRE_POST_SEMANTIC_MAP_PRE_SIZE = 150
PRE_POST_SEMANTIC_MAP_POST_SIZE = 170
PRE_POST_SEMANTIC_MAP_PRE_EDGEWIDTH = 1.35
PRE_POST_SEMANTIC_MAP_POST_EDGEWIDTH = 0.55
PRE_POST_SEMANTIC_MAP_LINE_ALPHA = 0.40
PRE_POST_SEMANTIC_MAP_LINEWIDTH = 0.90
PRE_POST_SEMANTIC_MAP_SPINE_WIDTH = 1.8
PRE_POST_SEMANTIC_MAP_LEGEND_FONTSIZE = VIZ_LEGEND_FONTSIZE + 8
PRE_POST_SEMANTIC_MAP_PANEL_TITLE_FONTSIZE = VIZ_PANEL_TITLE_FONTSIZE + 6
PRE_POST_SEMANTIC_MAP_AXIS_FONTSIZE = VIZ_AXIS_LABEL_FONTSIZE + 8
PRE_POST_SEMANTIC_MAP_TICK_FONTSIZE = VIZ_TICK_FONTSIZE + 6
PRE_POST_SEMANTIC_MAP_YLABEL_X = 0.004
PHASE_GRID_ROW_GAP = 0.36
PHASE_GRID_COL_GAP = 0.28
PHASE_GRID_SUPTITLE_FONTSIZE = VIZ_SUPTITLE_FONTSIZE
PHASE_GRID_SUPTITLE_Y = 0.975
PHASE_GRID_LEGEND_Y = 0.905
PHASE_GRID_LEGEND_FONTSIZE = VIZ_LEGEND_FONTSIZE
PHASE_GRID_PANEL_TOP = 0.82
PHASE_GRID_PANEL_TITLE_FONTSIZE = VIZ_PANEL_TITLE_FONTSIZE
PHASE_GRID_AXIS_FONTSIZE = VIZ_AXIS_LABEL_FONTSIZE
PHASE_GRID_TICK_FONTSIZE = VIZ_TICK_FONTSIZE
PHASE_GRID_SCATTER_SIZE = 78
PHASE_GRID_SEMANTIC_MAP_BOX_ASPECT = 1.0
PHASE_GRID_MAP_AXIS_MIN = -0.5
PHASE_GRID_MAP_AXIS_MAX = 0.5
PHASE_GRID_SEMANTIC_MAP_SUPXLABEL_Y = 0.028
PHASE_GRID_SEMANTIC_MAP_BOTTOM_EXTRA = 0.018
PHASE_GRID_SUPYLABEL_X = VIZ_SUPYLABEL_X
PHASE_GRID_FOOTNOTE_FONTSIZE = VIZ_FOOTNOTE_FONTSIZE
PHASE_GRID_FOOTNOTE_Y = VIZ_FOOTNOTE_Y
PHASE_GRID_FOOTNOTE_LINE_STEP = VIZ_FOOTNOTE_LINE_STEP
PHASE_GRID_BOX_FOOTNOTE_Y = 0.030
PHASE_GRID_BOX_FOOTNOTE_LINE_STEP = 0.030
PHASE_GRID_BOX_FOOTNOTE_BOTTOM_EXTRA = 0.014
PHASE_GRID_BOX_LEGEND_GAP = 0.085
PHASE_GRID_PANEL_COMPARISON_FONTSIZE = 11.5
PHASE_GRID_PANEL_COMPARISON_XY = (0.97, 0.97)
PHASE_GRID_PANEL_COMPARISON_LINE_STEP = 0.082
PHASE_GRID_PANEL_COMPARISON_BOX_PAD = 0.016
PHASE_GRID_COMPARISON_LABEL_SHORT = (
    ("PhD Students", "PhD"),
    ("Senior Scientists", "Senior Scientists"),
    ("GenAI", "GenAI"),
    ("Humans", "Humans"),
    ("Human", "Humans"),
)
PHASE_GRID_SEMANTIC_MAP_TITLE = (
    "Semantic distributions of theoretical explanations in embedding space (2D PCA)"
)
PHASE_GRID_SEMANTIC_MAP_METRIC = (
    "2D visualization using PCA (one point per theoretical explanation; "
    "Pre/Post and all groups share one pooled PCA per task; "
    "Post panels overlay pre-post trajectories)",
)
PHASE_GRID_CENTROID_BOX_METRIC = (
    "Metric: cosine distance from each respondent's embedding to its group centroid",
)
PHASE_GRID_CENTROID_DENSITY_METRIC = (
    "Metric: distribution of pairwise cosine distance within each group",
)
DIVERSITY_PREDICTION_FILENAME = "within_group_diversity_pre_post_predictions.png"
DIVERSITY_PREDICTION_CSV = "within_group_diversity_pre_post_predictions.csv"
HUMAN_VS_GENAI_PRE_POST_FILENAME = "human_vs_genai_pre_post_by_task.png"
HUMAN_VS_GENAI_PRE_POST_CSV = "human_vs_genai_pre_post_by_task.csv"
HUMAN_COLLAPSED_GROUP = "Human"
GENAI_COLLAPSED_GROUP = "GenAI"

DIVERSITY_TASK_PANEL_ORDER = [
    "race/main-effects",
    "race/soi",
    "gender/main-effects",
    "gender/soi",
]

DIVERSITY_PREDICTION_FOOTNOTE = (
    "Error bars: bootstrap 95% CI.",
    "One-sided Welch t-test (directional: Humans > GenAI).",
    SIG_LEVEL_LEGEND,
)
DIVERSITY_PREDICTION_PAIRWISE_FOOTNOTE = (
    "Error bars: bootstrap 95% CI.",
    "One-sided permutation test on group mean pairwise distance "
    "(directional: Humans > GenAI).",
    SIG_LEVEL_LEGEND,
)
DIVERSITY_PRED_SUPTITLE = (
    "Within-group variability of theoretical explanations (Humans vs GenAI)"
)
DIVERSITY_PRED_METRIC_SUBTITLE = (
    "Metric: mean(cosine distance to group centroid), higher = more dispersed"
)
DIVERSITY_PRED_PAIRWISE_METRIC_SUBTITLE = (
    "Metric: mean pairwise cosine distance among group members, higher = more dispersed"
)

DIVERSITY_PRED_FIGSIZE = (11.2, 12.4)
DIVERSITY_PRED_SUPTITLE_FONTSIZE = VIZ_SUPTITLE_FONTSIZE
DIVERSITY_PRED_PANEL_TITLE_FONTSIZE = VIZ_PANEL_TITLE_FONTSIZE
DIVERSITY_PRED_XTICK_FONTSIZE = VIZ_TICK_FONTSIZE
DIVERSITY_PRED_YTICK_FONTSIZE = VIZ_TICK_FONTSIZE
DIVERSITY_PRED_YLABEL_FONTSIZE = VIZ_AXIS_LABEL_FONTSIZE
DIVERSITY_PRED_YLABEL_X = VIZ_SUPYLABEL_X
DIVERSITY_PRED_FOOTNOTE_FONTSIZE = VIZ_FOOTNOTE_FONTSIZE
DIVERSITY_PRED_FOOTNOTE_Y = VIZ_FOOTNOTE_Y
DIVERSITY_PRED_FOOTNOTE_LINE_STEP = VIZ_FOOTNOTE_LINE_STEP
DIVERSITY_PRED_BRACKET_FONTSIZE = VIZ_BRACKET_FONTSIZE
DIVERSITY_PRED_ROW_GAP = 0.34
DIVERSITY_PRED_COL_GAP = 0.28
DIVERSITY_PRED_BOX_ASPECT = 0.92
DIVERSITY_PRED_BAR_WIDTH = 0.52
DIVERSITY_PRED_PRE_X = np.array([0.0, 1.0])
DIVERSITY_PRED_POST_X = np.array([2.75, 3.75])
DIVERSITY_PRED_X_MARGIN = 0.42
DIVERSITY_PRED_YLIM_TOP_PAD = 1.22

WITHIN_GROUP_VAR_FILENAME = "within_group_variability_pre_post_by_task.png"
WITHIN_GROUP_VAR_CSV = "within_group_variability_pre_post_by_task.csv"
WITHIN_GROUP_VAR_SUPTITLE = (
    "Within-group variability of theoretical explanations\n"
    "(pre-ML v.s. post-ML)"
)
WITHIN_GROUP_VAR_METRIC_SUBTITLE = (
    "Metric: mean(cosine distance to within-group centroid), higher = more dispersed"
)
WITHIN_GROUP_VAR_FOOTNOTE = (
    "Error bars: bootstrap 95% CI.",
    "One-sided paired t-test on group mean distance (directional: post < pre).",
    SIG_LEVEL_LEGEND,
)
WITHIN_GROUP_VAR_PRE_COLOR = "#4C72B0"
WITHIN_GROUP_VAR_POST_COLOR = "#D0D0D0"
WITHIN_GROUP_VAR_BAR_WIDTH = 0.50
WITHIN_GROUP_VAR_GROUP_X = {
    "student": np.array([0.0, 0.55]),
    "senior": np.array([2.4, 2.95]),
    "GenAI": np.array([4.8, 5.35]),
}
WITHIN_GROUP_VAR_COLLAPSED_FILENAME = (
    "within_group_variability_pre_post_by_task_collapsed.png"
)
WITHIN_GROUP_VAR_COLLAPSED_CSV = (
    "within_group_variability_pre_post_by_task_collapsed.csv"
)
WITHIN_GROUP_VAR_COLLAPSED_SUPTITLE = WITHIN_GROUP_VAR_SUPTITLE
WITHIN_GROUP_VAR_COLLAPSED_GROUP_X = {
    "Human": np.array([0.0, 0.55]),
    "GenAI": np.array([2.4, 2.95]),
}
WITHIN_GROUP_VAR_YTICK_MIN = 0.05
WITHIN_GROUP_VAR_YTICK_STEP = 0.05
WITHIN_GROUP_VAR_LEGEND_Y_SHIFT = 0.010
WITHIN_GROUP_VAR_X_MARGIN = 0.35
COMPARISON_FOOTNOTE_BOTTOM_PAD = 0.022
COMPARISON_FOOTNOTE_LINE_HEIGHT = 0.018
COMPARISON_FOOTNOTE_XLABEL_GAP = 0.010
COMPARISON_FOOTNOTE_XLABEL_HEIGHT = 0.030
WITHIN_GROUP_VAR_CENTROID_SUBDIR = "centroid_distance"
WITHIN_GROUP_VAR_PAIRWISE_SUBDIR = "mean_pairwise_cosine_distance"
WITHIN_GROUP_VAR_PAIRWISE_METRIC_SUBTITLE = (
    "Metric: mean pairwise cosine distance among group members, higher = more dispersed"
)
WITHIN_GROUP_VAR_PAIRWISE_FOOTNOTE = (
    "Error bars: bootstrap 95% CI.",
    "One-sided paired permutation test on group mean pairwise distance "
    "(directional: post < pre).",
    SIG_LEVEL_LEGEND,
)
WITHIN_GROUP_VAR_PAIRWISE_YLABEL = "Mean pairwise cosine distance"

apply_plot_style()


def resolve_input_parquet(embedding_dir: str | None = None) -> Path:
    """Resolve embeddings_wide.parquet from a full embedding-set folder path."""
    sets = discover_embedding_set_dirs(embedding_dir)
    if len(sets) != 1:
        raise ValueError(
            f"Expected one embedding set, found {len(sets)} under {sets[0].parent if sets else embedding_dir}. "
            "Use discover_embedding_set_dirs() for batch processing."
        )
    return sets[0] / "embeddings_wide.parquet"


def discover_embedding_set_dirs(embedding_dir: str | None = None) -> List[Path]:
    """Return embedding-set folders that contain embeddings_wide.parquet."""
    folder = DEFAULT_EMBEDDING_DIR if embedding_dir is None else Path(embedding_dir).expanduser()
    if not folder.is_absolute():
        folder = Path.cwd() / folder

    if not folder.exists():
        raise FileNotFoundError(f"Embedding path does not exist: {folder}")

    direct = folder / "embeddings_wide.parquet"
    if direct.exists():
        return [folder]

    set_dirs = sorted({path.parent for path in folder.rglob("embeddings_wide.parquet")})
    if not set_dirs:
        raise FileNotFoundError(
            f"No embeddings_wide.parquet found under {folder}. "
            "Pass --embedding-set as a leaf folder or a parent directory to batch all sets."
        )
    return set_dirs


def resolve_output_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path | None = None,
) -> Path:
    """Central clustering CSV dir: the embedding set folder itself."""
    root = embeddings_root or infer_embeddings_root(embedding_set_dir)
    return resolve_task_data_dir(root, embedding_set_dir, embedding_col)


def available_embedding_columns(df: pd.DataFrame, requested: List[str]) -> List[str]:
    """Keep requested embedding columns that exist in the parquet."""
    missing = [c for c in requested if c not in df.columns]
    if missing:
        print(f"Skipping missing embedding columns: {missing}")
    present = [c for c in requested if c in df.columns]
    if not present:
        raise ValueError(
            "No requested embedding columns found in parquet. "
            f"Available columns: {list(df.columns)}"
        )
    return present


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", x).strip("_")


def parse_embedding_cell(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, list):
        return np.asarray(x, dtype=float)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=float)

    raise TypeError(f"Unsupported embedding cell type: {type(x)}")


def stack_embeddings(df: pd.DataFrame, embedding_col: str) -> np.ndarray:
    vectors = [parse_embedding_cell(x) for x in df[embedding_col]]
    lengths = [len(v) for v in vectors]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Inconsistent vector lengths in {embedding_col}: "
            f"{pd.Series(lengths).value_counts().to_dict()}"
        )

    X = np.vstack(vectors).astype(float)

    if np.isnan(X).any():
        raise ValueError(f"NaN found in embedding column: {embedding_col}")

    return X


def ordered_groups(df: pd.DataFrame, group_col: str = PARTICIPANT_TYPE_COL) -> List[str]:
    group_order = (
        GROUP_ORDER_COLLAPSED
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL
        else GROUP_ORDER
    )
    existing = list(df[group_col].dropna().unique())
    ordered = [g for g in group_order if g in existing]
    ordered += [g for g in existing if g not in ordered]
    return ordered


def with_collapsed_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COLLAPSED_PARTICIPANT_TYPE_COL] = out[PARTICIPANT_TYPE_COL].map(
        PARTICIPANT_TYPE_TO_COLLAPSED
    )
    return out


def upper_triangle_values(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def mean_pairwise_cosine_distance_normalized(Xn: np.ndarray) -> float:
    """
    Mean pairwise cosine distance for L2-normalized rows.

    Uses Gram matrix: for unit vectors, mean distance = 1 - mean cosine similarity.
    """
    Xn = np.asarray(Xn, dtype=np.float64)
    n = Xn.shape[0]
    if n < 2:
        return np.nan
    gram = Xn @ Xn.T
    mean_sim = (gram.sum() - n) / (n * (n - 1))
    return float(1.0 - mean_sim)


def normalize_embedding_rows(X: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(X, dtype=np.float64))


def mean_pairwise_cosine_distance(X: np.ndarray) -> float:
    """Mean cosine distance over all unordered pairs in X (rows = embeddings)."""
    Xn = normalize_embedding_rows(X)
    n = len(Xn)
    if n < 2:
        return np.nan
    return mean_pairwise_cosine_distance_normalized(Xn)


def bootstrap_mpwd_ci(
    X: np.ndarray,
    *,
    n_boot: int = MPWD_N_BOOT,
    alpha: float = 0.05,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for mean pairwise cosine distance (row resampling)."""
    Xn = normalize_embedding_rows(X)
    n = len(Xn)
    if n < 2:
        v = mean_pairwise_cosine_distance_normalized(Xn) if n == 1 else np.nan
        return v, v
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = mean_pairwise_cosine_distance_normalized(Xn[rng.integers(0, n, size=n)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def _pooled_mpwd_from_normalized_mats(mats: list[np.ndarray]) -> float:
    """Pair-count-weighted mean of within-matrix mean pairwise distances."""
    num = 0.0
    den = 0
    for Xn in mats:
        n = len(Xn)
        if n < 2:
            continue
        n_pairs = n * (n - 1) // 2
        num += float(mean_pairwise_cosine_distance_normalized(Xn)) * n_pairs
        den += n_pairs
    if den == 0:
        return np.nan
    return num / den


def bootstrap_pooled_mpwd_ci(
    Xs: list[np.ndarray],
    *,
    n_boot: int = MPWD_N_BOOT,
    alpha: float = 0.05,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for pooled MPWD via within-task participant resampling.

    Pairwise distances are not independent, so bootstrapping the bag of all pairs
    understates uncertainty. Resample embedding rows within each task matrix, then
    recompute the pair-count-weighted pooled mean pairwise distance.
    """
    mats = [normalize_embedding_rows(X) for X in Xs if len(X) >= 2]
    if not mats:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot_mats = [
            Xn[rng.integers(0, len(Xn), size=len(Xn))]
            for Xn in mats
        ]
        boots[i] = _pooled_mpwd_from_normalized_mats(boot_mats)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def p_value_pooled_paired_permutation_mpwd_post_lt_pre(
    pre_Xs: list[np.ndarray],
    post_Xs: list[np.ndarray],
    *,
    n_perm: int = MPWD_N_PERM,
    seed: int = ANALYSIS_SEED,
) -> float:
    """H1: pooled post MPWD < pooled pre MPWD (paired within-task label swaps)."""
    if len(pre_Xs) != len(post_Xs) or not pre_Xs:
        return np.nan
    pre_ns: list[np.ndarray] = []
    post_ns: list[np.ndarray] = []
    for X_pre, X_post in zip(pre_Xs, post_Xs):
        X_pre_n = normalize_embedding_rows(X_pre)
        X_post_n = normalize_embedding_rows(X_post)
        if len(X_pre_n) < 2 or len(X_pre_n) != len(X_post_n):
            return np.nan
        pre_ns.append(X_pre_n)
        post_ns.append(X_post_n)

    pre_obs = _pooled_mpwd_from_normalized_mats(pre_ns)
    post_obs = _pooled_mpwd_from_normalized_mats(post_ns)
    if not np.isfinite(pre_obs) or not np.isfinite(post_obs):
        return np.nan
    delta_obs = post_obs - pre_obs

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        pre_perm_mats: list[np.ndarray] = []
        post_perm_mats: list[np.ndarray] = []
        for X_pre_n, X_post_n in zip(pre_ns, post_ns):
            swap = rng.random(len(X_pre_n)) < 0.5
            pre_perm_mats.append(np.where(swap[:, None], X_post_n, X_pre_n))
            post_perm_mats.append(np.where(swap[:, None], X_pre_n, X_post_n))
        delta_perm = (
            _pooled_mpwd_from_normalized_mats(post_perm_mats)
            - _pooled_mpwd_from_normalized_mats(pre_perm_mats)
        )
        if delta_perm <= delta_obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def ci_errorbar_offsets(
    mean: float,
    ci_low: float,
    ci_high: float,
) -> tuple[float, float]:
    low = max(0.0, mean - ci_low) if np.isfinite(mean) and np.isfinite(ci_low) else 0.0
    high = max(0.0, ci_high - mean) if np.isfinite(mean) and np.isfinite(ci_high) else 0.0
    return low, high


def p_value_paired_permutation_mpwd_post_lt_pre(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    *,
    n_perm: int = MPWD_N_PERM,
    seed: int = ANALYSIS_SEED,
) -> float:
    """H1: post-phase mean pairwise distance < pre-phase (paired label swap)."""
    X_pre_n = normalize_embedding_rows(X_pre)
    X_post_n = normalize_embedding_rows(X_post)
    n = len(X_pre_n)
    if n < 2 or len(X_post_n) != n:
        return np.nan

    pre_obs = mean_pairwise_cosine_distance_normalized(X_pre_n)
    post_obs = mean_pairwise_cosine_distance_normalized(X_post_n)
    if not np.isfinite(pre_obs) or not np.isfinite(post_obs):
        return np.nan
    delta_obs = post_obs - pre_obs

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pre_perm = np.where(swap[:, None], X_post_n, X_pre_n)
        post_perm = np.where(swap[:, None], X_pre_n, X_post_n)
        delta_perm = (
            mean_pairwise_cosine_distance_normalized(post_perm)
            - mean_pairwise_cosine_distance_normalized(pre_perm)
        )
        if delta_perm <= delta_obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def p_value_permutation_mpwd_group_greater(
    X_left: np.ndarray,
    X_right: np.ndarray,
    *,
    n_perm: int = MPWD_N_PERM,
    seed: int = ANALYSIS_SEED,
) -> float:
    """H1: left group's MPWD > right group's MPWD (label permutation)."""
    X_left_n = normalize_embedding_rows(X_left)
    X_right_n = normalize_embedding_rows(X_right)
    n_left, n_right = len(X_left_n), len(X_right_n)
    if n_left < 2 or n_right < 2:
        return np.nan
    observed = (
        mean_pairwise_cosine_distance_normalized(X_left_n)
        - mean_pairwise_cosine_distance_normalized(X_right_n)
    )
    if not np.isfinite(observed):
        return np.nan
    combined = np.vstack([X_left_n, X_right_n])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(combined))
        diff = (
            mean_pairwise_cosine_distance_normalized(combined[perm[:n_left]])
            - mean_pairwise_cosine_distance_normalized(combined[perm[n_left:]])
        )
        if diff >= observed:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def group_centroid(Xg: np.ndarray) -> np.ndarray:
    """L2-normalized mean vector (centroid) of unit-norm embeddings."""
    c = Xg.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 1e-12:
        c = c / norm
    return c


def find_medoid(X: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Medoid = actual participant whose average cosine distance to others is smallest.
    Returns local medoid index and distance matrix.
    """
    D = cosine_distances(X)
    mean_d = D.mean(axis=1)
    medoid_idx = int(np.argmin(mean_d))
    return medoid_idx, D


def embedding_set_label(embedding_dir: Path) -> str:
    try:
        rel = embedding_dir.relative_to(DEFAULT_EMBEDDINGS_ROOT)
        parts = [format_embedding_set_part(part) for part in rel.parts]
        return " · ".join(parts)
    except ValueError:
        return format_embedding_set_part(embedding_dir.name)


def format_embedding_set_part(part: str) -> str:
    key = part.lower()
    if key in EMBEDDING_SET_PART_LABELS:
        return EMBEDDING_SET_PART_LABELS[key]
    label = part.replace("-", " ").title()
    return label.replace("Ml", "ML")


def semantic_map_main_title(projection_name: str, n_components: int) -> str:
    if projection_name == "PCA" and n_components == 2:
        return PHASE_GRID_SEMANTIC_MAP_TITLE
    return (
        f"Theoretical explanations in embedding space "
        f"({n_components}D {projection_name})"
    )


def add_theory_figure_titles(
    fig,
    plot_title: str,
    embedding_set_label_text: str,
    detail_line: str = "",
    *,
    title_y: float = 0.96,
    subtitle_y: float = 0.885,
    detail_on_new_line: bool = False,
    detail_y: float = 0.838,
) -> None:
    """Main title = what the figure shows; subtitle = dataset context (+ optional method)."""
    fig.text(
        0.5,
        title_y,
        plot_title,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=MAP_FONT_TITLE,
        fontweight="bold",
    )
    if detail_line and not detail_on_new_line:
        subtitle = f"{embedding_set_label_text}  ·  {detail_line}"
    else:
        subtitle = embedding_set_label_text
    fig.text(
        0.5,
        subtitle_y,
        subtitle,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=MAP_FONT_SUBTITLE,
        color="#333333",
    )
    if detail_line and detail_on_new_line:
        fig.text(
            0.5,
            detail_y,
            detail_line,
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=MAP_FONT_SUBTITLE,
            color="#333333",
        )


def semantic_map_filename(
    n_components: int,
    *,
    collapsed: bool = False,
) -> str:
    suffix = "_collapsed" if collapsed else ""
    if n_components == 3:
        return f"01_semantic_space_map_3d{suffix}.png"
    return f"01_semantic_space_map{suffix}.png"


def compute_projection(
    X: np.ndarray,
    method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    n_components: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Project high-dimensional embeddings to 2D or 3D with PCA for visualization.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")

    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
        coords = reducer.fit_transform(X)
        return coords, "UMAP"

    if method == "umap" and not HAS_UMAP:
        print("umap-learn not installed. Falling back to PCA.")

    coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return coords, "PCA"


def semantic_neighbor_degree(
    X: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float,
) -> Tuple[np.ndarray, float]:
    """
    Degree = number of same-group semantic neighbors above a global similarity threshold.

    A high-degree explanation is a candidate "core" explanation.
    Low-degree explanations form the semantic tail.
    """
    S = cosine_similarity(X)
    global_sims = upper_triangle_values(S)
    threshold = float(np.quantile(global_sims, threshold_quantile))

    degree = np.zeros(X.shape[0], dtype=int)

    for group in np.unique(labels):
        idx = np.where(labels == group)[0]
        Sg = S[np.ix_(idx, idx)]

        adj = Sg >= threshold
        np.fill_diagonal(adj, False)

        degree[idx] = adj.sum(axis=1)

    return degree, threshold


def make_point_metrics(
    df: pd.DataFrame,
    X: np.ndarray,
    degree: np.ndarray,
) -> pd.DataFrame:
    """
    Point-level metrics for labeling outliers / core points.
    """
    labels = df[PARTICIPANT_TYPE_COL].values
    names = df[PARTICIPANT_NAME_COL].values

    global_medoid_idx, D_global = find_medoid(X)
    dist_to_global_medoid = D_global[:, global_medoid_idx]

    collapsed_labels = np.array(
        [PARTICIPANT_TYPE_TO_COLLAPSED.get(g, g) for g in labels]
    )
    dist_to_collapsed_group_centroid = np.zeros(len(labels), dtype=float)
    for cgroup in GROUP_ORDER_COLLAPSED:
        cidx = np.where(collapsed_labels == cgroup)[0]
        if len(cidx) == 0:
            continue
        c_centroid = group_centroid(X[cidx])
        dist_to_collapsed_group_centroid[cidx] = cosine_distances(
            X[cidx], c_centroid.reshape(1, -1)
        ).ravel()

    rows = []

    for group in ordered_groups(df):
        idx = np.where(labels == group)[0]
        Xg = X[idx]
        local_medoid_idx, Dg = find_medoid(Xg)

        group_medoid_global_idx = idx[local_medoid_idx]
        dist_to_group_medoid = D_global[:, group_medoid_global_idx]

        centroid = group_centroid(Xg)
        dist_to_group_centroid = cosine_distances(X, centroid.reshape(1, -1)).ravel()

        Sg = cosine_similarity(Xg)
        centrality_local = (Sg.sum(axis=1) - 1) / (len(idx) - 1)

        for pos, global_i in enumerate(idx):
            rows.append({
                "participant_name": names[global_i],
                "participant_type": group,
                COLLAPSED_PARTICIPANT_TYPE_COL: collapsed_labels[global_i],
                "semantic_neighbor_degree": int(degree[global_i]),
                "distance_to_global_medoid": float(dist_to_global_medoid[global_i]),
                "distance_to_group_medoid": float(dist_to_group_medoid[global_i]),
                "distance_to_group_centroid": float(dist_to_group_centroid[global_i]),
                "distance_to_collapsed_group_centroid": float(
                    dist_to_collapsed_group_centroid[global_i]
                ),
                "within_group_centrality": float(centrality_local[pos]),
                "is_global_medoid": global_i == global_medoid_idx,
                "is_group_medoid": global_i == group_medoid_global_idx,
            })

    return pd.DataFrame(rows)


def semantic_map_2d_bounds(
    coords: np.ndarray,
    projection_name: str,
    *,
    pad_frac: float | None = None,
    pad_frac_right: float | None = None,
) -> dict[str, float]:
    """Compute padded 2D map axis bounds for one or more coordinate sets."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    if projection_name == "UMAP":
        x_pad_left = max(x_range * MAP_PAD_FRAC, MAP_UMAP_PAD_MIN)
        x_pad_right = max(x_range * MAP_PAD_FRAC_RIGHT, MAP_UMAP_PAD_RIGHT_MIN)
        y_pad = max(y_range * MAP_PAD_FRAC, MAP_UMAP_PAD_MIN)
    else:
        pf = MAP_PAD_FRAC if pad_frac is None else pad_frac
        pfr = MAP_PAD_FRAC_RIGHT_PCA if pad_frac_right is None else pad_frac_right
        x_pad_left = x_range * pf
        x_pad_right = x_range * pfr
        y_pad = y_range * pf

    return {
        "x_left": x_min - x_pad_left,
        "x_right": x_max + x_pad_right,
        "y_bottom": y_min - y_pad,
        "y_top": y_max + y_pad,
        "projection_name": projection_name,
    }


def square_axis_bounds(bounds: dict[str, float]) -> dict[str, float]:
    """Expand the shorter axis so x/y spans match (for equal-aspect square panels)."""
    x_left = bounds["x_left"]
    x_right = bounds["x_right"]
    y_bottom = bounds["y_bottom"]
    y_top = bounds["y_top"]
    x_center = (x_left + x_right) / 2.0
    y_center = (y_bottom + y_top) / 2.0
    half_span = max((x_right - x_left) / 2.0, (y_top - y_bottom) / 2.0)
    return {
        **bounds,
        "x_left": x_center - half_span,
        "x_right": x_center + half_span,
        "y_bottom": y_center - half_span,
        "y_top": y_center + half_span,
    }


def apply_semantic_map_2d_bounds(
    ax,
    bounds: dict[str, float],
    *,
    box_aspect: float | None = None,
) -> dict[str, float]:
    projection_name = bounds.get("projection_name", "PCA")
    x_left = bounds["x_left"]
    x_right = bounds["x_right"]
    y_bottom = bounds["y_bottom"]
    y_top = bounds["y_top"]
    x_range = x_right - x_left

    if projection_name == "UMAP":
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    else:
        tick_step = 0.2 if x_range <= 1.5 else 0.5
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.set_aspect("equal", adjustable="datalim")
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    ax.autoscale(False)
    ax.margins(0)
    return bounds


def apply_semantic_map_2d_limits(
    ax,
    coords: np.ndarray,
    projection_name: str,
) -> dict[str, float]:
    """Pad axes for legend room. Returns axis bounds for legend placement."""
    bounds = semantic_map_2d_bounds(coords, projection_name)
    return apply_semantic_map_2d_bounds(ax, bounds)


def plot_semantic_map(
    df: pd.DataFrame,
    coords: np.ndarray,
    projection_name: str,
    embedding_name: str,
    embedding_dim: int,
    embedding_set_label_text: str,
    outpath: str,
    label_points: bool,
    n_components: int = 2,
    collapse_human: bool = False,
) -> None:
    """2D or 3D semantic map of the explanation embedding space."""
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    groups = ordered_groups(plot_df, group_col)
    group_counts = plot_df[group_col].value_counts().to_dict()

    if n_components == 3:
        fig = plt.figure(figsize=MAP_FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=MAP_FIGSIZE_2D)

    for group in groups:
        mask = plot_df[group_col].values == group
        color = colors_map.get(group, "#888888")
        if n_components == 3:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=color,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=BAR_EDGE_WIDTH,
                depthshade=False,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=BAR_EDGE_WIDTH,
            )

    if label_points:
        names = plot_df[PARTICIPANT_NAME_COL].values
        for i, name in enumerate(names):
            if n_components == 3:
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    coords[i, 2],
                    str(name),
                    fontsize=7,
                    alpha=0.85,
                )
            else:
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    str(name),
                    fontsize=7,
                    alpha=0.85,
                )

    map_title = semantic_map_main_title(projection_name, n_components)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.82)
    add_theory_figure_titles(
        fig,
        map_title,
        embedding_set_label_text,
        title_y=0.94,
        subtitle_y=0.875,
    )

    axis_label = f"{projection_name} dimension"
    x_label, y_label = f"{axis_label} 1", f"{axis_label} 2"
    z_label = f"{axis_label} 3"
    if n_components == 3:
        ax.set_xlabel(x_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_ylabel(y_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_zlabel(z_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.tick_params(axis="x", labelsize=MAP_FONT_TICK)
        ax.tick_params(axis="y", labelsize=MAP_FONT_TICK)
        ax.tick_params(axis="z", labelsize=MAP_FONT_TICK)
    else:
        ax.set_xlabel(x_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        ax.set_ylabel(y_label, fontsize=MAP_FONT_AXIS, fontweight="bold", labelpad=12)
        style_axes(ax)
        ax.tick_params(axis="both", labelsize=MAP_FONT_TICK)
        ax.grid(alpha=0.2, zorder=0)
        apply_semantic_map_2d_limits(ax, coords, projection_name)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors_map[group],
            markeredgecolor=BAR_EDGE_COLOR,
            markeredgewidth=BAR_EDGE_WIDTH,
            markersize=8,
        )
        for group in groups
    ]
    if collapse_human:
        legend_labels = collapsed_legend_labels(
            groups,
            {group: int(group_counts.get(group, 0)) for group in groups},
        )
    else:
        legend_labels = [
            legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(group, group),
                int(group_counts.get(group, 0)),
            )
            for group in groups
        ]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        framealpha=1.0,
        fontsize=VIZ_LEGEND_FONTSIZE,
    )

    save_figure_pdf_svg(fig, outpath)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    n1, n2 = len(arr_a), len(arr_b)
    var1, var2 = np.var(arr_a, ddof=1), np.var(arr_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return np.nan
    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled)


def p_value_permutation_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    observed = float(np.mean(arr_a) - np.mean(arr_b))
    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def comparison_pairs_for_groups(
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> List[Tuple[str, str, str]]:
    """Return (left_group_key, right_group_key, display_label) for pairwise tests."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL and {"Human", "GenAI"} <= set(groups):
        return [
            ("Human", "GenAI", comparison_pair_label("Human", "GenAI")),
        ]

    if set(groups) >= {"student", "senior", "GenAI"}:
        return [
            ("senior", "student", comparison_pair_label("Senior Scientists", "PhD Students")),
            ("student", "GenAI", comparison_pair_label("PhD Students", "GenAI")),
            ("senior", "GenAI", comparison_pair_label("Senior Scientists", "GenAI")),
        ]

    pairs: List[Tuple[str, str, str]] = []
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            left_label = PARTICIPANT_TYPE_TO_LEGEND.get(left, left)
            right_label = PARTICIPANT_TYPE_TO_LEGEND.get(right, right)
            pairs.append(
                (left, right, comparison_pair_label(left_label, right_label))
            )
    return pairs


def group_metric_values(
    data: pd.DataFrame,
    group_col: str,
    group: str,
    value_col: str,
) -> np.ndarray:
    vals = data.loc[data[group_col] == group, value_col].values.astype(float)
    return vals[~np.isnan(vals)]


def build_pairwise_inference_rows(
    data: pd.DataFrame,
    value_col: str,
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    metric_name: str,
    seed: int = 42,
) -> List[dict]:
    rows: List[dict] = []
    for left, right, label in comparison_pairs_for_groups(groups, group_col=group_col):
        vals_left = group_metric_values(data, group_col, left, value_col)
        vals_right = group_metric_values(data, group_col, right, value_col)
        welch_p = p_value_welch_ttest(vals_left, vals_right)
        perm_p = p_value_permutation_mean_diff(
            vals_left, vals_right, seed=seed
        )
        rows.append({
            "metric": metric_name,
            "comparison": label,
            "left_group": left,
            "right_group": right,
            "left_mean": float(np.mean(vals_left)) if len(vals_left) else np.nan,
            "right_mean": float(np.mean(vals_right)) if len(vals_right) else np.nan,
            "mean_diff_left_minus_right": (
                float(np.mean(vals_left) - np.mean(vals_right))
                if len(vals_left) and len(vals_right)
                else np.nan
            ),
            "welch_pvalue": welch_p,
            "permutation_pvalue": perm_p,
            "cohens_d": cohens_d(vals_left, vals_right),
            "significance": significance_label(welch_p),
        })
    return rows


def format_diversity_comparison_line(
    comparison: str,
    welch_p: float,
    cohens_d_val: float = np.nan,
    *,
    include_cohens_d: bool = True,
) -> str:
    sig = significance_label(welch_p)
    d_suffix = ""
    if include_cohens_d and np.isfinite(cohens_d_val):
        d_suffix = f", d={cohens_d_val:.2f}"
    return f"{comparison}: {sig}{d_suffix}"


def comparisons_for_metric(
    pairwise_df: pd.DataFrame,
    metric: str,
    *,
    include_cohens_d: bool = True,
) -> List[Tuple[str, float]]:
    if pairwise_df is None or pairwise_df.empty:
        return []
    rows = pairwise_df.loc[pairwise_df["metric"] == metric]
    comparisons: List[Tuple[str, float]] = []
    for _, row in rows.iterrows():
        comparisons.append(
            (str(row["comparison"]), float(row["welch_pvalue"]))
        )
    return comparisons


def batch_visualizations_root(embeddings_root: Path) -> Path:
    """Figure/CSV root: embedding_analysis/ (sibling of embeddings root)."""
    return embeddings_root.parent


def comparisons_pre_post_dir(embeddings_root: Path, subdir: str) -> Path:
    return (
        batch_visualizations_root(embeddings_root)
        / COMPARISONS_PRE_POST_SUBDIR
        / subdir
    )


def batch_phase_dir(embeddings_root: Path, phase: str) -> Path:
    return batch_visualizations_root(embeddings_root) / phase


DATA_SUBDIR = "semantic_clustering"  # legacy name; CSVs now live in each embedding set dir
NETWORK_SUBDIR = "network"


def infer_embeddings_root(embedding_set_dir: Path) -> Path:
    if embedding_set_dir.name not in PHASE_NAMES:
        raise ValueError(
            f"Cannot infer embeddings root from {embedding_set_dir}; "
            f"expected phase folder name in {PHASE_NAMES}."
        )
    return embedding_set_dir.parent.parent.parent


def task_phase_slug(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> str:
    task_key = embedding_set_dir.relative_to(embeddings_root).parent
    phase = embedding_set_dir.name
    task_slug = str(task_key).replace("/", "_")
    return f"{phase}__{task_slug}__{safe_name(embedding_col)}"


def resolve_task_data_dir(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> Path:
    """Write/read clustering CSVs beside the set's embeddings_wide.parquet."""
    del embeddings_root, embedding_col  # one embedding col per set in current pipeline
    return Path(embedding_set_dir)


def resolve_network_dir(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
) -> Path:
    task_key = embedding_set_dir.relative_to(embeddings_root).parent
    phase = embedding_set_dir.name
    outdir = (
        batch_visualizations_root(embeddings_root)
        / NETWORK_SUBDIR
        / task_key
        / phase
        / safe_name(embedding_col)
    )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def resolve_network_outpath(
    embeddings_root: Path,
    embedding_set_dir: Path,
    embedding_col: str,
    *,
    collapsed: bool = False,
) -> Path:
    filename = (
        THRESHOLD_NETWORK_COLLAPSED_FILENAME
        if collapsed
        else THRESHOLD_NETWORK_FILENAME
    )
    return resolve_network_dir(embeddings_root, embedding_set_dir, embedding_col) / filename


def figure_comparison_footer_layout(
    n_comp_lines: int,
    *,
    footnote_lines: Tuple[str, ...] = DIVERSITY_SIG_FOOTNOTE,
) -> Tuple[float, float, float]:
    return centroid_distribution_bottom_layout(
        n_comp_lines,
        n_footnote_lines=len(footnote_lines),
    )


def draw_figure_footnote_lines(
    fig,
    y: float,
    lines: Tuple[str, ...],
    *,
    line_step: float = VIZ_FOOTNOTE_LINE_STEP,
    fontsize: float = VIZ_FOOTNOTE_FONTSIZE,
) -> None:
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            y - i * line_step,
            line,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
            zorder=1,
        )


def attach_centered_comparisons(
    fig,
    ax,
    comparisons: List[Tuple[str, float]],
    *,
    comp_bottom: float,
    footnote_y: float,
    footnote_lines: Tuple[str, ...] = DIVERSITY_SIG_FOOTNOTE,
    max_width_frac: float = BOX_COMP_MAX_WIDTH_FRAC,
    center_x: float | None = None,
    footnote_line_step: float = CENTROID_FOOTNOTE_LINE_STEP,
) -> None:
    if not comparisons:
        return
    fig.canvas.draw()
    if center_x is None:
        bbox = ax.get_position()
        panel_w = bbox.x1 - bbox.x0
        center_x = (bbox.x0 + bbox.x1) / 2
        max_width = panel_w * max_width_frac
    else:
        max_width = max_width_frac
    draw_centered_comparison_box(
        fig,
        comparisons,
        center_x=center_x,
        box_bottom=comp_bottom,
        min_box_width=0.0,
        max_box_width=max_width,
    )
    draw_figure_footnote_lines(
        fig, footnote_y, footnote_lines, line_step=footnote_line_step
    )


def diversity_summary_for_group(
    diversity_summary_df: pd.DataFrame | None,
    group: str,
) -> dict | None:
    if diversity_summary_df is None or diversity_summary_df.empty:
        return None
    sub = diversity_summary_df.loc[diversity_summary_df["group"] == group]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def welch_comparisons_for_distance(
    point_metrics: pd.DataFrame,
    distance_col: str,
    groups: List[str],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """Pairwise Welch t-tests aligned with main_effects quant panels."""
    rows = build_pairwise_inference_rows(
        point_metrics,
        distance_col,
        groups,
        group_col=group_col,
        metric_name="centroid_distance",
        seed=seed,
    )
    return [(row["comparison"], row["welch_pvalue"]) for row in rows]


def histogram_bin_count(n: int) -> int:
    return max(5, min(12, int(np.sqrt(n))))


def pairwise_histogram_bin_count(n_pairs: int) -> int:
    return max(8, min(30, int(np.sqrt(n_pairs))))


def dagostino_skew_test(vals: np.ndarray) -> Tuple[float, float]:
    """D'Agostino K² test: H0 population skewness is zero."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 8:
        return np.nan, np.nan
    stat, p = skewtest(arr, nan_policy="omit")
    return float(stat), float(p)


def compute_centroid_skewness_stats(
    point_metrics: pd.DataFrame,
    distance_col: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> pd.DataFrame:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    groups = [
        g for g in preferred_order if g in point_metrics[group_col].unique()
    ]
    groups += [g for g in point_metrics[group_col].unique() if g not in groups]

    rows = []
    for group in groups:
        vals = point_metrics.loc[
            point_metrics[group_col] == group, distance_col
        ].values.astype(float)
        vals = vals[~np.isnan(vals)]
        dag_stat, dag_p = dagostino_skew_test(vals)
        rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(vals),
            "skewness": float(skew(vals, bias=False)) if len(vals) >= 3 else np.nan,
            "dagostino_statistic": dag_stat,
            "dagostino_pvalue": dag_p,
            "dagostino_significance": significance_label(dag_p),
        })
    return pd.DataFrame(rows)


def pairwise_cosine_distance_group_series(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> List[Tuple[str, np.ndarray, int]]:
    """Return (group, all pairwise distances, n_members) per group."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        labels = plot_df[group_col].values
    else:
        labels = df[group_col].values

    groups = ordered_groups(
        plot_df if group_col == COLLAPSED_PARTICIPANT_TYPE_COL else df,
        group_col,
    )
    series: List[Tuple[str, np.ndarray, int]] = []
    for group in groups:
        idx = np.where(labels == group)[0]
        if len(idx) < 2:
            continue
        dists = upper_triangle_values(cosine_distances(X[idx]))
        series.append((group, dists.astype(float), len(idx)))
    return series


def compute_pairwise_distance_skewness_stats(
    group_series: List[Tuple[str, np.ndarray, int]],
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> pd.DataFrame:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        display_names_map = DISPLAY_LABELS
    else:
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    rows = []
    for group, vals, n_members in group_series:
        vals = vals[~np.isnan(vals)]
        dag_stat, dag_p = dagostino_skew_test(vals)
        rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": n_members,
            "n_pairs": len(vals),
            "skewness": float(skew(vals, bias=False)) if len(vals) >= 3 else np.nan,
            "dagostino_statistic": dag_stat,
            "dagostino_pvalue": dag_p,
            "dagostino_significance": significance_label(dag_p),
        })
    return pd.DataFrame(rows)


def prepare_pairwise_distance_plot_data(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> dict:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    group_series = pairwise_cosine_distance_group_series(df, X, group_col=group_col)
    skew_df = compute_pairwise_distance_skewness_stats(
        group_series, group_col=group_col
    )
    skew_by_group = skew_df.set_index("group")["skewness"].to_dict()

    return {
        "group_col": group_col,
        "color_map": color_map,
        "display_names_map": display_names_map,
        "group_series": group_series,
        "skew_df": skew_df,
        "skew_by_group": skew_by_group,
    }


def centroid_distribution_bottom_layout(
    n_comp_lines: int,
    *,
    n_footnote_lines: int = len(CENTROID_MEAN_SIG_FOOTNOTE),
) -> Tuple[float, float, float]:
    """Return (comparison_box_bottom, footnote_y, subplot_bottom)."""
    footnote_y = CENTROID_FOOTNOTE_BOTTOM_Y + CENTROID_FOOTNOTE_LINE_STEP * max(
        n_footnote_lines - 1, 0
    )
    footnote_top = footnote_y + CENTROID_FOOTNOTE_TEXT_HEIGHT
    comp_bottom = footnote_top + CENTROID_FOOTNOTE_TO_COMP_GAP
    comp_height = comparison_box_height(n_comp_lines) if n_comp_lines else 0.0
    subplot_bottom = (
        comp_bottom + comp_height + CENTROID_AXIS_GAP if n_comp_lines else 0.14
    )
    return comp_bottom, footnote_y, subplot_bottom


def prepare_centroid_distance_plot_data(
    point_metrics: pd.DataFrame,
    distance_col: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    diversity_summary_df: pd.DataFrame | None = None,
    diversity_pairwise_df: pd.DataFrame | None = None,
) -> dict:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        preferred_order = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        preferred_order = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    groups = [
        g for g in preferred_order if g in point_metrics[group_col].unique()
    ]
    groups += [g for g in point_metrics[group_col].unique() if g not in groups]

    group_series: List[Tuple[str, np.ndarray]] = []
    for group in groups:
        vals = point_metrics.loc[
            point_metrics[group_col] == group, distance_col
        ].values.astype(float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        group_series.append((group, vals))

    skew_df = compute_centroid_skewness_stats(
        point_metrics, distance_col, group_col=group_col
    )
    skew_by_group = skew_df.set_index("group")["skewness"].to_dict()

    if diversity_pairwise_df is not None:
        comparisons = comparisons_for_metric(
            diversity_pairwise_df, "centroid_distance", include_cohens_d=False
        )
        inference_rows = diversity_pairwise_df.loc[
            diversity_pairwise_df["metric"] == "centroid_distance"
        ].to_dict("records")
    else:
        inference_rows = build_pairwise_inference_rows(
            point_metrics,
            distance_col,
            groups,
            group_col=group_col,
            metric_name="centroid_distance",
        )
        comparisons = [
            (str(row["comparison"]), float(row["welch_pvalue"]))
            for row in inference_rows
        ]

    mean_ci_by_group: Dict[str, Tuple[float, float, float]] = {}
    for group, vals in group_series:
        mean_dist = float(np.mean(vals))
        div_row = diversity_summary_for_group(diversity_summary_df, group)
        if div_row is not None:
            ci_lo = float(div_row["centroid_distance_ci_low"])
            ci_hi = float(div_row["centroid_distance_ci_high"])
        else:
            ci_lo, ci_hi = bootstrap_mean_ci(vals)
        mean_ci_by_group[group] = (mean_dist, ci_lo, ci_hi)

    return {
        "group_col": group_col,
        "color_map": color_map,
        "display_names_map": display_names_map,
        "group_series": group_series,
        "skew_df": skew_df,
        "skew_by_group": skew_by_group,
        "comparisons": comparisons,
        "inference_rows": inference_rows,
        "mean_ci_by_group": mean_ci_by_group,
    }


def plot_pairwise_distance_distribution(
    plot_data: dict,
    outpath: str,
    embedding_set_label_text: str,
) -> None:
    """Overlapping density histograms of all within-group pairwise cosine distances."""
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    skew_by_group = plot_data["skew_by_group"]
    group_series = plot_data["group_series"]

    fig, ax = plt.subplots(figsize=CENTROID_DIST_FIGSIZE)
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []

    for group, vals, n_members in group_series:
        color = color_map[group]
        ax.hist(
            vals,
            bins=pairwise_histogram_bin_count(len(vals)),
            alpha=CENTROID_BOX_FACE_ALPHA,
            density=True,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                alpha=CENTROID_BOX_FACE_ALPHA,
            )
        )
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            base_label = legend_entry(
                display_names_map.get(group, group),
                n_members,
                include_composition=(group == "Human"),
            )
        else:
            base_label = legend_entry(
                display_names_map.get(group, group),
                n_members,
            )
        group_skew = skew_by_group.get(group, np.nan)
        if np.isfinite(group_skew):
            legend_labels.append(f"{base_label}, skew={group_skew:.2f}")
        else:
            legend_labels.append(base_label)

    ax._viz_bold_xticks = False
    style_axes(ax)
    ax.set_xlabel(
        "Pairwise cosine distance",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        labelpad=10,
    )
    fig.supylabel(
        "Density",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        x=BOX_SUPYLABEL_X,
    )
    ax.tick_params(axis="x", labelsize=DIAG_FONT_TICK)
    ax.tick_params(axis="y", labelsize=DIAG_FONT_TICK)
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=VIZ_LEGEND_FONTSIZE,
    )

    fig.subplots_adjust(
        left=BOX_SUBPLOT_LEFT,
        right=0.98,
        bottom=0.14,
        top=BOX_SUBPLOT_TOP,
    )
    add_theory_figure_titles(
        fig,
        "Within-Group Pairwise Cosine Distance Distribution",
        embedding_set_label_text,
    )
    save_figure_pdf_svg(fig, outpath)


def plot_centroid_distance_distribution(
    plot_data: dict,
    outpath: str,
    embedding_set_label_text: str,
) -> None:
    """Deprecated alias — use plot_pairwise_distance_distribution."""
    plot_pairwise_distance_distribution(plot_data, outpath, embedding_set_label_text)


def plot_centroid_distance_boxplot(
    plot_data: dict,
    outpath: str,
    embedding_set_label_text: str,
) -> None:
    """Standard box plots (median, IQR, whiskers, outliers) with Welch comparisons."""
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    group_series = plot_data["group_series"]
    comparisons = plot_data["comparisons"]

    n_comp_lines = len(comparisons)
    comp_bottom, footnote_y, subplot_bottom = centroid_distribution_bottom_layout(
        n_comp_lines,
        n_footnote_lines=len(CENTROID_MEAN_SIG_FOOTNOTE),
    )

    fig, ax = plt.subplots(figsize=CENTROID_DIST_FIGSIZE)
    x = np.arange(len(group_series))
    box_vals = [vals for _, vals in group_series]
    bp = ax.boxplot(
        box_vals,
        positions=x,
        widths=CENTROID_BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        whis=1.5,
        medianprops={
            "color": BOXPLOT_STAT_COLOR,
            "linewidth": 1.5,
            "linestyle": "--",
        },
        meanprops={
            "marker": "^",
            "markerfacecolor": BOXPLOT_STAT_COLOR,
            "markeredgecolor": BOXPLOT_STAT_COLOR,
            "markersize": 7,
        },
        whiskerprops={
            "color": BOXPLOT_EDGE_COLOR,
            "linewidth": BOXPLOT_WHISKER_WIDTH,
            "solid_capstyle": "butt",
        },
        capprops={
            "color": BOXPLOT_EDGE_COLOR,
            "linewidth": BOXPLOT_WHISKER_WIDTH,
        },
        boxprops={
            "linewidth": BOXPLOT_WHISKER_WIDTH,
            "edgecolor": BOXPLOT_EDGE_COLOR,
        },
        flierprops={
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "white",
            "markeredgecolor": BOXPLOT_EDGE_COLOR,
            "alpha": 0.9,
        },
    )
    for patch, (group, _) in zip(bp["boxes"], group_series):
        patch.set_facecolor(color_map[group])
        patch.set_alpha(CENTROID_BOX_FACE_ALPHA)
        patch.set_edgecolor(BOXPLOT_EDGE_COLOR)

    all_vals = np.concatenate(box_vals)
    data_hi = float(np.max(all_vals))
    data_lo = float(np.min(all_vals))
    y_pad = max((data_hi - data_lo) * 0.08, 0.008)
    ax.set_ylim(bottom=max(0.0, data_lo - y_pad), top=data_hi + y_pad)

    xticks: List[str] = []
    for group, vals in group_series:
        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            xticks.append(
                legend_entry(
                    display_names_map.get(group, group),
                    len(vals),
                    include_composition=(group == "Human"),
                )
            )
        else:
            xticks.append(
                legend_entry(display_names_map.get(group, group), len(vals))
            )

    ax._viz_bold_xticks = False
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=BOX_FONT_XTICK)
    ax.tick_params(axis="x", labelsize=BOX_FONT_XTICK, pad=12)
    fig.supylabel(
        "Cosine distance to group centroid",
        fontsize=DIAG_FONT_AXIS,
        fontweight="bold",
        x=BOX_SUPYLABEL_X,
    )
    ax.tick_params(axis="y", labelsize=DIAG_FONT_TICK)

    boxplot_legend_handles = centroid_boxplot_stat_legend_handles() + [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=BOXPLOT_EDGE_COLOR,
            markersize=5,
            linestyle="None",
            label="Outliers",
        ),
    ]
    ax.legend(
        handles=boxplot_legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=VIZ_LEGEND_FONTSIZE,
    )

    fig.subplots_adjust(
        left=BOX_SUBPLOT_LEFT,
        right=0.98,
        bottom=subplot_bottom,
        top=BOX_SUBPLOT_TOP,
    )
    add_theory_figure_titles(
        fig,
        "Within-Group Distance to Group Centroid",
        embedding_set_label_text,
    )

    if comparisons:
        attach_centered_comparisons(
            fig,
            ax,
            comparisons,
            comp_bottom=comp_bottom,
            footnote_y=footnote_y,
            footnote_lines=CENTROID_MEAN_SIG_FOOTNOTE,
        )
        fig.subplots_adjust(left=BOX_SUBPLOT_LEFT, top=BOX_SUBPLOT_TOP)

    save_figure_pdf_svg(fig, outpath)


def plot_centroid_distance_figures(
    df: pd.DataFrame,
    X: np.ndarray,
    point_metrics: pd.DataFrame,
    distance_col: str,
    distribution_outpath: str,
    boxplot_outpath: str,
    embedding_set_label_text: str,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    diversity_summary_df: pd.DataFrame | None = None,
    diversity_pairwise_df: pd.DataFrame | None = None,
) -> None:
    pairwise_plot_data = prepare_pairwise_distance_plot_data(
        df, X, group_col=group_col
    )
    plot_pairwise_distance_distribution(
        pairwise_plot_data,
        distribution_outpath,
        embedding_set_label_text,
    )
    centroid_plot_data = prepare_centroid_distance_plot_data(
        point_metrics,
        distance_col,
        group_col=group_col,
        diversity_summary_df=diversity_summary_df,
        diversity_pairwise_df=diversity_pairwise_df,
    )
    plot_centroid_distance_boxplot(
        centroid_plot_data,
        boxplot_outpath,
        embedding_set_label_text,
    )


def compute_pairwise_similarity_summary(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> pd.DataFrame:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        display_names_map = DISPLAY_LABELS
    else:
        plot_df = df
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    labels = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)

    rows = []
    for group in groups:
        idx = np.where(labels == group)[0]
        if len(idx) < 2:
            continue
        vals = upper_triangle_values(cosine_similarity(X[idx]))
        rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(idx),
            "mean_pairwise_similarity": float(np.mean(vals)),
            "var_pairwise_similarity": float(np.var(vals)),
        })
    return pd.DataFrame(rows)


def similarity_distance_layout(sim: np.ndarray, *, seed: int) -> np.ndarray:
    """2D MDS on cosine distance so inter-node spacing reflects dissimilarity."""
    n = sim.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.array([[0.5, 0.5]])

    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)

    if n == 2:
        separation = float(dist[0, 1])
        coords = np.array([[0.5 - separation / 2, 0.5], [0.5 + separation / 2, 0.5]])
        return normalize_layout_coords(coords)

    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    ).fit_transform(dist)
    return normalize_layout_coords(coords)


def normalize_layout_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return coords
    if len(coords) == 1:
        return np.array([[0.5, 0.5]])
    coords -= coords.min(axis=0)
    span = coords.max(axis=0)
    span = np.where(span < 1e-12, 1.0, span)
    coords = coords / span
    return NETWORK_LAYOUT_PAD + coords * (1.0 - 2.0 * NETWORK_LAYOUT_PAD)


def load_theory_texts_for_embedding_set(embedding_set_dir: Path) -> dict[str, str]:
    """Map participant display name → cleaned theory explanation text."""
    run_info_path = embedding_set_dir / "run_info.json"
    if not run_info_path.exists():
        return {}

    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    csv_path = Path(run_info["input_file"])
    if not csv_path.is_absolute():
        csv_path = (SURVEY_ROOT / csv_path).resolve()
    text_col = run_info["text_column"]
    name_col = run_info["participant_name_column"]

    source = pd.read_csv(csv_path)
    if text_col not in source.columns:
        norm_target = " ".join(text_col.split()).lower()
        matches = [
            col
            for col in source.columns
            if " ".join(str(col).split()).lower() == norm_target
        ]
        if not matches:
            return {}
        text_col = matches[0]
    if name_col not in source.columns:
        return {}

    texts: dict[str, str] = {}
    for _, row in source.iterrows():
        name = clean_participant_display_name(str(row[name_col]).strip())
        if not name or name.lower() == "nan":
            continue
        raw = row[text_col]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            text = ""
        else:
            text = " ".join(str(raw).replace("\r", " ").replace("\n", " ").split()).strip()
        texts[name] = text
    return texts


def build_group_threshold_network_payload(
    df: pd.DataFrame,
    X: np.ndarray,
    similarity_threshold: float,
    threshold_quantile: float,
    embedding_set_label_text: str,
    *,
    seed: int,
    group_col: str = PARTICIPANT_TYPE_COL,
    theory_by_name: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
) -> dict:
    """Interactive network payload mirroring plot_group_threshold_networks."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        color_map = GROUP_COLORS_COLLAPSED
        display_names_map = DISPLAY_LABELS
    else:
        plot_df = df
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND

    labels = plot_df[group_col].values
    names = plot_df[PARTICIPANT_NAME_COL].values
    word_counts = (
        plot_df["text_word_count"].values if "text_word_count" in plot_df.columns else None
    )
    groups = ordered_groups(plot_df, group_col)
    group_counts = plot_df[group_col].value_counts().to_dict()
    quantile_pct = int(round(threshold_quantile * 100))
    theory_by_name = theory_by_name or {}

    panels: list[dict] = []
    for panel_idx, group in enumerate(groups):
        idx = np.where(labels == group)[0]
        if len(idx) == 0:
            continue

        Xg = X[idx]
        Sg = cosine_similarity(Xg)
        layout = similarity_distance_layout(Sg, seed=seed + panel_idx)

        node_records: list[dict] = []
        for local_i, global_i in enumerate(idx):
            name = clean_participant_display_name(names[global_i])
            if name_map and group in HUMAN_NETWORK_PANEL_IDS:
                display_name = format_contributor_id(name, str(group), name_map)
            else:
                display_name = name
            degree = int((Sg[local_i] >= similarity_threshold).sum()) - 1
            degree = max(degree, 0)
            node_records.append(
                {
                    "id": f"{group}-{local_i}",
                    "name": display_name,
                    "x": float(layout[local_i, 0]),
                    "y": float(layout[local_i, 1]),
                    "degree": degree,
                    "words": int(word_counts[global_i]) if word_counts is not None else None,
                    "explanation": theory_by_name.get(name, ""),
                }
            )

        links: list[dict] = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                sim = float(Sg[i, j])
                if sim >= similarity_threshold:
                    links.append(
                        {
                            "source": f"{group}-{i}",
                            "target": f"{group}-{j}",
                            "similarity": round(sim, 4),
                        }
                    )

        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            panel_title = collapsed_legend_labels(
                [group],
                {group: int(group_counts.get(group, 0))},
            )[0]
        else:
            panel_title = legend_entry(
                display_names_map.get(group, group),
                len(idx),
            )

        panels.append(
            {
                "id": group,
                "title": panel_title,
                "color": color_map[group],
                "nodes": node_records,
                "links": links,
            }
        )

    return {
        "title": "Semantic Network",
        "subtitle": embedding_set_label_text,
        "threshold": round(float(similarity_threshold), 4),
        "threshold_quantile_pct": quantile_pct,
        "collapsed": group_col == COLLAPSED_PARTICIPANT_TYPE_COL,
        "panels": panels,
    }


def export_network_interactive_demos(
    *,
    df: pd.DataFrame,
    X: np.ndarray,
    similarity_threshold: float,
    threshold_quantile: float,
    embedding_set_label_text: str,
    network_dir: Path,
    embedding_set_dir: Path,
    seed: int,
    name_map: dict[str, str] | None = None,
) -> None:
    """Write self-contained interactive HTML next to static network PNGs."""
    ni_pkg = PACKAGE_ROOT / "network"
    if str(ni_pkg) not in sys.path:
        sys.path.insert(0, str(ni_pkg))
    try:
        from network_interactive.render import render_network_interactive_html
    except ImportError:
        return

    theory_by_name = load_theory_texts_for_embedding_set(embedding_set_dir)
    if name_map is None:
        try:
            embeddings_root = infer_embeddings_root(embedding_set_dir)
            name_map = respondent_name_map_for_embeddings_root(
                embeddings_root,
                seed=seed,
            )
        except ValueError:
            human_names = df.loc[
                df[PARTICIPANT_TYPE_COL].isin(HUMAN_PARTICIPANT_TYPES),
                PARTICIPANT_NAME_COL,
            ].astype(str)
            name_map = build_contributor_code_map(human_names, seed=seed)
    variants = (
        (PARTICIPANT_TYPE_COL, NETWORK_INTERACTIVE_HTML),
        (COLLAPSED_PARTICIPANT_TYPE_COL, NETWORK_INTERACTIVE_COLLAPSED_HTML),
    )
    for group_col, html_name in variants:
        payload = build_group_threshold_network_payload(
            df,
            X,
            similarity_threshold,
            threshold_quantile,
            embedding_set_label_text,
            seed=seed,
            group_col=group_col,
            theory_by_name=theory_by_name,
            name_map=name_map,
        )
        render_network_interactive_html(payload, network_dir / html_name)


def plot_group_threshold_networks(
    df: pd.DataFrame,
    X: np.ndarray,
    similarity_threshold: float,
    threshold_quantile: float,
    embedding_set_label_text: str,
    outpath: str,
    *,
    seed: int,
    group_col: str = PARTICIPANT_TYPE_COL,
) -> None:
    """Within-group semantic network; node spacing from 2D MDS on cosine distance."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
        color_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    labels = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)
    group_counts = plot_df[group_col].value_counts().to_dict()
    quantile_pct = int(round(threshold_quantile * 100))

    n_groups = len(groups)
    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(4.8 * n_groups, 5.6),
        squeeze=False,
    )

    for panel_idx, (ax, group) in enumerate(zip(axes[0], groups)):
        idx = np.where(labels == group)[0]
        if len(idx) == 0:
            ax.set_visible(False)
            continue

        Xg = X[idx]
        Sg = cosine_similarity(Xg)
        layout = similarity_distance_layout(Sg, seed=seed + panel_idx)

        segments = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if Sg[i, j] >= similarity_threshold:
                    segments.append([layout[i], layout[j]])

        color = color_map[group]
        if segments:
            lc = LineCollection(
                segments,
                colors=NETWORK_EDGE_COLOR,
                linewidths=NETWORK_EDGE_WIDTH,
                alpha=NETWORK_EDGE_ALPHA,
                zorder=1,
            )
            ax.add_collection(lc)

        ax.scatter(
            layout[:, 0],
            layout[:, 1],
            s=NETWORK_NODE_SIZE,
            c=color,
            alpha=SCATTER_ALPHA,
            edgecolors=BAR_EDGE_COLOR,
            linewidths=BAR_EDGE_WIDTH,
            zorder=3,
        )

        if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
            panel_title = collapsed_legend_labels(
                [group],
                {group: int(group_counts.get(group, 0))},
            )[0]
        else:
            panel_title = legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(group, group),
                len(idx),
            )
        ax.set_title(
            panel_title,
            fontsize=DIAG_FONT_AXIS,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=BOX_SUBPLOT_TOP, wspace=0.28)
    add_theory_figure_titles(
        fig,
        "Semantic Threshold Network Within Each Group",
        embedding_set_label_text,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=NETWORK_EDGE_COLOR,
            linewidth=NETWORK_EDGE_WIDTH,
            alpha=NETWORK_EDGE_ALPHA,
            label=(
                f"Edge: cosine similarity ≥ {similarity_threshold:.3f} "
                f"({quantile_pct}th percentile of all pairs)"
            ),
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="None",
            alpha=0.0,
            label="The number of edges indicates within-group semantic connectivity",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#666666",
        facecolor="white",
        fontsize=VIZ_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.6,
    )

    save_figure_pdf_svg(fig, outpath)


def hdbscan_min_cluster_size(n: int) -> int:
    """Small groups need a small minimum; larger n still uses 2 (see tuning on cosine distances)."""
    if n < 3:
        return 1
    return 2


def hdbscan_cluster_selection_epsilon(embedding_col: str) -> float:
    return HDBSCAN_CLUSTER_SELECTION_EPSILON_BY_EMBEDDING.get(
        embedding_col,
        0,
    )


def run_hdbscan_within_group(
    Xg: np.ndarray,
    *,
    cluster_selection_epsilon,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> Tuple[np.ndarray, int, int, float]:
    """
    HDBSCAN on precomputed cosine distances (unit-norm embeddings).

    min_samples=1 is required here: with min_samples>=2 on cosine-distance
    matrices, this embedding space often collapses to all-noise labels.
    cluster_selection_epsilon widens the main cluster in tighter groups (GenAI)
    while leaving more PhD respondents as tail outliers.
    """
    n = Xg.shape[0]
    epsilon = float(cluster_selection_epsilon)
    if n < 3:
        return np.zeros(n, dtype=int), 1, 1, epsilon

    min_cs = (
        int(min_cluster_size)
        if min_cluster_size is not None
        else hdbscan_min_cluster_size(n)
    )
    min_cs = max(1, min(min_cs, n))
    min_ss = 1 if min_samples is None else max(1, int(min_samples))
    distance_matrix = cosine_distances(Xg)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=min_ss,
        metric="precomputed",
        cluster_selection_method="eom",
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(distance_matrix)
    return labels.astype(int), min_cs, min_ss, epsilon


def core_pct_from_embeddings(
    Xg: np.ndarray,
    *,
    cluster_selection_epsilon: float,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> float:
    labels, _, _, _ = run_hdbscan_within_group(
        Xg,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return float(summarize_core_tail(labels)["core_pct"])


def wilson_proportion_ci(
    k: int,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (percent scale)."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * lo, 100.0 * hi


def core_pct_with_wilson_ci(labels: np.ndarray) -> tuple[float, float, float]:
    """
    Point core % from HDBSCAN labels + Wilson 95% CI for the core proportion.

    The CI conditions on the full-sample cluster labels (sampling uncertainty of
    the proportion). HDBSCAN hyperparameter uncertainty is assessed separately
    via ``core_pct_parameter_sensitivity`` (with-replacement bootstrap is
    unsuitable here: duplicate embeddings collapse cosine distances and
    artificially inflate core %).
    """
    summary = summarize_core_tail(labels)
    point = float(summary["core_pct"])
    lo, hi = wilson_proportion_ci(int(summary["core_n"]), int(summary["n"]))
    return point, float(lo), float(hi)


# Default sensitivity grid around the tuned epsilon (0.1365 for 3072d).
HDBSCAN_SENSITIVITY_EPSILONS = (0.0, 0.05, 0.10, 0.1365, 0.15, 0.20)
HDBSCAN_SENSITIVITY_MIN_CLUSTER_SIZES = tuple(range(2, 11))


def core_pct_parameter_sensitivity(
    Xg: np.ndarray,
    *,
    base_epsilon: float,
    epsilons: tuple[float, ...] = HDBSCAN_SENSITIVITY_EPSILONS,
    min_cluster_sizes: tuple[int, ...] = HDBSCAN_SENSITIVITY_MIN_CLUSTER_SIZES,
) -> list[dict]:
    """Core % across a modest HDBSCAN hyperparameter grid."""
    n = int(Xg.shape[0])
    rows: list[dict] = []
    for eps in epsilons:
        for min_cs in min_cluster_sizes:
            if min_cs > n:
                continue
            core = core_pct_from_embeddings(
                Xg,
                cluster_selection_epsilon=float(eps),
                min_cluster_size=int(min_cs),
            )
            rows.append({
                "cluster_selection_epsilon": float(eps),
                "min_cluster_size": int(min_cs),
                "core_pct": float(core),
                "is_base": bool(
                    abs(float(eps) - float(base_epsilon)) < 1e-12 and min_cs == 2
                ),
            })
    return rows


def summarize_core_tail(labels: np.ndarray) -> Dict[str, float]:
    """Core = any HDBSCAN cluster; tail = noise (-1)."""
    n = len(labels)
    tail_mask = labels == -1
    core_mask = ~tail_mask
    core_n = int(core_mask.sum())
    tail_n = int(tail_mask.sum())

    if core_n == 0:
        largest_cluster_n = 0
        n_clusters = 0
    else:
        core_labels = labels[core_mask]
        _, counts = np.unique(core_labels, return_counts=True)
        largest_cluster_n = int(counts.max())
        n_clusters = int(len(counts))

    return {
        "n": n,
        "core_n": core_n,
        "tail_n": tail_n,
        "core_pct": 100.0 * core_n / n,
        "tail_pct": 100.0 * tail_n / n,
        "n_clusters": n_clusters,
        "largest_cluster_n": largest_cluster_n,
        "largest_cluster_pct": 100.0 * largest_cluster_n / n,
    }


def compute_semantic_clustering_tables(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    group_col: str = PARTICIPANT_TYPE_COL,
    cluster_selection_epsilon: float = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        plot_df = with_collapsed_group(df)
    else:
        plot_df = df

    labels_col = plot_df[group_col].values
    groups = ordered_groups(plot_df, group_col)
    participant_rows: List[dict] = []
    summary_rows: List[dict] = []

    for group in groups:
        idx = np.where(labels_col == group)[0]
        Xg = X[idx]
        cluster_labels, min_cs, min_ss, epsilon = run_hdbscan_within_group(
            Xg,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        summary = summarize_core_tail(cluster_labels)
        summary_rows.append({
            "participant_type": group,
            "min_cluster_size": min_cs,
            "min_samples": min_ss,
            "cluster_selection_epsilon": epsilon,
            **summary,
        })

        for local_i, global_i in enumerate(idx):
            participant_rows.append({
                PARTICIPANT_NAME_COL: plot_df.iloc[global_i][PARTICIPANT_NAME_COL],
                group_col: group,
                "hdbscan_label": int(cluster_labels[local_i]),
                "is_core": bool(cluster_labels[local_i] >= 0),
                "is_tail": bool(cluster_labels[local_i] == -1),
            })

    return pd.DataFrame(participant_rows), pd.DataFrame(summary_rows)
def run_semantic_clustering_analysis(
    df: pd.DataFrame,
    X: np.ndarray,
    space_outdir: str,
    embedding_set_label_text: str,
    *,
    embedding_col: str,
) -> Dict[str, dict]:
    """Compute HDBSCAN labels and save collapsed core/tail tables (for core_tail fig)."""
    del embedding_set_label_text  # kept for call-site compatibility
    if not HAS_HDBSCAN:
        print("hdbscan not installed. Skipping semantic clustering.")
        return {}

    cluster_selection_epsilon = hdbscan_cluster_selection_epsilon(embedding_col)
    # Only Humans|GenAI collapsed tables are consumed (fig_core_tail_structure).
    group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    suffix = "_collapsed"
    participant_df, summary_df = compute_semantic_clustering_tables(
        df,
        X,
        group_col=group_col,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    clustering_results: Dict[str, dict] = {
        suffix: {
            "participant_df": participant_df,
            "summary_df": summary_df,
            "group_col": group_col,
        }
    }
    participant_df.to_csv(
        os.path.join(
            space_outdir,
            SEMANTIC_CLUSTERING_PARTICIPANT_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        os.path.join(
            space_outdir,
            SEMANTIC_CLUSTERING_SUMMARY_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )
    print("\nHDBSCAN core/tail summary (Humans|GenAI):")
    print(summary_df.to_string(index=False))
    return clustering_results


def merge_point_metrics_with_clustering(
    point_metrics: pd.DataFrame,
    participant_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    cluster_cols = [PARTICIPANT_NAME_COL, group_col, "is_tail", "hdbscan_label"]
    merged = point_metrics.merge(
        participant_df[cluster_cols],
        on=[PARTICIPANT_NAME_COL, group_col],
        how="left",
    )
    merged["is_tail"] = merged["is_tail"].fillna(False).astype(float)
    return merged


def run_diversity_inference_analysis(
    point_metrics: pd.DataFrame,
    participant_df: pd.DataFrame,
    pairwise_summary_df: pd.DataFrame,
    space_outdir: str,
    *,
    group_col: str,
    distance_col: str,
    suffix: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs and pairwise tests for prediction-2 diversity metrics."""
    if group_col == COLLAPSED_PARTICIPANT_TYPE_COL:
        display_names_map = DISPLAY_LABELS
        preferred_order = GROUP_ORDER_COLLAPSED
    else:
        display_names_map = PARTICIPANT_TYPE_TO_LEGEND
        preferred_order = GROUP_ORDER

    merged = merge_point_metrics_with_clustering(
        point_metrics, participant_df, group_col=group_col
    )
    groups = [g for g in preferred_order if g in merged[group_col].unique()]
    groups += [g for g in merged[group_col].unique() if g not in groups]

    summary_rows: List[dict] = []
    for group in groups:
        centroid_vals = group_metric_values(merged, group_col, group, distance_col)
        centrality_vals = group_metric_values(
            merged, group_col, group, "within_group_centrality"
        )
        tail_vals = group_metric_values(merged, group_col, group, "is_tail")

        pooled_row = pairwise_summary_df.loc[
            pairwise_summary_df["group"] == group
        ]
        pooled_mean_sim = (
            float(pooled_row["mean_pairwise_similarity"].iloc[0])
            if len(pooled_row)
            else np.nan
        )
        pooled_var_sim = (
            float(pooled_row["var_pairwise_similarity"].iloc[0])
            if len(pooled_row)
            else np.nan
        )

        c_lo, c_hi = bootstrap_mean_ci(centroid_vals, seed=seed)
        s_lo, s_hi = bootstrap_mean_ci(centrality_vals, seed=seed + 1)
        t_lo, t_hi = bootstrap_mean_ci(tail_vals, seed=seed + 2)

        summary_rows.append({
            "group": group,
            "display_label": display_names_map.get(group, group),
            "n": len(centroid_vals),
            "mean_centroid_distance": float(np.mean(centroid_vals)),
            "centroid_distance_ci_low": c_lo,
            "centroid_distance_ci_high": c_hi,
            "mean_within_group_centrality": float(np.mean(centrality_vals)),
            "within_group_centrality_ci_low": s_lo,
            "within_group_centrality_ci_high": s_hi,
            "tail_pct": 100.0 * float(np.mean(tail_vals)),
            "tail_pct_ci_low": 100.0 * t_lo,
            "tail_pct_ci_high": 100.0 * t_hi,
            "mean_pairwise_similarity_pooled": pooled_mean_sim,
            "var_pairwise_similarity_pooled": pooled_var_sim,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(
            space_outdir,
            Q4_DIVERSITY_SUMMARY_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pairwise_rows: List[dict] = []
    for metric_name, col in (
        ("centroid_distance", distance_col),
        ("within_group_centrality", "within_group_centrality"),
        ("tail_indicator", "is_tail"),
    ):
        pairwise_rows.extend(
            build_pairwise_inference_rows(
                merged,
                col,
                groups,
                group_col=group_col,
                metric_name=metric_name,
                seed=seed,
            )
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(
        os.path.join(
            space_outdir,
            Q4_DIVERSITY_PAIRWISE_CSV.format(suffix=suffix),
        ),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nQ4 diversity inference ({suffix or '3 groups'}):")
    print(summary_df.to_string(index=False))
    print(pairwise_df.to_string(index=False))
    return summary_df, pairwise_df


# ---------------------------------------------------------------------
# One embedding space
# ---------------------------------------------------------------------

def visualize_one_space(
    df: pd.DataFrame,
    embedding_col: str,
    space_outdir: str,
    network_dir: Path,
    threshold_quantile: float,
    seed: int,
    embedding_set_label_text: str,
    embedding_set_dir: Path,
) -> None:
    os.makedirs(space_outdir, exist_ok=True)
    network_dir = Path(network_dir)
    network_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Visualizing embedding space: {embedding_col}")
    print("=" * 80)

    X_raw = stack_embeddings(df, embedding_col)
    X = normalize(X_raw)

    _, similarity_threshold = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )

    run_semantic_clustering_analysis(
        df=df,
        X=X,
        space_outdir=space_outdir,
        embedding_set_label_text=embedding_set_label_text,
        embedding_col=embedding_col,
    )

    export_network_interactive_demos(
        df=df,
        X=X,
        similarity_threshold=similarity_threshold,
        threshold_quantile=threshold_quantile,
        embedding_set_label_text=embedding_set_label_text,
        network_dir=network_dir,
        embedding_set_dir=embedding_set_dir,
        seed=seed,
    )

    print(f"Similarity threshold used: {similarity_threshold:.4f}")
    print("Saved clustering CSV tables to:")
    print(space_outdir)
    print("Saved interactive network demos (*.html) to:")
    print(network_dir)


def task_label_from_key(task_key: str) -> str:
    return " · ".join(
        format_embedding_set_part(part) for part in task_key.split("/")
    )


# ---------------------------------------------------------------------
# Batch phase grids (2×2 across four tasks, by pre-ML / post-ML)
# ---------------------------------------------------------------------

def discover_phase_task_dirs(
    embeddings_root: Path,
    phase: str,
) -> List[Tuple[str, Path]]:
    dirs: List[Tuple[str, Path]] = []
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        set_dir = embeddings_root / task_key / phase
        if (set_dir / "embeddings_wide.parquet").exists():
            dirs.append((task_key, set_dir))
    return dirs


def load_phase_task_bundle(
    set_dir: Path,
    embedding_col: str,
    *,
    threshold_quantile: float,
) -> dict:
    df = read_anonymized_embeddings_wide(set_dir / "embeddings_wide.parquet")
    X_raw = stack_embeddings(df, embedding_col)
    X = normalize(X_raw)
    degree, _ = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )
    point_metrics = make_point_metrics(df, X, degree)
    cluster_selection_epsilon = hdbscan_cluster_selection_epsilon(embedding_col)
    _, clustering_3g = compute_semantic_clustering_tables(
        df,
        X,
        group_col=PARTICIPANT_TYPE_COL,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    _, clustering_collapsed = compute_semantic_clustering_tables(
        df,
        X,
        group_col=COLLAPSED_PARTICIPANT_TYPE_COL,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    return {
        "df": df,
        "X": X,
        "point_metrics": point_metrics,
        "clustering_3g": clustering_3g,
        "clustering_collapsed": clustering_collapsed,
    }


def make_phase_grid_axes() -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=PHASE_GRID_FIGSIZE,
        gridspec_kw={
            "hspace": PHASE_GRID_ROW_GAP,
            "wspace": PHASE_GRID_COL_GAP,
        },
    )
    return fig, axes.ravel()


def phase_grid_layout_adjust(
    fig,
    *,
    footnote_lines: int = 0,
    bottom_extra: float = 0.0,
    top: float | None = None,
) -> None:
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=top if top is not None else PHASE_GRID_PANEL_TOP,
        bottom=0.08
        + footnote_lines * PHASE_GRID_FOOTNOTE_LINE_STEP
        + bottom_extra,
        hspace=PHASE_GRID_ROW_GAP,
        wspace=PHASE_GRID_COL_GAP,
    )


def add_phase_grid_figure_legend(
    fig,
    handles: list,
    labels: list[str],
    *,
    ncol: int,
    bbox_y: float | None = None,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] | None = None,
    fontsize: float | None = None,
):
    legend_y = (bbox_y if bbox_y is not None else PHASE_GRID_LEGEND_Y) + VIZ_LEGEND_Y_SHIFT
    anchor = bbox_to_anchor if bbox_to_anchor is not None else (0.5, legend_y)
    legend = fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=True,
        fontsize=PHASE_GRID_LEGEND_FONTSIZE if fontsize is None else fontsize,
        borderaxespad=0.0,
    )
    return legend


def _legend_blank_handle() -> Line2D:
    return Line2D([], [], linestyle="None", alpha=0, label=" ")


def centroid_boxplot_two_column_legend_entries(
    group_handles: list,
    group_labels: list[str],
    stat_handles: list,
) -> Tuple[list, list[str], int]:
    """Column 1 = groups, column 2 = median/mean/whiskers (mpl fills ncol by column)."""
    stat_labels = [handle.get_label() for handle in stat_handles]
    n_rows = max(len(group_handles), len(stat_handles))
    blank = _legend_blank_handle()
    handles: list = []
    labels: list[str] = []
    for i in range(n_rows):
        if i < len(group_handles):
            handles.append(group_handles[i])
            labels.append(group_labels[i])
        else:
            handles.append(blank)
            labels.append(" ")
    for i in range(n_rows):
        if i < len(stat_handles):
            handles.append(stat_handles[i])
            labels.append(stat_labels[i])
        else:
            handles.append(blank)
            labels.append(" ")
    return handles, labels, n_rows


def add_phase_grid_centroid_boxplot_legend(
    fig,
    group_handles: list,
    group_labels: list[str],
    stat_handles: list,
    *,
    bbox_y: float | None = None,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> int:
    """Single framed legend: col 1 = groups, col 2 = median/mean/whiskers."""
    legend_y = (bbox_y if bbox_y is not None else PHASE_GRID_LEGEND_Y) + VIZ_LEGEND_Y_SHIFT
    handles, labels, n_rows = centroid_boxplot_two_column_legend_entries(
        group_handles, group_labels, stat_handles
    )
    anchor = bbox_to_anchor if bbox_to_anchor is not None else (0.5, legend_y)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=2,
        frameon=True,
        fontsize=PHASE_GRID_LEGEND_FONTSIZE,
        borderaxespad=0.0,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    return n_rows


def phase_grid_group_counts(df: pd.DataFrame, *, collapse_human: bool) -> dict[str, int]:
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        groups = GROUP_ORDER_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        groups = GROUP_ORDER
    counts = plot_df[group_col].value_counts().to_dict()
    return {g: int(counts.get(g, 0)) for g in groups}


def phase_grid_group_legend_handles_labels(
    *,
    collapse_human: bool,
    n_by_group: dict[str, int],
    include_human_composition: bool = True,
    face_alpha: float = CENTROID_BOX_FACE_ALPHA,
    italic_uppercase_n: bool = False,
) -> Tuple[list, list[str], int]:
    if collapse_human:
        groups = GROUP_ORDER_COLLAPSED
        color_map = GROUP_COLORS_COLLAPSED
        labels = [
            legend_entry(
                g,
                n_by_group.get(g, 0),
                include_composition=(include_human_composition and g == "Human"),
                italic_uppercase_n=italic_uppercase_n,
            )
            for g in groups
        ]
    else:
        groups = GROUP_ORDER
        color_map = GROUP_COLORS_BY_PARTICIPANT_TYPE
        labels = [
            legend_entry(
                PARTICIPANT_TYPE_TO_LEGEND.get(g, g),
                n_by_group.get(g, 0),
                italic_uppercase_n=italic_uppercase_n,
            )
            for g in groups
        ]
    handles = [
        Patch(
            facecolor=color_map[g],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            alpha=face_alpha,
        )
        for g in groups
    ]
    return handles, labels, len(groups)
def shorten_panel_comparison_label(label: str) -> str:
    out = label
    for long, short in PHASE_GRID_COMPARISON_LABEL_SHORT:
        out = out.replace(long, short)
    return out.replace(" vs. ", "–").replace(" vs ", "–")


def format_panel_comparison_line(label: str, p: float) -> str:
    """Short pair label with explicit p-value and significance stars."""
    short = shorten_panel_comparison_label(label)
    return f"{short}: p={fmt_p(p)} {significance_label(p)}"


def attach_panel_comparisons(
    ax,
    comparisons: List[Tuple[str, float]],
) -> None:
    """Welch p-values in the upper-right corner."""
    if not comparisons:
        return

    header = "Welch p"
    lines = [header] + [
        format_panel_comparison_line(label, pval) for label, pval in comparisons
    ]
    n = len(lines)
    x_right, y_top = PHASE_GRID_PANEL_COMPARISON_XY
    line_step = PHASE_GRID_PANEL_COMPARISON_LINE_STEP
    pad = PHASE_GRID_PANEL_COMPARISON_BOX_PAD
    fontsize = PHASE_GRID_PANEL_COMPARISON_FONTSIZE

    max_chars = max(len(s) for s in lines)
    box_w = min(0.88, 0.0205 * max_chars + 2 * pad)
    box_h = n * line_step + 2 * pad
    box_x0 = x_right - box_w
    box_y0 = y_top - box_h

    ax.add_patch(
        FancyBboxPatch(
            (box_x0, box_y0),
            box_w,
            box_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#FFFFFF",
            edgecolor="#9E9E9E",
            linewidth=0.85,
            alpha=0.96,
            zorder=5,
            clip_on=False,
        )
    )
    for i, text in enumerate(lines):
        ax.text(
            x_right - pad,
            y_top - pad - i * line_step,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
            fontweight="normal",
            fontstyle="italic" if i == 0 else "normal",
            color="#333333",
            zorder=6,
            clip_on=False,
        )


def draw_phase_grid_footnote(
    fig,
    lines: Tuple[str, ...],
    *,
    y: float = PHASE_GRID_FOOTNOTE_Y,
    line_step: float = PHASE_GRID_FOOTNOTE_LINE_STEP,
) -> None:
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            y - i * line_step,
            line,
            ha="center",
            va="bottom",
            fontsize=PHASE_GRID_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def centroid_boxplot_stat_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=BOXPLOT_STAT_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=BOXPLOT_STAT_COLOR,
            markerfacecolor=BOXPLOT_STAT_COLOR,
            markeredgecolor=BOXPLOT_STAT_COLOR,
            markersize=5,
            linestyle="None",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            color=BOXPLOT_EDGE_COLOR,
            linewidth=BOXPLOT_WHISKER_WIDTH,
            label="Whiskers (1.5×IQR)",
        ),
    ]


def draw_semantic_map_panel(
    ax,
    df: pd.DataFrame,
    coords: np.ndarray,
    *,
    collapse_human: bool,
    panel_title: str,
    axis_bounds: dict[str, float] | None = None,
    scatter_size: float | None = None,
    box_aspect: float | None = PHASE_GRID_SEMANTIC_MAP_BOX_ASPECT,
    title_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    hollow: bool = False,
) -> None:
    if collapse_human:
        plot_df = with_collapsed_group(df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_COLLAPSED
    else:
        plot_df = df
        group_col = PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    groups = ordered_groups(plot_df, group_col)
    marker_size = (
        PHASE_GRID_SCATTER_SIZE if scatter_size is None else float(scatter_size)
    )
    title_fs = (
        PHASE_GRID_PANEL_TITLE_FONTSIZE
        if title_fontsize is None
        else float(title_fontsize)
    )
    tick_fs = (
        PHASE_GRID_TICK_FONTSIZE if tick_fontsize is None else float(tick_fontsize)
    )

    for group in groups:
        mask = plot_df[group_col].values == group
        color = colors_map.get(group, "#888888")
        if hollow:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                facecolors="none",
                edgecolors=color,
                s=marker_size,
                linewidths=PRE_POST_SEMANTIC_MAP_PRE_EDGEWIDTH,
                zorder=2,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=marker_size,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=BAR_EDGE_WIDTH,
                zorder=2,
            )

    if axis_bounds is not None:
        apply_semantic_map_2d_bounds(
            ax,
            axis_bounds,
            box_aspect=box_aspect,
        )
    else:
        apply_semantic_map_2d_bounds(
            ax,
            semantic_map_2d_bounds(coords, "PCA"),
            box_aspect=box_aspect,
        )
    ax.set_title(
        panel_title,
        fontweight="bold",
        fontsize=title_fs,
        pad=8,
        loc="left",
    )
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(PRE_POST_SEMANTIC_MAP_SPINE_WIDTH)


def _semantic_map_name_index(df: pd.DataFrame) -> dict[str, int]:
    names = df[PARTICIPANT_NAME_COL].astype(str).map(clean_participant_display_name)
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        if name and name not in out:
            out[name] = i
    return out


def draw_semantic_map_panel_with_trajectories(
    ax,
    df_pre: pd.DataFrame,
    coords_pre: np.ndarray,
    df_post: pd.DataFrame,
    coords_post: np.ndarray,
    *,
    collapse_human: bool,
    panel_title: str,
    axis_bounds: dict[str, float],
    box_aspect: float | None = PHASE_GRID_SEMANTIC_MAP_BOX_ASPECT,
    title_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    show_phase_legend: bool = True,
) -> None:
    """Post-ML panel: pre→post trajectories; pre hollow, post filled."""
    if collapse_human:
        plot_pre = with_collapsed_group(df_pre)
        plot_post = with_collapsed_group(df_post)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_COLLAPSED
    else:
        plot_pre = df_pre
        plot_post = df_post
        group_col = PARTICIPANT_TYPE_COL
        colors_map = GROUP_COLORS_BY_PARTICIPANT_TYPE

    pre_by_name = _semantic_map_name_index(plot_pre)
    post_names = (
        plot_post[PARTICIPANT_NAME_COL]
        .astype(str)
        .map(clean_participant_display_name)
        .tolist()
    )
    post_groups = plot_post[group_col].astype(str).tolist()

    for post_i, name in enumerate(post_names):
        pre_i = pre_by_name.get(name)
        if pre_i is None:
            continue
        color = colors_map.get(post_groups[post_i], "#888888")
        ax.plot(
            [float(coords_pre[pre_i, 0]), float(coords_post[post_i, 0])],
            [float(coords_pre[pre_i, 1]), float(coords_post[post_i, 1])],
            color=color,
            alpha=PRE_POST_SEMANTIC_MAP_LINE_ALPHA,
            linewidth=PRE_POST_SEMANTIC_MAP_LINEWIDTH,
            zorder=1,
            solid_capstyle="round",
        )

    groups = ordered_groups(plot_post, group_col)
    for group in groups:
        color = colors_map.get(group, "#888888")
        pre_mask = plot_pre[group_col].values == group
        post_mask = plot_post[group_col].values == group
        if np.any(pre_mask):
            ax.scatter(
                coords_pre[pre_mask, 0],
                coords_pre[pre_mask, 1],
                facecolors="none",
                edgecolors=color,
                s=PRE_POST_SEMANTIC_MAP_PRE_SIZE,
                linewidths=PRE_POST_SEMANTIC_MAP_PRE_EDGEWIDTH,
                zorder=2,
            )
        if np.any(post_mask):
            ax.scatter(
                coords_post[post_mask, 0],
                coords_post[post_mask, 1],
                c=color,
                s=PRE_POST_SEMANTIC_MAP_POST_SIZE,
                alpha=SCATTER_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                linewidths=PRE_POST_SEMANTIC_MAP_POST_EDGEWIDTH,
                zorder=3,
            )

    apply_semantic_map_2d_bounds(ax, axis_bounds, box_aspect=box_aspect)
    title_fs = (
        PHASE_GRID_PANEL_TITLE_FONTSIZE
        if title_fontsize is None
        else float(title_fontsize)
    )
    tick_fs = (
        PHASE_GRID_TICK_FONTSIZE if tick_fontsize is None else float(tick_fontsize)
    )
    ax.set_title(panel_title, fontweight="bold", fontsize=title_fs, pad=8, loc="left")
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(PRE_POST_SEMANTIC_MAP_SPINE_WIDTH)

    if show_phase_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#555555",
                markeredgewidth=PRE_POST_SEMANTIC_MAP_PRE_EDGEWIDTH,
                linestyle="None",
                markersize=8,
                label="pre-ML",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="#555555",
                markeredgecolor=BAR_EDGE_COLOR,
                markeredgewidth=PRE_POST_SEMANTIC_MAP_POST_EDGEWIDTH,
                linestyle="None",
                markersize=8,
                label="post-ML",
            ),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            fontsize=max(9.0, float(tick_fs) - 2.0),
            handletextpad=0.35,
            borderpad=0.35,
            labelspacing=0.25,
            borderaxespad=0.35,
        )
        leg.get_frame().set_edgecolor("#666666")
        leg.get_frame().set_linewidth(0.7)
        leg.set_zorder(5)


def centroid_boxplot_ylim(
    bp: dict,
    *,
    pad_frac: float = 0.10,
    min_pad: float = 0.008,
) -> tuple[float, float]:
    whisker_ends = [
        float(y)
        for whisker in bp["whiskers"]
        for y in whisker.get_ydata()
    ]
    if not whisker_ends:
        return 0.0, 1.0
    y_hi = max(whisker_ends)
    y_lo = min(whisker_ends)
    y_pad = max((y_hi - y_lo) * pad_frac, min_pad)
    return max(0.0, y_lo - y_pad), y_hi + y_pad


def draw_centroid_boxplot_panel(
    ax,
    plot_data: dict,
    *,
    panel_title: str,
    show_group_xticklabels: bool = True,
) -> None:
    color_map = plot_data["color_map"]
    display_names_map = plot_data["display_names_map"]
    group_col = plot_data["group_col"]
    group_series = plot_data["group_series"]

    x = np.arange(len(group_series))
    box_vals = [vals for _, vals in group_series]
    bp = ax.boxplot(
        box_vals,
        positions=x,
        widths=CENTROID_BOX_WIDTH * 0.85,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        whis=1.5,
        medianprops={
            "color": BOXPLOT_STAT_COLOR,
            "linewidth": 1.2,
            "linestyle": "--",
        },
        meanprops={
            "marker": "^",
            "markerfacecolor": BOXPLOT_STAT_COLOR,
            "markeredgecolor": BOXPLOT_STAT_COLOR,
            "markersize": 5,
        },
        whiskerprops={"color": BOXPLOT_EDGE_COLOR, "linewidth": 1.1},
        capprops={"color": BOXPLOT_EDGE_COLOR, "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": BOXPLOT_EDGE_COLOR},
    )
    for patch, (group, _) in zip(bp["boxes"], group_series):
        patch.set_facecolor(color_map[group])
        patch.set_alpha(CENTROID_BOX_FACE_ALPHA)
        patch.set_edgecolor(BOXPLOT_EDGE_COLOR)

    xticks: List[str] = []
    for group, _vals in group_series:
        xticks.append(display_names_map.get(group, group))

    ax._viz_bold_xticks = False
    style_axes(ax)
    ax.set_xticks(x)
    if show_group_xticklabels:
        ax.set_xticklabels(xticks, fontsize=PHASE_GRID_TICK_FONTSIZE)
        ax.tick_params(axis="x", labelsize=PHASE_GRID_TICK_FONTSIZE, pad=8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.set_title(
        panel_title,
        fontweight="normal",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.set_ylim(*centroid_boxplot_ylim(bp))


def draw_pairwise_distribution_panel(
    ax,
    plot_data: dict,
    *,
    panel_title: str,
) -> None:
    color_map = plot_data["color_map"]
    group_series = plot_data["group_series"]

    for group, vals, _n_members in group_series:
        ax.hist(
            vals,
            bins=pairwise_histogram_bin_count(len(vals)),
            alpha=CENTROID_BOX_FACE_ALPHA,
            density=True,
            color=color_map[group],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
        )

    style_axes(ax)
    ax.set_title(
        panel_title,
        fontweight="normal",
        fontsize=PHASE_GRID_PANEL_TITLE_FONTSIZE,
        pad=8,
    )
    ax.tick_params(axis="both", labelsize=PHASE_GRID_TICK_FONTSIZE)


def draw_centroid_distribution_panel(
    ax,
    plot_data: dict,
    *,
    panel_title: str,
) -> None:
    """Deprecated alias — use draw_pairwise_distribution_panel."""
    draw_pairwise_distribution_panel(ax, plot_data, panel_title=panel_title)
def phase_grid_task_pca_coords(
    task_bundles: List[Tuple[str, dict]],
    *,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> list[tuple[str, dict, np.ndarray]]:
    panel_data: list[tuple[str, dict, np.ndarray]] = []
    for task_key, bundle in task_bundles:
        coords, _ = compute_projection(
            X=bundle["X"],
            method="pca",
        seed=seed,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
            n_components=2,
        )
        panel_data.append((task_key, bundle, coords))
    return panel_data


def phase_grid_semantic_map_axis_bounds(
    panel_data: list[tuple[str, dict, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Fixed square PCA axis limits for 2×2 phase-grid semantic maps."""
    _ = panel_data
    return {
        "x_left": PHASE_GRID_MAP_AXIS_MIN,
        "x_right": PHASE_GRID_MAP_AXIS_MAX,
        "y_bottom": PHASE_GRID_MAP_AXIS_MIN,
        "y_top": PHASE_GRID_MAP_AXIS_MAX,
        "projection_name": "PCA",
    }



def plot_phase_grid_semantic_maps(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
    axis_bounds: dict[str, float] | None = None,
    panel_data: list[tuple[str, dict, np.ndarray]] | None = None,
) -> None:
    _ = phase_label
    suptitle = PHASE_GRID_SEMANTIC_MAP_TITLE
    fig, axes = make_phase_grid_axes()
    if panel_data is None:
        panel_data = phase_grid_task_pca_coords(
            task_bundles,
            seed=seed,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
        )

    unified_bounds = axis_bounds
    if unified_bounds is None:
        unified_bounds = phase_grid_semantic_map_axis_bounds(panel_data)

    for ax, (task_key, bundle, coords) in zip(axes, panel_data):
        draw_semantic_map_panel(
            ax,
            bundle["df"],
            coords,
            collapse_human=collapse_human,
            panel_title=task_label_from_key(task_key),
            axis_bounds=unified_bounds,
        )

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
        face_alpha=SCATTER_ALPHA,
    )
    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=(),
        suptitle_fontsize=PHASE_GRID_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    add_phase_grid_figure_legend(
        fig,
        handles,
        labels,
        ncol=ncol,
        bbox_y=header.legend_y,
    )

    axis_label = "PCA dimension"
    fig.supxlabel(
        f"{axis_label} 1",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=PHASE_GRID_SEMANTIC_MAP_SUPXLABEL_Y,
    )
    fig.supylabel(
        f"{axis_label} 2",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    phase_grid_layout_adjust(
        fig,
        top=header.panel_top,
        bottom_extra=PHASE_GRID_SEMANTIC_MAP_BOTTOM_EXTRA,
    )
    save_figure_pdf_svg(fig, outpath)


def build_pre_post_shared_pca_panels(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    *,
    seed: int,
) -> list[tuple[str, str, dict, np.ndarray]]:
    """Per task: fit PCA on pooled Pre+Post (all groups), then project each phase.

    Returns panels in 2×4 reading order (row 1 = Pre-ML, row 2 = Post-ML):
    Pre:  task1 … task4; Post: task1 … task4
    (task order = DIVERSITY_TASK_PANEL_ORDER).
    """
    if "pre-ML" not in phase_bundles or "post-ML" not in phase_bundles:
        raise ValueError("Need both pre-ML and post-ML bundles.")
    pre_by_task = dict(phase_bundles["pre-ML"])
    post_by_task = dict(phase_bundles["post-ML"])
    coords_by_phase: dict[str, list[tuple[str, str, dict, np.ndarray]]] = {
        "Pre-ML": [],
        "Post-ML": [],
    }
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        if task_key not in pre_by_task or task_key not in post_by_task:
            raise ValueError(f"Missing task for pre/post map: {task_key}")
        pre_b = pre_by_task[task_key]
        post_b = post_by_task[task_key]
        # All respondents (Human + GenAI) × both phases → one shared basis.
        X_pool = np.vstack([pre_b["X"], post_b["X"]])
        pca = PCA(n_components=2, random_state=seed).fit(X_pool)
        for phase, bundle in (("Pre-ML", pre_b), ("Post-ML", post_b)):
            coords = pca.transform(bundle["X"])
            coords_by_phase[phase].append((task_key, phase, bundle, coords))
    return coords_by_phase["Pre-ML"] + coords_by_phase["Post-ML"]


def format_semantic_map_panel_title(
    letter: str,
    task_key: str,
    phase: str,
) -> str:
    """Single-line panel title: ``a.  Topic · Design · Pre-ML``."""
    task = task_label_from_key(task_key).replace(
        "Second-Order Interactions", "Interactions"
    )
    return f"{letter}.  {task} · {phase}"


def _axes_title_artist(ax):
    """Title Text that actually holds the string (``loc='left'`` → ``_left_title``)."""
    for artist in (getattr(ax, "_left_title", None), ax.title, getattr(ax, "_right_title", None)):
        if artist is not None and artist.get_text():
            return artist
    return ax.title


def plot_pre_post_semantic_maps_expanded(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    outpath: Path,
    *,
    collapse_human: bool,
    seed: int,
) -> None:
    """2×4 map: row 1 Pre-ML, row 2 Post-ML (pooled Pre+Post PCA per task).

    Pre-ML panels use solid markers. Post-ML panels overlay pre→post
    trajectories (hollow pre, solid post) with a per-panel phase legend.
    """
    panels = build_pre_post_shared_pca_panels(phase_bundles, seed=seed)
    pre_by_task = {
        task_key: (bundle, coords)
        for task_key, phase, bundle, coords in panels
        if phase == "Pre-ML"
    }
    row_hspace = PRE_POST_SEMANTIC_MAP_HSPACE
    col_wspace = PRE_POST_SEMANTIC_MAP_WSPACE
    fig, axes = plt.subplots(
        2,
        4,
        figsize=PRE_POST_SEMANTIC_MAP_FIGSIZE,
        gridspec_kw={"hspace": row_hspace, "wspace": col_wspace},
    )
    axes_flat = axes.ravel()
    unified_bounds = phase_grid_semantic_map_axis_bounds()

    for i, (ax, (task_key, phase, bundle, coords)) in enumerate(
        zip(axes_flat, panels)
    ):
        letter = chr(ord("a") + i)
        title = format_semantic_map_panel_title(letter, task_key, phase)
        if phase == "Post-ML":
            pre_bundle, pre_coords = pre_by_task[task_key]
            draw_semantic_map_panel_with_trajectories(
                ax,
                pre_bundle["df"],
                pre_coords,
                bundle["df"],
                coords,
                collapse_human=collapse_human,
                panel_title=title,
                axis_bounds=unified_bounds,
                box_aspect=1.0,
                title_fontsize=PRE_POST_SEMANTIC_MAP_PANEL_TITLE_FONTSIZE,
                tick_fontsize=PRE_POST_SEMANTIC_MAP_TICK_FONTSIZE,
            )
        else:
            draw_semantic_map_panel(
                ax,
                bundle["df"],
                coords,
                collapse_human=collapse_human,
                panel_title=title,
                axis_bounds=unified_bounds,
                scatter_size=PRE_POST_SEMANTIC_MAP_SCATTER_SIZE,
                box_aspect=1.0,
                title_fontsize=PRE_POST_SEMANTIC_MAP_PANEL_TITLE_FONTSIZE,
                tick_fontsize=PRE_POST_SEMANTIC_MAP_TICK_FONTSIZE,
                hollow=False,
            )

    sample_df = phase_bundles["pre-ML"][0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
        include_human_composition=False,
        face_alpha=SCATTER_ALPHA,
        italic_uppercase_n=True,
    )
    legend_title_gap = PRE_POST_SEMANTIC_MAP_LEGEND_TITLE_GAP
    # Temporary legend to measure height (centered after layout).
    legend = add_phase_grid_figure_legend(
        fig,
        handles,
        labels,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=PRE_POST_SEMANTIC_MAP_LEGEND_FONTSIZE,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    legend_h = float(legend_bbox.y1 - legend_bbox.y0)
    title_heights = []
    for ax in axes[0]:
        title = _axes_title_artist(ax)
        if not title.get_text():
            continue
        tb = title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        ab = ax.get_position()
        title_heights.append(float(tb.y1 - ab.y1))
    title_reserve = max(title_heights) if title_heights else 0.05
    panel_top = min(0.92, 0.995 - legend_h - legend_title_gap - title_reserve)

    fig.supxlabel(
        "PCA dimension 1",
        fontsize=PRE_POST_SEMANTIC_MAP_AXIS_FONTSIZE,
        fontweight="bold",
        y=0.004,
    )
    fig.supylabel(
        "PCA dimension 2",
        fontsize=PRE_POST_SEMANTIC_MAP_AXIS_FONTSIZE,
        fontweight="bold",
        x=PRE_POST_SEMANTIC_MAP_YLABEL_X,
    )
    fig.subplots_adjust(
        left=PRE_POST_SEMANTIC_MAP_LEFT,
        right=PRE_POST_SEMANTIC_MAP_RIGHT,
        top=panel_top,
        bottom=PRE_POST_SEMANTIC_MAP_BOTTOM,
        hspace=row_hspace,
        wspace=col_wspace,
    )

    def _pin_legend_centered() -> None:
        """Center legend above the top row, flush above panel titles."""
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        top_title_y1 = max(
            (
                _axes_title_artist(ax)
                .get_window_extent(rend)
                .transformed(fig.transFigure.inverted())
                .y1
                for ax in axes[0]
                if _axes_title_artist(ax).get_text()
            ),
            default=panel_top,
        )
        legend_y = min(0.998, float(top_title_y1) + legend_h + legend_title_gap)
        legend.set_loc("upper center")
        legend.set_bbox_to_anchor((0.5, legend_y))

    _pin_legend_centered()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    if float(legend_bbox.y1) > 0.999:
        panel_top -= float(legend_bbox.y1) - 0.999
        fig.subplots_adjust(
            left=PRE_POST_SEMANTIC_MAP_LEFT,
            right=PRE_POST_SEMANTIC_MAP_RIGHT,
            top=panel_top,
            bottom=PRE_POST_SEMANTIC_MAP_BOTTOM,
            hspace=row_hspace,
            wspace=col_wspace,
        )
        _pin_legend_centered()

    # Fixed figure frame (no tight crop) so axes/legend alignment is stable.
    save_figure_pdf_svg(
        fig,
        outpath,
        bbox_inches=None,
        pad_inches=PRE_POST_SEMANTIC_MAP_PAD_INCHES,
    )
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


def plot_pre_post_centroid_boxplots(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    outpath: Path,
    *,
    collapse_human: bool,
) -> None:
    """2×4 centroid boxplots: row 1 Pre-ML, row 2 Post-ML (shared y-limits)."""
    if "pre-ML" not in phase_bundles or "post-ML" not in phase_bundles:
        raise ValueError("Need both pre-ML and post-ML bundles.")
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    distance_col = (
        "distance_to_collapsed_group_centroid"
        if collapse_human
        else "distance_to_group_centroid"
    )
    pre_by_task = dict(phase_bundles["pre-ML"])
    post_by_task = dict(phase_bundles["post-ML"])
    panels: list[tuple[str, str, dict]] = []
    for phase_key, phase_label, by_task in (
        ("pre-ML", "Pre-ML", pre_by_task),
        ("post-ML", "Post-ML", post_by_task),
    ):
        for task_key in DIVERSITY_TASK_PANEL_ORDER:
            if task_key not in by_task:
                raise ValueError(f"Missing task for centroid map: {task_key}")
            panels.append((task_key, phase_label, by_task[task_key]))

    row_hspace = 0.42
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20.0, 12.0),
        gridspec_kw={"hspace": row_hspace, "wspace": 0.22},
    )
    axes_flat = axes.ravel()
    for ax, (task_key, phase_label, bundle) in zip(axes_flat, panels):
        plot_data = prepare_centroid_distance_plot_data(
            bundle["point_metrics"],
            distance_col,
            group_col=group_col,
        )
        draw_centroid_boxplot_panel(
            ax,
            plot_data,
            panel_title=f"{task_label_from_key(task_key)}\n{phase_label}",
            show_group_xticklabels=False,
        )
        attach_panel_comparisons(ax, plot_data.get("comparisons") or [])

    # Shared y-axis across Pre/Post panels for visual comparison.
    y0 = min(ax.get_ylim()[0] for ax in axes_flat)
    y1 = max(ax.get_ylim()[1] for ax in axes_flat)
    for ax in axes_flat:
        ax.set_ylim(y0, y1)

    sample_df = phase_bundles["pre-ML"][0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, _ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
    )
    # Nature style: no on-figure title. Three-group figure: legend upper-right;
    # collapsed Human|GenAI keeps centered legend.
    if collapse_human:
        legend_n_rows = add_phase_grid_centroid_boxplot_legend(
            fig,
            handles,
            labels,
            centroid_boxplot_stat_legend_handles(),
            bbox_y=0.90 - VIZ_LEGEND_Y_SHIFT,
        )
        legend = fig.legends[-1] if fig.legends else None
        legend_anchor_x = 0.5
        if legend is not None:
            legend.set_bbox_to_anchor((legend_anchor_x, 0.90))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if legend is not None:
            legend_bbox = legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            top_limit = 0.96
            if float(legend_bbox.y1) > top_limit:
                legend.set_bbox_to_anchor(
                    (legend_anchor_x, 0.90 - (float(legend_bbox.y1) - top_limit))
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer).transformed(
                    fig.transFigure.inverted()
                )
            title_heights = []
            for ax in axes[0]:
                if not ax.title.get_text():
                    continue
                tb = ax.title.get_window_extent(renderer).transformed(
                    fig.transFigure.inverted()
                )
                title_heights.append(float(tb.y1 - ax.get_position().y1))
            title_reserve = max(title_heights) if title_heights else 0.05
            panel_top = float(legend_bbox.y0) - title_reserve - 0.02
        else:
            panel_top = figure_legend_panel_top(
                0.90,
                n_items=legend_n_rows,
                ncol=2,
                fig_height_in=float(fig.get_size_inches()[1]),
                legend_fontsize=PHASE_GRID_LEGEND_FONTSIZE,
                gap_below_legend=PHASE_GRID_BOX_LEGEND_GAP,
                row_spacing=1.65,
            )
    else:
        # Reserve a tight strip above panels; legend sits just above titles.
        legend_n_rows = add_phase_grid_centroid_boxplot_legend(
            fig,
            handles,
            labels,
            centroid_boxplot_stat_legend_handles(),
            loc="lower right",
            bbox_to_anchor=(0.99, 0.88),
        )
        legend = fig.legends[-1] if fig.legends else None
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_h = 0.09
        if legend is not None:
            legend_bbox = legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            legend_h = float(legend_bbox.y1 - legend_bbox.y0)
        title_heights = []
        for ax in axes[0]:
            if not ax.title.get_text():
                continue
            tb = ax.title.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            title_heights.append(float(tb.y1 - ax.get_position().y1))
        title_reserve = max(title_heights) if title_heights else 0.05
        # Small gap between legend bottom and panel titles.
        panel_top = 0.995 - legend_h - 0.008 - title_reserve

    fig.supylabel(
        "Cosine distance to group centroid",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=0.02,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=panel_top,
        bottom=0.06,
        hspace=row_hspace,
        wspace=0.22,
    )
    if legend is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        top_title_y1 = max(
            (
                ax.title.get_window_extent(renderer)
                .transformed(fig.transFigure.inverted())
                .y1
                for ax in axes[0]
                if ax.title.get_text()
            ),
            default=panel_top,
        )
        if collapse_human:
            overflow = float(top_title_y1) - (float(legend_bbox.y0) - 0.012)
            if overflow > 0:
                panel_top -= overflow
                fig.subplots_adjust(
                    left=0.07,
                    right=0.99,
                    top=panel_top,
                    bottom=0.06,
                    hspace=row_hspace,
                    wspace=0.22,
                )
        else:
            # Pin legend lower-right just above the top-row titles.
            legend.set_loc("lower right")
            legend.set_bbox_to_anchor((0.99, float(top_title_y1) + 0.016))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bbox = legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            if float(legend_bbox.y1) > 0.995:
                shift = float(legend_bbox.y1) - 0.995
                panel_top -= shift
                fig.subplots_adjust(
                    left=0.07,
                    right=0.99,
                    top=panel_top,
                    bottom=0.06,
                    hspace=row_hspace,
                    wspace=0.22,
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                top_title_y1 = max(
                    (
                        ax.title.get_window_extent(renderer)
                        .transformed(fig.transFigure.inverted())
                        .y1
                        for ax in axes[0]
                        if ax.title.get_text()
                    ),
                    default=panel_top,
                )
                legend.set_bbox_to_anchor((0.99, float(top_title_y1) + 0.016))

    save_figure_pdf_svg(fig, outpath)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


CENTROID_PRE_POST_BAR_WIDTH = 0.36
CENTROID_PRE_POST_HATCH = "///"

# Nature-style combined Pre|Post dispersion figure (no figure title / footnotes).
CENTROID_PAIRWISE_TWO_PANEL_FIG = "centroid_pairwise_pre_post_two_panel.png"
CENTROID_PAIRWISE_TWO_PANEL_FIGSIZE = (20.8, 10.4)
CENTROID_PAIRWISE_TWO_PANEL_WSPACE = 0.12
CENTROID_PAIRWISE_TWO_PANEL_LEGEND_Y = 1.02
CENTROID_PAIRWISE_TWO_PANEL_TITLES = (
    "a. Distance to group centroid",
    "b. Mean pairwise cosine distance",
)
CENTROID_PAIRWISE_TWO_PANEL_YLABELS = (
    "Mean cosine distance to group centroid",
    WITHIN_GROUP_VAR_PAIRWISE_YLABEL,
)

# Pooled across all task × effect cells.
# 1×4 left→right: centroid dist. | centroid bars | pairwise dist. | pairwise bars
WITHIN_GROUP_DISPERSION_POOLED_FIG = (
    "within_group_dispersion_pre_post_pooled_three_panel.png"
)
WITHIN_GROUP_DISPERSION_POOLED_FIG_3GROUP = (
    "within_group_dispersion_pre_post_pooled_three_panel_3group.png"
)
WITHIN_GROUP_DISPERSION_POOLED_FIG_COMBINED = (
    "within_group_dispersion.png"
)
WITHIN_GROUP_DISPERSION_POOLED_FIGSIZE = (22.0, 9.2)
WITHIN_GROUP_DISPERSION_POOLED_COMBINED_FIGSIZE = (23.4, 12.0)
WITHIN_GROUP_DISPERSION_POOLED_WSPACE = 0.08
WITHIN_GROUP_DISPERSION_POOLED_BOX_ASPECT = 1.0
WITHIN_GROUP_DISPERSION_POOLED_SPINE_WIDTH = 1.15
WITHIN_GROUP_DISPERSION_POOLED_SUBPLOT = dict(
    left=0.155, right=0.99, top=0.935, bottom=0.095
)
WITHIN_GROUP_DISPERSION_POOLED_TITLE_STEMS = (
    "Distance to group centroid distributions",
    "Pairwise distance distributions",
    "Mean distance to group centroid",
    "Mean pairwise distance",
)
WITHIN_GROUP_DISPERSION_POOLED_TITLES = tuple(
    f"{letter}. {stem}"
    for letter, stem in zip("abcd", WITHIN_GROUP_DISPERSION_POOLED_TITLE_STEMS)
)
WITHIN_GROUP_DISPERSION_POOLED_YLABELS = (
    "Distance to group centroid",
    "Density",
    "Mean distance to group centroid",
    "Mean pairwise distance",
)
WITHIN_GROUP_DISPERSION_POOLED_TITLE_FS = 21.0
WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS = 17.5
WITHIN_GROUP_DISPERSION_POOLED_TICK_FS = 17.5
WITHIN_GROUP_DISPERSION_POOLED_YTICK_FS = 19.5
WITHIN_GROUP_DISPERSION_POOLED_YLABEL_FS = 19.5
WITHIN_GROUP_DISPERSION_POOLED_XTICK_FS = 21.5
WITHIN_GROUP_DISPERSION_POOLED_DENSITY_XLABEL_FS = 21.5
WITHIN_GROUP_DISPERSION_POOLED_P_FS = 18.0
WITHIN_GROUP_DISPERSION_POOLED_TASK_KEY = "pooled"
WITHIN_GROUP_DISPERSION_POOLED_GROUP_X_COLLAPSED = {
    "Human": np.array([0.0, 0.55]),
    "GenAI": np.array([1.70, 2.25]),
}
WITHIN_GROUP_DISPERSION_POOLED_GROUP_X_3GROUP = {
    "student": np.array([0.0, 0.48]),
    "senior": np.array([1.35, 1.83]),
    "GenAI": np.array([2.70, 3.18]),
}
WITHIN_GROUP_DISPERSION_POOLED_XTICK_SHORT = {
    "student": "PhD",
    "senior": "Senior",
    "GenAI": "GenAI",
    "Human": "Humans",
}



def _pre_post_bars_summary_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "collapsed" in df.columns:
        collapsed = df["collapsed"].astype(bool)
        if collapsed.any():
            df = df.loc[collapsed].copy()
        else:
            df = df.loc[~collapsed].copy()
    df = df.loc[df["participant_group"].isin(GROUP_ORDER_COLLAPSED)].copy()
    return df


def _pre_post_phase_legend_handles() -> list:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label="Pre-ML",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PHASE_HATCH_COLOR,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=CENTROID_PRE_POST_HATCH,
            label="Post-ML",
        ),
    ]


def _draw_pre_post_human_genai_bars_on_axes(
    axes,
    summary_df: pd.DataFrame,
    *,
    pre_col: str,
    post_col: str,
    p_col: str,
) -> None:
    """Fill a 2×2 axes grid with Pre/Post Human|GenAI bars + paired brackets."""
    df = _pre_post_bars_summary_frame(summary_df)
    if df.empty:
        raise ValueError("No collapsed Human/GenAI rows for Pre|Post bar panel.")

    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    task_keys = sorted(df["task_key"].unique(), key=lambda key: order.get(key, 999))
    groups = [g for g in GROUP_ORDER_COLLAPSED if g in set(df["participant_group"])]
    group_x = {
        "Human": np.array([0.0, 0.42]),
        "GenAI": np.array([1.55, 1.97]),
    }
    bar_width = CENTROID_PRE_POST_BAR_WIDTH
    axes_flat = np.asarray(axes).ravel()
    ylim_top = within_group_var_figure_ylim_top(
        df,
        groups=groups,
        pre_col=pre_col,
        post_col=post_col,
    )
    yticks = within_group_var_yticks(ylim_top)

    for ax, task_key in zip(axes_flat, task_keys):
        task_df = df.loc[df["task_key"] == task_key]
        for group in groups:
            row = task_df.loc[task_df["participant_group"] == group].iloc[0]
            xs = group_x[group]
            color = GROUP_COLORS_COLLAPSED[group]
            pre_mean = float(row[pre_col])
            post_mean = float(row[post_col])
            pre_err_lo, pre_err_hi = ci_errorbar_offsets(
                pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
            )
            post_err_lo, post_err_hi = ci_errorbar_offsets(
                post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
            )

            ax.bar(
                xs[0],
                pre_mean,
                bar_width,
                color=color,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            ax.errorbar(
                xs[0],
                pre_mean,
                yerr=[[pre_err_lo], [pre_err_hi]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )
            post_bars = ax.bar(
                xs[1],
                post_mean,
                bar_width,
                color=color,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            post_bars[0].set_hatch(CENTROID_PRE_POST_HATCH)
            ax.errorbar(
                xs[1],
                post_mean,
                yerr=[[post_err_lo], [post_err_hi]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )

        ax.set_xticks([group_x[group].mean() for group in groups])
        ax.set_xticklabels(
            [display_label(group) for group in groups],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
        title = task_label_from_key(task_key).replace(
            "Second-Order Interactions", "Interactions"
        )
        ax.set_title(
            title,
            fontweight="bold",
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            pad=8,
        )
        ax.tick_params(axis="y", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
        ax.set_xlim(-0.35, group_x[groups[-1]][1] + 0.45)
        ax.set_ylim(0.0, ylim_top)
        ax.set_yticks(yticks)
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, task_key in zip(axes_flat, task_keys):
        task_df = df.loc[df["task_key"] == task_key]
        for group in groups:
            row = task_df.loc[task_df["participant_group"] == group].iloc[0]
            xs = group_x[group]
            pre_mean = float(row[pre_col])
            post_mean = float(row[post_col])
            _, pre_err_hi = ci_errorbar_offsets(
                pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
            )
            _, post_err_hi = ci_errorbar_offsets(
                post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
            )
            draw_paired_pre_post_bracket(
                ax,
                float(xs[0]),
                float(xs[1]),
                max(pre_mean + pre_err_hi, post_mean + post_err_hi),
                float(row[p_col]),
                fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
            )


def plot_centroid_pairwise_pre_post_two_panel(
    centroid_summary_df: pd.DataFrame,
    pairwise_summary_df: pd.DataFrame,
    outpath: Path,
) -> None:
    """Nature-style (a)|(b) Pre/Post bars: centroid + pairwise; no title/footnotes."""
    fig = plt.figure(figsize=CENTROID_PAIRWISE_TWO_PANEL_FIGSIZE)
    # Leave a thin top band for the Pre/Post legend (no figure title).
    fig.subplots_adjust(top=0.94)
    subfigs = fig.subfigures(1, 2, wspace=CENTROID_PAIRWISE_TWO_PANEL_WSPACE)
    panel_specs = (
        (
            centroid_summary_df,
            "pre_mean_cosine_distance",
            "post_mean_cosine_distance",
            "paired_p_one_sided_post_lt_pre",
            CENTROID_PAIRWISE_TWO_PANEL_TITLES[0],
            CENTROID_PAIRWISE_TWO_PANEL_YLABELS[0],
        ),
        (
            pairwise_summary_df,
            "pre_mean_pairwise_cosine_distance",
            "post_mean_pairwise_cosine_distance",
            "paired_p_one_sided_post_lt_pre",
            CENTROID_PAIRWISE_TWO_PANEL_TITLES[1],
            CENTROID_PAIRWISE_TWO_PANEL_YLABELS[1],
        ),
    )
    for subfig, (summary_df, pre_col, post_col, p_col, panel_title, ylabel) in zip(
        subfigs, panel_specs
    ):
        axes = subfig.subplots(
            2,
            2,
            gridspec_kw={
                "hspace": DIVERSITY_PRED_ROW_GAP,
                "wspace": DIVERSITY_PRED_COL_GAP,
            },
        )
        _draw_pre_post_human_genai_bars_on_axes(
            axes,
            summary_df,
            pre_col=pre_col,
            post_col=post_col,
            p_col=p_col,
        )
        subfig.suptitle(
            panel_title,
            fontsize=DIVERSITY_PRED_SUPTITLE_FONTSIZE,
            fontweight="bold",
            y=0.995,
        )
        subfig.supylabel(
            ylabel,
            fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
        )
        subfig.subplots_adjust(
            left=0.14,
            right=0.98,
            top=0.88,
            bottom=0.08,
            hspace=DIVERSITY_PRED_ROW_GAP,
            wspace=DIVERSITY_PRED_COL_GAP,
        )

    fig.legend(
        handles=_pre_post_phase_legend_handles(),
        labels=["Pre-ML", "Post-ML"],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=VIZ_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, CENTROID_PAIRWISE_TWO_PANEL_LEGEND_Y),
        borderaxespad=0.0,
    )
    save_figure_pdf_svg(fig, outpath, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


def save_centroid_pairwise_pre_post_two_panel(
    embeddings_root: Path,
    *,
    centroid_summary_df: pd.DataFrame | None = None,
    pairwise_summary_df: pd.DataFrame | None = None,
) -> Path | None:
    """Write the combined Nature-style bars figure when both summaries are available."""
    base = comparisons_pre_post_dir(
        embeddings_root, COMPARISONS_WITHIN_GROUP_VAR_SUBDIR
    )
    centroid_csv = base / WITHIN_GROUP_VAR_CENTROID_SUBDIR / WITHIN_GROUP_VAR_COLLAPSED_CSV
    pairwise_csv = base / WITHIN_GROUP_VAR_PAIRWISE_SUBDIR / WITHIN_GROUP_VAR_COLLAPSED_CSV
    if centroid_summary_df is None:
        if not centroid_csv.exists():
            return None
        centroid_summary_df = pd.read_csv(centroid_csv)
    if pairwise_summary_df is None:
        if not pairwise_csv.exists():
            return None
        pairwise_summary_df = pd.read_csv(pairwise_csv)

    outpath = base / CENTROID_PAIRWISE_TWO_PANEL_FIG
    plot_centroid_pairwise_pre_post_two_panel(
        centroid_summary_df,
        pairwise_summary_df,
        outpath,
    )
    return outpath


def plot_phase_grid_centroid_boxplots(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    distance_col = (
        "distance_to_collapsed_group_centroid"
        if collapse_human
        else "distance_to_group_centroid"
    )
    suptitle = f"Within-group distance to centroid ({phase_label})"
    fig, axes = make_phase_grid_axes()
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        plot_data = prepare_centroid_distance_plot_data(
            bundle["point_metrics"],
            distance_col,
            group_col=group_col,
        )
        draw_centroid_boxplot_panel(
            ax,
            plot_data,
            panel_title=task_label_from_key(task_key),
        )
        attach_panel_comparisons(ax, plot_data.get("comparisons") or [])

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, _ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
    )
    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=PHASE_GRID_CENTROID_BOX_METRIC,
        suptitle_fontsize=PHASE_GRID_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    stat_handles = centroid_boxplot_stat_legend_handles()
    legend_n_rows = add_phase_grid_centroid_boxplot_legend(
        fig,
        handles,
        labels,
        stat_handles,
        bbox_y=header.legend_y,
    )

    fig.supylabel(
        "Cosine distance to group centroid",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    legend_anchor_y = header.legend_y + VIZ_LEGEND_Y_SHIFT
    panel_top = figure_legend_panel_top(
        legend_anchor_y,
        n_items=legend_n_rows,
        ncol=2,
        fig_height_in=float(fig.get_size_inches()[1]),
        legend_fontsize=PHASE_GRID_LEGEND_FONTSIZE,
        gap_below_legend=PHASE_GRID_BOX_LEGEND_GAP,
        row_spacing=1.65,
    )
    phase_grid_layout_adjust(
        fig,
        top=panel_top,
        footnote_lines=len(CENTROID_MEAN_SIG_FOOTNOTE),
        bottom_extra=PHASE_GRID_BOX_FOOTNOTE_BOTTOM_EXTRA,
    )
    draw_phase_grid_footnote(
        fig,
        CENTROID_MEAN_SIG_FOOTNOTE,
        y=PHASE_GRID_BOX_FOOTNOTE_Y,
        line_step=PHASE_GRID_BOX_FOOTNOTE_LINE_STEP,
    )
    save_figure_pdf_svg(fig, outpath)


def plot_pre_post_pairwise_distributions(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    outpath: Path,
    *,
    collapse_human: bool,
) -> None:
    """2×4 pairwise-distance densities: row 1 Pre-ML, row 2 Post-ML."""
    if "pre-ML" not in phase_bundles or "post-ML" not in phase_bundles:
        raise ValueError("Need both pre-ML and post-ML bundles.")
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    pre_by_task = dict(phase_bundles["pre-ML"])
    post_by_task = dict(phase_bundles["post-ML"])
    panels: list[tuple[str, str, dict]] = []
    for phase_label, by_task in (
        ("Pre-ML", pre_by_task),
        ("Post-ML", post_by_task),
    ):
        for task_key in DIVERSITY_TASK_PANEL_ORDER:
            if task_key not in by_task:
                raise ValueError(f"Missing task for pairwise map: {task_key}")
            panels.append((task_key, phase_label, by_task[task_key]))

    row_hspace = 0.42
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20.0, 12.0),
        gridspec_kw={"hspace": row_hspace, "wspace": 0.22},
    )
    axes_flat = axes.ravel()
    for ax, (task_key, phase_label, bundle) in zip(axes_flat, panels):
        plot_data = prepare_pairwise_distance_plot_data(
            bundle["df"],
            bundle["X"],
            group_col=group_col,
        )
        draw_pairwise_distribution_panel(
            ax,
            plot_data,
            panel_title=f"{task_label_from_key(task_key)}\n{phase_label}",
        )

    x0 = min(ax.get_xlim()[0] for ax in axes_flat)
    x1 = max(ax.get_xlim()[1] for ax in axes_flat)
    y0 = min(ax.get_ylim()[0] for ax in axes_flat)
    y1 = max(ax.get_ylim()[1] for ax in axes_flat)
    for ax in axes_flat:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    sample_df = phase_bundles["pre-ML"][0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
        face_alpha=CENTROID_BOX_FACE_ALPHA,
    )
    legend_title_gap = 0.012
    legend = add_phase_grid_figure_legend(
        fig,
        handles,
        labels,
        ncol=ncol,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.88),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    legend_h = float(legend_bbox.y1 - legend_bbox.y0)
    title_heights = []
    for ax in axes[0]:
        if not ax.title.get_text():
            continue
        tb = ax.title.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        title_heights.append(float(tb.y1 - ax.get_position().y1))
    title_reserve = max(title_heights) if title_heights else 0.05
    panel_top = 0.995 - legend_h - legend_title_gap - title_reserve

    fig.supylabel(
        "Density",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=0.02,
    )
    fig.supxlabel(
        "Pairwise cosine distance",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=0.02,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=panel_top,
        bottom=0.08,
        hspace=row_hspace,
        wspace=0.22,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_title_y1 = max(
        (
            ax.title.get_window_extent(renderer)
            .transformed(fig.transFigure.inverted())
            .y1
            for ax in axes[0]
            if ax.title.get_text()
        ),
        default=panel_top,
    )
    legend.set_loc("lower right")
    legend.set_bbox_to_anchor((0.99, float(top_title_y1) + legend_title_gap))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    if float(legend_bbox.y1) > 0.995:
        panel_top -= float(legend_bbox.y1) - 0.995
        fig.subplots_adjust(
            left=0.07,
            right=0.99,
            top=panel_top,
            bottom=0.08,
            hspace=row_hspace,
            wspace=0.22,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top_title_y1 = max(
            (
                ax.title.get_window_extent(renderer)
                .transformed(fig.transFigure.inverted())
                .y1
                for ax in axes[0]
                if ax.title.get_text()
            ),
            default=panel_top,
        )
        legend.set_bbox_to_anchor((0.99, float(top_title_y1) + legend_title_gap))
    save_figure_pdf_svg(fig, outpath)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


def _collect_pooled_centroid_paired_vals(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    group: str,
    *,
    group_col: str,
    distance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate paired pre/post centroid distances across all task×effect cells."""
    pre_by_task = dict(phase_bundles["pre-ML"])
    post_by_task = dict(phase_bundles["post-ML"])
    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        pre_pm = pre_by_task[task_key]["point_metrics"]
        post_pm = post_by_task[task_key]["point_metrics"]
        pre_g = pre_pm.loc[
            pre_pm[group_col] == group,
            [PARTICIPANT_NAME_COL, distance_col],
        ].rename(columns={distance_col: "pre"})
        post_g = post_pm.loc[
            post_pm[group_col] == group,
            [PARTICIPANT_NAME_COL, distance_col],
        ].rename(columns={distance_col: "post"})
        merged = pre_g.merge(post_g, on=PARTICIPANT_NAME_COL, how="inner")
        if merged.empty:
            continue
        pre_parts.append(merged["pre"].to_numpy(dtype=float))
        post_parts.append(merged["post"].to_numpy(dtype=float))
    if not pre_parts:
        raise ValueError(f"No pooled centroid pairs for group {group!r}")
    return np.concatenate(pre_parts), np.concatenate(post_parts)


def _collect_pooled_pairwise_distance_vals(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    group: str,
    phase: str,
    *,
    group_col: str,
) -> np.ndarray:
    """Concatenate within-group pairwise distances across all task×effect cells."""
    by_task = dict(phase_bundles[phase])
    parts: list[np.ndarray] = []
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        bundle = by_task[task_key]
        series = pairwise_cosine_distance_group_series(
            bundle["df"],
            bundle["X"],
            group_col=group_col,
        )
        for g, vals, _n in series:
            if g == group:
                parts.append(np.asarray(vals, dtype=float))
                break
    if not parts:
        raise ValueError(
            f"No pooled pairwise distances for {group!r} ({phase})"
        )
    return np.concatenate(parts)


def _collect_pooled_paired_group_embedding_matrices(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    group: str,
    *,
    group_col: str,
    collapse_human: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Paired pre/post embedding matrices per task for one group."""
    pre_by_task = dict(phase_bundles["pre-ML"])
    post_by_task = dict(phase_bundles["post-ML"])
    pre_Xs: list[np.ndarray] = []
    post_Xs: list[np.ndarray] = []
    for task_key in DIVERSITY_TASK_PANEL_ORDER:
        pre_bundle = pre_by_task[task_key]
        post_bundle = post_by_task[task_key]
        pre_df = (
            with_collapsed_group(pre_bundle["df"])
            if collapse_human
            else pre_bundle["df"]
        )
        post_df = (
            with_collapsed_group(post_bundle["df"])
            if collapse_human
            else post_bundle["df"]
        )
        pre_names = pre_df[PARTICIPANT_NAME_COL].astype(str).to_numpy()
        post_names = post_df[PARTICIPANT_NAME_COL].astype(str).to_numpy()
        pre_labs = pre_df[group_col].to_numpy()
        post_labs = post_df[group_col].to_numpy()
        pre_name_to_i = {
            name: i
            for i, (name, lab) in enumerate(zip(pre_names, pre_labs))
            if lab == group
        }
        post_name_to_i = {
            name: i
            for i, (name, lab) in enumerate(zip(post_names, post_labs))
            if lab == group
        }
        common = sorted(set(pre_name_to_i) & set(post_name_to_i))
        if len(common) < 2:
            continue
        pre_idx = np.array([pre_name_to_i[n] for n in common], dtype=int)
        post_idx = np.array([post_name_to_i[n] for n in common], dtype=int)
        pre_Xs.append(np.asarray(pre_bundle["X"][pre_idx], dtype=float))
        post_Xs.append(np.asarray(post_bundle["X"][post_idx], dtype=float))
    if not pre_Xs:
        raise ValueError(f"No pooled paired embeddings for group {group!r}")
    return pre_Xs, post_Xs


def _pooled_dispersion_group_setup(*, collapse_human: bool) -> dict:
    if collapse_human:
        return {
            "collapse_human": True,
            "groups": list(GROUP_ORDER_COLLAPSED),
            "group_col": COLLAPSED_PARTICIPANT_TYPE_COL,
            "distance_col": "distance_to_collapsed_group_centroid",
            "color_map": GROUP_COLORS_COLLAPSED,
            "group_x": WITHIN_GROUP_DISPERSION_POOLED_GROUP_X_COLLAPSED,
            "group_label": display_label,
            "box_width": 0.42,
            "bar_width": CENTROID_PRE_POST_BAR_WIDTH,
        }
    return {
        "collapse_human": False,
        "groups": list(GROUP_ORDER),
        "group_col": PARTICIPANT_TYPE_COL,
        "distance_col": "distance_to_group_centroid",
        "color_map": GROUP_COLORS_BY_PARTICIPANT_TYPE,
        "group_x": WITHIN_GROUP_DISPERSION_POOLED_GROUP_X_3GROUP,
        "group_label": lambda g: PARTICIPANT_TYPE_TO_LEGEND.get(g, g),
        "box_width": 0.36,
        "bar_width": CENTROID_PRE_POST_BAR_WIDTH * 0.85,
    }


def compute_pooled_within_group_dispersion_summaries(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    *,
    collapse_human: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build bar summaries + raw pooled arrays for densities/boxplots."""
    if "pre-ML" not in phase_bundles or "post-ML" not in phase_bundles:
        raise ValueError("Need both pre-ML and post-ML bundles.")

    setup = _pooled_dispersion_group_setup(collapse_human=collapse_human)
    groups = setup["groups"]
    group_col = setup["group_col"]
    distance_col = setup["distance_col"]
    group_label = setup["group_label"]

    centroid_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    pooled_arrays: dict = {
        "pairwise": {},  # (group, phase) -> vals
        "centroid": {},  # (group, phase) -> vals
    }

    for group in groups:
        pre_c, post_c = _collect_pooled_centroid_paired_vals(
            phase_bundles,
            group,
            group_col=group_col,
            distance_col=distance_col,
        )
        pre_c_mean = float(np.mean(pre_c))
        post_c_mean = float(np.mean(post_c))
        pre_c_lo, pre_c_hi = bootstrap_mean_ci(pre_c, seed=ANALYSIS_SEED)
        post_c_lo, post_c_hi = bootstrap_mean_ci(post_c, seed=ANALYSIS_SEED + 1)
        paired_p_c = p_value_paired_one_sided_post_lt_pre(pre_c, post_c)
        centroid_rows.append({
            "metric_type": "centroid_distance",
            "task_key": WITHIN_GROUP_DISPERSION_POOLED_TASK_KEY,
            "task_label": "Pooled",
            "collapsed": collapse_human,
            "participant_group": group,
            "group_label": group_label(group),
            "n_paired": int(len(pre_c)),
            "pre_mean_cosine_distance": pre_c_mean,
            "post_mean_cosine_distance": post_c_mean,
            "pre_ci_low": pre_c_lo,
            "pre_ci_high": pre_c_hi,
            "post_ci_low": post_c_lo,
            "post_ci_high": post_c_hi,
            "paired_p_one_sided_post_lt_pre": paired_p_c,
            "paired_significance": significance_label(paired_p_c),
        })
        pooled_arrays["centroid"][(group, "pre-ML")] = pre_c
        pooled_arrays["centroid"][(group, "post-ML")] = post_c

        pre_p = _collect_pooled_pairwise_distance_vals(
            phase_bundles, group, "pre-ML", group_col=group_col
        )
        post_p = _collect_pooled_pairwise_distance_vals(
            phase_bundles, group, "post-ML", group_col=group_col
        )
        pre_Xs, post_Xs = _collect_pooled_paired_group_embedding_matrices(
            phase_bundles,
            group,
            group_col=group_col,
            collapse_human=collapse_human,
        )
        pre_p_mean = float(_pooled_mpwd_from_normalized_mats(
            [normalize_embedding_rows(X) for X in pre_Xs]
        ))
        post_p_mean = float(_pooled_mpwd_from_normalized_mats(
            [normalize_embedding_rows(X) for X in post_Xs]
        ))
        pre_p_lo, pre_p_hi = bootstrap_pooled_mpwd_ci(
            pre_Xs, seed=ANALYSIS_SEED + 2
        )
        post_p_lo, post_p_hi = bootstrap_pooled_mpwd_ci(
            post_Xs, seed=ANALYSIS_SEED + 3
        )
        paired_p_p = p_value_pooled_paired_permutation_mpwd_post_lt_pre(
            pre_Xs, post_Xs
        )
        pairwise_rows.append({
            "metric_type": "mean_pairwise_cosine_distance",
            "task_key": WITHIN_GROUP_DISPERSION_POOLED_TASK_KEY,
            "task_label": "Pooled",
            "collapsed": collapse_human,
            "participant_group": group,
            "group_label": group_label(group),
            "n_paired": int(len(pre_c)),
            "pre_mean_pairwise_cosine_distance": pre_p_mean,
            "post_mean_pairwise_cosine_distance": post_p_mean,
            "pre_ci_low": pre_p_lo,
            "pre_ci_high": pre_p_hi,
            "post_ci_low": post_p_lo,
            "post_ci_high": post_p_hi,
            "paired_p_one_sided_post_lt_pre": paired_p_p,
            "paired_significance": significance_label(paired_p_p),
        })
        pooled_arrays["pairwise"][(group, "pre-ML")] = pre_p
        pooled_arrays["pairwise"][(group, "post-ML")] = post_p

    return pd.DataFrame(centroid_rows), pd.DataFrame(pairwise_rows), pooled_arrays


def _draw_pooled_pre_post_bars_on_ax(
    ax,
    summary_df: pd.DataFrame,
    *,
    groups: list[str],
    group_x: dict[str, np.ndarray],
    color_map: dict,
    pre_col: str,
    post_col: str,
    p_col: str,
    bar_width: float,
    xtick_labels: dict[str, str],
) -> None:
    """Draw Pre/Post bars + paired significance brackets for one pooled panel."""
    df = summary_df.loc[summary_df["participant_group"].isin(groups)].copy()
    if df.empty:
        raise ValueError("No rows for pooled Pre|Post bar panel.")
    for group in groups:
        row = df.loc[df["participant_group"] == group].iloc[0]
        xs = group_x[group]
        color = color_map[group]
        pre_mean = float(row[pre_col])
        post_mean = float(row[post_col])
        pre_err_lo, pre_err_hi = ci_errorbar_offsets(
            pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
        )
        post_err_lo, post_err_hi = ci_errorbar_offsets(
            post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
        )
        ax.bar(
            xs[0],
            pre_mean,
            bar_width,
            color=color,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
        )
        ax.errorbar(
            xs[0],
            pre_mean,
            yerr=[[pre_err_lo], [pre_err_hi]],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )
        post_bars = ax.bar(
            xs[1],
            post_mean,
            bar_width,
            color=color,
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
        )
        post_bars[0].set_hatch(CENTROID_PRE_POST_HATCH)
        ax.errorbar(
            xs[1],
            post_mean,
            yerr=[[post_err_lo], [post_err_hi]],
            fmt="none",
            ecolor="black",
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            zorder=3,
        )

    ax.set_xticks([group_x[g].mean() for g in groups])
    ax.set_xticklabels(
        [xtick_labels.get(g, g) for g in groups],
        fontsize=WITHIN_GROUP_DISPERSION_POOLED_TICK_FS,
    )
    ax.set_xlim(-0.45, group_x[groups[-1]][1] + 0.45)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    for group in groups:
        row = df.loc[df["participant_group"] == group].iloc[0]
        xs = group_x[group]
        pre_mean = float(row[pre_col])
        post_mean = float(row[post_col])
        _, pre_err_hi = ci_errorbar_offsets(
            pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
        )
        _, post_err_hi = ci_errorbar_offsets(
            post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
        )
        p_val = float(row[p_col])
        draw_paired_pre_post_bracket(
            ax,
            float(xs[0]),
            float(xs[1]),
            max(pre_mean + pre_err_hi, post_mean + post_err_hi),
            p_val,
            fontsize=WITHIN_GROUP_DISPERSION_POOLED_P_FS,
            label=format_p_value_with_trailing_stars(p_val),
            fontweight="normal",
            color="black",
        )


def _pooled_panel_titles(title_letters: str | tuple[str, ...] = "abcd") -> tuple[str, ...]:
    letters = tuple(title_letters)
    if len(letters) != 4:
        raise ValueError(f"Need 4 panel letters, got {letters!r}")
    return tuple(
        f"{letter}. {stem}"
        for letter, stem in zip(letters, WITHIN_GROUP_DISPERSION_POOLED_TITLE_STEMS)
    )


def _draw_within_group_dispersion_pooled_into(
    legend_row,
    body,
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    *,
    collapse_human: bool,
    title_letters: str | tuple[str, ...] = "abcd",
) -> None:
    """Draw one pooled 1×4 plate into existing legend + body subfigures."""
    setup = _pooled_dispersion_group_setup(collapse_human=collapse_human)
    groups = setup["groups"]
    color_map = setup["color_map"]
    group_x = setup["group_x"]
    box_width = setup["box_width"]
    bar_width = setup["bar_width"]
    titles = _pooled_panel_titles(title_letters)
    xtick_labels = {
        g: WITHIN_GROUP_DISPERSION_POOLED_XTICK_SHORT.get(
            g, setup["group_label"](g)
        )
        for g in groups
    }

    centroid_summary_df, pairwise_summary_df, pooled = (
        compute_pooled_within_group_dispersion_summaries(
            phase_bundles, collapse_human=collapse_human
        )
    )

    sample_df = phase_bundles["pre-ML"][0][1]["df"]
    n_by_group = phase_grid_group_counts(
        sample_df, collapse_human=collapse_human
    )
    group_handles, group_labels, _ = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
        include_human_composition=collapse_human,
        face_alpha=BAR_ALPHA,
        italic_uppercase_n=True,
    )
    if collapse_human:
        # Prefer Senior Scientists first in the Humans composition note.
        group_labels = [
            (
                f"{display_label('Human')} "
                f"($\\mathit{{N}}$ = {n_by_group.get('Human', 0)}, "
                "Senior Scientists + PhD Students)"
                if g == "Human"
                else lab
            )
            for g, lab in zip(groups, group_labels)
        ]
    phase_handles = _pre_post_phase_legend_handles()
    individual_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="#6E6E6E",
        markerfacecolor="#6E6E6E",
        markeredgecolor="white",
        markeredgewidth=0.4,
        markersize=5.5,
        linestyle="None",
        label="Individual",
    )
    panel_b_legend_handles = [
        Patch(
            facecolor="#D0D0D0",
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label="Box (IQR)",
        ),
        Line2D(
            [0],
            [0],
            color=BOXPLOT_STAT_COLOR,
            linewidth=1.15,
            linestyle="-",
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=BOXPLOT_STAT_COLOR,
            markerfacecolor=BOXPLOT_STAT_COLOR,
            markeredgecolor=BOXPLOT_STAT_COLOR,
            markersize=5,
            linestyle="None",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            color=BOXPLOT_EDGE_COLOR,
            linewidth=BOXPLOT_WHISKER_WIDTH,
            label="Whiskers (1.5×IQR)",
        ),
        individual_handle,
    ]
    shared_handles = group_handles + phase_handles
    shared_labels = group_labels + ["Pre-ML", "Post-ML"]
    legend_ncol = len(shared_handles)

    legend_row.legend(
        handles=shared_handles,
        labels=shared_labels,
        loc="center",
        ncol=legend_ncol,
        frameon=True,
        fontsize=19.5,
        bbox_to_anchor=(0.5, 0.48),
        borderaxespad=0.0,
        columnspacing=1.6,
        handletextpad=0.55,
        handlelength=1.6,
    )

    panel_a, panel_b, panel_c, panel_d = body.subfigures(
        1, 4, wspace=WITHIN_GROUP_DISPERSION_POOLED_WSPACE
    )

    def _style_pooled_panel_title(ax, title: str) -> None:
        # Clear default title; place text so we can align to y-axis label.
        ax.set_title("")
        ax._pooled_panel_title = ax.text(
            0.0,
            1.045,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=WITHIN_GROUP_DISPERSION_POOLED_TITLE_FS,
            fontweight="bold",
            linespacing=1.0,
            clip_on=False,
        )

    def _style_pooled_axis_fonts(ax) -> None:
        ax.tick_params(axis="x", labelsize=WITHIN_GROUP_DISPERSION_POOLED_XTICK_FS)
        ax.tick_params(axis="y", labelsize=WITHIN_GROUP_DISPERSION_POOLED_YTICK_FS)
        for lab in ax.get_xticklabels():
            lab.set_fontsize(WITHIN_GROUP_DISPERSION_POOLED_XTICK_FS)
        for lab in ax.get_yticklabels():
            lab.set_fontsize(WITHIN_GROUP_DISPERSION_POOLED_YTICK_FS)
        ax.xaxis.label.set_fontsize(WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS)
        ax.yaxis.label.set_fontsize(WITHIN_GROUP_DISPERSION_POOLED_YLABEL_FS)
        ax.xaxis.label.set_fontweight("normal")
        ax.yaxis.label.set_fontweight("normal")
        for spine in ax.spines.values():
            spine.set_linewidth(WITHIN_GROUP_DISPERSION_POOLED_SPINE_WIDTH)

    def _square_axes(ax) -> None:
        ax.set_box_aspect(WITHIN_GROUP_DISPERSION_POOLED_BOX_ASPECT)
        # Keep the square plot flush under the title (not vertically centered).
        ax.set_anchor("N")

    def _draw_pooled_mean_bars(
        subfig,
        summary_df,
        *,
        pre_col,
        post_col,
        p_col,
        title,
        ylabel,
        ylim_top: float,
        ytick_step: float | None = None,
    ):
        ax = subfig.subplots(1, 1)
        _draw_pooled_pre_post_bars_on_ax(
            ax,
            summary_df,
            groups=groups,
            group_x=group_x,
            color_map=color_map,
            pre_col=pre_col,
            post_col=post_col,
            p_col=p_col,
            bar_width=bar_width,
            xtick_labels=xtick_labels,
        )
        ax.set_title("")
        _square_axes(ax)
        ax.set_ylim(0.0, ylim_top)
        step = WITHIN_GROUP_VAR_YTICK_STEP if ytick_step is None else ytick_step
        ax.set_yticks(np.arange(0.0, ylim_top + step * 0.01, step))
        _style_pooled_panel_title(ax, title)
        ax.set_ylabel(
            ylabel,
            fontsize=WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS,
            fontweight="normal",
        )
        _style_pooled_axis_fonts(ax)
        subfig.subplots_adjust(**WITHIN_GROUP_DISPERSION_POOLED_SUBPLOT)
        return ax

    # --- a. Centroid distance boxplots + individual points ---
    box_ax = panel_a.subplots(1, 1)
    for group in groups:
        color = color_map[group]
        for phase, x, hatch in (
            ("pre-ML", group_x[group][0], None),
            ("post-ML", group_x[group][1], CENTROID_PRE_POST_HATCH),
        ):
            vals = np.asarray(pooled["centroid"][(group, phase)], dtype=float)
            # Individuals behind the box so median/mean stay visible.
            box_ax.scatter(
                np.full(len(vals), x, dtype=float),
                vals,
                s=22 if not collapse_human else 28,
                color=color,
                alpha=0.45,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
            bp = box_ax.boxplot(
                [vals],
                positions=[x],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                whis=1.5,
                medianprops={
                    "color": BOXPLOT_STAT_COLOR,
                    "linewidth": 1.15,
                    "linestyle": "-",
                    "solid_capstyle": "butt",
                    "zorder": 5,
                },
                meanprops={
                    "marker": "^",
                    "markerfacecolor": BOXPLOT_STAT_COLOR,
                    "markeredgecolor": BOXPLOT_STAT_COLOR,
                    "markersize": 5.5,
                    "zorder": 6,
                },
                whiskerprops={
                    "color": BOXPLOT_EDGE_COLOR,
                    "linewidth": 1.1,
                    "zorder": 4,
                },
                capprops={
                    "color": BOXPLOT_EDGE_COLOR,
                    "linewidth": 1.1,
                    "zorder": 4,
                },
                boxprops={
                    "linewidth": BAR_EDGE_WIDTH,
                    "edgecolor": BAR_EDGE_COLOR,
                    "zorder": 3,
                },
                zorder=3,
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(BAR_ALPHA)
            if hatch is not None:
                bp["boxes"][0].set_hatch(hatch)
            for med in bp["medians"]:
                med.set_zorder(5)
                med.set_linewidth(1.15)
                med.set_linestyle("-")
                med.set_color(BOXPLOT_STAT_COLOR)
            for mean_art in bp["means"]:
                mean_art.set_zorder(6)

    box_ax.set_xticks([group_x[g].mean() for g in groups])
    box_ax.set_xticklabels(
        [xtick_labels[g] for g in groups],
        fontsize=WITHIN_GROUP_DISPERSION_POOLED_TICK_FS,
    )
    box_ax.set_xlim(-0.55, group_x[groups[-1]][1] + 0.55)
    box_ax.set_ylim(0.0, 0.6)
    _square_axes(box_ax)
    for side in ("top", "right", "bottom", "left"):
        box_ax.spines[side].set_visible(True)
        box_ax.spines[side].set_linewidth(WITHIN_GROUP_DISPERSION_POOLED_SPINE_WIDTH)
    box_ax.grid(axis="y", alpha=0.25, zorder=0)
    _style_pooled_panel_title(box_ax, titles[0])
    box_ax.set_ylabel(
        WITHIN_GROUP_DISPERSION_POOLED_YLABELS[0],
        fontsize=WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS,
        fontweight="normal",
    )
    _style_pooled_axis_fonts(box_ax)
    panel_a_annot_fs = max(WITHIN_GROUP_DISPERSION_POOLED_TICK_FS - 2.0, 8.0)
    box_ax.legend(
        handles=panel_b_legend_handles,
        labels=[h.get_label() for h in panel_b_legend_handles],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        frameon=True,
        fontsize=panel_a_annot_fs,
        borderaxespad=0.15,
        labelspacing=0.28,
        handlelength=1.25,
        handletextpad=0.35,
        borderpad=0.28,
    )
    panel_a.subplots_adjust(**WITHIN_GROUP_DISPERSION_POOLED_SUBPLOT)

    # --- b. Pairwise distance densities ---
    dens_ax = panel_b.subplots(1, 1)
    all_pw = np.concatenate(
        [
            pooled["pairwise"][(g, ph)]
            for g in groups
            for ph in ("pre-ML", "post-ML")
        ]
    )
    n_bins = pairwise_histogram_bin_count(len(all_pw))
    shared_bins = np.linspace(
        float(np.min(all_pw)), float(np.max(all_pw)), n_bins + 1
    )
    # Draw all Pre first, then all Post, so Post sits above Pre.
    for group in groups:
        color = color_map[group]
        pre_h = dens_ax.hist(
            pooled["pairwise"][(group, "pre-ML")],
            bins=shared_bins,
            density=True,
            alpha=0.55,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
            zorder=2,
        )
        for patch in pre_h[2]:
            patch.set_zorder(2)
    for group in groups:
        color = color_map[group]
        post_h = dens_ax.hist(
            pooled["pairwise"][(group, "post-ML")],
            bins=shared_bins,
            density=True,
            alpha=0.82,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH * 0.6,
            zorder=3,
        )
        for patch in post_h[2]:
            patch.set_hatch(CENTROID_PRE_POST_HATCH)
            patch.set_zorder(3)
    for side in ("top", "right", "bottom", "left"):
        dens_ax.spines[side].set_visible(True)
        dens_ax.spines[side].set_linewidth(WITHIN_GROUP_DISPERSION_POOLED_SPINE_WIDTH)
    _square_axes(dens_ax)
    dens_ax.set_ylim(0.0, 13.0)
    dens_ax.set_yticks(np.arange(0.0, 13.0 + 0.01, 2.0))
    dens_ax.grid(axis="y", alpha=0.25, zorder=0)
    dens_ax.set_xlabel(
        "Pairwise distance",
        fontsize=WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS,
        fontweight="normal",
    )
    dens_ax.set_ylabel(
        WITHIN_GROUP_DISPERSION_POOLED_YLABELS[1],
        fontsize=WITHIN_GROUP_DISPERSION_POOLED_AXIS_FS,
        fontweight="normal",
    )
    _style_pooled_axis_fonts(dens_ax)
    dens_ax.xaxis.label.set_fontsize(WITHIN_GROUP_DISPERSION_POOLED_DENSITY_XLABEL_FS)
    _style_pooled_panel_title(dens_ax, titles[1])
    panel_b.subplots_adjust(**WITHIN_GROUP_DISPERSION_POOLED_SUBPLOT)

    # --- c. Mean centroid distance bars ---
    _draw_pooled_mean_bars(
        panel_c,
        centroid_summary_df,
        pre_col="pre_mean_cosine_distance",
        post_col="post_mean_cosine_distance",
        p_col="paired_p_one_sided_post_lt_pre",
        title=titles[2],
        ylabel=WITHIN_GROUP_DISPERSION_POOLED_YLABELS[2],
        ylim_top=0.3,
    )

    # --- d. Mean pairwise distance bars ---
    _draw_pooled_mean_bars(
        panel_d,
        pairwise_summary_df,
        pre_col="pre_mean_pairwise_cosine_distance",
        post_col="post_mean_pairwise_cosine_distance",
        p_col="paired_p_one_sided_post_lt_pre",
        title=titles[3],
        ylabel=WITHIN_GROUP_DISPERSION_POOLED_YLABELS[3],
        ylim_top=0.6,
        ytick_step=0.1,
    )


def _align_pooled_panel_titles_to_ylabel_left(fig) -> None:
    """Shift panel titles so left edge matches the y-axis text label's left."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        text = getattr(ax, "_pooled_panel_title", None)
        if text is None:
            continue
        ylab = ax.yaxis.label
        if ylab is None or not str(ylab.get_text()).strip():
            continue
        left_px = float(ylab.get_window_extent(renderer).x0)
        x_axes = float(ax.transAxes.inverted().transform((left_px, 0.0))[0])
        text.set_position((x_axes, 1.045))


def plot_within_group_dispersion_pooled_three_panel(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    outpath: Path,
    *,
    collapse_human: bool = True,
    title_letters: str | tuple[str, ...] = "abcd",
) -> None:
    """Pooled Pre|Post dispersion in a 1×4 plate.

    Left→right: centroid distributions | pairwise densities |
    centroid mean bars | pairwise mean bars.

    collapse_human=True → Humans|GenAI; False → PhD|Senior|GenAI.
    """
    fig = plt.figure(figsize=WITHIN_GROUP_DISPERSION_POOLED_FIGSIZE)
    legend_row, body = fig.subfigures(
        2,
        1,
        height_ratios=(0.10, 1.0),
        hspace=0.06,
    )
    _draw_within_group_dispersion_pooled_into(
        legend_row,
        body,
        phase_bundles,
        collapse_human=collapse_human,
        title_letters=title_letters,
    )
    _align_pooled_panel_titles_to_ylabel_left(fig)
    save_figure_pdf_svg(fig, outpath, bbox_inches="tight", pad_inches=0.30)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


def plot_within_group_dispersion_pooled_combined(
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    outpath: Path,
) -> None:
    """Stack Humans|GenAI (a–d) above PhD|Senior|GenAI (e–h) into one 2×4 plate."""
    # Keep figure height close to 2×(square panel + legend). A taller canvas
    # leaves empty space under each N-anchored square row, which stacks into a
    # large visual gap between the two plates.
    fig = plt.figure(figsize=WITHIN_GROUP_DISPERSION_POOLED_COMBINED_FIGSIZE)
    top_row, bottom_row = fig.subfigures(
        2,
        1,
        height_ratios=(1.0, 1.0),
        hspace=0.08,
    )
    for row_subfig, collapse_human, letters, legend_hspace in (
        (top_row, True, "abcd", 0.10),
        (bottom_row, False, "efgh", 0.12),
    ):
        legend_row, body = row_subfig.subfigures(
            2,
            1,
            height_ratios=(0.12, 1.0),
            hspace=legend_hspace,
        )
        _draw_within_group_dispersion_pooled_into(
            legend_row,
            body,
            phase_bundles,
            collapse_human=collapse_human,
            title_letters=letters,
        )
    _align_pooled_panel_titles_to_ylabel_left(fig)
    save_figure_pdf_svg(fig, outpath, bbox_inches="tight", pad_inches=0.20)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")



def save_within_group_dispersion_pooled_three_panel(
    embeddings_root: Path,
    phase_bundles: dict[str, List[Tuple[str, dict]]],
    *,
    collapse_human: bool = True,
) -> Path:
    """Write pooled three-panel under outputs/within_group_variability/."""
    base = comparisons_pre_post_dir(
        embeddings_root, COMPARISONS_WITHIN_GROUP_VAR_SUBDIR
    )
    base.mkdir(parents=True, exist_ok=True)
    outpath = base / (
        WITHIN_GROUP_DISPERSION_POOLED_FIG
        if collapse_human
        else WITHIN_GROUP_DISPERSION_POOLED_FIG_3GROUP
    )
    plot_within_group_dispersion_pooled_three_panel(
        phase_bundles,
        outpath,
        collapse_human=collapse_human,
    )
    return outpath


def save_within_group_dispersion_pooled_combined(
    embeddings_root: Path,
    phase_bundles: dict[str, List[Tuple[str, dict]]],
) -> Path:
    """Write stacked Humans|GenAI + PhD|Senior|GenAI 2×4 pooled plate."""
    base = comparisons_pre_post_dir(
        embeddings_root, COMPARISONS_WITHIN_GROUP_VAR_SUBDIR
    )
    base.mkdir(parents=True, exist_ok=True)
    outpath = base / WITHIN_GROUP_DISPERSION_POOLED_FIG_COMBINED
    plot_within_group_dispersion_pooled_combined(phase_bundles, outpath)
    return outpath


def plot_phase_grid_pairwise_distributions(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    group_col = (
        COLLAPSED_PARTICIPANT_TYPE_COL
        if collapse_human
        else PARTICIPANT_TYPE_COL
    )
    suptitle = f"Within-group pairwise-distance distribution ({phase_label})"
    fig, axes = make_phase_grid_axes()
    for ax, (task_key, bundle) in zip(axes, task_bundles):
        plot_data = prepare_pairwise_distance_plot_data(
            bundle["df"],
            bundle["X"],
            group_col=group_col,
        )
        draw_pairwise_distribution_panel(
            ax,
            plot_data,
            panel_title=task_label_from_key(task_key),
        )

    sample_df = task_bundles[0][1]["df"]
    n_by_group = phase_grid_group_counts(sample_df, collapse_human=collapse_human)
    handles, labels, ncol = phase_grid_group_legend_handles_labels(
        collapse_human=collapse_human,
        n_by_group=n_by_group,
        face_alpha=CENTROID_BOX_FACE_ALPHA,
    )
    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=PHASE_GRID_CENTROID_DENSITY_METRIC,
        suptitle_fontsize=PHASE_GRID_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    add_phase_grid_figure_legend(
        fig,
        handles,
        labels,
        ncol=ncol,
        bbox_y=header.legend_y,
    )

    fig.supylabel(
        "Density",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        x=PHASE_GRID_SUPYLABEL_X,
    )
    fig.supxlabel(
        "Pairwise cosine distance",
        fontsize=PHASE_GRID_AXIS_FONTSIZE,
        fontweight="bold",
        y=0.045,
    )
    phase_grid_layout_adjust(
        fig,
        bottom_extra=0.02,
        top=header.panel_top,
    )
    save_figure_pdf_svg(fig, outpath)
    print(f"Saved figure: {outpath.with_suffix('.png')} (+ svg)")


def plot_phase_grid_centroid_distributions(
    task_bundles: List[Tuple[str, dict]],
    outpath: Path,
    *,
    phase_label: str,
    collapse_human: bool,
) -> None:
    """Deprecated alias — use plot_phase_grid_pairwise_distributions."""
    plot_phase_grid_pairwise_distributions(
        task_bundles,
        outpath,
        phase_label=phase_label,
        collapse_human=collapse_human,
    )


def run_phase_grid_visualizations(
    embeddings_root: Path,
    embedding_col: str,
    *,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    seed: int = ANALYSIS_SEED,
    umap_neighbors: int = UMAP_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
) -> None:
    phase_bundles: dict[str, List[Tuple[str, dict]]] = {}
    for phase in PHASE_NAMES:
        task_dirs = discover_phase_task_dirs(embeddings_root, phase)
        if len(task_dirs) != len(DIVERSITY_TASK_PANEL_ORDER):
            found = [key for key, _ in task_dirs]
            missing = [k for k in DIVERSITY_TASK_PANEL_ORDER if k not in found]
            print(
                f"Skipping {phase} phase grid: expected "
                f"{len(DIVERSITY_TASK_PANEL_ORDER)} tasks, "
                f"missing {missing}."
            )
            continue

        task_bundles: List[Tuple[str, dict]] = []
        for task_key, set_dir in task_dirs:
            print(f"  Loading {phase} · {task_label_from_key(task_key)}")
            task_bundles.append((
                task_key,
                load_phase_task_bundle(
                    set_dir,
                    embedding_col,
                    threshold_quantile=threshold_quantile,
                ),
            ))
        phase_bundles[phase] = task_bundles

    if len(phase_bundles) == len(PHASE_NAMES):
        compare_outdir = comparisons_pre_post_dir(
            embeddings_root, "semantic_map_PCA"
        )
        compare_outdir.mkdir(parents=True, exist_ok=True)
        plot_pre_post_semantic_maps_expanded(
            phase_bundles,
            compare_outdir / "semantic_space_map.png",
            collapse_human=True,
            seed=seed,
        )
        plot_pre_post_semantic_maps_expanded(
            phase_bundles,
            compare_outdir / "semantic_space_map_three_groups.png",
            collapse_human=False,
            seed=seed,
        )
        print(f"Saved Pre|Post expanded semantic maps to: {compare_outdir}")

        pooled_combined_path = save_within_group_dispersion_pooled_combined(
            embeddings_root,
            phase_bundles,
        )
        print(
            "Saved within-group dispersion pooled combined: "
            f"{pooled_combined_path}"
        )


# ---------------------------------------------------------------------
# Plot 07 — Human vs GenAI diversity predictions (pre/post, collapsed)
# ---------------------------------------------------------------------

def discover_pre_post_task_pairs(embeddings_root: Path) -> List[Tuple[Path, Path, str]]:
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


def diversity_space_dir(
    embedding_set_dir: Path,
    embedding_col: str,
    embeddings_root: Path | None = None,
) -> Path:
    return resolve_output_dir(embedding_set_dir, embedding_col, embeddings_root)


def diversity_csv_paths(embedding_set_dir: Path, embedding_col: str) -> dict[str, Path]:
    collapsed = "_collapsed"
    space_dir = diversity_space_dir(embedding_set_dir, embedding_col)
    return {
        "summary": space_dir / Q4_DIVERSITY_SUMMARY_CSV.format(suffix=collapsed),
        "pairwise": space_dir / Q4_DIVERSITY_PAIRWISE_CSV.format(suffix=collapsed),
    }


def require_diversity_csvs(paths: dict[str, Path], embedding_set_dir: Path) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing diversity CSV(s). Run analysis on this embedding set first:\n"
            f"  {embedding_set_dir}\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def summary_group_value(summary_df: pd.DataFrame, group: str, column: str) -> float:
    match = summary_df.loc[summary_df["group"] == group]
    if match.empty:
        raise ValueError(
            f"No row for {group!r} in diversity summary "
            f"(available: {summary_df['group'].tolist()})"
        )
    return float(match.iloc[0][column])


def human_vs_genai_pairwise_row(pairwise_df: pd.DataFrame) -> pd.Series:
    match = pairwise_df.loc[
        (pairwise_df["metric"] == "centroid_distance")
        & (pairwise_df["left_group"] == HUMAN_COLLAPSED_GROUP)
        & (pairwise_df["right_group"] == GENAI_COLLAPSED_GROUP)
    ]
    if match.empty:
        raise ValueError(
            "No Human vs GenAI centroid_distance row in diversity pairwise CSV."
        )
    return match.iloc[0]


def group_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


def analyze_diversity_prediction_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
) -> dict:
    pre_dists = participant_collapsed_centroid_distances(pre_dir, embedding_col)
    post_dists = participant_collapsed_centroid_distances(post_dir, embedding_col)

    pre_human = group_mean(pre_dists[HUMAN_COLLAPSED_GROUP])
    pre_genai = group_mean(pre_dists[GENAI_COLLAPSED_GROUP])
    post_human = group_mean(post_dists[HUMAN_COLLAPSED_GROUP])
    post_genai = group_mean(post_dists[GENAI_COLLAPSED_GROUP])

    pre_p = p_value_welch_ttest_one_sided(
        pre_dists[HUMAN_COLLAPSED_GROUP],
        pre_dists[GENAI_COLLAPSED_GROUP],
        alternative="greater",
    )
    post_p = p_value_welch_ttest_one_sided(
        post_dists[HUMAN_COLLAPSED_GROUP],
        post_dists[GENAI_COLLAPSED_GROUP],
        alternative="greater",
    )
    pre_gap = abs(pre_human - pre_genai)
    post_gap = abs(post_human - post_genai)

    pre_human_ci_lo, pre_human_ci_hi = bootstrap_mean_ci(
        pre_dists[HUMAN_COLLAPSED_GROUP], seed=ANALYSIS_SEED
    )
    pre_genai_ci_lo, pre_genai_ci_hi = bootstrap_mean_ci(
        pre_dists[GENAI_COLLAPSED_GROUP], seed=ANALYSIS_SEED + 1
    )
    post_human_ci_lo, post_human_ci_hi = bootstrap_mean_ci(
        post_dists[HUMAN_COLLAPSED_GROUP], seed=ANALYSIS_SEED + 2
    )
    post_genai_ci_lo, post_genai_ci_hi = bootstrap_mean_ci(
        post_dists[GENAI_COLLAPSED_GROUP], seed=ANALYSIS_SEED + 3
    )

    pre_human_more_dispersed = pre_human > pre_genai
    pre_p2a_supported = bool(pre_human_more_dispersed and pre_p < 0.05)
    post_human_more_dispersed = post_human > post_genai
    post_p2b_supported = bool(post_p >= 0.05)

    return {
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "metric_type": "centroid_distance",
        "pre_human_mean_cosine_distance": pre_human,
        "pre_genai_mean_cosine_distance": pre_genai,
        "post_human_mean_cosine_distance": post_human,
        "post_genai_mean_cosine_distance": post_genai,
        "pre_human_ci_low": pre_human_ci_lo,
        "pre_human_ci_high": pre_human_ci_hi,
        "pre_genai_ci_low": pre_genai_ci_lo,
        "pre_genai_ci_high": pre_genai_ci_hi,
        "post_human_ci_low": post_human_ci_lo,
        "post_human_ci_high": post_human_ci_hi,
        "post_genai_ci_low": post_genai_ci_lo,
        "post_genai_ci_high": post_genai_ci_hi,
        "pre_human_minus_genai_gap": pre_human - pre_genai,
        "post_human_minus_genai_gap": post_human - post_genai,
        "pre_abs_gap": pre_gap,
        "post_abs_gap": post_gap,
        "abs_gap_delta_post_minus_pre": post_gap - pre_gap,
        "pre_welch_p_one_sided": pre_p,
        "post_welch_p_one_sided": post_p,
        "pre_significance": significance_label(pre_p),
        "post_significance": significance_label(post_p),
        "pre_human_more_dispersed_than_genai": pre_human_more_dispersed,
        "pre_p2a_supported": pre_p2a_supported,
        "post_human_more_dispersed_than_genai": post_human_more_dispersed,
        "post_p2b_supported": post_p2b_supported,
    }


def analyze_diversity_prediction_task_pairwise(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
) -> dict:
    X_pre_human = collapsed_group_embedding_matrix(
        pre_dir, embedding_col, HUMAN_COLLAPSED_GROUP
    )
    X_pre_genai = collapsed_group_embedding_matrix(
        pre_dir, embedding_col, GENAI_COLLAPSED_GROUP
    )
    X_post_human = collapsed_group_embedding_matrix(
        post_dir, embedding_col, HUMAN_COLLAPSED_GROUP
    )
    X_post_genai = collapsed_group_embedding_matrix(
        post_dir, embedding_col, GENAI_COLLAPSED_GROUP
    )

    pre_human = mean_pairwise_cosine_distance(X_pre_human)
    pre_genai = mean_pairwise_cosine_distance(X_pre_genai)
    post_human = mean_pairwise_cosine_distance(X_post_human)
    post_genai = mean_pairwise_cosine_distance(X_post_genai)

    pre_p = p_value_permutation_mpwd_group_greater(X_pre_human, X_pre_genai)
    post_p = p_value_permutation_mpwd_group_greater(X_post_human, X_post_genai)
    pre_gap = abs(pre_human - pre_genai)
    post_gap = abs(post_human - post_genai)

    pre_human_ci_lo, pre_human_ci_hi = bootstrap_mpwd_ci(X_pre_human, seed=ANALYSIS_SEED)
    pre_genai_ci_lo, pre_genai_ci_hi = bootstrap_mpwd_ci(X_pre_genai, seed=ANALYSIS_SEED + 1)
    post_human_ci_lo, post_human_ci_hi = bootstrap_mpwd_ci(
        X_post_human, seed=ANALYSIS_SEED + 2
    )
    post_genai_ci_lo, post_genai_ci_hi = bootstrap_mpwd_ci(
        X_post_genai, seed=ANALYSIS_SEED + 3
    )

    pre_human_more_dispersed = pre_human > pre_genai
    pre_p2a_supported = bool(pre_human_more_dispersed and pre_p < 0.05)
    post_human_more_dispersed = post_human > post_genai
    post_p2b_supported = bool(post_p >= 0.05)

    return {
        "metric_type": "mean_pairwise_cosine_distance",
        "task_key": task_key,
        "task_label": task_label_from_key(task_key),
        "embedding_col": embedding_col,
        "pre_human_mean_pairwise_cosine_distance": pre_human,
        "pre_genai_mean_pairwise_cosine_distance": pre_genai,
        "post_human_mean_pairwise_cosine_distance": post_human,
        "post_genai_mean_pairwise_cosine_distance": post_genai,
        "pre_human_ci_low": pre_human_ci_lo,
        "pre_human_ci_high": pre_human_ci_hi,
        "pre_genai_ci_low": pre_genai_ci_lo,
        "pre_genai_ci_high": pre_genai_ci_hi,
        "post_human_ci_low": post_human_ci_lo,
        "post_human_ci_high": post_human_ci_hi,
        "post_genai_ci_low": post_genai_ci_lo,
        "post_genai_ci_high": post_genai_ci_hi,
        "pre_human_minus_genai_gap": pre_human - pre_genai,
        "post_human_minus_genai_gap": post_human - post_genai,
        "pre_abs_gap": pre_gap,
        "post_abs_gap": post_gap,
        "abs_gap_delta_post_minus_pre": post_gap - pre_gap,
        "pre_welch_p_one_sided": pre_p,
        "post_welch_p_one_sided": post_p,
        "pre_significance": significance_label(pre_p),
        "post_significance": significance_label(post_p),
        "pre_human_more_dispersed_than_genai": pre_human_more_dispersed,
        "pre_p2a_supported": pre_p2a_supported,
        "post_human_more_dispersed_than_genai": post_human_more_dispersed,
        "post_p2b_supported": post_p2b_supported,
    }


def diversity_panel_ylim_top(panel_max: float) -> float:
    raw = panel_max * DIVERSITY_PRED_YLIM_TOP_PAD
    step = 0.02 if raw < 0.3 else 0.05
    return float(np.ceil(raw / step) * step)


def within_group_var_figure_ylim_top(
    summary_df: pd.DataFrame,
    *,
    groups: list[str],
    pre_col: str,
    post_col: str,
) -> float:
    """Shared y-axis top for all panels in a within-group variability figure."""
    figure_max = 0.0
    for task_key in summary_df["task_key"].unique():
        task_df = summary_df.loc[summary_df["task_key"] == task_key]
        for group in groups:
            row = task_df.loc[task_df["participant_group"] == group].iloc[0]
            pre_mean = float(row[pre_col])
            post_mean = float(row[post_col])
            _, pre_err_hi = ci_errorbar_offsets(
                pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
            )
            _, post_err_hi = ci_errorbar_offsets(
                post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
            )
            figure_max = max(figure_max, pre_mean + pre_err_hi, post_mean + post_err_hi)
    return diversity_panel_ylim_top(figure_max)


def mean_col_to_ci_cols(mean_col: str) -> tuple[str, str]:
    prefix = mean_col.split("_mean_", 1)[0]
    return f"{prefix}_ci_low", f"{prefix}_ci_high"


def diversity_pred_figure_ylim_top(
    plot_df: pd.DataFrame,
    value_cols: tuple[str, ...],
) -> float:
    """Shared y-axis top for all panels in a human-vs-GenAI pre/post figure."""
    figure_max = 0.0
    for _, row in plot_df.iterrows():
        for col in value_cols:
            val = float(row[col])
            ci_lo_col, ci_hi_col = mean_col_to_ci_cols(col)
            if ci_lo_col in row.index and ci_hi_col in row.index:
                _, err_hi = ci_errorbar_offsets(
                    val, float(row[ci_lo_col]), float(row[ci_hi_col])
                )
                figure_max = max(figure_max, val + err_hi)
            else:
                figure_max = max(figure_max, val)
    return diversity_panel_ylim_top(figure_max)


def diversity_pred_xlim() -> tuple[float, float]:
    left = (
        DIVERSITY_PRED_PRE_X[0]
        - DIVERSITY_PRED_BAR_WIDTH / 2
        - DIVERSITY_PRED_X_MARGIN
    )
    right = (
        DIVERSITY_PRED_POST_X[1]
        + DIVERSITY_PRED_BAR_WIDTH / 2
        + DIVERSITY_PRED_X_MARGIN
    )
    return left, right


def footnote_visual_line_count(footnote: tuple[str, ...]) -> int:
    return sum(line.count("\n") + 1 for line in footnote)


def diversity_comparison_bottom_layout(
    footnote: tuple[str, ...],
) -> tuple[float, float]:
    """Return (footnote_y_top, subplot_bottom) stacked from figure bottom up."""
    n = footnote_visual_line_count(footnote)
    step = DIVERSITY_PRED_FOOTNOTE_LINE_STEP
    footnote_y = COMPARISON_FOOTNOTE_BOTTOM_PAD + (n - 1) * step
    footnote_top = footnote_y + COMPARISON_FOOTNOTE_LINE_HEIGHT
    subplot_bottom = (
        footnote_top
        + COMPARISON_FOOTNOTE_XLABEL_GAP
        + COMPARISON_FOOTNOTE_XLABEL_HEIGHT
    )
    return footnote_y, subplot_bottom


def draw_diversity_prediction_footnote(
    fig,
    footnote_lines: tuple[str, ...],
    y: float = DIVERSITY_PRED_FOOTNOTE_Y,
) -> None:
    for i, line in enumerate(footnote_lines):
        fig.text(
            0.5,
            y - i * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=DIVERSITY_PRED_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def diversity_pred_bar_top_with_ci(row: pd.Series, mean_col: str) -> float:
    val = float(row[mean_col])
    ci_lo_col, ci_hi_col = mean_col_to_ci_cols(mean_col)
    if ci_lo_col in row.index and ci_hi_col in row.index:
        _, err_hi = ci_errorbar_offsets(
            val, float(row[ci_lo_col]), float(row[ci_hi_col])
        )
        return val + err_hi
    return val


def draw_diversity_pred_bar_with_ci(
    ax,
    x: float,
    row: pd.Series,
    mean_col: str,
    *,
    color: str,
) -> None:
    mean = float(row[mean_col])
    ax.bar(
        x,
        mean,
        DIVERSITY_PRED_BAR_WIDTH,
        color=color,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=2,
    )
    ci_lo_col, ci_hi_col = mean_col_to_ci_cols(mean_col)
    if ci_lo_col not in row.index or ci_hi_col not in row.index:
        return
    err_lo, err_hi = ci_errorbar_offsets(
        mean, float(row[ci_lo_col]), float(row[ci_hi_col])
    )
    ax.errorbar(
        x,
        mean,
        yerr=[[err_lo], [err_hi]],
        fmt="none",
        ecolor="black",
        elinewidth=ERROR_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        zorder=3,
    )


def plot_diversity_pre_post_predictions(
    summary_df: pd.DataFrame,
    outpath: Path,
    *,
    human_pre_col: str,
    genai_pre_col: str,
    human_post_col: str,
    genai_post_col: str,
    p_pre_col: str,
    p_post_col: str,
    ylabel: str,
    metric_subtitle: str,
    footnote: tuple[str, ...],
) -> None:
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    plot_df = summary_df.sort_values(
        "task_key",
        key=lambda s: s.map(order),
    ).reset_index(drop=True)

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
    human_color = GROUP_COLORS_COLLAPSED[HUMAN_COLLAPSED_GROUP]
    genai_color = GROUP_COLORS_COLLAPSED[GENAI_COLLAPSED_GROUP]
    ylim_top = diversity_pred_figure_ylim_top(
        plot_df,
        (human_pre_col, genai_pre_col, human_post_col, genai_post_col),
    )
    yticks = within_group_var_yticks(ylim_top)

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        draw_diversity_pred_bar_with_ci(
            ax,
            DIVERSITY_PRED_PRE_X[0],
            row,
            human_pre_col,
            color=human_color,
        )
        draw_diversity_pred_bar_with_ci(
            ax,
            DIVERSITY_PRED_PRE_X[1],
            row,
            genai_pre_col,
            color=genai_color,
        )
        draw_diversity_pred_bar_with_ci(
            ax,
            DIVERSITY_PRED_POST_X[0],
            row,
            human_post_col,
            color=human_color,
        )
        draw_diversity_pred_bar_with_ci(
            ax,
            DIVERSITY_PRED_POST_X[1],
            row,
            genai_post_col,
            color=genai_color,
        )

        ax.set_xticks(
            [DIVERSITY_PRED_PRE_X.mean(), DIVERSITY_PRED_POST_X.mean()]
        )
        ax.set_xticklabels(
            ["pre-ML", "post-ML"],
            fontsize=DIVERSITY_PRED_XTICK_FONTSIZE,
        )
        ax.set_title(
            row["task_label"],
            fontweight="bold",
            fontsize=DIVERSITY_PRED_PANEL_TITLE_FONTSIZE,
            pad=10,
        )
        ax.tick_params(axis="y", labelsize=DIVERSITY_PRED_YTICK_FONTSIZE)
        ax.set_xlim(*diversity_pred_xlim())
        ax.set_ylim(0.0, ylim_top)
        ax.set_yticks(yticks)
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows()):
        pre_top = max(
            diversity_pred_bar_top_with_ci(row, human_pre_col),
            diversity_pred_bar_top_with_ci(row, genai_pre_col),
        )
        post_top = max(
            diversity_pred_bar_top_with_ci(row, human_post_col),
            diversity_pred_bar_top_with_ci(row, genai_post_col),
        )
        draw_paired_pre_post_bracket(
            ax,
            DIVERSITY_PRED_PRE_X[0],
            DIVERSITY_PRED_PRE_X[1],
            pre_top,
            float(row[p_pre_col]),
            fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
        )
        draw_paired_pre_post_bracket(
            ax,
            DIVERSITY_PRED_POST_X[0],
            DIVERSITY_PRED_POST_X[1],
            post_top,
            float(row[p_post_col]),
            fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
        )

    fig.supylabel(
        ylabel,
        fontweight="bold",
        x=DIVERSITY_PRED_YLABEL_X,
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    footnote_y, subplot_bottom = diversity_comparison_bottom_layout(footnote)
    header = layout_title_and_metric(
        fig,
        suptitle=DIVERSITY_PRED_SUPTITLE,
        metric_lines=metric_subtitle,
        suptitle_fontsize=DIVERSITY_PRED_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=header.panel_top,
        bottom=subplot_bottom,
        hspace=DIVERSITY_PRED_ROW_GAP,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=human_color, label=display_label(HUMAN_COLLAPSED_GROUP)),
        plt.Rectangle((0, 0), 1, 1, color=genai_color, label="GenAI"),
    ]
    fig.legend(
        legend_handles,
        [display_label(HUMAN_COLLAPSED_GROUP), "GenAI"],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=VIZ_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, header.legend_y + VIZ_LEGEND_Y_SHIFT),
        borderaxespad=0.0,
    )
    draw_diversity_prediction_footnote(fig, footnote, y=footnote_y)
    save_figure_pdf_svg(fig, outpath)


def participant_collapsed_centroid_distance_tables(
    embedding_set_dir: Path,
    embedding_col: str,
) -> dict[str, pd.DataFrame]:
    """Per-respondent cosine distance to collapsed within-group centroid."""
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    plot_df = with_collapsed_group(df)
    collapsed = plot_df[COLLAPSED_PARTICIPANT_TYPE_COL].values
    names = plot_df[PARTICIPANT_NAME_COL].values
    out: dict[str, pd.DataFrame] = {}
    for group in (HUMAN_COLLAPSED_GROUP, GENAI_COLLAPSED_GROUP):
        idx = np.where(collapsed == group)[0]
        if len(idx) == 0:
            raise ValueError(f"No {group!r} rows in {embedding_set_dir}")
        centroid = group_centroid(X[idx])
        dists = cosine_distances(X[idx], centroid.reshape(1, -1)).ravel()
        out[group] = pd.DataFrame(
            {
                PARTICIPANT_NAME_COL: names[idx],
                "cosine_distance": dists,
            }
        )
    return out


def participant_collapsed_centroid_distances(
    embedding_set_dir: Path,
    embedding_col: str,
) -> dict[str, np.ndarray]:
    tables = participant_collapsed_centroid_distance_tables(
        embedding_set_dir, embedding_col
    )
    return {
        group: table["cosine_distance"].to_numpy(dtype=float)
        for group, table in tables.items()
    }


def collapsed_group_embedding_matrix(
    embedding_set_dir: Path,
    embedding_col: str,
    group: str,
) -> np.ndarray:
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    plot_df = with_collapsed_group(df)
    sub = plot_df.loc[plot_df[COLLAPSED_PARTICIPANT_TYPE_COL] == group]
    if sub.empty:
        raise ValueError(f"No {group!r} rows in {embedding_set_dir}")
    return normalize(stack_embeddings(sub, embedding_col))


def participant_three_group_centroid_distance_tables(
    embedding_set_dir: Path,
    embedding_col: str,
) -> dict[str, pd.DataFrame]:
    """Per-respondent cosine distance to that respondent's own group centroid."""
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    groups = df[PARTICIPANT_TYPE_COL].values
    names = df[PARTICIPANT_NAME_COL].values
    out: dict[str, pd.DataFrame] = {}
    for group in GROUP_ORDER:
        idx = np.where(groups == group)[0]
        if len(idx) == 0:
            continue
        centroid = group_centroid(X[idx])
        dists = cosine_distances(X[idx], centroid.reshape(1, -1)).ravel()
        out[group] = pd.DataFrame(
            {
                PARTICIPANT_NAME_COL: names[idx],
                "cosine_distance": dists,
            }
        )
    return out


def paired_group_embedding_matrices(
    pre_dir: Path,
    post_dir: Path,
    embedding_col: str,
    group: str,
    *,
    collapsed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (n, d) embedding matrices for one group across pre/post phases."""
    pre_df = pd.read_parquet(pre_dir / "embeddings_wide.parquet")
    post_df = pd.read_parquet(post_dir / "embeddings_wide.parquet")
    if collapsed:
        pre_df = with_collapsed_group(pre_df)
        post_df = with_collapsed_group(post_df)
        group_col = COLLAPSED_PARTICIPANT_TYPE_COL
    else:
        group_col = PARTICIPANT_TYPE_COL

    pre_g = pre_df.loc[pre_df[group_col] == group, [PARTICIPANT_NAME_COL, embedding_col]]
    post_g = post_df.loc[post_df[group_col] == group, [PARTICIPANT_NAME_COL, embedding_col]]
    merged = pre_g.merge(post_g, on=PARTICIPANT_NAME_COL, suffixes=("_pre", "_post"))
    if merged.empty:
        raise ValueError(f"No paired rows for group {group!r}")

    X_pre = normalize(stack_embeddings(
        merged.rename(columns={f"{embedding_col}_pre": embedding_col}),
        embedding_col,
    ))
    X_post = normalize(stack_embeddings(
        merged.rename(columns={f"{embedding_col}_post": embedding_col}),
        embedding_col,
    ))
    return X_pre, X_post


def analyze_within_group_variability_task(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
    *,
    collapsed: bool = False,
) -> list[dict]:
    if collapsed:
        pre_tables = participant_collapsed_centroid_distance_tables(
            pre_dir, embedding_col
        )
        post_tables = participant_collapsed_centroid_distance_tables(
            post_dir, embedding_col
        )
        groups = GROUP_ORDER_COLLAPSED
        group_label = display_label
    else:
        pre_tables = participant_three_group_centroid_distance_tables(
            pre_dir, embedding_col
        )
        post_tables = participant_three_group_centroid_distance_tables(
            post_dir, embedding_col
        )
        groups = GROUP_ORDER
        group_label = lambda group: PARTICIPANT_TYPE_TO_LEGEND[group]

    rows: list[dict] = []
    for group in groups:
        if group not in pre_tables or group not in post_tables:
            raise ValueError(
                f"Missing group {group!r} in pre/post tables for task {task_key}"
            )
        merged = pre_tables[group].merge(
            post_tables[group],
            on=PARTICIPANT_NAME_COL,
            how="inner",
            suffixes=("_pre", "_post"),
        )
        if merged.empty:
            raise ValueError(
                f"No paired {group!r} respondents for task {task_key}"
            )
        pre_vals = merged["cosine_distance_pre"].to_numpy(dtype=float)
        post_vals = merged["cosine_distance_post"].to_numpy(dtype=float)
        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        pre_ci_lo, pre_ci_hi = bootstrap_mean_ci(pre_vals, seed=ANALYSIS_SEED)
        post_ci_lo, post_ci_hi = bootstrap_mean_ci(post_vals, seed=ANALYSIS_SEED + 1)
        paired_p = p_value_paired_one_sided_post_lt_pre(pre_vals, post_vals)
        post_lower = post_mean < pre_mean
        rows.append({
            "metric_type": "centroid_distance",
            "task_key": task_key,
            "task_label": task_label_from_key(task_key),
            "embedding_col": embedding_col,
            "collapsed": collapsed,
            "participant_group": group,
            "group_label": group_label(group),
            "n_paired": len(merged),
            "pre_mean_cosine_distance": pre_mean,
            "post_mean_cosine_distance": post_mean,
            "pre_ci_low": pre_ci_lo,
            "pre_ci_high": pre_ci_hi,
            "post_ci_low": post_ci_lo,
            "post_ci_high": post_ci_hi,
            "mean_delta_post_minus_pre": post_mean - pre_mean,
            "paired_p_one_sided_post_lt_pre": paired_p,
            "paired_significance": significance_label(paired_p),
            "post_lower_than_pre": post_lower,
            "directional_supported": bool(post_lower and paired_p < 0.05),
        })
    return rows


def analyze_within_group_variability_task_pairwise(
    pre_dir: Path,
    post_dir: Path,
    task_key: str,
    embedding_col: str,
    *,
    collapsed: bool = False,
) -> list[dict]:
    if collapsed:
        groups = GROUP_ORDER_COLLAPSED
        group_label = display_label
    else:
        groups = GROUP_ORDER
        group_label = lambda group: PARTICIPANT_TYPE_TO_LEGEND[group]

    rows: list[dict] = []
    for group in groups:
        X_pre, X_post = paired_group_embedding_matrices(
            pre_dir, post_dir, embedding_col, group, collapsed=collapsed
        )
        pre_mean = mean_pairwise_cosine_distance(X_pre)
        post_mean = mean_pairwise_cosine_distance(X_post)
        pre_ci_lo, pre_ci_hi = bootstrap_mpwd_ci(X_pre, seed=ANALYSIS_SEED)
        post_ci_lo, post_ci_hi = bootstrap_mpwd_ci(X_post, seed=ANALYSIS_SEED + 1)
        paired_p = p_value_paired_permutation_mpwd_post_lt_pre(X_pre, X_post)
        post_lower = post_mean < pre_mean
        rows.append({
            "metric_type": "mean_pairwise_cosine_distance",
            "task_key": task_key,
            "task_label": task_label_from_key(task_key),
            "embedding_col": embedding_col,
            "collapsed": collapsed,
            "participant_group": group,
            "group_label": group_label(group),
            "n_paired": len(X_pre),
            "pre_mean_pairwise_cosine_distance": pre_mean,
            "post_mean_pairwise_cosine_distance": post_mean,
            "pre_ci_low": pre_ci_lo,
            "pre_ci_high": pre_ci_hi,
            "post_ci_low": post_ci_lo,
            "post_ci_high": post_ci_hi,
            "mean_delta_post_minus_pre": post_mean - pre_mean,
            "paired_p_one_sided_post_lt_pre": paired_p,
            "paired_significance": significance_label(paired_p),
            "post_lower_than_pre": post_lower,
            "directional_supported": bool(post_lower and paired_p < 0.05),
        })
    return rows


def within_group_var_xlim(
    group_x: dict[str, np.ndarray],
    groups: list[str],
    *,
    bar_width: float = WITHIN_GROUP_VAR_BAR_WIDTH,
) -> tuple[float, float]:
    left = (
        group_x[groups[0]][0]
        - bar_width / 2
        - WITHIN_GROUP_VAR_X_MARGIN
    )
    right = (
        group_x[groups[-1]][1]
        + bar_width / 2
        + WITHIN_GROUP_VAR_X_MARGIN
    )
    return left, right


def within_group_var_yticks(ylim_top: float) -> np.ndarray:
    step = WITHIN_GROUP_VAR_YTICK_STEP
    return np.arange(WITHIN_GROUP_VAR_YTICK_MIN, ylim_top + step * 0.01, step)


def draw_within_group_var_footnote(
    fig,
    footnote_lines: tuple[str, ...],
    y: float = DIVERSITY_PRED_FOOTNOTE_Y,
) -> None:
    for i, line in enumerate(footnote_lines):
        fig.text(
            0.5,
            y - i * DIVERSITY_PRED_FOOTNOTE_LINE_STEP,
            line,
            ha="center",
            va="bottom",
            fontsize=DIVERSITY_PRED_FOOTNOTE_FONTSIZE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )


def plot_within_group_variability_pre_post(
    summary_df: pd.DataFrame,
    outpath: Path,
    *,
    groups: list[str],
    group_x: dict[str, np.ndarray],
    group_labels: dict[str, str],
    suptitle: str,
    pre_col: str,
    post_col: str,
    p_col: str,
    ylabel: str,
    metric_subtitle: str,
    footnote: tuple[str, ...],
    bar_width: float = WITHIN_GROUP_VAR_BAR_WIDTH,
) -> None:
    order = {key: i for i, key in enumerate(DIVERSITY_TASK_PANEL_ORDER)}
    task_keys = sorted(
        summary_df["task_key"].unique(),
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
    axes_flat = axes.ravel()
    ylim_top = within_group_var_figure_ylim_top(
        summary_df,
        groups=groups,
        pre_col=pre_col,
        post_col=post_col,
    )
    yticks = within_group_var_yticks(ylim_top)

    for ax, task_key in zip(axes_flat, task_keys):
        task_df = summary_df.loc[summary_df["task_key"] == task_key]

        for group in groups:
            row = task_df.loc[task_df["participant_group"] == group].iloc[0]
            xs = group_x[group]
            pre_mean = float(row[pre_col])
            post_mean = float(row[post_col])
            pre_lo = float(row["pre_ci_low"])
            pre_hi = float(row["pre_ci_high"])
            post_lo = float(row["post_ci_low"])
            post_hi = float(row["post_ci_high"])
            pre_err_lo, pre_err_hi = ci_errorbar_offsets(pre_mean, pre_lo, pre_hi)
            post_err_lo, post_err_hi = ci_errorbar_offsets(post_mean, post_lo, post_hi)

            ax.bar(
                xs[0],
                pre_mean,
                bar_width,
                color=WITHIN_GROUP_VAR_PRE_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            ax.errorbar(
                xs[0],
                pre_mean,
                yerr=[[pre_err_lo], [pre_err_hi]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )
            ax.bar(
                xs[1],
                post_mean,
                bar_width,
                color=WITHIN_GROUP_VAR_POST_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )
            ax.errorbar(
                xs[1],
                post_mean,
                yerr=[[post_err_lo], [post_err_hi]],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                zorder=3,
            )

        ax.set_xticks([group_x[group].mean() for group in groups])
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
        ax.set_xlim(*within_group_var_xlim(group_x, groups, bar_width=bar_width))
        ax.set_ylim(0.0, ylim_top)
        ax.set_yticks(yticks)
        ax.set_box_aspect(DIVERSITY_PRED_BOX_ASPECT)
        ax.grid(axis="y", alpha=0.25)

    for ax, task_key in zip(axes_flat, task_keys):
        task_df = summary_df.loc[summary_df["task_key"] == task_key]
        for group in groups:
            row = task_df.loc[task_df["participant_group"] == group].iloc[0]
            xs = group_x[group]
            pre_mean = float(row[pre_col])
            post_mean = float(row[post_col])
            _, pre_err_hi = ci_errorbar_offsets(
                pre_mean, float(row["pre_ci_low"]), float(row["pre_ci_high"])
            )
            _, post_err_hi = ci_errorbar_offsets(
                post_mean, float(row["post_ci_low"]), float(row["post_ci_high"])
            )
            draw_paired_pre_post_bracket(
                ax,
                xs[0],
                xs[1],
                max(pre_mean + pre_err_hi, post_mean + post_err_hi),
                float(row[p_col]),
                fontsize=DIVERSITY_PRED_BRACKET_FONTSIZE,
            )

    fig.supylabel(
        ylabel,
        fontweight="bold",
        x=DIVERSITY_PRED_YLABEL_X,
        fontsize=DIVERSITY_PRED_YLABEL_FONTSIZE,
    )
    footnote_y, subplot_bottom = diversity_comparison_bottom_layout(footnote)
    header = layout_title_and_metric(
        fig,
        suptitle=suptitle,
        metric_lines=metric_subtitle,
        suptitle_fontsize=DIVERSITY_PRED_SUPTITLE_FONTSIZE,
        suptitle_line_spacing=VIZ_SUPTITLE_LINE_SPACING,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=header.panel_top,
        bottom=subplot_bottom,
        hspace=DIVERSITY_PRED_ROW_GAP,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WITHIN_GROUP_VAR_PRE_COLOR, label="pre-ML"),
        plt.Rectangle((0, 0), 1, 1, color=WITHIN_GROUP_VAR_POST_COLOR, label="post-ML"),
    ]
    fig.legend(
        handles,
        ["pre-ML", "post-ML"],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=VIZ_LEGEND_FONTSIZE,
        bbox_to_anchor=(
            0.5,
            header.legend_y + VIZ_LEGEND_Y_SHIFT + WITHIN_GROUP_VAR_LEGEND_Y_SHIFT,
        ),
        borderaxespad=0.0,
    )
    draw_within_group_var_footnote(fig, footnote, y=footnote_y)
    save_figure_pdf_svg(fig, outpath)


def _within_group_var_three_group_labels() -> dict[str, str]:
    return {group: PARTICIPANT_TYPE_TO_LEGEND[group] for group in GROUP_ORDER}


def _within_group_var_collapsed_labels() -> dict[str, str]:
    return {group: display_label(group) for group in GROUP_ORDER_COLLAPSED}


def run_within_group_variability_variant(
    embeddings_root: Path,
    embedding_col: str,
    outdir: Path,
    *,
    collapsed: bool,
    csv_name: str,
    fig_name: str,
    suptitle: str,
    groups: list[str],
    group_x: dict[str, np.ndarray],
    group_labels: dict[str, str],
    variant_label: str,
    analyze_task,
    pre_col: str,
    post_col: str,
    p_col: str,
    ylabel: str,
    metric_subtitle: str,
    footnote: tuple[str, ...],
    bar_width: float = WITHIN_GROUP_VAR_BAR_WIDTH,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pre_dir, post_dir, task_key in discover_pre_post_task_pairs(embeddings_root):
        print(f"\n=== Within-group variability ({variant_label}) · {task_label_from_key(task_key)} ===")
        task_rows = analyze_task(
            pre_dir, post_dir, task_key, embedding_col, collapsed=collapsed
        )
        rows.extend(task_rows)
        for row in task_rows:
            print(
                f"  {row['group_label']}: pre {row[pre_col]:.4f} "
                f"→ post {row[post_col]:.4f} "
                f"(Δ {row['mean_delta_post_minus_pre']:+.4f}) | "
                f"Paired directional p: {row[p_col]:.4f} "
                f"{row['paired_significance']} | "
                f"Supported: {row['directional_supported']}"
            )

    summary_df = pd.DataFrame(rows)
    csv_path = outdir / csv_name
    fig_path = outdir / fig_name
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_within_group_variability_pre_post(
        summary_df,
        fig_path,
        groups=groups,
        group_x=group_x,
        group_labels=group_labels,
        suptitle=suptitle,
        pre_col=pre_col,
        post_col=post_col,
        p_col=p_col,
        ylabel=ylabel,
        metric_subtitle=metric_subtitle,
        footnote=footnote,
        bar_width=bar_width,
    )

    n_groups = len(summary_df)
    n_supported = int(summary_df["directional_supported"].sum())
    print(
        f"\nWithin-group variability ({variant_label}) summary: directional support "
        f"{n_supported}/{n_groups} group×task cells"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    return summary_df


def _run_human_vs_genai_by_phase(
    embeddings_root: Path,
    embedding_col: str,
    outdir: Path,
    *,
    analyze_task,
    human_pre_col: str,
    genai_pre_col: str,
    human_post_col: str,
    genai_post_col: str,
    p_pre_col: str,
    p_post_col: str,
    ylabel: str,
    metric_subtitle: str,
    footnote: tuple[str, ...],
    metric_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pre_dir, post_dir, task_key in discover_pre_post_task_pairs(embeddings_root):
        print(
            f"\n=== Human vs GenAI ({metric_label}) · "
            f"{task_label_from_key(task_key)} ==="
        )
        row = analyze_task(pre_dir, post_dir, task_key, embedding_col)
        rows.append(row)
        print(
            f"  Pre: Human {row[human_pre_col]:.4f} vs GenAI {row[genai_pre_col]:.4f} "
            f"({row['pre_significance']}) | Pre supported: {row['pre_p2a_supported']}"
        )
        print(
            f"  Post: Human {row[human_post_col]:.4f} vs GenAI {row[genai_post_col]:.4f} "
            f"({row['post_significance']}) | "
            f"gap {row['pre_abs_gap']:.4f} → {row['post_abs_gap']:.4f} | "
            f"Post converged (NS) supported: {row['post_p2b_supported']}"
        )

    summary_df = pd.DataFrame(rows)
    csv_path = outdir / HUMAN_VS_GENAI_PRE_POST_CSV
    fig_path = outdir / HUMAN_VS_GENAI_PRE_POST_FILENAME
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_diversity_pre_post_predictions(
        summary_df,
        fig_path,
        human_pre_col=human_pre_col,
        genai_pre_col=genai_pre_col,
        human_post_col=human_post_col,
        genai_post_col=genai_post_col,
        p_pre_col=p_pre_col,
        p_post_col=p_post_col,
        ylabel=ylabel,
        metric_subtitle=metric_subtitle,
        footnote=footnote,
    )
    n_p2a = int(summary_df["pre_p2a_supported"].sum())
    n_p2b = int(summary_df["post_p2b_supported"].sum())
    n_tasks = len(summary_df)
    print(
        f"\nHuman vs GenAI ({metric_label}) summary across {n_tasks} tasks: "
        f"pre Human > GenAI {n_p2a}/{n_tasks}, post NS {n_p2b}/{n_tasks}"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    return summary_df


def _run_within_group_variability_metric_family(
    embeddings_root: Path,
    embedding_col: str,
    outdir: Path,
    *,
    analyze_task,
    pre_col: str,
    post_col: str,
    p_col: str,
    ylabel: str,
    metric_subtitle: str,
    footnote: tuple[str, ...],
    metric_label: str,
    analyze_human_vs_genai_task,
    human_pre_col: str,
    genai_pre_col: str,
    human_post_col: str,
    genai_post_col: str,
    human_vs_genai_footnote: tuple[str, ...],
    human_vs_genai_metric_subtitle: str,
    human_vs_genai_ylabel: str,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    run_within_group_variability_variant(
        embeddings_root,
        embedding_col,
        outdir,
        collapsed=False,
        csv_name=WITHIN_GROUP_VAR_CSV,
        fig_name=WITHIN_GROUP_VAR_FILENAME,
        suptitle=WITHIN_GROUP_VAR_SUPTITLE,
        groups=GROUP_ORDER,
        group_x=WITHIN_GROUP_VAR_GROUP_X,
        group_labels=_within_group_var_three_group_labels(),
        variant_label=f"{metric_label} · PhD Students, Senior Scientists, GenAI",
        analyze_task=analyze_task,
        pre_col=pre_col,
        post_col=post_col,
        p_col=p_col,
        ylabel=ylabel,
        metric_subtitle=metric_subtitle,
        footnote=footnote,
    )
    run_within_group_variability_variant(
        embeddings_root,
        embedding_col,
        outdir,
        collapsed=True,
        csv_name=WITHIN_GROUP_VAR_COLLAPSED_CSV,
        fig_name=WITHIN_GROUP_VAR_COLLAPSED_FILENAME,
        suptitle=WITHIN_GROUP_VAR_COLLAPSED_SUPTITLE,
        groups=GROUP_ORDER_COLLAPSED,
        group_x=WITHIN_GROUP_VAR_COLLAPSED_GROUP_X,
        group_labels=_within_group_var_collapsed_labels(),
        variant_label=f"{metric_label} · Humans collapsed, GenAI",
        analyze_task=analyze_task,
        pre_col=pre_col,
        post_col=post_col,
        p_col=p_col,
        ylabel=ylabel,
        metric_subtitle=metric_subtitle,
        footnote=footnote,
    )
    return _run_human_vs_genai_by_phase(
        embeddings_root,
        embedding_col,
        outdir,
        analyze_task=analyze_human_vs_genai_task,
        human_pre_col=human_pre_col,
        genai_pre_col=genai_pre_col,
        human_post_col=human_post_col,
        genai_post_col=genai_post_col,
        p_pre_col="pre_welch_p_one_sided",
        p_post_col="post_welch_p_one_sided",
        ylabel=human_vs_genai_ylabel,
        metric_subtitle=human_vs_genai_metric_subtitle,
        footnote=human_vs_genai_footnote,
        metric_label=metric_label,
    )


def run_within_group_variability_comparison(
    embeddings_root: Path,
    embedding_col: str,
) -> pd.DataFrame:
    base_outdir = comparisons_pre_post_dir(
        embeddings_root, COMPARISONS_WITHIN_GROUP_VAR_SUBDIR
    )
    centroid_outdir = base_outdir / WITHIN_GROUP_VAR_CENTROID_SUBDIR
    pairwise_outdir = base_outdir / WITHIN_GROUP_VAR_PAIRWISE_SUBDIR

    _run_within_group_variability_metric_family(
        embeddings_root,
        embedding_col,
        centroid_outdir,
        analyze_task=analyze_within_group_variability_task,
        pre_col="pre_mean_cosine_distance",
        post_col="post_mean_cosine_distance",
        p_col="paired_p_one_sided_post_lt_pre",
        ylabel="Mean cosine distance to group centroid",
        metric_subtitle=WITHIN_GROUP_VAR_METRIC_SUBTITLE,
        footnote=WITHIN_GROUP_VAR_FOOTNOTE,
        metric_label="centroid distance",
        analyze_human_vs_genai_task=analyze_diversity_prediction_task,
        human_pre_col="pre_human_mean_cosine_distance",
        genai_pre_col="pre_genai_mean_cosine_distance",
        human_post_col="post_human_mean_cosine_distance",
        genai_post_col="post_genai_mean_cosine_distance",
        human_vs_genai_footnote=DIVERSITY_PREDICTION_FOOTNOTE,
        human_vs_genai_metric_subtitle=DIVERSITY_PRED_METRIC_SUBTITLE,
        human_vs_genai_ylabel="Mean cosine distance to group centroid",
    )
    pairwise_df = _run_within_group_variability_metric_family(
        embeddings_root,
        embedding_col,
        pairwise_outdir,
        analyze_task=analyze_within_group_variability_task_pairwise,
        pre_col="pre_mean_pairwise_cosine_distance",
        post_col="post_mean_pairwise_cosine_distance",
        p_col="paired_p_one_sided_post_lt_pre",
        ylabel=WITHIN_GROUP_VAR_PAIRWISE_YLABEL,
        metric_subtitle=WITHIN_GROUP_VAR_PAIRWISE_METRIC_SUBTITLE,
        footnote=WITHIN_GROUP_VAR_PAIRWISE_FOOTNOTE,
        metric_label="mean pairwise cosine distance",
        analyze_human_vs_genai_task=analyze_diversity_prediction_task_pairwise,
        human_pre_col="pre_human_mean_pairwise_cosine_distance",
        genai_pre_col="pre_genai_mean_pairwise_cosine_distance",
        human_post_col="post_human_mean_pairwise_cosine_distance",
        genai_post_col="post_genai_mean_pairwise_cosine_distance",
        human_vs_genai_footnote=DIVERSITY_PREDICTION_PAIRWISE_FOOTNOTE,
        human_vs_genai_metric_subtitle=DIVERSITY_PRED_PAIRWISE_METRIC_SUBTITLE,
        human_vs_genai_ylabel=WITHIN_GROUP_VAR_PAIRWISE_YLABEL,
    )
    return pairwise_df


def run_diversity_prediction_comparison(
    embeddings_root: Path,
    embedding_col: str,
) -> pd.DataFrame:
    """Deprecated alias — diversity outputs live under within_group_variability/."""
    return run_within_group_variability_comparison(embeddings_root, embedding_col)


# ---------------------------------------------------------------------
# Per-set runners (clustering CSVs vs network HTML)
# ---------------------------------------------------------------------

def run_semantic_clustering_for_embedding_set(
    embedding_set_dir: Path,
    embeddings_root: Path,
) -> Path:
    """Write collapsed HDBSCAN clustering CSVs beside embeddings_wide.parquet."""
    input_path = embedding_set_dir / "embeddings_wide.parquet"
    set_label = embedding_set_label(embedding_set_dir)

    df = pd.read_parquet(input_path)
    columns = available_embedding_columns(df, DEFAULT_EMBEDDING_COLUMNS)
    required = [PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {embedding_set_dir}: {missing}")

    code_map = respondent_name_map_for_embeddings_root(embeddings_root)
    df = anonymize_participant_names_df(df, code_map)

    last_outdir: Path | None = None
    for col in tqdm(columns, desc=f"clustering · {set_label}"):
        space_outdir = resolve_task_data_dir(
            embeddings_root, embedding_set_dir, col
        )
        space_outdir.mkdir(parents=True, exist_ok=True)
        X = normalize(stack_embeddings(df, col))
        run_semantic_clustering_analysis(
            df=df,
            X=X,
            space_outdir=str(space_outdir),
            embedding_set_label_text=set_label,
            embedding_col=col,
        )
        last_outdir = space_outdir

    assert last_outdir is not None
    print(f"Saved clustering CSVs to: {last_outdir}")
    return last_outdir


def run_network_html_for_embedding_set(
    embedding_set_dir: Path,
    embeddings_root: Path,
) -> Path:
    """Write interactive semantic-threshold network HTML under network/."""
    input_path = embedding_set_dir / "embeddings_wide.parquet"
    set_label = embedding_set_label(embedding_set_dir)

    df = pd.read_parquet(input_path)
    columns = available_embedding_columns(df, DEFAULT_EMBEDDING_COLUMNS)
    required = [PARTICIPANT_NAME_COL, PARTICIPANT_TYPE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {embedding_set_dir}: {missing}")

    code_map = respondent_name_map_for_embeddings_root(embeddings_root)
    df = anonymize_participant_names_df(df, code_map)

    last_outdir: Path | None = None
    for col in tqdm(columns, desc=f"network html · {set_label}"):
        network_dir = resolve_network_dir(embeddings_root, embedding_set_dir, col)
        X = normalize(stack_embeddings(df, col))
        _, similarity_threshold = semantic_neighbor_degree(
            X,
            labels=df[PARTICIPANT_TYPE_COL].values,
            threshold_quantile=THRESHOLD_QUANTILE,
        )
        export_network_interactive_demos(
            df=df,
            X=X,
            similarity_threshold=similarity_threshold,
            threshold_quantile=THRESHOLD_QUANTILE,
            embedding_set_label_text=set_label,
            network_dir=network_dir,
            embedding_set_dir=embedding_set_dir,
            seed=ANALYSIS_SEED,
        )
        last_outdir = network_dir

    assert last_outdir is not None
    print(f"Saved network HTML to: {last_outdir}")
    return last_outdir


def run_visualizations_for_embedding_set(
    embedding_set_dir: Path,
    embeddings_root: Path,
) -> Path:
    """Backward-compatible alias: network HTML only (clustering is separate)."""
    return run_network_html_for_embedding_set(embedding_set_dir, embeddings_root)


# Prefer thin fig_*.py / run_*.py entrypoints under embedding_analysis/; this module is a library.
