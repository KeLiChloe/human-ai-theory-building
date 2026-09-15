"""
Main Effects Analysis
=====================
Quantitative Metrics (Q1 + Q2 + Q3 vs ML vector)

Two metrics per respondent per task:

1. Cosine Similarity (binary signed)
   Vector: +1 / −1 / 0  (selected with sign / not selected)
   Range: [−1, 1]

All metrics compared across Senior Scientist vs PhD Student groups.
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent  # survey-analysis/
TEXTUAL_DIR = ROOT / "textual_analysis"
FORECASTS_DIR = ROOT / "forecasts"
LIB = Path(__file__).resolve().parent
for p in (ROOT, TEXTUAL_DIR, FORECASTS_DIR, LIB):
    if str(p) not in sys.path:
        sys.path.append(str(p))
from common.stats_utils import bootstrap_ci_half_width, welch_test
from common.viz_config import COLOR_AGG_HUMAN, COLOR_ML_FEATURE_HIGHLIGHT, GROUP_COLORS
from viz_style import (
    FOOTNOTE_LINE_STEP,
    SUBPLOT_LEFT,
    SUBPLOT_RIGHT,
    comparison_box_height,
    comparison_pair_label,
    draw_centered_comparison_box,
    draw_sig_footnote,
    SIG_LEVEL_LEGEND,
)

MAIN_EFFECTS_WELCH_THREE_GROUP_FOOTNOTE = (
    "Two-sided Welch t-test on pairwise group mean ML feature-selection accuracy "
    "(PhD Students vs Senior Scientists, PhD Students vs GenAI, Senior Scientists vs GenAI).",
    SIG_LEVEL_LEGEND,
)
MAIN_EFFECTS_WELCH_HUMAN_GENAI_FOOTNOTE = (
    "Two-sided Welch t-test on mean ML feature-selection accuracy (Humans vs GenAI).",
    SIG_LEVEL_LEGEND,
)

# Bottom stack for panel-03 summary bars: footnote → Welch box → x tick labels → axes.
QUANT_FOOTNOTE_Y = 0.0
QUANT_FOOTNOTE_LINE_HEIGHT = 0.016
QUANT_BOX_FOOTNOTE_GAP = 0.020
QUANT_XTICK_GAP = 0.092
QUANT_SUMMARY_SUBPLOT_TOP = 0.90
QUANT_BOX_MAX_WIDTH_FRAC = 0.90
QUANT_SUMMARY_SAVE_PAD = 0.14


def quant_summary_bottom_layout(
    n_comp_lines: int,
    *,
    n_footnote_lines: int = 2,
) -> tuple[float, float, float]:
    """Return (footnote_y, comparison_box_bottom, subplot_bottom) from figure bottom."""
    footnote_y = QUANT_FOOTNOTE_Y
    footnote_top = (
        footnote_y
        + (n_footnote_lines - 1) * FOOTNOTE_LINE_STEP
        + QUANT_FOOTNOTE_LINE_HEIGHT
    )
    comp_box_bottom = footnote_top + QUANT_BOX_FOOTNOTE_GAP
    comp_box_top = comp_box_bottom + comparison_box_height(n_comp_lines)
    subplot_bottom = comp_box_top + QUANT_XTICK_GAP
    return footnote_y, comp_box_bottom, subplot_bottom


def save_figure(out_path, *, pad_inches: float = 0.04):
    """Export high-res PNG only."""
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=900, bbox_inches="tight", pad_inches=pad_inches)

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 900,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
FORECASTS = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "anonymous_survey_data.csv"
ML_PATH  = FORECASTS / "ML_results_main_effects.json"
OUT_DIR  = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

headers = rows[0]
data    = rows[1:]

with open(ML_PATH) as f:
    _ml_raw = json.load(f)   # {"race": [{rank, feature, sign}, ...], "gender": [...]}

# Parse into lookup structures used across the file
# ml_signs[task]  : {feature: sign} — for binary vector & exact recovery
ml_signs  = {}
for task, entries in _ml_raw.items():
    sorted_entries = sorted(entries, key=lambda e: e["rank"])
    ml_signs[task]  = {e["feature"]: e["sign"] for e in sorted_entries}

# ── Feature list ──────────────────────────────────────────────────────────────
FEATURES = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]
FEAT_IDX = {f: i for i, f in enumerate(FEATURES)}
SIGN_MAP = {"+": 1, "-": -1}

# Pooled respondent subsets for aggregate metrics (not per-bar group means).
# Human = PhD Students (0) + Senior Scientist (1) combined; GenAI (2) is computed separately.
HUMAN_GROUP_IDS = frozenset({"0", "1"})
SENIOR_GROUP_IDS = frozenset({"1"})
PHD_GROUP_IDS = frozenset({"0"})
GENAI_GROUP_IDS = frozenset({"2"})

# ── Column index maps ─────────────────────────────────────────────────────────
group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)
r1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
g1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")

r3_cols = {re.sub(r"^Q Race\.3 \(sign\) - ",   "", h): i
           for i, h in enumerate(headers) if re.match(r"^Q Race\.3 \(sign\) - ",   h)}
g3_cols = {re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
           for i, h in enumerate(headers) if re.match(r"^Q Gender\.3 \(sign\) - ", h)}
topic_expert_col = next(i for i, h in enumerate(headers) if h.strip() == "topic_expert")


def _is_topic_expert(row: list[str]) -> bool:
    """Human-only topic expertise from CSV ``topic_expert``.

    Coding: ``1`` = topic expert, ``0`` = human non-expert, ``-1`` = N/A (GenAI; ignored).
    """
    return row[topic_expert_col].strip() == "1"

# ── Vector builders ───────────────────────────────────────────────────────────
def build_binary_vector(q1_col, q3_col_map, row):
    """±1 / 0 signed vector."""
    vec = np.zeros(len(FEATURES))
    cell = row[q1_col].strip()
    if not cell:
        return None
    for feat in cell.split(","):
        feat = feat.strip()
        if feat not in FEAT_IDX:
            continue
        sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
        vec[FEAT_IDX[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec

def build_ml_binary_vector(signs_dict):
    """±1 / 0 ML vector from {feature: sign} dict."""
    vec = np.zeros(len(FEATURES))
    for feat, sign_str in signs_dict.items():
        if feat in FEAT_IDX:
            vec[FEAT_IDX[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec

# ── Metric functions ──────────────────────────────────────────────────────────
def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


RANDOM_BENCHMARK_N = 1000
RANDOM_BENCHMARK_SEED = 20260715


def random_benchmark_cosine(ml_vec, vec_dim, nonzero_count, n_random=RANDOM_BENCHMARK_N):
    """Monte Carlo estimate of E[cosine(random signed sparse vector, ML vector)].

    Uses the mean of per-draw cosines (not cosine of the averaged vector). The
    latter has near-zero norm after averaging and produces unstable angles.
    """
    rng = np.random.default_rng(RANDOM_BENCHMARK_SEED)
    ml_norm = np.linalg.norm(ml_vec)
    if ml_norm == 0:
        return np.nan
    scores = np.empty(n_random, dtype=float)
    for i in range(n_random):
        vec = np.zeros(vec_dim, dtype=float)
        idx = rng.choice(vec_dim, size=nonzero_count, replace=False)
        vec[idx] = rng.choice([-1, 1], size=nonzero_count)
        vnorm = np.linalg.norm(vec)
        scores[i] = float(np.dot(vec, ml_vec) / (vnorm * ml_norm)) if vnorm else np.nan
    return float(np.nanmean(scores))


def compute_hit_rate(q1_col, ml_signs_dict, row):
    """Fraction of ML-important features that the respondent selected (sign-agnostic)."""
    cell = row[q1_col].strip()
    if not cell:
        return np.nan
    selected = {f.strip() for f in cell.split(",") if f.strip() in FEAT_IDX}
    ml_feats = set(ml_signs_dict.keys())
    return len(selected & ml_feats) / len(ml_feats) if ml_feats else np.nan

def compute_sign_align_rate(q1_col, q3_col_map, ml_signs_dict, row):
    """Among selected ML-important features, fraction with correct sign.
    Returns nan if the respondent selected no ML features at all."""
    cell = row[q1_col].strip()
    if not cell:
        return np.nan
    selected = [f.strip() for f in cell.split(",") if f.strip() in FEAT_IDX]
    overlapping = [f for f in selected if f in ml_signs_dict]
    if not overlapping:
        return np.nan
    aligned = sum(
        1 for f in overlapping
        if (row[q3_col_map[f]].strip() if f in q3_col_map else "") == ml_signs_dict[f]
    )
    return aligned / len(overlapping)


def aggregated_sign_align_majority_excluding_ties(q1_col, q3_col_map, ml_signs_dict, pooled_subset_ids):
    """Majority-sign vote per ML feature over rows in pooled_subset_ids; ties excluded.

    Typical use: HUMAN_GROUP_IDS (Senior Scientist+PhD pooled) or GENAI_GROUP_IDS alone.
    """
    vote_sums = {feat: 0 for feat in ml_signs_dict}
    for row in data:
        if row[group_col].strip() not in pooled_subset_ids:
            continue
        cell = row[q1_col].strip()
        if not cell:
            continue
        for feat in cell.split(","):
            feat = feat.strip()
            if feat not in ml_signs_dict or feat not in FEAT_IDX:
                continue
            sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            vote_sums[feat] += SIGN_MAP.get(sign_str, 0)

    aligned = 0
    considered = 0
    tie_count = 0
    for feat, sum_vote in vote_sums.items():
        if sum_vote == 0:
            tie_count += 1
            continue
        considered += 1
        majority_sign = 1 if sum_vote > 0 else -1
        if majority_sign == SIGN_MAP.get(ml_signs_dict[feat], 0):
            aligned += 1
    rate = (aligned / considered) if considered else np.nan
    return rate, tie_count, considered, len(vote_sums)

# ── Pre-build ML vectors ───────────────────────────────────────────────────────
ml_bin_race    = build_ml_binary_vector(ml_signs["race"])
ml_bin_gender  = build_ml_binary_vector(ml_signs["gender"])

RANDOM_BENCHMARK_BY_TASK = {
    "cos_race": random_benchmark_cosine(ml_bin_race, len(FEATURES), 5),
    "cos_gender": random_benchmark_cosine(ml_bin_gender, len(FEATURES), 5),
}

# ── Compute all metrics per respondent ────────────────────────────────────────
records = []

for row in data:
    gid = row[group_col].strip()   # 0=PhD, 1=Senior, 2=GenAI

    # Binary signed vectors
    vr_bin = build_binary_vector(r1_col, r3_cols, row)
    vg_bin = build_binary_vector(g1_col, g3_cols, row)

    records.append({
        "group": gid,
        "is_topic_expert": _is_topic_expert(row),
        "vec_race_bin": vr_bin,
        "vec_gender_bin": vg_bin,
        "cos_race":         cosine_sim(vr_bin, ml_bin_race)   if vr_bin is not None else np.nan,
        "cos_gender":       cosine_sim(vg_bin, ml_bin_gender) if vg_bin is not None else np.nan,
        "hit_race":         compute_hit_rate(r1_col, ml_signs["race"],   row),
        "hit_gender":       compute_hit_rate(g1_col, ml_signs["gender"], row),
        "sign_align_race":  compute_sign_align_rate(r1_col, r3_cols, ml_signs["race"],   row),
        "sign_align_gender": compute_sign_align_rate(g1_col, g3_cols, ml_signs["gender"], row),
    })

# ── Summary stats helper ──────────────────────────────────────────────────────
def group_stats(key, group_val):
    vals = [r[key] for r in records
            if r["group"] == group_val and not np.isnan(r[key])]
    if not vals:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, sims=[])
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def all_stats(key):
    vals = [r[key] for r in records if not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def human_stats(key):
    vals = [r[key] for r in records
            if r["group"] in HUMAN_GROUP_IDS and not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def genai_stats(key):
    vals = [r[key] for r in records
            if r["group"] in GENAI_GROUP_IDS and not np.isnan(r[key])]
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)


def topic_expert_stats(key):
    vals = [r[key] for r in records
            if r.get("is_topic_expert") and not np.isnan(r[key])]
    if not vals:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, sims=[])
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)


def non_topic_expert_stats(key):
    vals = [
        r[key] for r in records
        if r["group"] in HUMAN_GROUP_IDS
        and not r.get("is_topic_expert")
        and not np.isnan(r[key])
    ]
    if not vals:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, sims=[])
    return dict(n=len(vals), mean=np.mean(vals),
                median=np.median(vals), std=np.std(vals), sims=vals)

def format_legend_value(value):
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def _collapsed_panel_ylim(means, errs, default_ylim):
    err_vals = [0 if not np.isfinite(e) else e for e in errs]
    data_lo = min(m - e for m, e in zip(means, err_vals))
    data_hi = max(m + e for m, e in zip(means, err_vals))
    span = max(data_hi - data_lo, 0.01)
    pad = max(span * 0.18, 0.04)
    ymin = data_lo - pad
    ymax = data_hi + pad
    ymax += (ymax - ymin) * 0.14
    return max(ymin, default_ylim[0]), min(ymax, default_ylim[1])


def _cosine_aggregation_scores(pts, ml_vec):
    def _agg(group_ids):
        vecs = [p["vec"] for p in pts if p["group"] in group_ids]
        return cosine_sim(np.sum(vecs, axis=0), ml_vec) if vecs else np.nan

    return {
        "senior": _agg(SENIOR_GROUP_IDS),
        "phd": _agg(PHD_GROUP_IDS),
        "human": _agg(HUMAN_GROUP_IDS),
        "genai": _agg(GENAI_GROUP_IDS),
    }


def aggregation_by_table_group(task_key: str, group_id: str) -> float:
    """Aggregated cosine for cosine-table group ids (human/0/1/topic/2)."""
    vecs = vecs_by_table_group(task_key, group_id)
    ml_vec = ml_vec_for_task(task_key)
    return cosine_sim(np.sum(vecs, axis=0), ml_vec) if len(vecs) else np.nan


def ml_vec_for_task(task_key: str):
    if task_key == "cos_race":
        return ml_bin_race
    if task_key == "cos_gender":
        return ml_bin_gender
    raise ValueError(task_key)


def vecs_by_table_group(task_key: str, group_id: str):
    """Prediction vectors for cosine-table group ids (human/0/1/topic/non_topic/2)."""
    if task_key == "cos_race":
        vec_key = "vec_race_bin"
    elif task_key == "cos_gender":
        vec_key = "vec_gender_bin"
    else:
        raise ValueError(task_key)

    if group_id == "topic":
        vecs = [
            r[vec_key]
            for r in records
            if r.get("is_topic_expert") and r[vec_key] is not None
        ]
    elif group_id == "non_topic":
        vecs = [
            r[vec_key]
            for r in records
            if r["group"] in HUMAN_GROUP_IDS
            and not r.get("is_topic_expert")
            and r[vec_key] is not None
        ]
    elif group_id == "human":
        vecs = [
            r[vec_key]
            for r in records
            if r["group"] in HUMAN_GROUP_IDS and r[vec_key] is not None
        ]
    else:
        vecs = [
            r[vec_key]
            for r in records
            if r["group"] == group_id and r[vec_key] is not None
        ]
    return np.asarray(vecs, dtype=float) if vecs else np.zeros((0, len(FEATURES)), dtype=float)


def _draw_horizontal_reference(ax, y, color, linestyle, linewidth=1.5, alpha=0.95, zorder=1):
    if not np.isnan(y):
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _legend_line(color, linestyle, label, linewidth=1.5, marker=None, markevery=None, markersize=6.5):
    kwargs = {"color": color, "linestyle": linestyle, "linewidth": linewidth, "label": label}
    if marker:
        kwargs.update(marker=marker, markevery=markevery or [1], markersize=markersize,
                      markerfacecolor=color, markeredgecolor="#2a2a2a", markeredgewidth=0.55)
        return plt.Line2D([0, 0.5, 1], [0, 0, 0], **kwargs)
    return plt.Line2D([0], [0], **kwargs)


def _build_sorted_figure_legend(
    ax,
    aggregations,
    extra_lines,
    scatter_legend=True,
    *,
    collapsed=False,
    group_means=None,
):
    """Legend entries include numeric values for aggregation reference lines."""
    handles = []
    if scatter_legend:
        if collapsed:
            human_mean = format_legend_value(
                (group_means or {}).get("human", np.nan)
            )
            genai_mean = format_legend_value(
                (group_means or {}).get("genai", np.nan)
            )
            handles.extend([
                plt.Line2D(
                    [0], [0], marker="o", linestyle="None", color=COLOR_AGG_HUMAN,
                    markersize=8, label=f"Humans (mean = {human_mean})",
                ),
                plt.Line2D(
                    [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["genai"],
                    markersize=8, label=f"GenAI (mean = {genai_mean})",
                ),
            ])
        else:
            senior_mean = format_legend_value(
                (group_means or {}).get("senior", np.nan)
            )
            phd_mean = format_legend_value(
                (group_means or {}).get("phd", np.nan)
            )
            genai_mean = format_legend_value(
                (group_means or {}).get("genai", np.nan)
            )
            handles.extend([
                plt.Line2D(
                    [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["senior"],
                    markersize=8, label=f"Senior Scientists (mean = {senior_mean})",
                ),
                plt.Line2D(
                    [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["phd"],
                    markersize=8, label=f"PhD Students (mean = {phd_mean})",
                ),
                plt.Line2D(
                    [0], [0], marker="o", linestyle="None", color=GROUP_COLORS["genai"],
                    markersize=8, label=f"GenAI (mean = {genai_mean})",
                ),
            ])
    handles.extend([
        _legend_line(
            COLOR_AGG_HUMAN, "-",
            f"Aggregated Humans = {format_legend_value(aggregations['human'])}",
        ),
        _legend_line(
            GROUP_COLORS["genai"], "-",
            f"Aggregated GenAI = {format_legend_value(aggregations['genai'])}",
        ),
    ])
    handles.extend(extra_lines)
    lg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=13.5)
    lg.get_frame().set_edgecolor("#666666")
    lg.get_frame().set_linewidth(0.8)
    return lg


def main() -> None:
    # ── Print summary tables ──────────────────────────────────────────────────────
    METRICS = [
        ("cos_race",        "cos_gender",        "Cosine Similarity (binary)"),
        ("hit_race",        "hit_gender",        "Hit Rate"),
        ("sign_align_race", "sign_align_gender", "Sign Alignment Rate"),
    ]

    for r_key, g_key, metric_label in METRICS:
        for task_key, task_label in [(r_key, "Racial Inequality"),
                                     (g_key, "Gender Inequality")]:
            e  = group_stats(task_key, "1")
            ne = group_stats(task_key, "0")
            ga = group_stats(task_key, "2")
            al = all_stats(task_key)
            hu = human_stats(task_key)
            print(f"\n── {task_label} — {metric_label} ──────────────────────────")
            print(f"{'Group':<15} {'N':>4} {'Mean':>7} {'Median':>8} {'SD':>7}")
            print("─" * 46)
            print(f"{'Senior Scientist':<15} {e['n']:>4} {e['mean']:>7.4f} {e['median']:>8.4f} {e['std']:>7.4f}")
            print(f"{'PhD Students':<15} {ne['n']:>4} {ne['mean']:>7.4f} {ne['median']:>8.4f} {ne['std']:>7.4f}")
            print(f"{'GenAI':<15} {ga['n']:>4} {ga['mean']:>7.4f} {ga['median']:>8.4f} {ga['std']:>7.4f}")
            print(f"{'All':<15} {al['n']:>4} {al['mean']:>7.4f} {al['median']:>8.4f} {al['std']:>7.4f}")
            for a_lbl, a_grp, b_lbl, b_grp in [
                ("Senior Scientist", e, "PhD Students", ne),
                ("Senior Scientist", e, "GenAI", ga),
                ("PhD Students", ne, "GenAI", ga),
                ("GenAI", ga, "All Humans", hu),
            ]:
                p = welch_test(a_grp, b_grp)
                if not np.isnan(p):
                    print(f"  Welch t-test ({a_lbl} vs {b_lbl}) = p = {p:.4f}")

    race_sign_agg, race_tie, race_considered, race_total = aggregated_sign_align_majority_excluding_ties(
        r1_col, r3_cols, ml_signs["race"], HUMAN_GROUP_IDS
    )
    gender_sign_agg, gender_tie, gender_considered, gender_total = aggregated_sign_align_majority_excluding_ties(
        g1_col, g3_cols, ml_signs["gender"], HUMAN_GROUP_IDS
    )
    print(
        f"\n[Main Effects] Aggregated sign alignment (excluding ties) — Race: "
        f"{race_sign_agg:.4f} | ties excluded = {race_tie}/{race_total} | considered = {race_considered}"
    )
    print(
        f"[Main Effects] Aggregated sign alignment (excluding ties) — Gender: "
        f"{gender_sign_agg:.4f} | ties excluded = {gender_tie}/{gender_total} | considered = {gender_considered}"
    )



if __name__ == "__main__":
    main()
