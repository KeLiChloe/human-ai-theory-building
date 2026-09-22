"""
Second-Order Interactions (SOI) Analysis
=======================================
Descriptive Analysis — Interaction Selection Frequency (Q6-Q8)

Each respondent selects top-3 second-order interactions.
Interaction is treated as an unordered pair.

Figures produced:
  01a / 01b  — overall frequency (all / Humans / GenAI)
Bar labels for ML top-3 interactions: n selected (sign-alignment % vs LR).
"""

import csv
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # survey-analysis/
FORECASTS_DIR = ROOT / "forecasts"
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS_DIR, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from common.viz_config import COLOR_ML_FEATURE_DEFAULT, COLOR_ML_FEATURE_HIGHLIGHT

COLOR_DEFAULT = COLOR_ML_FEATURE_DEFAULT
COLOR_ML = COLOR_ML_FEATURE_HIGHLIGHT

FEATURE_LABELS = {
    "social_science": "Social Science",
    "natural_science": "Natural Science",
    "engineering_and_technology": "Engineering & Tech",
    "num_authors": "Num. Authors",
    "female": "Female",
    "asian": "Asian",
    "black": "Black",
    "hispanic_and_other": "Hispanic",
    "white": "White",
    "authors_race_diversity_score": "Author Race Diversity",
    "country_race_diversity_score": "Country Race Diversity",
    "news_inequality_mentions_3_years": '"Inequality" Mentions in News (3yr)',
    "paper_inequality_mentions_3_years": '"Inequality" Mentions in Papers (3yr)',
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
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

FORECASTS = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "anonymous_survey_data.csv"
ML_PATH = FORECASTS / "ML_results_second_order_interactions.json"
OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def canon_pair(a, b):
    return tuple(sorted((a.strip(), b.strip())))


def parse_pair(cell, valid_features):
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2:
        return None
    if parts[0] not in valid_features or parts[1] not in valid_features:
        return None
    if parts[0] == parts[1]:
        return None
    return canon_pair(parts[0], parts[1])


with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
headers = rows[0]
data = rows[1:]

with open(ML_PATH) as f:
    ml_raw = json.load(f)

features = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]
feature_set = set(features)

pairs = list(combinations(sorted(features), 2))  # 78 unordered pairs
pair_label = {
    p: f"{FEATURE_LABELS.get(p[0], p[0])} * {FEATURE_LABELS.get(p[1], p[1])}"
    for p in pairs
}

ml_pairs = {
    task: {canon_pair(e["feature_1"], e["feature_2"]) for e in entries}
    for task, entries in ml_raw.items()
}
ml_signs = {
    task: {
        canon_pair(e["feature_1"], e["feature_2"]): e["sign"]
        for e in entries
    }
    for task, entries in ml_raw.items()
}

group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)
r_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
]
g_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
]
r_sign_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 3rd)"),
]
g_sign_cols = [
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 1st)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 2nd)"),
    next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 3rd)"),
]


def _row_matches_group(gid: str, group_filter: str | None) -> bool:
    if group_filter is None:
        return True
    if group_filter == "senior":
        return gid == "1"
    if group_filter == "phd":
        return gid == "0"
    if group_filter == "genai":
        return gid == "2"
    if group_filter == "human":
        return gid in {"0", "1"}
    raise ValueError(group_filter)


def count_soi(cols, group_filter=None):
    c = Counter({p: 0 for p in pairs})
    n = 0
    for row in data:
        gid = row[group_col].strip()
        if not _row_matches_group(gid, group_filter):
            continue
        n += 1
        for col in cols:
            p = parse_pair(row[col], feature_set)
            if p is not None:
                c[p] += 1
    return c, n


def sign_align_among_selectors(task: str, group_filter=None):
    """Among selectors of each ML top-3 interaction, count LR sign alignment."""
    pair_cols = r_cols if task == "race" else g_cols
    sign_cols = r_sign_cols if task == "race" else g_sign_cols
    signs = ml_signs[task]
    out = {p: {"n_selected": 0, "n_aligned": 0} for p in signs}
    for row in data:
        gid = row[group_col].strip()
        if not _row_matches_group(gid, group_filter):
            continue
        chosen: dict[tuple[str, str], str] = {}
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc], feature_set)
            if p is None:
                continue
            chosen[p] = row[sc].strip()
        for p, ml_sign in signs.items():
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
    return f"{n_sel}  ({pct:.0f}% sign-aligned)"


def plot_overall(
    counts,
    n,
    ml_set,
    title,
    out_path,
    *,
    task_key: str,
    group_filter=None,
    top_n=10,
    subgroup_label="All Respondents",
    mark_unselected_ml_ticks=False,
):
    ranked = sorted(pairs, key=lambda p: counts[p], reverse=True)[:top_n]
    if mark_unselected_ml_ticks:
        missing_ml = [p for p in ml_set if p not in ranked]
        if missing_ml:
            non_ml_ranked = [p for p in ranked if p not in ml_set]
            keep_non_ml = max(0, top_n - len(ml_set))
            ranked = non_ml_ranked[:keep_non_ml] + [p for p in ranked if p in ml_set] + missing_ml

    labels = []
    for p in ranked:
        lbl = pair_label[p]
        add_ml_mark = mark_unselected_ml_ticks and (p in ml_set) and (counts[p] == 0)
        labels.append(f"{lbl} (ML top-3)" if add_ml_mark else lbl)
    vals = [counts[p] for p in ranked]
    colors = [COLOR_ML if p in ml_set else COLOR_DEFAULT for p in ranked]
    align = sign_align_among_selectors(task_key, group_filter=group_filter)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(ranked))
    bars = ax.barh(
        y, vals, color=colors, height=0.65, edgecolor="white", linewidth=0.6,
    )
    for b, v, pair in zip(bars, vals, ranked):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2,
            _format_bar_label(v, align.get(pair)),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    for t in ax.get_yticklabels():
        if "(ML top-3)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of respondents selecting interaction\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=10,
    )
    ax.set_title(
        f"{title}\n(Top-3 interactions selected by respondents, {subgroup_label}, N = {n})",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0, n + n * 0.38)
    ax.axvline(n, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_ML, label="In ML top-3"),
            mpatches.Patch(color=COLOR_DEFAULT, label="Not in ML top-3"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


race_counts, n_race = count_soi(r_cols)
gender_counts, n_gender = count_soi(g_cols)
race_senior, n_race_senior = count_soi(r_cols, group_filter="senior")
race_phd, n_race_phd = count_soi(r_cols, group_filter="phd")
race_gen, n_race_gen = count_soi(r_cols, group_filter="genai")
race_human, n_race_human = count_soi(r_cols, group_filter="human")
gend_senior, n_gend_senior = count_soi(g_cols, group_filter="senior")
gend_phd, n_gend_phd = count_soi(g_cols, group_filter="phd")
gend_gen, n_gend_gen = count_soi(g_cols, group_filter="genai")
gend_human, n_gend_human = count_soi(g_cols, group_filter="human")

print(f"Respondents — Race: all={n_race}, senior={n_race_senior}, phd={n_race_phd}, genai={n_race_gen}")
print(f"Respondents — Gender: all={n_gender}, senior={n_gend_senior}, phd={n_gend_phd}, genai={n_gend_gen}")


def run_all() -> None:
    plot_overall(
        race_counts, n_race, ml_pairs["race"], "SOI Selection Frequency — Race",
        OUT_DIR / "01a_soi_race.png",
        task_key="race",
        top_n=8,
        subgroup_label="All Respondents",
    )
    plot_overall(
        race_human, n_race_human, ml_pairs["race"], "SOI Selection Frequency — Race",
        OUT_DIR / "01a_soi_race_humans.png",
        task_key="race",
        group_filter="human",
        top_n=8,
        subgroup_label="Humans",
    )
    plot_overall(
        race_gen, n_race_gen, ml_pairs["race"], "SOI Selection Frequency — Race",
        OUT_DIR / "01a_soi_race_genai.png",
        task_key="race",
        group_filter="genai",
        top_n=8,
        subgroup_label="GenAI",
        mark_unselected_ml_ticks=True,
    )
    plot_overall(
        gender_counts, n_gender, ml_pairs["gender"], "SOI Selection Frequency — Gender",
        OUT_DIR / "01b_soi_gender.png",
        task_key="gender",
        top_n=8,
        subgroup_label="All Respondents",
    )
    plot_overall(
        gend_human, n_gend_human, ml_pairs["gender"], "SOI Selection Frequency — Gender",
        OUT_DIR / "01b_soi_gender_humans.png",
        task_key="gender",
        group_filter="human",
        top_n=8,
        subgroup_label="Humans",
    )
    plot_overall(
        gend_gen, n_gend_gen, ml_pairs["gender"], "SOI Selection Frequency — Gender",
        OUT_DIR / "01b_soi_gender_genai.png",
        task_key="gender",
        group_filter="genai",
        top_n=8,
        subgroup_label="GenAI",
        mark_unselected_ml_ticks=True,
    )

    for task_name, counts, n, ml_set in [
        ("RACE", race_counts, n_race, ml_pairs["race"]),
        ("GENDER", gender_counts, n_gender, ml_pairs["gender"]),
    ]:
        print(f"\n── {task_name} SOI Top-15 ─────────────────────────────────────────")
        ranked = sorted(pairs, key=lambda p: counts[p], reverse=True)[:15]
        for i, p in enumerate(ranked, start=1):
            mark = "(ML)" if p in ml_set else " "
            print(f"{i:>2}. {pair_label[p]:<75} {counts[p]:>3} ({counts[p] / n * 100:>5.1f}%) {mark}")


if __name__ == "__main__":
    run_all()
