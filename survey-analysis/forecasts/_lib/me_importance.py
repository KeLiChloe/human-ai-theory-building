"""
Main Effects Analysis
=====================
Descriptive Analysis — Feature Selection Frequency (Q1)

Count how many times each feature is selected into the top 5
by respondents. Race and Gender tasks are analysed separately.

Figures produced:
  01a / 01b  — overall frequency (all / Humans / GenAI)
Bar labels for ML top-5 features: n selected (sign-alignment % vs LR).
"""

import csv
import json
import re
import sys
from collections import Counter
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

ml_top5 = {task: {e["feature"] for e in entries}
           for task, entries in _ml_raw.items()}
ml_signs = {
    task: {e["feature"]: e["sign"] for e in entries}
    for task, entries in _ml_raw.items()
}

# ── Feature list & labels ─────────────────────────────────────────────────────
FEATURES = [
    re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
    for h in headers
    if re.match(r"^Q Race\.2 \(rank\) - ", h)
]

FEATURE_LABELS = {
    "social_science":                    "Social Science",
    "natural_science":                   "Natural Science",
    "engineering_and_technology":        "Engineering & Tech",
    "num_authors":                       "Num. Authors",
    "female":                            "Female",
    "asian":                             "Asian",
    "black":                             "Black",
    "hispanic_and_other":                "Hispanic",
    "white":                             "White",
    "authors_race_diversity_score":      "Author Race Diversity",
    "country_race_diversity_score":      "Country Race Diversity",
    "news_inequality_mentions_3_years":  "\"Inequality\" Mentions in News (3yr)",
    "paper_inequality_mentions_3_years": "\"Inequality\" Mentions in Papers (3yr)",
}

# ── Column indices ────────────────────────────────────────────────────────────
group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)
r1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
g1_col     = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")
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


# ── Count selection frequency ─────────────────────────────────────────────────
def count_selections(col_idx, group_filter=None):
    """
    group_filter: None | 'senior' | 'phd' | 'genai' | 'human'
    """
    counter = Counter({f: 0 for f in FEATURES})
    n = 0
    for row in data:
        gid = row[group_col].strip()
        if not _row_matches_group(gid, group_filter):
            continue
        cell = row[col_idx].strip()
        if not cell:
            continue
        n += 1
        for feat in cell.split(","):
            feat = feat.strip()
            if feat in counter:
                counter[feat] += 1
    return counter, n


def sign_align_among_selectors(task: str, group_filter=None):
    """Among selectors of each ML top-5 feature, count LR sign alignment."""
    q1 = r1_col if task == "race" else g1_col
    q3 = r3_cols if task == "race" else g3_cols
    signs = ml_signs[task]
    out = {f: {"n_selected": 0, "n_aligned": 0} for f in signs}
    for row in data:
        gid = row[group_col].strip()
        if not _row_matches_group(gid, group_filter):
            continue
        cell = row[q1].strip()
        if not cell:
            continue
        selected = {x.strip() for x in cell.split(",") if x.strip()}
        for feat, ml_sign in signs.items():
            if feat not in selected:
                continue
            out[feat]["n_selected"] += 1
            human = row[q3[feat]].strip() if feat in q3 else ""
            if human == ml_sign:
                out[feat]["n_aligned"] += 1
    return out


def _format_bar_label(n_sel: int, align: dict[str, int] | None) -> str:
    if align is None or align["n_selected"] <= 0:
        return f"{n_sel}"
    pct = align["n_aligned"] / align["n_selected"] * 100
    return f"{n_sel}  ({pct:.0f}% sign-aligned)"


# All / Senior Scientist / PhD Students / GenAI counts
race_counts,     n_race     = count_selections(r1_col)
gender_counts,   n_gender   = count_selections(g1_col)
race_senior,        n_race_senior = count_selections(r1_col, group_filter="senior")
race_phd,        n_race_phd = count_selections(r1_col, group_filter="phd")
race_gen,        n_race_gen = count_selections(r1_col, group_filter="genai")
race_human,      n_race_human = count_selections(r1_col, group_filter="human")
gender_senior,      n_gend_senior = count_selections(g1_col, group_filter="senior")
gender_phd,      n_gend_phd = count_selections(g1_col, group_filter="phd")
gender_gen,      n_gend_gen = count_selections(g1_col, group_filter="genai")
gender_human,    n_gend_human = count_selections(g1_col, group_filter="human")

print(f"Respondents — Race:   all={n_race}, senior={n_race_senior}, phd={n_race_phd}, genai={n_race_gen}")
print(f"Respondents — Gender: all={n_gender}, senior={n_gend_senior}, phd={n_gend_phd}, genai={n_gend_gen}")

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_DEFAULT    = COLOR_ML_FEATURE_DEFAULT
COLOR_ML         = COLOR_ML_FEATURE_HIGHLIGHT

# ── Figure 1: Overall frequency ───────────────────────────────────────────────
def plot_overall(
    counts,
    ml_top5_set,
    task_label,
    n_resp,
    out_path,
    *,
    task_key: str,
    group_filter=None,
    subgroup_label="All Respondents",
    mark_unselected_ml_ticks=False,
):
    sorted_feats = sorted(FEATURES, key=lambda f: counts[f], reverse=True)
    values  = [counts[f] for f in sorted_feats]
    colors  = [COLOR_ML if f in ml_top5_set else COLOR_DEFAULT
               for f in sorted_feats]
    align = sign_align_among_selectors(task_key, group_filter=group_filter)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sorted_feats))
    bars  = ax.barh(y_pos, values, color=colors, edgecolor="white",
                    linewidth=0.6, height=0.65)

    for bar, v, feat in zip(bars, values, sorted_feats):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            _format_bar_label(v, align.get(feat)),
            va="center",
            ha="left",
            fontsize=9,
        )

    tick_labels = []
    for f in sorted_feats:
        lbl = FEATURE_LABELS[f]
        add_ml_mark = mark_unselected_ml_ticks and (f in ml_top5_set) and (counts[f] == 0)
        tick_labels.append(f"{lbl} (ML top-5)" if add_ml_mark else lbl)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontsize=10.5)
    for t in ax.get_yticklabels():
        if "(ML top-5)" in t.get_text():
            t.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Number of respondents selecting feature\n"
        "(% = percent of selectors sign-aligned with LR)",
        fontsize=10,
    )
    ax.set_title(
        f"Feature Selection Frequency — {task_label}\n"
        f"(Top-5 features selected by respondents, {subgroup_label}, N = {n_resp})",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlim(0, n_resp + n_resp * 0.38)
    ax.axvline(n_resp, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    legend_handles = [
        mpatches.Patch(color=COLOR_ML,      label="In ML top-5"),
        mpatches.Patch(color=COLOR_DEFAULT, label="Not in ML top-5"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


def run_all() -> None:
    # ── Run all plots ─────────────────────────────────────────────────────────
    plot_overall(race_counts,   ml_top5["race"],
                 "Race",  n_race,
                 OUT_DIR / "01a_race.png",
                 task_key="race",
                 subgroup_label="All Respondents")
    plot_overall(race_human,    ml_top5["race"],
                 "Race",  n_race_human,
                 OUT_DIR / "01a_race_humans.png",
                 task_key="race",
                 group_filter="human",
                 subgroup_label="Humans")
    plot_overall(race_gen,      ml_top5["race"],
                 "Race",  n_race_gen,
                 OUT_DIR / "01a_race_genai.png",
                 task_key="race",
                 group_filter="genai",
                 subgroup_label="GenAI",
                 mark_unselected_ml_ticks=True)

    plot_overall(gender_counts, ml_top5["gender"],
                 "Gender", n_gender,
                 OUT_DIR / "01b_gender.png",
                 task_key="gender",
                 subgroup_label="All Respondents")
    plot_overall(gender_human,  ml_top5["gender"],
                 "Gender", n_gend_human,
                 OUT_DIR / "01b_gender_humans.png",
                 task_key="gender",
                 group_filter="human",
                 subgroup_label="Humans")
    plot_overall(gender_gen,    ml_top5["gender"],
                 "Gender", n_gend_gen,
                 OUT_DIR / "01b_gender_genai.png",
                 task_key="gender",
                 group_filter="genai",
                 subgroup_label="GenAI",
                 mark_unselected_ml_ticks=True)


    # ── Print summary tables ──────────────────────────────────────────────────
    for task, counts, n, senior_c, n_senior, phd_c, n_phd, gen_c, n_gen, ml_key in [
        ("RACE",   race_counts,   n_race,   race_senior, n_race_senior, race_phd, n_race_phd, race_gen, n_race_gen, "race"),
        ("GENDER", gender_counts, n_gender, gender_senior, n_gend_senior, gender_phd, n_gend_phd, gender_gen, n_gend_gen, "gender"),
    ]:
        print(f"\n── {task} — Feature Selection Frequency ───────────────────────────────────────")
        print(f"{'Rank':<5} {'Feature':<35} {'All':>5} {'All%':>6}  {'Sen':>4} {'Sen%':>6}  {'PhD':>4} {'PhD%':>6}  {'Gen':>4} {'Gen%':>6}  ML")
        print("─" * 80)
        for rank, f in enumerate(
            sorted(FEATURES, key=lambda x: counts[x], reverse=True), 1
        ):
            ml_mark = "(ML)" if f in ml_top5[ml_key] else " "
            print(f"{rank:<5} {FEATURE_LABELS[f]:<35} "
                  f"{counts[f]:>5}  {counts[f]/n*100:>4.0f}%  "
                  f"{senior_c[f]:>4}  {senior_c[f]/n_senior*100:>4.0f}%  "
                  f"{phd_c[f]:>4}  {phd_c[f]/n_phd*100:>4.0f}%  "
                  f"{gen_c[f]:>4}  {gen_c[f]/n_gen*100:>4.0f}%  {ml_mark}")
        print("─" * 80)
        print("  (ML) = also in ML top-5")


if __name__ == "__main__":
    run_all()
