"""
Aggregation analysis: gain and aggregated accuracy (2×2 tables).

Uses the same record-building and filtering logic as:
  - forecasts/_lib/me_quant.py
  - forecasts/_lib/soi_quant.py

Δ Humans = Aggregated Humans cosine − mean individual Humans cosine (same pts pool as 06_*)
Δ GenAI = Aggregated GenAI cosine − mean individual GenAI cosine
Gap     = Δ Humans − Δ GenAI

Aggregated accuracy = cosine similarity of the summed prediction vector to the ML benchmark.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GAIN_TABLE_STEM = "aggregation_gain_gap_table"
ACCURACY_TABLE_STEM = "aggregation_accuracy_table"
COMBINED_TABLE_STEM = "aggregation_table"

OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SIGN_MAP = {"+": 1, "-": -1}
HUMAN_GROUP_IDS = frozenset({"0", "1"})
GENAI_GROUP_IDS = frozenset({"2"})


def find_col(headers: list[str], prefix: str) -> int:
    exact = [i for i, h in enumerate(headers) if h.strip() == prefix]
    if len(exact) == 1:
        return exact[0]
    matches = [i for i, h in enumerate(headers) if h.strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column for {prefix}, got {matches}")
    return matches[0]


def canon_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip(), b.strip())))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def aggregation_scores(pts: list[dict], ml_vec: np.ndarray, group_ids: frozenset[str]) -> float:
    vecs = [p["vec"] for p in pts if p["group"] in group_ids]
    return cosine_sim(np.sum(vecs, axis=0), ml_vec) if vecs else np.nan


def mean_score(pts: list[dict], group_ids: frozenset[str]) -> float:
    scores = [p["score"] for p in pts if p["group"] in group_ids]
    return float(np.mean(scores)) if scores else np.nan


def compute_from_plot_pts(pts: list[dict], ml_vec: np.ndarray) -> dict[str, float]:
    avg_human = mean_score(pts, HUMAN_GROUP_IDS)
    avg_genai = mean_score(pts, GENAI_GROUP_IDS)
    agg_human = aggregation_scores(pts, ml_vec, HUMAN_GROUP_IDS)
    agg_genai = aggregation_scores(pts, ml_vec, GENAI_GROUP_IDS)
    gain_human = agg_human - avg_human
    gain_genai = agg_genai - avg_genai
    return {
        "n_human": sum(1 for p in pts if p["group"] in HUMAN_GROUP_IDS),
        "n_genai": sum(1 for p in pts if p["group"] in GENAI_GROUP_IDS),
        "avg_human": avg_human,
        "avg_genai": avg_genai,
        "agg_human": agg_human,
        "agg_genai": agg_genai,
        "gain_human": gain_human,
        "gain_genai": gain_genai,
        "gain_gap": gain_human - gain_genai,
    }


def load_main_effects_records(
    headers: list[str], data: list[list[str]]
) -> tuple[list[dict], dict[str, np.ndarray]]:
    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feat_idx = {f: i for i, f in enumerate(features)}

    group_col = find_col(headers, "student_0, senior_1, genAI_2")
    topic_expert_col = next(i for i, h in enumerate(headers) if h.strip() == "topic_expert")
    r1_col = find_col(headers, "Q Race.1")
    g1_col = find_col(headers, "Q Gender.1")
    r3_cols = {
        re.sub(r"^Q Race\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers) if re.match(r"^Q Race\.3 \(sign\) - ", h)
    }
    g3_cols = {
        re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers) if re.match(r"^Q Gender\.3 \(sign\) - ", h)
    }

    ml_signs = json.loads((FORECASTS / "ML_results_main_effects.json").read_text(encoding="utf-8"))

    def build_binary_vector(q1_col: int, q3_col_map: dict[str, int], row: list[str]) -> np.ndarray | None:
        vec = np.zeros(len(features))
        cell = row[q1_col].strip()
        if not cell:
            return None
        for feat in cell.split(","):
            feat = feat.strip()
            if feat not in feat_idx:
                continue
            sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
            vec[feat_idx[feat]] = SIGN_MAP.get(sign_str, 0)
        return vec

    def build_ml_binary_vector(signs_dict: dict[str, str]) -> np.ndarray:
        vec = np.zeros(len(features))
        for feat, sign_str in signs_dict.items():
            if feat in feat_idx:
                vec[feat_idx[feat]] = SIGN_MAP.get(sign_str, 0)
        return vec

    ml_race = build_ml_binary_vector({e["feature"]: e["sign"] for e in ml_signs["race"]})
    ml_gender = build_ml_binary_vector({e["feature"]: e["sign"] for e in ml_signs["gender"]})

    records = []
    for row in data:
        gid = row[group_col].strip()
        vr = build_binary_vector(r1_col, r3_cols, row)
        vg = build_binary_vector(g1_col, g3_cols, row)
        records.append({
            "group": gid,
            "is_topic_expert": row[topic_expert_col].strip() == "1",
            "vec_race_bin": vr,
            "vec_gender_bin": vg,
            "cos_race": cosine_sim(vr, ml_race) if vr is not None else np.nan,
            "cos_gender": cosine_sim(vg, ml_gender) if vg is not None else np.nan,
        })

    return records, {"Race": ml_race, "Gender": ml_gender}


def load_soi_records(
    headers: list[str], data: list[list[str]]
) -> tuple[list[dict], dict[str, np.ndarray]]:
    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feature_set = set(features)
    pairs = list(combinations(sorted(features), 2))
    pair_idx = {p: i for i, p in enumerate(pairs)}

    group_col = next(i for i, h in enumerate(headers) if "senior_1" in h)
    topic_expert_col = next(i for i, h in enumerate(headers) if h.strip() == "topic_expert")
    r_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
    ]
    r_sign_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 3rd)"),
    ]
    g_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
    ]
    g_sign_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 3rd)"),
    ]

    ml_raw = json.loads((FORECASTS / "ML_results_second_order_interactions.json").read_text(encoding="utf-8"))

    def parse_pair(cell: str) -> tuple[str, str] | None:
        cell = cell.strip()
        if not cell or "," not in cell:
            return None
        parts = [x.strip() for x in cell.split(",")]
        if len(parts) != 2 or parts[0] == parts[1]:
            return None
        if parts[0] not in feature_set or parts[1] not in feature_set:
            return None
        return canon_pair(parts[0], parts[1])

    def build_binary_vector(pair_cols: list[int], sign_cols: list[int], row: list[str]) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc])
            if p is not None:
                vec[pair_idx[p]] = SIGN_MAP.get(row[sc].strip(), 0)
        return vec

    def build_ml_binary(entries: list[dict]) -> np.ndarray:
        vec = np.zeros(len(pairs))
        for e in entries:
            p = canon_pair(e["feature_1"], e["feature_2"])
            if p in pair_idx:
                vec[pair_idx[p]] = SIGN_MAP.get(e["sign"], 0)
        return vec

    ml_race = build_ml_binary(ml_raw["race"])
    ml_gender = build_ml_binary(ml_raw["gender"])

    records = []
    for row in data:
        gid = row[group_col].strip()
        hr = build_binary_vector(r_pair_cols, r_sign_cols, row)
        hg = build_binary_vector(g_pair_cols, g_sign_cols, row)
        records.append({
            "group": gid,
            "is_topic_expert": row[topic_expert_col].strip() == "1",
            "vec_race_bin": hr,
            "vec_gender_bin": hg,
            "cos_race": cosine_sim(hr, ml_race),
            "cos_gender": cosine_sim(hg, ml_gender),
        })

    return records, {"Race": ml_race, "Gender": ml_gender}


def plot_pts_main_effects(records: list[dict], task_key: str, vec_key: str) -> list[dict]:
    return [
        {
            "score": r[task_key],
            "group": r["group"],
            "is_topic_expert": bool(r.get("is_topic_expert")),
            "vec": r[vec_key],
        }
        for r in records
        if not np.isnan(r[task_key]) and r[vec_key] is not None
    ]


def plot_pts_soi(records: list[dict], task_key: str, vec_key: str) -> list[dict]:
    return [
        {
            "score": r[task_key],
            "group": r["group"],
            "is_topic_expert": bool(r.get("is_topic_expert")),
            "vec": r[vec_key],
        }
        for r in records
        if not np.isnan(r[task_key])
    ]


def format_gain_cell(g: dict[str, float]) -> str:
    return (
        f"{g['gain_gap']:+.3f} "
        f"(Humans Δ={g['gain_human']:+.3f}, GenAI Δ={g['gain_genai']:+.3f})"
    )


def format_accuracy_cell(g: dict[str, float]) -> str:
    return (
        f"Humans={g['agg_human']:.3f}, GenAI={g['agg_genai']:.3f}"
    )


def _latex_signed(value: float, *, decimals: int = 3) -> str:
    text = f"{value:+.{decimals}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def _latex_unsigned(value: float, *, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def _latex_bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _latex_pair_cells(
    human: float,
    genai: float,
    *,
    signed: bool,
) -> tuple[str, str]:
    fmt = _latex_signed if signed else _latex_unsigned
    if np.isclose(human, genai):
        human_wins = genai_wins = False
    elif human > genai:
        human_wins, genai_wins = True, False
    else:
        human_wins, genai_wins = False, True
    human_tex = fmt(human)
    genai_tex = fmt(genai)
    return (
        _latex_bold(human_tex) if human_wins else human_tex,
        _latex_bold(genai_tex) if genai_wins else genai_tex,
    )


def _latex_gain_pair_cells(g: dict[str, float]) -> tuple[str, str]:
    return _latex_pair_cells(
        float(g["gain_human"]),
        float(g["gain_genai"]),
        signed=True,
    )


def _latex_accuracy_pair_cells(g: dict[str, float]) -> tuple[str, str]:
    return _latex_pair_cells(
        float(g["agg_human"]),
        float(g["agg_genai"]),
        signed=False,
    )


TASK_LABELS = {
    "Race": "Racial Inequality",
    "Gender": "Gender Inequality",
}


ANALYSIS_SECTIONS = (
    ("Main Effects", "Main Effects"),
    ("Interactions", "Interactions"),
)


def build_aggregation_combined_tex(
    table: dict[str, dict[str, dict[str, float]]],
    tasks: list[str],
    *,
    include_caption: bool = True,
    gain_only: bool = False,
    show_title: bool = True,
    show_notes: bool = True,
) -> str:
    if gain_only:
        n_cols = 3
        col_spec = "@{}lcc@{}"
        body_rows: list[str] = []
        for section_idx, (analysis_key, section_label) in enumerate(ANALYSIS_SECTIONS):
            if section_idx > 0:
                body_rows.append("\\midrule")
            body_rows.append(
                f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textbf{{{section_label}}}}} \\\\"
            )
            for task in tasks:
                gain_h, gain_g = _latex_gain_pair_cells(table[task][analysis_key])
                body_rows.append(f"{TASK_LABELS[task]} & {gain_h} & {gain_g} \\\\")
        tabular = [
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            "Task & Humans $\\Delta$ & GenAI $\\Delta$ \\\\",
            "\\midrule",
            *body_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
        title = "Aggregation Gain"
        label = "tab:aggregation_gain"
        notes = [
            "{\\footnotesize ",
            "Aggregation gain ($\\Delta$) = aggregated accuracy $-$ mean individual accuracy.}",
        ]
    else:
        n_cols = 5
        body_rows = []
        for section_idx, (analysis_key, section_label) in enumerate(ANALYSIS_SECTIONS):
            if section_idx > 0:
                body_rows.append("\\midrule")
            body_rows.append(
                f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textbf{{{section_label}}}}} \\\\"
            )
            for task in tasks:
                acc_h, acc_g = _latex_accuracy_pair_cells(table[task][analysis_key])
                gain_h, gain_g = _latex_gain_pair_cells(table[task][analysis_key])
                body_rows.append(
                    f"{TASK_LABELS[task]} & {acc_h} & {acc_g} & {gain_h} & {gain_g} \\\\"
                )
        tabular = [
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            " & \\multicolumn{2}{c}{Aggregated accuracy}",
            " & \\multicolumn{2}{c}{Aggregation gain ($\\Delta$)} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "Task & Humans & GenAI & Humans & GenAI \\\\",
            "\\midrule",
            *body_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
        title = "Aggregation Accuracy and Gain"
        label = "tab:aggregation_accuracy_gain"
        notes = [
            "{\\footnotesize ",
            "Aggregated accuracy = accuracy of the summed prediction vector ",
            "relative to the ML benchmark.\\\\",
            "Aggregation gain ($\\Delta$) = aggregated accuracy $-$ mean individual accuracy.}",
        ]

    if include_caption:
        lines = [
            "% Auto-generated by aggregation.py",
            "",
            "{\\footnotesize",
            "\\setlength{\\tabcolsep}{7pt}",
            "\\renewcommand{\\arraystretch}{1.35}",
            "\\begin{center}",
            f"\\captionof{{table}}{{{title}}}",
            f"\\label{{{label}}}",
            "\\par\\vspace{0.3em}",
            *notes,
            "\\par\\vspace{0.55em}",
            *tabular,
            "\\end{center}",
            "}",
            "",
        ]
    else:
        title_line = (
            f"{{\\normalsize\\textbf{{{title}}}}}\\\\[0.45em]"
            if show_title
            else ""
        )
        note_block = [*notes, "\\\\[0.65em]"] if show_notes else []
        lines = [
            "% Auto-generated by aggregation.py (embed / standalone crop)",
            "",
            "\\begin{minipage}{0.95\\textwidth}",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{7pt}",
            "\\renewcommand{\\arraystretch}{1.35}",
            *([title_line] if title_line else []),
            *note_block,
            *tabular,
            "\\end{minipage}",
            "",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    csv_path = ROOT / "anonymous_survey_data.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    headers, data = rows[0], rows[1:]

    me_records, me_ml = load_main_effects_records(headers, data)
    soi_records, soi_ml = load_soi_records(headers, data)

    tasks = ["Race", "Gender"]
    table: dict[str, dict[str, dict[str, float]]] = {}
    detail_rows: list[dict[str, object]] = []

    for task in tasks:
        table[task] = {}
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"

        me_pts = plot_pts_main_effects(me_records, task_key, vec_key)
        g_me = compute_from_plot_pts(me_pts, me_ml[task])
        table[task]["Main Effects"] = g_me
        detail_rows.append({"task": task, "analysis": "Main Effects", **g_me})

        soi_pts = plot_pts_soi(soi_records, task_key, vec_key)
        g_soi = compute_from_plot_pts(soi_pts, soi_ml[task])
        table[task]["Interactions"] = g_soi
        detail_rows.append({"task": task, "analysis": "Interactions", **g_soi})

    long_csv = OUT_DIR / "aggregation_detail.csv"
    with long_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task", "analysis", "n_human", "n_genai",
            "avg_human", "avg_genai", "agg_human", "agg_genai",
            "gain_human", "gain_genai", "gain_gap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({
                k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                for k in fieldnames
            })

    for stale in (
        OUT_DIR / "aggregation_gain_gap_table.md",
        OUT_DIR / "aggregation_gain_detail.csv",
        OUT_DIR / f"{GAIN_TABLE_STEM}.tex",
        OUT_DIR / f"{GAIN_TABLE_STEM}_standalone.tex",
        OUT_DIR / f"{GAIN_TABLE_STEM}_standalone.pdf",
        OUT_DIR / f"{ACCURACY_TABLE_STEM}.tex",
        OUT_DIR / f"{ACCURACY_TABLE_STEM}_standalone.tex",
        OUT_DIR / f"{ACCURACY_TABLE_STEM}_standalone.pdf",
        OUT_DIR / "aggregation_accuracy_table.csv",
        OUT_DIR / "aggregation_gain_gap_table.csv",
        OUT_DIR / f"{COMBINED_TABLE_STEM}.tex",
        OUT_DIR / f"{COMBINED_TABLE_STEM}_standalone.tex",
        OUT_DIR / f"{COMBINED_TABLE_STEM}_standalone.pdf",
        OUT_DIR / f"{COMBINED_TABLE_STEM}.pdf",
    ):
        if stale.is_file():
            stale.unlink()

    print("Aggregation accuracy and gain")
    print("(aligned with 06_* figure legend + scatter means)\n")
    col_w = 44
    header = f"{'':<10} {'Main Effects':>{col_w}} {'Interactions':>{col_w}}"
    print(header)
    print("-" * len(header))
    for task in tasks:
        print(
            f"{task:<10} {format_accuracy_cell(table[task]['Main Effects']):>{col_w}} "
            f"{format_accuracy_cell(table[task]['Interactions']):>{col_w}}"
        )
        print(
            f"{'':<10} {format_gain_cell(table[task]['Main Effects']):>{col_w}} "
            f"{format_gain_cell(table[task]['Interactions']):>{col_w}}"
        )

    print("\n06 figure cross-check (Aggregated / mean individual):")
    for task in tasks:
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"
        for label, pts, ml_map in [
            ("Main Effects", plot_pts_main_effects(me_records, task_key, vec_key), me_ml),
            ("Interactions", plot_pts_soi(soi_records, task_key, vec_key), soi_ml),
        ]:
            g = compute_from_plot_pts(pts, ml_map[task])
            print(
                f"  {label} {task}: "
                f"mean H={g['avg_human']:.3f} agg H={g['agg_human']:.3f} | "
                f"mean AI={g['avg_genai']:.3f} agg AI={g['agg_genai']:.3f}"
            )

    print(f"\nSaved: {long_csv}")


if __name__ == "__main__":
    main()
