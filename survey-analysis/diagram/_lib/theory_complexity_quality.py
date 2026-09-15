"""
Theory complexity → LLM-assessed quality ratings (panel c).

Same sample logic as panel b, plus Post-ML as separate observations:

  Quality = β0 + β1 C_z + β2 GenAI + β3 (C_z × GenAI) + δ + ε

Diagram metrics (Q5 Pre / Q13 Post):
  Y = main-effects LLM quality only (Q4 / Q12)
  FE = task × phase (reference = Racial × Pre-ML)

Theory word count (mean ME+SOI text length; Pre raw / Post LLM_refined):
  Y = main-effects and SOI LLM quality stacked (Q4+Q10 / Q12+Q15)
  FE = task × effect × phase (reference = Racial × ME × Pre-ML)

C is phase-matched and standardized over person×task×phase rows.
Person-clustered (CR1) SEs.

Outputs under diagram/outputs/:
  theory_complexity_quality_summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

LIB = Path(__file__).resolve().parent
DIAGRAM = LIB.parent
ROOT = DIAGRAM.parent
FORECASTS = ROOT / "forecasts"
for p in (ROOT, DIAGRAM, LIB, FORECASTS / "_lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation import GENAI_GROUP_IDS, HUMAN_GROUP_IDS  # noqa: E402
from theory_efficiency import (  # noqa: E402
    _ci95,
    _cluster_cr1_cov,
    _find_col,
    _fmt_coef_ci_cell,
    _to_float,
    _z,
)
from theory_word_count import (  # noqa: E402
    combined_theory_word_count,
    theory_text_columns,
    word_count,
)

OUT_DIR = DIAGRAM / "outputs"
CSV_PATH = ROOT / "anonymous_survey_data.csv"
GENAI_CSV = (
    ROOT
    / "textual_analysis"
    / "theory_quality_rating"
    / "theory_ratings_genai_evaluator.csv"
)
MODEL_TAG = "gpt-5.5"

COMPLEXITY_METRICS = (
    ("n_paths", "Number of paths"),
    ("max_path_len", "Maximum path length"),
    ("n_latents", "Number of latent variables"),
    ("n_words", "Theory word count"),
)
DIAGRAM_KEYS = ("n_paths", "max_path_len", "n_latents")

TASKS = (
    ("Race", "Racial inequality"),
    ("Gender", "Gender inequality"),
)

# phase_key, phase_label, diagram Q#, ME quality Q#, SOI quality Q#
PHASES = (
    ("Pre", "Pre-ML", "5", "4", "10"),
    ("Post", "Post-ML", "13", "12", "15"),
)


def _quality_lookup(rater: str = MODEL_TAG) -> dict[tuple[str, str, str, str], float]:
    """(participant_ID, task, effect, phase) -> overall_quality from GenAI ratings CSV."""
    with GENAI_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[str, str, str, str], float] = {}
    for r in rows:
        if str(r.get("rater_identifier", "")).strip() != rater:
            continue
        v = _to_float(r.get("overall_quality", ""))
        if v is None:
            continue
        key = (
            str(r.get("participant_ID", "")).strip(),
            str(r.get("task", "")).strip().lower(),
            str(r.get("effect", "")).strip().lower(),
            str(r.get("phase", "")).strip().lower(),
        )
        out[key] = float(v)
    return out


def _complexity_col(task: str, qnum: str, ckey: str) -> str:
    label = dict(COMPLEXITY_METRICS)[ckey]
    return f"Q {task}.{qnum} {label}"


def build_person_phase_rows() -> list[dict]:
    """One row per person × task × phase with phase-matched C and qualities."""
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]
    group_col = _find_col(headers, "student_0, senior_1, genAI_2")
    topic_col = _find_col(headers, "topic_expert")
    name_col = _find_col(headers, "Participant ID")
    q_lookup = _quality_lookup()

    out: list[dict] = []
    for i, row in enumerate(data):
        gid = row[group_col].strip()
        if gid in HUMAN_GROUP_IDS:
            group = "Human"
        elif gid in GENAI_GROUP_IDS:
            group = "GenAI"
        else:
            continue
        is_topic = (
            gid in HUMAN_GROUP_IDS
            and len(row) > topic_col
            and row[topic_col].strip() == "1"
        )
        pid = row[name_col].strip() if len(row) > name_col else ""
        for task_key, task_label in TASKS:
            for phase_key, phase_label, c_q, q_me, q_soi in PHASES:
                vals: dict[str, float] = {}
                ok = True
                for ckey in DIAGRAM_KEYS:
                    col = _find_col(headers, _complexity_col(task_key, c_q, ckey))
                    v = _to_float(row[col]) if len(row) > col else None
                    if v is None:
                        ok = False
                        break
                    vals[ckey] = v
                if not ok:
                    continue
                me_name, soi_name = theory_text_columns(task_key, phase=phase_key)
                me_col = _find_col(headers, me_name)
                soi_col = _find_col(headers, soi_name)
                me_text = row[me_col] if len(row) > me_col else ""
                soi_text = row[soi_col] if len(row) > soi_col else ""
                wc_me = float(word_count(me_text))
                wc_soi = float(word_count(soi_text))
                wc = float(combined_theory_word_count(me_text, soi_text))
                if wc <= 0:
                    continue
                vals["n_words"] = wc
                vals["n_words_me"] = wc_me
                vals["n_words_soi"] = wc_soi

                phase_l = "pre" if phase_key == "Pre" else "post"
                task_l = task_key.lower()
                q_me_v = q_lookup.get((pid, task_l, "main", phase_l))
                q_soi_v = q_lookup.get((pid, task_l, "interactions", phase_l))
                if q_me_v is None or not np.isfinite(q_me_v):
                    continue
                soi_q = (
                    float(q_soi_v)
                    if q_soi_v is not None and np.isfinite(q_soi_v)
                    else float("nan")
                )
                out.append({
                    "person_id": i,
                    "group": group,
                    "group_id": gid,
                    "is_topic_expert": is_topic,
                    "task": task_key,
                    "task_label": task_label,
                    "phase": phase_key,
                    "phase_label": phase_label,
                    "quality_me": float(q_me_v),
                    "quality_soi": soi_q,
                    # Backward-compatible alias for ratios (ME quality).
                    "quality": float(q_me_v),
                    **vals,
                })
    return out


def complexity_z_stats(rows: list[dict]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for ckey, _ in COMPLEXITY_METRICS:
        xs = np.asarray([float(r[ckey]) for r in rows], dtype=float)
        mu = float(xs.mean())
        sd = float(xs.std(ddof=0))
        if sd <= 0:
            sd = 1.0
        out[ckey] = (mu, sd)
    return out


def _task_phase_fe(task: str, phase: str) -> list[float]:
    """task × phase FE; reference = Racial × Pre-ML."""
    return [
        1.0 if task == "Race" and phase == "Post" else 0.0,
        1.0 if task == "Gender" and phase == "Pre" else 0.0,
        1.0 if task == "Gender" and phase == "Post" else 0.0,
    ]


def _task_effect_phase_fe(task: str, effect: str, phase: str) -> list[float]:
    """
    task × effect × phase FE; reference = Racial × ME × Pre-ML.
    effect is ``ME`` or ``SOI``.
    """
    dummies: list[float] = []
    for t in ("Race", "Gender"):
        for e in ("ME", "SOI"):
            for p in ("Pre", "Post"):
                if (t, e, p) == ("Race", "ME", "Pre"):
                    continue
                dummies.append(
                    1.0 if task == t and effect == e and phase == p else 0.0
                )
    return dummies


def fit_pooled_models(
    rows: list[dict],
    z_stats: dict[str, tuple[float, float]],
) -> list[dict]:
    """
    Diagram: ME quality only + δ_{task×phase}.
    Word count: ME+SOI quality stacked + δ_{task×effect×phase}.
    """
    results: list[dict] = []
    for ckey, clabel in COMPLEXITY_METRICS:
        mu, sd = z_stats[ckey]
        y_list: list[float] = []
        C_list: list[float] = []
        genai_list: list[float] = []
        cell_rows: list[list[float]] = []
        clusters: list[int] = []

        for r in rows:
            task = str(r["task"])
            phase = str(r["phase"])
            cz = _z(float(r[ckey]), mu, sd)
            genai = 1.0 if r["group"] == "GenAI" else 0.0
            pid = int(r["person_id"])

            if ckey == "n_words":
                for effect, qkey in (("ME", "quality_me"), ("SOI", "quality_soi")):
                    yv = r[qkey]
                    if not np.isfinite(yv):
                        continue
                    y_list.append(float(yv))
                    C_list.append(cz)
                    genai_list.append(genai)
                    cell_rows.append(
                        _task_effect_phase_fe(task, effect, phase)
                    )
                    clusters.append(pid)
            else:
                yv = r["quality_me"]
                if not np.isfinite(yv):
                    continue
                y_list.append(float(yv))
                C_list.append(cz)
                genai_list.append(genai)
                cell_rows.append(_task_phase_fe(task, phase))
                clusters.append(pid)

        y = np.asarray(y_list, dtype=float)
        C = np.asarray(C_list, dtype=float)
        genai = np.asarray(genai_list, dtype=float)
        inter = C * genai
        cells = np.asarray(cell_rows, dtype=float)
        X = np.column_stack([np.ones(len(y)), C, genai, inter, cells])
        cl = np.asarray(clusters, dtype=int)
        beta, se, p, cov, g = _cluster_cr1_cov(X, y, cl)
        df = float(max(g - 1, 1))

        slope_h = float(beta[1])
        slope_g = float(beta[1] + beta[3])
        se_slope_g = float(np.sqrt(max(
            cov[1, 1] + cov[3, 3] + 2.0 * cov[1, 3], 0.0
        )))
        ci_c = _ci95(float(beta[1]), float(se[1]), df)
        ci_genai = _ci95(float(beta[2]), float(se[2]), df)
        ci_inter = _ci95(float(beta[3]), float(se[3]), df)
        ci_slope_g = _ci95(slope_g, se_slope_g, df)

        results.append({
            "complexity_key": ckey,
            "complexity_label": clabel,
            "c_mean": mu,
            "c_sd": sd,
            "n": float(len(y)),
            "n_clusters": float(g),
            "df": df,
            "beta_c": float(beta[1]),
            "se_c": float(se[1]),
            "p_c": float(p[1]),
            "ci_c_lo": ci_c[0],
            "ci_c_hi": ci_c[1],
            "beta_genai": float(beta[2]),
            "se_genai": float(se[2]),
            "p_genai": float(p[2]),
            "ci_genai_lo": ci_genai[0],
            "ci_genai_hi": ci_genai[1],
            "beta_inter": float(beta[3]),
            "se_inter": float(se[3]),
            "p_inter": float(p[3]),
            "ci_inter_lo": ci_inter[0],
            "ci_inter_hi": ci_inter[1],
            "slope_human": slope_h,
            "slope_genai": slope_g,
            "se_slope_genai": se_slope_g,
            "ci_slope_human_lo": ci_c[0],
            "ci_slope_human_hi": ci_c[1],
            "ci_slope_genai_lo": ci_slope_g[0],
            "ci_slope_genai_hi": ci_slope_g[1],
        })
    return results


def build_panel_c_tex(pooled_fits: list[dict]) -> str:
    lines: list[str] = []
    for f in pooled_fits:
        se_sg = float(f["se_slope_genai"])
        if np.isfinite(se_sg) and se_sg > 0:
            p_slope_g = float(
                2 * t_dist.sf(abs(f["slope_genai"] / se_sg), f["df"])
            )
        else:
            p_slope_g = float("nan")
        lines.append(
            " & ".join([
                f["complexity_label"],
                _fmt_coef_ci_cell(
                    f["beta_c"], f["ci_c_lo"], f["ci_c_hi"], f["p_c"]
                ),
                _fmt_coef_ci_cell(
                    f["beta_genai"], f["ci_genai_lo"], f["ci_genai_hi"], f["p_genai"]
                ),
                _fmt_coef_ci_cell(
                    f["beta_inter"], f["ci_inter_lo"], f["ci_inter_hi"], f["p_inter"]
                ),
                _fmt_coef_ci_cell(
                    f["slope_human"],
                    f["ci_slope_human_lo"],
                    f["ci_slope_human_hi"],
                    f["p_c"],
                ),
                _fmt_coef_ci_cell(
                    f["slope_genai"],
                    f["ci_slope_genai_lo"],
                    f["ci_slope_genai_hi"],
                    p_slope_g,
                ),
            ])
            + r" \\"
        )
        lines.append("")

    return "\n".join([
        "% Auto-generated by diagram/_lib/theory_complexity_quality.py",
        "",
        r"{\small\textbf{c.\enspace Association between complexity and "
        r"LLM-assessed theory quality ratings}}\\[0.25em]",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\textbf{Complexity metric ($C$)}",
        r"& \textbf{$\boldsymbol{\beta}_{1}$ ($C$)}",
        r"& \textbf{$\boldsymbol{\beta}_{2}$ (GenAI)}",
        r"& \textbf{$\boldsymbol{\beta}_{3}$ ($C{\times}$GenAI)}",
        r"& \textbf{Human slope}",
        r"& \textbf{GenAI slope} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
    ])


# Keep old name for any external imports.
build_panel_d_tex = build_panel_c_tex


def panel_c_tex() -> str:
    """Panel (c) TeX body for the combined theory-complexity figure."""
    rows = build_person_phase_rows()
    z_stats = complexity_z_stats(rows)
    return build_panel_c_tex(fit_pooled_models(rows, z_stats))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_person_phase_rows()
    z_stats = complexity_z_stats(rows)
    print(f"Person×task×phase base rows: {len(rows)}")
    for ckey, clabel in COMPLEXITY_METRICS:
        mu, sd = z_stats[ckey]
        print(f"  {clabel}: mean={mu:.3f}, SD={sd:.3f}")

    pooled = fit_pooled_models(rows, z_stats)

    summary = []
    for f in pooled:
        summary.append({
            "complexity": f["complexity_key"],
            "complexity_label": f["complexity_label"],
            "n": int(f["n"]),
            "n_clusters": int(f["n_clusters"]),
            "beta_c": f["beta_c"],
            "p_c": f["p_c"],
            "beta_genai": f["beta_genai"],
            "p_genai": f["p_genai"],
            "beta_interaction": f["beta_inter"],
            "p_interaction": f["p_inter"],
            "slope_human": f["slope_human"],
            "slope_genai": f["slope_genai"],
            "ci_c_lo": f["ci_c_lo"],
            "ci_c_hi": f["ci_c_hi"],
            "ci_slope_genai_lo": f["ci_slope_genai_lo"],
            "ci_slope_genai_hi": f["ci_slope_genai_hi"],
        })
        print(
            f"  {f['complexity_label']:28} N={int(f['n']):4d}  "
            f"β1={f['beta_c']:+.3f} (p={f['p_c']:.3f})  "
            f"human={f['slope_human']:+.3f}  genai={f['slope_genai']:+.3f}"
        )
    _write_csv(OUT_DIR / "theory_complexity_quality_summary.csv", summary)


if __name__ == "__main__":
    main()
