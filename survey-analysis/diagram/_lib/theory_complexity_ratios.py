"""
Complexity–performance ratios for Debate 4 / parsimony.

  ratio = outcome / C

Higher = more outcome per unit complexity (more “efficient”).

Accuracy / C (Pre-ML only, like panel b):
  Diagram:    acc_ME / C_diagram
  Word count: acc_ME / len(ME text) and acc_SOI / len(SOI text)
              (matched explanation lengths; stacked into the group mean)

Quality / C (Pre+Post, like panel c):
  Diagram:    quality_ME / C_diagram
  Word count: quality_ME / len(ME text) and quality_SOI / len(SOI text)
              per task × phase (up to 2×2×2 ratios per person)

C for diagrams is raw diagram complexity. Rows with C ≤ 0 or non-finite
outcome are dropped.

Comparisons (Welch two-sided):
  Humans vs GenAI; Senior vs PhD; Topic Experts vs Non-Experts.

Outputs under diagram/outputs/:
  theory_complexity_ratios_summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent
DIAGRAM = LIB.parent
ROOT = DIAGRAM.parent
FORECASTS = ROOT / "forecasts"
FORECASTS_LIB = FORECASTS / "_lib"
for p in (ROOT, DIAGRAM, LIB, FORECASTS_LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import bootstrap_mean_ci, p_value_welch_ttest  # noqa: E402
from theory_complexity_quality import (  # noqa: E402
    COMPLEXITY_METRICS,
    build_person_phase_rows,
)
from theory_efficiency import build_person_rows  # noqa: E402

OUT_DIR = DIAGRAM / "outputs"

# (comparison_key, latex comparison cell, left mask, right mask)
COMPARISONS = (
    (
        "human_vs_genai",
        r"Humans (Group 1) vs.\ GenAI (Group 2)",
        "human",
        "genai",
    ),
    (
        "senior_vs_phd",
        r"Senior (Group 1) vs.\ PhD (Group 2)",
        "senior",
        "phd",
    ),
    (
        "expert_vs_non",
        r"Experts (Group 1) vs.\ Non-Experts (Group 2)",
        "expert",
        "non_expert",
    ),
)


def _stars(p: float) -> str:
    if not np.isfinite(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    return r"^{*}"


def _fmt_mean(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    return f"{float(arr.mean()):.3f}"


def _fmt_p(p: float) -> str:
    """Always show numeric p; stars only when p < 0.05."""
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return rf"$<0.001{_stars(p)}$"
    return rf"${p:.3f}{_stars(p)}$"


def _one_ratio(outcome: float, c: float) -> float | None:
    if not np.isfinite(outcome) or not np.isfinite(c) or c <= 0:
        return None
    return float(outcome) / float(c)


def _mask_rows(rows: list[dict], which: str) -> list[dict]:
    if which == "human":
        return [r for r in rows if r["group"] == "Human"]
    if which == "genai":
        return [r for r in rows if r["group"] == "GenAI"]
    if which == "senior":
        return [r for r in rows if r.get("group_id") == "1"]
    if which == "phd":
        return [r for r in rows if r.get("group_id") == "0"]
    if which == "expert":
        return [
            r for r in rows
            if r["group"] == "Human" and r.get("is_topic_expert")
        ]
    if which == "non_expert":
        return [
            r for r in rows
            if r["group"] == "Human" and not r.get("is_topic_expert")
        ]
    raise ValueError(which)


def _forecast_ratios(rows: list[dict], ckey: str) -> np.ndarray:
    """Pre-ML accuracy/C. Diagram: ME only; word count: matched ME/SOI lengths."""
    vals: list[float] = []
    for r in rows:
        if ckey == "n_words":
            for ykey, ckey_part in (
                ("acc_me", "n_words_me"),
                ("acc_soi", "n_words_soi"),
            ):
                ratio = _one_ratio(float(r[ykey]), float(r[ckey_part]))
                if ratio is not None:
                    vals.append(ratio)
        else:
            ratio = _one_ratio(float(r["acc_me"]), float(r[ckey]))
            if ratio is not None:
                vals.append(ratio)
    return np.asarray(vals, dtype=float)


def _quality_ratios(rows: list[dict], ckey: str) -> np.ndarray:
    """Pre+Post quality/C. Diagram: ME only; word count: matched ME/SOI lengths."""
    vals: list[float] = []
    for r in rows:
        if ckey == "n_words":
            for ykey, ckey_part in (
                ("quality_me", "n_words_me"),
                ("quality_soi", "n_words_soi"),
            ):
                ratio = _one_ratio(float(r[ykey]), float(r[ckey_part]))
                if ratio is not None:
                    vals.append(ratio)
        else:
            q = float(r.get("quality_me", r["quality"]))
            ratio = _one_ratio(q, float(r[ckey]))
            if ratio is not None:
                vals.append(ratio)
    return np.asarray(vals, dtype=float)


def summarize_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    p = p_value_welch_ttest(left, right)
    l_lo, l_hi = bootstrap_mean_ci(left, seed=seed)
    r_lo, r_hi = bootstrap_mean_ci(right, seed=seed + 17)
    return {
        "n_left": int(np.isfinite(left).sum()),
        "n_right": int(np.isfinite(right).sum()),
        "mean_left": float(np.nanmean(left)) if left.size else float("nan"),
        "mean_right": float(np.nanmean(right)) if right.size else float("nan"),
        "ci_left_lo": float(l_lo),
        "ci_left_hi": float(l_hi),
        "ci_right_lo": float(r_lo),
        "ci_right_hi": float(r_hi),
        "p": float(p),
        "seed": seed,
    }


def build_ratio_rows() -> list[dict]:
    """One summary row per comparison × complexity × outcome."""
    forecast_rows = build_person_rows()
    quality_rows = build_person_phase_rows()
    out: list[dict] = []
    seed = 20260816

    for comp_key, comp_label, left_name, right_name in COMPARISONS:
        for ckey, clabel in COMPLEXITY_METRICS:
            for outcome, source_rows, ratio_fn in (
                ("forecast", forecast_rows, _forecast_ratios),
                ("quality", quality_rows, _quality_ratios),
            ):
                left_rows = _mask_rows(source_rows, left_name)
                right_rows = _mask_rows(source_rows, right_name)
                left = ratio_fn(left_rows, ckey)
                right = ratio_fn(right_rows, ckey)
                s = summarize_pair(left, right, seed=seed)
                out.append({
                    "comparison_key": comp_key,
                    "comparison_label": comp_label,
                    "left_name": left_name,
                    "right_name": right_name,
                    "outcome": outcome,
                    "complexity_key": ckey,
                    "complexity_label": clabel,
                    "left_vals": left,
                    "right_vals": right,
                    **s,
                })
                seed += 1
    return out


def build_panel_d_tex(rows: list[dict]) -> str:
    """Comparison × metric; Accuracy/C and Quality/C side by side."""
    # index: (comp_key, ckey) -> {forecast: row, quality: row}
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        key = (str(r["comparison_key"]), str(r["complexity_key"]))
        by.setdefault(key, {})[str(r["outcome"])] = r

    n_metrics = len(COMPLEXITY_METRICS)
    lines: list[str] = []
    for ci, (comp_key, comp_label, _, _) in enumerate(COMPARISONS):
        for mi, (ckey, clabel) in enumerate(COMPLEXITY_METRICS):
            f = by[(comp_key, ckey)]["forecast"]
            q = by[(comp_key, ckey)]["quality"]
            comp_cell = (
                rf"\multirow{{{n_metrics}}}{{*}}{{{comp_label}}}"
                if mi == 0
                else ""
            )
            lines.append(
                " & ".join([
                    comp_cell,
                    clabel,
                    _fmt_mean(np.asarray(f["left_vals"], dtype=float)),
                    _fmt_mean(np.asarray(f["right_vals"], dtype=float)),
                    _fmt_p(float(f["p"])),
                    _fmt_mean(np.asarray(q["left_vals"], dtype=float)),
                    _fmt_mean(np.asarray(q["right_vals"], dtype=float)),
                    _fmt_p(float(q["p"])),
                ])
                + r" \\"
            )
        if ci < len(COMPARISONS) - 1:
            lines.append(r"\midrule")

    return "\n".join([
        "% Auto-generated by diagram/_lib/theory_complexity_ratios.py",
        "",
        r"{\small\textbf{d.\enspace Performance--complexity ratios "
        r"(higher $=$ more efficient)}}\\[0.25em]",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}ll*{6}{c}@{}}",
        r"\toprule",
        r"&",
        r"& \multicolumn{3}{c}{\textbf{Accuracy / $C$}}",
        r"& \multicolumn{3}{c}{\textbf{Quality rating / $C$}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\textbf{Comparison} & \textbf{Complexity metric ($C$)}",
        r"& \textbf{Group 1} & \textbf{Group 2} & \textbf{Two-sided $P$}",
        r"& \textbf{Group 1} & \textbf{Group 2} & \textbf{Two-sided $P$} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
    ])


def panel_d_tex() -> str:
    """Panel (d) TeX body for the combined theory-complexity figure."""
    return build_panel_d_tex(build_ratio_rows())


def write_ratio_summary(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "comparison": r["comparison_key"],
            "comparison_label": r["comparison_label"],
            "complexity": r["complexity_key"],
            "complexity_label": r["complexity_label"],
            "outcome": r["outcome"],
            "n_left": r["n_left"],
            "n_right": r["n_right"],
            "mean_left": r["mean_left"],
            "mean_right": r["mean_right"],
            "p": r["p"],
        })
    csv_path = OUT_DIR / "theory_complexity_ratios_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {csv_path}")
    return csv_path


def main() -> None:
    rows = build_ratio_rows()
    write_ratio_summary(rows)
    for r in rows:
        print(
            f"  {r['comparison_label']:28} {r['complexity_label']:28} "
            f"{r['outcome']:8}  "
            f"L={r['mean_left']:.3f}  R={r['mean_right']:.3f}  "
            f"p={r['p']:.3g}"
        )


if __name__ == "__main__":
    main()
