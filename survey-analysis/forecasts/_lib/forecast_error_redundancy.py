"""
Forecast / error redundancy: mean pairwise cosine similarities within group.

For each (analysis, task) and group (Humans, GenAI):

  Forecast similarity = mean_{i<j} cos(h_i, h_j)
  Error similarity    = mean_{i<j} cos(e_i, e_j),  e_i = h_i − b

where b is the ML benchmark vector.

Inference uses a forecaster-level permutation test of the Human−GenAI
difference (does not treat pairwise observations as independent).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation import (
    GENAI_GROUP_IDS,
    HUMAN_GROUP_IDS,
    load_main_effects_records,
    load_soi_records,
    plot_pts_main_effects,
    plot_pts_soi,
)
OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)
N_PERM = 5000
N_BOOT = 2000
SEED = 20260715
TASKS = ("Race", "Gender")
TASK_LABELS = {
    "Race": "Racial Inequality",
    "Gender": "Gender Inequality",
}
ANALYSIS_SECTIONS = (
    ("Main Effects", "Main Effects"),
    ("Interactions", "Interactions"),
)


def _stack_vecs(pts: list[dict]) -> np.ndarray:
    return np.asarray([p["vec"] for p in pts], dtype=float)


def mean_pairwise_cosine(vecs: np.ndarray) -> float:
    """Mean cos(v_i, v_j) over i < j. Zero-norm rows are dropped."""
    if vecs.ndim != 2 or vecs.shape[0] < 2:
        return np.nan
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 0
    if int(np.sum(valid)) < 2:
        return np.nan
    unit = vecs[valid] / norms[valid, None]
    gram = unit @ unit.T
    n = gram.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(gram[iu]))


def bootstrap_mean_pairwise_cosine(
    vecs: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI for mean pairwise cosine."""
    point = mean_pairwise_cosine(vecs)
    n = int(vecs.shape[0])
    if n < 2 or not np.isfinite(point):
        return float(point), float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = mean_pairwise_cosine(vecs[idx])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def similarity_stats(vecs: np.ndarray, ml_vec: np.ndarray) -> dict[str, float]:
    err = vecs - ml_vec.reshape(1, -1)
    return {
        "n": float(vecs.shape[0]),
        "n_pairs": float(vecs.shape[0] * (vecs.shape[0] - 1) // 2) if vecs.shape[0] >= 2 else 0.0,
        "forecast_sim": mean_pairwise_cosine(vecs),
        "error_sim": mean_pairwise_cosine(err),
    }


def permute_group_diff(
    all_vecs: np.ndarray,
    human_mask: np.ndarray,
    ml_vec: np.ndarray,
    *,
    metric: str,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Two-sided forecaster-level permutation test of Human − GenAI similarity."""
    n_human = int(np.sum(human_mask))
    n = all_vecs.shape[0]
    human_vecs = all_vecs[human_mask]
    genai_vecs = all_vecs[~human_mask]

    if metric == "forecast":
        obs_h = mean_pairwise_cosine(human_vecs)
        obs_g = mean_pairwise_cosine(genai_vecs)
    else:
        obs_h = mean_pairwise_cosine(human_vecs - ml_vec)
        obs_g = mean_pairwise_cosine(genai_vecs - ml_vec)
    obs_diff = obs_h - obs_g

    count_extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        h_idx = perm[:n_human]
        g_idx = perm[n_human:]
        if metric == "forecast":
            d_h = mean_pairwise_cosine(all_vecs[h_idx])
            d_g = mean_pairwise_cosine(all_vecs[g_idx])
        else:
            d_h = mean_pairwise_cosine(all_vecs[h_idx] - ml_vec)
            d_g = mean_pairwise_cosine(all_vecs[g_idx] - ml_vec)
        if abs(d_h - d_g) >= abs(obs_diff) - 1e-15:
            count_extreme += 1

    # Add-one smoothing: min p = 1/(N_perm+1) when count_extreme=0
    p_val = (count_extreme + 1) / (n_perm + 1)
    return {
        "obs_human": obs_h,
        "obs_genai": obs_g,
        "diff": obs_diff,
        "p_perm": float(p_val),
        "n_extreme": float(count_extreme),
    }


def analyze_pool(
    pts: list[dict],
    ml_vec: np.ndarray,
    *,
    analysis: str,
    task: str,
    seed: int,
) -> dict[str, float]:
    humans = [p for p in pts if p["group"] in HUMAN_GROUP_IDS]
    genai = [p for p in pts if p["group"] in GENAI_GROUP_IDS]
    h_vecs = _stack_vecs(humans)
    g_vecs = _stack_vecs(genai)
    h_stats = similarity_stats(h_vecs, ml_vec)
    g_stats = similarity_stats(g_vecs, ml_vec)

    all_vecs = np.vstack([h_vecs, g_vecs])
    human_mask = np.array([True] * len(h_vecs) + [False] * len(g_vecs), dtype=bool)

    # Deterministic per-cell streams (do NOT use built-in hash: PYTHONHASHSEED randomizes it)
    cell_id = {
        ("Main Effects", "Race"): 11,
        ("Main Effects", "Gender"): 12,
        ("Interactions", "Race"): 21,
        ("Interactions", "Gender"): 22,
    }[(analysis, task)]
    forecast_rng = np.random.default_rng([seed, cell_id, 1])
    error_rng = np.random.default_rng([seed, cell_id, 2])
    boot_rng = np.random.default_rng([seed, cell_id, 3])

    forecast_perm = permute_group_diff(
        all_vecs, human_mask, ml_vec, metric="forecast", n_perm=N_PERM, rng=forecast_rng
    )
    error_perm = permute_group_diff(
        all_vecs, human_mask, ml_vec, metric="error", n_perm=N_PERM, rng=error_rng
    )

    f_h, f_h_lo, f_h_hi = bootstrap_mean_pairwise_cosine(
        h_vecs, n_boot=N_BOOT, rng=boot_rng
    )
    f_g, f_g_lo, f_g_hi = bootstrap_mean_pairwise_cosine(
        g_vecs, n_boot=N_BOOT, rng=boot_rng
    )
    e_h, e_h_lo, e_h_hi = bootstrap_mean_pairwise_cosine(
        h_vecs - ml_vec.reshape(1, -1), n_boot=N_BOOT, rng=boot_rng
    )
    e_g, e_g_lo, e_g_hi = bootstrap_mean_pairwise_cosine(
        g_vecs - ml_vec.reshape(1, -1), n_boot=N_BOOT, rng=boot_rng
    )

    return {
        "n_human": h_stats["n"],
        "n_genai": g_stats["n"],
        "n_pairs_human": h_stats["n_pairs"],
        "n_pairs_genai": g_stats["n_pairs"],
        "forecast_sim_human": f_h,
        "forecast_sim_human_ci_lo": f_h_lo,
        "forecast_sim_human_ci_hi": f_h_hi,
        "forecast_sim_genai": f_g,
        "forecast_sim_genai_ci_lo": f_g_lo,
        "forecast_sim_genai_ci_hi": f_g_hi,
        "forecast_diff": forecast_perm["diff"],
        "forecast_p": forecast_perm["p_perm"],
        "error_sim_human": e_h,
        "error_sim_human_ci_lo": e_h_lo,
        "error_sim_human_ci_hi": e_h_hi,
        "error_sim_genai": e_g,
        "error_sim_genai_ci_lo": e_g_lo,
        "error_sim_genai_ci_hi": e_g_hi,
        "error_diff": error_perm["diff"],
        "error_p": error_perm["p_perm"],
    }


def _latex_num(value: float, *, decimals: int = 3, signed: bool = False) -> str:
    if np.isnan(value):
        return "—"
    text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def _latex_p(p: float) -> str:
    """Show p to 3 decimals with stars; values below 0.001 as <0.001***."""
    if np.isnan(p):
        return "—"
    if p < 0.001:
        text, stars = "$<$0.001", "$^{***}$"
    elif p < 0.01:
        text, stars = f"{p:.3f}", "$^{**}$"
    elif p < 0.05:
        text, stars = f"{p:.3f}", "$^{*}$"
    else:
        text, stars = f"{p:.3f}", ""
    return f"{text}{stars}"


def _bold_if(text: str, *, bold: bool) -> str:
    return f"\\textbf{{{text}}}" if bold else text


def build_redundancy_tex(table: dict[str, dict[str, dict[str, float]]]) -> str:
    """Stacked Main Effects / Interactions; Humans vs GenAI forecast & error similarity."""
    body: list[str] = []
    for section_idx, (analysis_key, section_label) in enumerate(ANALYSIS_SECTIONS):
        if section_idx > 0:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{section_label}}}}} \\\\")
        for task in TASKS:
            g = table[task][analysis_key]
            # Bold the lower similarity (more diverse / less redundant)
            f_h, f_g = g["forecast_sim_human"], g["forecast_sim_genai"]
            e_h, e_g = g["error_sim_human"], g["error_sim_genai"]
            fh_tex = _bold_if(_latex_num(f_h), bold=f_h < f_g)
            fg_tex = _bold_if(_latex_num(f_g), bold=f_g < f_h)
            eh_tex = _bold_if(_latex_num(e_h), bold=e_h < e_g)
            eg_tex = _bold_if(_latex_num(e_g), bold=e_g < e_h)
            body.append(
                f"{TASK_LABELS[task]} & "
                f"{fh_tex} & {fg_tex} & {_latex_p(g['forecast_p'])} & "
                f"{eh_tex} & {eg_tex} & {_latex_p(g['error_p'])} \\\\"
            )

    return "\n".join([
        "% Auto-generated by forecast_error_redundancy.py",
        "",
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{5.5pt}",
        "\\renewcommand{\\arraystretch}{1.35}",
        "\\begin{center}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Prediction similarity} "
        "& \\multicolumn{3}{c}{Error similarity} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Task & Humans & GenAI & $p$ & Humans & GenAI & $p$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
        "}",
        "",
    ])


def main() -> None:
    csv_path = ROOT / "anonymous_survey_data.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]

    me_records, me_ml = load_main_effects_records(headers, data)
    soi_records, soi_ml = load_soi_records(headers, data)

    table: dict[str, dict[str, dict[str, float]]] = {}
    detail_rows: list[dict[str, object]] = []

    for task in TASKS:
        table[task] = {}
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"

        for analysis, pts_fn, ml_map in (
            ("Main Effects", plot_pts_main_effects, me_ml),
            ("Interactions", plot_pts_soi, soi_ml),
        ):
            pts = pts_fn(me_records if analysis == "Main Effects" else soi_records, task_key, vec_key)
            pts = [p for p in pts if p.get("vec") is not None and np.linalg.norm(p["vec"]) > 0]
            stats = analyze_pool(
                pts, ml_map[task], analysis=analysis, task=task, seed=SEED
            )
            table[task][analysis] = stats
            detail_rows.append({"task": task, "analysis": analysis, **stats})

    detail_csv = OUT_DIR / "forecast_error_redundancy_detail.csv"
    fieldnames = [
        "task", "analysis",
        "n_human", "n_genai", "n_pairs_human", "n_pairs_genai",
        "forecast_sim_human", "forecast_sim_human_ci_lo", "forecast_sim_human_ci_hi",
        "forecast_sim_genai", "forecast_sim_genai_ci_lo", "forecast_sim_genai_ci_hi",
        "forecast_diff", "forecast_p",
        "error_sim_human", "error_sim_human_ci_lo", "error_sim_human_ci_hi",
        "error_sim_genai", "error_sim_genai_ci_lo", "error_sim_genai_ci_hi",
        "error_diff", "error_p",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({
                k: (f"{row[k]:.8f}" if isinstance(row[k], float) else row[k])
                for k in fieldnames
            })

    print(
        f"Forecast / error redundancy "
        f"(permutation N={N_PERM}, bootstrap N={N_BOOT}, seed={SEED})"
    )
    print(f"Saved: {detail_csv}")

    print()

    for analysis_key, _ in ANALYSIS_SECTIONS:
        print(f"=== {analysis_key} ===")
        for task in TASKS:
            g = table[task][analysis_key]
            print(
                f"  {task}: forecast H={g['forecast_sim_human']:.3f} AI={g['forecast_sim_genai']:.3f} "
                f"diff={g['forecast_diff']:+.3f} p={g['forecast_p']:.4f} | "
                f"error H={g['error_sim_human']:.3f} AI={g['error_sim_genai']:.3f} "
                f"diff={g['error_diff']:+.3f} p={g['error_p']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
