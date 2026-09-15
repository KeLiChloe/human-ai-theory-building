"""
Diversity coefficient in aggregation-gain OLS, by task × effect.

Equal-sized resampled crowds (Humans & GenAI, k=2..min n).

Two diversity measures:
  D^f_c = 1 − mean pairwise forecast cosine
  D^e_c = 1 − mean pairwise error cosine,  e_i = h_i − b

Gain Δ_c = agg cosine − mean individual cosine.

Within each (task, effect) cell, OLS (HC1):

  Δ_c = β_0 + γ D_c + β_1 Human_c + β_2 Acc̄_c + β_3 k_c + ε_c

Reports γ for each cell, once for each diversity measure.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation import (  # noqa: E402
    GENAI_GROUP_IDS,
    HUMAN_GROUP_IDS,
    cosine_sim,
    load_main_effects_records,
    load_soi_records,
    plot_pts_main_effects,
    plot_pts_soi,
)
OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)
B = 200  # draws per (analysis, task, group, k)
SEED = 20260715
K_MIN = 2
TASKS = ("Race", "Gender")
TASK_LABELS = {
    "Race": "Racial inequality",
    "Gender": "Gender inequality",
}
ANALYSES = (
    ("Main Effects", "Main effects"),
    ("Interactions", "Interactions"),
)

# (key in draws, print label)
DIVERSITY_SPECS = (
    ("diversity_forecast", "Prediction diversity"),
    ("diversity_error", "Error diversity"),
)


def mean_pairwise_cosine(vecs: np.ndarray) -> float:
    if vecs.ndim != 2 or vecs.shape[0] < 2:
        return np.nan
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 0
    if int(np.sum(valid)) < 2:
        return np.nan
    unit = vecs[valid] / norms[valid, None]
    gram = unit @ unit.T
    iu = np.triu_indices(gram.shape[0], k=1)
    return float(np.mean(gram[iu]))


def _split_pools(pts: list[dict]) -> tuple[list[dict], list[dict]]:
    humans = [p for p in pts if p["group"] in HUMAN_GROUP_IDS]
    genai = [p for p in pts if p["group"] in GENAI_GROUP_IDS]
    return humans, genai


def sample_crowd_row(
    pool: list[dict],
    ml_vec: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    idx = rng.choice(len(pool), size=k, replace=False)
    selected = [pool[i] for i in idx]
    scores = np.asarray([p["score"] for p in selected], dtype=float)
    vecs = np.asarray([p["vec"] for p in selected], dtype=float)
    if np.any(np.isnan(scores)):
        return None
    mean_ind = float(np.mean(scores))
    agg = cosine_sim(np.sum(vecs, axis=0), ml_vec)
    if np.isnan(agg):
        return None
    forecast_sim = mean_pairwise_cosine(vecs)
    error_sim = mean_pairwise_cosine(vecs - ml_vec.reshape(1, -1))
    if np.isnan(forecast_sim) or np.isnan(error_sim):
        return None
    return {
        "mean_ind": mean_ind,
        "agg": float(agg),
        "gain": float(agg - mean_ind),
        "diversity_forecast": float(1.0 - forecast_sim),
        "diversity_error": float(1.0 - error_sim),
        "forecast_sim": float(forecast_sim),
        "error_sim": float(error_sim),
    }


def collect_draws(
    pts: list[dict],
    ml_vec: np.ndarray,
    *,
    analysis: str,
    task: str,
    rng: np.random.Generator,
    n_draws: int,
) -> list[dict[str, object]]:
    humans, genai = _split_pools(pts)
    k_max = min(len(humans), len(genai))
    rows: list[dict[str, object]] = []
    for group_name, pool, is_human in (
        ("Humans", humans, 1),
        ("GenAI", genai, 0),
    ):
        for k in range(K_MIN, k_max + 1):
            for draw in range(n_draws):
                stats = sample_crowd_row(pool, ml_vec, k, rng)
                if stats is None:
                    continue
                rows.append({
                    "analysis": analysis,
                    "task": task,
                    "group": group_name,
                    "is_human": is_human,
                    "k": k,
                    "draw": draw,
                    **stats,
                })
    return rows


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """OLS with HC1 robust standard errors."""
    n, p = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = max(n - rank, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((p, p))
    scale = n / df
    for i in range(n):
        xi = X[i : i + 1].T
        meat += float(resid[i] ** 2) * (xi @ xi.T)
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_stats), df)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"beta": beta, "se": se, "p": p_vals, "r2": np.array([r2]), "n": np.array([n])}


def design_matrix(
    rows: list[dict[str, object]],
    *,
    diversity_key: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Within-cell: Δ ~ D + Human + Acc̄ + k."""
    y = np.asarray([float(r["gain"]) for r in rows], dtype=float)
    X = np.column_stack([
        np.ones(len(rows)),
        np.asarray([float(r[diversity_key]) for r in rows]),
        np.asarray([float(r["is_human"]) for r in rows]),
        np.asarray([float(r["mean_ind"]) for r in rows]),
        np.asarray([float(r["k"]) for r in rows]),
    ])
    names = ["intercept", "diversity", "human", "mean_ind", "k"]
    return X, y, names


def fit_ols_coefs(
    rows: list[dict[str, object]],
    *,
    diversity_key: str,
) -> dict[str, float | dict[str, float]]:
    """Return per-term coef/SE/p plus model R² and N."""
    X, y, names = design_matrix(rows, diversity_key=diversity_key)
    fit = ols_fit(X, y)
    out: dict[str, float | dict[str, float]] = {
        "r2": float(fit["r2"][0]),
        "n": float(fit["n"][0]),
    }
    for i, name in enumerate(names):
        out[name] = {
            "coef": float(fit["beta"][i]),
            "se": float(fit["se"][i]),
            "p": float(fit["p"][i]),
        }
    return out


# Focal term in the paper table (controls Human, Acc̄, k estimated but not shown).
TABLE_TERMS = ("diversity",)
TABLE_GAMMA_HEADERS = (
    r"$\hat{\gamma}_{F}$",
    r"$\hat{\gamma}_{E}$",
)


def _latex_num(x: float, *, decimals: int = 3, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "---"
    if signed:
        text = f"{x:+.{decimals}f}"
    else:
        text = f"{x:.{decimals}f}"
    if text.startswith("-"):
        return f"$-{text[1:]}$"
    if text.startswith("+"):
        return f"$+{text[1:]}$"
    return text


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def _latex_coef_stars(coef: float, p: float, *, bold: bool = False) -> str:
    num = _latex_num(coef)
    stars = _stars(p)
    if bold:
        if num.startswith("$") and num.endswith("$"):
            num_out = rf"$\boldsymbol{{{num[1:-1]}}}$"
        else:
            num_out = rf"\textbf{{{num}}}"
        return f"{num_out}{stars}"
    if num.startswith("$") and num.endswith("$") and stars.startswith("$"):
        return f"${num[1:-1]}{stars[1:]}"
    return f"{num}{stars}"


def build_diversity_coef_tex(
    results_by_measure: dict[str, dict[tuple[str, str], dict]],
    *,
    n_per_cell: int,
) -> str:
    del n_per_cell
    n_gamma = len(TABLE_GAMMA_HEADERS)
    n_cols = 1 + n_gamma
    header_row = "\n& ".join(
        rf"$\boldsymbol{{{h[1:-1]}}}$" if h.startswith("$") and h.endswith("$") else h
        for h in TABLE_GAMMA_HEADERS
    )

    body: list[str] = [
        "Task",
        f"& {header_row} \\\\",
        r"\midrule",
        "",
    ]
    for section_idx, (analysis_key, analysis_label) in enumerate(ANALYSES):
        if section_idx > 0:
            body.append(r"\addlinespace[0.35em]")
        body.append(
            rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{analysis_label}}}}} \\"
        )
        body.append("")
        for task in TASKS:
            cells = [TASK_LABELS[task]]
            for div_key, _ in DIVERSITY_SPECS:
                fit = results_by_measure[div_key][(analysis_key, task)]
                d = fit["diversity"]
                assert isinstance(d, dict)
                cells.append(_latex_coef_stars(d["coef"], d["p"], bold=False))
            body.append("\n& ".join(cells) + r" \\")
            body.append("")

    return "\n".join([
        "% Auto-generated by diversity_explains_gain.py",
        "",
        r"\begin{threeparttable}",
        rf"\begin{{tabular}}{{@{{}}l@{{\hspace{{24pt}}}}*{{{n_gamma}}}{{c}}@{{}}}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Reports the diversity coefficient $\hat{\gamma}$ from "
        r"$\Delta_{c}=\beta_{0}+\gamma D_{c}+\beta_{1}\mathrm{Human}_{c}"
        r"+\beta_{2}\overline{\mathrm{Acc}}_{c}+\beta_{3}k_{c}+\varepsilon_{c}$, "
        r"where $D^{F}$ is prediction diversity and $D^{E}$ is error diversity "
        r"(controls estimated but not shown). "
        r"Stars mark two-sided HC1 $t$-tests of $H_0$: $\gamma=0$ "
        r"($^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$; "
        r"approximate under overlapping resampled crowds).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        "",
    ])


def fit_diversity_coef(
    rows: list[dict[str, object]],
    *,
    diversity_key: str,
) -> dict[str, float]:
    """Backward-compatible: return diversity-term coef only."""
    fit = fit_ols_coefs(rows, diversity_key=diversity_key)
    d = fit["diversity"]  # type: ignore[index]
    assert isinstance(d, dict)
    return {
        "coef": float(d["coef"]),
        "se": float(d["se"]),
        "p": float(d["p"]),
        "r2": float(fit["r2"]),  # type: ignore[arg-type]
        "n": float(fit["n"]),  # type: ignore[arg-type]
    }


def main() -> None:
    csv_path = ROOT / "anonymous_survey_data.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]

    me_records, me_ml = load_main_effects_records(headers, data)
    soi_records, soi_ml = load_soi_records(headers, data)

    rng = np.random.default_rng(SEED)
    all_rows: list[dict[str, object]] = []

    for task in TASKS:
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"
        me_pts = plot_pts_main_effects(me_records, task_key, vec_key)
        me_pts = [p for p in me_pts if p.get("vec") is not None]
        all_rows.extend(
            collect_draws(
                me_pts, me_ml[task],
                analysis="Main Effects", task=task, rng=rng, n_draws=B,
            )
        )
        soi_pts = plot_pts_soi(soi_records, task_key, vec_key)
        soi_pts = [p for p in soi_pts if p.get("vec") is not None]
        all_rows.extend(
            collect_draws(
                soi_pts, soi_ml[task],
                analysis="Interactions", task=task, rng=rng, n_draws=B,
            )
        )

    detail_csv = OUT_DIR / "diversity_explains_gain_draws.csv"
    fieldnames = [
        "analysis", "task", "group", "is_human", "k", "draw",
        "mean_ind", "agg", "gain",
        "diversity_forecast", "diversity_error",
        "forecast_sim", "error_sim",
    ]
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                k: (f"{r[k]:.8f}" if isinstance(r[k], float) else r[k])
                for k in fieldnames
            })

    results_by_measure: dict[str, dict[tuple[str, str], dict]] = {}
    coef_rows: list[dict[str, object]] = []
    n_per_cell = 0

    for div_key, _ in DIVERSITY_SPECS:
        results: dict[tuple[str, str], dict] = {}
        for analysis_key, _ in ANALYSES:
            for task in TASKS:
                rows = [
                    r for r in all_rows
                    if r["analysis"] == analysis_key and r["task"] == task
                ]
                g = fit_ols_coefs(rows, diversity_key=div_key)
                results[(analysis_key, task)] = g
                n_per_cell = int(g["n"])  # type: ignore[arg-type]
                for term in ("intercept", *TABLE_TERMS):
                    term_fit = g[term]
                    assert isinstance(term_fit, dict)
                    coef_rows.append({
                        "diversity": div_key,
                        "analysis": analysis_key,
                        "task": task,
                        "term": term,
                        "coef": term_fit["coef"],
                        "se": term_fit["se"],
                        "p": term_fit["p"],
                        "r2": g["r2"],
                        "n": g["n"],
                    })
        results_by_measure[div_key] = results

    coef_csv = OUT_DIR / "diversity_explains_gain_coefs.csv"
    with coef_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["diversity", "analysis", "task", "term", "coef", "se", "p", "r2", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in coef_rows:
            writer.writerow({
                k: (f"{r[k]:.8f}" if isinstance(r[k], float) else r[k])
                for k in fields
            })

    print(f"Diversity coefficient γ by task × effect (B={B}, seed={SEED})")
    print(f"Draws: {len(all_rows)}; N per cell = {n_per_cell}")
    print(f"Saved: {detail_csv}")
    print(f"Saved: {coef_csv}")

    print()
    for div_key, label in DIVERSITY_SPECS:
        print(f"=== {label} ===")
        for analysis_key, _ in ANALYSES:
            for task in TASKS:
                g = results_by_measure[div_key][(analysis_key, task)]
                d = g["diversity"]
                assert isinstance(d, dict)
                print(
                    f"  {analysis_key} × {task}: γ={d['coef']:+.4f} "
                    f"(SE={d['se']:.4f}, p={d['p']:.4g}, R²={g['r2']:.3f})"
                )
        print()


if __name__ == "__main__":
    main()
