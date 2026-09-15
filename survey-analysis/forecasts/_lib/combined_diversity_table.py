"""
Nature-style prediction/error diversity table
(Humans vs GenAI; D = 1 − mean pairwise cosine) plus γ̂_P / γ̂_E.

Reads existing analysis CSVs.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS, LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.latex_table_pdf import compile_standalone_table  # noqa: E402

OUT_DIR = FORECASTS / "outputs"
TABLE_STEM = "fig_forecast_diversity"
# Temp compile stem (tectonic writes ``{stem}_standalone.*``).
_COMPILE_STEM = "forecast_diversity_combined_table"

REDUNDANCY_CSV = OUT_DIR / "forecast_error_redundancy_detail.csv"
GAIN_COEF_CSV = OUT_DIR / "diversity_explains_gain_coefs.csv"

TASKS = ("Race", "Gender")
TASK_LABELS = {
    "Race": "Racial inequality",
    "Gender": "Gender inequality",
}
ANALYSES = (
    ("Main Effects", "Main effects"),
    ("Interactions", "Interactions"),
)

# Leftover / renamed outputs (removed by request).
_LEGACY_DAP_OUTPUTS = (
    OUT_DIR / "forecast_diversity_dap_by_lambda.pdf",
    OUT_DIR / "forecast_diversity_dap_by_lambda.svg",
    OUT_DIR / "forecast_diversity_dap_by_lambda_components.csv",
    OUT_DIR / f"{_COMPILE_STEM}.tex",
    OUT_DIR / f"{_COMPILE_STEM}_standalone.tex",
    OUT_DIR / f"{_COMPILE_STEM}_standalone.pdf",
    OUT_DIR / f"{_COMPILE_STEM}_standalone.svg",
    OUT_DIR / f"{_COMPILE_STEM}_standalone.png",
)


def _latex_num(value: float, *, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        return f"$-{text[1:]}$"
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


def _latex_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        text = "$<$0.001"
    else:
        text = f"{p:.3f}"
    return f"{text}{_stars(p)}"


def _latex_coef_stars(coef: float, p: float) -> str:
    num = _latex_num(coef)
    stars = _stars(p)
    if num.startswith("$") and num.endswith("$") and stars.startswith("$"):
        return f"${num[1:-1]}{stars[1:]}"
    return f"{num}{stars}"


def _diversity_from_sim(sim: float) -> float:
    if not np.isfinite(sim):
        return float("nan")
    return 1.0 - sim


def _diversity_ci_from_sim(sim_lo: float, sim_hi: float) -> tuple[float, float]:
    """Map similarity CI → diversity CI (D = 1 − sim; bounds flip)."""
    if not (np.isfinite(sim_lo) and np.isfinite(sim_hi)):
        return float("nan"), float("nan")
    return 1.0 - sim_hi, 1.0 - sim_lo


def _latex_plain_num(value: float, *, decimals: int = 3) -> str:
    """Plain decimal for use inside math-mode CI brackets."""
    if not np.isfinite(value):
        return r"\mathrm{---}"
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        return f"-{text[1:]}"
    return text


def _latex_point_ci(point: float, lo: float, hi: float, *, decimals: int = 3) -> str:
    if not np.isfinite(point):
        return "—"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _latex_num(point, decimals=decimals)
    return (
        r"\begin{tabular}{@{}c@{}}"
        rf"{_latex_num(point, decimals=decimals)}\\"
        rf"$[{_latex_plain_num(lo, decimals=decimals)},\ "
        rf"{_latex_plain_num(hi, decimals=decimals)}]$"
        r"\end{tabular}"
    )


def _latex_coef_ci_stars(
    coef: float, se: float, p: float, *, decimals: int = 3, z: float = 1.96
) -> str:
    """Point estimate with asymptotic 95% CI and significance stars."""
    if not np.isfinite(coef):
        return "—"
    stars = _stars(p)
    if not np.isfinite(se):
        return f"{_latex_num(coef, decimals=decimals)}{stars}"
    lo = coef - z * se
    hi = coef + z * se
    point = _latex_num(coef, decimals=decimals)
    # Keep stars on the point line.
    if point.startswith("$") and point.endswith("$") and stars.startswith("$"):
        point_stars = f"${point[1:-1]}{stars[1:]}"
    else:
        point_stars = f"{point}{stars}"
    return (
        r"\begin{tabular}{@{}c@{}}"
        rf"{point_stars}\\"
        rf"$[{_latex_plain_num(lo, decimals=decimals)},\ "
        rf"{_latex_plain_num(hi, decimals=decimals)}]$"
        r"\end{tabular}"
    )


def load_redundancy(
    path: Path = REDUNDANCY_CSV,
) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["analysis"], row["task"])
            out[key] = {
                k: float(row[k])
                for k in row
                if k not in ("task", "analysis")
            }
    return out


def load_gamma(
    path: Path = GAIN_COEF_CSV,
) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["term"] != "diversity":
                continue
            key = (row["diversity"], row["analysis"], row["task"])
            out[key] = {
                "coef": float(row["coef"]),
                "se": float(row["se"]),
                "p": float(row["p"]),
            }
    return out


def build_diversity_tex(
    redundancy: dict[tuple[str, str], dict[str, float]],
    gamma: dict[tuple[str, str, str], dict[str, float]],
) -> str:
    n_cols = 9
    body: list[str] = []
    for section_idx, (analysis_key, section_label) in enumerate(ANALYSES):
        if section_idx > 0:
            body.append(r"\midrule")
        body.append(
            rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{section_label}}}}} \\"
        )
        for task in TASKS:
            red = redundancy[(analysis_key, task)]
            g_f = gamma[("diversity_forecast", analysis_key, task)]
            g_e = gamma[("diversity_error", analysis_key, task)]

            d_fh = _diversity_from_sim(red["forecast_sim_human"])
            d_fh_lo, d_fh_hi = _diversity_ci_from_sim(
                red["forecast_sim_human_ci_lo"], red["forecast_sim_human_ci_hi"]
            )
            d_fg = _diversity_from_sim(red["forecast_sim_genai"])
            d_fg_lo, d_fg_hi = _diversity_ci_from_sim(
                red["forecast_sim_genai_ci_lo"], red["forecast_sim_genai_ci_hi"]
            )
            d_eh = _diversity_from_sim(red["error_sim_human"])
            d_eh_lo, d_eh_hi = _diversity_ci_from_sim(
                red["error_sim_human_ci_lo"], red["error_sim_human_ci_hi"]
            )
            d_eg = _diversity_from_sim(red["error_sim_genai"])
            d_eg_lo, d_eg_hi = _diversity_ci_from_sim(
                red["error_sim_genai_ci_lo"], red["error_sim_genai_ci_hi"]
            )

            body.append(
                f"{TASK_LABELS[task]} & "
                f"{_latex_point_ci(d_fh, d_fh_lo, d_fh_hi)} & "
                f"{_latex_point_ci(d_fg, d_fg_lo, d_fg_hi)} & "
                f"{_latex_p(red['forecast_p'])} & "
                f"{_latex_coef_ci_stars(g_f['coef'], g_f['se'], g_f['p'])} & "
                f"{_latex_point_ci(d_eh, d_eh_lo, d_eh_hi)} & "
                f"{_latex_point_ci(d_eg, d_eg_lo, d_eg_hi)} & "
                f"{_latex_p(red['error_p'])} & "
                f"{_latex_coef_ci_stars(g_e['coef'], g_e['se'], g_e['p'])} \\\\"
            )
    return "\n".join([
        "% Auto-generated by combined_diversity_table.py",
        "% Diversity cells: point + 95% bootstrap CI (D = 1 − mean pairwise cosine).",
        "% Gamma cells: point + asymptotic 95% CI (coef ± 1.96 SE) with stars.",
        "",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}l*{8}{c}@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Prediction diversity} "
        r"& \multicolumn{4}{c}{Error diversity} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Task & Humans & GenAI & $P$ & $\hat{\gamma}_{P}$"
        r" & Humans & GenAI & $P$ & $\hat{\gamma}_{E}$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
    ])


def write_combined_diversity_table(
    *,
    redundancy_csv: Path = REDUNDANCY_CSV,
    gain_coef_csv: Path = GAIN_COEF_CSV,
    out_dir: Path = OUT_DIR,
) -> Path:
    if not redundancy_csv.is_file():
        raise FileNotFoundError(f"Missing redundancy CSV: {redundancy_csv}")
    if not gain_coef_csv.is_file():
        raise FileNotFoundError(f"Missing gain-coef CSV: {gain_coef_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tex = build_diversity_tex(
        load_redundancy(redundancy_csv),
        load_gamma(gain_coef_csv),
    )
    dest_png = out_dir / f"{TABLE_STEM}.png"
    dest_svg = out_dir / f"{TABLE_STEM}.svg"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        png_tmp = compile_standalone_table(
            tmp_dir,
            _COMPILE_STEM,
            tex,
            output_format="svg+png",
            png_dpi=400,
            crop="standalone",
        )
        svg_tmp = tmp_dir / f"{_COMPILE_STEM}_standalone.svg"
        shutil.move(str(png_tmp), str(dest_png))
        if svg_tmp.is_file():
            shutil.move(str(svg_tmp), str(dest_svg))
    for legacy in _LEGACY_DAP_OUTPUTS:
        legacy.unlink(missing_ok=True)
    return dest_png


def main() -> None:
    png = write_combined_diversity_table()
    print(f"Saved: {png}")
    print(f"Saved: {OUT_DIR / f'{TABLE_STEM}.svg'}")


if __name__ == "__main__":
    main()
