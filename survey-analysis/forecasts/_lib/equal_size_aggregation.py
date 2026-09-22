"""
Equal-sized crowd aggregation analysis.

Matches Humans and GenAI at the same crowd size k by repeatedly sampling
k forecasters without replacement from each pool, aggregating forecast
vectors with the same sum→cosine rule as aggregation / 06_*,
and summarizing aggregated accuracy, raw gain, and ceiling-normalized gain.

Normalized gain = (agg − mean) / (1 − mean), NaN when remaining room ≤ ε.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
for p in (ROOT, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregation import (
    HUMAN_GROUP_IDS,
    GENAI_GROUP_IDS,
    compute_from_plot_pts,
    cosine_sim,
    load_main_effects_records,
    load_soi_records,
    plot_pts_main_effects,
    plot_pts_soi,
)
from common.viz_config import COLOR_AGG_HUMAN, GROUP_COLORS

OUT_DIR = FORECASTS / "outputs"
OUT_DIR.mkdir(exist_ok=True)

B = 1000
SEED = 20260714
NORM_EPS = 1e-6
K_MIN = 2

TASKS = ("Race", "Gender")

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "legend.frameon": False,
    "grid.alpha": 0.22,
    "grid.linestyle": ":",
    "lines.linewidth": 1.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})


def _split_pools(pts: list[dict]) -> tuple[list[dict], list[dict]]:
    humans = [p for p in pts if p["group"] in HUMAN_GROUP_IDS]
    genai = [p for p in pts if p["group"] in GENAI_GROUP_IDS]
    return humans, genai


def _sample_metrics(
    pool: list[dict],
    ml_vec: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Return mean_ind, agg, gain, norm_gain for one subsample of size k."""
    idx = rng.choice(len(pool), size=k, replace=False)
    selected = [pool[i] for i in idx]
    scores = np.array([p["score"] for p in selected], dtype=float)
    vecs = np.array([p["vec"] for p in selected], dtype=float)
    mean_ind = float(np.mean(scores))
    agg = cosine_sim(np.sum(vecs, axis=0), ml_vec)
    gain = agg - mean_ind
    room = 1.0 - mean_ind
    norm_gain = float(gain / room) if room > NORM_EPS else np.nan
    return mean_ind, agg, gain, norm_gain


def resample_curve(
    pool: list[dict],
    ml_vec: np.ndarray,
    k_values: list[int],
    n_draws: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, np.ndarray]]:
    """For each k, arrays of length n_draws for mean_ind / agg / gain / norm_gain."""
    out: dict[int, dict[str, np.ndarray]] = {}
    n = len(pool)
    for k in k_values:
        if k > n:
            continue
        mean_inds = np.empty(n_draws)
        aggs = np.empty(n_draws)
        gains = np.empty(n_draws)
        norms = np.empty(n_draws)
        for b in range(n_draws):
            mean_inds[b], aggs[b], gains[b], norms[b] = _sample_metrics(
                pool, ml_vec, k, rng
            )
        out[k] = {
            "mean_ind": mean_inds,
            "agg": aggs,
            "gain": gains,
            "norm_gain": norms,
        }
    return out


def _array_summary(arr: np.ndarray) -> dict[str, float]:
    """Mean / median / subsample percentiles, plus 95% CI of the Monte Carlo mean."""
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p025": np.nan,
            "p975": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }
    mean = float(np.mean(clean))
    n = int(clean.size)
    if n >= 2:
        se = float(np.std(clean, ddof=1) / np.sqrt(n))
        half = 1.96 * se
    else:
        half = 0.0
    return {
        "mean": mean,
        "median": float(np.median(clean)),
        "p025": float(np.percentile(clean, 2.5)),
        "p975": float(np.percentile(clean, 97.5)),
        "ci_lo": mean - half,
        "ci_hi": mean + half,
    }


def summarize_curves(
    analysis: str,
    task: str,
    group: str,
    n_pool: int,
    curves: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k, arrays in sorted(curves.items()):
        mean_s = _array_summary(arrays["mean_ind"])
        agg_s = _array_summary(arrays["agg"])
        gain_s = _array_summary(arrays["gain"])
        norm_s = _array_summary(arrays["norm_gain"])
        rows.append({
            "analysis": analysis,
            "task": task,
            "group": group,
            "k": k,
            "n_pool": n_pool,
            "n_draws": len(arrays["agg"]),
            "mean_individual_mean": mean_s["mean"],
            "mean_individual_median": mean_s["median"],
            "mean_individual_p025": mean_s["p025"],
            "mean_individual_p975": mean_s["p975"],
            "mean_individual_ci_lo": mean_s["ci_lo"],
            "mean_individual_ci_hi": mean_s["ci_hi"],
            "agg_mean": agg_s["mean"],
            "agg_median": agg_s["median"],
            "agg_p025": agg_s["p025"],
            "agg_p975": agg_s["p975"],
            "agg_ci_lo": agg_s["ci_lo"],
            "agg_ci_hi": agg_s["ci_hi"],
            "gain_mean": gain_s["mean"],
            "gain_median": gain_s["median"],
            "gain_p025": gain_s["p025"],
            "gain_p975": gain_s["p975"],
            "gain_ci_lo": gain_s["ci_lo"],
            "gain_ci_hi": gain_s["ci_hi"],
            "norm_gain_mean": norm_s["mean"],
            "norm_gain_median": norm_s["median"],
            "norm_gain_p025": norm_s["p025"],
            "norm_gain_p975": norm_s["p975"],
            "norm_gain_ci_lo": norm_s["ci_lo"],
            "norm_gain_ci_hi": norm_s["ci_hi"],
        })
    return rows


def _plot_series(
    ax,
    k_values: list[int],
    rows_by_k: dict[int, dict[str, object]],
    *,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    color: str,
    label: str,
):
    ks = [k for k in k_values if k in rows_by_k]
    if not ks:
        return
    means = [float(rows_by_k[k][mean_key]) for k in ks]
    los = [float(rows_by_k[k][lo_key]) for k in ks]
    his = [float(rows_by_k[k][hi_key]) for k in ks]
    ax.plot(ks, means, color=color, label=label)
    ax.fill_between(ks, los, his, color=color, alpha=0.18, linewidth=0)


def _fill_metric_2x2_axes(
    axes,
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    *,
    mean_key: str,
    lo_key: str,
    hi_key: str,
    ref_human_key: str | None,
    ref_genai_key: str | None,
    ylabel: str,
    panel_letters: tuple[str, str, str, str] = ("a", "b", "c", "d"),
):
    """Draw the 2×2 equal-size metric panels onto an existing axes grid."""
    panels = [
        (panel_letters[0], "Main Effects", "Race", "Racial Inequality — Main Effects"),
        (panel_letters[1], "Main Effects", "Gender", "Gender Inequality — Main Effects"),
        (panel_letters[2], "Interactions", "Race", "Racial Inequality — Interactions"),
        (panel_letters[3], "Interactions", "Gender", "Gender Inequality — Interactions"),
    ]
    draw_refs = ref_human_key is not None and ref_genai_key is not None
    flat_axes = np.asarray(axes).ravel()

    for ax, (letter, analysis, task, panel_title) in zip(flat_axes, panels):
        task_rows = [
            r for r in summary_rows
            if r["analysis"] == analysis and r["task"] == task
        ]
        human_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "Humans"}
        genai_by_k = {int(r["k"]): r for r in task_rows if r["group"] == "GenAI"}
        k_values = sorted(set(human_by_k) | set(genai_by_k))

        _plot_series(
            ax, k_values, human_by_k,
            mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
            color=COLOR_AGG_HUMAN, label="Humans",
        )
        _plot_series(
            ax, k_values, genai_by_k,
            mean_key=mean_key, lo_key=lo_key, hi_key=hi_key,
            color=GROUP_COLORS["genai"], label="GenAI",
        )
        if draw_refs:
            ref = full_refs[analysis][task]
            ax.axhline(ref[ref_human_key], color=COLOR_AGG_HUMAN, ls="--", lw=1.15, alpha=0.75)
            ax.axhline(ref[ref_genai_key], color=GROUP_COLORS["genai"], ls="--", lw=1.15, alpha=0.75)

        ax.grid(True, axis="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=22, length=6.5, width=1.3)
        if k_values:
            ax.set_xlim(k_values[0] - 0.35, k_values[-1] + 0.35)
            ax.set_xticks(_crowd_size_ticks(k_values))

        ax.set_title(
            f"{letter}.  {panel_title}",
            fontsize=22,
            fontweight="bold",
            pad=12,
            loc="left",
        )
        ax.set_ylabel(ylabel, fontsize=23)
        # sharex hides top-row labels; keep Crowd size on every panel for the paper figure.
        ax.set_xlabel("Crowd size", fontsize=23)
        ax.tick_params(axis="x", labelbottom=True)


def plot_main_text_accuracy_and_gain_combined(
    summary_rows: list[dict[str, object]],
    full_refs: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
):
    """Combined 2×4 figure: row 1 aggregated accuracy; row 2 aggregation gain."""
    fig, axes = plt.subplots(
        2, 4, figsize=(23.0, 11.2), sharex=False, sharey=False,
    )
    axes_acc = axes[0, :]
    axes_gain = axes[1, :]

    _fill_metric_2x2_axes(
        axes_acc,
        summary_rows,
        full_refs,
        mean_key="agg_mean",
        lo_key="agg_p025",
        hi_key="agg_p975",
        ref_human_key="avg_human",
        ref_genai_key="avg_genai",
        ylabel="Aggregated accuracy",
        panel_letters=("a", "b", "c", "d"),
    )
    _fill_metric_2x2_axes(
        axes_gain,
        summary_rows,
        full_refs,
        mean_key="gain_mean",
        lo_key="gain_p025",
        hi_key="gain_p975",
        ref_human_key=None,
        ref_genai_key=None,
        ylabel="Aggregation gain",
        panel_letters=("e", "f", "g", "h"),
    )

    # Keep y-axis label only on the leftmost panel of each row.
    for ax in list(axes_acc[1:]) + list(axes_gain[1:]):
        ax.set_ylabel("")

    legend_handles = [
        Line2D([0], [0], color=COLOR_AGG_HUMAN, lw=3.0, label="Humans"),
        Line2D([0], [0], color=GROUP_COLORS["genai"], lw=3.0, label="GenAI"),
        Line2D(
            [0], [0], color="0.35", lw=1.8, ls="--",
            label="Mean accuracy (same color)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=24,
        frameon=False,
        handlelength=3.4,
        columnspacing=2.4,
        handletextpad=0.9,
    )
    fig.subplots_adjust(
        left=0.06, right=0.995, top=0.86, bottom=0.09, hspace=0.52, wspace=0.30,
    )

    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    for fmt in ("svg", "png"):
        fig.savefig(
            Path(f"{stem}.{fmt}"),
            format=fmt,
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def run_one(
    analysis: str,
    pts: list[dict],
    ml_vec: np.ndarray,
    task: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    humans, genai = _split_pools(pts)
    full = compute_from_plot_pts(pts, ml_vec)
    # Sample each group up to its own pool size (Humans to n_H, GenAI to n_AI).
    human_ks = list(range(K_MIN, len(humans) + 1))
    genai_ks = list(range(K_MIN, len(genai) + 1))

    human_curves = resample_curve(humans, ml_vec, human_ks, B, rng)
    genai_curves = resample_curve(genai, ml_vec, genai_ks, B, rng)

    rows = []
    rows.extend(summarize_curves(analysis, task, "Humans", len(humans), human_curves))
    rows.extend(summarize_curves(analysis, task, "GenAI", len(genai), genai_curves))
    return rows, full


def _crowd_size_ticks(k_values: list[int]) -> list[int]:
    if not k_values:
        return []
    lo, hi = k_values[0], k_values[-1]
    span = hi - lo
    # Sparse ticks for the dense 2×4 combined figure.
    if span > 40:
        step = 20
    elif span > 20:
        step = 10
    else:
        step = 5
    # Evenly spaced labels; stop at the last grid point ≤ hi (e.g. 62 when hi=73).
    last_label = hi - ((hi - lo) % step)
    ticks = list(range(lo, last_label + 1, step))
    # Always show the true max crowd size when it is not already a grid point.
    if ticks and ticks[-1] != hi and (hi - ticks[-1]) >= max(3, step // 3):
        ticks.append(hi)
    return ticks


def main():
    csv_path = ROOT / "anonymous_survey_data.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows_csv = list(csv.reader(f))
    headers, data = rows_csv[0], rows_csv[1:]

    me_records, me_ml = load_main_effects_records(headers, data)
    soi_records, soi_ml = load_soi_records(headers, data)

    rng = np.random.default_rng(SEED)
    all_summary: list[dict[str, object]] = []
    full_refs: dict[str, dict[str, dict[str, float]]] = {
        "Main Effects": {},
        "Interactions": {},
    }

    for task in TASKS:
        task_key = "cos_race" if task == "Race" else "cos_gender"
        vec_key = "vec_race_bin" if task == "Race" else "vec_gender_bin"

        me_pts = plot_pts_main_effects(me_records, task_key, vec_key)
        me_rows, me_full = run_one("Main Effects", me_pts, me_ml[task], task, rng)
        all_summary.extend(me_rows)
        full_refs["Main Effects"][task] = me_full

        soi_pts = plot_pts_soi(soi_records, task_key, vec_key)
        soi_rows, soi_full = run_one("Interactions", soi_pts, soi_ml[task], task, rng)
        all_summary.extend(soi_rows)
        full_refs["Interactions"][task] = soi_full

    summary_csv = OUT_DIR / "fig_equal_size_aggregation_summary.csv"
    fieldnames = [
        "analysis", "task", "group", "k", "n_pool", "n_draws",
        "mean_individual_mean", "mean_individual_median",
        "mean_individual_p025", "mean_individual_p975",
        "mean_individual_ci_lo", "mean_individual_ci_hi",
        "agg_mean", "agg_median", "agg_p025", "agg_p975",
        "agg_ci_lo", "agg_ci_hi",
        "gain_mean", "gain_median", "gain_p025", "gain_p975",
        "gain_ci_lo", "gain_ci_hi",
        "norm_gain_mean", "norm_gain_median", "norm_gain_p025", "norm_gain_p975",
        "norm_gain_ci_lo", "norm_gain_ci_hi",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_summary:
            writer.writerow({
                k: (
                    f"{row[k]:.8f}" if isinstance(row[k], float) else row[k]
                )
                for k in fieldnames
            })

    combined_fig = OUT_DIR / "fig_equal_size_aggregation.png"
    plot_main_text_accuracy_and_gain_combined(all_summary, full_refs, combined_fig)

    for stale in (
        OUT_DIR / "equal_size_aggregation_accuracy_gain_combined.png",
        OUT_DIR / "equal_size_aggregation_accuracy_gain_combined.svg",
        OUT_DIR / "equal_size_aggregation_accuracy_gain_combined.pdf",
        OUT_DIR / "equal_size_aggregation_summary.csv",
        OUT_DIR / "aggregation_table_and_accuracy_combined.pdf",
        OUT_DIR / "aggregation_accuracy_and_table_combined.pdf",
        OUT_DIR / "aggregation_accuracy_and_table_combined.tex",
        OUT_DIR / "equal_size_aggregation_accuracy_panelA.pdf",
        OUT_DIR / "aggregation_gain_table_embed.tex",
        OUT_DIR / "aggregation_gain_table_embed_standalone.tex",
        OUT_DIR / "aggregation_gain_table_embed_standalone.pdf",
        OUT_DIR / "equal_size_aggregation_accuracy_2x2.pdf",
        OUT_DIR / "equal_size_aggregation_accuracy_2x2.svg",
        OUT_DIR / "equal_size_aggregation_gain_2x2.pdf",
        OUT_DIR / "equal_size_aggregation_gain_2x2.svg",
        OUT_DIR / "equal_size_aggregation_accuracy_2x2_human_subgroups.pdf",
        OUT_DIR / "equal_size_aggregation_accuracy_2x2_human_subgroups.svg",
        OUT_DIR / "equal_size_aggregation_summary_human_subgroups.csv",
        OUT_DIR / "equal_size_aggregation_curves.pdf",
        OUT_DIR / "equal_size_aggregation_curves.svg",
    ):
        if stale.is_file():
            stale.unlink()

    # Console checks
    print(f"Equal-sized aggregation (B={B}, seed={SEED})")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {combined_fig}\n")

    for analysis in ("Main Effects", "Interactions"):
        print(f"=== {analysis} ===")
        for task in TASKS:
            ref = full_refs[analysis][task]
            print(
                f"  Full crowds {task}: "
                f"n_H={int(ref['n_human'])} n_AI={int(ref['n_genai'])} | "
                f"agg H={ref['agg_human']:.4f} AI={ref['agg_genai']:.4f} | "
                f"gain H={ref['gain_human']:+.4f} AI={ref['gain_genai']:+.4f}"
            )
            k_star = int(ref["n_genai"])
            for group in ("Humans", "GenAI"):
                match = [
                    r for r in all_summary
                    if r["analysis"] == analysis
                    and r["task"] == task
                    and r["group"] == group
                    and int(r["k"]) == k_star
                ]
                if not match:
                    continue
                r = match[0]
                tag = "full-pool check" if group == "GenAI" else "matched-k"
                print(
                    f"    k={k_star} {group} ({tag}): "
                    f"agg={float(r['agg_mean']):.4f} "
                    f"gain={float(r['gain_mean']):+.4f} "
                    f"norm={float(r['norm_gain_mean']):+.4f}"
                )
                if group == "GenAI":
                    delta = abs(float(r["agg_mean"]) - ref["agg_genai"])
                    print(f"      |agg_mean − full GenAI agg| = {delta:.6f}")
            # Matched-k gain gap at k_star
            h = next(
                r for r in all_summary
                if r["analysis"] == analysis and r["task"] == task
                and r["group"] == "Humans" and int(r["k"]) == k_star
            )
            g = next(
                r for r in all_summary
                if r["analysis"] == analysis and r["task"] == task
                and r["group"] == "GenAI" and int(r["k"]) == k_star
            )
            print(
                f"    Matched-k gain gap (H−AI) at k={k_star}: "
                f"{float(h['gain_mean']) - float(g['gain_mean']):+.4f} "
                f"(norm: {float(h['norm_gain_mean']) - float(g['norm_gain_mean']):+.4f})"
            )
        print()


if __name__ == "__main__":
    main()
