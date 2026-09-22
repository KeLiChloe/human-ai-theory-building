"""
2×2 sorted cosine-accuracy panels:
Main Effects / Interactions × Race / Gender.

Output: forecasts/outputs/fig_sorted_individuals.{svg,png} (Humans pooled).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FORECASTS = Path(__file__).resolve().parent.parent
ROOT = FORECASTS.parent
LIB = Path(__file__).resolve().parent
TEXTUAL_DIR = ROOT / "textual_analysis"
for p in (ROOT, TEXTUAL_DIR, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.viz_config import COLOR_AGG_HUMAN, GROUP_COLORS  # noqa: E402
from viz_style import apply_plot_style  # noqa: E402

import me_quant as me  # noqa: E402
import soi_quant as soi  # noqa: E402

OUT_PATH = FORECASTS / "outputs" / "fig_sorted_individuals.png"
COLOR_RANDOM_BENCH = "#4A4A4A"
COLOR_ML_BENCH = "#000000"
# Theoretical E[cosine] under random ±1 sparse selection is exactly 0.
NULL_CHANCE_BASELINE = 0.0
NULL_CHANCE_LEGEND = "Random Baseline = 0"


def _draw_panel(
    ax,
    *,
    records: list[dict],
    task_key: str,
    vec_key: str,
    ml_vec: np.ndarray,
    random_bench: float,
    panel_title: str,
    letter: str,
    agg_fn,
    legend_fn,
    legend_line_fn,
    format_value,
    human_ids,
    legend_fontsize: float = 11,
) -> None:
    pts = [
        {"score": r[task_key], "group": r["group"], "vec": r[vec_key]}
        for r in records
        if not np.isnan(r[task_key]) and r.get(vec_key) is not None
    ]
    pts.sort(key=lambda x: x["score"])

    n_pt = len(pts)
    total_n = len(records)
    x = np.linspace(1.0, float(total_n), n_pt) if n_pt else np.array([])
    y = np.array([p["score"] for p in pts])
    is_human = np.array([p["group"] in human_ids for p in pts], dtype=bool)
    is_senior = np.array([p["group"] == "1" for p in pts], dtype=bool)
    is_phd = np.array([p["group"] == "0" for p in pts], dtype=bool)
    is_gen = np.array([p["group"] == "2" for p in pts], dtype=bool)

    color_genai = GROUP_COLORS["genai"]
    edge, lw = "#555555", 0.45
    ax.scatter(
        x[is_human], y[is_human],
        c=COLOR_AGG_HUMAN, marker="o", s=28,
        alpha=1.0, edgecolors=edge, linewidths=lw, zorder=5,
    )
    ax.scatter(
        x[is_gen], y[is_gen],
        c=color_genai, marker="o", s=28,
        alpha=1.0, edgecolors=edge, linewidths=lw, zorder=5,
    )

    aggregations = agg_fn(pts, ml_vec)
    group_means = {
        "human": float(np.mean(y[is_human])) if np.any(is_human) else np.nan,
        "senior": float(np.mean(y[is_senior])) if np.any(is_senior) else np.nan,
        "phd": float(np.mean(y[is_phd])) if np.any(is_phd) else np.nan,
        "genai": float(np.mean(y[is_gen])) if np.any(is_gen) else np.nan,
    }

    for y_val, color, ls in (
        (aggregations["human"], COLOR_AGG_HUMAN, "-"),
        (aggregations["genai"], color_genai, "-"),
        (random_bench, COLOR_RANDOM_BENCH, "--"),
        (1.0, COLOR_ML_BENCH, ":"),
    ):
        if not np.isnan(y_val):
            ax.axhline(y_val, color=color, linestyle=ls, linewidth=1.25, alpha=0.95, zorder=1)

    extra_legend = [
        legend_line_fn(
            COLOR_RANDOM_BENCH, "--",
            NULL_CHANCE_LEGEND,
        ),
        legend_line_fn(
            COLOR_ML_BENCH, ":",
            "ML = 1.000",
            linewidth=1.35,
        ),
    ]
    lg = legend_fn(
        ax, aggregations, extra_legend,
        collapsed=True, group_means=group_means,
    )
    # Rebuild with tighter vertical spacing; put ML just above Aggregated Humans.
    handles = list(lg.legend_handles)
    labels = [t.get_text() for t in lg.get_texts()]
    lg.remove()
    ml_i = next((i for i, t in enumerate(labels) if t.startswith("ML")), None)
    agg_i = next((i for i, t in enumerate(labels) if t.startswith("Aggregated Humans")), None)
    if ml_i is not None and agg_i is not None and ml_i != agg_i:
        ml_h, ml_t = handles.pop(ml_i), labels.pop(ml_i)
        if ml_i < agg_i:
            agg_i -= 1
        handles.insert(agg_i, ml_h)
        labels.insert(agg_i, ml_t)
    lg = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        frameon=True,
        fontsize=legend_fontsize,
        labelspacing=0.28,
        borderpad=0.35,
        handletextpad=0.45,
        handlelength=1.55,
        borderaxespad=0.4,
    )
    lg.get_frame().set_edgecolor("#666666")
    lg.get_frame().set_linewidth(0.7)

    ax.set_title(
        f"{letter}.  {panel_title}",
        fontsize=15,
        fontweight="bold",
        pad=8,
        loc="left",
    )
    ax.set_xlim(0, total_n + 1)
    ax.set_ylim(-1.0, 1.0)
    step = max(1, (total_n + 7) // 8)
    ticks = list(range(0, total_n + 1, step))
    if ticks[-1] != total_n:
        ticks.append(total_n)
    ax.set_xticks(ticks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.2)


def plot_sorted_individuals(out_path: Path = OUT_PATH) -> Path:
    # me/soi modules set Times; re-apply Nature-style Helvetica/Arial for this figure.
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), sharey=True)
    panels = [
        (
            axes[0, 0], "a",
            me.records, "cos_race", "vec_race_bin", me.ml_bin_race,
            NULL_CHANCE_BASELINE,
            "Racial Inequality — Main Effects",
            me._cosine_aggregation_scores,
            me._build_sorted_figure_legend,
            me._legend_line,
            me.format_legend_value,
            me.HUMAN_GROUP_IDS,
        ),
        (
            axes[0, 1], "b",
            me.records, "cos_gender", "vec_gender_bin", me.ml_bin_gender,
            NULL_CHANCE_BASELINE,
            "Gender Inequality — Main Effects",
            me._cosine_aggregation_scores,
            me._build_sorted_figure_legend,
            me._legend_line,
            me.format_legend_value,
            me.HUMAN_GROUP_IDS,
        ),
        (
            axes[1, 0], "c",
            soi.records, "cos_race", "vec_race_bin", soi.ml_bin_r,
            NULL_CHANCE_BASELINE,
            "Racial Inequality — Interactions",
            soi._cosine_aggregation_scores,
            soi._build_sorted_figure_legend,
            soi._legend_line,
            soi.format_legend_value,
            soi.HUMAN_GROUP_IDS,
        ),
        (
            axes[1, 1], "d",
            soi.records, "cos_gender", "vec_gender_bin", soi.ml_bin_g,
            NULL_CHANCE_BASELINE,
            "Gender Inequality — Interactions",
            soi._cosine_aggregation_scores,
            soi._build_sorted_figure_legend,
            soi._legend_line,
            soi.format_legend_value,
            soi.HUMAN_GROUP_IDS,
        ),
    ]
    for args in panels:
        _draw_panel(
            args[0],
            records=args[2],
            task_key=args[3],
            vec_key=args[4],
            ml_vec=args[5],
            random_bench=args[6],
            panel_title=args[7],
            letter=args[1],
            agg_fn=args[8],
            legend_fn=args[9],
            legend_line_fn=args[10],
            format_value=args[11],
            human_ids=args[12],
        )

    for ax in axes[1, :]:
        ax.set_xlabel(
            "Contributor rank (sorted low to high by cosine similarity)",
            fontsize=13.5,
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Cosine similarity", fontsize=13.5)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    for fmt in ("svg", "png"):
        path = Path(f"{stem}.{fmt}")
        fig.savefig(path, format=fmt, dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    path = plot_sorted_individuals()
    print(f"Figure saved → {path}")


if __name__ == "__main__":
    main()
