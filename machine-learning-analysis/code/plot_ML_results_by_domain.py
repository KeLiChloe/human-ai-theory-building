"""
Redraw racial / gender ML-results figures from JSON/PKL (vector text).

Each figure:
  Row a: Main effects (RF | LR)
  Row b: Second-order interactions (RF | LR)
  Row c: RF out-of-sample classification, y-axis aligned with a1/b1
         and stretched to the right edge of a2/b2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from feature_sign_Logistic import PLOT_FONT, draw_coef_summary_on_ax
from feature_importance_RF import draw_feature_importance_on_ax
from ml_classification import load_rf_classification_results


def _load_coef_dataframe(json_path: Path) -> tuple[dict, pd.DataFrame]:
    with json_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    run_config = payload["run_config"]
    coef_df = pd.DataFrame(payload["ranking"]).rename(
        columns={
            "feature": "Feature",
            "mean_coef": "Mean",
            "lower95_ci": "Lower95CI",
            "upper95_ci": "Upper95CI",
            "mean_pval": "MeanPval",
            "stars": "Stars",
        }
    )
    return run_config, coef_df


def _load_ranking_dataframe(json_path: Path) -> tuple[dict, pd.DataFrame]:
    with json_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    run_config = payload["run_config"]
    ranking_df = pd.DataFrame(payload["ranking"]).rename(
        columns={
            "feature": "Feature",
            "votes": "Votes",
            "feature_importance": "Feature Importance",
        }
    )
    return run_config, ranking_df


def _domain_specs(base: Path) -> dict[str, dict]:
    return {
        "race": {
            "output_stem": "ML_results_racial",
            "jobs": [
                {
                    "rf_json": base / "race/main_effects/RF_feature_importance_results.json",
                    "lr_json": base / "race/main_effects/LR_sign_results.json",
                    "rf_title": "Random forest — main-effect votes",
                    "lr_title": "Logistic regression — main-effect coefficients",
                    "labels": ("a1", "a2"),
                    "rf_top_k": 5,
                },
                {
                    "rf_json": base / "race/soi/RF_feature_importance_results_soi.json",
                    "lr_json": base / "race/soi/LR_sign_results_soi.json",
                    "rf_title": "Random forest — interaction votes",
                    "lr_title": "Logistic regression — interaction coefficients",
                    "labels": ("b1", "b2"),
                    "rf_top_k": 3,
                },
            ],
            "classif": {
                "pkl": base / "race/RF_classification_results_seed300.pkl",
                "label": "c",
            },
        },
        "gender": {
            "output_stem": "ML_results_gender",
            "jobs": [
                {
                    "rf_json": base / "gender/main_effects/RF_feature_importance_results.json",
                    "lr_json": base / "gender/main_effects/LR_sign_results.json",
                    "rf_title": "Random forest — main-effect votes",
                    "lr_title": "Logistic regression — main-effect coefficients",
                    "labels": ("a1", "a2"),
                    "rf_top_k": 5,
                },
                {
                    "rf_json": base / "gender/soi/RF_feature_importance_results_soi.json",
                    "lr_json": base / "gender/soi/LR_sign_results_soi.json",
                    "rf_title": "Random forest — interaction votes",
                    "lr_title": "Logistic regression — interaction coefficients",
                    "labels": ("b1", "b2"),
                    "rf_top_k": 3,
                },
            ],
            "classif": {
                "pkl": base / "gender/RF_classification_results_seed102.pkl",
                "label": "c",
            },
        },
    }


def _load_rf_df(json_path: Path, top_k: int | None = 50):
    run_config, ranking_df = _load_ranking_dataframe(json_path)
    k = top_k if top_k is not None else run_config["plot_top_k"]
    combined = ranking_df.head(k).copy()
    combined = combined.sort_values(
        by=["Votes", "Feature Importance"], ascending=[True, True]
    )
    return run_config, combined


def _load_lr_df(json_path: Path):
    return _load_coef_dataframe(json_path)


def _add_panel_label(
    ax,
    code: str,
    title: str = "",
    fontsize: int = 150,
    align_to_yticks: bool = False,
    x_fig: float | None = None,
) -> None:
    """Panel code + short title. Left-align to y-ticks or a shared figure x."""
    code_dot = code if str(code).endswith(".") else f"{code}."
    text = f"{code_dot}  {title}".rstrip()
    fig = ax.figure
    if x_fig is not None:
        trans = blended_transform_factory(fig.transFigure, ax.transAxes)
        ax.text(
            x_fig,
            1.04,
            text,
            transform=trans,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            fontname=PLOT_FONT,
            clip_on=False,
        )
        return
    if align_to_yticks:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xs = [
            lab.get_window_extent(renderer=renderer).x0
            for lab in ax.get_yticklabels()
            if lab.get_visible() and str(lab.get_text()).strip()
        ]
        if xs:
            y_disp = ax.transAxes.transform((0.0, 1.0))[1]
            x_axes = ax.transAxes.inverted().transform((min(xs), y_disp))[0]
            ax.text(
                x_axes,
                1.04,
                text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                fontname=PLOT_FONT,
                clip_on=False,
            )
            return
    ax.text(
        0.0,
        1.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        fontname=PLOT_FONT,
        clip_on=False,
    )


def _column_title_x_fig(axes_list: list) -> float | None:
    """Shared figure-x at the leftmost y-tick ink in a column."""
    if not axes_list:
        return None
    fig = axes_list[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xs = []
    for ax in axes_list:
        for lab in ax.get_yticklabels():
            if not lab.get_visible() or not str(lab.get_text()).strip():
                continue
            xs.append(lab.get_window_extent(renderer=renderer).x0)
    if not xs:
        return None
    return float(fig.transFigure.inverted().transform((min(xs), 0.0))[0])


def _unify_ylabel_pad(axes_list: list, gap_pt: float = 14.0) -> None:
    """Match visual gap from y-label ink to the axis across panels."""
    if not axes_list:
        return
    fig = axes_list[0].figure
    dpi = float(fig.dpi)
    for ax in axes_list:
        ax.tick_params(axis="y", pad=gap_pt)
        for lab in ax.get_yticklabels():
            lab.set_ha("right")
            lab.set_va("center")
            lab.set_rotation_mode("anchor")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_gap_px = gap_pt * dpi / 72.0
    for ax in axes_list:
        spine_x = ax.transAxes.transform((0.0, 0.5))[0]
        rights = [
            lab.get_window_extent(renderer=renderer).x1
            for lab in ax.get_yticklabels()
            if lab.get_visible() and str(lab.get_text()).strip()
        ]
        if not rights:
            continue
        cur_gap_px = spine_x - max(rights)
        cur_pad = float(ax.yaxis.get_tick_padding())
        ax.tick_params(axis="y", pad=cur_pad + (target_gap_px - cur_gap_px) * 72.0 / dpi)
        for lab in ax.get_yticklabels():
            lab.set_ha("right")
            lab.set_va("center")
            lab.set_rotation_mode("anchor")
    fig.canvas.draw()


def _style_panel_frame(ax, color: str = "#000000", linewidth: float = 5.5) -> None:
    """Nature/PNAS-like frames: true black, thicker spines."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)
    ax.tick_params(axis="both", which="both", color=color, width=2.8)


# Panel c only. Klein blue, 大红, green, black — kept apart by hue and dash.
PANEL_C_LINE_WIDTH = 14.4
PANEL_C_FRAME_WIDTH = 9.5
PANEL_C_SERIES = (
    ("accuracy", "Accuracy", "#002FA7", "-"),
    ("precision", "Precision", "#E60012", (0, (9, 3.5))),
    ("recall", "Recall", "#009E49", (0, (8, 2.5, 2.2, 2.5))),
    ("f1", "F1 Score", "#000000", (0, (2.4, 2.4))),
)


def _draw_rf_classification_on_ax(ax, payload, show_legend: bool = True):
    """Threshold metrics; AUC in legend."""
    cfg = payload["run_config"]
    thresholds = payload["thresholds"]
    for key, label, color, linestyle in PANEL_C_SERIES:
        ax.plot(
            thresholds,
            payload[key],
            label=label,
            color=color,
            linewidth=PANEL_C_LINE_WIDTH,
            linestyle=linestyle,
        )
    ax.plot([], [], " ", label=f"AUC = {cfg['auc']:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(
        "Decision threshold",
        fontsize=168,
        fontname=PLOT_FONT,
        labelpad=52,
    )
    ax.set_ylabel("Metric Value", fontsize=168, fontname=PLOT_FONT, labelpad=52)
    if show_legend:
        ax.legend(
            prop={"family": PLOT_FONT, "size": 120},
            frameon=True,
            fancybox=False,
            edgecolor="#9a9a9a",
            facecolor="none",
            framealpha=1.0,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=2,
            columnspacing=1.0,
            handlelength=2.4,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.25,
            borderaxespad=0.2,
        )
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.tick_params(axis="both", labelsize=144, pad=14)
    # Keep x=0; drop y=0 so the two zeros do not collide at the origin.
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontname(PLOT_FONT)
    return ax


def _load_jobs(jobs: list[dict]) -> list[dict]:
    loaded = []
    for job in jobs:
        rf_cfg, rf_df = _load_rf_df(Path(job["rf_json"]), top_k=job.get("rf_top_k", 50))
        lr_cfg, lr_df = _load_lr_df(Path(job["lr_json"]))
        n_row = max(len(rf_df), min(len(lr_df), 10))
        loaded.append({
            **job,
            "rf_cfg": rf_cfg,
            "rf_df": rf_df,
            "lr_cfg": lr_cfg,
            "lr_df": lr_df,
            "n_row": n_row,
        })
    return loaded


def plot_domain_figure(
    output_stem: Path,
    jobs: list[dict],
    classif_job: dict,
    rf_width_ratio: float = 1.45,
    gap_ratio: float = 3.00,
    lr_width_ratio: float = 1.35,
) -> Path:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    loaded = _load_jobs(jobs)
    pkl_path = Path(classif_job["pkl"])
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Missing RF classification pkl: {pkl_path}\n"
            "Run ml_classification.py first."
        )
    classif_payload = load_rf_classification_results(pkl_path)

    height_ratios = [max(22, row["n_row"] * 4.20) for row in loaded]
    height_ratios.append(68)
    fig_w = 92
    fig_h = max(sum(height_ratios) * 0.82 + 2.2, 138)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    outer = GridSpec(
        nrows=len(loaded) + 1,
        ncols=1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.54,
        left=0.10,
        right=0.985,
        top=0.96,
        bottom=0.04,
    )

    rf_axes = []
    lr_axes = []
    cbar_axes = []

    for i, row in enumerate(loaded):
        is_soi = bool(row["rf_cfg"]["add_SOI"])
        # Same RF column as a1 so b1 matches a1 width; extra b1–b2 gap comes from b2.
        row_rf = rf_width_ratio
        if is_soi:
            # Keep a bit more RF–LR gap than row a, but not so wide that b2 shrinks.
            row_gap = 3.10
            row_lr = (rf_width_ratio + gap_ratio + lr_width_ratio) - row_rf - row_gap
        else:
            row_gap = gap_ratio
            row_lr = lr_width_ratio
        row_wspace = 0.22
        panels = outer[i].subgridspec(
            1,
            3,
            width_ratios=[row_rf, row_gap, row_lr],
            wspace=row_wspace,
        )
        # First-row colorbar sits a little farther from the bars than row b.
        rf_cbar_wspace = 0.20 if i == 0 else 0.10
        rf_block = panels[0, 0].subgridspec(
            1, 2, width_ratios=[1.0, 0.048], wspace=rf_cbar_wspace
        )
        ax_rf = fig.add_subplot(rf_block[0, 0])
        cax_rf = fig.add_subplot(rf_block[0, 1])
        ax_lr = fig.add_subplot(panels[0, 2])
        rf_axes.append(ax_rf)
        lr_axes.append(ax_lr)
        cbar_axes.append(cax_rf)

        draw_feature_importance_on_ax(
            ax_rf,
            fig,
            row["rf_df"],
            row["rf_cfg"]["importance_type"],
            add_SOI=is_soi,
            show_title=False,
            task_label=row["rf_cfg"].get("task", ""),
            cax=cax_rf,
            xlim_pad_frac=0.32,
            xlim_pad_min=200.0,
            x_max=1000,
            y_tick_size=150,
            y_tick_rotation=25,
            x_tick_size=72 if is_soi else 64,
            x_label_size=88 if is_soi else 80,
            colorbar_tick_size=100 if is_soi else 92,
            colorbar_label_size=110 if is_soi else 100,
            show_colorbar_ticks=True,
            colorbar_nbins=2,
            wrap_interactions=is_soi,
            bar_label_size=110 if is_soi else 102,
        )
        draw_coef_summary_on_ax(
            ax_lr,
            row["lr_df"],
            row["lr_cfg"]["split_n"],
            task_label=row["lr_cfg"].get("task", ""),
            top_n=10,
            show_title=False,
            show_legend=False,
            y_tick_size=130 if is_soi else 140,
            y_tick_rotation=25,
            x_tick_size=92,
            x_label_size=100,
            coef_label_size=100 if is_soi else 90,
            legend_size=26,
            legend_title_size=28,
            pos_color="#e53935",
            neg_color="#1e88e5",
            bar_alpha=0.62,
            x_nbins=4,
        )

    _unify_ylabel_pad(rf_axes, gap_pt=16)
    _unify_ylabel_pad(lr_axes, gap_pt=14)
    rf_title_x = _column_title_x_fig(rf_axes)
    lr_title_x = _column_title_x_fig(lr_axes)
    for ax_rf, ax_lr, row in zip(rf_axes, lr_axes, loaded):
        left_lab, right_lab = row["labels"]
        _add_panel_label(
            ax_rf, left_lab, row["rf_title"], x_fig=rf_title_x
        )
        _add_panel_label(
            ax_lr, right_lab, row["lr_title"], x_fig=lr_title_x
        )

    ax_c_placeholder = fig.add_subplot(outer[len(loaded)])
    ax_c_placeholder.set_axis_off()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    slot_pos = ax_c_placeholder.get_position()
    ax_c_placeholder.remove()

    # Center c under the visual a/b block (axes + y-tick labels + colorbars),
    # not just the bare axes boxes (labels pull the perceived center leftward).
    inv = fig.transFigure.inverted()
    x0s, x1s = [], []
    for ax in (*rf_axes, *lr_axes, *cbar_axes):
        bbox = ax.get_window_extent(renderer=renderer)
        x0s.append(bbox.x0)
        x1s.append(bbox.x1)
        for lab in ax.get_yticklabels():
            if not lab.get_visible() or not str(lab.get_text()).strip():
                continue
            lb = lab.get_window_extent(renderer=renderer)
            x0s.append(lb.x0)
            x1s.append(lb.x1)
    span_left = float(inv.transform((min(x0s), 0.0))[0])
    span_right = float(inv.transform((max(x1s), 0.0))[0])
    span_width = span_right - span_left

    # Hang c from the top of its GridSpec row so b–c equals a–b (same hspace).
    c_width = span_width
    c_height = c_width * fig.get_figwidth() / fig.get_figheight()
    max_h = slot_pos.height * 0.98
    if c_height > max_h:
        c_height = max_h
        c_width = c_height * fig.get_figheight() / fig.get_figwidth()
    scale = 0.90
    c_width *= scale
    c_height *= scale
    c_left = span_left + 0.5 * (span_width - c_width)
    c_bottom = slot_pos.y1 - c_height

    ax_c = fig.add_axes([c_left, c_bottom, c_width, c_height])
    _draw_rf_classification_on_ax(ax_c, classif_payload, show_legend=True)
    _add_panel_label(
        ax_c,
        classif_job["label"],
        "Random forest — out-of-sample classification performance",
    )
    for ax in (*rf_axes, *lr_axes, ax_c):
        _style_panel_frame(ax, linewidth=PANEL_C_FRAME_WIDTH)
    for ax in cbar_axes:
        _style_panel_frame(ax)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.35)
    # Cap PNG raster size so viewers don't OOM (figure is ~90×140 in).
    fig_w_in, fig_h_in = fig.get_size_inches()
    png_dpi = min(150.0, 4500.0 / max(fig_w_in, fig_h_in))
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print("✅ Wrote:")
    print(f"   → {svg_path}")
    print(f"   → {png_path} (dpi={png_dpi:.1f})")
    return svg_path


def main():
    parser = argparse.ArgumentParser(
        description="Redraw racial/gender ML-results figures from JSON/PKL."
    )
    parser.add_argument(
        "--domain",
        choices=["race", "gender", "both"],
        default="both",
    )
    args = parser.parse_args()

    # Anchor to package root so cwd does not matter (any clone / any machine).
    root = Path(__file__).resolve().parent.parent
    base = root / "models"
    outdir = root / "figures"
    specs = _domain_specs(base)
    domains = ["race", "gender"] if args.domain == "both" else [args.domain]

    for key in domains:
        spec = specs[key]
        plot_domain_figure(
            outdir / spec["output_stem"],
            jobs=spec["jobs"],
            classif_job=spec["classif"],
        )


if __name__ == "__main__":
    main()
