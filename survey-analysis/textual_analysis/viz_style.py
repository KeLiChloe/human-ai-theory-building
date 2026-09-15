"""Shared matplotlib style for textual_analysis visualization scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from matplotlib.offsetbox import AnchoredOffsetbox, AnchoredText, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.transforms import Bbox, blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.viz_config import COLOR_AGG_HUMAN, GROUP_COLORS

SIG_TEXT_COLOR = "#C62828"
FOOTNOTE_COLOR = "#444444"
BOX_EDGE_NEUTRAL = "#555555"
PHASE_HATCH_COLOR = "#9e9e9e"

GROUP_COLORS_TEXT = {
    "PhD Students": GROUP_COLORS["phd"],
    "Senior Scientists": GROUP_COLORS["senior"],
    "Topic Experts": GROUP_COLORS["topic"],
    "Non Topic Experts": GROUP_COLORS["non_topic"],
    "GenAI": GROUP_COLORS["genai"],
}
GROUP_COLORS_COLLAPSED = {
    "Human": COLOR_AGG_HUMAN,
    "Topic Experts": GROUP_COLORS["topic"],
    "GenAI": GROUP_COLORS["genai"],
}

GROUP_ORDER = ["PhD Students", "Senior Scientists", "GenAI"]
GROUP_ORDER_COLLAPSED = ["Human", "GenAI"]

DISPLAY_LABELS = {
    "PhD Students": "PhD Students",
    "Senior Scientists": "Senior Scientists",
    "Topic Experts": "Topic Experts",
    "Non Topic Experts": "Non-Experts",
    "GenAI": "GenAI",
    "Human": "Humans",
}
HUMAN_DISPLAY_LABEL = DISPLAY_LABELS["Human"]
HUMAN_COMPOSITION_NOTE = "PhD Students + Senior Scientists"

SAVE_DPI = 600
SAVE_PAD_INCHES = 0.08
FONT_TITLE = 20
TITLE_PAD = 22
FONT_AXIS_LABEL = 13
FONT_TICK = 11.5
FONT_LEGEND = 13
FONT_COMPARISON = 11
FONT_FOOTNOTE = 10
SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.98
SUBPLOT_TOP = 0.87

FOOTNOTE_Y = 0.038
SIG_LEVEL_LEGEND = (
    "NS (p ≥ 0.05), * (p < 0.05), ** (p < 0.01), *** (p < 0.001)"
)
METRIC_SUBTITLE_Y = 0.935
METRIC_SUBTITLE_FONTSIZE = 19
METRIC_SUBTITLE_LINE_STEP = 0.028
FIGURE_SUPTITLE_Y = 0.99
FIGURE_TITLE_METRIC_GAP = 0.014
FIGURE_METRIC_LEGEND_GAP = 0.030
FIGURE_LEGEND_PANEL_GAP = 0.060

# Unified typography for theory_space_structure visualization figures.
VIZ_SUPTITLE_FONTSIZE = 26
VIZ_SUPTITLE_LINE_SPACING = 1.35
VIZ_PANEL_TITLE_FONTSIZE = 19
VIZ_AXIS_LABEL_FONTSIZE = 23
VIZ_TICK_FONTSIZE = 18
VIZ_FOOTNOTE_FONTSIZE = 17
VIZ_FOOTNOTE_LINE_STEP = 0.017
VIZ_FOOTNOTE_Y = 0.058
VIZ_LEGEND_FONTSIZE = 18
VIZ_LEGEND_Y_SHIFT = 0.012
VIZ_BRACKET_FONTSIZE = 18
VIZ_SUPYLABEL_X = 0.02
VIZ_HEADER_VERTICAL_SHIFT = 0.028
# Legacy defaults; prefer layout_title_and_metric() for per-figure placement.
FIGURE_WITH_METRIC_PANEL_TOP = 0.78
FIGURE_WITH_METRIC_LEGEND_Y = 0.875
WELCH_TWO_SIDED_THREE_GROUP_FOOTNOTE = (
    "Two-sided Welch t-test on pairwise group mean differences "
    "(PhD Students vs Senior Scientists, PhD Students vs GenAI, "
    "Senior Scientists vs GenAI).",
    SIG_LEVEL_LEGEND,
)
WELCH_TWO_SIDED_HUMAN_GENAI_FOOTNOTE = (
    "Two-sided Welch t-test on mean difference (Humans vs GenAI).",
    SIG_LEVEL_LEGEND,
)
WELCH_TWO_SIDED_PAIRWISE_FOOTNOTE = (
    "Two-sided Welch t-test on pairwise group mean differences.",
    SIG_LEVEL_LEGEND,
)
SIG_FOOTNOTE = WELCH_TWO_SIDED_PAIRWISE_FOOTNOTE
ASSESSMENT_SIG_FOOTNOTE = (
    "Two-sided Welch t-test on mean quality score (between groups, per phase).",
    "Two-sided paired t-test on mean quality score (within-group Pre vs Post).",
    SIG_LEVEL_LEGEND,
)
FOOTNOTE_LINE_STEP = 0.014
PAIRED_BRACKET_LIFT = 0.035
PAIRED_BRACKET_HEIGHT = 0.045
PAIRED_BRACKET_LABEL_GAP = 0.014
PAIRWISE_BRACKET_TIER_STEP = 0.058
# Below x-tick labels (axes y < 0, blended with data x).
PAIRED_BRACKET_BELOW_AXIS_ARM = -0.20
PAIRED_BRACKET_BELOW_AXIS_LINE = -0.28
PAIRED_BRACKET_BELOW_AXIS_LABEL = -0.36

COMPARISON_BOX_BOTTOM = 0.085
COMPARISON_AXIS_GAP = 0.065
COMPARISON_LINE_STEP = 0.028
PHASE_COMPARE_LINE_STEP = COMPARISON_LINE_STEP
COMPARISON_BOX_PAD = 0.010
FIGURE_COMPARE_PAD_PX = 8
PHASE_BOX_SIDE_MARGIN = 0.08
BOX_GAP = 0.08
BOX_WIDTH_ONE_LINE = 0.34
BOX_WIDTH_THREE_LINES = 0.35
BOX_WIDTH_CENTERED = 0.62
BOX_STYLE_PAD = 0.008

BAR_ALPHA = 0.95
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.7
ERROR_CAPSIZE = 3
ERROR_LINEWIDTH = 1.0

COMPARE_PAD_PX = 12
COMPARE_LINE_STEP_METRIC = 0.045

# Nature figure fonts: sans-serif, preferably Helvetica or Arial (same across all
# figures). Greek letters: use unicode in Helvetica/Arial, or Symbol when
# hand-editing in Illustrator (matplotlib mathtext stays on Helvetica).
RC_PARAMS = {
    "figure.dpi": 180,
    "savefig.dpi": SAVE_DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:oblique",
    "mathtext.bf": "Helvetica:bold",
    "mathtext.sf": "Helvetica",
    "mathtext.default": "regular",
    "font.size": 10.5,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": FONT_LEGEND,
    "legend.frameon": False,
    "grid.alpha": 0.2,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_plot_style() -> None:
    """Apply shared Nature-oriented figure style (Helvetica / Arial)."""
    plt.rcParams.update(RC_PARAMS)


def set_figure_title(ax, title: str, *, pad: float | None = None) -> None:
    ax.set_title(
        title,
        fontweight="bold",
        fontsize=FONT_TITLE,
        pad=TITLE_PAD if pad is None else pad,
    )


def display_label(group_key: str) -> str:
    return DISPLAY_LABELS.get(group_key, group_key)


def comparison_pair_label(left_key: str, right_key: str) -> str:
    return f"{display_label(left_key)} vs. {display_label(right_key)}"


def legend_entry(group_key: str, n: int, *, include_composition: bool = False) -> str:
    label = display_label(group_key)
    if include_composition and group_key == "Human":
        return f"{label} (n={n}, {HUMAN_COMPOSITION_NOTE})"
    return f"{label} (n={n})"


def collapsed_legend_labels(groups: list[str], ns: dict[str, int]) -> list[str]:
    return [
        legend_entry(g, ns[g], include_composition=(g == "Human"))
        for g in groups
    ]


def add_legend(ax, handles: list, labels: list[str], *, loc: str = "upper left"):
    return ax.legend(
        handles,
        labels,
        loc=loc,
        frameon=False,
        fontsize=FONT_LEGEND,
    )


def _apply_bold_xticklabels(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")


def style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    if getattr(ax, "_viz_bold_xticks", False):
        _apply_bold_xticklabels(ax)


def set_axis_labels(
    ax,
    xlabel: str | None,
    ylabel: str,
    *,
    xlabel_pad: int = 10,
    bold_xticks: bool = False,
) -> None:
    ax._viz_bold_xticks = bold_xticks
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if bold_xticks:
        _apply_bold_xticklabels(ax)


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=SAVE_DPI, pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def save_figure_pdf_svg(
    fig,
    out_path: Path | str,
    *,
    pad_inches: float | None = None,
    bbox_inches: str | None = None,
    close: bool = True,
) -> list[Path]:
    """Save SVG and PNG using ``out_path`` stem (suffix ignored)."""
    stem = Path(out_path).with_suffix("")
    pad = SAVE_PAD_INCHES if pad_inches is None else pad_inches
    kwargs: dict = {"pad_inches": pad, "dpi": SAVE_DPI}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    saved: list[Path] = []
    for fmt in ("svg", "png"):
        path = Path(f"{stem}.{fmt}")
        fig.savefig(path, format=fmt, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def format_p_value_label(p: float, *, stars: bool = False) -> str:
    """Compact p-value text for figure brackets (italic P).

    If ``stars`` is True and p < 0.05, append a second line such as ``(***)``.
    """
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        text = r"$\mathit{P}$ < 0.001"
    else:
        text = f"$\\mathit{{P}}$ = {p:.3f}"
    if stars and is_significant(p):
        text = f"{text}\n({significance_label(p)})"
    return text


def format_p_value_with_trailing_stars(p: float) -> str:
    """Italic capital P with trailing stars, e.g. ``P < 0.001***``."""
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return r"$\mathit{P}$ < 0.001***"
    stars = significance_label(p)
    if stars == "NS":
        stars = ""
    return f"$\\mathit{{P}}$ = {p:.3f}{stars}"


def italic_p_textareas(label: str, *, fontsize: float, color: str) -> list:
    """Offsetbox pieces: italic P plus upright remainder (legend rows)."""
    text = (
        str(label)
        .strip()
        .replace(r"$\mathit{P}$", "P")
        .replace("$P$", "P")
    )
    props = {"size": fontsize, "color": color}
    if text.startswith("P"):
        return [
            TextArea(" ", textprops=props),
            TextArea("P", textprops={**props, "style": "italic"}),
            TextArea(text[1:], textprops=props),
        ]
    return [TextArea(f" {label}", textprops=props)]

def significance_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"


def is_significant(p: float) -> bool:
    return np.isfinite(p) and p < 0.05


def sig_label_is_significant(sig: str) -> bool:
    """True for *, **, ***; false for NS / n/a."""
    return str(sig).strip() not in {"NS", "n/a", "NA", ""}


def _axes_text_width(
    ax,
    text: str,
    *,
    fontsize: float,
    fontweight: str = "normal",
) -> float:
    """Approximate text width in axes coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    tmp = ax.text(
        0.0,
        0.0,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
    )
    bb = tmp.get_window_extent(renderer=renderer)
    tmp.remove()
    inv = ax.transAxes.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y0))
    return abs(x1 - x0)


def draw_pre_post_sig_columns(
    ax,
    columns: list[tuple[float, str, list[tuple[str, str]]]]
    | list[tuple[str, list[tuple[str, str]]]],
    *,
    y0: float = -0.14,
    fontsize: float,
    linespacing: float = 1.65,
    framed: bool = True,
    col_gap: float = 0.055,
) -> None:
    """
    Bottom Pre/Post notes: bold headers; significant rows in red.

    Columns are packed tightly and the whole framed panel is centered under the axes.
    ``columns`` may be ``(header, rows)`` or legacy ``(x, header, rows)`` (x ignored).
    """
    normalized: list[tuple[str, list[tuple[str, str]]]] = []
    for item in columns:
        if len(item) == 3 and isinstance(item[0], (int, float)):
            normalized.append((str(item[1]), list(item[2])))
        else:
            normalized.append((str(item[0]), list(item[1])))

    if not normalized:
        return

    bbox = ax.get_position()
    ax_h_in = max(bbox.height * float(ax.figure.get_figheight()), 1e-6)
    line_dy = (fontsize * linespacing) / 72.0 / ax_h_in
    n_rows = max(len(rows) for _, rows in normalized)
    n_lines = 1 + n_rows

    # Measure content widths, then pack columns and center the panel.
    col_widths: list[float] = []
    for header, rows in normalized:
        candidates = [header] + [f"{sig:<3}  {lab}" for sig, lab in rows]
        weights = ["bold"] + ["normal"] * (len(candidates) - 1)
        w = max(
            _axes_text_width(ax, t, fontsize=fontsize, fontweight=wt)
            for t, wt in zip(candidates, weights)
        )
        col_widths.append(w)

    content_w = sum(col_widths) + col_gap * max(len(normalized) - 1, 0)
    pad_x = 0.028
    pad_y = 0.55 * line_dy
    box_w = content_w + 2 * pad_x
    box_left = 0.5 - box_w / 2
    box_right = box_left + box_w
    box_top = y0 + pad_y
    box_bottom = y0 - n_lines * line_dy - pad_y

    xs: list[float] = []
    x_cursor = box_left + pad_x
    for w in col_widths:
        xs.append(x_cursor)
        x_cursor += w + col_gap

    if framed:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (box_left, box_bottom),
                box_right - box_left,
                box_top - box_bottom,
                boxstyle="round,pad=0.012",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=PHASE_HATCH_COLOR,
                linewidth=0.8,
                alpha=1.0,
                clip_on=False,
                zorder=1,
            )
        )

    for x, (header, rows) in zip(xs, normalized):
        ax.text(
            x,
            y0,
            header,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color="#333333",
            clip_on=False,
            zorder=2,
        )
        for i, (sig, lab) in enumerate(rows):
            color = SIG_TEXT_COLOR if sig_label_is_significant(sig) else "#555555"
            ax.text(
                x,
                y0 - (i + 1) * line_dy,
                f"{sig:<3}  {lab}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=2,
            )


def format_comparison_line(label: str, p: float) -> str:
    return f"{label}: {significance_label(p)}"


def format_paired_pre_post_line(p: float) -> str:
    return significance_label(p)


def comparison_box_height(n_lines: int) -> float:
    """Tight height estimate for bottom margin (matches bbox-hug phase boxes)."""
    return 0.026 + COMPARISON_LINE_STEP * max(n_lines - 1, 0) + 0.016


def comparison_box_width(n_lines: int, *, centered: bool = False) -> float:
    if centered:
        return BOX_WIDTH_CENTERED
    return BOX_WIDTH_ONE_LINE if n_lines == 1 else BOX_WIDTH_THREE_LINES


def phase_center_x(box_width: float) -> dict[str, float]:
    gap = BOX_GAP
    span = 2 * box_width + gap
    max_span = 1.0 - 2 * PHASE_BOX_SIDE_MARGIN
    if span > max_span:
        gap = max(0.06, max_span - 2 * box_width)
        span = 2 * box_width + gap
    left_edge = (1.0 - span) / 2.0
    left_edge = max(left_edge, PHASE_BOX_SIDE_MARGIN)
    if left_edge + span > 1.0 - PHASE_BOX_SIDE_MARGIN:
        left_edge = 1.0 - PHASE_BOX_SIDE_MARGIN - span
    return {
        "Pre-ML": left_edge + box_width / 2,
        "Post-ML": left_edge + box_width + gap + box_width / 2,
    }


def apply_bottom_layout(
    fig,
    n_lines: int,
    *,
    box_bottom: float = COMPARISON_BOX_BOTTOM,
    axis_gap: float = COMPARISON_AXIS_GAP,
) -> None:
    box_height = comparison_box_height(n_lines)
    comparison_top = box_bottom + box_height
    bottom_margin = min(0.90, comparison_top + axis_gap)
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=bottom_margin,
    )


def draw_metric_subtitle(
    fig,
    lines: str | tuple[str, ...] | list[str],
    *,
    y: float = METRIC_SUBTITLE_Y,
    fontsize: float = METRIC_SUBTITLE_FONTSIZE,
    line_step: float = METRIC_SUBTITLE_LINE_STEP,
) -> int:
    """Place metric definition lines directly below the figure suptitle."""
    content = (lines,) if isinstance(lines, str) else tuple(lines)
    for i, line in enumerate(content):
        fig.text(
            0.5,
            y - i * line_step,
            line,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight="normal",
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
        )
    return len(content)


@dataclass(frozen=True)
class TitleBlockLayout:
    legend_y: float
    panel_top: float


def _text_block_height_frac(
    fontsize: float,
    n_lines: int,
    fig_height_in: float,
    *,
    line_spacing: float = 1.0,
) -> float:
    return (fontsize / 72.0 / fig_height_in) * n_lines * line_spacing


def layout_title_and_metric(
    fig,
    *,
    suptitle: str,
    metric_lines: str | tuple[str, ...] | list[str],
    suptitle_fontsize: float,
    suptitle_y: float = FIGURE_SUPTITLE_Y,
    suptitle_line_spacing: float = 1.0,
    metric_fontsize: float = METRIC_SUBTITLE_FONTSIZE,
    metric_line_step: float = METRIC_SUBTITLE_LINE_STEP,
    gap_title_metric: float = FIGURE_TITLE_METRIC_GAP,
    gap_metric_legend: float = FIGURE_METRIC_LEGEND_GAP,
    gap_legend_panels: float = FIGURE_LEGEND_PANEL_GAP,
    vertical_shift: float = 0.0,
    metric_vertical_shift: float = 0.0,
) -> TitleBlockLayout:
    """Place suptitle (top-aligned) and metric lines below it without overlap."""
    metric_content = (metric_lines,) if isinstance(metric_lines, str) else tuple(metric_lines)
    n_metric = len(metric_content)
    n_title = suptitle.count("\n") + 1
    fig_h = float(fig.get_size_inches()[1])

    title_h = _text_block_height_frac(
        suptitle_fontsize,
        n_title,
        fig_h,
        line_spacing=suptitle_line_spacing,
    )
    if n_metric == 0:
        metric_h = 0.0
    else:
        metric_h = _text_block_height_frac(metric_fontsize, 1, fig_h)
        metric_h += max(n_metric - 1, 0) * metric_line_step

    title = fig.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=suptitle_fontsize,
        y=suptitle_y,
        va="top",
    )
    title.set_linespacing(suptitle_line_spacing)

    metric_y = (
        suptitle_y - title_h - gap_title_metric + vertical_shift + metric_vertical_shift
    )
    if n_metric:
        draw_metric_subtitle(
            fig,
            metric_content,
            y=metric_y,
            fontsize=metric_fontsize,
            line_step=metric_line_step,
        )

    legend_y = metric_y - metric_h - gap_metric_legend
    panel_top = legend_y - gap_legend_panels
    return TitleBlockLayout(legend_y=legend_y, panel_top=panel_top)


def figure_legend_panel_top(
    legend_anchor_y: float,
    *,
    n_items: int,
    ncol: int,
    fig_height_in: float,
    legend_fontsize: float = VIZ_LEGEND_FONTSIZE,
    gap_below_legend: float = 0.032,
    row_spacing: float = 1.55,
) -> float:
    """Figure ``top`` for subplots_adjust below a multi-row figure legend."""
    n_rows = (n_items + max(ncol, 1) - 1) // max(ncol, 1)
    row_h = (legend_fontsize / 72.0 / fig_height_in) * row_spacing
    return legend_anchor_y - n_rows * row_h - gap_below_legend


def draw_sig_footnote(
    fig,
    y: float = FOOTNOTE_Y,
    *,
    text: str | tuple[str, ...] | list[str] | None = None,
    line_step: float = FOOTNOTE_LINE_STEP,
) -> None:
    content = SIG_FOOTNOTE if text is None else text
    lines = [content] if isinstance(content, str) else list(content)
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            y - i * line_step,
            line,
            ha="center",
            va="bottom",
            fontsize=FONT_FOOTNOTE,
            color=FOOTNOTE_COLOR,
            transform=fig.transFigure,
            clip_on=False,
            zorder=1,
        )


def draw_pairwise_sig_legend(
    ax,
    columns: list[tuple[float, str, list[tuple[str, str]]]]
    | list[tuple[str, list[tuple[str, str]]]],
    *,
    loc: str = "upper right",
    fontsize: float = 6.5,
) -> None:
    """In-axes framed legend for pairwise Pre/Post significance (no below-axis box)."""
    normalized = _normalize_sig_columns(columns)
    if not normalized:
        return

    lines: list[str] = []
    for i, (header, rows) in enumerate(normalized):
        if i > 0:
            lines.append("")
        lines.append(str(header))
        for sig, lab in rows:
            lines.append(f"{sig:<3}  {lab}")

    box = AnchoredText(
        "\n".join(lines),
        loc=loc,
        prop={"size": fontsize, "color": "#333333"},
        frameon=True,
        borderpad=0.35,
        pad=0.25,
    )
    box.patch.set_boxstyle("round,pad=0.25")
    box.patch.set_facecolor("white")
    box.patch.set_edgecolor(PHASE_HATCH_COLOR)
    box.patch.set_linewidth(0.8)
    box.patch.set_alpha(1.0)
    ax.add_artist(box)

NON_TOPIC_SWATCH_COLOR = "#111111"


def draw_pairwise_sig_color_legend(
    ax,
    columns: list[tuple[str, list[tuple[str | float, tuple[str, str]]]]],
    *,
    group_colors: dict[str, str],
    loc: str = "upper right",
    fontsize: float = 6.5,
    swatch_size: float | None = None,
    layout: str = "columns",
    column_sep: float = 16,
    label_pvalues: bool = False,
    sig_text_color: str | None = None,
) -> None:
    """In-axes legend: colored swatches vs swatches + stars or p-values."""
    if not columns:
        return

    sw = sh = swatch_size if swatch_size is not None else max(5.5, fontsize * 0.95)
    highlight = SIG_TEXT_COLOR if sig_text_color is None else sig_text_color

    def _swatch(color: str) -> DrawingArea:
        da = DrawingArea(sw, sh)
        da.add_artist(
            mpatches.Rectangle(
                (0, 0),
                sw,
                sh,
                facecolor=color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.5,
                alpha=BAR_ALPHA,
            )
        )
        return da

    def _group_marker(group: str):
        if group == "Non Topic Experts":
            return _swatch(NON_TOPIC_SWATCH_COLOR)
        return _swatch(group_colors[group])

    def _row_label(sig_or_p: str | float) -> tuple[str, bool]:
        if label_pvalues and isinstance(sig_or_p, (int, float)):
            p = float(sig_or_p)
            return format_p_value_with_trailing_stars(p), is_significant(p)
        sig = str(sig_or_p)
        return sig, sig_label_is_significant(sig)

    def _comparison_row(sig_or_p: str | float, left: str, right: str) -> HPacker:
        label, sig_flag = _row_label(sig_or_p)
        if sig_text_color is not None:
            text_color = highlight
        else:
            text_color = highlight if sig_flag else "#555555"
        return HPacker(
            children=[
                _group_marker(left),
                TextArea(" vs ", textprops={"size": fontsize, "color": "#666666"}),
                _group_marker(right),
                *italic_p_textareas(label, fontsize=fontsize, color=text_color),
            ],
            align="center",
            pad=0,
            sep=1,
        )

    def _phase_column(
        header: str, pairs: list[tuple[str | float, tuple[str, str]]]
    ) -> VPacker:
        col_rows: list = [
            TextArea(header, textprops={"size": fontsize, "weight": "bold", "color": "#333333"})
        ]
        for sig_or_p, (left, right) in pairs:
            col_rows.append(_comparison_row(sig_or_p, left, right))
        return VPacker(children=col_rows, align="left", pad=0, sep=2)

    uses_non_topic = any(
        left == "Non Topic Experts" or right == "Non Topic Experts"
        for _header, pairs in columns
        for _sig, (left, right) in pairs
    )

    if layout == "columns":
        body = HPacker(
            children=[_phase_column(header, pairs) for header, pairs in columns],
            align="top",
            pad=0,
            sep=column_sep,
        )
    else:
        rows: list = []
        for i, (header, pairs) in enumerate(columns):
            if i > 0:
                rows.append(TextArea("", textprops={"size": fontsize * 0.35}))
            rows.append(
                TextArea(
                    header, textprops={"size": fontsize, "weight": "bold", "color": "#333333"}
                )
            )
            for sig, (left, right) in pairs:
                rows.append(_comparison_row(sig, left, right))
        body = VPacker(children=rows, align="left", pad=0, sep=2)

    if uses_non_topic:
        footnote = HPacker(
            children=[
                _swatch(NON_TOPIC_SWATCH_COLOR),
                TextArea(
                    f" = {display_label('Non Topic Experts')}",
                    textprops={"size": max(fontsize - 0.5, 5.5), "color": "#555555"},
                ),
            ],
            align="center",
            pad=0,
            sep=2,
        )
        child = VPacker(children=[body, footnote], align="left", pad=0, sep=2)
    else:
        child = body

    anchored = AnchoredOffsetbox(
        loc=loc,
        child=child,
        pad=0.22,
        borderpad=0.22,
        frameon=True,
        bbox_to_anchor=(1.0, 1.02),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set_boxstyle("round,pad=0.12")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor(PHASE_HATCH_COLOR)
    anchored.patch.set_linewidth(0.8)
    anchored.patch.set_alpha(1.0)
    ax.add_artist(anchored)


def draw_paired_pre_post_bracket(
    ax,
    x_pre: float,
    x_post: float,
    y_base: float,
    p: float,
    *,
    fontsize: float | None = None,
    placement: str = "above",
    below_arm: float | None = None,
    below_line: float | None = None,
    below_label: float | None = None,
    label: str | None = None,
    color: str | None = None,
    fontweight: str | None = None,
) -> None:
    """Bracket between Pre/Post bars for within-group paired comparison."""
    if fontsize is None:
        fontsize = FONT_COMPARISON
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    sig = is_significant(p)
    if color is None:
        color = SIG_TEXT_COLOR if sig else "black"
    if fontweight is None:
        weight = "bold" if sig else "normal"
    else:
        weight = fontweight
    if label is None:
        label = format_paired_pre_post_line(p)

    if placement == "below":
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        y_arm = (
            PAIRED_BRACKET_BELOW_AXIS_ARM
            if below_arm is None
            else below_arm
        )
        y_line = (
            PAIRED_BRACKET_BELOW_AXIS_LINE
            if below_line is None
            else below_line
        )
        label_y = (
            PAIRED_BRACKET_BELOW_AXIS_LABEL
            if below_label is None
            else below_label
        )
        ax.plot(
            [x_pre, x_pre, x_post, x_post],
            [y_arm, y_line, y_line, y_arm],
            transform=trans,
            color=color,
            linewidth=1.1,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            (x_pre + x_post) / 2,
            label_y,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            clip_on=False,
            zorder=7,
        )
        return

    y_bar = y_base + span * PAIRED_BRACKET_LIFT
    y_tip = y_bar + span * PAIRED_BRACKET_HEIGHT
    label_y = y_tip + span * PAIRED_BRACKET_LABEL_GAP
    ax.plot(
        [x_pre, x_pre, x_post, x_post],
        [y_bar, y_tip, y_tip, y_bar],
        color=color,
        linewidth=1.1,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        (x_pre + x_post) / 2,
        label_y,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.18,
        clip_on=False,
        zorder=7,
    )



draw_pre_post_bracket = draw_paired_pre_post_bracket

def draw_pairwise_group_brackets(
    ax,
    x,
    group_order: list[str] | tuple[str, ...],
    pairwise: tuple[tuple[str, str], ...],
    phase_order: tuple[str, ...],
    phase_offsets: dict[str, float],
    phase_comp_sigs: dict[str, list[str]],
    phase_bar_tops: dict[str, float],
    *,
    fontsize: float,
    significant_only: bool = True,
) -> None:
    """Bracket annotations for between-group Welch comparisons (per phase)."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    tier_step = span * PAIRWISE_BRACKET_TIER_STEP
    groups = list(group_order)

    for phase in phase_order:
        y_base = float(phase_bar_tops.get(phase, 0.0))
        tier = 0
        for (left, right), sig in zip(pairwise, phase_comp_sigs[phase]):
            if significant_only and not sig_label_is_significant(sig):
                continue
            if left not in groups or right not in groups:
                continue
            x_left = float(x[groups.index(left)] + phase_offsets[phase])
            x_right = float(x[groups.index(right)] + phase_offsets[phase])
            color = SIG_TEXT_COLOR if sig_label_is_significant(sig) else "#555555"
            weight = "bold" if sig_label_is_significant(sig) else "normal"
            y_bar = y_base + span * PAIRED_BRACKET_LIFT + tier * tier_step
            y_tip = y_bar + span * PAIRED_BRACKET_HEIGHT
            label_y = y_tip + span * PAIRED_BRACKET_LABEL_GAP
            ax.plot(
                [x_left, x_left, x_right, x_right],
                [y_bar, y_tip, y_tip, y_bar],
                color=color,
                linewidth=1.1,
                clip_on=False,
                zorder=6,
            )
            ax.text(
                (x_left + x_right) / 2,
                label_y,
                sig,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight=weight,
                color=color,
                clip_on=False,
                zorder=7,
            )
            tier += 1


def _add_comparison_box(
    fig,
    box_left: float,
    box_bottom: float,
    box_width: float,
    box_height: float,
    has_sig: bool,
) -> None:
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (box_left, box_bottom),
            box_width,
            box_height,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            clip_on=False,
            linewidth=1.2 if has_sig else 0.9,
            zorder=2,
        )
    )


def _figure_text_width(fig, text: str, *, fontsize: float) -> float:
    """Return text width in figure coordinates."""
    renderer = fig.canvas.get_renderer()
    tmp = fig.text(0.5, 0.5, text, fontsize=fontsize, transform=fig.transFigure)
    bb = tmp.get_window_extent(renderer=renderer)
    tmp.remove()
    inv = fig.transFigure.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y0))
    return abs(x1 - x0)


def _auto_box_width(
    fig,
    lines: list[tuple[str, float]],
    *,
    min_width: float,
    max_width: float | None = None,
) -> float:
    if not lines:
        return min_width
    texts = [format_comparison_line(label, p) for label, p in lines]
    text_w = max(_figure_text_width(fig, t, fontsize=FONT_COMPARISON) for t in texts)
    pad = 2 * COMPARISON_BOX_PAD + 0.02
    cap = 1.0 - 2 * PHASE_BOX_SIDE_MARGIN if max_width is None else max_width
    return min(max(text_w + pad, min_width), cap)


def draw_centered_comparison_box(
    fig,
    comparisons: list[tuple[str, float]],
    *,
    center_x: float = 0.5,
    box_bottom: float = COMPARISON_BOX_BOTTOM,
    min_box_width: float | None = None,
    max_box_width: float | None = None,
) -> None:
    n = len(comparisons)
    min_width = (
        comparison_box_width(n, centered=True)
        if min_box_width is None
        else min_box_width
    )
    box_width = _auto_box_width(
        fig, comparisons, min_width=min_width, max_width=max_box_width
    )
    box_height = comparison_box_height(n)
    box_top = box_bottom + box_height
    box_left = center_x - box_width / 2
    has_sig = any(is_significant(p) for _, p in comparisons)

    _add_comparison_box(
        fig, box_left, box_bottom, box_width, box_height, has_sig
    )

    line_ys = [
        box_top - COMPARISON_BOX_PAD - i * COMPARISON_LINE_STEP for i in range(n)
    ]
    for (label, pval), y in zip(comparisons, line_ys):
        sig = is_significant(pval)
        fig.text(
            center_x,
            y,
            format_comparison_line(label, pval),
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=FONT_COMPARISON,
            fontweight="bold" if sig else "normal",
            color=SIG_TEXT_COLOR if sig else "black",
            clip_on=False,
            zorder=3,
        )


def snug_comparison_box_width(
    fig,
    lines: list[tuple[str, float]],
    *,
    max_width: float | None = None,
) -> float:
    """Text-fitted comparison box width (no default minimum)."""
    return _auto_box_width(fig, lines, min_width=0.0, max_width=max_width)


def shared_phase_box_width(
    fig,
    phase_lines: list[list[tuple[str, float]]],
    n_lines: int,
) -> float:
    fig.canvas.draw()
    min_width = comparison_box_width(n_lines)
    return max(
        _auto_box_width(fig, lines, min_width=min_width) for lines in phase_lines
    )


def _draw_snug_figure_patch(fig, text_objs: list, has_sig: bool) -> None:
    """Wrap figure-coordinate text objects in a tight rounded box (diagram style)."""
    if not text_objs:
        return
    renderer = fig.canvas.get_renderer()
    bb = Bbox.union([t.get_window_extent(renderer=renderer) for t in text_objs])
    bb = Bbox.from_extents(
        bb.x0 - FIGURE_COMPARE_PAD_PX,
        bb.y0 - FIGURE_COMPARE_PAD_PX,
        bb.x1 + FIGURE_COMPARE_PAD_PX,
        bb.y1 + FIGURE_COMPARE_PAD_PX,
    )
    inv = fig.transFigure.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            clip_on=False,
            linewidth=1.2 if has_sig else 0.9,
            zorder=3,
        )
    )


def draw_phase_comparison_box(
    fig,
    phase: str,
    center_x: float,
    lines: list[tuple[str, float]],
    *,
    box_width: float | None = None,
) -> None:
    del box_width  # spacing handled by caller; box size hugs text
    n = len(lines)
    has_sig = any(is_significant(p) for _, p in lines)
    n_with_header = n + 1
    y_top = COMPARISON_BOX_BOTTOM + comparison_box_height(n_with_header) - 0.006
    text_objs = []
    text_objs.append(
        fig.text(
            center_x,
            y_top,
            phase,
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=FONT_COMPARISON,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=4,
        )
    )
    for i, (label, pval) in enumerate(lines):
        sig = is_significant(pval)
        text_objs.append(
            fig.text(
                center_x,
                y_top - (i + 1) * PHASE_COMPARE_LINE_STEP,
                format_comparison_line(label, pval),
                transform=fig.transFigure,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )
    fig.canvas.draw()
    _draw_snug_figure_patch(fig, text_objs, has_sig)


def draw_snug_footer_comparison_box(
    fig,
    lines: list[tuple[str, float]],
    *,
    center_x: float = 0.5,
) -> None:
    """Single footer box with snug bbox (diagram / assessment style, no phase header)."""
    n = len(lines)
    has_sig = any(is_significant(p) for _, p in lines)
    y_top = COMPARISON_BOX_BOTTOM + comparison_box_height(n) - 0.006
    text_objs = []
    for i, (label, pval) in enumerate(lines):
        sig = is_significant(pval)
        text_objs.append(
            fig.text(
                center_x,
                y_top - i * PHASE_COMPARE_LINE_STEP,
                format_comparison_line(label, pval),
                transform=fig.transFigure,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )
    fig.canvas.draw()
    _draw_snug_figure_patch(fig, text_objs, has_sig)


def _place_metric_comparison_texts(
    ax,
    x_idx: int,
    comparison_lines: list[str],
    p_values: list[float],
    y: float,
    line_step: float,
) -> tuple[list, bool]:
    trans = ax.get_xaxis_transform()
    has_sig = any(is_significant(p) for p in p_values)
    text_objs = []

    for i, (line, p) in enumerate(zip(comparison_lines, p_values)):
        sig = is_significant(p)
        text_objs.append(
            ax.text(
                x_idx,
                y - i * line_step,
                line,
                transform=trans,
                ha="center",
                va="top",
                fontsize=FONT_COMPARISON,
                fontweight="bold" if sig else "normal",
                color=SIG_TEXT_COLOR if sig else "black",
                clip_on=False,
                zorder=4,
            )
        )

    return text_objs, has_sig


def _draw_metric_patch(ax, text_objs: list, has_sig: bool) -> None:
    if not text_objs:
        return

    trans = ax.get_xaxis_transform()
    renderer = ax.figure.canvas.get_renderer()
    bb = Bbox.union([t.get_window_extent(renderer=renderer) for t in text_objs])
    bb = Bbox.from_extents(
        bb.x0 - COMPARE_PAD_PX,
        bb.y0 - COMPARE_PAD_PX,
        bb.x1 + COMPARE_PAD_PX,
        bb.y1 + COMPARE_PAD_PX,
    )

    inv = trans.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, y1 = inv.transform((bb.x1, bb.y1))

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad={BOX_STYLE_PAD}",
            transform=trans,
            facecolor="white",
            edgecolor=SIG_TEXT_COLOR if has_sig else BOX_EDGE_NEUTRAL,
            alpha=0.96,
            linewidth=1.2 if has_sig else 0.9,
            clip_on=False,
            zorder=3,
        )
    )


def draw_metric_comparison_boxes(
    ax,
    comparisons: list[tuple[int, list[str], list[float], float, float]],
) -> None:
    placed: list[tuple[list, bool]] = []
    for x_idx, lines, p_values, y, line_step in comparisons:
        placed.append(
            _place_metric_comparison_texts(ax, x_idx, lines, p_values, y, line_step)
        )

    ax.figure.canvas.draw()

    for text_objs, has_sig in placed:
        _draw_metric_patch(ax, text_objs, has_sig)


def finalize_metric_figure(
    fig,
    ax,
    out_path: Path,
    *,
    layout: dict,
    comparisons: list[tuple[int, list[str], list[float], float, float]],
    footnote_y: float = FOOTNOTE_Y,
    footnote_text: str | tuple[str, ...] | None = None,
) -> None:
    fig.subplots_adjust(**layout)
    draw_metric_comparison_boxes(ax, comparisons)
    draw_sig_footnote(fig, y=footnote_y, text=footnote_text)
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}")
