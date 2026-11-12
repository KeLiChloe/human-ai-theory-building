"""Theory-level deepest mechanism: pre-ML vs post-ML.

Each theory (participant × task) is one equal-weight unit.
Among addressed pairs (post: also dedupe unordered feature pairs), take the
deepest mechanism:

  curvature (terminal_curvature | mixed_terminal)
    > upstream (nonterminal_upstream_interaction)
    > restatement (restatement_only | unclear)

Theories with no addressed pairs (all not_addressed) are coded none.
Main comparisons keep only theories with a non-none depth in BOTH pre and post;
excluded counts are reported in the summary CSV and stdout.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TEXTUAL = HERE.parent
if str(TEXTUAL) not in sys.path:
    sys.path.insert(0, str(TEXTUAL))

from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    apply_plot_style,
    save_figure_pdf_svg,
    set_figure_title,
)

PRE_PATH = HERE / "moderation_coding_pre_ML.json"
POST_PATH = HERE / "moderation_coding_post_ML.json"
OUT_PATH = HERE / "outputs" / "fig_theory_depth_pre_post.png"
SUMMARY_PATH = HERE / "outputs" / "fig_theory_depth_pre_post_summary.csv"
UNIT_PATH = HERE / "outputs" / "fig_theory_depth_pre_post_units.csv"

POST_PAIR_ORDER = (
    "USER_SELECTED_rank_1",
    "USER_SELECTED_rank_2",
    "USER_SELECTED_rank_3",
    "ML_rank_1",
    "ML_rank_2",
    "ML_rank_3",
)

DEPTH = {
    "terminal_curvature": 2,
    "mixed_terminal": 2,
    "nonterminal_upstream_interaction": 1,
    "restatement_only": 0,
    "unclear": 0,
}
DEPTH_LABEL = {
    2: "Terminates in curvature",
    1: "Unexplained upstream",
    0: "No moderation mechanism",
}
# Stack bottom -> top: curvature, upstream, restatement (legend same order).
DEPTH_ORDER = (2, 1, 0)
# Light fills so dark in-bar labels stay readable (incl. hatched post bars).
DEPTH_COLORS = {
    2: "#C4A1D6",  # soft purple — curvature
    1: "#F2C14E",  # soft gold — upstream
    0: "#C5CED6",  # soft slate — restatement
}
LEGEND_DEPTH_ORDER = DEPTH_ORDER
GROUP_ORDER = ("PhD Students", "Senior Scientists", "GenAI")
GROUP_CODE = {"0": "PhD Students", "1": "Senior Scientists", "2": "GenAI"}
PHASE_ORDER = ("pre", "post")
PHASE_LABEL = {"pre": "Pre-ML", "post": "Post-ML"}
PHASE_HATCH = {"pre": "", "post": "///"}
ANNOTATE_MIN_PCT = 8.0


def feature_key(pair: dict) -> tuple[str, str]:
    features = pair["features"]
    return tuple(sorted(features))  # type: ignore[return-value]


def addressed_labels_pre(pairs: list[dict]) -> list[str]:
    labels = []
    for pair in pairs:
        label = pair["primary_classification"]
        if label == "not_addressed":
            continue
        if label not in DEPTH:
            raise ValueError(f"Unknown classification: {label!r}")
        labels.append(label)
    return labels


def addressed_labels_post(pairs: list[dict]) -> list[str]:
    by_id = {pair["pair_id"]: pair for pair in pairs}
    missing = [pair_id for pair_id in POST_PAIR_ORDER if pair_id not in by_id]
    if missing:
        raise ValueError(f"Missing post pair_id(s): {missing}")
    seen: set[tuple[str, str]] = set()
    labels = []
    for pair_id in POST_PAIR_ORDER:
        pair = by_id[pair_id]
        label = pair["primary_classification"]
        if label == "not_addressed":
            continue
        if label not in DEPTH:
            raise ValueError(f"Unknown classification: {label!r}")
        key = feature_key(pair)
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def deepest_depth(labels: list[str]) -> int | None:
    if not labels:
        return None
    return max(DEPTH[label] for label in labels)


def load_units() -> list[dict[str, object]]:
    pre_records = {
        (rec["participant_id"], rec["task"]): rec
        for rec in json.loads(PRE_PATH.read_text(encoding="utf-8"))["records"]
        if rec.get("coding") is not None
    }
    post_records = {
        (rec["participant_id"], rec["task"]): rec
        for rec in json.loads(POST_PATH.read_text(encoding="utf-8"))["records"]
        if rec.get("coding") is not None
    }
    if set(pre_records) != set(post_records):
        raise ValueError("pre/post theory keys do not match")

    units = []
    for key in sorted(pre_records):
        pre = pre_records[key]
        post = post_records[key]
        group = GROUP_CODE[pre["group_code"]]
        if GROUP_CODE[post["group_code"]] != group:
            raise ValueError(f"{key} has conflicting groups")
        pre_depth = deepest_depth(addressed_labels_pre(pre["coding"]["pairs"]))
        post_depth = deepest_depth(addressed_labels_post(post["coding"]["pairs"]))
        if pre_depth is None or post_depth is None:
            change = "excluded_none"
        elif post_depth > pre_depth:
            change = "deeper"
        elif post_depth < pre_depth:
            change = "shallower"
        else:
            change = "same"
        units.append({
            "participant_id": key[0],
            "task": key[1],
            "group": group,
            "pre_depth": pre_depth,
            "post_depth": post_depth,
            "pre_label": None if pre_depth is None else DEPTH_LABEL[pre_depth],
            "post_label": None if post_depth is None else DEPTH_LABEL[post_depth],
            "change": change,
            "in_main": pre_depth is not None and post_depth is not None,
        })
    return units


def write_units(units: list[dict[str, object]]) -> None:
    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UNIT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(units[0]))
        writer.writeheader()
        writer.writerows(units)


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_units(units: list[dict[str, object]]) -> list[dict[str, object]]:
    return [unit for unit in units if unit["in_main"]]


def exclusion_report(units: list[dict[str, object]]) -> dict[str, int]:
    pre_none = sum(unit["pre_depth"] is None for unit in units)
    post_none = sum(unit["post_depth"] is None for unit in units)
    either = sum(unit["pre_depth"] is None or unit["post_depth"] is None for unit in units)
    both = sum(unit["pre_depth"] is None and unit["post_depth"] is None for unit in units)
    return {
        "n_theories": len(units),
        "n_main": len(main_units(units)),
        "n_excluded": either,
        "n_pre_none": pre_none,
        "n_post_none": post_none,
        "n_both_none": both,
    }


def _share(sub: list[dict[str, object]], phase: str, depth: int) -> tuple[float, int]:
    n = len(sub)
    count = sum(unit[f"{phase}_depth"] == depth for unit in sub)
    pct = 100.0 * count / n if n else float("nan")
    return pct, count


def plot(units: list[dict[str, object]]) -> list[dict[str, object]]:
    apply_plot_style()
    main = main_units(units)
    report = exclusion_report(units)
    table_rows: list[dict[str, object]] = []

    for key, value in report.items():
        table_rows.append({
            "panel": "exclusion",
            "group": "ALL",
            "contrast": key,
            "percentage": "",
            "count": value,
            "n_theories": report["n_main"],
        })

    fig, ax = plt.subplots(figsize=(11.4, 8.8))
    # Extra space between groups for pre/post pairs.
    x = np.array([0.0, 1.15, 2.3])
    width = 0.38
    phase_offset = {"pre": -width / 2 - 0.02, "post": width / 2 + 0.02}

    # Draw stacked bars; keep bottoms for in-bar labels.
    bottoms = {phase: np.zeros(len(GROUP_ORDER)) for phase in PHASE_ORDER}
    segment_meta: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for depth in DEPTH_ORDER:
        for phase in PHASE_ORDER:
            vals = []
            for group in GROUP_ORDER:
                sub = [unit for unit in main if unit["group"] == group]
                pct, count = _share(sub, phase, depth)
                vals.append(pct)
                table_rows.append({
                    "panel": "depth_composition",
                    "group": group,
                    "contrast": f"{phase} | {DEPTH_LABEL[depth]}",
                    "percentage": round(pct, 2),
                    "count": count,
                    "n_theories": len(sub),
                })
            vals_arr = np.array(vals, dtype=float)
            ax.bar(
                x + phase_offset[phase],
                vals_arr,
                width=width,
                bottom=bottoms[phase],
                color=DEPTH_COLORS[depth],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch=PHASE_HATCH[phase],
                zorder=2,
            )
            segment_meta.append((phase, depth, bottoms[phase].copy(), vals_arr))
            bottoms[phase] = bottoms[phase] + vals_arr

    # Percentage labels on sufficiently tall segments.
    for phase, depth, bottom, vals in segment_meta:
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if not np.isfinite(v) or v < ANNOTATE_MIN_PCT:
                continue
            ax.text(
                x[i] + phase_offset[phase],
                b + v / 2,
                f"{v:.0f}%",
                ha="center",
                va="center",
                fontsize=13,
                color="#1A1A1A",
                fontweight="semibold",
                zorder=3,
            )

    # Pre / Post labels under each bar pair.
    for i, group in enumerate(GROUP_ORDER):
        n = sum(unit["group"] == group for unit in main)
        ax.text(
            x[i] + phase_offset["pre"],
            -4.5,
            "Pre",
            ha="center",
            va="top",
            fontsize=14,
            color="#444444",
            clip_on=False,
        )
        ax.text(
            x[i] + phase_offset["post"],
            -4.5,
            "Post",
            ha="center",
            va="top",
            fontsize=14,
            color="#444444",
            clip_on=False,
        )
        ax.text(
            x[i],
            -11.5,
            group,
            ha="center",
            va="top",
            fontsize=15,
            color="#111111",
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            x[i],
            -16.8,
            f"(n = {n} theories)",
            ha="center",
            va="top",
            fontsize=13,
            color="#444444",
            clip_on=False,
        )

    ax.set_xticks([])
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage of theories", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_figure_title(ax, "Deepest mechanism per theory", pad=16)

    mechanism_handles = [
        mpatches.Patch(
            facecolor=DEPTH_COLORS[depth],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            alpha=BAR_ALPHA,
            label=DEPTH_LABEL[depth],
        )
        for depth in LEGEND_DEPTH_ORDER
    ]
    phase_handles = [
        mpatches.Patch(
            facecolor="#D0D0D0",
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=PHASE_HATCH[phase],
            label=PHASE_LABEL[phase],
        )
        for phase in PHASE_ORDER
    ]
    leg1 = fig.legend(
        handles=mechanism_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        fontsize=13,
        title="Category",
        title_fontsize=13,
        borderaxespad=0.0,
        borderpad=0.6,
        labelspacing=0.35,
        columnspacing=1.2,
    )
    fig.add_artist(leg1)
    leg2 = fig.legend(
        handles=phase_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.81),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        fontsize=13,
        title="Phase",
        title_fontsize=13,
        borderaxespad=0.0,
        borderpad=0.6,
        labelspacing=0.35,
        columnspacing=1.4,
    )
    _ = leg2
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.76))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_summary(table_rows)
    save_figure_pdf_svg(fig, OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    return table_rows


def main() -> None:
    units = load_units()
    write_units(units)
    report = exclusion_report(units)
    plot(units)
    main = main_units(units)
    print(
        f"wrote {OUT_PATH}\n"
        f"theories={report['n_theories']}, main={report['n_main']}, "
        f"excluded={report['n_excluded']} "
        f"(pre-none={report['n_pre_none']}, post-none={report['n_post_none']}, both={report['n_both_none']})"
    )
    print("main deepest pre:", Counter(unit["pre_label"] for unit in main))
    print("main deepest post:", Counter(unit["post_label"] for unit in main))
    print("paired change:", Counter(unit["change"] for unit in main))
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {UNIT_PATH}")


if __name__ == "__main__":
    main()
