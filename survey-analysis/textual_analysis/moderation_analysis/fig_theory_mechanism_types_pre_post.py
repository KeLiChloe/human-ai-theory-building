"""Pair-level mechanism types: pre-ML vs post-ML.

Unit = (theory, interaction) pair = participant × task × addressed feature pair.
Post-ML also dedupes unordered feature pairs.

Each unit gets one mutually exclusive label from primary_classification:

  restatement_only  <- restatement_only | unclear
  upstream_only     <- nonterminal_upstream_interaction
  curvature_only    <- terminal_curvature
  both              <- mixed_terminal
      (both means this single pair's explanation has upstream and terminates
       in curvature — not co-occurrence across different pairs)

not_addressed pairs are excluded.
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
OUT_JOINT = HERE / "outputs" / "fig_theory_mechanism_joint_type_pre_post.png"
SUMMARY_JOINT = HERE / "outputs" / "fig_theory_mechanism_joint_type_pre_post_summary.csv"
UNIT_PATH = HERE / "outputs" / "fig_theory_mechanism_types_pre_post_units.csv"

# Legacy presence outputs (no longer generated; remove if present).
OUT_PRESENCE = HERE / "outputs" / "fig_theory_mechanism_presence_pre_post.png"
SUMMARY_PRESENCE = HERE / "outputs" / "fig_theory_mechanism_presence_pre_post_summary.csv"

POST_PAIR_ORDER = (
    "USER_SELECTED_rank_1",
    "USER_SELECTED_rank_2",
    "USER_SELECTED_rank_3",
    "ML_rank_1",
    "ML_rank_2",
    "ML_rank_3",
)

JOINT_FROM_PRIMARY = {
    "restatement_only": "restatement_only",
    "unclear": "restatement_only",
    "nonterminal_upstream_interaction": "upstream_only",
    "terminal_curvature": "curvature_only",
    "mixed_terminal": "both",
}

JOINT_ORDER = ("restatement_only", "upstream_only", "curvature_only", "both")
JOINT_LABEL = {
    "restatement_only": "No moderation mechanism",
    "upstream_only": "Upstream only",
    "curvature_only": "Curvature only",
    "both": "Both curvature and upstream",
}
JOINT_COLORS = {
    "restatement_only": "#C5CED6",
    "upstream_only": "#F2C14E",
    "curvature_only": "#A8D5A2",
    "both": "#C4A1D6",
}

GROUP_ORDER = ("PhD Students", "Senior Scientists", "GenAI")
GROUP_CODE = {"0": "PhD Students", "1": "Senior Scientists", "2": "GenAI"}
PHASE_ORDER = ("pre", "post")
PHASE_LABEL = {"pre": "Pre-ML", "post": "Post-ML"}
PHASE_HATCH = {"pre": "", "post": "///"}
ANNOTATE_MIN_PCT = 8.0


def feature_key(pair: dict) -> tuple[str, str]:
    return tuple(sorted(pair["features"]))  # type: ignore[return-value]


def pair_joint(primary: str) -> str:
    if primary not in JOINT_FROM_PRIMARY:
        raise ValueError(f"Unknown classification: {primary!r}")
    return JOINT_FROM_PRIMARY[primary]


def iter_pre_pairs(record: dict) -> list[dict[str, object]]:
    rows = []
    for pair in record["coding"]["pairs"]:
        primary = pair["primary_classification"]
        if primary == "not_addressed":
            continue
        rows.append({
            "pair_id": f"rank_{pair['rank']}",
            "features": feature_key(pair),
            "primary_classification": primary,
            "joint_type": pair_joint(primary),
        })
    return rows


def iter_post_pairs(record: dict) -> list[dict[str, object]]:
    by_id = {pair["pair_id"]: pair for pair in record["coding"]["pairs"]}
    missing = [pair_id for pair_id in POST_PAIR_ORDER if pair_id not in by_id]
    if missing:
        raise ValueError(f"Missing post pair_id(s): {missing}")
    seen: set[tuple[str, str]] = set()
    rows = []
    for pair_id in POST_PAIR_ORDER:
        pair = by_id[pair_id]
        primary = pair["primary_classification"]
        if primary == "not_addressed":
            continue
        key = feature_key(pair)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "pair_id": pair_id,
            "features": key,
            "primary_classification": primary,
            "joint_type": pair_joint(primary),
        })
    return rows


def load_units() -> list[dict[str, object]]:
    """One row per (phase, theory, addressed pair)."""
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
        for phase, record, extractor in (
            ("pre", pre, iter_pre_pairs),
            ("post", post, iter_post_pairs),
        ):
            for pair in extractor(record):
                units.append({
                    "participant_id": key[0],
                    "task": key[1],
                    "group": group,
                    "phase": phase,
                    "pair_id": pair["pair_id"],
                    "feature_a": pair["features"][0],
                    "feature_b": pair["features"][1],
                    "primary_classification": pair["primary_classification"],
                    "joint_type": pair["joint_type"],
                })
    return units


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _add_group_xticks(
    ax,
    x: np.ndarray,
    units: list[dict[str, object]],
    phase_offset: dict,
) -> None:
    for i, group in enumerate(GROUP_ORDER):
        n_pre = sum(u["group"] == group and u["phase"] == "pre" for u in units)
        n_post = sum(u["group"] == group and u["phase"] == "post" for u in units)
        ax.text(
            x[i] + phase_offset["pre"], -4.5, "Pre",
            ha="center", va="top", fontsize=14, color="#444444", clip_on=False,
        )
        ax.text(
            x[i] + phase_offset["post"], -4.5, "Post",
            ha="center", va="top", fontsize=14, color="#444444", clip_on=False,
        )
        ax.text(
            x[i], -11.5, group,
            ha="center", va="top", fontsize=15, color="#111111",
            fontweight="bold", clip_on=False,
        )
        ax.text(
            x[i], -16.8, f"(n = {n_pre} / {n_post} pairs)",
            ha="center", va="top", fontsize=13, color="#444444", clip_on=False,
        )


def plot_joint(units: list[dict[str, object]]) -> list[dict[str, object]]:
    """Mutually exclusive pair types; stacked shares sum to 100% within phase."""
    apply_plot_style()
    table_rows: list[dict[str, object]] = []
    table_rows.append({
        "panel": "counts",
        "group": "ALL",
        "contrast": "n_pair_units",
        "percentage": "",
        "count": len(units),
        "n_pairs": len(units),
    })

    fig, ax = plt.subplots(figsize=(11.4, 8.8))
    x = np.array([0.0, 1.15, 2.3])
    width = 0.38
    phase_offset = {"pre": -width / 2 - 0.02, "post": width / 2 + 0.02}
    bottoms = {phase: np.zeros(len(GROUP_ORDER)) for phase in PHASE_ORDER}
    segment_meta: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    for joint in JOINT_ORDER:
        for phase in PHASE_ORDER:
            vals = []
            for group in GROUP_ORDER:
                sub = [
                    u for u in units
                    if u["group"] == group and u["phase"] == phase
                ]
                n = len(sub)
                count = sum(u["joint_type"] == joint for u in sub)
                pct = 100.0 * count / n if n else float("nan")
                vals.append(pct)
                table_rows.append({
                    "panel": "joint_type",
                    "group": group,
                    "contrast": f"{phase} | {JOINT_LABEL[joint]}",
                    "percentage": round(pct, 2),
                    "count": count,
                    "n_pairs": n,
                })
            vals_arr = np.array(vals, dtype=float)
            ax.bar(
                x + phase_offset[phase],
                vals_arr,
                width=width,
                bottom=bottoms[phase],
                color=JOINT_COLORS[joint],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch=PHASE_HATCH[phase],
                zorder=2,
            )
            segment_meta.append((phase, joint, bottoms[phase].copy(), vals_arr))
            bottoms[phase] = bottoms[phase] + vals_arr

    for phase, joint, bottom, vals in segment_meta:
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
                fontweight="bold",
                zorder=3,
            )

    _add_group_xticks(ax, x, units, phase_offset)
    ax.set_xticks([])
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage of theory–pair units", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_figure_title(ax, "Pair-level mechanism type", pad=16)

    type_handles = [
        mpatches.Patch(
            facecolor=JOINT_COLORS[key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            alpha=BAR_ALPHA,
            label=JOINT_LABEL[key],
        )
        for key in JOINT_ORDER
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
        handles=type_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
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
    fig.legend(
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
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.76))
    save_figure_pdf_svg(fig, OUT_JOINT, bbox_inches="tight")
    plt.close(fig)
    return table_rows


def main() -> None:
    units = load_units()
    write_csv(UNIT_PATH, units)
    joint_rows = plot_joint(units)
    write_csv(SUMMARY_JOINT, joint_rows)

    # Presence figure was theory-level co-occurrence; drop stale outputs.
    for path in (
        OUT_PRESENCE,
        OUT_PRESENCE.with_suffix(".svg"),
        OUT_PRESENCE.with_suffix(".pdf"),
        SUMMARY_PRESENCE,
    ):
        if path.exists():
            path.unlink()

    print(f"wrote {OUT_JOINT}")
    print(f"pair units={len(units)}")
    for phase in PHASE_ORDER:
        print(
            phase,
            Counter(u["joint_type"] for u in units if u["phase"] == phase),
        )


if __name__ == "__main__":
    main()
