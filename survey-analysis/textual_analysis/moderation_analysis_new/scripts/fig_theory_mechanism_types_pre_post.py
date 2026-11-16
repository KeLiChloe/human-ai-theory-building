"""Parallel mechanism presence before and after ML.

Each mechanism is coded independently. Undiscussed pairs are excluded; post-ML unordered
feature pairs are deduplicated. The three binary presence flags are retained
in the units CSV, alongside their combination.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File-only plotting also works without a graphical session.
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from coding_schema import load_coding_records, mechanism_combination

HERE = Path(__file__).resolve().parents[1]
TEXTUAL = HERE.parent
if str(TEXTUAL) not in sys.path:
    sys.path.insert(0, str(TEXTUAL))

from viz_style import (  # noqa: E402
    BAR_ALPHA,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    apply_plot_style,
)

PRE_PATH = HERE / "outputs" / "moderation_coding_pre_ML_v2.json"
POST_PATH = HERE / "outputs" / "moderation_coding_post_ML_v2.json"
SUMMARY_JOINT = HERE / "outputs" / "fig_theory_mechanism_joint_type_pre_post_v2_summary.csv"
UNIT_PATH = HERE / "outputs" / "fig_theory_mechanism_types_pre_post_v2_units.csv"

# Main figure and summary: overlapping mechanism presence, not exclusive combinations.
OUT_PRESENCE = HERE / "outputs" / "fig_theory_mechanism_presence_pre_post_v2.svg"
SUMMARY_PRESENCE = HERE / "outputs" / "fig_theory_mechanism_presence_pre_post_v2_summary.csv"

POST_PAIR_ORDER = (
    "USER_SELECTED_rank_1",
    "USER_SELECTED_rank_2",
    "USER_SELECTED_rank_3",
    "ML_rank_1",
    "ML_rank_2",
    "ML_rank_3",
)

JOINT_ORDER = (
    "no_mechanism", "interaction_only", "curvature_only", "curvature_interaction",
    "other_only", "curvature_other", "interaction_other", "all_three",
)
JOINT_LABEL = {
    "no_mechanism": "No coded mechanism",
    "interaction_only": "Interaction-based only",
    "curvature_only": "Curvature only",
    "curvature_interaction": "Curvature + interaction",
    "other_only": "Other only",
    "curvature_other": "Curvature + other",
    "interaction_other": "Interaction + other",
    "all_three": "All three mechanisms",
}
PRESENCE_ORDER = ("curvature", "interaction_based", "other_mechanism", "no_mechanism")
PRESENCE_LABEL = {
    "curvature": "Curvature", "interaction_based": "Interaction-based",
    "other_mechanism": "Other mechanism", "no_mechanism": "No mechanism",
}
PRESENCE_COLORS = {
    "curvature": "#A8D5A2", "interaction_based": "#F2C14E",
    "other_mechanism": "#F2BACB", "no_mechanism": "#C5CED6",
}

GROUP_ORDER = ("PhD Students", "Senior Scientists", "GenAI")
GROUP_CODE = {"0": "PhD Students", "1": "Senior Scientists", "2": "GenAI"}
PHASE_ORDER = ("pre", "post")
PHASE_LABEL = {"pre": "Pre-ML", "post": "Post-ML"}
PHASE_HATCH = {"pre": "", "post": "///"}


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.08)


def feature_key(pair: dict) -> tuple[str, str]:
    return tuple(sorted(pair["features"]))  # type: ignore[return-value]


def iter_pre_pairs(record: dict) -> list[dict[str, object]]:
    rows = []
    for pair in record["coding"]["pairs"]:
        if pair["discussed"] == "no":
            continue
        rows.append({
            "pair_id": f"rank_{pair['rank']}",
            "features": feature_key(pair),
            "no_mechanism_reason": pair["no_mechanism_reason"],
            "joint_type": mechanism_combination(pair),
            "curvature_present": pair["curvature"]["present"],
            "interaction_based_present": pair["interaction_based"]["present"],
            "other_mechanism_present": pair["other_mechanism"]["present"],
            "interaction_location": pair["interaction_based"]["interaction_location"],
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
        if pair["discussed"] == "no":
            continue
        key = feature_key(pair)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "pair_id": pair_id,
            "features": key,
            "no_mechanism_reason": pair["no_mechanism_reason"],
            "joint_type": mechanism_combination(pair),
            "curvature_present": pair["curvature"]["present"],
            "interaction_based_present": pair["interaction_based"]["present"],
            "other_mechanism_present": pair["other_mechanism"]["present"],
            "interaction_location": pair["interaction_based"]["interaction_location"],
        })
    return rows


def load_units() -> list[dict[str, object]]:
    """One row per (phase, theory, discussed feature pair)."""
    pre_records = {
        (rec["participant_id"], rec["task"]): rec
        for rec in load_coding_records(PRE_PATH)
        if rec.get("coding") is not None
    }
    post_records = {
        (rec["participant_id"], rec["task"]): rec
        for rec in load_coding_records(POST_PATH)
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
                    "curvature_present": pair["curvature_present"],
                    "interaction_based_present": pair["interaction_based_present"],
                    "other_mechanism_present": pair["other_mechanism_present"],
                    "no_mechanism_reason": pair["no_mechanism_reason"],
                    "joint_type": pair["joint_type"],
                    "interaction_location": pair["interaction_location"],
                })
    return units


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_presence(units: list[dict[str, object]]) -> list[dict[str, object]]:
    """Use the same discussed-pair denominator for every independent judgment."""
    rows = []
    for group in GROUP_ORDER:
        for phase in PHASE_ORDER:
            sub = [u for u in units if u["group"] == group and u["phase"] == phase]
            for mechanism in PRESENCE_ORDER:
                if mechanism == "no_mechanism":
                    count = sum(all(u[f"{key}_present"] == "no" for key in PRESENCE_ORDER[:3]) for u in sub)
                else:
                    count = sum(u[f"{mechanism}_present"] == "yes" for u in sub)
                rows.append({
                    "group": group, "phase": phase, "mechanism": mechanism,
                    "count": count, "n_pairs": len(sub),
                    "percentage": 100.0 * count / len(sub) if sub else float("nan"),
                })
    return rows


def summarize_combinations(units: list[dict[str, object]]) -> list[dict[str, object]]:
    """Retain the eight detailed combinations for checking and further analysis."""
    rows = []
    for group in GROUP_ORDER:
        for phase in PHASE_ORDER:
            sub = [u for u in units if u["group"] == group and u["phase"] == phase]
            for joint in JOINT_ORDER:
                count = sum(u["joint_type"] == joint for u in sub)
                rows.append({"group": group, "phase": phase, "joint_type": joint,
                             "count": count, "n_pairs": len(sub),
                             "percentage": 100.0 * count / len(sub) if sub else float("nan")})
    return rows


def plot_presence(rows: list[dict[str, object]]) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharey=True)
    lookup = {(r["group"], r["phase"], r["mechanism"]): r for r in rows}
    x = np.arange(len(PRESENCE_ORDER))
    width = 0.34
    for ax, group in zip(axes, GROUP_ORDER):
        for phase, offset in (("pre", -width / 2), ("post", width / 2)):
            values = [lookup[(group, phase, key)]["percentage"] for key in PRESENCE_ORDER]
            bars = ax.bar(
                x + offset, values, width=width,
                color=[PRESENCE_COLORS[key] for key in PRESENCE_ORDER],
                edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH,
                alpha=BAR_ALPHA, hatch=PHASE_HATCH[phase], zorder=3,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    text = f"{value:.1f}%" if 0 < value < 1 else f"{value:.0f}%"
                    ax.text(bar.get_x()+bar.get_width()/2, value+1.6, text,
                            ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, ["Curvature", "Interaction-\nbased", "Other\nmechanism", "No\nmechanism"], fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(group, fontsize=16, fontweight="bold", pad=24)
        n_pre = lookup[(group, "pre", "curvature")]["n_pairs"]
        n_post = lookup[(group, "post", "curvature")]["n_pairs"]
        ax.text(0.5, 1.015, f"Discussed pairs: pre n={n_pre}, post n={n_post}",
                transform=ax.transAxes, ha="center", fontsize=10, color="#555555")
    axes[0].set_ylabel("Percentage of discussed theory–pair units", fontsize=12)
    handles = [mpatches.Patch(facecolor="#DDDDDD", edgecolor=BAR_EDGE_COLOR,
                             hatch=PHASE_HATCH[phase], label=PHASE_LABEL[phase]) for phase in PHASE_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
               frameon=False, fontsize=12)
    fig.suptitle("Mechanism presence before and after ML", fontsize=20, y=0.99)
    fig.text(0.5, 0.025,
             "Mechanisms can coexist; percentages need not sum to 100%. No mechanism = all three marked no.\n"
             "Undiscussed pairs excluded; post-ML pairs deduplicated within each theory.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.2, wspace=0.13)
    save_svg(fig, OUT_PRESENCE)
    plt.close(fig)


def plot_stacked(rows: list[dict[str, object]], out_path: Path) -> None:
    """Eight exclusive combinations; each group/phase sums to 100%."""
    apply_plot_style()
    order = ("no_mechanism", "other_only", "interaction_only", "interaction_other",
             "curvature_only", "curvature_interaction", "curvature_other", "all_three")
    colors = {
        "no_mechanism": "#C5CED6", "other_only": "#F2BACB",
        "interaction_only": "#F2C14E", "interaction_other": "#E99B70",
        "curvature_only": "#A8D5A2", "curvature_interaction": "#8AA8CB",
        "curvature_other": "#C4A1D6", "all_three": "#7F6A99",
    }
    lookup = {(r["group"], r["phase"], r["joint_type"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        bottoms = np.zeros(2)
        for key in order:
            values = np.array([lookup[(group, phase, key)]["percentage"] for phase in PHASE_ORDER])
            ax.bar([0, 1], values, bottom=bottoms, width=0.62, color=colors[key],
                   edgecolor="white", linewidth=0.8, alpha=BAR_ALPHA, zorder=3)
            for i, value in enumerate(values):
                if np.isfinite(value) and value >= 4:
                    ax.text(i, bottoms[i]+value/2, f"{value:.0f}%", ha="center", va="center", fontsize=12)
            bottoms += values
        labels = [f"{PHASE_LABEL[phase]}\n(n={lookup[(group, phase, order[0])]['n_pairs']})" for phase in PHASE_ORDER]
        ax.set_xticks([0, 1], labels, fontsize=12)
        ax.set_xlim(-0.65, 1.65)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(group, fontsize=16, pad=14, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Percentage of discussed theory–pair units", fontsize=12)
    handles = [mpatches.Patch(facecolor=colors[key], label=JOINT_LABEL[key]) for key in order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4,
               frameon=False, fontsize=10.5, columnspacing=1.4)
    fig.suptitle("Mechanism combinations before and after ML", fontsize=20, y=0.99)
    fig.text(0.5, 0.025, "Exclusive combinations sum to 100%; segments below 4% are not labeled.\n"
             "Undiscussed pairs excluded; post-ML pairs deduplicated within each theory.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.16, wspace=0.14)
    save_svg(fig, out_path)
    plt.close(fig)


def summarize_changes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(r["group"], r["phase"], r["mechanism"]): r for r in rows}
    result = []
    for group in GROUP_ORDER:
        for mechanism in PRESENCE_ORDER:
            pre = lookup[(group, "pre", mechanism)]
            post = lookup[(group, "post", mechanism)]
            result.append({
                "group": group, "mechanism": mechanism,
                "pre_count": pre["count"], "pre_n_pairs": pre["n_pairs"], "pre_percentage": pre["percentage"],
                "post_count": post["count"], "post_n_pairs": post["n_pairs"], "post_percentage": post["percentage"],
                "change_percentage_points": post["percentage"] - pre["percentage"],
            })
    return result


def plot_changes(rows: list[dict[str, object]], out_path: Path) -> None:
    """Signed marginal changes, calculated before rounding percentages."""
    apply_plot_style()
    lookup = {(r["group"], r["mechanism"]): r for r in rows}
    max_change = max(abs(float(r["change_percentage_points"])) for r in rows)
    limit = max(10, int(np.ceil(max_change/5))*5+5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharex=True)
    positions = np.arange(4)[::-1]
    for ax, group in zip(axes, GROUP_ORDER):
        values = [lookup[(group, key)]["change_percentage_points"] for key in PRESENCE_ORDER]
        ax.barh(positions, values, height=0.56, color=[PRESENCE_COLORS[k] for k in PRESENCE_ORDER],
                edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, alpha=BAR_ALPHA, zorder=3)
        ax.axvline(0, color="#333333", linewidth=1.1, zorder=4)
        for y, value in zip(positions, values):
            label = "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
            ax.text(value+(0.55 if value >= 0 else -0.55), y, label,
                    va="center", ha="left" if value >= 0 else "right", fontsize=11)
        ax.set_yticks(positions, ["Curvature", "Interaction-\nbased", "Other\nmechanism", "No\nmechanism"], fontsize=11)
        ax.set_xlim(-limit, limit)
        ax.set_xticks(np.arange(-limit, limit+1, 10))
        ax.set_ylim(-0.6, 3.7)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(group, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("← Decrease     Increase →", fontsize=11, labelpad=10, fontfamily="DejaVu Sans")
    fig.suptitle("Change in mechanism presence: Post − Pre", fontsize=20, y=0.98)
    fig.text(0.5, 0.865, "Percentage-point change; zero indicates no change", ha="center", fontsize=12, color="#444444")
    fig.text(0.5, 0.025, "Differences use each phase's discussed-pair denominator; mechanisms may coexist.\n"
             "Post-ML pairs deduplicated within theory. This is a change in proportions, not matched-pair transitions.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.75, bottom=0.22, wspace=0.5)
    save_svg(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v2",), default="v2",
                        help="Select coding inputs and give all output files the matching version suffix")
    parser.add_argument("--chart", choices=("presence", "stacked", "change", "all"), default="presence",
                        help="Select a figure; all generates presence, combinations and percentage-point changes")
    args = parser.parse_args()
    global PRE_PATH, POST_PATH, OUT_PRESENCE, SUMMARY_PRESENCE, SUMMARY_JOINT, UNIT_PATH
    suffix = "_v2"
    PRE_PATH = HERE / "outputs" / f"moderation_coding_pre_ML{suffix}.json"
    POST_PATH = HERE / "outputs" / f"moderation_coding_post_ML{suffix}.json"
    OUT_PRESENCE = HERE / "outputs" / f"fig_theory_mechanism_presence_pre_post{suffix}.svg"
    SUMMARY_PRESENCE = HERE / "outputs" / f"fig_theory_mechanism_presence_pre_post{suffix}_summary.csv"
    SUMMARY_JOINT = HERE / "outputs" / f"fig_theory_mechanism_joint_type_pre_post{suffix}_summary.csv"
    UNIT_PATH = HERE / "outputs" / f"fig_theory_mechanism_types_pre_post{suffix}_units.csv"
    units = load_units()
    write_csv(UNIT_PATH, units)
    combinations = summarize_combinations(units)
    write_csv(SUMMARY_JOINT, combinations)
    rows = summarize_presence(units)
    write_csv(SUMMARY_PRESENCE, rows)
    if args.chart in ("presence", "all"):
        plot_presence(rows)
        print(f"wrote {OUT_PRESENCE}")
    if args.chart in ("stacked", "all"):
        stacked = HERE / "outputs" / f"fig_theory_mechanism_joint_type_pre_post{suffix}.svg"
        plot_stacked(combinations, stacked)
        print(f"wrote {stacked}")
    if args.chart in ("change", "all"):
        changes = summarize_changes(rows)
        change_path = HERE / "outputs" / f"fig_theory_mechanism_change_pre_post{suffix}.svg"
        write_csv(change_path.with_name(change_path.stem+"_summary.csv"), changes)
        plot_changes(changes, change_path)
        print(f"wrote {change_path}")
    print(f"pair units={len(units)}")
    for phase in PHASE_ORDER:
        print(phase, Counter(u["joint_type"] for u in units if u["phase"] == phase))


if __name__ == "__main__":
    main()
