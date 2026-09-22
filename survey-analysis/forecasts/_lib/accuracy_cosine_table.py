"""LaTeX + PDF tables for cosine-similarity accuracy (main effects & SOI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import shutil
import tempfile
import zlib

import numpy as np

from common.latex_table_pdf import compile_standalone_table
from common.stats_utils import (
    bootstrap_mean_ci,
    p_value_aggregated_cosine_diff_permutation,
    welch_test,
)

SIG_FOOTNOTE_LATEX = (
    "Mean: two-sided Welch $t$-tests on individual cosine similarities. "
    "Aggregated: two-sided contributor-level permutation tests of "
    "$|\\mathrm{agg}_A-\\mathrm{agg}_B|$ (sum-then-cosine). "
    "Cells report $p$ (3 d.p.; $<$0.001 when smaller) with "
    "* ($p < 0.05$), ** ($p < 0.01$), *** ($p < 0.001$)."
)

AGG_PERM_N = 5000
AGG_PERM_SEED = 20260821

COSINE_TASK_DEFS = (
    ("cos_race", "Racial Inequality"),
    ("cos_gender", "Gender Inequality"),
)

# Mean/CI table groups (descriptive; Topic/Non-Experts overlap PhD/Senior).
ALL_GROUP_DEFS = (
    ("human", "Humans", "Humans"),
    ("1", "Senior Scientists", "Senior"),
    ("0", "PhD Students", "PhD"),
    ("topic", "Topic Experts", "Topic"),
    ("non_topic", "Non-Experts", "Non-Exp."),
    ("2", "GenAI", "GenAI"),
)

# Welch comparisons for Panel B.
WELCH_COMPARISONS = (
    ("human", "2", "Humans vs.\\ GenAI"),
    ("topic", "2", "Topic Experts vs.\\ GenAI"),
    ("1", "2", "Senior Scientists vs.\\ GenAI"),
    ("0", "1", "PhD Students vs.\\ Senior Scientists"),
    ("topic", "non_topic", "Topic Experts vs.\\ Non-Experts"),
)

# Panel A: Effect | Task | Group | N | Mean | CI | Agg
PANEL_A_N_COLS = 7
# Panel B: Comparison | Effect | Task | p_mean | p_agg
PANEL_B_N_COLS = 5


def _rule_full() -> tuple[str, ...]:
    """Solid rule across all columns (e.g. Main Effects ↔ SOI)."""
    return ("\\hline",)


def _rule_from_col(start_col: int, n_cols: int) -> tuple[str, ...]:
    """Solid rule from ``start_col`` through the last column."""
    return (f"\\cline{{{start_col}-{n_cols}}}",)


def _panel_title(letter: str, title: str) -> str:
    """Bold panel letter with period, then bold title (no 'Panel A/B')."""
    return (
        f"{{\\small\\textbf{{{letter}.}}\\enspace "
        f"\\textbf{{{title}}}}}\\\\[0.35em]"
    )


@dataclass(frozen=True)
class CosineTableConfig:
    stem: str
    analysis_name: str
    label_prefix: str
    generator_note: str


def _latex_sig_cell(p: float) -> str:
    """Format p as numeric value; add stars when p < 0.05 (no color)."""
    if not np.isfinite(p):
        return "---"
    if p < 0.001:
        return "$<$0.001$^{***}$"
    if p < 0.01:
        return f"{p:.3f}$^{{**}}$"
    if p < 0.05:
        return f"{p:.3f}$^{{*}}$"
    return f"{p:.3f}"


def _best_row_indices(stats_list: list[dict]) -> set[int]:
    means = [float(s["mean"]) for s in stats_list]
    finite = [m for m in means if np.isfinite(m)]
    if not finite:
        return set()
    best = max(finite)
    return {i for i, m in enumerate(means) if np.isfinite(m) and m == best}


def _latex_bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _format_agg(agg: float, *, bold: bool = False) -> str:
    if not np.isfinite(agg):
        return "---"
    tex = f"{agg:.3f}"
    return _latex_bold(tex) if bold else tex


def _mean_ci_row(
    group_label: str,
    stats: dict,
    *,
    values_key: str,
    agg: float,
    task_cell: str | None = None,
    effect_cell: str | None = None,
    include_effect: bool = False,
    bold_mean: bool = False,
    bold_agg: bool = False,
) -> str:
    del bold_agg  # Aggregated Accuracy is never bolded
    mean = float(stats["mean"])
    n = int(stats["n"])
    lo, hi = bootstrap_mean_ci(np.asarray(stats[values_key], dtype=float))
    if not np.isfinite(mean):
        mean_tex = "---"
        ci_tex = "---"
    else:
        mean_tex = f"{mean:.3f}"
        ci_tex = (
            f"$[{lo:.3f},\\,{hi:.3f}]$"
            if np.isfinite(lo) and np.isfinite(hi)
            else "---"
        )
    if bold_mean:
        group_label = _latex_bold(group_label)
        if mean_tex != "---":
            mean_tex = _latex_bold(mean_tex)
    agg_tex = _format_agg(agg, bold=False)
    if include_effect:
        effect_prefix = f"{effect_cell} & " if effect_cell else "& "
    else:
        effect_prefix = ""
    task_prefix = f"{task_cell} & " if task_cell else "& "
    return (
        f"{effect_prefix}{task_prefix}{group_label} & {n} & {mean_tex} & {ci_tex} "
        f"& {agg_tex} \\\\"
    )


def _multirow_task_cell(task_label: str, n_rows: int) -> str:
    return f"\\multirow{{{n_rows}}}{{*}}{{{task_label}}}"


def _stats_for_group(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    key: str,
    group_id: str,
) -> dict:
    if group_id == "human":
        return stats_human(key)
    if group_id == "topic":
        return stats_topic_expert(key)
    if group_id == "non_topic":
        return stats_non_topic_expert(key)
    return stats_by_group(key, group_id)


def _best_agg_indices(aggs: list[float]) -> set[int]:
    finite = [a for a in aggs if np.isfinite(a)]
    if not finite:
        return set()
    best = max(finite)
    return {i for i, a in enumerate(aggs) if np.isfinite(a) and a == best}


def _mean_ci_rows_all_groups(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    agg_by_group: Callable[[str, str], float],
    *,
    values_key: str,
) -> list[str]:
    rows: list[str] = []
    n_groups = len(ALL_GROUP_DEFS)
    n_tasks = len(COSINE_TASK_DEFS)
    for task_idx, (key, task_label) in enumerate(COSINE_TASK_DEFS):
        task_stats = [
            _stats_for_group(
                stats_by_group,
                stats_human,
                stats_topic_expert,
                stats_non_topic_expert,
                key,
                group_id,
            )
            for group_id, _, _ in ALL_GROUP_DEFS
        ]
        task_aggs = [
            float(agg_by_group(key, group_id))
            for group_id, _, _ in ALL_GROUP_DEFS
        ]
        best_mean = _best_row_indices(task_stats)
        best_agg = _best_agg_indices(task_aggs)
        for i, (_, group_label, _) in enumerate(ALL_GROUP_DEFS):
            task_cell = _multirow_task_cell(task_label, n_groups) if i == 0 else None
            rows.append(
                _mean_ci_row(
                    group_label,
                    task_stats[i],
                    values_key=values_key,
                    agg=task_aggs[i],
                    task_cell=task_cell,
                    bold_mean=i in best_mean,
                    bold_agg=i in best_agg,
                )
            )
    return rows


def _mean_ci_table_tex(
    mean_rows: list[str],
    *,
    config: CosineTableConfig,
) -> list[str]:
    return [
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.35}",
        "\\begin{tabular}{@{}l@{}}",
        _panel_title("a", "Mean prediction accuracy by group"),
        "\\begin{tabular}{@{}clcccc@{}}",
        "\\toprule",
        "\\textbf{Task} & \\textbf{Group} & $\\mathbf{N}$ & \\textbf{Mean Accuracy} "
        "& \\textbf{95\\% CI} & \\textbf{Aggregated Accuracy} \\\\",
        "\\midrule",
        *mean_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\",
        "\\end{tabular}",
        "}",
    ]


def _welch_p_for_pair(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    key: str,
    id_a: str,
    id_b: str,
    *,
    values_key: str,
) -> float:
    a = _stats_for_group(
        stats_by_group,
        stats_human,
        stats_topic_expert,
        stats_non_topic_expert,
        key,
        id_a,
    )
    b = _stats_for_group(
        stats_by_group,
        stats_human,
        stats_topic_expert,
        stats_non_topic_expert,
        key,
        id_b,
    )
    return welch_test(a, b, values_key=values_key)


def _agg_perm_p_for_pair(
    vecs_by_group: Callable[[str, str], np.ndarray],
    ml_vec_for_task: Callable[[str], np.ndarray],
    key: str,
    id_a: str,
    id_b: str,
    *,
    seed: int,
) -> float:
    return p_value_aggregated_cosine_diff_permutation(
        vecs_by_group(key, id_a),
        vecs_by_group(key, id_b),
        ml_vec_for_task(key),
        n_perm=AGG_PERM_N,
        seed=seed,
    )


def _agg_perm_seed(effect_label: str, key: str, id_a: str, id_b: str) -> int:
    """Stable seed so panel-b aggregated p-values are reproducible."""
    tag = f"{effect_label}|{key}|{id_a}|{id_b}|{AGG_PERM_SEED}".encode()
    return int(zlib.crc32(tag) & 0x7FFFFFFF)


def _welch_comparison_rows(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    *,
    values_key: str,
) -> list[str]:
    """Panel B: Comparison × Task (Comparison first)."""
    rows: list[str] = []
    n_comp = len(WELCH_COMPARISONS)
    n_tasks = len(COSINE_TASK_DEFS)
    for comp_idx, (id_a, id_b, label) in enumerate(WELCH_COMPARISONS):
        for task_idx, (key, task_label) in enumerate(COSINE_TASK_DEFS):
            cell = _latex_sig_cell(
                _welch_p_for_pair(
                    stats_by_group,
                    stats_human,
                    stats_topic_expert,
                    stats_non_topic_expert,
                    key,
                    id_a,
                    id_b,
                    values_key=values_key,
                )
            )
            comp_cell = (
                _multirow_task_cell(label, n_tasks) if task_idx == 0 else None
            )
            comp_prefix = f"{comp_cell} & " if comp_cell else "& "
            rows.append(f"{comp_prefix}{task_label} & {cell} \\\\")
        if comp_idx < n_comp - 1:
            rows.extend(_rule_full())
    return rows


def _welch_table_tex(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    *,
    values_key: str,
    config: CosineTableConfig,
) -> list[str]:
    body = _welch_comparison_rows(
        stats_by_group,
        stats_human,
        stats_topic_expert,
        stats_non_topic_expert,
        values_key=values_key,
    )
    return [
        "\\par\\vspace{1.0em}",
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\begin{tabular}{@{}l@{}}",
        _panel_title("b", "Two-sided Welch's $t$-tests of mean prediction accuracy"),
        "\\begin{tabular}{@{}llc@{}}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{Task} & $\\mathbf{p}$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\",
        "\\end{tabular}",
        "}",
        "",
    ]


def build_cosine_combined_tex(
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_genai: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    agg_by_group: Callable[[str, str], float],
    *,
    values_key: str,
    config: CosineTableConfig,
) -> str:
    """One combined figure: Panel A (means) + Panel B (Welch tests)."""
    del stats_genai  # GenAI comes from stats_by_group("2")
    mean_rows = _mean_ci_rows_all_groups(
        stats_by_group,
        stats_human,
        stats_topic_expert,
        stats_non_topic_expert,
        agg_by_group,
        values_key=values_key,
    )
    return "\n".join([
        f"% {config.generator_note}",
        "",
        f"\\label{{tab:{config.label_prefix}_cosine_similarity}}",
        "",
        *_mean_ci_table_tex(mean_rows, config=config),
        *_welch_table_tex(
            stats_by_group,
            stats_human,
            stats_topic_expert,
            stats_non_topic_expert,
            values_key=values_key,
            config=config,
        ),
    ]) + "\n"


def write_cosine_tables(
    out_dir: Path,
    stats_by_group: Callable[[str, str], dict],
    stats_human: Callable[[str], dict],
    stats_genai: Callable[[str], dict],
    stats_topic_expert: Callable[[str], dict],
    stats_non_topic_expert: Callable[[str], dict],
    agg_by_group: Callable[[str, str], float],
    *,
    values_key: str,
    config: CosineTableConfig,
) -> Path:
    """Compile one standalone PNG: all-group means + exclusive Welch panels."""
    body = build_cosine_combined_tex(
        stats_by_group,
        stats_human,
        stats_genai,
        stats_topic_expert,
        stats_non_topic_expert,
        agg_by_group,
        values_key=values_key,
        config=config,
    )
    return compile_standalone_table(out_dir, config.stem, body)


write_cosine_tables_pdf = write_cosine_tables


@dataclass(frozen=True)
class CosineEffectSource:
    """One effect block (Main Effects or SOI) for the merged table."""

    effect_label: str
    stats_by_group: Callable[[str, str], dict]
    stats_human: Callable[[str], dict]
    stats_topic_expert: Callable[[str], dict]
    stats_non_topic_expert: Callable[[str], dict]
    agg_by_group: Callable[[str, str], float]
    values_key: str
    vecs_by_group: Callable[[str, str], np.ndarray]
    ml_vec_for_task: Callable[[str], np.ndarray]


def _mean_ci_rows_with_effect(sources: list[CosineEffectSource]) -> list[str]:
    rows: list[str] = []
    n_groups = len(ALL_GROUP_DEFS)
    n_tasks = len(COSINE_TASK_DEFS)
    rows_per_effect = n_tasks * n_groups
    for effect_idx, src in enumerate(sources):
        for task_idx, (key, task_label) in enumerate(COSINE_TASK_DEFS):
            task_stats = [
                _stats_for_group(
                    src.stats_by_group,
                    src.stats_human,
                    src.stats_topic_expert,
                    src.stats_non_topic_expert,
                    key,
                    group_id,
                )
                for group_id, _, _ in ALL_GROUP_DEFS
            ]
            task_aggs = [
                float(src.agg_by_group(key, group_id))
                for group_id, _, _ in ALL_GROUP_DEFS
            ]
            best_mean = _best_row_indices(task_stats)
            best_agg = _best_agg_indices(task_aggs)
            for i, (_, group_label, _) in enumerate(ALL_GROUP_DEFS):
                is_first_of_effect = task_idx == 0 and i == 0
                effect_cell = (
                    _multirow_task_cell(src.effect_label, rows_per_effect)
                    if is_first_of_effect
                    else None
                )
                task_cell = (
                    _multirow_task_cell(task_label, n_groups) if i == 0 else None
                )
                rows.append(
                    _mean_ci_row(
                        group_label,
                        task_stats[i],
                        values_key=src.values_key,
                        agg=task_aggs[i],
                        effect_cell=effect_cell,
                        task_cell=task_cell,
                        include_effect=True,
                        bold_mean=i in best_mean,
                        bold_agg=i in best_agg,
                    )
                )
            if task_idx < n_tasks - 1:
                # Racial ↔ Gender: from Task column (col 2).
                rows.extend(_rule_from_col(2, PANEL_A_N_COLS))
        if effect_idx < len(sources) - 1:
            # Main Effects ↔ SOI: full width from Effect column.
            rows.extend(_rule_full())
    return rows


def _mean_ci_table_tex_with_effect(mean_rows: list[str]) -> list[str]:
    return [
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\begin{tabular}{@{}l@{}}",
        _panel_title("a", "Mean prediction accuracy by group"),
        "\\begin{tabular}{@{}lclcccc@{}}",
        "\\toprule",
        "\\textbf{Effect} & \\textbf{Task} & \\textbf{Group} & $\\mathbf{N}$ "
        "& \\textbf{Mean Accuracy} & \\textbf{95\\% CI} "
        "& \\textbf{Aggregated Accuracy} \\\\",
        "\\midrule",
        *mean_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\",
        "\\end{tabular}",
        "}",
    ]


def _welch_comparison_rows_with_effect(
    sources: list[CosineEffectSource],
) -> list[str]:
    """Panel B: Comparison × Effect × Task with mean and aggregated p columns."""
    rows: list[str] = []
    n_effects = len(sources)
    n_tasks = len(COSINE_TASK_DEFS)
    n_comp = len(WELCH_COMPARISONS)
    rows_per_comparison = n_effects * n_tasks
    for comp_idx, (id_a, id_b, label) in enumerate(WELCH_COMPARISONS):
        row_in_comp = 0
        for effect_idx, src in enumerate(sources):
            for task_idx, (key, task_label) in enumerate(COSINE_TASK_DEFS):
                p_mean = _welch_p_for_pair(
                    src.stats_by_group,
                    src.stats_human,
                    src.stats_topic_expert,
                    src.stats_non_topic_expert,
                    key,
                    id_a,
                    id_b,
                    values_key=src.values_key,
                )
                p_agg = _agg_perm_p_for_pair(
                    src.vecs_by_group,
                    src.ml_vec_for_task,
                    key,
                    id_a,
                    id_b,
                    seed=_agg_perm_seed(src.effect_label, key, id_a, id_b),
                )
                mean_cell = _latex_sig_cell(p_mean)
                agg_cell = _latex_sig_cell(p_agg)
                comp_cell = (
                    _multirow_task_cell(label, rows_per_comparison)
                    if row_in_comp == 0
                    else None
                )
                effect_cell = (
                    _multirow_task_cell(src.effect_label, n_tasks)
                    if task_idx == 0
                    else None
                )
                comp_prefix = f"{comp_cell} & " if comp_cell else "& "
                effect_prefix = f"{effect_cell} & " if effect_cell else "& "
                rows.append(
                    f"{comp_prefix}{effect_prefix}{task_label} "
                    f"& {mean_cell} & {agg_cell} \\\\"
                )
                row_in_comp += 1
            if effect_idx < n_effects - 1:
                # Main Effects ↔ SOI: from Effect column (col 2).
                rows.extend(_rule_from_col(2, PANEL_B_N_COLS))
        if comp_idx < n_comp - 1:
            # Between comparisons: full width from leftmost.
            rows.extend(_rule_full())
    return rows


def _welch_table_tex_with_effect(sources: list[CosineEffectSource]) -> list[str]:
    body = _welch_comparison_rows_with_effect(sources)
    return [
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{5.5pt}",
        "\\renewcommand{\\arraystretch}{1.25}",
        "\\begin{tabular}{@{}l@{}}",
        _panel_title(
            "b",
            "Two-sided tests of mean and aggregated prediction accuracy",
        ),
        "\\begin{tabular}{@{}lllcc@{}}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{Effect} & \\textbf{Task} "
        "& $\\mathbf{p}_{\\mathrm{mean}}$ "
        "& $\\mathbf{p}_{\\mathrm{agg}}$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\",
        "\\end{tabular}",
        "}",
    ]


def build_cosine_me_soi_combined_tex(sources: list[CosineEffectSource]) -> str:
    """Merged Main Effects + SOI figure with an Effect column."""
    mean_rows = _mean_ci_rows_with_effect(sources)
    # Use \\ + \\noalign{\\vspace{...}} — \\[dim] is unreliable with nested
    # tabulars / braced panel groups and was not changing the gap.
    return "\n".join([
        "% Auto-generated combined Main Effects + Second-Order Interactions",
        "",
        "\\label{tab:cosine_similarity_me_soi}",
        "",
        "% Natural-width stack (avoid \\par / \\textwidth right padding).",
        "\\begin{tabular}{@{}l@{}}",
        *_mean_ci_table_tex_with_effect(mean_rows),
        "\\\\",
        "\\noalign{\\vspace{2.2em}}",
        *_welch_table_tex_with_effect(sources),
        "\\\\",
        "\\end{tabular}",
        "",
    ]) + "\n"


def write_cosine_me_soi_tables(
    out_dir: Path,
    sources: list[CosineEffectSource],
    *,
    stem: str = "fig_accuracy_tables_all",
) -> Path:
    """Write merged ME+SOI cosine table SVG to ``out_dir`` (no ``.tex`` artifacts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = build_cosine_me_soi_combined_tex(sources)
    dest = out_dir / f"{stem}.svg"
    with tempfile.TemporaryDirectory() as tmp:
        svg_tmp = compile_standalone_table(
            Path(tmp),
            stem,
            body,
            output_format="svg",
            crop="standalone",
        )
        # compile writes ``{stem}_standalone.svg`` in the temp dir
        shutil.move(str(svg_tmp), str(dest))
    return dest


MAIN_EFFECTS_COSINE_TABLE_CONFIG = CosineTableConfig(
    stem="03_cosine_similarity_tables",
    analysis_name="Main Effects",
    label_prefix="me",
    generator_note="Auto-generated by forecasts/fig_accuracy_tables_all.py",
)

SOI_COSINE_TABLE_CONFIG = CosineTableConfig(
    stem="03_soi_cosine_similarity_tables",
    analysis_name="Second-Order Interactions",
    label_prefix="soi",
    generator_note="Auto-generated by forecasts/fig_accuracy_tables_all.py",
)
