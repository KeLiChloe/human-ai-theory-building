"""Pre|Post semantic space maps (collapsed Humans + three-group variants)."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_legacy import (
    ANALYSIS_SEED,
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    DIVERSITY_TASK_PANEL_ORDER,
    PHASE_NAMES,
    THRESHOLD_QUANTILE,
    available_embedding_columns,
    comparisons_pre_post_dir,
    discover_phase_task_dirs,
    load_phase_task_bundle,
    plot_pre_post_semantic_maps_expanded,
    read_anonymized_embeddings_wide,
    task_label_from_key,
)


def load_phase_bundles(
    embeddings_root: Path,
    embedding_col: str,
    *,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> dict[str, list[tuple[str, dict]]]:
    """Load pre-ML / post-ML task bundles required for semantic maps."""
    phase_bundles: dict[str, list[tuple[str, dict]]] = {}
    for phase in PHASE_NAMES:
        task_dirs = discover_phase_task_dirs(embeddings_root, phase)
        if len(task_dirs) != len(DIVERSITY_TASK_PANEL_ORDER):
            found = [key for key, _ in task_dirs]
            missing = [k for k in DIVERSITY_TASK_PANEL_ORDER if k not in found]
            raise FileNotFoundError(
                f"Missing {phase} tasks under {embeddings_root}: {missing} "
                f"(found {found})."
            )
        task_bundles: list[tuple[str, dict]] = []
        for task_key, set_dir in task_dirs:
            print(f"  Loading {phase} · {task_label_from_key(task_key)}")
            task_bundles.append(
                (
                    task_key,
                    load_phase_task_bundle(
                        set_dir,
                        embedding_col,
                        threshold_quantile=threshold_quantile,
                    ),
                )
            )
        phase_bundles[phase] = task_bundles
    return phase_bundles


def run(
    embeddings_root: Path | None = None,
    embedding_col: str | None = None,
    *,
    seed: int = ANALYSIS_SEED,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> Path:
    """Write semantic_space_map.png/.svg and semantic_space_map_three_groups.*."""
    root = (embeddings_root or DEFAULT_EMBEDDINGS_ROOT).expanduser().resolve()
    sample_dirs = discover_phase_task_dirs(root, "pre-ML")
    if not sample_dirs:
        raise FileNotFoundError(f"No pre-ML embedding sets under {root}")
    sample_df = read_anonymized_embeddings_wide(
        sample_dirs[0][1] / "embeddings_wide.parquet"
    )
    col = available_embedding_columns(
        sample_df,
        [embedding_col] if embedding_col else DEFAULT_EMBEDDING_COLUMNS,
    )[0]

    print(f"Embeddings root: {root}")
    print(f"Embedding column: {col}")
    phase_bundles = load_phase_bundles(
        root, col, threshold_quantile=threshold_quantile
    )

    compare_outdir = comparisons_pre_post_dir(root, "semantic_map_PCA")
    compare_outdir.mkdir(parents=True, exist_ok=True)
    plot_pre_post_semantic_maps_expanded(
        phase_bundles,
        compare_outdir / "semantic_space_map.png",
        collapse_human=True,
        seed=seed,
    )
    plot_pre_post_semantic_maps_expanded(
        phase_bundles,
        compare_outdir / "semantic_space_map_three_groups.png",
        collapse_human=False,
        seed=seed,
    )
    print(f"Saved Pre|Post expanded semantic maps to: {compare_outdir}")
    return compare_outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre|Post semantic space maps (Humans collapsed + three groups)."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help="Root folder containing topic/task/pre-ML and post-ML sets.",
    )
    parser.add_argument(
        "--embedding-col",
        default=None,
        help="Embedding column (default: raw 3072d).",
    )
    args = parser.parse_args()
    run(args.embeddings_root, args.embedding_col)


__all__ = [
    "load_phase_bundles",
    "main",
    "plot_pre_post_semantic_maps_expanded",
    "run",
]


if __name__ == "__main__":
    main()
