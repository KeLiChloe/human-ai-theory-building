"""Self pre–post embedding shift + pooled three-panel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_legacy import (
    COMPARISONS_SELF_SUBDIR,
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    available_embedding_columns,
    comparisons_pre_post_dir,
    read_anonymized_embeddings_wide,
)
from compare_legacy import (
    discover_task_pairs,
    run_self_pre_post_embedding_distance,
    run_shift_conditioned_on_pre_ml_accuracy,
)


def run(
    embeddings_root: Path | None = None,
    embedding_col: str | None = None,
) -> Path:
    """Write MixedLM CSVs + theory_revision figure (no trajectory figs)."""
    root = (embeddings_root or DEFAULT_EMBEDDINGS_ROOT).expanduser().resolve()
    sample_pair = discover_task_pairs(root)[0]
    sample_df = read_anonymized_embeddings_wide(
        sample_pair[0] / "embeddings_wide.parquet"
    )
    col = available_embedding_columns(
        sample_df,
        [embedding_col] if embedding_col else DEFAULT_EMBEDDING_COLUMNS,
    )[0]

    outdir = comparisons_pre_post_dir(root, COMPARISONS_SELF_SUBDIR)
    print(f"Embeddings root: {root}")
    print(f"Embedding column: {col}")
    print(f"Output: {outdir}")

    participant_df = run_self_pre_post_embedding_distance(root, col, outdir)
    run_shift_conditioned_on_pre_ml_accuracy(participant_df, outdir)
    print(f"Self pre–post embedding distance outputs: {outdir}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Theory-revision figure and MixedLM CSVs (panel d)."
        )
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


__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
