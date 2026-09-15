"""Core vs tail Pre|Post summary (prediction 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_legacy import (
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    available_embedding_columns,
    read_anonymized_embeddings_wide,
)
from compare_legacy import (
    discover_task_pairs,
    run_all_prediction1,
)


def run(
    embeddings_root: Path | None = None,
    embedding_col: str | None = None,
) -> None:
    """Read clustering CSVs; write outputs/core_tail_structure/core_tail_structure.{svg,png}.

    Prerequisite: run_semantic_clustering.py
    """
    root = (embeddings_root or DEFAULT_EMBEDDINGS_ROOT).expanduser().resolve()
    sample_pair = discover_task_pairs(root)[0]
    sample_df = read_anonymized_embeddings_wide(
        sample_pair[0] / "embeddings_wide.parquet"
    )
    col = available_embedding_columns(
        sample_df,
        [embedding_col] if embedding_col else DEFAULT_EMBEDDING_COLUMNS,
    )[0]

    print(f"Embeddings root: {root}")
    print(f"Embedding column: {col}")
    run_all_prediction1(root, col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Core–tail Pre|Post summary (Humans and GenAI)."
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


__all__ = ["main", "run", "run_all_prediction1"]


if __name__ == "__main__":
    main()
