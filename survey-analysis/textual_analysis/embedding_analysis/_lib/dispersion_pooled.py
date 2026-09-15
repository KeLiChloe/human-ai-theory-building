"""Pooled within-group dispersion Pre|Post combined figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_legacy import (
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    THRESHOLD_QUANTILE,
    available_embedding_columns,
    discover_phase_task_dirs,
    plot_within_group_dispersion_pooled_combined,
    read_anonymized_embeddings_wide,
    save_within_group_dispersion_pooled_combined,
)
from semantic_map import load_phase_bundles


def run(
    embeddings_root: Path | None = None,
    embedding_col: str | None = None,
    *,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> Path:
    """Write within_group_dispersion.{png,svg} only."""
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
    outpath = save_within_group_dispersion_pooled_combined(root, phase_bundles)
    print(f"Saved within-group dispersion pooled combined: {outpath}")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pooled Pre|Post within-group dispersion "
            "(Humans|GenAI stacked over PhD|Senior|GenAI)."
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


__all__ = [
    "main",
    "plot_within_group_dispersion_pooled_combined",
    "run",
    "save_within_group_dispersion_pooled_combined",
]


if __name__ == "__main__":
    main()
