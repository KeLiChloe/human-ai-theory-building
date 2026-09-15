"""Write collapsed HDBSCAN clustering CSVs for fig_core_tail_structure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PACKAGE = Path(__file__).resolve().parent
LIB = PACKAGE / "_lib"
TEXTUAL = PACKAGE.parent
SURVEY = TEXTUAL.parent
for p in (SURVEY, TEXTUAL, PACKAGE, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analysis_legacy import (  # noqa: E402
    DEFAULT_EMBEDDINGS_ROOT,
    discover_embedding_set_dirs,
    embedding_set_label,
    infer_embeddings_root,
    run_semantic_clustering_for_embedding_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write semantic_clustering_*_collapsed.csv beside each "
            "embeddings_wide.parquet (input for fig_core_tail_structure)."
        )
    )
    parser.add_argument(
        "--embedding-set",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help=(
            "One embedding-set folder, or a parent folder "
            "(default: embeddings_openai_text-embedding-3-large/)."
        ),
    )
    args = parser.parse_args()

    embedding_set = args.embedding_set
    batch_root = Path(embedding_set).expanduser()
    if not batch_root.is_absolute():
        batch_root = (Path.cwd() / batch_root).resolve()

    embedding_set_dirs = discover_embedding_set_dirs(str(embedding_set))
    is_batch = len(embedding_set_dirs) > 1
    embeddings_root = (
        batch_root if is_batch else infer_embeddings_root(embedding_set_dirs[0])
    )
    print(f"Processing {len(embedding_set_dirs)} embedding set(s).")
    for embedding_set_dir in embedding_set_dirs:
        set_label = embedding_set_label(embedding_set_dir)
        print(f"\n=== clustering · {set_label} ({embedding_set_dir}) ===")
        run_semantic_clustering_for_embedding_set(embedding_set_dir, embeddings_root)
    print("\nDone.")
    print(f"Clustering CSVs saved under each set in: {embeddings_root}")


if __name__ == "__main__":
    main()
