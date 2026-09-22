"""Embedding parquet I/O and row stacking helpers."""

from __future__ import annotations

from analysis_legacy import (  # noqa: F401
    available_embedding_columns,
    load_phase_task_bundle,
    ordered_groups,
    parse_embedding_cell,
    stack_embeddings,
    with_collapsed_group,
)
