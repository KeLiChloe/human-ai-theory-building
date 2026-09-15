"""Path resolution and embedding-set discovery (package root = embedding_analysis/)."""

from __future__ import annotations

from analysis_legacy import (  # noqa: F401
    PACKAGE_ROOT,
    SCRIPT_DIR,
    DEFAULT_EMBEDDINGS_ROOT,
    VISUALIZATIONS_DIRNAME,
    batch_visualizations_root,
    batch_phase_dir,
    comparisons_pre_post_dir,
    discover_embedding_set_dirs,
    discover_phase_task_dirs,
    discover_pre_post_task_pairs,
    embedding_set_label,
    format_embedding_set_part,
    infer_embeddings_root,
    resolve_input_parquet,
    resolve_network_dir,
    resolve_network_outpath,
    resolve_output_dir,
    resolve_task_data_dir,
    safe_name,
    task_label_from_key,
    task_phase_slug,
)
