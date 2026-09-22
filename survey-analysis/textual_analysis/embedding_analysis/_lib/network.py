"""Semantic threshold network figures + per-set data CSVs + interactive HTML."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import normalize

from analysis_legacy import (
    ANALYSIS_SEED,
    COLLAPSED_PARTICIPANT_TYPE_COL,
    DEFAULT_EMBEDDING_COLUMNS,
    DEFAULT_EMBEDDINGS_ROOT,
    NETWORK_INTERACTIVE_COLLAPSED_HTML,
    NETWORK_INTERACTIVE_HTML,
    NETWORK_INTERACTIVE_INDEX,
    NETWORK_SUBDIR,
    PACKAGE_ROOT,
    PARTICIPANT_TYPE_COL,
    THRESHOLD_QUANTILE,
    available_embedding_columns,
    batch_visualizations_root,
    build_group_threshold_network_payload,
    discover_embedding_set_dirs,
    embedding_set_label,
    export_network_interactive_demos,
    infer_embeddings_root,
    load_theory_texts_for_embedding_set,
    plot_group_threshold_networks,
    resolve_network_dir,
    respondent_name_map_for_embeddings_root,
    run_network_html_for_embedding_set,
    semantic_neighbor_degree,
    stack_embeddings,
)


def collect_network_demo_jobs(
    embedding_set_dir: Path,
    embeddings_root: Path,
    embedding_col: str,
    *,
    seed: int = ANALYSIS_SEED,
    threshold_quantile: float = THRESHOLD_QUANTILE,
    name_map: dict[str, str] | None = None,
) -> list[tuple[dict, Path, dict]]:
    """Build payloads and index metadata without writing HTML yet."""
    df = pd.read_parquet(embedding_set_dir / "embeddings_wide.parquet")
    X = normalize(stack_embeddings(df, embedding_col))
    _, similarity_threshold = semantic_neighbor_degree(
        X,
        labels=df[PARTICIPANT_TYPE_COL].values,
        threshold_quantile=threshold_quantile,
    )
    label_text = embedding_set_label(embedding_set_dir)
    theory_by_name = load_theory_texts_for_embedding_set(embedding_set_dir)
    network_dir = resolve_network_dir(embeddings_root, embedding_set_dir, embedding_col)
    jobs: list[tuple[dict, Path, dict]] = []

    variants = (
        (
            PARTICIPANT_TYPE_COL,
            NETWORK_INTERACTIVE_HTML,
            "Three groups (PhDs, Senior Scientists, GenAI)",
        ),
        (
            COLLAPSED_PARTICIPANT_TYPE_COL,
            NETWORK_INTERACTIVE_COLLAPSED_HTML,
            "Two groups (Humans and GenAI)",
        ),
    )
    for group_col, html_name, variant_label in variants:
        payload = build_group_threshold_network_payload(
            df,
            X,
            similarity_threshold,
            threshold_quantile,
            label_text,
            seed=seed,
            group_col=group_col,
            theory_by_name=theory_by_name,
            name_map=name_map,
        )
        outpath = network_dir / html_name
        rel_href = outpath.relative_to(
            batch_visualizations_root(embeddings_root) / NETWORK_SUBDIR
        )
        parts = embedding_set_dir.relative_to(embeddings_root).parts
        task, theory_type, phase = parts[0], parts[1], parts[2]
        variant_key = (
            "collapsed"
            if group_col == COLLAPSED_PARTICIPANT_TYPE_COL
            else "three_groups"
        )
        jobs.append(
            (
                payload,
                outpath,
                {
                    "href": str(rel_href).replace("\\", "/"),
                    "task": task,
                    "theory_type": theory_type,
                    "phase": phase,
                    "variant_key": variant_key,
                    "title": label_text,
                    "subtitle": payload["title"],
                    "variant": variant_label,
                },
            )
        )
    return jobs


def refresh_network_interactive_index(
    embeddings_root: Path,
    embedding_col: str | None = None,
) -> Path:
    """Rebuild network/index.html gallery after batch network exports."""
    ni_pkg = PACKAGE_ROOT / "network"
    import sys

    if str(ni_pkg) not in sys.path:
        sys.path.insert(0, str(ni_pkg))
    from network_interactive.render import (  # noqa: E402
        render_network_index_html,
        render_network_interactive_html,
    )

    set_dirs = discover_embedding_set_dirs(str(embeddings_root))
    name_map = respondent_name_map_for_embeddings_root(
        embeddings_root, seed=ANALYSIS_SEED
    )
    all_jobs: list[tuple[dict, Path, dict]] = []
    for set_dir in set_dirs:
        cols = available_embedding_columns(
            pd.read_parquet(set_dir / "embeddings_wide.parquet"),
            [embedding_col] if embedding_col else DEFAULT_EMBEDDING_COLUMNS,
        )
        for col in cols:
            all_jobs.extend(
                collect_network_demo_jobs(
                    set_dir,
                    embeddings_root,
                    col,
                    name_map=name_map,
                )
            )

    all_index = [entry for _, _, entry in all_jobs]
    for payload, outpath, _ in all_jobs:
        render_network_interactive_html(payload, outpath, nav_entries=all_index)

    index_path = (
        batch_visualizations_root(embeddings_root)
        / NETWORK_SUBDIR
        / NETWORK_INTERACTIVE_INDEX
    )
    render_network_index_html(all_index, index_path)
    print(f"Gallery index: {index_path}")
    return index_path


def run(
    embedding_set: str | Path,
    *,
    refresh_index: bool | None = None,
) -> Path:
    """Write interactive network HTML for one set or a batch root."""
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
        print(f"\n=== network html · {set_label} ({embedding_set_dir}) ===")
        run_network_html_for_embedding_set(embedding_set_dir, embeddings_root)

    if refresh_index is None:
        refresh_index = is_batch
    if refresh_index:
        sample_df = pd.read_parquet(
            embedding_set_dirs[0] / "embeddings_wide.parquet"
        )
        embedding_col = available_embedding_columns(
            sample_df, DEFAULT_EMBEDDING_COLUMNS
        )[0]
        refresh_network_interactive_index(embeddings_root, embedding_col)

    out_root = batch_visualizations_root(embeddings_root) / NETWORK_SUBDIR
    print("\nDone.")
    print(f"Network HTML saved under: {out_root}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive semantic-threshold network HTML under network/."
    )
    parser.add_argument(
        "--embedding-set",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help=(
            "One embedding-set folder (with embeddings_wide.parquet), "
            "or a parent folder (default: embeddings_openai_text-embedding-3-large/) to process all sets."
        ),
    )
    parser.add_argument(
        "--refresh-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rebuild network/index.html (default: on for batch roots).",
    )
    args = parser.parse_args()
    run(args.embedding_set, refresh_index=args.refresh_index)


# Re-exports used by other modules / tests
__all__ = [
    "build_group_threshold_network_payload",
    "collect_network_demo_jobs",
    "export_network_interactive_demos",
    "main",
    "plot_group_threshold_networks",
    "refresh_network_interactive_index",
    "run",
    "run_network_html_for_embedding_set",
]


if __name__ == "__main__":
    main()
