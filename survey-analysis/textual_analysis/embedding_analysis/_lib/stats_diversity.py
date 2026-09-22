"""Within-group diversity / MPWD statistics used by active figures."""

from __future__ import annotations

from analysis_legacy import (  # noqa: F401
    bootstrap_mpwd_ci,
    bootstrap_pooled_mpwd_ci,
    build_pairwise_inference_rows,
    ci_errorbar_offsets,
    cohens_d,
    comparison_pairs_for_groups,
    comparisons_for_metric,
    diversity_summary_for_group,
    find_medoid,
    format_diversity_comparison_line,
    group_centroid,
    group_metric_values,
    mean_pairwise_cosine_distance,
    mean_pairwise_cosine_distance_normalized,
    normalize_embedding_rows,
    p_value_paired_permutation_mpwd_post_lt_pre,
    p_value_permutation_mean_diff,
    p_value_permutation_mpwd_group_greater,
    p_value_pooled_paired_permutation_mpwd_post_lt_pre,
    pairwise_cosine_distance_group_series,
    run_diversity_inference_analysis,
    upper_triangle_values,
    welch_comparisons_for_distance,
)
