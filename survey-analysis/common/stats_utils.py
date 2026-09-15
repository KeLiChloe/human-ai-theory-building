"""Shared bootstrap CI and Welch t-test helpers for survey figures."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, ttest_ind, ttest_rel


def binary_split_membership(*, in_high_group: bool, want_high: bool) -> bool:
    """Membership for a single boolean moderator (e.g. has background vs not)."""
    return in_high_group if want_high else not in_high_group


def gender_split_membership(*, is_female: bool, is_male: bool, want_female: bool) -> bool:
    """Membership for Female-vs-Male splits (two distinct indicators, not one flag)."""
    return is_female if want_female else is_male


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci_half_width(
    sample_values,
    *,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> float:
    """Symmetric half-width for bar whiskers (max distance from mean to bootstrap bounds)."""
    arr = np.asarray(sample_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed, alpha=alpha)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    return max(mean - lo, hi - mean)


def p_value_welch_ttest(a, b) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    try:
        return float(ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def p_value_welch_ttest_one_sided(a, b, *, alternative: str = "greater") -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    try:
        return float(
            ttest_ind(
                arr_a,
                arr_b,
                equal_var=False,
                nan_policy="omit",
                alternative=alternative,
            ).pvalue
        )
    except Exception:
        return np.nan


def welch_test(a: dict, b: dict, values_key: str = "sims") -> float:
    return p_value_welch_ttest(a[values_key], b[values_key])


def p_value_paired_ttest(pre, post) -> float:
    arr_pre = np.asarray(pre, dtype=float)
    arr_post = np.asarray(post, dtype=float)
    mask = np.isfinite(arr_pre) & np.isfinite(arr_post)
    arr_pre = arr_pre[mask]
    arr_post = arr_post[mask]
    if len(arr_pre) < 2:
        return np.nan
    try:
        return float(ttest_rel(arr_pre, arr_post, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def p_value_paired_ttest_pairs(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return np.nan
    pre = [p[0] for p in pairs]
    post = [p[1] for p in pairs]
    return p_value_paired_ttest(pre, post)


def p_value_paired_one_sided_post_lt_pre(pre, post) -> float:
    """H1: post < pre (e.g. lower within-group distance post-data)."""
    arr_pre = np.asarray(pre, dtype=float)
    arr_post = np.asarray(post, dtype=float)
    mask = np.isfinite(arr_pre) & np.isfinite(arr_post)
    arr_pre = arr_pre[mask]
    arr_post = arr_post[mask]
    if len(arr_pre) < 2:
        return np.nan
    try:
        return float(
            ttest_rel(arr_pre, arr_post, alternative="greater").pvalue
        )
    except Exception:
        return np.nan


def p_value_exact_mcnemar_two_sided(n_b: int, n_c: int) -> float:
    """Two-sided exact McNemar test on discordant pair counts (b, c).

    Under H0, discordant outcomes are equiprobable: b ~ Binomial(b+c, 1/2).
    Returns 1.0 when there are no discordant pairs.
    """
    b = int(n_b)
    c = int(n_c)
    n = b + c
    if n <= 0:
        return 1.0
    try:
        return float(binomtest(b, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def p_value_aggregated_cosine_diff_permutation(
    vecs_a,
    vecs_b,
    ml_vec,
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test of |agg(A) − agg(B)|.

    Aggregated accuracy is cosine(sum of prediction vectors, ML benchmark).
    Under H0, group labels are exchangeable within the pooled set of
    contributors. Uses add-one smoothing:
    p = (#{|δ_perm| ≥ |δ_obs|} + 1) / (n_perm + 1).
    """
    a = np.asarray(vecs_a, dtype=float)
    b = np.asarray(vecs_b, dtype=float)
    ml = np.asarray(ml_vec, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[0] < 1:
        return np.nan
    if a.shape[1] != b.shape[1] or a.shape[1] != ml.shape[0]:
        return np.nan

    def _agg(vecs: np.ndarray) -> float:
        if vecs.shape[0] == 0:
            return np.nan
        s = np.sum(vecs, axis=0)
        denom = float(np.linalg.norm(s) * np.linalg.norm(ml))
        return float(np.dot(s, ml) / denom) if denom else np.nan

    obs_a = _agg(a)
    obs_b = _agg(b)
    if not np.isfinite(obs_a) or not np.isfinite(obs_b):
        return np.nan
    obs = abs(obs_a - obs_b)
    pool = np.vstack([a, b])
    n_a = a.shape[0]
    n = pool.shape[0]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        diff = abs(_agg(pool[perm[:n_a]]) - _agg(pool[perm[n_a:]]))
        if not np.isfinite(diff):
            continue
        if diff >= obs - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def p_value_paired_permutation_two_sided(
    pre,
    post,
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test on mean(post − pre).

    Under H0, randomly flips the sign of each paired difference (equivalent to
    swapping pre/post within pairs). Uses add-one smoothing:
    p = (#{|δ_perm| ≥ |δ_obs|} + 1) / (n_perm + 1).
    """
    arr_pre = np.asarray(pre, dtype=float)
    arr_post = np.asarray(post, dtype=float)
    mask = np.isfinite(arr_pre) & np.isfinite(arr_post)
    arr_pre = arr_pre[mask]
    arr_post = arr_post[mask]
    n = len(arr_pre)
    if n < 1:
        return np.nan
    diffs = arr_post - arr_pre
    obs = float(np.mean(diffs))
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.mean(signs * diffs))) >= abs(obs) - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)
