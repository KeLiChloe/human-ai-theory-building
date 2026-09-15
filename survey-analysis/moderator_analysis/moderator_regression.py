"""
Q8 moderator analysis: forecasting accuracy by forecaster gender, expert, and senior.

Sample: all human forecasters (PhD students + senior scientists), n = 73.

Naming (do not conflate):
  - Expert: topic publications — ≥1 peer-reviewed article on racial or gender inequality.
  - Senior: job rank — Senior Scientist (group id 1) vs PhD Student (group id 0).

Produces a moderator results table PDF in moderator_analysis/outputs/.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, t as t_dist

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TEXTUAL_DIR = ROOT / "textual_analysis"
for p in (ROOT, TEXTUAL_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from common.stats_utils import (  # noqa: E402
    binary_split_membership,
    bootstrap_ci_half_width,
    gender_split_membership,
    p_value_welch_ttest,
    p_value_welch_ttest_one_sided,
)
from viz_style import SIG_LEVEL_LEGEND, draw_sig_footnote, significance_label  # noqa: E402

CSV_PATH = ROOT / "anonymous_survey_data.csv"
ME_ML_PATH = ROOT / "forecasts" / "ML_results_main_effects.json"
SOI_ML_PATH = ROOT / "forecasts" / "ML_results_second_order_interactions.json"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

HUMAN_GROUP_IDS = frozenset({"0", "1"})
SIGN_MAP = {"+": 1, "-": -1}

MOD_COLOR = {"high": "#C44E52", "low": "#4C72B0"}
MOD_LABELS = {
    "female": ("Female", "Male"),
    "expert": ("≥1 publication", "No publications"),
    "senior": ("Senior Scientist", "PhD Student"),
}
ROW_LABELS = {
    "female": "Female contributor",
    "expert": "Expert (inequality-related topic publication)",
    "senior": "Seniority (Senior Scientist vs. PhD Student)",
}
TABLE_MODERATOR_KEYS = ("female", "expert", "senior")
MODERATOR_SPECS = [
    ("female", "Female contributor", "female"),
    ("expert", "Expert (inequality-related topic publication)", "expert"),
]
PREREG_PANEL_SPECS = [
    ("female", "cos_race", "Racial Inequality", "Female Contributor — Racial Inequality Task"),
    ("female", "cos_gender", "Gender Inequality", "Female Contributor — Gender Inequality Task"),
    ("expert", "cos_race", "Racial Inequality", "Expert — Racial Inequality Task"),
    ("expert", "cos_gender", "Gender Inequality", "Expert — Gender Inequality Task"),
]
OLS_PREDICTORS = [
    ("female", "female"),
    ("expert", "expert"),
    ("senior", "senior"),
]
OLS_FOOTNOTE = (
    "OLS univariate regressions, one-sided tests "
    "(H1: moderator-high group > moderator-low group)."
)
YLIM = (-0.05, 1.05)
YLABEL = "Prediction Accuracy"

TASK_SPECS = {
    "race": {
        "acc_key": "cos_race",
        "title": "Racial Inequality Task",
        "rows": [
            ("female",),
            ("expert",),
            ("senior",),
        ],
    },
    "gender": {
        "acc_key": "cos_gender",
        "title": "Gender Inequality Task",
        "rows": [
            ("female",),
            ("expert",),
            ("senior",),
        ],
    },
}

ANALYSIS_SPECS = {
    "main_effects": {
        "label": "Main Effects",
        "filename_suffix": "main_effects",
    },
    "soi": {
        "label": "Second-Order Interactions",
        "filename_suffix": "soi",
    },
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 900,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass
class HumanRecord:
    cos_race: float
    cos_gender: float
    is_female: bool
    is_male: bool
    race_pub_count: float
    gender_pub_count: float
    is_senior: bool
    is_expert: bool  # topic_expert==1 (human-only; -1=GenAI N/A, ignored)

    @property
    def race_background(self) -> bool:
        return self.race_pub_count > 0

    @property
    def gender_background(self) -> bool:
        return self.gender_pub_count > 0


def parse_pub_count(cell: str) -> float:
    s = cell.strip()
    if s and s.replace(".", "", 1).isdigit():
        return float(s)
    return 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def canon_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip(), b.strip())))


def parse_pair(cell: str, valid_features: set[str]) -> tuple[str, str] | None:
    cell = cell.strip()
    if not cell or "," not in cell:
        return None
    parts = [x.strip() for x in cell.split(",")]
    if len(parts) != 2 or parts[0] == parts[1]:
        return None
    if parts[0] not in valid_features or parts[1] not in valid_features:
        return None
    return canon_pair(parts[0], parts[1])


def _load_csv() -> tuple[list[str], list[list[str]]]:
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _demographic_cols(headers: list[str]) -> dict[str, int]:
    return {
        "senior": next(i for i, h in enumerate(headers) if "senior_1" in h),
        "gender": next(i for i, h in enumerate(headers) if h == "What is your gender? - Selected Choice"),
        "topic_expert": next(i for i, h in enumerate(headers) if h.strip() == "topic_expert"),
        "race_pub": next(
            i for i, h in enumerate(headers)
            if h.startswith(
                "Approximately how many peer-reviewed academic articles have you published on topics related to racial inequality"
            )
        ),
        "gender_pub": next(
            i for i, h in enumerate(headers)
            if h.startswith(
                "Approximately how many peer-reviewed academic articles have you published on topics related to gender inequality"
            )
        ),
    }


def _human_record_from_row(
    row: list[str],
    cols: dict[str, int],
    cos_race: float,
    cos_gender: float,
) -> HumanRecord:
    gender = row[cols["gender"]].strip()
    gid = row[cols["senior"]].strip()
    return HumanRecord(
        cos_race=cos_race,
        cos_gender=cos_gender,
        is_female=(gender == "Female"),
        is_male=(gender == "Male"),
        race_pub_count=parse_pub_count(row[cols["race_pub"]]),
        gender_pub_count=parse_pub_count(row[cols["gender_pub"]]),
        is_senior=(gid == "1"),
        is_expert=(row[cols["topic_expert"]].strip() == "1"),
    )


def build_binary_vector(q1_col, q3_col_map, row, feat_idx):
    vec = np.zeros(len(feat_idx))
    cell = row[q1_col].strip()
    if not cell:
        return None
    for feat in cell.split(","):
        feat = feat.strip()
        if feat not in feat_idx:
            continue
        sign_str = row[q3_col_map[feat]].strip() if feat in q3_col_map else ""
        vec[feat_idx[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec


def build_ml_binary_vector(signs_dict, feat_idx):
    vec = np.zeros(len(feat_idx))
    for feat, sign_str in signs_dict.items():
        if feat in feat_idx:
            vec[feat_idx[feat]] = SIGN_MAP.get(sign_str, 0)
    return vec


def load_main_effects_records() -> list[HumanRecord]:
    headers, data = _load_csv()
    cols = _demographic_cols(headers)

    with open(ME_ML_PATH) as f:
        ml_raw = json.load(f)
    ml_signs = {
        task: {e["feature"]: e["sign"] for e in sorted(entries, key=lambda x: x["rank"])}
        for task, entries in ml_raw.items()
    }

    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers
        if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feat_idx = {f: i for i, f in enumerate(features)}
    r1_col = next(i for i, h in enumerate(headers) if h.strip() == "Q Race.1")
    g1_col = next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.1")
    r3_cols = {
        re.sub(r"^Q Race\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers) if re.match(r"^Q Race\.3 \(sign\) - ", h)
    }
    g3_cols = {
        re.sub(r"^Q Gender\.3 \(sign\) - ", "", h): i
        for i, h in enumerate(headers) if re.match(r"^Q Gender\.3 \(sign\) - ", h)
    }

    ml_race = build_ml_binary_vector(ml_signs["race"], feat_idx)
    ml_gender = build_ml_binary_vector(ml_signs["gender"], feat_idx)

    records: list[HumanRecord] = []
    for row in data:
        if row[cols["senior"]].strip() not in HUMAN_GROUP_IDS:
            continue
        vr = build_binary_vector(r1_col, r3_cols, row, feat_idx)
        vg = build_binary_vector(g1_col, g3_cols, row, feat_idx)
        records.append(
            _human_record_from_row(
                row,
                cols,
                cosine_sim(vr, ml_race) if vr is not None else np.nan,
                cosine_sim(vg, ml_gender) if vg is not None else np.nan,
            )
        )
    return records


def load_soi_records() -> list[HumanRecord]:
    headers, data = _load_csv()
    cols = _demographic_cols(headers)

    with open(SOI_ML_PATH) as f:
        ml_raw = json.load(f)

    features = [
        re.sub(r"^Q Race\.2 \(rank\) - ", "", h)
        for h in headers
        if re.match(r"^Q Race\.2 \(rank\) - ", h)
    ]
    feature_set = set(features)
    pairs = list(combinations(sorted(features), 2))
    pair_idx = {p: i for i, p in enumerate(pairs)}

    r_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.8 (SOI, 3rd)"),
    ]
    r_sign_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Race.9 (SOI, sign, 3rd)"),
    ]
    g_pair_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.6 (SOI, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.7 (SOI, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.8 (SOI, 3rd)"),
    ]
    g_sign_cols = [
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 1st)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 2nd)"),
        next(i for i, h in enumerate(headers) if h.strip() == "Q Gender.9 (SOI, sign, 3rd)"),
    ]

    def _build_vec(pair_cols, sign_cols, row):
        vec = np.zeros(len(pairs))
        for pc, sc in zip(pair_cols, sign_cols):
            p = parse_pair(row[pc], feature_set)
            if p is None:
                continue
            vec[pair_idx[p]] = SIGN_MAP.get(row[sc].strip(), 0)
        return vec

    def _build_ml(entries):
        vec = np.zeros(len(pairs))
        for e in entries:
            p = canon_pair(e["feature_1"], e["feature_2"])
            if p in pair_idx:
                vec[pair_idx[p]] = SIGN_MAP.get(e["sign"], 0)
        return vec

    ml_race = _build_ml(ml_raw["race"])
    ml_gender = _build_ml(ml_raw["gender"])

    records: list[HumanRecord] = []
    for row in data:
        if row[cols["senior"]].strip() not in HUMAN_GROUP_IDS:
            continue
        hr = _build_vec(r_pair_cols, r_sign_cols, row)
        hg = _build_vec(g_pair_cols, g_sign_cols, row)
        records.append(
            _human_record_from_row(
                row,
                cols,
                cosine_sim(hr, ml_race),
                cosine_sim(hg, ml_gender),
            )
        )
    return records


def group_values(records: list[HumanRecord], acc_key: str, mod_key: str, high: bool) -> list[float]:
    out = []
    for r in records:
        acc = getattr(r, acc_key)
        if np.isnan(acc):
            continue
        if mod_key == "female":
            in_group = gender_split_membership(
                is_female=r.is_female, is_male=r.is_male, want_female=high,
            )
        elif mod_key == "expert":
            in_group = binary_split_membership(in_high_group=r.is_expert, want_high=high)
        elif mod_key == "senior":
            in_group = r.is_senior if high else not r.is_senior
        else:
            raise ValueError(mod_key)
        if in_group:
            out.append(acc)
    return out


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": np.nan, "sims": []}
    return {"n": len(values), "mean": float(np.mean(values)), "sims": values}


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - rank
    if df <= 0:
        nan = np.full(k, np.nan)
        return beta, nan, nan
    mse = float(resid @ resid) / df
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_stats), df)
    return beta, se, p_vals


def ols_hc1_inference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with HC1 heteroskedasticity-robust standard errors."""
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - rank
    if df <= 0:
        nan = np.full(k, np.nan)
        return beta, nan, nan
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for i in range(n):
        xi = X[i : i + 1].T
        meat += float(resid[i] ** 2) * (xi @ xi.T)
    cov = (n / df) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_stats), df)
    return beta, se, p_vals


def predictor_value(r: HumanRecord, pred_key: str) -> float:
    if pred_key == "female":
        return float(r.is_female)
    if pred_key == "expert":
        return float(r.is_expert)
    if pred_key == "senior":
        return float(r.is_senior)
    raise ValueError(pred_key)


def run_univariate_regression(
    records: list[HumanRecord],
    acc_key: str,
    pred_key: str,
) -> dict:
    rows = []
    for r in records:
        acc = getattr(r, acc_key)
        if np.isnan(acc):
            continue
        rows.append((acc, predictor_value(r, pred_key)))
    if len(rows) < 4:
        return {}
    y = np.array([t[0] for t in rows])
    x = np.array([t[1] for t in rows])
    X = np.column_stack([np.ones(len(rows)), x])
    beta, se, p_vals = ols_hc1_inference(X, y)
    t_stat = beta[1] / se[1]
    df = len(rows) - 2
    p_one_sided = float(t_dist.sf(t_stat, df))
    return {
        "intercept": {"coef": float(beta[0]), "se": float(se[0]), "p": float(p_vals[0])},
        pred_key: {
            "coef": float(beta[1]),
            "se": float(se[1]),
            "p": float(p_vals[1]),
            "p_one_sided": p_one_sided,
        },
    }


def run_joint_moderator_regression(
    records: list[HumanRecord],
    acc_key: str,
    pred_keys: tuple[str, ...] = TABLE_MODERATOR_KEYS,
) -> dict:
    """OLS of accuracy on Female + Expert + Senior (HC1)."""
    y_list: list[float] = []
    x_rows: list[list[float]] = []
    for r in records:
        acc = getattr(r, acc_key)
        if np.isnan(acc):
            continue
        y_list.append(float(acc))
        x_rows.append([predictor_value(r, k) for k in pred_keys])
    n = len(y_list)
    k = 1 + len(pred_keys)
    if n <= k:
        return {}
    y = np.asarray(y_list, dtype=float)
    X = np.column_stack([np.ones(n), np.asarray(x_rows, dtype=float)])
    beta, se, p_vals = ols_hc1_inference(X, y)
    df = n - X.shape[1]
    out: dict = {
        "n": n,
        "intercept": {
            "coef": float(beta[0]),
            "se": float(se[0]),
            "p": float(p_vals[0]),
        },
    }
    for j, key in enumerate(pred_keys, start=1):
        t_stat = beta[j] / se[j]
        out[key] = {
            "coef": float(beta[j]),
            "se": float(se[j]),
            "p": float(p_vals[j]),
            "p_one_sided": float(t_dist.sf(t_stat, df)),
        }
    return out


def run_pooled_multivariable_regression(
    pred_keys: tuple[str, ...] = TABLE_MODERATOR_KEYS,
) -> dict:
    """
    Pooled multivariable OLS across four task × forecast-type cells:

        Accuracy_it = β0 + β1 Female_i + β2 Expert_i + β3 Senior_i + γ_t + ε_it

    γ_t are fixed effects for the four task-by-effect combinations; standard
    errors are clustered by forecaster (CR1).
    """
    me_records = load_main_effects_records()
    soi_records = load_soi_records()
    if len(me_records) != len(soi_records):
        raise ValueError("Main-effects and SOI human samples differ in length")

    cell_specs = (
        (me_records, "cos_race"),
        (me_records, "cos_gender"),
        (soi_records, "cos_race"),
        (soi_records, "cos_gender"),
    )
    y_list: list[float] = []
    x_rows: list[list[float]] = []
    fe_ids: list[int] = []
    clusters: list[int] = []
    for person_id, (r_me, r_soi) in enumerate(zip(me_records, soi_records)):
        # Demographics are person-level; either record is fine.
        for cell_id, (records, acc_key) in enumerate(cell_specs):
            r = records[person_id]
            acc = getattr(r, acc_key)
            if np.isnan(acc):
                continue
            y_list.append(float(acc))
            x_rows.append([predictor_value(r_me, k) for k in pred_keys])
            fe_ids.append(cell_id)
            clusters.append(person_id)

    n = len(y_list)
    n_pred = len(pred_keys)
    n_cells = len(cell_specs)
    # Intercept + predictors + (n_cells - 1) FE dummies (omit cell 0).
    k = 1 + n_pred + (n_cells - 1)
    if n <= k:
        return {}

    y = np.asarray(y_list, dtype=float)
    fe = np.asarray(fe_ids, dtype=int)
    fe_dummies = np.column_stack([(fe == j).astype(float) for j in range(1, n_cells)])
    X = np.column_stack([
        np.ones(n),
        np.asarray(x_rows, dtype=float),
        fe_dummies,
    ])
    beta, se, p_vals, g = ols_cluster_inference(X, y, np.asarray(clusters, dtype=int))
    df = max(g - 1, 1)
    out: dict = {
        "n_obs": n,
        "n_clusters": g,
        "intercept": {
            "coef": float(beta[0]),
            "se": float(se[0]),
            "p": float(p_vals[0]),
        },
    }
    for j, key in enumerate(pred_keys, start=1):
        t_stat = beta[j] / se[j]
        out[key] = {
            "coef": float(beta[j]),
            "se": float(se[j]),
            "p": float(p_vals[j]),
            "p_one_sided": float(t_dist.sf(t_stat, df)),
        }
    return out


def ols_cluster_inference(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """OLS with cluster-robust (CR1) standard errors; return beta, se, two-sided p, G."""
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    uniq, inv = np.unique(clusters, return_inverse=True)
    g = len(uniq)
    if g <= k or n <= k:
        nan = np.full(k, np.nan)
        return beta, nan, nan, g

    meat = np.zeros((k, k))
    for j in range(g):
        idx = inv == j
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)

    xtx_inv = np.linalg.inv(X.T @ X)
    # CR1 finite-sample correction.
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - rank))
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    df = max(g - 1, 1)
    t_stats = beta / se
    p_vals = 2 * t_dist.sf(np.abs(t_stats), df)
    return beta, se, p_vals, g


def run_pooled_univariate_regression(
    records: list[HumanRecord],
    pred_key: str,
) -> dict:
    """Pool Race+Gender accuracies (two rows per person); cluster SE by person."""
    y_list: list[float] = []
    x_list: list[float] = []
    c_list: list[int] = []
    for person_id, r in enumerate(records):
        x = predictor_value(r, pred_key)
        for acc_key in ("cos_race", "cos_gender"):
            acc = getattr(r, acc_key)
            if np.isnan(acc):
                continue
            y_list.append(float(acc))
            x_list.append(float(x))
            c_list.append(person_id)
    if len(y_list) < 4:
        return {}
    y = np.asarray(y_list, dtype=float)
    x = np.asarray(x_list, dtype=float)
    clusters = np.asarray(c_list, dtype=int)
    X = np.column_stack([np.ones(len(y_list)), x])
    beta, se, p_vals, g = ols_cluster_inference(X, y, clusters)
    df = max(g - 1, 1)
    t_stat = beta[1] / se[1]
    p_one_sided = float(t_dist.sf(t_stat, df))
    return {
        "n_obs": len(y_list),
        "n_clusters": g,
        "intercept": {"coef": float(beta[0]), "se": float(se[0]), "p": float(p_vals[0])},
        pred_key: {
            "coef": float(beta[1]),
            "se": float(se[1]),
            "p": float(p_vals[1]),
            "p_one_sided": p_one_sided,
        },
    }


def pooled_accuracy(r: HumanRecord) -> float:
    """Mean of racial- and gender-inequality accuracies for one person."""
    vals = [v for v in (r.cos_race, r.cos_gender) if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def pooled_group_values(
    records: list[HumanRecord],
    mod_key: str,
    high: bool,
) -> list[float]:
    """Person-level pooled accuracy, split by moderator group."""
    out: list[float] = []
    for r in records:
        acc = pooled_accuracy(r)
        if np.isnan(acc):
            continue
        if mod_key == "female":
            in_group = gender_split_membership(
                is_female=r.is_female, is_male=r.is_male, want_female=high,
            )
        elif mod_key == "expert":
            in_group = binary_split_membership(
                in_high_group=r.is_expert, want_high=high,
            )
        elif mod_key == "senior":
            in_group = r.is_senior if high else not r.is_senior
        else:
            raise ValueError(mod_key)
        if in_group:
            out.append(acc)
    return out


def collect_pooled_panel_rows() -> list[dict]:
    """Pooled Race+Gender rows within Main effects and within SOI (Welch)."""
    pools = (
        ("main_effects", "Main effects", load_main_effects_records),
        ("soi", "Second-order interactions", load_soi_records),
    )
    rows: list[dict] = []
    for analysis_key, analysis_label, loader in pools:
        records = loader()
        for mod_key in TABLE_MODERATOR_KEYS:
            hi = summary_stats(pooled_group_values(records, mod_key, True))
            lo = summary_stats(pooled_group_values(records, mod_key, False))
            p_val, _ = welch_test_for_moderator(mod_key, hi["sims"], lo["sims"])
            beta = (
                float(hi["mean"] - lo["mean"])
                if np.isfinite(hi["mean"]) and np.isfinite(lo["mean"])
                else np.nan
            )
            rows.append({
                "analysis": analysis_label,
                "analysis_key": analysis_key,
                "moderator": _latex_moderator_label(mod_key),
                "mod_key": mod_key,
                "beta": beta,
                "p_one_sided": p_val,
                "n_high": hi["n"],
                "n_low": lo["n"],
                "mean_high": hi["mean"],
                "mean_low": lo["mean"],
            })
    return rows


def ols_test_for_moderator(
    records: list[HumanRecord],
    acc_key: str,
    mod_key: str,
) -> tuple[float, float]:
    """One-sided OLS p-value and β for moderator-high > moderator-low."""
    reg = run_univariate_regression(records, acc_key, mod_key)
    if mod_key not in reg:
        return np.nan, np.nan
    stats = reg[mod_key]
    return stats["p_one_sided"], stats["coef"]


def welch_test_for_moderator(mod_key: str, hi: list[float], lo: list[float]) -> tuple[float, str]:
    return (
        p_value_welch_ttest_one_sided(hi, lo, alternative="greater"),
        "one-sided (high > low)",
    )


def welch_p_label(mod_key: str, p_val: float) -> str:
    if not np.isfinite(p_val):
        return "p1 = n/a"
    return f"p1 = {p_val:.3f}"


def ols_p_label(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "p1 = n/a"
    return f"p1 = {p_val:.3f}"


def _append_binary_row(
    rows: list[dict],
    *,
    analysis: str,
    acc_key: str,
    task_label: str,
    mod_key: str,
    mod_name: str,
    records: list[HumanRecord],
    prereg_q8: bool,
) -> None:
    labels = MOD_LABELS.get(mod_key, MOD_LABELS["expert"])
    hi = summary_stats(group_values(records, acc_key, mod_key, True))
    lo = summary_stats(group_values(records, acc_key, mod_key, False))
    p_val, test_kind = welch_test_for_moderator(mod_key, hi["sims"], lo["sims"])
    rows.append({
        "analysis": analysis,
        "prereg_q8": prereg_q8,
        "task": task_label,
        "moderator": mod_name,
        "mod_key": mod_key,
        "group_high": labels[0],
        "group_low": labels[1],
        "n_high": hi["n"],
        "n_low": lo["n"],
        "mean_high": hi["mean"],
        "mean_low": lo["mean"],
        "welch_p": p_val,
        "welch_test": test_kind,
    })


def collect_results(records: list[HumanRecord], *, analysis: str) -> list[dict]:
    tasks = [
        ("cos_race", "Racial Inequality"),
        ("cos_gender", "Gender Inequality"),
    ]
    prereg_pairs = {(m, a) for m, a, _, _ in PREREG_PANEL_SPECS}
    rows = []

    for acc_key, task_label in tasks:
        for mod_key, mod_name, _label_key in MODERATOR_SPECS:
            is_prereg = (mod_key, acc_key) in prereg_pairs
            _append_binary_row(
                rows,
                analysis=analysis,
                acc_key=acc_key,
                task_label=task_label,
                mod_key=mod_key,
                mod_name=mod_name,
                records=records,
                prereg_q8=is_prereg,
            )

        for pred_key, pred_label in OLS_PREDICTORS:
            reg = run_univariate_regression(records, acc_key, pred_key)
            if pred_key not in reg:
                continue
            stats = reg[pred_key]
            rows.append({
                "analysis": analysis,
                "prereg_q8": False,
                "task": task_label,
                "moderator": f"OLS (univariate):{pred_label}",
                "group_high": "",
                "group_low": "",
                "n_high": "",
                "n_low": len([r for r in records if not np.isnan(getattr(r, acc_key))]),
                "mean_high": stats["coef"],
                "mean_low": stats["se"],
                "welch_p_two_sided": stats["p"],
            })

        for pub_key, pub_label in [
            ("race_pub_count", "race publication count"),
            ("gender_pub_count", "gender publication count"),
        ]:
            acc_vals, pub_vals = [], []
            for r in records:
                acc = getattr(r, acc_key)
                if np.isnan(acc):
                    continue
                acc_vals.append(acc)
                pub_vals.append(getattr(r, pub_key))
            rho, p_sp = spearmanr(acc_vals, pub_vals)
            rows.append({
                "analysis": analysis,
                "prereg_q8": False,
                "task": task_label,
                "moderator": f"Spearman: {pub_label}",
                "group_high": "",
                "group_low": "",
                "n_high": "",
                "n_low": len(acc_vals),
                "mean_high": float(rho),
                "mean_low": np.nan,
                "welch_p_two_sided": float(p_sp),
            })
    return rows


def collect_figure_panel_rows() -> list[dict]:
    """One row per task × forecast type from joint multivariate OLS."""
    panel_specs = [
        ("race", "main_effects", load_main_effects_records),
        ("race", "soi", load_soi_records),
        ("gender", "main_effects", load_main_effects_records),
        ("gender", "soi", load_soi_records),
    ]
    rows: list[dict] = []
    for task_key, analysis_key, loader in panel_specs:
        records = loader()
        task_spec = TASK_SPECS[task_key]
        analysis_spec = ANALYSIS_SPECS[analysis_key]
        acc_key = task_spec["acc_key"]
        reg = run_joint_moderator_regression(records, acc_key)
        row: dict = {
            "task": task_spec["title"],
            "analysis": analysis_spec["label"],
            "analysis_key": analysis_key,
            "n_model": reg.get("n", np.nan),
            # Joint OLS: intercept + Female + Expert + Senior
            "df": float(reg["n"] - (1 + len(TABLE_MODERATOR_KEYS)))
            if np.isfinite(reg.get("n", np.nan))
            else np.nan,
        }
        for mod_key in TABLE_MODERATOR_KEYS:
            stats = reg.get(mod_key, {})
            row[f"beta_{mod_key}"] = stats.get("coef", np.nan)
            row[f"se_{mod_key}"] = stats.get("se", np.nan)
            row[f"p_{mod_key}"] = stats.get("p_one_sided", np.nan)
        rows.append(row)
    return rows


def collect_pooled_multivariable_rows() -> list[dict]:
    """Three partial associations from the pooled multivariable model."""
    reg = run_pooled_multivariable_regression()
    rows: list[dict] = []
    for mod_key in TABLE_MODERATOR_KEYS:
        stats = reg.get(mod_key, {})
        rows.append({
            "mod_key": mod_key,
            "moderator": _latex_moderator_label(mod_key),
            "beta": stats.get("coef", np.nan),
            "se": stats.get("se", np.nan),
            "p": stats.get("p_one_sided", np.nan),
            "n_obs": reg.get("n_obs", np.nan),
            "n_clusters": reg.get("n_clusters", np.nan),
        })
    return rows


def _latex_moderator_label(mod_key: str) -> str:
    return {
        "female": "Female",
        "expert": "Expert",
        "senior": "Senior",
    }[mod_key]


def _latex_task_label(task_title: str) -> str:
    return (
        task_title
        .replace(" Task", "")
        .replace("Inequality", "inequality")
    )


def _latex_forecast_type_label(analysis_label: str) -> str:
    return {
        "Main Effects": "Main effects",
        "Second-Order Interactions": "Interactions",
    }.get(analysis_label, analysis_label)


def _latex_sig_superscript(p: float) -> str:
    sig = significance_label(p)
    if sig in ("NS", "n/a"):
        return ""
    return f"$^{{{sig}}}$"


def _latex_format_p(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def _latex_format_beta(beta: float, p: float, se: float = np.nan, df: float = np.nan) -> str:
    if not np.isfinite(beta):
        return "---"
    stars = _latex_sig_superscript(p)
    if np.isfinite(se) and np.isfinite(df) and df > 0 and se >= 0:
        tcrit = float(t_dist.ppf(0.975, df))
        lo = beta - tcrit * se
        hi = beta + tcrit * se

        def _one(v: float) -> str:
            if v < 0:
                return f"-{abs(v):.3f}"
            return f"{v:.3f}"

        coef = f"-{abs(beta):.3f}" if beta < 0 else f"{beta:.3f}"
        return (
            r"\begin{tabular}{@{}c@{}}"
            + coef
            + stars
            + r"\\"
            + rf"$[{_one(lo)},\ {_one(hi)}]$"
            + r"\end{tabular}"
        )
    if beta < 0:
        return f"$-{abs(beta):.3f}${stars}"
    return f"{beta:.3f}{stars}"


def _latex_format_mean_n(mean: float, n: int) -> str:
    if not np.isfinite(mean):
        return "---"
    return f"{mean:.3f} ({n})"


def _compile_table_pdf(standalone_path: Path) -> Path:
    pdf_path = standalone_path.with_suffix(".pdf")
    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError(
            "tectonic not found; install with: brew install tectonic"
        )
    subprocess.run(
        [tectonic, standalone_path.name],
        cwd=standalone_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF not produced: {pdf_path}")
    return pdf_path


def write_figure_results_pdf(out_dir: Path | None = None) -> Path:
    """Build Nature-style moderator table PNG (+ SVG); no ``.tex`` left in ``out_dir``."""
    target = out_dir or OUT_DIR
    target.mkdir(exist_ok=True)
    rows = collect_figure_panel_rows()

    stem = "moderator_regression"

    # Group rows by task for multirow + midrule layout.
    by_task: list[tuple[str, list[dict]]] = []
    for row in rows:
        task = _latex_task_label(str(row["task"]))
        if not by_task or by_task[-1][0] != task:
            by_task.append((task, [row]))
        else:
            by_task[-1][1].append(row)

    body_lines = [
        "% Auto-generated by moderator_analysis/moderator_regression.py",
        r"\label{tab:moderator_regression}",
        "",
        r"\begin{threeparttable}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{8pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Prediction type}",
        r"& \textbf{\textit{Female} ($\beta_1$)}",
        r"& \textbf{\textit{Expertise} ($\beta_2$)}",
        r"& \textbf{\textit{Seniority} ($\beta_3$)} \\",
        r"\midrule",
    ]

    for task_idx, (task, task_rows) in enumerate(by_task):
        n_task = len(task_rows)
        for i, row in enumerate(task_rows):
            analysis_display = _latex_forecast_type_label(str(row["analysis"]))
            df = float(row.get("df", np.nan))
            beta_cells = [
                _latex_format_beta(
                    float(row[f"beta_{k}"]),
                    float(row[f"p_{k}"]),
                    se=float(row[f"se_{k}"]),
                    df=df,
                )
                for k in TABLE_MODERATOR_KEYS
            ]
            task_cell = (
                rf"\multirow{{{n_task}}}{{*}}{{{task}}}" if i == 0 else ""
            )
            body_lines.append(
                f"{task_cell} & {analysis_display} & "
                f"{beta_cells[0]} & {beta_cells[1]} & {beta_cells[2]} \\\\"
            )
        if task_idx < len(by_task) - 1:
            body_lines.append(r"\midrule")

    body_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{threeparttable}",
        "",
    ])

    tex_body = "\n".join(body_lines)
    standalone = "\n".join([
        "% Compile:",
        f"%   tectonic {stem}_standalone.tex",
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{threeparttable}",
        r"\usepackage{amsmath}",
        "",
        r"\begin{document}",
        rf"\input{{{stem}.tex}}",
        r"\end{document}",
        "",
    ])

    dest_png = target / f"{stem}.png"
    dest_svg = target / f"{stem}.svg"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tex_body_path = tmp_dir / f"{stem}.tex"
        tex_standalone_path = tmp_dir / f"{stem}_standalone.tex"
        tex_body_path.write_text(tex_body, encoding="utf-8")
        tex_standalone_path.write_text(standalone, encoding="utf-8")

        pdf_path = _compile_table_pdf(tex_standalone_path)
        png_tmp = pdf_path.with_suffix(".png")
        svg_tmp = pdf_path.with_suffix(".svg")

        from common.latex_table_pdf import _pdf_to_png, _pdf_to_svg, _trim_png_whitespace

        _pdf_to_png(pdf_path, png_tmp, dpi=400)
        _trim_png_whitespace(png_tmp)
        try:
            _pdf_to_svg(pdf_path, svg_tmp)
        except Exception:
            pass
        pdf_path.unlink(missing_ok=True)

        shutil.move(str(png_tmp), str(dest_png))
        if svg_tmp.is_file():
            shutil.move(str(svg_tmp), str(dest_svg))

    # Drop leftovers from older runs / names.
    for stale in (
        target / f"{stem}.tex",
        target / f"{stem}_standalone.tex",
        target / f"{stem}_standalone.png",
        target / f"{stem}_standalone.svg",
        target / "moderator_forecasting_accuracy_table.tex",
        target / "moderator_forecasting_accuracy_table_standalone.tex",
        target / "moderator_forecasting_accuracy_table_standalone.png",
        target / "moderator_forecasting_accuracy_table_standalone.svg",
    ):
        stale.unlink(missing_ok=True)

    return dest_png


def _panel_ylim(means, errs):
    err_vals = [0 if not np.isfinite(e) else e for e in errs]
    data_lo = min(m - e for m, e in zip(means, err_vals))
    data_hi = max(m + e for m, e in zip(means, err_vals))
    pad = max((data_hi - data_lo) * 0.18, 0.04)
    ymin = max(data_lo - pad, YLIM[0])
    ymax = min(data_hi + pad + (data_hi - data_lo) * 0.12, YLIM[1])
    return ymin, ymax


def plot_moderator_panel(
    ax,
    hi: dict,
    lo: dict,
    labels: tuple[str, str],
    mod_key: str,
    records: list[HumanRecord],
    acc_key: str,
):
    groups = [f"{labels[0]} (n={hi['n']})", f"{labels[1]} (n={lo['n']})"]
    means = [hi["mean"], lo["mean"]]
    errs = [bootstrap_ci_half_width(hi["sims"]), bootstrap_ci_half_width(lo["sims"])]
    colors = [MOD_COLOR["high"], MOD_COLOR["low"]]
    ylo, yhi = _panel_ylim(means, errs)
    bars = ax.bar(
        groups, means, yerr=errs, color=colors, width=0.58,
        capsize=4, error_kw={"linewidth": 1.0},
        edgecolor="white", linewidth=0.8,
    )
    for bar, m, err in zip(bars, means, errs):
        y_text = m + (0 if not np.isfinite(err) else err) + (yhi - ylo) * 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(ylo, yhi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    p_val, beta = ols_test_for_moderator(records, acc_key, mod_key)
    sig = significance_label(p_val)
    if sig in ("NS", "n/a"):
        annot = sig
    else:
        annot = f"{sig} (β = {beta:.2f}, {ols_p_label(p_val)})"
    ax.text(
        0.98, 0.96, annot,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="#C62828" if sig not in ("NS", "n/a") else "#555555",
        fontweight="bold" if sig not in ("NS", "n/a") else "normal",
    )


def _plot_panel_figure(
    *,
    task_key: str,
    analysis_key: str,
    records: list[HumanRecord],
    out_dir: Path = OUT_DIR,
) -> Path:
    task_spec = TASK_SPECS[task_key]
    analysis_spec = ANALYSIS_SPECS[analysis_key]
    acc_key = task_spec["acc_key"]

    n_panels = len(task_spec["rows"])
    fig_h = 3.2 * n_panels + 1.2
    fig, axes = plt.subplots(n_panels, 1, figsize=(5.8, fig_h))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"{task_spec['title']} — {analysis_spec['label']}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    for row_i, (mod_key,) in enumerate(task_spec["rows"]):
        ax = axes[row_i]
        labels = MOD_LABELS[mod_key if mod_key in MOD_LABELS else "expert"]
        hi = summary_stats(group_values(records, acc_key, mod_key, True))
        lo = summary_stats(group_values(records, acc_key, mod_key, False))
        plot_moderator_panel(ax, hi, lo, labels, mod_key, records, acc_key)
        ax.set_title(ROW_LABELS[mod_key], fontsize=10, fontweight="bold", pad=10)

    fig.supylabel(YLABEL, fontsize=11, fontweight="bold", x=0.04)

    legend_handles = [
        mpatches.Patch(facecolor=MOD_COLOR["high"], edgecolor="white", label="Moderator-high group"),
        mpatches.Patch(facecolor=MOD_COLOR["low"], edgecolor="white", label="Moderator-low group"),
        plt.Line2D([0], [0], color="none", label="Bars: mean; whiskers: 95% bootstrap CI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.14),
    )

    footnotes = (OLS_FOOTNOTE, SIG_LEVEL_LEGEND)
    draw_sig_footnote(fig, y=0.07, text=footnotes, line_step=0.024)
    plt.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.20, hspace=0.42)

    filename = f"moderator_regression_{task_key}_{analysis_spec['filename_suffix']}.png"
    out = out_dir / filename
    plt.savefig(out, dpi=900, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    return out


def plot_all_task_figures(out_dir: Path | None = None) -> list[Path]:
    target = out_dir or OUT_DIR
    me_records = load_main_effects_records()
    soi_records = load_soi_records()
    paths = []
    for task_key in ("race", "gender"):
        paths.append(
            _plot_panel_figure(
                task_key=task_key,
                analysis_key="main_effects",
                records=me_records,
                out_dir=target,
            )
        )
        paths.append(
            _plot_panel_figure(
                task_key=task_key,
                analysis_key="soi",
                records=soi_records,
                out_dir=target,
            )
        )
    return paths


def print_summary(rows: list[dict], records: list[HumanRecord], *, title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print("\n--- Q8 pre-registered tests ---")
    for r in rows:
        if not r.get("prereg_q8"):
            continue
        print(
            f"{r['task']:22} | {r['moderator']:35} | "
            f"{r['group_high']} (n={r['n_high']}, M={r['mean_high']:.3f}) vs "
            f"{r['group_low']} (n={r['n_low']}, M={r['mean_low']:.3f}) | "
            f"{welch_p_label(r['mod_key'], r['welch_p'])} ({r['welch_test']})"
        )
    print("\n--- OLS univariate (one moderator per model; all humans, n=73) ---")
    for r in rows:
        if r["moderator"].startswith("OLS (univariate):"):
            print(
                f"  {r['task']:22} {r['moderator']:32} "
                f"β={r['mean_high']:.4f} (SE={r['mean_low']:.4f}), p={r['welch_p_two_sided']:.4f}"
            )


def main() -> None:
    me_records = load_main_effects_records()
    soi_records = load_soi_records()
    n_phd = sum(1 for r in me_records if not r.is_senior)
    n_senior = sum(1 for r in me_records if r.is_senior)
    n_topic_expert = sum(1 for r in me_records if r.is_expert)
    print(
        f"Loaded {len(me_records)} humans "
        f"({n_phd} PhD students, {n_senior} seniors; "
        f"{n_topic_expert} topic experts)."
    )

    me_rows = collect_results(me_records, analysis="main_effects")
    soi_rows = collect_results(soi_records, analysis="second_order_interactions")

    png_path = write_figure_results_pdf()
    print_summary(
        me_rows,
        me_records,
        title="Main Effects Prediction Accuracy",
    )
    print_summary(
        soi_rows,
        soi_records,
        title="Second-Order Interactions",
    )
    print(f"\nSaved: {png_path}")


if __name__ == "__main__":
    main()
