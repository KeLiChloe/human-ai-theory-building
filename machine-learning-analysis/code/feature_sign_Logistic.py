import json
import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import statsmodels.api as sm


from scipy import stats
from matplotlib.patches import Patch

import matplotlib as mpl
from tqdm import tqdm

# Nature-compatible Helvetica (seaborn set_theme may reset fonts; re-apply when plotting)
PLOT_FONT = "Helvetica"
PLOT_FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.framealpha": 1.0,
    "axes.unicode_minus": False,
}
mpl.rcParams.update(PLOT_FONT_RC)

# Paper feature names (spaces → underscores). Map internal column names → paper names.
PAPER_FEATURE_NAMES = {
    "number_of_authors": "num_authors",
    "num_authors": "num_authors",
    "female": "female",
    "female_mean": "female",
    "female_score_mean": "female",
    "asian": "asian",
    "asian_composition": "asian",
    "black": "black",
    "black_composition": "black",
    "white": "white",
    "white_composition": "white",
    "hispanic": "hispanic",
    "hispanic_composition": "hispanic",
    "hispanic_and_other": "hispanic",
    "social_science": "social_science",
    "social_sciences": "social_science",
    "natural_science": "natural_science",
    "natural_sciences": "natural_science",
    "engineering_and_technology": "engineering_and_technology",
    "authors_race_diversity_score": "authors_race_diversity_score",
    "paper_race_shannon_entropy": "authors_race_diversity_score",
    "country_race_diversity_score": "country_race_diversity_score",
    "country_race_shannon_entropy_mean": "country_race_diversity_score",
    "news_inequality_mentions_3_years": "news_inequality_mentions_3_years",
    "news_ineq_3yr_avg": "news_inequality_mentions_3_years",
    "paper_inequality_mentions_3_years": "paper_inequality_mentions_3_years",
    "acad_ineq_3yr_avg": "paper_inequality_mentions_3_years",
}

FEATURE_LABELS = PAPER_FEATURE_NAMES


def format_feature_label(feature_name):
    """Map internal feature names to paper underscore names (for plots)."""
    if not isinstance(feature_name, str):
        return feature_name
    if " * " in feature_name:
        parts = feature_name.split(" * ")
        return " * ".join(PAPER_FEATURE_NAMES.get(part, part) for part in parts)
    return PAPER_FEATURE_NAMES.get(feature_name, feature_name)


def infer_task_label(file_path, model_save_dir):
    context = f"{file_path} {model_save_dir}".lower()
    if "race" in context:
        return "Race Task"
    if "gender" in context:
        return "Gender Task"
    return "Task"


def artifact_output_dir(base_model_save_dir, add_SOI):
    """Figure output under …/main_effects or …/soi (``.pkl`` stays in ``model_save_dir``)."""
    sub = "soi" if add_SOI else "main_effects"
    out = os.path.join(base_model_save_dir, sub)
    os.makedirs(out, exist_ok=True)
    return out


def results_output_base(figures_dir, add_SOI):
    soi_suffix = "_soi" if add_SOI else ""
    return os.path.join(
        figures_dir,
        f"LR_sign_results{soi_suffix}",
    )


def save_experiment_results(figures_dir, add_SOI, coef_df, run_config):
    """Save ``run_config`` and coefficient ``ranking`` to a single JSON file."""
    output_base = results_output_base(figures_dir, add_SOI)
    ranked = coef_df.reindex(coef_df["Mean"].abs().sort_values(ascending=False).index)

    ranking = [
        {
            "rank": rank,
            "feature": row["Feature"],
            "mean_coef": round(float(row["Mean"]), 6),
            "lower95_ci": round(float(row["Lower95CI"]), 6),
            "upper95_ci": round(float(row["Upper95CI"]), 6),
            "mean_pval": round(float(row["MeanPval"]), 6),
            "stars": row["Stars"],
        }
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1)
    ]

    payload = {
        "run_config": run_config,
        "ranking": ranking,
    }
    json_path = f"{output_base}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("✅ Saved experiment results to:")
    print(f"   → {json_path}")


# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert ndarray back to DataFrame, keeping column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

def initial_screen_features_RF(X, y, threshold=0.01):
    """
    Select important features based on Random Forest importance scores.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Minimum importance score to retain a feature.

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Filter features based on importance threshold
    selected_features = importance_df[importance_df['Importance'] > threshold]['Feature']
    X_filtered = X[selected_features]

    print("\nFeature Importance:")
    print(importance_df)
    print(f"\nSelected {len(selected_features)} features based on importance > {threshold}")

    return X_filtered

def initial_screen_features_lasso(X, y, alpha=0.001):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        alpha (float): Regularization strength (LASSO parameter).

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(X, y)

    # Get coefficients and feature importance
    coefficients = lasso_model.coef_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    # Filter features based on non-zero coefficients
    selected_features = importance_df[importance_df['Coefficient'] != 0]['Feature']
    X_filtered = X[selected_features]

    print("\nLASSO Feature Coefficients:")
    print(importance_df[importance_df['Coefficient'] != 0])
    print(f"\nSelected {len(selected_features)} features with non-zero coefficients")

    return X_filtered



def tune_hyperparameters(X_train, y_train, subset_id, model_save_dir):
    """
    Tune Logistic Regression hyperparameters using GridSearchCV.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series or array): Training labels.
        subset_id (int): Subset index for tracking and saving results.
        model_save_dir (str): Directory to save best parameter files.

    Returns:
        dict: Best hyperparameters for Logistic Regression.
    """
    print(f"🔍 Tuning hyperparameters for subset {subset_id}...")

    # Define parameter grid
    param_grid = {
        "solver": ["liblinear", "lbfgs"],   # support both L1 & L2
        "max_iter": [500, 1000],
        "C": [0.01, 0.1, 0.3, 1, 3, 10],
        "class_weight": ["balanced"],
    }

    # Logistic Regression model
    base_model = LogisticRegression(random_state=42)

    # Stratified 5-fold CV (for balanced evaluation)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use ROC-AUC as the metric — more robust than accuracy for classification
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Extract best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"✅ Best params for subset {subset_id}: {best_params}")
    print(f"   Mean CV ROC-AUC: {best_score:.4f}")

    # Save best parameters to file
    best_params_path = os.path.join(model_save_dir, f"LR_best_params_subset_{subset_id}.pkl")
    with open(best_params_path, "wb") as f:
        pickle.dump(best_params, f)

    return best_params



def add_second_order_interactions(X, sep=" * "):
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    X_interactions = X_interactions[:, X.shape[1]:]  # Remove original features
    
    # Get all feature names
    original_names = X.columns
    raw_feature_names = poly.get_feature_names_out(original_names)

    # Keep only interaction terms' names
    raw_interaction_names = raw_feature_names[X.shape[1]:]
    custom_feature_names = [name.replace(" ", sep) for name in raw_interaction_names]
    
    return pd.DataFrame(X_interactions, columns=custom_feature_names)


# load and prepare data with second-order interactions, and do initail feature screening
def load_and_prepare_data(file_path, columns_to_use=None, add_SOI=False, random_seed=None, shuffle=True):
    df = pd.read_csv(file_path)
    
    
    # Create the target variable
    df['target'] = df['count_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    # df['target'] = df["AI_label"].apply(lambda x: 1 if x == 1 else 0)
    
    y = df['target']

    
    
    # Select features 
    X = df[columns_to_use]
    print(f"Initial number of features: {X.shape[1]}")

    
    if add_SOI:
        # Add second-order interaction features
        X = add_second_order_interactions(X)
        print(f"Number of features after adding second-order interactions: {X.shape[1]}")
        
        # Initial feature screening using Random Forest importance
        X = initial_screen_features_RF(X, y, threshold=0.02)
        print(f"Number of features after initial screening: {X.shape[1]}")
        
    if shuffle:
        print(f"Shuffling dataset with random seed: {random_seed}")
        dataset = pd.concat([X, y], axis=1).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        X = dataset.iloc[:, :-1]  # Features
        y = dataset.iloc[:, -1]   # Target
    else:
        print("Skipping shuffle in load (replication loop will reshuffle).")

    return X, y, df



def compute_coef_summary(all_coefs, all_pvals, feature_names):
    coef_matrix = np.vstack(all_coefs)
    pval_matrix = np.vstack(all_pvals)
    n_runs = coef_matrix.shape[0]

    mean_coef = coef_matrix.mean(axis=0)
    std_coef = coef_matrix.std(axis=0, ddof=1)
    se = std_coef / np.sqrt(n_runs)

    t_val = stats.t.ppf(1 - 0.025, df=n_runs - 1)
    ci_lower = mean_coef - t_val * se
    ci_upper = mean_coef + t_val * se
    mean_pvals = pval_matrix.mean(axis=0)

    stars = []
    for p in mean_pvals:
        if p < 0.001:
            stars.append("***")
        elif p < 0.01:
            stars.append("**")
        elif p < 0.05:
            stars.append("*")
        else:
            stars.append("")

    return pd.DataFrame({
        "Feature": list(feature_names),
        "Mean": mean_coef,
        "Lower95CI": ci_lower,
        "Upper95CI": ci_upper,
        "MeanPval": mean_pvals,
        "Stars": stars,
    })


def draw_coef_summary_on_ax(
    ax,
    coef_df,
    split_N,
    task_label="",
    top_n=15,
    show_title=True,
    show_legend=True,
    y_tick_size: float = 22,
    y_tick_rotation: float = 25,
    x_tick_size: float = 20,
    x_label_size: float = 18,
    coef_label_size: float = 16,
    legend_size: float = 14,
    legend_title_size: float = 16,
    pos_color: str = "#e53935",
    neg_color: str = "#1e88e5",
    bar_alpha: float = 0.62,
    x_nbins: int = 4,
):
    """Draw the LR coefficient panel onto ``ax`` (standalone styling)."""
    sns.set_theme(style="white")
    mpl.rcParams.update(PLOT_FONT_RC)

    top_coef = coef_df.reindex(
        coef_df["Mean"].abs().sort_values(ascending=False).head(top_n).index
    )
    top_coef = top_coef.iloc[::-1]
    ci_upper = top_coef["Upper95CI"].values

    colors = top_coef["Mean"].apply(lambda x: pos_color if x > 0 else neg_color)
    if split_N > 1:
        xerr = [
            top_coef["Mean"] - top_coef["Lower95CI"],
            top_coef["Upper95CI"] - top_coef["Mean"],
        ]
    else:
        xerr = None

    top_coef["Feature_Display"] = top_coef["Feature"].map(format_feature_label)

    bars = ax.barh(
        top_coef["Feature_Display"],
        top_coef["Mean"],
        xerr=xerr,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4 if split_N > 1 else 0,
        alpha=bar_alpha,
    )

    for bar in bars:
        y_mid = bar.get_y() + bar.get_height() / 2
        ax.axhline(
            y_mid,
            color="#c0c0c0",
            linestyle="--",
            linewidth=0.9,
            zorder=0,
        )

    # Per-panel x-limits: as tight as the data allow (always include 0).
    if split_N > 1:
        x_min = float(np.nanmin(top_coef["Lower95CI"].to_numpy()))
        x_max = float(np.nanmax(top_coef["Upper95CI"].to_numpy()))
    else:
        x_min = float(np.nanmin(top_coef["Mean"].to_numpy()))
        x_max = float(np.nanmax(top_coef["Mean"].to_numpy()))
    x_min = min(0.0, x_min)
    x_max = max(0.0, x_max)
    span = max(x_max - x_min, 0.1)
    pad = max(0.04, 0.08 * span)
    ax.set_xlim(x_min - pad, x_max + pad)

    for bar, coef, star in zip(bars, top_coef["Mean"], top_coef["Stars"]):
        label = f"{coef:.3f} ({star})" if star else f"{coef:.3f}"
        ax.text(
            0.5 * coef,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="center",
            fontsize=coef_label_size,
            color="black",
            fontname=PLOT_FONT,
            zorder=5,
            clip_on=True,
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    if split_N > 1:
        x_label = "Mean coefficient (95% CI)"
    else:
        x_label = "Mean coefficient (single run)"
    ax.set_xlabel(x_label, fontsize=x_label_size, labelpad=30, fontname=PLOT_FONT)
    if show_title:
        title = f"Logistic Regression Coefficients ({task_label})"
        ax.set_title(title, fontsize=25, fontweight="bold", pad=30, fontname=PLOT_FONT)
    else:
        ax.set_title("")

    ax.grid(False)

    if show_legend:
        significance_legend = [
            Patch(facecolor="none", edgecolor="none", label="*    p < 0.05"),
            Patch(facecolor="none", edgecolor="none", label="**   p < 0.01"),
            Patch(facecolor="none", edgecolor="none", label="***  p < 0.001"),
        ]
        ax.legend(
            handles=significance_legend,
            loc="lower right",
            frameon=True,
            fontsize=legend_size,
            title="Significance",
            borderpad=0.8,
            handlelength=0,
            handletextpad=0.4,
            prop={"family": "sans-serif", "size": legend_size},
            title_fontproperties={"family": "sans-serif", "size": legend_title_size},
        )

    ax.tick_params(axis="y", labelsize=y_tick_size, pad=8)
    ax.tick_params(axis="x", labelsize=x_tick_size)
    x0, x1 = ax.get_xlim()
    xticks = np.linspace(x0, x1, max(2, int(x_nbins)))
    ax.set_xticks(xticks)
    # Compact labels; keep equal spacing visible.
    step = abs(xticks[-1] - xticks[0]) / max(1, len(xticks) - 1)
    if step >= 0.5:
        fmt = "{:.1f}"
    elif step >= 0.05:
        fmt = "{:.2f}"
    else:
        fmt = "{:.3f}"
    ax.set_xticklabels([fmt.format(t) for t in xticks])
    plt.sca(ax)
    plt.yticks(rotation=y_tick_rotation)
    for label in ax.get_yticklabels():
        label.set_fontname(PLOT_FONT)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    for label in ax.get_xticklabels():
        label.set_fontname(PLOT_FONT)


def _run_all_subsets_for_one_replicate(
    rep_idx,
    X_raw,
    y_raw,
    replicate_seed,
    split_n,
    load_existing_best_params,
    model_save_dir,
    scale,
):
    dataset_rep = (
        pd.concat([X_raw, y_raw], axis=1)
        .sample(frac=1, random_state=replicate_seed)
        .reset_index(drop=True)
    )
    X_rep = dataset_rep.iloc[:, :-1]
    y_rep = dataset_rep.iloc[:, -1]
    idx_parts = np.array_split(np.arange(len(X_rep)), split_n)
    subsets_X = [X_rep.iloc[p] for p in idx_parts]
    subsets_y = [y_rep.iloc[p] for p in idx_parts]

    rep_coefs = []
    rep_pvals = []
    for i in range(split_n):
        subset_run_seed = int(replicate_seed)
        print(f"\nProcessing replicate {rep_idx + 1}, subset {i + 1}...")
        X_split, y_split = subsets_X[i], subsets_y[i]

        if load_existing_best_params:
            with open(f"{model_save_dir}/LR_best_params_subset_{i}.pkl", "rb") as file:
                best_params = pickle.load(file)
        else:
            best_params = tune_hyperparameters(X_split, y_split, i, model_save_dir)

        X_train, X_test, y_train, y_test = train_test_split(
            X_split, y_split, test_size=0.2, random_state=subset_run_seed
        )

        if scale:
            X_train, X_test = scale_data(X_train, X_test)

        model = LogisticRegression(**best_params, random_state=subset_run_seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Rep {rep_idx + 1}, Split {i + 1}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}, AUC: {auc_score:.2f}")

        coefs = model.coef_[0]
        pvals = np.ones_like(coefs)
        try:
            X_train_sm = sm.add_constant(X_train)
            sm_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
            pvals = sm_model.pvalues[1:].values
        except Exception as e:
            print(f"⚠️ Failed to compute p-values for replicate {rep_idx + 1}, subset {i}: {e}")

        rep_coefs.append(coefs)
        rep_pvals.append(pvals)

    return rep_coefs, rep_pvals


def main(
    file_path,
    load_existing_best_params,
    model_save_dir,
    Split_N,
    columns_to_use=None,
    add_SOI=False,
    scale=False,
    n_replicates=100,
    initial_seed_starter=42,
):
    """
    Find robust feature importance rankings using shuffled and split subsets.
    """
    figures_dir = artifact_output_dir(model_save_dir, add_SOI)

    # Step 1: Load and prepare data with optional second-order interactions
    print("Initial sequence seed starter:", initial_seed_starter)
    replicate_seeds = [int(initial_seed_starter + rep * 100_003 + 17) for rep in range(n_replicates)]
    preview_n = min(5, len(replicate_seeds))
    print(f"Replicate seeds (first {preview_n}): {replicate_seeds[:preview_n]}")
    X, y, _ = load_and_prepare_data(file_path, columns_to_use, add_SOI, random_seed=None, shuffle=False)

    task_label = infer_task_label(file_path, model_save_dir)
    print(f"Subsample robustness: {n_replicates} reshuffles × {Split_N} subsamples = {n_replicates * Split_N} runs")
    all_coefs = []
    all_pvals = [] 
    for rep in tqdm(range(n_replicates), desc="Replication (reshuffle)", total=n_replicates):
        rep_seed = replicate_seeds[rep]
        print(f"[Replicate {rep + 1}/{n_replicates}] seed = {rep_seed}")
        rep_coefs, rep_pvals = _run_all_subsets_for_one_replicate(
            rep,
            X,
            y,
            rep_seed,
            Split_N,
            load_existing_best_params,
            model_save_dir,
            scale,
        )
        all_coefs.extend(rep_coefs)
        all_pvals.extend(rep_pvals)

    plot_top_k = 15
    coef_df = compute_coef_summary(all_coefs, all_pvals, X.columns)
    save_experiment_results(
        figures_dir,
        add_SOI,
        coef_df,
        run_config={
            "task": task_label,
            "file_path": file_path,
            "model_save_dir": model_save_dir,
            "model_name": "logistic_regression",
            "add_SOI": add_SOI,
            "scale": scale,
            "split_n": Split_N,
            "n_replicates": n_replicates,
            "plot_top_k": plot_top_k,
            "total_runs": n_replicates * Split_N,
            "initial_seed_starter": initial_seed_starter,
            "replicate_seeds": [int(s) for s in replicate_seeds],
            "load_existing_best_params": load_existing_best_params,
        },
    )


    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistic feature-sign analysis with vote aggregation")
    parser.add_argument("--file_path", required=True, help="Path to dataset CSV")
    parser.add_argument("--model_save_dir", required=True, help="Model output directory")
    parser.add_argument(
        "--add_SOI",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add second-order interaction features (default: False)",
    )
    parser.add_argument(
        "--initial_seed_starter",
        type=int,
        required=True,
        help="Base seed for replicate reshuffles",
    )
    args = parser.parse_args()

    file_path = args.file_path
    model_save_dir = args.model_save_dir
    add_SOI = args.add_SOI
    initial_seed_starter = args.initial_seed_starter

    # Select features, dropping non-relevant columns
    columns_to_use = [# 'title', 'paper_abstract', 'count_inequality_words',
                       'female', # 'female_max', 'female_min','first_author_female_score',
                       'num_authors',
                       'natural_science', 'engineering_and_technology', 'social_science',
                       'country_race_diversity_score', # 'country_race_simpson_index_mean', 'country_race_inverse_dominance_mean',
                       'authors_race_diversity_score', # 'paper_race_simpson_index', 'paper_race_inverse_dominance',
                        'white', 'asian','black', 'hispanic',
                        #'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_t-3', 
                        'paper_inequality_mentions_3_years', 
                        #'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_t-3', 
                        'news_inequality_mentions_3_years', 
                        # 'news_gender_ineq_t-0', 'news_gender_ineq_t-1', 'news_gender_ineq_t-2', 'news_gender_ineq_t-3', 'news_gender_ineq_3yr_avg', 
                        # 'news_econ_ineq_t-0', 'news_econ_ineq_t-1', 'news_econ_ineq_t-2', 'news_econ_ineq_t-3', 'news_econ_ineq_3yr_avg', 
                        # 'news_race_ineq_t-0', 'news_race_ineq_t-1', 'news_race_ineq_t-2', 'news_race_ineq_t-3', 
                        # 'news_race_ineq_3yr_avg'
                        ]
    
    scale = True
    load_existing_best_params = True

    split_N = 10
    n_replicates = 100

    main(
        file_path,
        load_existing_best_params,
        model_save_dir,
        split_N,
        columns_to_use,
        add_SOI,
        scale,
        n_replicates=n_replicates,
        initial_seed_starter=initial_seed_starter,
    )

    print(
        "\n--- Run parameters ---\n"
        f"  scale: {scale}\n"
        f"  load_existing_best_params: {load_existing_best_params}\n"
        f"  add_SOI: {add_SOI}\n"
        f"  split_N: {split_N}\n"
        f"  n_replicates: {n_replicates}\n"
        f"  initial_seed_starter: {initial_seed_starter}\n"
        f"  columns_to_use: {columns_to_use}\n"
        "--- End ---\n"
    )

# Example usage:
#   python code/feature_sign_Logistic.py \
#       --file_path data/race_ineq_dataset.csv \
#       --model_save_dir models/race \
#       --initial_seed_starter 99
#