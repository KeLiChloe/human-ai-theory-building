import json
import os
import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier  # optional; unused when RF-only
except ImportError:  # pragma: no cover
    XGBClassifier = None
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
try:
    import shap
except ImportError:  # pragma: no cover
    shap = None
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,   # TrueType; Nature-compatible editable text
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

PLOT_FONT = "Helvetica"

# Paper feature names (spaces → underscores). Map internal column names → paper names.
PAPER_FEATURE_NAMES = {
    "number_of_authors": "num_authors",
    "num_authors": "num_authors",
    "female": "female",
    "female_mean": "female",
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

# Back-compat alias used elsewhere
FEATURE_LABELS = PAPER_FEATURE_NAMES


def format_feature_label(feature_name, wrap_interactions=False):
    """Map internal feature names to paper underscore names (for plots)."""
    if not isinstance(feature_name, str):
        return feature_name
    if " * " in feature_name:
        parts = [PAPER_FEATURE_NAMES.get(part, part) for part in feature_name.split(" * ")]
        if wrap_interactions and len(parts) >= 2:
            # Two-line interaction: keeps the y-axis narrow on SOI panels.
            return f"{parts[0]} *\n{parts[1]}"
        return " * ".join(parts)
    return PAPER_FEATURE_NAMES.get(feature_name, feature_name)


def wrap_feature_label(feature_name):
    if not isinstance(feature_name, str):
        return feature_name
    if " * " in feature_name:
        return feature_name.replace(" * ", "\n*\n")
    return feature_name


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


def results_output_base(figures_dir, importance_type, add_SOI):
    soi_suffix = "_soi" if add_SOI else ""
    return os.path.join(
        figures_dir,
        f"RF_feature_importance_results_{importance_type}{soi_suffix}",
    )


def save_experiment_results(
    figures_dir,
    importance_type,
    add_SOI,
    combined_df,
    run_config,
):
    """Save ``run_config`` and feature ``ranking`` to a single JSON file."""
    output_base = results_output_base(figures_dir, importance_type, add_SOI)

    ranking = [
        {
            "rank": rank,
            "feature": row["Feature"],
            "votes": int(row["Votes"]),
            "feature_importance": round(float(row["Feature Importance"]), 6),
        }
        for rank, (_, row) in enumerate(combined_df.iterrows(), start=1)
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


def get_importance_display_name(importance_type):
    names = {
        "mdi": "MDI (Gini) Importance",
        "permutation": "Permutation Importance",
        "shap": "SHAP Importance",
    }
    return names.get(importance_type, importance_type)


def compute_feature_importance(
    model, X_eval, y_eval, importance_type, random_seed
):
    if importance_type == "mdi":
        return model.feature_importances_

    if importance_type == "permutation":
        perm = permutation_importance(
            model,
            X_eval,
            y_eval,
            random_state=random_seed,
            scoring="roc_auc",
            n_jobs=1,
        )
        return perm.importances_mean

    if importance_type == "shap":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        # Binary classification may return a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Newer SHAP may return shape: (n_samples, n_features, n_classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

        return np.abs(shap_values).mean(axis=0)

    raise ValueError(f"Unsupported importance_type: {importance_type}")


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

def tune_hyperparameters(X_train, y_train, subset_id, model_save_dir, random_seed=42):
    """
    Tune hyperparameters for models using GridSearchCV.

    Parameters:
        X_train (array-like): Features for training.
        y_train (array-like): Labels for training.

    Returns:
        dict: Best hyperparameters for each model.
    """
    # Define hyperparameter grids
    # ================== 1. Random Forest: moderate search grid ==================
    # Tuned for ~100k rows / 10–15 features / ~4:6 class balance; 1–3 values per param
    param_grid_rf = {
        "n_estimators": [200, 400],      # tree count: medium and slightly larger
        "criterion": ["gini"],           # keep fixed for stability
        "max_depth": [12, 16, 20],       # three moderate depth options
        "min_samples_split": [5, 10],    # two settings to limit overfitting
        "min_samples_leaf": [3, 5],      # min 3 or 5 samples per leaf
        "bootstrap": [True],
        "max_features": ["sqrt", 0.8],   # two common feature-sampling options
    }

    # ================== 2. XGBoost: compact grid ==================
    # Keep the grid small while covering conservative and slightly aggressive settings
    param_grid_xgb = {
        "n_estimators": [300, 600],      # boosting rounds: medium and higher
        "learning_rate": [0.05, 0.1],    # smaller vs slightly larger learning rate
        "max_depth": [4, 5],             # moderate depth to limit overfitting
        "subsample": [0.8],              # row subsample for generalization
        "colsample_bytree": [0.8],       # feature subsample per tree
        "min_child_weight": [1, 5],      # min child weight: more flexible vs more conservative
        "reg_lambda": [1.0],             # fixed L2 regularization
        "reg_alpha": [0.0, 0.5],         # optional light L1 regularization
    }

    # Initialize models
    models = {
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            param_grid_rf,
        ),
        # "gradient_boosting": (XGBClassifier(random_state=42), param_grid_xgb),
    }

    # Tune models
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(f"{model_save_dir}/best_params_subset_{subset_id}.pkl", "wb") as file:
        pickle.dump(best_params, file)
    

    return best_params

# load and prepare data with second-order interactions, and do initail feature screening
def load_and_prepare_data(file_path, columns_to_use=None, add_SOI=False, random_seed=None, shuffle=True):
    df = pd.read_csv(file_path)
    
    
    # Create the target variable
    df['target'] = df['count_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    
   
    y = df['target']
    # check the distribution of target variable
    print("Target variable distribution:")
    print(y.value_counts()) 
    
    # Select features 
    X = df[columns_to_use]

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
        print("Skipping shuffle in load (features row order fixed; replication loop will reshuffle).")

    return X, y, df

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





def draw_feature_importance_on_ax(
    ax,
    fig,
    combined_df,
    importance_type,
    add_SOI=False,
    show_title=True,
    task_label="",
    cax=None,
    xlim_pad_frac: float = 0.12,
    xlim_pad_min: float = 80.0,
    colorbar_fraction: float = 0.046,
    y_tick_size: float | None = None,
    y_tick_rotation: float = 25,
    x_tick_size: float = 20,
    x_label_size: float = 24,
    colorbar_tick_size: float = 14,
    colorbar_label_size: float = 20,
    wrap_interactions: bool = False,
    bar_label_size: float = 18,
    show_colorbar_ticks: bool = True,
    colorbar_nbins: int = 5,
    x_max: float | None = None,
):
    """Draw the RF votes panel onto ``ax`` (standalone styling).

    ``xlim_pad_frac`` / ``xlim_pad_min`` control empty space to the right of
    bars (importance value labels). Smaller values shrink the *plot* x-range
    without changing tick font sizes. If ``x_max`` is set, the x-axis is fixed
    to ``[0, x_max]`` (e.g. 1000 runs).
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })

    combined_df = combined_df.copy()
    combined_df["Feature"] = combined_df["Feature"].map(
        lambda x: format_feature_label(x, wrap_interactions=wrap_interactions)
    )
    importance_label = get_importance_display_name(importance_type)

    norm = mcolors.Normalize(
        vmin=combined_df["Feature Importance"].min(),
        vmax=combined_df["Feature Importance"].max(),
    )
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = cmap(norm(combined_df["Feature Importance"].values))

    combined_df = combined_df.sort_values(
        by=["Votes", "Feature Importance"], ascending=[True, True]
    )

    votes_max = int(combined_df["Votes"].max())
    bars = ax.barh(
        combined_df["Feature"],
        combined_df["Votes"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        height=0.65,
        alpha=0.82,
    )

    # Pad so every MDI value fits outside the bar tip (always visible).
    if x_max is not None:
        # Axis extends past the last tick so bar-end numbers have room.
        x_limit = 1400.0 if float(x_max) == 1000 else float(x_max)
        ax.set_xlim(0, x_limit)
    else:
        label_pad = max(votes_max * xlim_pad_frac, xlim_pad_min)
        ax.set_xlim(0, votes_max + label_pad)
    label_offset = max(0.5, votes_max * 0.015)
    for bar, importance in zip(bars, combined_df["Feature Importance"]):
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.3f}",
            va="center",
            ha="left",
            fontsize=bar_label_size,
            color="black",
            fontname=PLOT_FONT,
            clip_on=False,
            zorder=5,
        )

    ranking_metric = {
        "mdi": "MDI",
        "permutation": "Permutation",
        "shap": "SHAP",
    }.get(importance_type, importance_type.upper())
    x_label = (
        f"Total number of votes\n"
        f"(voting metric in each run: {ranking_metric})"
    )
    ax.set_xlabel(x_label, fontsize=x_label_size, labelpad=42, fontname=PLOT_FONT)
    if show_title:
        effects_label = "Second-Order Interactions" if add_SOI else "Main Effects"
        title = f"Top Voted Features - {task_label} ({effects_label})"
        ax.set_title(title, fontsize=28, pad=18, weight="bold", fontname=PLOT_FONT)
    else:
        ax.set_title("")

    if y_tick_size is None:
        y_tick_size = 42 if add_SOI else 46
    ax.tick_params(axis="y", labelsize=y_tick_size, pad=10)
    ax.tick_params(axis="x", labelsize=x_tick_size)
    plt.sca(ax)
    plt.yticks(rotation=y_tick_rotation)
    for label in ax.get_yticklabels():
        label.set_fontname(PLOT_FONT)
        label.set_ha("right")
        label.set_va("center")
        label.set_rotation_mode("anchor")
        if wrap_interactions:
            label.set_linespacing(0.92)
    for label in ax.get_xticklabels():
        label.set_fontname(PLOT_FONT)

    ax.grid(axis="x", linestyle="--", alpha=0.6)
    if x_max is not None and float(x_max) == 1000:
        # Ticks stop at 1000; axis frame continues to 1200 for label padding.
        ax.set_xticks([0, 250, 500, 750, 1000])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune=None))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if cax is not None:
        cbar = fig.colorbar(sm, cax=cax)
    else:
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=colorbar_fraction)
    cbar.set_label(
        f"Averaged {importance_label}",
        fontsize=colorbar_label_size,
        labelpad=16,
        fontname=PLOT_FONT,
    )
    if show_colorbar_ticks:
        n_ticks = max(2, int(colorbar_nbins))
        vmin = float(norm.vmin)
        vmax = float(norm.vmax)
        ticks = np.linspace(vmin, vmax, n_ticks)
        # Equal spacing in data + on the bar; pick decimals so printed
        # labels keep that equal step (avoid .2f rounding artifacts).
        step = (vmax - vmin) / (n_ticks - 1) if n_ticks > 1 else 0.0
        if step >= 0.05:
            fmt = "{:.2f}"
        elif step >= 0.005:
            fmt = "{:.3f}"
        else:
            fmt = "{:.4f}"
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([fmt.format(t) for t in ticks])
        cbar.ax.tick_params(labelsize=colorbar_tick_size)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(PLOT_FONT)
    else:
        cbar.ax.set_yticks([])
        cbar.ax.tick_params(length=0, labelleft=False, labelright=False)
    return cbar


def load_best_params_cache(model_save_dir, split_n, source="global_pkl"):
    """
    Load hyperparameters for each of split_n subsamples.

    source:
        "global_pkl" — single file ``{model_save_dir}/best_params.pkl`` (e.g. from ml_classification.py),
          reused for every subset index.
        "subset_pkls" — one file per subset: ``best_params_subset_{i}.pkl`` (from tune_hyperparameters here).
    """
    if source == "global_pkl":
        path = os.path.join(model_save_dir, "best_params.pkl")
        with open(path, "rb") as fh:
            best_params = pickle.load(fh)
        print(f"Loaded hyperparameters from {path} (same dict for all {split_n} subsets)")
        return [best_params] * split_n

    if source == "subset_pkls":
        cache = []
        for i in range(split_n):
            path = os.path.join(model_save_dir, f"best_params_subset_{i}.pkl")
            with open(path, "rb") as fh:
                cache.append(pickle.load(fh))
        print(
            f"Loaded hyperparameters from {split_n} files: "
            f"best_params_subset_0.pkl … best_params_subset_{split_n - 1}.pkl"
        )
        return cache

    raise ValueError('source must be "global_pkl" or "subset_pkls"')


def _strip_rf_n_jobs(rf_params):
    """GridSearch may store n_jobs in best_params_; force single-thread fit."""
    out = dict(rf_params)
    out.pop("n_jobs", None)
    out["n_jobs"] = 1
    return out


def _run_all_subsets_for_one_replicate(
    rep_idx,
    X_raw,
    y_raw,
    replicate_seed,
    split_n,
    load_existing_best_params,
    model_save_dir,
    model_name,
    scale,
    importance_type,
    best_params_cache,
):
    """
    One full reshuffle of the full table, then train + importance on each of split_n subsets.
    Returns a list of length split_n (one importance DataFrame per subset).
    """
    dataset_rep = (
        pd.concat([X_raw, y_raw], axis=1)
        .sample(frac=1, random_state=replicate_seed)
        .reset_index(drop=True)
    )
    X_rep = dataset_rep.iloc[:, :-1]
    y_rep = dataset_rep.iloc[:, -1]
    # np.array_split(DataFrame/Series) triggers pandas swapaxes deprecation (numpy→pandas dispatch).
    idx_parts = np.array_split(np.arange(len(X_rep)), split_n)
    subsets_X = [X_rep.iloc[p] for p in idx_parts]
    subsets_y = [y_rep.iloc[p] for p in idx_parts]

    out_dfs = []
    for i in range(split_n):
        X_split = subsets_X[i]
        y_split = subsets_y[i]
        subset_run_seed = int(replicate_seed)

        if load_existing_best_params:
            best_params = best_params_cache[i]
        else:
            best_params = tune_hyperparameters(
                X_split, y_split, i, model_save_dir, random_seed=subset_run_seed
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_split, y_split, test_size=0.2, random_state=replicate_seed,
        )
        if scale:
            X_train, X_test = scale_data(X_train, X_test)

        if model_name == "random_forest":
            rf_params = _strip_rf_n_jobs(best_params["random_forest"])
            model = RandomForestClassifier(
                **rf_params, random_state=replicate_seed
            )
            model.fit(X_train, y_train)
        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(
                **best_params["gradient_boosting"], random_state=subset_run_seed
            )
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        importance_values = compute_feature_importance(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            importance_type=importance_type,
            random_seed=subset_run_seed,
        )
        out_dfs.append(
            pd.DataFrame(
                {
                    "Feature": X_train.columns,
                    "Feature Importance": importance_values,
                }
            )
        )
    return out_dfs


def main(
    file_path,
    load_existing_best_params,
    model_name,
    model_save_dir,
    Split_N,
    columns_to_use=None,
    add_SOI=False,
    scale=False,
    importance_type="mdi",
    n_replicates=100,
    best_params_source="global_pkl",
    initial_seed_starter=42,
):
    """
    Find robust feature importance rankings using shuffled and split subsets.

    Parameters:
        file_path (str): Path to the dataset.
        model_save_dir (str): Directory to save model parameters.
        best_params_source: ``"global_pkl"`` loads ``best_params.pkl``; ``"subset_pkls"`` loads per-subset files from tuning.

    Returns:
        DataFrame: Robust feature importance ranking.
    """
    figures_dir = artifact_output_dir(model_save_dir, add_SOI)

    # Step 1: Load and prepare data with second-order interactions
    print("Initial sequence seed starter:", initial_seed_starter)
    replicate_seeds = [int(initial_seed_starter + rep * 100_003 + 17) for rep in range(n_replicates)]
    preview_n = min(5, len(replicate_seeds))
    print(f"Replicate seeds (first {preview_n}): {replicate_seeds[:preview_n]}")
    X_raw, y_raw, _ = load_and_prepare_data(
        file_path, columns_to_use, add_SOI, random_seed=None, shuffle=False
    )
    task_label = infer_task_label(file_path, model_save_dir)
    print(f"Importance type: {importance_type}")
    print(f"Subsample robustness: {n_replicates} reshuffles × {Split_N} subsamples = {n_replicates * Split_N} runs")

    best_params_cache = (
        load_best_params_cache(model_save_dir, Split_N, source=best_params_source)
        if load_existing_best_params
        else None
    )


    all_feature_importances = []
    for rep in tqdm(
        range(n_replicates), desc="Replication (reshuffle)", total=n_replicates
    ):
        rep_seed = replicate_seeds[rep]
        print(f"[Replicate {rep + 1}/{n_replicates}] seed = {rep_seed}")
        all_feature_importances.extend(
            _run_all_subsets_for_one_replicate(
                rep,
                X_raw,
                y_raw,
                rep_seed,
                Split_N,
                load_existing_best_params,
                model_save_dir,
                model_name,
                scale,
                importance_type,
                best_params_cache,
            )
        )

    average_importance = pd.concat(all_feature_importances).groupby("Feature").mean().sort_values(by="Feature Importance", ascending=False)

    # Combine feature importance with votes
    Top_N = 5 #  it means as long as the feature is in the top N of any subset, it will be voted 
    feature_votes = {}
    for importance_df in all_feature_importances:
        top_features = importance_df.nlargest(Top_N, 'Feature Importance')['Feature']
        for feature in top_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
    votes_df = pd.DataFrame(list(feature_votes.items()), columns=['Feature', 'Votes'])

    # Keep ALL features that have an averaged importance. Votes=0 means never in
    # run-level top-N — that is NOT the same as importance=0.
    combined_df = (
        average_importance.reset_index()
        .merge(votes_df, on="Feature", how="left")
    )
    combined_df["Votes"] = combined_df["Votes"].fillna(0).astype(int)

    # Sort by Votes (descending) and Importance (descending)
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[False, False])
    print("\nFinal Feature Importance Values Ranking (Combined Votes and Feature Importance):")
    print(combined_df)

    plot_top_k = 3 if add_SOI else 5
    save_experiment_results(
        figures_dir,
        importance_type,
        add_SOI,
        combined_df,
        run_config={
            "task": task_label,
            "file_path": file_path,
            "model_save_dir": model_save_dir,
            "model_name": model_name,
            "importance_type": importance_type,
            "importance_label": get_importance_display_name(importance_type),
            "add_SOI": add_SOI,
            "scale": scale,
            "split_n": Split_N,
            "n_replicates": n_replicates,
            "vote_top_n": Top_N,
            "plot_top_k": plot_top_k,
            "total_runs": n_replicates * Split_N,
            "initial_seed_starter": initial_seed_starter,
            "replicate_seeds": [int(s) for s in replicate_seeds],
            "load_existing_best_params": load_existing_best_params,
            "best_params_source": best_params_source,
        },
    )

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF feature importance with vote aggregation")
    parser.add_argument("--file_path", required=True, help="Path to dataset CSV")
    parser.add_argument("--model_save_dir", required=True, help="Model output directory")
    parser.add_argument(
        "--importance_type",
        choices=["mdi", "permutation", "shap"],
        default="mdi",
        help="Importance metric (default: mdi)",
    )
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
    importance_type = args.importance_type
    add_SOI = args.add_SOI
    initial_seed_starter = args.initial_seed_starter

    model_name = 'random_forest' # [random_forest, gradient_boosting]
    
    columns_to_use = [# 'title', 'paper_abstract', 'count_inequality_words',
                       'female', # 'female_max', 'female_min','first_author_female_score',
                       'natural_science',  'social_science', 'engineering_and_technology',
                       'country_race_diversity_score', # 'country_race_simpson_index_mean', 'country_race_inverse_dominance_mean',
                       'authors_race_diversity_score', # 'paper_race_simpson_index', 'paper_race_inverse_dominance',
                        'white', 'asian', 'black', 'hispanic',
                        #'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_t-3', 
                        'paper_inequality_mentions_3_years', 
                        #'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_t-3', 
                        'news_inequality_mentions_3_years', 
                        # 'news_gender_ineq_t-0', 'news_gender_ineq_t-1', 'news_gender_ineq_t-2', 'news_gender_ineq_t-3', 
                        # 'news_gender_ineq_3yr_avg', 
                        # 'news_econ_ineq_t-0', 'news_econ_ineq_t-1', 'news_econ_ineq_t-2', 'news_econ_ineq_t-3', 'news_econ_ineq_3yr_avg', 
                        # 'news_race_ineq_t-0', 'news_race_ineq_t-1', 'news_race_ineq_t-2', 'news_race_ineq_t-3', 
                        # 'news_race_ineq_3yr_avg'
                        ]
    

    scale = True
    load_existing_best_params = True
    best_params_source = "subset_pkls"    # ["global_pkl", "subset_pkls"]

    Split_N = 10
    n_replicates = 100  # reshuffles; each uses Split_N subsamples → n_replicates * Split_N importance runs


    main(
        file_path,
        load_existing_best_params,
        model_name,
        model_save_dir,
        Split_N,
        columns_to_use,
        add_SOI,
        scale,
        importance_type,
        n_replicates=n_replicates,
        best_params_source=best_params_source,
        initial_seed_starter=initial_seed_starter,
    )

    print(
        "\n--- Run parameters ---\n"
        f"  scale: {scale}\n"
        f"  load_existing_best_params: {load_existing_best_params}\n"
        f"  best_params_source: {best_params_source}\n"
        f"  add_SOI: {add_SOI}\n"
        f"  Split_N: {Split_N}\n"
        f"  n_replicates: {n_replicates}\n"
        f"  initial_seed_starter: {initial_seed_starter}\n"
        f"  columns_to_use: {columns_to_use}\n"
        "--- End ---\n"
    )

# Example:
#   python code/feature_importance_RF.py \
#       --file_path data/race_ineq_dataset.csv \
#       --model_save_dir models/race \
#       --importance_type mdi \
#       --initial_seed_starter 102 \
#       --add_SOI