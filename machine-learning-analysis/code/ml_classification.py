import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle


def initial_screen_features_RF(X, y, threshold):
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
    random_seed = np.random.randint(10000)
    rf_model = RandomForestClassifier(random_state=random_seed)
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
    print(importance_df.head(10))
    print(f"\nSelected {len(selected_features)} features based on importance > {threshold}")

    return X_filtered

def initial_screen_features_lasso(X, y, threshold, random_seed=None):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Regularization strength (LASSO parameter).
        random_seed (int): RNG seed for LASSO.

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    if random_seed is None:
        random_seed = np.random.randint(10000)
    lasso_model = Lasso(alpha=threshold, random_state=random_seed)
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

def tune_hyperparameters(X_train, y_train, model_save_dir, add_SOI):
    """
    Tune hyperparameters for models using GridSearchCV.

    Parameters:
        X_train (array-like): Features for training.
        y_train (array-like): Labels for training.

    Returns:
        dict: Best hyperparameters for each model.
    """
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
        "n_jobs": [-1],
    }

    # ================== 2. Initialize models ==================
    random_seed = np.random.randint(10000)
    models = {
        "random_forest": (
            RandomForestClassifier(random_state=random_seed),
            param_grid_rf
        ),
    }

    # ================== 3. GridSearch ==================
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(
        f"{model_save_dir}/best_params.pkl",
        "wb"
    ) as file:
        pickle.dump(best_params, file)
    
    return best_params


# Function to load and prepare data
def load_and_prepare_data(file_path, feature_columns):
    df = pd.read_csv(file_path)
    
    # Create the target variable
    df['target'] = df['count_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    # df['target'] = df["AI_label"].apply(lambda x: 1 if int(x) == 1 else 0)

    y = df['target']
    print(f"Positive class ratio: {y.sum()}/{len(y)} = {y.mean():.4f}")

    # Select features 
    X = df[feature_columns]
    return X, y, df

def add_second_order_interactions(X):
    # PolynomialFeatures with degree 2 for second-order interactions
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)
    X = pd.DataFrame(X_interactions, columns=interaction_feature_names)
    
    return X

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert ndarray back to DataFrame, keeping column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

def infer_task_label(file_path, model_save_dir):
    context = f"{file_path} {model_save_dir}".lower()
    if "gender" in context:
        return "Gender"
    if "race" in context:
        return "Race"
    return "Task"


def classification_results_path(model_save_dir, seed):
    return os.path.join(model_save_dir, f"RF_classification_results_seed{seed}.pkl")


def build_rf_classification_payload(
    y_test,
    y_pred_proba,
    *,
    file_path,
    model_save_dir,
    add_SOI,
    scale,
    seed,
    feature_names,
    best_params_rf,
    task_label=None,
):
    """Serialize RF test-set curves for later redrawing (stacked figure e1/e2)."""
    thresholds, accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = float(roc_auc_score(y_test, y_pred_proba))
    return {
        "run_config": {
            "task": task_label or infer_task_label(file_path, model_save_dir),
            "file_path": file_path,
            "model_save_dir": model_save_dir,
            "add_SOI": bool(add_SOI),
            "scale": bool(scale),
            "seed": int(seed),
            "n_features": int(len(feature_names)),
            "feature_names": list(feature_names),
            "target": "count_inequality_words > 0",
            "auc": auc,
            "best_params_rf": best_params_rf or {},
        },
        "thresholds": np.asarray(thresholds, dtype=float),
        "accuracy": np.asarray(accuracy, dtype=float),
        "precision": np.asarray(precision, dtype=float),
        "recall": np.asarray(recall, dtype=float),
        "f1": np.asarray(f1, dtype=float),
        "fpr": np.asarray(fpr, dtype=float),
        "tpr": np.asarray(tpr, dtype=float),
    }


def save_rf_classification_results(payload, model_save_dir, seed):
    os.makedirs(model_save_dir, exist_ok=True)
    out_path = classification_results_path(model_save_dir, seed)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"✅ Saved RF classification results → {out_path}")
    return out_path


def load_rf_classification_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def draw_rf_classification_on_ax(ax, payload, show_title=True, show_legend=True):
    """Draw RF threshold metrics (and AUC) from a saved classification payload."""
    cfg = payload["run_config"]
    thresholds = payload["thresholds"]
    ax.plot(thresholds, payload["accuracy"], label="Accuracy", linewidth=2.2)
    ax.plot(thresholds, payload["precision"], label="Precision", linewidth=2.2)
    ax.plot(thresholds, payload["recall"], label="Recall", linewidth=2.2)
    ax.plot(thresholds, payload["f1"], label="F1 Score", linewidth=2.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Threshold", fontsize=22, fontname="Helvetica")
    ax.set_ylabel("Metric Value", fontsize=22, fontname="Helvetica")
    if show_title:
        soi = " With SOI" if cfg.get("add_SOI") else ""
        ax.set_title(
            f"Random forest{soi} (AUC = {cfg['auc']:.3f}; "
            f"feature # = {cfg['n_features']})",
            fontsize=24,
            fontweight="bold",
            fontname="Helvetica",
            pad=12,
        )
    if show_legend:
        ax.legend(fontsize=16, frameon=False, loc="best")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.tick_params(axis="both", labelsize=16)
    return ax


# Function to train and predict models
def train_and_predict_models(X_train_scaled, y_train, X_test_scaled, best_params=None, random_seed=None):
    """
    Train models using the best parameters and make predictions.

    Parameters:
        X_train_scaled (array-like): Scaled training features.
        y_train (array-like): Training labels.
        X_test_scaled (array-like): Scaled test features.
        best_params (dict): Tuned hyperparameters for each model (optional).
        random_seed (int): RNG seed for stochastic models / reproducibility.

    Returns:
        dict: Predicted probabilities for each model.
    """
    if random_seed is None:
        random_seed = np.random.randint(10000)
    models = {
        "random_forest": RandomForestClassifier(random_state=random_seed),
    }

    trained_models = {}
    predictions = {}

    # Update models with tuned hyperparameters
    if best_params:
        for model_name in models.keys():
            if model_name in best_params:
                models[model_name].set_params(**best_params[model_name])

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]

    return trained_models, predictions


# Function to calculate metrics for varying thresholds
def calculate_metrics(y_test, y_pred_proba):
    thresholds = np.arange(0.0, 1.05, 0.05)
    accuracy, precision, recall, f1 = [], [], [], []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        accuracy.append(accuracy_score(y_test, y_pred_thresh))
        precision.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recall.append(recall_score(y_test, y_pred_thresh))
        f1.append(f1_score(y_test, y_pred_thresh))
    return thresholds, accuracy, precision, recall, f1

# Function to perform cross-validation on the train set
def cross_validate_with_metrics(X_train, y_train, n_splits=5):
    random_seed = np.random.randint(10000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Initialize dictionaries to store metrics and ROC data

    cv_results = {
        "random_forest": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    }

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        

        _, predictions = train_and_predict_models(X_train_fold, y_train_fold, X_val_fold)

        for model_name, y_pred_proba in predictions.items():
            _, acc, prec, rec, f1 = calculate_metrics(y_val_fold, y_pred_proba)
            cv_results[model_name]["accuracy"].append(acc)
            cv_results[model_name]["precision"].append(prec)
            cv_results[model_name]["recall"].append(rec)
            cv_results[model_name]["f1"].append(f1)
            cv_results[model_name]["auc"].append(roc_auc_score(y_val_fold, y_pred_proba))

    # Average metrics across folds
    for model_name, metrics in cv_results.items():
        for metric, values in metrics.items():
            cv_results[model_name][metric] = np.mean(values, axis=0)

    return cv_results

# Main function
def main(file_path, model_save_dir, add_SOI, use_best_params, feature_columns, scale, seed=1024):

    task_label = infer_task_label(file_path, model_save_dir)
    print(f"Task: {task_label} | seed={seed}")

    # Step 1: Load and prepare data
    # y = 1 if count_inequality_words > 0 else 0
    X, y, _ = load_and_prepare_data(file_path, feature_columns)
    
    if add_SOI:
        X = add_second_order_interactions(X)
        X = initial_screen_features_lasso(X, y, threshold=0.005, random_seed=seed)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    feature_names = X_train.columns.tolist()
    print(f"Number of features: {len(feature_names)}")

    # Step 4: Train on full train set and predict on test set
    if use_best_params:
        with open(f"{model_save_dir}/best_params.pkl", "rb") as file:
            best_params = pickle.load(file)
    else:
        print("Tuning hyperparameters...")
        best_params = tune_hyperparameters(X_train, y_train, model_save_dir, add_SOI)
        print("\nBest Hyperparameters:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")
    
    train_models, predictions = train_and_predict_models(
        X_train, y_train, X_test, best_params, random_seed=seed
    )
    
    print("\nTest Results:")
    for model_name, y_pred_proba in predictions.items():
        print(f"\nModel: {model_name}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Persist RF curves for stacked figure (e1/e2)
    rf_payload = build_rf_classification_payload(
        y_test,
        predictions["random_forest"],
        file_path=file_path,
        model_save_dir=model_save_dir,
        add_SOI=add_SOI,
        scale=scale,
        seed=seed,
        feature_names=feature_names,
        best_params_rf=best_params.get("random_forest"),
        task_label=task_label,
    )
    save_rf_classification_results(rf_payload, model_save_dir, seed)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python code/ml_classification.py "
            "<file_path> <model_save_dir> [seed]"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    add_SOI = False
    
    use_best_params = True
    
    scale = True
    
    feature_columns =  [
                       'female', # 'female_max', 'female_min','first_author_female_score',
                       'num_authors',
                       'natural_science', 'engineering_and_technology', 'social_science',
                       'country_race_diversity_score', #'country_race_simpson_index_mean', 'country_race_inverse_dominance_mean',
                       'authors_race_diversity_score', #'paper_race_simpson_index', 'paper_race_inverse_dominance',
                        'white', 'asian', 'black', 'hispanic',
                        # 'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_t-3', 
                        'paper_inequality_mentions_3_years', 
                        # 'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_t-3', 
                        'news_inequality_mentions_3_years', 
                        # 'news_gender_ineq_t-0', 'news_gender_ineq_t-1', 'news_gender_ineq_t-2', 'news_gender_ineq_t-3', 'news_gender_ineq_3yr_avg', 
                        # 'news_econ_ineq_t-0', 'news_econ_ineq_t-1', 'news_econ_ineq_t-2', 'news_econ_ineq_t-3', 'news_econ_ineq_3yr_avg', 
                        # 'news_race_ineq_t-0', 'news_race_ineq_t-1', 'news_race_ineq_t-2', 'news_race_ineq_t-3', 'news_race_ineq_3yr_avg'
                        ]
    
    main(file_path, model_save_dir, add_SOI, use_best_params, feature_columns, scale, seed=seed)

