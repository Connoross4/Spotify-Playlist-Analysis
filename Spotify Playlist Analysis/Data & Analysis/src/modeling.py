from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = ROOT / "tracks_with_semantic.csv"
CLUSTERED_PATH = ROOT / "tracks_with_clusters.csv"
MODEL_RESULTS_PATH = ROOT / "model_results.csv"
SCORED_PATH = ROOT / "tracks_scored.csv"
MODEL_PLOTS_DIR = ROOT / "model_plots"


def _build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def _evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor, random_state: int = 42
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    results = []
    best_pipeline = None
    best_auc = -np.inf
    eval_outputs: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in models.items():
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        if isinstance(model, LogisticRegression):
            metrics["n_params_or_trees"] = model.coef_.size + model.intercept_.size
        elif isinstance(model, DecisionTreeClassifier):
            metrics["n_params_or_trees"] = model.tree_.node_count
        elif isinstance(model, RandomForestClassifier):
            metrics["n_params_or_trees"] = model.n_estimators
        elif isinstance(model, GradientBoostingClassifier):
            metrics["n_params_or_trees"] = model.n_estimators
        else:
            metrics["n_params_or_trees"] = np.nan

        results.append(metrics)

        eval_outputs[name] = {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_pipeline = clf

    results_df = pd.DataFrame(results)
    return best_pipeline, results_df, eval_outputs


def run_supervised_modeling() -> pd.DataFrame:
    df = pd.read_csv(LABELED_PATH)
    df = df.dropna(subset=["would_add_to_playlist"])
    df["would_add_to_playlist"] = df["would_add_to_playlist"].astype(int)

    numeric_features = [
        "duration_ms",
        "popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]
    categorical_features = ["cluster_label"]

    missing_features = [col for col in numeric_features + categorical_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in labeled data: {missing_features}")

    X = df[numeric_features + categorical_features].copy()
    y = df["would_add_to_playlist"]

    X["cluster_label"] = X["cluster_label"].astype("category")

    stratify_arg = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    best_pipeline, results_df, eval_outputs = _evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    MODEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(MODEL_RESULTS_PATH, index=False)
    print("Model evaluation results:")
    print(results_df)
    best_model_name = results_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    print(f"Selected best model based on ROC AUC: {best_model_name}")
    print(f"Saved model metrics to {MODEL_RESULTS_PATH}")

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    melted = results_df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / "model_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, outputs in eval_outputs.items():
        fpr, tpr, _ = roc_curve(outputs["y_true"], outputs["y_prob"])
        auc_val = roc_auc_score(outputs["y_true"], outputs["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / "roc_curves.png")
    plt.close()

    best_outputs = eval_outputs[best_model_name]
    cm = confusion_matrix(best_outputs["y_true"], best_outputs["y_pred"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / "confusion_matrix_best_model.png")
    plt.close()

    # Refit best model on all labeled data
    best_pipeline.fit(X, y)

    # Probability distribution on labeled data
    labeled_features = X.copy()
    labeled_features["cluster_label"] = labeled_features["cluster_label"].astype("category")
    labeled_scores = best_pipeline.predict_proba(labeled_features)[:, 1]

    plt.figure(figsize=(8, 5))
    sns.histplot(labeled_scores[y == 1], color="tab:blue", label="would_add=1", kde=True, stat="density", alpha=0.5)
    sns.histplot(labeled_scores[y == 0], color="tab:orange", label="would_add=0", kde=True, stat="density", alpha=0.5)
    plt.xlabel("Taste Score (P[would_add=1])")
    plt.ylabel("Density")
    plt.title("Taste Score Distribution by True Label (labeled set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_PLOTS_DIR / "taste_score_distribution.png")
    plt.close()

    # Score full catalog
    full_df = pd.read_csv(CLUSTERED_PATH)
    missing_full = [col for col in numeric_features + categorical_features if col not in full_df.columns]
    if missing_full:
        raise ValueError(f"Missing required features in full catalog: {missing_full}")

    # Attach labels for correctness if available
    if "id" in df.columns and "id" in full_df.columns:
        full_df = full_df.merge(df[["id", "would_add_to_playlist"]], on="id", how="left")
    else:
        full_df = full_df.reset_index().merge(
            df.reset_index()[["index", "would_add_to_playlist"]], on="index", how="left"
        ).drop(columns=["index"])

    full_features = full_df[numeric_features + categorical_features].copy()
    full_features["cluster_label"] = full_features["cluster_label"].astype("category")

    taste_scores = best_pipeline.predict_proba(full_features)[:, 1]
    pred_labels = best_pipeline.predict(full_features)

    full_df["taste_score"] = taste_scores
    full_df["pred_label"] = pred_labels
    full_df["correct_prediction"] = np.where(
        full_df["would_add_to_playlist"].notna(),
        full_df["pred_label"] == full_df["would_add_to_playlist"],
        np.nan,
    )

    full_df.to_csv(SCORED_PATH, index=False)
    print(f"Saved scored catalog to {SCORED_PATH}")

    return full_df
