"""Phase 3: Gradient Boosting conversion propensity model with SHAP explainability.

Predicts whether a session will result in a purchase based on session-level
and user-history features. Includes cross-validation, hyperparameter tuning,
and SHAP-based feature importance explanations.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EXCLUDE_COLS = [
    "computed_session_id", "user_id", "session_start", "session_end",
    "converted",
    "purchase_count",  # leaks the target
    "cart_value",  # partially leaks
]


def load_data():
    df = pd.read_parquet(DATA_DIR / "ml_features.parquet")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0)
    y = df["converted"]
    return X, y, feature_cols, df


def train_model(X, y, feature_cols):
    """Train GradientBoosting with cross-validation and return fitted model + metrics."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["roc_auc", "precision", "recall", "f1"],
        return_train_score=False,
    )

    print("\nCross-Validation Results:")
    for metric in ["roc_auc", "precision", "recall", "f1"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    metrics = {
        "cv_roc_auc": float(cv_results["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cv_results["test_roc_auc"].std()),
        "test_roc_auc": float(test_auc),
        "test_pr_auc": float(pr_auc),
    }

    return model, metrics, X_test, y_test, y_prob


def compute_shap(model, X, feature_cols):
    """Compute SHAP values for feature explanations."""
    import shap
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 10 Features (by SHAP importance):")
    print(shap_importance.head(10).to_string(index=False))

    return shap_values, shap_importance, explainer


def save_artifacts(model, metrics, shap_importance, feature_cols):
    """Save model and metadata for the dashboard."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "gb_conversion.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    shap_importance.to_csv(MODEL_DIR / "shap_importance.csv", index=False)

    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\nArtifacts saved to {MODEL_DIR}/")


def run():
    X, y, feature_cols, df = load_data()
    print(f"Dataset: {len(X):,} sessions, {len(feature_cols)} features")
    print(f"Conversion rate: {y.mean():.1%}\n")

    model, metrics, X_test, y_test, y_prob = train_model(X, y, feature_cols)
    shap_values, shap_importance, explainer = compute_shap(model, X_test, feature_cols)
    save_artifacts(model, metrics, shap_importance, feature_cols)

    return model, metrics, shap_importance


if __name__ == "__main__":
    run()
