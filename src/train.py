"""
Model training, cross-validation benchmarking, hyperparameter tuning,
and model serialization for Titanic survival classification.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data_loader import load_raw_data
from src.pipeline import build_full_pipeline, create_preprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def get_candidate_models() -> dict[str, any]:
    """Define dictionary of candidate classification algorithms."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_split=5, random_state=42
        ),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=42
        )
    }


def evaluate_models_cv(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Run 5-Fold Stratified Cross-Validation on all candidate models.
    Returns comparison DataFrame.
    """
    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    results = []
    print("\n" + "=" * 70)
    print("STARTING 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK")
    print("=" * 70)
    
    for name, clf in models.items():
        pipeline = build_full_pipeline(clf)
        cv_res = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        acc_mean = cv_res["test_accuracy"].mean()
        acc_std = cv_res["test_accuracy"].std()
        f1_mean = cv_res["test_f1"].mean()
        prec_mean = cv_res["test_precision"].mean()
        rec_mean = cv_res["test_recall"].mean()
        auc_mean = cv_res["test_roc_auc"].mean()
        
        print(f"[{name:22s}] Acc: {acc_mean:.4f} (+/- {acc_std:.4f}) | F1: {f1_mean:.4f} | AUC: {auc_mean:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy_Mean": acc_mean,
            "Accuracy_Std": acc_std,
            "F1_Mean": f1_mean,
            "Precision_Mean": prec_mean,
            "Recall_Mean": rec_mean,
            "ROC_AUC_Mean": auc_mean,
        })
        
    df_results = pd.DataFrame(results).sort_values(by="Accuracy_Mean", ascending=False)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_results.to_csv(os.path.join(REPORTS_DIR, "model_benchmark.csv"), index=False)
    print("Saved benchmark results to reports/model_benchmark.csv")
    return df_results


def tune_best_model(X: pd.DataFrame, y: pd.Series, best_model_name: str) -> any:
    """
    Perform hyperparameter tuning for top ensemble models.
    """
    print(f"\nFine-tuning hyperparameters for {best_model_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if "Gradient Boosting" in best_model_name:
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5],
            "classifier__min_samples_split": [4, 6, 8]
        }
        base_pipeline = build_full_pipeline(GradientBoostingClassifier(random_state=42))
    elif "XGBoost" in best_model_name:
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0]
        }
        base_pipeline = build_full_pipeline(XGBClassifier(eval_metric="logloss", random_state=42))
    else:
        param_grid = {
            "classifier__n_estimators": [150, 200, 250],
            "classifier__max_depth": [5, 6, 7],
            "classifier__min_samples_split": [4, 6, 8]
        }
        base_pipeline = build_full_pipeline(RandomForestClassifier(random_state=42))

    grid = GridSearchCV(base_pipeline, param_grid, cv=skf, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def generate_evaluation_visualizations(pipeline: any, X: pd.DataFrame, y: pd.Series) -> None:
    """Generate and save ROC curve, confusion matrix, and feature importances."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validated out-of-fold predictions
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        oof_preds[val_idx] = pipeline.predict(X_val)
        oof_probs[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    # 1. Confusion Matrix
    cm = confusion_matrix(y, oof_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Did Not Survive (0)", "Survived (1)"],
                yticklabels=["Did Not Survive (0)", "Survived (1)"])
    plt.title("Out-of-Fold Confusion Matrix (Titanic Best Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, oof_probs)
    auc_val = roc_auc_score(y, oof_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.title("ROC Curve - Stratified Cross-Validation")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {roc_path}")

    # 3. Feature Importance Analysis
    # Fit pipeline on all data to extract feature names
    pipeline.fit(X, y)
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    try:
        # Extract encoded feature names
        col_transformer = preprocessor.named_steps["encoder_scaler"]
        num_cols = ["Age", "Fare", "FarePerPerson", "FamilySize", "SibSp", "Parch"]
        cat_encoder = col_transformer.named_transformers_["cat"]
        cat_cols = cat_encoder.get_feature_names_out([
            "Pclass", "Sex", "Title", "Embarked", "CabinDeck", "AgeGroup", "IsAlone", "HasCabin"
        ]).tolist()
        all_features = num_cols + cat_cols
        
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
            plt.title("Top 15 Most Influential Features for Titanic Survival")
            plt.xlabel("Relative Feature Importance")
            plt.tight_layout()
            feat_path = os.path.join(FIGURES_DIR, "feature_importance.png")
            plt.savefig(feat_path, dpi=300)
            plt.close()
            print(f"Saved feature importance plot to {feat_path}")
            
            # Save top features to json
            top_feats = feat_df.to_dict(orient="records")
            with open(os.path.join(MODELS_DIR, "top_features.json"), "w") as f:
                json.dump(top_feats, f, indent=2)
    except Exception as e:
        print(f"Could not compute feature importances: {e}")


def train_and_save_pipeline() -> None:
    """Full workflow: benchmark, tune, generate figures, and save final pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, _ = load_raw_data()
    
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    
    # 1. Benchmark all models
    benchmark_df = evaluate_models_cv(X, y)
    best_model_row = benchmark_df.iloc[0]
    best_name = best_model_row["Model"]
    print(f"\nTop Performing Algorithm: {best_name} (Acc: {best_model_row['Accuracy_Mean']:.4f})")
    
    # 2. Hyperparameter tune top model
    best_pipeline = tune_best_model(X, y, best_name)
    
    # 3. Fit on complete training dataset
    print("\nFitting final tuned pipeline on all training data...")
    best_pipeline.fit(X, y)
    
    # 4. Generate evaluation plots
    generate_evaluation_visualizations(best_pipeline, X, y)
    
    # 5. Save model and metadata
    model_path = os.path.join(MODELS_DIR, "best_titanic_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved final trained pipeline to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": best_name,
        "cv_accuracy": float(best_model_row["Accuracy_Mean"]),
        "cv_f1": float(best_model_row["F1_Mean"]),
        "cv_roc_auc": float(best_model_row["ROC_AUC_Mean"]),
        "n_train_samples": len(train_df),
        "target_distribution": train_df["Survived"].value_counts().to_dict(),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    train_and_save_pipeline()
