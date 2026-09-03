"""
Inference and submission generation module for Titanic ML competition.
Generates Kaggle-compliant submission.csv and supports single-passenger API predictions.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
from src.data_loader import load_raw_data, TEST_PATH
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_titanic_model.joblib")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.csv")


def load_trained_model():
    """Load the serialized best pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )
    return joblib.load(MODEL_PATH)


def generate_kaggle_submission() -> pd.DataFrame:
    """
    Run inference on test.csv and produce validated submission.csv.
    """
    _, test_df = load_raw_data()
    pipeline = load_trained_model()
    
    print(f"\nRunning predictions on test set ({len(test_df)} passengers)...")
    preds = pipeline.predict(test_df)
    probs = pipeline.predict_proba(test_df)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    submission_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"].astype(int),
        "Survived": preds.astype(int)
    })
    
    # Validation checks
    assert len(submission_df) == 418, f"Expected 418 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["PassengerId", "Survived"], "Column headers incorrect"
    assert not submission_df.isnull().any().any(), "Found missing values in submission"
    assert set(submission_df["Survived"].unique()).issubset({0, 1}), "Predictions must be binary 0 or 1"
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Kaggle submission successfully verified and saved to: {SUBMISSION_PATH}")
    
    survived_count = (submission_df["Survived"] == 1).sum()
    survived_pct = (survived_count / len(submission_df)) * 100
    print(f"Predicted Test Survival: {survived_count} / {len(submission_df)} ({survived_pct:.2f}%)")
    
    return submission_df


def predict_single_passenger(passenger_dict: dict) -> dict:
    """
    Predict survival probability and binary outcome for a single passenger.
    Used by the interactive web application.
    """
    pipeline = load_trained_model()
    df_single = pd.DataFrame([passenger_dict])
    
    pred = int(pipeline.predict(df_single)[0])
    prob = float(pipeline.predict_proba(df_single)[0][1])
    
    return {
        "survived": pred,
        "survival_probability": round(prob * 100, 2),
        "risk_level": "High" if prob < 0.35 else ("Moderate" if prob < 0.65 else "Low"),
        "label": "Survived" if pred == 1 else "Did Not Survive"
    }


if __name__ == "__main__":
    generate_kaggle_submission()
