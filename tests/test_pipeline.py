"""Unit tests for ML pipeline transformations and model construction."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.pipeline import create_preprocessor, build_full_pipeline


def test_pipeline_transform():
    sample_df = pd.DataFrame([
        {
            "Pclass": 1,
            "Name": "Cumings, Mrs. John Bradley",
            "Sex": "female",
            "Age": 38.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 71.2833,
            "Cabin": "C85",
            "Embarked": "C"
        },
        {
            "Pclass": 3,
            "Name": "Heikkinen, Miss. Laina",
            "Sex": "female",
            "Age": 26.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.9250,
            "Cabin": None,
            "Embarked": "S"
        }
    ])
    
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(sample_df)
    
    # Check transformed output is 2D numpy array without NaNs
    assert X_trans.ndim == 2
    assert X_trans.shape[0] == 2
    assert X_trans.shape[1] > 10


def test_full_pipeline_fit_predict():
    train_df = pd.DataFrame([
        {"Pclass": 1, "Name": "A, Mrs. X", "Sex": "female", "Age": 30.0, "SibSp": 1, "Parch": 0, "Fare": 80.0, "Cabin": "B1", "Embarked": "C"},
        {"Pclass": 3, "Name": "B, Mr. Y", "Sex": "male", "Age": 22.0, "SibSp": 0, "Parch": 0, "Fare": 7.5, "Cabin": None, "Embarked": "S"},
        {"Pclass": 2, "Name": "C, Miss. Z", "Sex": "female", "Age": 14.0, "SibSp": 1, "Parch": 1, "Fare": 26.0, "Cabin": None, "Embarked": "S"},
        {"Pclass": 3, "Name": "D, Mr. W", "Sex": "male", "Age": 45.0, "SibSp": 0, "Parch": 0, "Fare": 8.0, "Cabin": None, "Embarked": "Q"},
    ])
    y = [1, 0, 1, 0]
    
    pipeline = build_full_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(train_df, y)
    
    preds = pipeline.predict(train_df)
    assert len(preds) == 4
    assert set(preds).issubset({0, 1})
