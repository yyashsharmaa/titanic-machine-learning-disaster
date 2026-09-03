"""Unit tests for inference and Kaggle submission constraints."""
import os
import pandas as pd
from src.predict import predict_single_passenger, SUBMISSION_PATH


def test_predict_single_passenger():
    # Only run if model exists
    from src.predict import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        return

    passenger = {
        "Pclass": 1,
        "Name": "Bukater, Miss. Rose DeWitt",
        "Sex": "female",
        "Age": 17.0,
        "SibSp": 0,
        "Parch": 1,
        "Fare": 227.5,
        "Cabin": "B51",
        "Embarked": "C"
    }
    
    res = predict_single_passenger(passenger)
    assert "survived" in res
    assert res["survived"] in [0, 1]
    assert 0.0 <= res["survival_probability"] <= 100.0
    assert res["risk_level"] in ["High", "Moderate", "Low"]
    assert res["label"] in ["Survived", "Did Not Survive"]


def test_submission_format_if_exists():
    if os.path.exists(SUBMISSION_PATH):
        df = pd.read_csv(SUBMISSION_PATH)
        assert len(df) == 418
        assert list(df.columns) == ["PassengerId", "Survived"]
        assert df["PassengerId"].iloc[0] == 892
        assert df["PassengerId"].iloc[-1] == 1309
        assert not df.isnull().any().any()
        assert set(df["Survived"].unique()).issubset({0, 1})
