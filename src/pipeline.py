"""
Machine learning pipeline construction for Titanic survival classification.
Builds leak-free ColumnTransformer and end-to-end Pipelines.
"""
from typing import Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.features import TitanicFeatureExtractor

NUMERICAL_FEATURES = ["Age", "Fare", "FarePerPerson", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Title",
    "Embarked",
    "CabinDeck",
    "AgeGroup",
    "IsAlone",
    "HasCabin"
]


def create_preprocessor() -> Pipeline:
    """
    Construct preprocessing pipeline:
    1. Extracts engineered features using TitanicFeatureExtractor.
    2. Scales numerical features.
    3. One-hot encodes categorical features.
    """
    column_transforms = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"
    )
    
    preprocessor = Pipeline([
        ("feature_extractor", TitanicFeatureExtractor()),
        ("encoder_scaler", column_transforms)
    ])
    
    return preprocessor


def build_full_pipeline(classifier: Any) -> Pipeline:
    """
    Wrap preprocessor and a given classifier into an end-to-end Pipeline.
    """
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])
