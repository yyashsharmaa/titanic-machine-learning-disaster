"""Unit tests for feature engineering transformations."""
import pandas as pd
from src.features import extract_title, extract_cabin_deck, TitanicFeatureExtractor


def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Cumings, Mrs. John Bradley") == "Mrs"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Futrelle, Mme. Jacques Heath") == "Mrs"
    assert extract_title("Reynaldo, Ms. Encarnacion") == "Miss"
    assert extract_title("Duff Gordon, Lady. (Lucille Christiana)") == "Rare"
    assert extract_title("Crosby, Capt. Edward Gifford") == "Rare"
    assert extract_title("Palsson, Master. Gosta Leonard") == "Master"
    assert extract_title(None) == "Mr"


def test_extract_cabin_deck():
    assert extract_cabin_deck("C85") == "C"
    assert extract_cabin_deck("E46") == "E"
    assert extract_cabin_deck("B96 B98") == "B"
    assert extract_cabin_deck(None) == "U"
    assert extract_cabin_deck("") == "U"
    assert extract_cabin_deck("Unknown") == "U"


def test_titanic_feature_extractor():
    raw_df = pd.DataFrame([
        {
            "Pclass": 1,
            "Name": "Allen, Miss. Elisabeth Walton",
            "Sex": "female",
            "Age": 29.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 211.3375,
            "Cabin": "B5",
            "Embarked": "S"
        },
        {
            "Pclass": 3,
            "Name": "Moran, Mr. James",
            "Sex": "male",
            "Age": None,
            "SibSp": 1,
            "Parch": 1,
            "Fare": None,
            "Cabin": None,
            "Embarked": None
        }
    ])
    
    extractor = TitanicFeatureExtractor()
    extractor.fit(raw_df)
    transformed_df = extractor.transform(raw_df)
    
    # Check Title
    assert transformed_df.loc[0, "Title"] == "Miss"
    assert transformed_df.loc[1, "Title"] == "Mr"
    
    # Check FamilySize & IsAlone
    assert transformed_df.loc[0, "FamilySize"] == 1
    assert transformed_df.loc[0, "IsAlone"] == 1
    assert transformed_df.loc[1, "FamilySize"] == 3
    assert transformed_df.loc[1, "IsAlone"] == 0
    
    # Check Deck & Cabin
    assert transformed_df.loc[0, "CabinDeck"] == "B"
    assert transformed_df.loc[0, "HasCabin"] == 1
    assert transformed_df.loc[1, "CabinDeck"] == "U"
    assert transformed_df.loc[1, "HasCabin"] == 0
    
    # Check missing imputation
    assert not transformed_df["Age"].isnull().any()
    assert not transformed_df["Fare"].isnull().any()
    assert not transformed_df["Embarked"].isnull().any()
    assert transformed_df.loc[1, "Embarked"] == "S"
