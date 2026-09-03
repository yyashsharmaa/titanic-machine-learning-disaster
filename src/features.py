"""
Feature engineering module for Titanic survival prediction.
Implements leak-free sklearn-compatible transformers for feature extraction and transformation.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def extract_title(name: str) -> str:
    """Extract and standardize title from passenger name."""
    if not isinstance(name, str):
        return "Mr"
    title_search = re.search(r" ([A-Za-z]+)\.", name)
    if not title_search:
        return "Mr"
    title = title_search.group(1)
    
    # Consolidate rare & equivalent titles
    if title in ["Mlle", "Ms"]:
        return "Miss"
    elif title in ["Mme"]:
        return "Mrs"
    elif title in ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]:
        return "Rare"
    elif title in ["Mr", "Miss", "Mrs", "Master"]:
        return title
    else:
        return "Rare"


def extract_cabin_deck(cabin: str) -> str:
    """Extract deck letter from cabin identifier or return 'U' (Unknown)."""
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin.strip()) == 0:
        return "U"
    deck = cabin.strip()[0].upper()
    if deck in ["A", "B", "C", "D", "E", "F", "G"]:
        return deck
    return "U"


class TitanicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs engineered features
    from raw Titanic passenger data without data leakage.
    """
    def __init__(self):
        self.median_fares_by_pclass_ = {}
        self.median_ages_by_title_pclass_ = {}
        self.global_median_age_ = 28.0
        self.global_median_fare_ = 14.45

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        
        # Calculate Title for age grouping during fit
        if "Name" in X_df.columns:
            titles = X_df["Name"].apply(extract_title)
        else:
            titles = pd.Series(["Mr"] * len(X_df), index=X_df.index)
            
        pclasses = X_df["Pclass"] if "Pclass" in X_df.columns else pd.Series([3] * len(X_df), index=X_df.index)
        
        # Learn median fare by Pclass
        if "Fare" in X_df.columns:
            fare_s = X_df["Fare"].dropna()
            self.global_median_fare_ = float(fare_s.median()) if not fare_s.empty else 14.45
            self.median_fares_by_pclass_ = X_df.dropna(subset=["Fare"]).groupby("Pclass")["Fare"].median().to_dict()
            
        # Learn median age by (Title, Pclass)
        if "Age" in X_df.columns:
            age_s = X_df["Age"].dropna()
            self.global_median_age_ = float(age_s.median()) if not age_s.empty else 28.0
            temp_df = pd.DataFrame({"Title": titles, "Pclass": pclasses, "Age": X_df["Age"]}).dropna(subset=["Age"])
            if not temp_df.empty:
                self.median_ages_by_title_pclass_ = temp_df.groupby(["Title", "Pclass"])["Age"].median().to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        
        # 1. Title Extraction
        if "Name" in X_df.columns:
            X_df["Title"] = X_df["Name"].apply(extract_title)
        else:
            X_df["Title"] = "Mr"
            
        # 2. Family Size & IsAlone
        sibsp = X_df["SibSp"].fillna(0) if "SibSp" in X_df.columns else 0
        parch = X_df["Parch"].fillna(0) if "Parch" in X_df.columns else 0
        X_df["FamilySize"] = sibsp + parch + 1
        X_df["IsAlone"] = (X_df["FamilySize"] == 1).astype(int)
        
        # 3. Cabin Deck
        if "Cabin" in X_df.columns:
            X_df["CabinDeck"] = X_df["Cabin"].apply(extract_cabin_deck)
            X_df["HasCabin"] = (X_df["CabinDeck"] != "U").astype(int)
        else:
            X_df["CabinDeck"] = "U"
            X_df["HasCabin"] = 0
            
        # 4. Contextual Fare Imputation & FarePerPerson
        if "Fare" in X_df.columns:
            def impute_fare(row):
                if pd.isna(row["Fare"]):
                    pclass = row.get("Pclass", 3)
                    val = self.median_fares_by_pclass_.get(pclass)
                    return self.global_median_fare_ if (val is None or pd.isna(val)) else val
                return row["Fare"]
            
            X_df["Fare"] = X_df.apply(impute_fare, axis=1)
            X_df["FarePerPerson"] = X_df["Fare"] / X_df["FamilySize"]
        else:
            X_df["Fare"] = self.global_median_fare_
            X_df["FarePerPerson"] = self.global_median_fare_
            
        # 5. Contextual Age Imputation
        if "Age" in X_df.columns:
            def impute_age(row):
                if pd.isna(row["Age"]):
                    key = (row["Title"], row.get("Pclass", 3))
                    val = self.median_ages_by_title_pclass_.get(key)
                    return self.global_median_age_ if (val is None or pd.isna(val)) else val
                return row["Age"]
            X_df["Age"] = X_df.apply(impute_age, axis=1)
            X_df["AgeGroup"] = pd.cut(
                X_df["Age"],
                bins=[-np.inf, 12, 18, 35, 60, np.inf],
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"]
            ).astype(str)
        else:
            X_df["Age"] = self.global_median_age_
            X_df["AgeGroup"] = "YoungAdult"
            
        # 6. Embarked Imputation (Mode is 'S')
        if "Embarked" in X_df.columns:
            X_df["Embarked"] = X_df["Embarked"].fillna("S")
        else:
            X_df["Embarked"] = "S"
            
        # 7. Sex normalization
        if "Sex" in X_df.columns:
            X_df["Sex"] = X_df["Sex"].str.lower().fillna("male")
        else:
            X_df["Sex"] = "male"
            
        return X_df
