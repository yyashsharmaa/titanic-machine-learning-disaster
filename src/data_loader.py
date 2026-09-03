"""
Data loader module for Titanic dataset.
Handles fetching, caching, loading, and initial schema validation.
"""
import os
import urllib.request
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
TRAIN_URL = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv"

TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
GENDER_SUBMISSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "gender_submission.csv"
)

def ensure_data_downloaded() -> None:
    """Download train.csv and test.csv if they do not exist locally."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    if not os.path.exists(TRAIN_PATH):
        print(f"Downloading train data from {TRAIN_URL} ...")
        urllib.request.urlretrieve(TRAIN_URL, TRAIN_PATH)
        print(f"Saved train data to {TRAIN_PATH}")
        
    if not os.path.exists(TEST_PATH):
        print(f"Downloading test data from {TEST_URL} ...")
        urllib.request.urlretrieve(TEST_URL, TEST_PATH)
        print(f"Saved test data to {TEST_PATH}")

    # Generate Kaggle benchmark baseline gender_submission.csv if not exists
    if not os.path.exists(GENDER_SUBMISSION_PATH):
        test_df = pd.read_csv(TEST_PATH)
        # Kaggle standard rule: females survive (1), males do not (0)
        gender_sub = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": (test_df["Sex"].str.lower() == "female").astype(int)
        })
        gender_sub.to_csv(GENDER_SUBMISSION_PATH, index=False)
        print(f"Generated benchmark gender_submission.csv at {GENDER_SUBMISSION_PATH}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure raw data exists and return (train_df, test_df).
    """
    ensure_data_downloaded()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


if __name__ == "__main__":
    train, test = load_raw_data()
    print("Train dataset shape:", train.shape)
    print("Test dataset shape:", test.shape)
    print("Train columns:", train.columns.tolist())
    print("Test columns:", test.columns.tolist())
    print("\nMissing values in Train:\n", train.isnull().sum()[train.isnull().sum() > 0])
    print("\nMissing values in Test:\n", test.isnull().sum()[test.isnull().sum() > 0])
