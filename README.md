# 🚢 RMS Titanic: Machine Learning from Disaster

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey.svg)](https://palletsprojects.com/p/flask/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An end-to-end, production-grade Machine Learning solution and interactive web application for Kaggle's legendary **Titanic: Machine Learning from Disaster** competition. Designed for top-tier competitive accuracy, zero data leakage, automated testing, and comprehensive internship / viva voce presentation.

---

## 📌 Project Overview

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg, resulting in the loss of **1,502 out of 2,224** passengers and crew. While some survival elements were stochastic, distinct socio-economic, logistical, and demographic groups experienced vastly different survival rates.

This project delivers:
1. **Empirical Exploratory Data Analysis (EDA)** answering: *"What sorts of people were more likely to survive?"*
2. **Leak-Free Feature Engineering & Pipeline**: Custom Scikit-Learn transformers preventing data leakage across folds.
3. **Rigorous Multi-Model Cross-Validation Benchmark**: Comparing Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Support Vector Machine (SVC), and XGBoost using Stratified 5-Fold CV.
4. **Hyperparameter Tuning & Explainability**: Fine-tuned tree ensembles with out-of-fold confusion matrix, ROC-AUC curves, and feature importance rankings.
5. **Kaggle Submission Engine**: Verified `submission.csv` format matching Kaggle's 418 test passenger requirement.
6. **Interactive Web Application**: A modern Flask dashboard featuring real-time survival probability prediction, passenger archetype presets, risk factor breakdowns, and benchmark metrics.
7. **Automated Unit Testing Suite**: Pytest coverage across feature extraction, pipeline integrity, and predictions.

---

## 📂 Repository Structure

```
Titanic - Machine Learning from Disaster/
├── data/
│   ├── raw/
│   │   ├── train.csv                # Kaggle ground-truth training data (891 passengers)
│   │   └── test.csv                 # Kaggle evaluation data (418 passengers)
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Auto-downloads raw data & baseline generator
│   ├── features.py                  # Sklearn-compatible feature extractor (leak-free)
│   ├── pipeline.py                  # ColumnTransformer & full pipeline builder
│   ├── train.py                     # Multi-model Stratified 5-Fold CV & tuning
│   └── predict.py                   # Kaggle submission generator & API inference
├── models/
│   ├── best_titanic_model.joblib    # Serialized tuned production pipeline
│   ├── model_metadata.json          # Metrics, best parameters, and train summary
│   └── top_features.json            # Ranked feature importance records
├── reports/
│   ├── figures/                     # High-resolution plots (ROC, CM, Feature Importance, EDA)
│   └── model_benchmark.csv          # Comparative metrics across all candidate algorithms
├── app/
│   ├── app.py                       # Flask server hosting REST API & dashboard
│   ├── static/
│   │   ├── style.css                # Modern glassmorphic responsive UI styles
│   │   └── app.js                   # Interactive client logic & real-time gauge
│   └── templates/
│       └── index.html               # Multi-tab single-page dashboard
├── scripts/
│   ├── run_eda.py                   # Automated EDA visualization pipeline
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_features.py             # Unit tests for title & deck extraction
│   ├── test_pipeline.py             # Pipeline transformation & leak-free validation
│   └── test_prediction.py           # Inference schema & Kaggle submission tests
├── submission.csv                   # Validated Kaggle submission file (418 rows)
├── gender_submission.csv            # Kaggle benchmark baseline file
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Documentation & reproduction guide
└── PROJECT_NOTES_AND_VIVA_PREP.md   # Comprehensive viva guide & ML theory
```

---

## 🛠️ Feature Engineering & Preprocessing

To ensure maximum predictive power without data leakage, all transformations are encapsulated in custom `scikit-learn` transformers:

| Feature Engineered | Logic / Formula | ML Rationale |
| :--- | :--- | :--- |
| **`Title`** | Regex extraction from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) | Isolates marital status, gender, and social status. Distinguishes young boys (`Master`, ~57% survival) from adult men (`Mr`, ~16% survival). |
| **`FamilySize`** | `SibSp + Parch + 1` | Captures passenger travel group size. Small families (2–4) survived at ~58%, whereas solo travelers (30%) and large families (&gt;4, ~15%) struggled. |
| **`IsAlone`** | `1 if FamilySize == 1 else 0` | Binary indicator isolating solo travelers who lacked family support networks. |
| **`CabinDeck`** | Extracted first letter from `Cabin` (`A`–`G`, or `'U'` for Unknown) | Cabin deck indicates proximity to the lifeboats on the top deck. Recorded cabin holders had ~67% survival vs ~30% for unrecorded. |
| **`HasCabin`** | `1 if CabinDeck != 'U' else 0` | Explicit missingness indicator capturing structural bias in record-keeping. |
| **`FarePerPerson`**| `Fare / FamilySize` | Normalizes group ticket purchases to reflect true individual socio-economic expenditure. |
| **`Age Imputation`**| Median grouped by `(Title, Pclass)` | Prevents global median bias (e.g., imputing 28 to a 1st class lady vs. a young boy). |
| **`Fare Imputation`**| Median grouped by `Pclass` | Fills test set missing fare based on passenger class standards. |

---

## 📊 Key Findings: "What Sorts of People Were More Likely to Survive?"

1. **Gender Priority**: Females had a **74.2%** survival rate compared to **18.9%** for males, driven by the maritime "Women and Children First" protocol.
2. **Socio-Economic Class**: 1st Class passengers survived at **63.0%**, 2nd Class at **47.3%**, and 3rd Class at **24.2%**, reflecting proximity to boat decks and evacuation priority.
3. **Children / Boys**: Boys with the title "Master" achieved an ~**57%** survival rate, contrasting sharply with adult men.
4. **Optimal Family Size**: Travelers in families of 2 to 4 enjoyed the highest survival rates (**~55–70%**), while solo travelers (**30.4%**) and large families of 5+ (**~15%**) suffered high mortality.
5. **Cabin & Deck**: Passengers on decks B, C, D, and E experienced survival rates above **65%**.

---

## 🚀 Quickstart & Reproduction

### 1. Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Raw Data
```powershell
.venv\Scripts\python src/data_loader.py
```

### 3. Generate EDA Visualizations
```powershell
.venv\Scripts\python scripts/run_eda.py
```

### 4. Train Models & Run Stratified 5-Fold Benchmark
```powershell
.venv\Scripts\python src/train.py
```

### 5. Generate Kaggle Submission File
```powershell
.venv\Scripts\python src/predict.py
```

### 6. Run Automated Pytest Suite
```powershell
.venv\Scripts\pytest tests/ -v
```

### 7. Launch Interactive Web Application
```powershell
.venv\Scripts\python app/app.py
```
Open your browser at **`http://127.0.0.1:5000`** to test passenger survival odds live!

---

## 🏆 Kaggle Submission Instructions

1. Log into your Kaggle account and navigate to [Titanic Submissions](https://www.kaggle.com/c/titanic/submit).
2. Upload the generated `submission.csv` file from the workspace root.
3. Verify your score and standing on the public leaderboard.
