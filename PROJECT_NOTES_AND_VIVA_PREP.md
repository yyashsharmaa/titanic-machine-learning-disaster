# 🎓 Titanic ML Project: Comprehensive Internship Notes & Viva Voce Prep Guide

This document is specifically tailored for internship evaluation, academic defense, technical interviews, and viva voce presentations for the **Titanic: Machine Learning from Disaster** project.

---

## 1. Executive Summary

- **Domain**: Binary Classification / Supervised Machine Learning
- **Problem Statement**: Predict whether a passenger survived the Titanic shipwreck ($y \in \{0, 1\}$) given their demographic, familial, and logistical attributes.
- **Dataset Size**:
  - Training Set: 891 instances, 12 features (includes ground truth `Survived`).
  - Test Set: 418 instances, 11 features (labels withheld by Kaggle).
- **Primary Metric**: Classification Accuracy (official Kaggle metric), supported by F1-Score, ROC-AUC, Precision, and Recall for holistic evaluation.
- **Core Architecture**: Leak-free `scikit-learn` Pipeline with custom feature transformers and Stratified 5-Fold Cross-Validation.

---

## 2. Key Statistical Insights (Answering Kaggle's Core Prompt)

### *"What sorts of people were more likely to survive?"*

| Dimension | Group | Survival Rate | Historical & Behavioral Rationale |
| :--- | :--- | :--- | :--- |
| **Gender** | Female | **74.2%** | Maritime evacuation protocol: *"Women and children first"*. |
| | Male | **18.9%** | Ordered to stand back; only allowed in boats if seats remained empty. |
| **Socio-Economic Class** | 1st Class | **63.0%** | Proximity to upper boat deck, stateroom evacuation alerts, priority boarding. |
| | 2nd Class | **47.3%** | Intermediate deck access. |
| | 3rd Class | **24.2%** | Steerage passengers situated deep in the hull; labyrinthine corridors and locked gate delays. |
| **Age / Children** | Master (Boys &lt; 14) | **~57.1%** | Young boys given child lifeboat priority, unlike adult men (`Mr`, 15.7%). |
| **Family Grouping** | Family of 2–4 | **~57.8%** | Mutual assistance and alerts without excessive coordination overhead. |
| | Solo (Alone) | **30.4%** | Nobody to awaken or guide them; high proportion of single young men in 3rd class. |
| | Large Family (&gt;4) | **~15.0%** | Extreme difficulty gathering all family members in dark lower corridors before boats launched. |
| **Cabin Deck** | Recorded Cabin | **66.7%** | Strong proxy for 1st Class staterooms and top-tier boat deck access. |
| | Missing Cabin (`U`) | **29.9%** | Strong proxy for steerage and lower crew decks. |

---

## 3. Data Preprocessing & Strict Anti-Leakage Protocol

### What is Data Leakage?
Data leakage occurs when information from outside the training dataset (such as target distributions, test set statistics, or validation folds) is used to create or tune the model. It causes optimistically inflated validation scores that fail to generalize in production or on Kaggle's hidden test set.

### How Our Architecture Enforces Zero Leakage:
1. **Encapsulation in `Pipeline`**:
   - Feature engineering (`TitanicFeatureExtractor`) and transformations (`StandardScaler`, `OneHotEncoder`) are wrapped inside a single `Pipeline`.
   - Fitting occurs **strictly** inside each cross-validation fold on $X_{\text{train}}$ only.
2. **Contextual Group Imputation**:
   - **`Age`**: Imputed using the median age of the passenger's `(Title, Pclass)` group learned strictly during `.fit()`. (e.g., A 1st class "Mrs." is imputed with ~40 years, whereas a 3rd class "Master" is imputed with ~4 years).
   - **`Fare`**: Imputed with the median fare of their `Pclass` learned during `.fit()`.
   - **`Embarked`**: Imputed with training mode (`'S'`).

---

## 4. Machine Learning Algorithms Benchmarked

### 1. Logistic Regression (Baseline)
- **Mathematical Form**:
  $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Role**: Establishes linear separability baseline. Fast, highly interpretable via log-odds weights.
- **Limitation**: Cannot capture non-linear feature interactions (such as the interaction between `Pclass` and `Sex` or non-linear `FamilySize` effects) without explicit polynomial features.

### 2. Random Forest Classifier
- **Mechanism**: Bagging (Bootstrap Aggregation) of decorrelated decision trees.
- **Why it shines**: Automatically captures non-linear splits (e.g., young boys surviving while older men perish) and handles feature interactions without manual interaction terms.
- **Hyperparameters tuned**: `n_estimators=200`, `max_depth=6`, `min_samples_split=5` to prevent memorization / overfitting.

### 3. Gradient Boosting / XGBoost
- **Mechanism**: Sequential boosting where each subsequent tree fits the negative gradient (pseudo-residuals) of the loss function:
  $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$
- **Shrinkage ($\eta$)**: Small learning rate ($\eta = 0.05$) ensures gradual convergence and superior generalization.
- **Why it wins**: Generally achieves the highest cross-validation accuracy and ROC-AUC on tabular datasets by focusing learning capacity on hard-to-classify edge cases.

### 4. Support Vector Machine (SVC)
- **Mechanism**: Maximizes the margin separating positive and negative classes in a kernel-transformed feature space (RBF Kernel: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$).
- **Requirement**: Crucially requires `StandardScaler` to prevent high-variance numerical features like `Fare` from dominating the distance metric.

---

## 5. Comprehensive Viva Voce / Technical Interview Q&A

### Q1: Why did you engineer the `Title` feature from passenger names?
> **Answer**: Passenger names by themselves are unique strings that risk severe overfitting if memorized. However, historical titles contain vital latent signals:
> 1. **Age proxy**: The title *"Master"* exclusively designated boys under age 14, who had an ~57% survival rate despite being male.
> 2. **Social hierarchy**: Titles like *"Countess"*, *"Col"*, *"Dr"*, and *"Major"* reflect military, medical, or noble status, granting influence during lifeboat loading.
> 3. **Marital status**: Separating *"Mrs"* from *"Miss"* captured family structure.

---

### Q2: Why is Accuracy insufficient on its own for classification?
> **Answer**: Accuracy can be deceptive if class imbalance exists (in Titanic, ~61.6% perished and ~38.4% survived). A naive dummy model predicting "0" for everyone would automatically achieve 61.6% accuracy without learning any patterns.
> To ensure true clinical and operational reliability, we also track:
> - **Precision**: Of all predicted survivors, how many actually survived?
> - **Recall**: Of all actual survivors, what fraction did the model identify?
> - **F1-Score**: Harmonic mean of Precision and Recall ($2 \times \frac{P \times R}{P + R}$).
> - **ROC-AUC**: Measures classification capability across all decision thresholds, independent of class prevalence.

---

### Q3: Why does `FamilySize` have an inverted-U / non-linear relationship with survival?
> **Answer**:
> - **Solo passengers (`FamilySize = 1`)**: Faced a ~30.4% survival rate because they had no one to alert them in their cabins or advocate for them.
> - **Small families (`FamilySize = 2 to 4`)**: Achieved peak survival (~57.8%) because spouses and parents assisted each other in reaching the upper deck without slowing the group down.
> - **Large families (`FamilySize > 4`)**: Survival dropped drastically to ~15%. Families like the Anderssons (7 members) and Sages (11 members) were unwilling to leave members behind, delaying their evacuation until all lifeboats had already departed.

---

### Q4: How did you prevent Data Leakage during feature engineering?
> **Answer**: We implemented a strict Scikit-Learn `Pipeline` where all stateful calculations (such as the median age for each Title/Pclass combination, the median fare for 3rd class, and the standard deviation of numerical scales) are computed **strictly during the `.fit()` step on the training folds**.
> In `.transform()`, the pre-calculated training parameters are applied to the test or validation set without recalculating or peeking at test statistics.

---

### Q5: How was the missing data in `Cabin` addressed without discarding 77% of the rows?
> **Answer**: In classical statistics, columns with >70% missingness are often dropped. However, in the Titanic dataset, **missingness in `Cabin` is not Missing Completely At Random (MCAR); it is Missing Not At Random (MNAR)**.
> Passengers with a recorded cabin were primarily affluent 1st Class passengers with a **66.7%** survival rate. Passengers with a missing cabin were predominantly steerage passengers with a **29.9%** survival rate.
> Instead of discarding the column, we extracted the **Cabin Deck** letter (`A`–`G`) and created a dedicated category `'U'` (Unknown) along with a binary indicator `HasCabin`. This preserved critical predictive variance.

---

### Q6: How does the interactive web application perform real-time inference?
> **Answer**: The Flask web app exposes a REST API (`/api/predict`). When the user inputs passenger attributes or selects a preset archetype, the payload is converted into a single-row Pandas DataFrame and fed directly into the serialized `joblib` pipeline (`best_titanic_model.joblib`).
> The pipeline handles extraction, imputation, scaling, and classification in milliseconds and returns both the predicted class (`0` or `1`) and the exact calibrated probability.

---

## 6. Industry & Production Deployment Considerations

If deploying this solution in an enterprise production environment:
1. **Containerization**: Package the Flask API, model artifacts, and dependencies into a lightweight Docker container (`python:3.13-slim`).
2. **Model Registry & Tracking**: Use MLflow or Weights & Biases to track experiments, hyperparameter configurations, and model versions.
3. **Drift Monitoring**: Implement Evidently AI or Prometheus metrics to detect data drift (covariate shift in passenger demographics) and concept drift (shifts in survival distributions).
4. **CI/CD Pipeline**: GitHub Actions running `pytest tests/` on every pull request before deployment to Kubernetes / cloud clusters.
