# Titanic Survival Predictor: End-to-End ML Pipeline

An end-to-end machine learning project predicting passenger survival on the Titanic using modular scikit-learn preprocessing pipelines and tuned classification models.

---

## Overview

This project builds a reproducible machine learning workflow from raw data ingestion to final model evaluation on holdout test data. It automates feature engineering, data imputation, encoding, model benchmarking, and hyperparameter tuning.

---

## Notebook Structure & Workflow

1. **Exploratory Data Analysis (EDA):**
   - Assessed missing values, distributions, and survival rates across passenger classes, sex, age, and embarkation points.
   - Evaluated feature correlations with survival.

2. **Custom Scikit-Learn Transformers:**
   - `MissingValueImputer`: Imputes missing `Age` (median) and `Embarked` (mode) values.
   - `TitleExtractor`: Parses social titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from passenger names.
   - `FamilySizeAdder` & `IsAloneCreator`: Constructs family dynamic metrics from `SibSp` and `Parch`.
   - `AgeBinner` & `FareBinner`: Categorizes continuous age and quartile-based fare amounts.
   - `ColumnDropper`: Removes redundant columns (`Cabin`, `Ticket`, `PassengerId`, `SibSp`, `Parch`, `Name`).

3. **Modular Preprocessing Pipeline:**
   - Bundles all custom transformers with standard scaling and one-hot encoding via `ColumnTransformer` to prevent data leakage.

4. **Model Exploration & Baseline Benchmarking:**
   - Evaluated models using 5-fold Stratified Cross-Validation across Accuracy, Precision, Recall, and F1 Score:
     - k-Nearest Neighbors (kNN)
     - Decision Tree
     - Bagging Classifier (kNN base estimator)
     - Random Forest

5. **Hyperparameter Tuning:**
   - Conducted grid and randomized search (`GridSearchCV`, `RandomizedSearchCV`) to optimize model hyperparameters.

---

## Key Solutions & Final Results

- **Best Solution:** A tuned **Random Forest Classifier** achieved the highest cross-validation score (~84.3%) and demonstrated strong generalization on unseen test data.
- **Runner-Up:** A tuned **Bagging Classifier (kNN)** delivered ~83.8% validation accuracy and 80.5% test accuracy.

### Test Set Performance (Random Forest)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **81.01%** |
| **Precision** | **79.41%** |
| **Recall** | **72.97%** |
| **F1 Score** | **76.06%** |

---

## Tech Stack

- **Python**
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib
