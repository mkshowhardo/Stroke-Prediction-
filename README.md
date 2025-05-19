# 🧠 Stroke Prediction Using Machine Learning

Predicting stroke risk using patient medical history and health indicators with end-to-end ML pipeline and interpretability.

---

## 📌 Project Overview

This project aims to predict whether a patient is likely to experience a stroke based on their demographic and medical features. It tackles real-world problems such as **missing data**, **class imbalance**, and **feature encoding**, while demonstrating the complete machine learning pipeline from data cleaning to model explainability.

---

## 📂 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Target Variable**: `stroke` (0 = No stroke, 1 = Stroke)
- **Features**: Age, hypertension, heart disease, BMI, glucose level, smoking status, etc.

---

## ✅ Objectives

- Perform detailed Exploratory Data Analysis (EDA)
- Handle missing values (e.g., BMI)
- Address class imbalance using SMOTE
- Engineer new features (age groups, risk scores)
- Train and evaluate models (Random Forest, Logistic Regression, XGBoost)
- Use SHAP for model interpretation

---

## 🛠️ Tools & Technologies

- Python 🐍
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- SHAP

---

## 📊 Key Highlights

| Task                      | Technique/Result                            |
|---------------------------|---------------------------------------------|
| Missing Values            | Filled BMI using median                     |
| Class Imbalance           | Used SMOTE for oversampling                 |
| Feature Engineering       | Created `age_group`, `risk_score`           |
| Best Model                | XGBoost                                     |
| Evaluation Metric         | F1-score (focus on minority class)          |
| Explainability            | SHAP summary & force plots                  |

---

## 📈 Sample Output

- **Top Features**: `avg_glucose_level`, `bmi`, `age`, `hypertension`
- **F1-score (minority class)**: ~0.85 (XGBoost)
- **Interpretability**: SHAP explains how each feature contributes to predictions

---

## 📁 Project Structure


