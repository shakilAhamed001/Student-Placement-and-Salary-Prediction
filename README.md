# 🎓 Student Placement & Salary Prediction
### End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📌 Project Overview

This project builds a **machine learning system to predict student placement outcomes and expected salary packages** based on academic and skill-related attributes.

The project demonstrates a **complete data science pipeline**, including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation
- Performance Comparison

---

# 🎯 Project Objectives

The project focuses on two prediction tasks:

### 1️⃣ Placement Prediction
Predict whether a student will be **placed or not placed**.

**Type:** Classification Problem

---

### 2️⃣ Salary Prediction
Predict the **salary package (LPA)** for students who are placed.

**Type:** Regression Problem

---

# 📊 Dataset Features

Example features used in the dataset include:

- CGPA
- Branch
- College Tier
- Internships
- Coding Skills
- Communication Skills
- Projects
- Placement Status
- Salary Package (LPA)

⚠️ `salary_package_lpa` contains missing values for **unplaced students**, so salary prediction is performed only on placed students.

---

# 🔎 Exploratory Data Analysis

Key insights discovered:

- Placement probability increases with **higher CGPA**.
- **Coding skills and internships** strongly impact placement chances.
- Placement rates vary across **college tiers and branches**.
- Salary distribution is **right-skewed**, meaning a few students receive high packages.

EDA visualizations included:

- Placement distribution
- Salary distribution
- Feature correlation heatmap
- Branch-wise placement analysis

---

# 🤖 Machine Learning Models

## Placement Prediction (Classification)

| Model | Accuracy | Precision | Recall |
|------|------|------|------|
| Logistic Regression | 0.6907 | 0.70 | 0.95 |
| Random Forest Classifier | 0.6844 | 0.70 | 0.93 |

📌 Logistic Regression achieved slightly better recall for predicting placed students.

---

## Salary Prediction (Regression)

| Model | R² Score | MAE |
|------|------|------|
| Linear Regression | 0.7914 | 0.9454 |
| Random Forest Regressor | 0.7566 | 1.0207 |
| XGBoost Regressor | 0.7674 | 0.9999 |
| LightGBM Regressor | 0.7824 | 0.9653 |

📌 Linear Regression produced the best R² score.

---


# 🧰 Tech Stack

| Tool | Purpose |
|-----|------|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-Learn | Machine learning models |
| XGBoost | Gradient boosting |
| LightGBM | Efficient gradient boosting |

---

