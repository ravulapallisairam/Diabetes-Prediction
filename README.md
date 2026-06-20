# 🩺 Diabetes Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning techniques to predict whether a patient has diabetes based on medical attributes such as glucose level, BMI, age, insulin level, blood pressure, and pregnancies.

The goal is to analyze patient health data, build predictive models, and evaluate their performance in identifying diabetes risk.

---

## Dataset

* Diabetes Dataset
* 768 patient records
* 8 medical features
* Target Variable:

  * 0 = No Diabetes
  * 1 = Diabetes

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

### 1. Data Loading

* Imported dataset into Python
* Explored structure and features

### 2. Data Cleaning

* Checked for missing values
* Replaced invalid values
* Prepared data for modeling

### 3. Exploratory Data Analysis (EDA)

* Distribution Analysis
* Correlation Analysis
* Feature Comparison
* Statistical Summaries

### 4. Data Visualization

Created visualizations including:

* Feature Distributions
* Correlation Heatmap
* Boxplots
* Confusion Matrix
* ROC Curve
* Feature Importance Chart
* Decision Tree Visualization

### 5. Machine Learning Models

Implemented and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 6. Model Evaluation

Evaluated models using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve
* AUC Score

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 70.1%    |
| Decision Tree       | 78.6%    |
| Random Forest       | 77.9%    |

🏆 Best Performing Model: Decision Tree Classifier

---

## Key Findings

* Glucose was the most important predictor of diabetes.
* BMI and Age also showed strong influence.
* Decision Tree achieved the highest accuracy.
* Machine Learning can effectively identify diabetes risk patterns from patient health data.

---

## How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the project:

python diabetes_project.py

---

## Learning Outcomes

Through this project, I gained practical experience in:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Machine Learning
* Classification Models
* Model Evaluation
* Data Visualization
* Feature Importance Analysis

---

## Author

Machine Learning Project developed using Python and Scikit-learn for diabetes prediction and healthcare data analysis.
