---

# Student Academic Success Prediction Project

## a. Problem Statement

The objective of this project is to predict student academic outcomes—specifically identifying students at risk of dropping out versus those likely to graduate—using early-semester performance and demographic data.

By leveraging machine learning, institutions can implement proactive intervention strategies to improve student retention rates.

---

## b. Dataset Description  *(1 Mark)*

The project utilizes the **"Predict Students' Dropout and Academic Success"** dataset from the UCI Machine Learning Repository.

* **Target Variable:** `Target`
  *(Multi-class classification: Graduate, Dropout, Enrolled)*
* **Number of Instances:** 4,424 records
* **Number of Features:** 36 features
  *(Demographic, socio-economic, and academic data)*

### Class Distribution

* **Graduate:** 2,209
* **Dropout:** 1,421
* **Enrolled:** 794

---

## c. Models Used  *(6 Marks)*

The following table summarizes the performance of six classification models across key evaluation metrics:

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.7525   | 0.8694 | 0.7341    | 0.7525 | 0.7363   | 0.5932 |
| Decision Tree            | 0.7164   | 0.7702 | 0.7119    | 0.7164 | 0.7103   | 0.5381 |
| kNN                      | 0.6983   | 0.7912 | 0.6891    | 0.6983 | 0.6879   | 0.5050 |
| Naive Bayes              | 0.6927   | 0.8015 | 0.6809    | 0.6927 | 0.6798   | 0.4943 |
| Random Forest (Ensemble) | 0.7605   | 0.8676 | 0.7465    | 0.7605 | 0.7464   | 0.6073 |
| XGBoost (Ensemble)       | 0.7638   | 0.8750 | 0.7541    | 0.7638 | 0.7553   | 0.6139 |

---

## Observations on Model Performance  *(3 Marks)*

### Logistic Regression

* Performed surprisingly well for a linear model.
* Achieved the **second-highest AUC (0.8694)**.
* Indicates a strong linear relationship between features and student outcomes.

### Decision Tree

* Showed moderate performance.
* Likely suffered from overfitting or high variance.
* Lower MCC compared to ensemble methods.

### kNN

* Delivered the second-lowest accuracy (**0.6983**).
* Suggests local feature proximity is not the most reliable predictor for this high-dimensional dataset.

### Naive Bayes

* Weakest performer overall (**MCC: 0.4943**).
* Performance likely affected by correlated academic features violating the independence assumption.

### Random Forest (Ensemble)

* Highly stable performer.
* Strong **F1-score (0.7464)**.
* Successfully reduced variance issues seen in the single Decision Tree.

### XGBoost (Ensemble)

* **Top-performing model across all metrics**.
* Highest Accuracy (**0.7638**) and MCC (**0.6139**).
* Best captured complex, non-linear interactions in the student dataset.

---
