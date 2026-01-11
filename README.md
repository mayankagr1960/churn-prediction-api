# 📊 Customer Churn Prediction – End-to-End Machine Learning Project

## 📌 Problem Statement
Customer churn is a critical challenge for subscription-based businesses.  
The objective of this project is to **predict whether a customer will churn** and **identify the key drivers behind churn**, enabling businesses to take proactive retention actions.

This project demonstrates a **full machine learning lifecycle**, from data exploration to deployment.

---

## 📂 Dataset
- **Telco Customer Churn Dataset**
- ~7,000 customer records
- Mix of numerical and categorical features:
  - Tenure
  - Monthly & total charges
  - Contract type
  - Payment method
  - Gender
  - Customer churn (target)

---

## ⚙️ Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data preprocessing & feature engineering
3. Train-Test Split
4. Baseline modeling (Logistic Regression)
5. Advanced modeling (Random Forest)
6. Key metrics evaluation
7. Hyperparameter tuning (GridSearchCV)
8. Model explainability (SHAP)
9. Import and export of model
10. API deployment using FastAPI

---

## 🧠 Models Used
- Logistic Regression – baseline & interpretability
- Random Forest (Tuned) – Final production model

---

## 📈 Model Performance

Class	Presicion	Recall	F1-Score
0	    0.91	    0.77	0.83
1	    0.54	    0.76	0.63
Overall Accuracy			0.77
ROC-AUC                     0.85


### Business Interpretation
- The model achieves an overall accuracy of **77%**, indicating solid general performance on unseen data.
- For Class 0 (Non-Churn), the model shows high precision **(0.91)** and a strong F1-score **(0.83)**, meaning it reliably identifies non-churning customers.
- For Class 1 (Churn), the model attains a recall of **0.76**, showing it successfully captures most churn cases, which is critical for retention strategies.
- The F1-score of **0.63** for churners reflects a balanced trade-off between precision and recall in an imbalanced dataset.
- With a ROC-AUC of **0.85** and **77%** overall accuracy, the model reliably distinguishes between churn and non-churn customers, supporting **data-driven decision-making**.


### Business Actions
- The model helps the business identify customers likely to leave early, allowing teams to take timely retention actions.
- The predictions are reliable and consistent, giving confidence in using the model for customer-focused decisions.

---

## 🔍 Key Churn Drivers (SHAP Explainability)
- Low customer tenure
- Month-to-month contracts
- High monthly charges
- High total charges
- Lack of long-term commitment
- Senior and non-senior citizens

SHAP was used to explain:
- Global feature importance
- Individual customer predictions

<img width="457" height="680" alt="SHAP_explanation" src="https://github.com/user-attachments/assets/b708420b-f7cf-4bd8-b497-e633beed5365" />

---

## 🚀 Live API Deployment

The trained model is deployed as a REST API using FastAPI.

<img width="626" height="331" alt="image" src="https://github.com/user-attachments/assets/304bf0d1-7c05-4389-9e6f-346c1bae5539" />

