<h1 align="center">Employee Expense Reimbursement Fraud Detection</h1>

<p align="center">
<b>An AI-Driven Internal Audit Platform for Financial Compliance and Fraud Analytics</b><br>
Corporate-grade machine learning project that identifies fraudulent expense reimbursements through data-driven insights.
</p>

---

## 1. Overview

This project implements a machine learning–based system to detect potential fraud in employee reimbursement claims.  
It leverages **CatBoost**, **data preprocessing pipelines**, **interactive visual analytics**, and **automated PDF reporting** to support corporate audit teams with transparent, interpretable insights.

---

## 2. Key Features

- Predicts the **probability of fraudulent expense claims** with confidence levels.  
- **Interactive Streamlit dashboard** with professional UI for finance and HR review.  
- **Dynamic visualizations** such as Expense Distribution and Fraud Rate by Department.  
- **Model performance charts** showing accuracy and ROC–AUC visually.  
- **Confidence and contributing factor insights** for each prediction.  
- Generates **professional PDF reports** summarizing each claim analysis.  
- Built with reproducible pipelines for **feature engineering and preprocessing**.

---

## 3. Technologies Used

| **Category**        | **Details** |
|----------------------|-------------|
| Machine Learning     | CatBoost, Scikit-Learn |
| Data Processing      | Pandas, NumPy |
| Visualization        | Matplotlib |
| PDF Generation       | FPDF |
| Deployment           | Streamlit |
| Language / Runtime   | Python 3.13 |


---

## 4. Dataset

- Dataset: `employee_expense_fraud_dataset.csv`  
- Contains employee expense details with approved vs. claimed amounts, submission delays, ratings, and receipts.  
- Target variable: `Is_Fraudulent` (binary classification)
- Data used for model training, visualization insights, and prediction interface.

---

## 5. Project Structure

<pre>
employee_expense_fraud_project/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── preprocessor.pkl
│   └── CatBoost_best_model.pkl
│
├── data/
│   └── employee_expense_fraud_dataset.csv
│
├── src/
│   ├── preprocessing_feature_engineering.py
│   ├── train_ensemble_models.py
│   └── eda_preprocessing.py
│
├── requirements.txt
└── README.md
</pre>

---

## 6. Installation and Setup

**Step 1:** Clone the Repository  
```bash
git clone https://github.com/Ashwinikrishnan-205/employee-expense-fraud-detection.git
cd employee-expense-fraud-detection
```

**Step 2:** Create and Activate a Virtual Environment  
```bash
python -m venv venv
venv\Scripts\activate
```

**Step 3:** Install Dependencies  
```bash
pip install -r requirements.txt
```

**Step 4:** Run the Streamlit Application  
```bash
streamlit run app/streamlit_app.py
```

Then open the browser window at: [http://localhost:8501](http://localhost:8501)


---

## 7. Model Performance Summary

| **Metric**          | **Value**              |
|----------------------|------------------------|
| Training Accuracy    | 75.25 %                |
| ROC–AUC Score        | 0.79                   |
| Model Type           | CatBoost Classifier    |
| Objective            | Binary Classification  |

---

## 8. Visual Overview

The dashboard includes:

- Bar chart comparing Training Accuracy and ROC–AUC.
- Histogram showing Expense Amount distribution.
- Fraud Probability chart by department.

---

## 8. Outputs

- Fraud probability and confidence level per claim.  
- Key risk factors and interpretability summary.  
- Interactive visualization dashboard for dataset insights.
- Professional PDF report export (`Expense_Fraud_Report.pdf`).  
- Streamlined dashboard for HR and finance teams.  

---

## 9. Developer Information

<p align="justify">
<b>Author:</b> Ashwini Krishnan<br>
<b>Year:</b> 2025<br>
<b>Focus Areas:</b> Machine Learning · Fraud Analytics · Corporate Data Science<br>
<b>GitHub:</b> <a href="https://github.com/Ashwinikrishnan-205" target="_blank">Ashwinikrishnan-205</a>
</p>

---

## 10. License

<p align="justify">
This project was created for educational and professional demonstration purposes.  
All rights reserved © 2025 <b>Ashwini Krishnan</b>.
</p>

---
