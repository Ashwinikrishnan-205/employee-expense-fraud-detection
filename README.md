<div align="center">

#  Employee Expense Reimbursement Fraud Detection  
**Developed by Ashwini Krishnan | Machine Learning Project 2025**

</div>

---

## 1. Overview  
This project is an **AI-driven internal audit system** designed to detect potential fraud in employee expense reimbursement claims.  
It leverages advanced machine learning (**CatBoost**) models, a corporate-grade **Streamlit executive dashboard**, and automated **PDF reporting** to strengthen financial governance and compliance.

The system provides:  
- Intelligent fraud probability prediction using structured expense features.  
- Real-time executive dashboard for claim analysis and insights.  
- Confidence-based model interpretation with key contributing factors.  
- Auto-generated, professional-grade PDF audit reports.

---

## 2. Business Context  
Organizations process thousands of employee reimbursement claims every month. Manual reviews are often inconsistent, subjective, and time-consuming.  
This project automates claim verification by flagging anomalies such as **inflated expenses**, **missing receipts**, and **unusual approval ratios**, helping finance departments maintain transparency and operational integrity.

---

## 3. Key Features  
- **Executive Streamlit Dashboard** — professional corporate UI for financial analysis.  
- **AI-Powered Fraud Detection** — built on CatBoost ensemble classification model.  
- **Advanced Analytics Expander** — interpretable AI insights for decision makers.  
- **Automated PDF Audit Reports** — generates single-page audit-grade summaries.  
- **Dataset & Model Overview** — togglable transparency section for reviewers.

---

## 4. Tech Stack  

| Component | Technology |
|------------|-------------|
| Interface | Streamlit |
| Machine Learning | CatBoost |
| Data Processing | Pandas, Scikit-Learn |
| Visualization | Matplotlib |
| PDF Generation | FPDF |
| Language | Python 3.13 |
| Deployment | Streamlit Cloud / Local Execution |

---

## 5. Project Structure  

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


6. Installation and Setup
Step 1: Clone the Repository

git clone https://github.com/Ashwinikrishnan-205/employee-expense-fraud-detection.git
cd employee-expense-fraud-detection

Step 2: Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate     

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Streamlit Application
streamlit run app/streamlit_app.py

Then open the browser window at:
http://localhost:8501

7. Model Performance Summary

| Metric            | Value                 |
| ----------------- | --------------------- |
| Training Accuracy | 75.25 %               |
| ROC–AUC Score     | 0.79                  |
| Model Type        | CatBoost Classifier   |
| Objective         | Binary Classification |

8. Outputs

Fraud probability and confidence level per claim.

Key risk factors and interpretability section.

Professional PDF report export (Expense_Fraud_Report.pdf).

Streamlined interface for audit and HR review teams.

9. Developer Information

Author: Ashwini Krishnan
Year: 2025
Focus Areas: Machine Learning · Fraud Analytics · Corporate Data Science
GitHub: Ashwinikrishnan-205

10. License

This project was created for educational and professional demonstration purposes.
All rights reserved © 2025 Ashwini Krishnan.