# Employee Expense Reimbursement Fraud Detection  
**Developed by Ashwini Krishnan | Machine Learning Project 2025**

---

### 1. Overview  
This project is an AI-driven internal audit solution designed to detect potential fraud in employee expense reimbursement claims.  
It leverages advanced machine learning (CatBoost) models, corporate analytics dashboards (Streamlit), and automated PDF reporting to enhance financial governance and compliance efficiency.

The system provides:  
- Fraud probability prediction using structured claim features.  
- Real-time executive dashboard for claim analysis.  
- Confidence-based model insights for risk interpretation.  
- Auto-generated professional PDF audit summaries.

---

### 2. Business Context  
Organizations process thousands of employee reimbursement claims monthly. Manual reviews are often inconsistent and prone to bias or oversight.  
This solution automates claim assessment by identifying anomalies such as inflated expenses, missing receipts, and abnormal approval ratios — enabling finance teams to act proactively.

---

### 3. Features  
- **Streamlit Executive Dashboard:** Corporate-style, interactive user interface.  
- **AI-Powered Fraud Prediction:** Uses CatBoost classifier trained on structured expense data.  
- **Advanced Analytics Expander:** Provides interpretive insights (risk confidence, key factors).  
- **Automated PDF Reporting:** Generates a clean, one-page professional audit summary.  
- **Configurable Data View:** Dataset and model performance toggles for transparency.

---

### 4. Tech Stack  
| Component | Technology |
|------------|-------------|
| Front-End Dashboard | Streamlit |
| Machine Learning Model | CatBoost |
| Data Processing | Pandas, Scikit-Learn |
| Visualization | Matplotlib |
| PDF Report Engine | FPDF |
| Language | Python 3.13 |
| Deployment | Streamlit Cloud / Local Execution |

---

### 5. Project Architecture  

employee_expense_fraud_project/
│
├── app/
│ └── streamlit_app.py # Main Streamlit dashboard
├── models/
│ ├── preprocessor.pkl
│ └── CatBoost_best_model.pkl
├── data/
│ └── employee_expense_fraud_dataset.csv
├── src/
│ ├── preprocessing_feature_engineering.py
│ ├── train_ensemble_models.py
│ └── eda_preprocessing.py
├── requirements.txt
└── README.md

### 6. Installation & Setup  

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Ashwinikrishnan-205/employee-expense-fraud-detection.git
   cd employee-expense-fraud-detection

2. Create and Activate Virtual Environment
   
python -m venv venv
venv\Scripts\activate       # On Windows

3. Install Dependencies
   
pip install -r requirements.txt

4. Run the Application

streamlit run app/streamlit_app.py

5. Access Dashboard

The app will open automatically in your browser:
http://localhost:8501

6. Model Performance Summary
   
   | Metric            | Value                 |
| ----------------- | --------------------- |
| Training Accuracy | 75.25 %               |
| ROC–AUC Score     | 0.79                  |
| Model Type        | CatBoost Classifier   |
| Objective         | Binary Classification |

7. Outputs

Fraud Probability per claim

Risk Confidence Analysis

Contributing Factor Report

PDF Export: Expense_Fraud_Report.pdf

8. Developer Information

Author: Ashwini Krishnan
Year: 2025
Focus Areas: Machine Learning · Fraud Analytics · Corporate Data Science
GitHub: https://github.com/Ashwinikrishnan-205

9. License

This project is developed for educational and professional demonstration purposes.
All rights reserved © 2025 Ashwini Krishnan.

   
