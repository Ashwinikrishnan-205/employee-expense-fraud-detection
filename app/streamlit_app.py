# Employee Expense Reimbursement Fraud Detection

import streamlit as st
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="Please replace `use_container_width`")

# ------------------------------------------------------
# Page configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="Employee Expense Reimbursement Fraud Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------
# Load Assets
# ------------------------------------------------------
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/CatBoost_best_model.pkl")
    dataset = pd.read_csv("data/employee_expense_fraud_dataset.csv")
    return preprocessor, model, dataset

preprocessor, model, dataset = load_assets()

# ------------------------------------------------------
# Corporate Styling
# ------------------------------------------------------
st.markdown("""
    <style>
        html, body, [class*="stAppViewContainer"] {
            background-color: #0f1116 !important;
            color: #e8e8e8 !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
        }

        .main-header {
            font-size: 50px !important;
            font-weight: 800 !important;
            text-align: center !important;
            letter-spacing: 0.4px !important;
            margin-top: 10px !important;
            margin-bottom: 5px !important;
            background: linear-gradient(to right, #3a8fdc, #8fc9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .title-accent {
            width: 150px !important;
            height: 4px !important;
            background: linear-gradient(to right, #3a8fdc, #4ca1ff) !important;
            box-shadow: 0 0 10px rgba(74, 144, 226, 0.5);
            margin: 8px auto 25px auto !important;
            border-radius: 2px !important;
        }
        .sub-header {
            font-size: 19px !important;
            color: #a9bcd6 !important;
            text-align: center !important;
            letter-spacing: 0.3px !important;
            margin-top: 2px !important;
            margin-bottom: 35px !important;
            font-weight: 500 !important;
        }
        .paragraph {
            color: #d3d3d3 !important;
            font-size: 16px !important;
            text-align: justify !important;
            line-height: 1.6 !important;
            max-width: 900px !important;
            margin: auto !important;
            margin-bottom: 30px !important;
        }
        .section-title {
            font-size: 25px !important;
            font-weight: 700 !important;
            color: #f2f2f2 !important;
            margin-top: 50px !important;
            border-left: 4px solid #3a8fdc !important;
            padding-left: 10px !important;
            position: relative;
        }
        .section-title::after {
            content: "";
            position: absolute;
            left: 10px;
            bottom: -4px;
            width: 60px;
            height: 3px;
            background: linear-gradient(to right, #3a8fdc, #73c2ff);
            box-shadow: 0 0 8px rgba(74, 144, 226, 0.4);
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; width: 0; }
            to { opacity: 1; width: 60px; }
        }

        button[kind="primary"]:hover {
            box-shadow: 0 0 12px rgba(58,143,220,0.6) !important;
            transition: 0.3s ease-in-out !important;
        }

        [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            animation: fadeUp 0.6s ease-in-out;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .footer {
            text-align: right !important;
            color: #6f737a !important;
            font-size: 14px !important;
            margin-top: 60px !important;
            margin-right: 15px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Header Section
# ------------------------------------------------------
st.markdown("<p class='main-header'>Employee Expense Reimbursement Fraud Detection</p>", unsafe_allow_html=True)
st.markdown("<div class='title-accent'></div>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>An AI-Driven Internal Audit System for Financial Compliance and Expense Fraud Detection</p>", unsafe_allow_html=True)
st.markdown("""
<p class='paragraph'>
This platform enables corporate finance and compliance departments to evaluate employee reimbursement requests using data-driven insights. 
The system leverages ensemble learning on verified financial datasets to identify anomalies, detect potentially fraudulent submissions, 
and strengthen overall audit governance within organizations.
</p>
""", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------
# Dataset and Model Insights
# ------------------------------------------------------
st.markdown("<p class='section-title'>1. Dataset and Model Insights</p>", unsafe_allow_html=True)
st.markdown("#### Customize Dashboard View")

show_dataset = st.checkbox(" Show Dataset Preview", value=True)
show_model_summary = st.checkbox(" Show Model Performance Summary", value=True)

st.markdown("<div class='insight-container'>", unsafe_allow_html=True)

if show_dataset:
    st.dataframe(dataset, use_container_width=True, height=380)
    st.caption(f"Dataset Size: {dataset.shape[0]} rows × {dataset.shape[1]} columns")

if show_model_summary:
    st.markdown("### Model Performance Overview")
    
    st.markdown("""
    <style>
        .perf-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #181b22;
            border: 1px solid #2e2e2e;
            border-radius: 8px;
        }
        .perf-table th, .perf-table td {
            border: 1px solid #2e2e2e;
            padding: 10px 15px;
            text-align: left;
            color: #d4d4d4;
        }
        .perf-table th {
            background-color: #22252e;
            font-weight: 700;
            color: #76baff;
        }
        .perf-table tr:hover {
            background-color: #1f232d;
        }
    </style>

    <table class='perf-table'>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Algorithm</td><td>CatBoost Classifier</td></tr>
        <tr><td>Model Type</td><td>Ensemble Gradient Boosting</td></tr>
        <tr><td>Training Accuracy</td><td>75.25%</td></tr>
        <tr><td>ROC–AUC Score</td><td>0.79</td></tr>
        <tr><td>Preprocessing Pipeline</td><td>Encoding + Scaling</td></tr>
        <tr><td>Objective</td><td>Binary Classification</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # Add a small visual summary
    import matplotlib.pyplot as plt
    perf_labels = ["Training Accuracy", "ROC–AUC"]
    perf_values = [75.25, 79.0]

# 🔹 Shorter, cleaner chart for corporate view
    fig, ax = plt.subplots(figsize=(4.2, 1.5))  # replaces the older 4.5,1.8 line
    ax.barh(perf_labels, perf_values, color="#3a8fdc", alpha=0.85)
    ax.grid(False)  # removes extra background gridlines for a clean look

    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)", color="#c9c9c9", fontsize=9)
    ax.tick_params(colors="#c9c9c9", labelsize=9)

    for i, v in enumerate(perf_values):
        ax.text(v + 1, i, f"{v:.1f}%", color="#ffffff", fontsize=9, va="center")

    ax.set_facecolor("#181b22")
    fig.patch.set_facecolor("#0f1116")
    st.pyplot(fig)


# ------------------------------------------------------
# Visual Insights Section (Before Prediction)
# ------------------------------------------------------
st.markdown("<p class='section-title'>1.5 Visual Insights</p>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.markdown("#### Expense Amount Distribution")
    fig1, ax1 = plt.subplots(figsize=(4.5, 3))
    ax1.hist(dataset["Expense_Amount"], bins=25, color="#3a8fdc", alpha=0.8)
    ax1.set_xlabel("Expense Amount", color="#c9c9c9")
    ax1.set_ylabel("Frequency", color="#c9c9c9")
    ax1.tick_params(colors="#c9c9c9", labelsize=8)
    ax1.set_facecolor("#181b22")
    fig1.patch.set_facecolor("#0f1116")
    st.pyplot(fig1)

with colB:
    st.markdown("#### Fraud Rate by Department")
    dept_fraud = dataset.groupby("Department")["Is_Fraudulent"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(4.5, 3))
    dept_fraud.plot(kind="bar", ax=ax2, color="#4ca1ff", alpha=0.85)
    ax2.set_ylabel("Fraud Probability", color="#c9c9c9")
    ax2.tick_params(colors="#c9c9c9", labelsize=8, rotation=45)
    ax2.set_facecolor("#181b22")
    fig2.patch.set_facecolor("#0f1116")
    st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------
# Prediction Interface (Rest Unchanged)
# ------------------------------------------------------
# (keep your prediction + PDF section exactly as in your last code)


# Prediction Interface

st.markdown("<p class='section-title'>2. Predict New Expense Claim</p>", unsafe_allow_html=True)

with st.form("fraud_prediction_form"):
    col1, col2, col3 = st.columns([1.1, 1.1, 0.9])
    with col1:
        department = st.selectbox("Department", ["Finance", "IT", "HR", "Sales", "Operations"])
        employee_level = st.selectbox("Employee Level", ["Junior", "Mid", "Senior", "Executive"])
        expense_type = st.selectbox("Expense Type", ["Travel", "Food", "Stay", "Supplies", "Client Entertainment"])
        country = st.selectbox("Country", ["India", "USA", "UK", "Singapore", "UAE"])
    with col2:
        expense_amount = st.number_input("Expense Amount (INR)", min_value=50.0, max_value=5000.0, value=1000.0)
        approved_amount = st.number_input("Approved Amount (INR)", min_value=50.0, max_value=5000.0, value=850.0)
        submission_delay = st.slider("Submission Delay (Days)", 0, 30, 10)
        has_receipt = st.selectbox("Receipt Attached", ["Yes", "No"])
    with col3:
        employee_rating = st.slider("Employee Rating (1–5)", 1.0, 5.0, 3.5, 0.1)
        years_with_company = st.number_input("Years with Company", min_value=0, max_value=30, value=5)
        claim_difference = round(expense_amount - approved_amount, 2)
        approval_ratio = round(approved_amount / expense_amount, 2)

    submitted = st.form_submit_button("Predict Fraud Probability")

confidence_text = ""
if submitted:
    with st.spinner("Analyzing claim and generating prediction..."):
        has_receipt_num = 1 if has_receipt == "Yes" else 0
        input_data = pd.DataFrame({
            "Department": [department],
            "Employee_Level": [employee_level],
            "Expense_Type": [expense_type],
            "Country": [country],
            "Expense_Amount": [expense_amount],
            "Approved_Amount": [approved_amount],
            "Submission_Delay_Days": [submission_delay],
            "Employee_Rating": [employee_rating],
            "Years_With_Company": [years_with_company],
            "Has_Receipt": [has_receipt_num],
            "Claim_Difference": [claim_difference],
            "Approval_Ratio": [approval_ratio]
        })

        processed_input = preprocessor.transform(input_data)
        prediction_prob = model.predict_proba(processed_input)[0][1]
        prediction = model.predict(processed_input)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(f"""
            <div class='result-box' style='background-color:#2b0e0e;
            color:#ffb3b3;border-left:5px solid #ff4d4d;
            box-shadow:0 0 12px rgba(255,0,0,0.3);'>
            <b>Result:</b> Fraudulent Claim<br>
            <b>Predicted Probability of Fraud:</b> {prediction_prob:.2%}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-box' style='background-color:#132b13;
            color:#b2ffb2;border-left:5px solid #00cc66;
            box-shadow:0 0 12px rgba(0,255,100,0.25);'>
            <b>Result:</b> Genuine Claim<br>
            <b>Predicted Probability of Fraud:</b> {prediction_prob:.2%}
            </div>
        """, unsafe_allow_html=True)

    with st.expander("Advanced Analytics (Model Insights)", expanded=False):
        st.markdown("""
            <style>
                [data-testid="stExpander"] {
                    border: 1px solid #2e2e2e !important;
                    box-shadow: 0 0 8px rgba(58,143,220,0.3) !important;
                    border-radius: 8px !important;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
                .insight-box {
                    background-color: #181b22;
                    border: 1px solid #2b2f36;
                    border-radius: 10px;
                    padding: 20px 25px;
                    margin-top: 10px;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.4);
                }
                .insight-header {
                    color: #4da6ff;
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 8px;
                }
                .insight-text {
                    color: #cccccc;
                    font-size: 15px;
                    line-height: 1.6;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("<div class='insight-header'>Model Confidence & Risk Assessment</div>", unsafe_allow_html=True)

        if prediction_prob < 0.25:
            confidence_text = "Low risk — claim is strongly consistent with genuine reimbursement patterns."
        elif prediction_prob < 0.6:
            confidence_text = "Moderate risk — claim shares some similarity with known fraudulent cases. Requires HR review."
        else:
            confidence_text = "High risk — claim aligns with multiple fraud-indicator variables. Immediate verification advised."

        st.markdown(f"<p class='insight-text'>{confidence_text}</p>", unsafe_allow_html=True)

        st.markdown("<br><div class='insight-header'>Key Contributing Factors</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='insight-text'>
            <li><b>Claim Difference</b>: Large gaps between claimed and approved amounts raise suspicion.</li>
            <li><b>Submission Delay</b>: Delayed reimbursements often correlate with manipulated entries.</li>
            <li><b>Employee Rating</b>: Lower ratings increase fraud probability due to reduced trust score.</li>
            <li><b>Has Receipt</b>: Missing receipts are strong fraud predictors in the dataset.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    from fpdf import FPDF
    import tempfile

    def generate_pdf(input_data, prediction, prediction_prob, confidence_text):
        def _safe_text(x):
            return str(x).replace("—", "-").replace("–", "-").replace("₹", "INR").encode("latin-1", "replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=12)

        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, _safe_text("Employee Expense Reimbursement Fraud Report"), ln=True, align="C")
        pdf.set_draw_color(0, 51, 102)
        pdf.set_line_width(0.6)
        pdf.line(10, 25, 200, 25)
        pdf.ln(8)

        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 10, _safe_text("Claim Details"), ln=True)
        pdf.set_font("Arial", size=11)
        for i, (key, value) in enumerate(input_data.items()):
            if i % 2 == 0:
                pdf.cell(95, 8, _safe_text(f"{key}: {value}"), 0, 0)
            else:
                pdf.cell(95, 8, _safe_text(f"{key}: {value}"), 0, 1)
        pdf.ln(4)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, _safe_text("Prediction Summary"), ln=True)
        pdf.set_font("Arial", size=11)
        result_text = "Fraudulent Claim" if prediction == 1 else "Genuine Claim"
        result_color = (204, 0, 0) if prediction == 1 else (0, 102, 51)
        pdf.set_text_color(*result_color)
        pdf.cell(200, 8, _safe_text(f"Outcome: {result_text}"), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 8, _safe_text(f"Fraud Probability: {prediction_prob:.2%}"), ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, _safe_text("Confidence Analysis"), ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, _safe_text(confidence_text or "Model confidence unavailable."))
        pdf.multi_cell(0, 8, _safe_text("Overall assessment: Model assessment based on behavioral and transactional anomalies."))
        pdf.ln(4)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, _safe_text("Key Contributing Factors"), ln=True)
        pdf.set_font("Arial", size=11)
        factors = [
            "1. Claim Difference – Large discrepancies between claimed and approved values.",
            "2. Submission Delay – Longer delays increase anomaly likelihood.",
            "3. Employee Rating – Lower trust score increases fraud tendency.",
            "4. Missing Receipts – Absence of proof often correlates with fraudulent claims."
        ]
        for f in factors:
            pdf.multi_cell(0, 8, _safe_text(f))
        pdf.ln(2)

        pdf.set_y(-25)
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, _safe_text("Generated by Employee Expense Reimbursement Fraud Detection System"))
        pdf.cell(0, 5, _safe_text("Developed by Ashwini Krishnan | Machine Learning Project 2025"), 0, 0, 'C')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name

    report_file = generate_pdf(
        input_data={
            "Department": department,
            "Employee Level": employee_level,
            "Expense Type": expense_type,
            "Country": country,
            "Expense Amount (INR)": expense_amount,
            "Approved Amount (INR)": approved_amount,
            "Submission Delay (Days)": submission_delay,
            "Employee Rating": employee_rating,
            "Years with Company": years_with_company,
            "Receipt Attached": has_receipt,
        },
        prediction=prediction,
        prediction_prob=prediction_prob,
        confidence_text=confidence_text
    )

    with open(report_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
        st.download_button(
            label=" Export Claim Report (PDF)",
            data=PDFbyte,
            file_name="Expense_Fraud_Report.pdf",
            mime="application/octet-stream",
            use_container_width=True,
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Developed by Ashwini Krishnan | Machine Learning Project 2025</p>", unsafe_allow_html=True)
