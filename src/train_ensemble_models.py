# Employee Expense Fraud Detection
# 3: Ensemble Model Training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import numpy as np

# Load data
df = pd.read_csv("data/employee_expense_fraud_dataset.csv")
X = df.drop(columns=["Employee_ID", "Is_Fraudulent"])
y = df["Is_Fraudulent"]

# Load preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Split data again
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Transform data
X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=42)
}

# Evaluate models
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)
    y_prob = model.predict_proba(X_test_prep)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)

# Save best model
best_model_name = results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
joblib.dump(best_model, f"models/{best_model_name}_best_model.pkl")

print(f"\nBest model: {best_model_name} (saved successfully)")
