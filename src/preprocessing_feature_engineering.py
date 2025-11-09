# =====================================
# Employee Expense Fraud Detection
# Step 2: Data Preprocessing & Feature Engineering
# =====================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv("data/employee_expense_fraud_dataset.csv")

print("Data loaded successfully. Shape:", df.shape)

# Drop non-useful ID column
df = df.drop(columns=["Employee_ID"])

# Separate target and features
X = df.drop("Is_Fraudulent", axis=1)
y = df["Is_Fraudulent"]

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print("\nNumeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# Numeric preprocessing: impute + scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing: impute + one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Fit the preprocessor
preprocessor.fit(X_train)

# Save the preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nPreprocessing pipeline created and saved successfully.")
