# =====================================
# Employee Expense Fraud Detection
# Step 1: Exploratory Data Analysis
# =====================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/employee_expense_fraud_dataset.csv")

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print("\nPreview of data:")
print(df.head())

# -------------------------------
# Basic Information
# -------------------------------
print("\nData information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nUnique value count per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# -------------------------------
# Check for Missing Values and Duplicates
# -------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_rows}")

# -------------------------------
# Target Variable Distribution
# -------------------------------
print("\nTarget distribution (Is_Fraudulent):")
fraud_dist = df["Is_Fraudulent"].value_counts(normalize=True) * 100
print(fraud_dist)

sns.countplot(x="Is_Fraudulent", data=df)
plt.title("Fraud vs Genuine Distribution")
plt.xlabel("Is_Fraudulent (0=Genuine, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Correlation Analysis
# -------------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numerical Features")
plt.show()

print("\nEDA completed successfully.")
