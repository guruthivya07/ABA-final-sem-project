# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# TITLE
# -------------------------------
st.title("Loan Default Prediction App")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("ABA data set EXCEL.csv")

# -------------------------------
# DATA CLEANING
# -------------------------------

# Fill missing values
if 'LoanAmount' in df.columns:
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

if 'Credit_History' in df.columns:
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Convert Loan_Status if exists (before encoding)
if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Drop missing
df = df.dropna()

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# FIND TARGET COLUMN (SAFE)
# -------------------------------

# Debug (optional)
print("Columns:", df.columns)

# Find any column containing Loan_Status
# -------------------------------
# FIND TARGET COLUMN (SMART FIX)
# -------------------------------

print("Columns:", df.columns)

# Try to find Loan_Status first
target_cols = [col for col in df.columns if "Loan_Status" in col]

# If not found → fallback to Credit_History
if len(target_cols) == 0:
    st.warning("Loan_Status not found → using Credit_History as target")

    if 'Credit_History' not in df.columns:
        st.error("No valid target column found!")
        st.stop()
    
    target_col = 'Credit_History'
else:
    target_col = target_cols[0]

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.write("Model Accuracy:", model.score(X, y))

# -------------------------------
# USER INPUT
# -------------------------------
st.sidebar.header("Enter Customer Details")

ApplicantIncome = st.sidebar.number_input("Applicant Income", 0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", 0)
LoanAmount = st.sidebar.number_input("Loan Amount", 0)
Loan_Amount_Term = st.sidebar.number_input("Loan Term", 0)
Credit_History = st.sidebar.selectbox("Credit History", [0, 1])

# Input dataframe
input_data = pd.DataFrame({
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History]
})

# Match columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")