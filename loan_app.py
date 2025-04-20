import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦")

# --- Load or Train & Save Model ---
@st.cache_resource
def load_or_train_model():
    # Check if saved models exist
    if all(os.path.exists(f) for f in [
        'rf_model.pkl', 'scaler.pkl', 'le_edu.pkl', 'le_self.pkl', 'le_status.pkl'
    ]):
        # Load from disk
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_edu = joblib.load('le_edu.pkl')
        le_self = joblib.load('le_self.pkl')
        le_status = joblib.load('le_status.pkl')
        features = ['cibil_score', 'income_annum', 'education', 'self_employed']
        return model, scaler, le_edu, le_self, le_status, features

    # If not found, train new model and save
    df = pd.read_csv('loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    le_edu = LabelEncoder()
    le_self = LabelEncoder()
    le_status = LabelEncoder()

    df['education'] = le_edu.fit_transform(df['education'])
    df['self_employed'] = le_self.fit_transform(df['self_employed'])
    df['loan_status'] = le_status.fit_transform(df['loan_status'])

    features = ['cibil_score', 'income_annum', 'education', 'self_employed']
    X = df[features]
    y = df['loan_status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Save models
    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_edu, 'le_edu.pkl')
    joblib.dump(le_self, 'le_self.pkl')
    joblib.dump(le_status, 'le_status.pkl')

    return model, scaler, le_edu, le_self, le_status, features

# Load the models and encoders
model, scaler, le_edu, le_self, le_status, features = load_or_train_model()

# --- Streamlit UI ---
st.title("🏦 Loan Eligibility Predictor")

st.markdown("Enter the details below to check if you're eligible for a loan:")

# Input fields
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
income_annum = st.number_input("Annual Income (in ₹)", min_value=100000, step=10000, value=5000000)

education = st.selectbox("Education", le_edu.classes_)
self_employed = st.selectbox("Self Employed", le_self.classes_)

# Predict button
if st.button("Check Eligibility"):
    try:
        # Encode inputs
        edu_encoded = le_edu.transform([education])[0]
        emp_encoded = le_self.transform([self_employed])[0]

        input_df = pd.DataFrame([{
            'cibil_score': cibil_score,
            'income_annum': income_annum,
            'education': edu_encoded,
            'self_employed': emp_encoded
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        #prediction = model.predict(input_scaled)[0]  # Pass the scaled input directly
        prediction = model.predict(pd.DataFrame(input_scaled, columns=features))[0]
        result = le_status.inverse_transform([prediction])[0]

        if result == 'Approved':
            st.success("🎉 You are **Eligible** for the loan!")
        else:
            st.error("❌ You are **Not Eligible** for the loan.")

    except Exception as e:
        st.error(f"Error: {e}")

#streamlit run loan_app.py