import streamlit as st
import pandas as pd
import joblib
import json
from tensorflow import keras
 
# ---------------------------------------------------------
# Load saved artifacts (from your Colab "Save Model and Scaler" step)
# Make sure the 'artifacts' folder sits next to this app.py
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("artifacts/scaler.joblib")
    model = keras.models.load_model("artifacts/churn_model.keras")
    with open("artifacts/metadata.json") as f:
        metadata = json.load(f)
    return scaler, model, metadata
 
scaler, model, metadata = load_artifacts()
threshold = metadata["threshold"]
feature_columns = metadata["feature_columns"]
 
st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦")
st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter a customer's details below to predict whether they are likely to churn.")
 
# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)
 
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
 
    with col2:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
 
    submitted = st.form_submit_button("Predict Churn")
 
# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    raw_df = pd.DataFrame([{
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": 1 if has_cr_card == "Yes" else 0,
        "is_active_member": 1 if is_active_member == "Yes" else 0,
        "estimated_salary": estimated_salary,
    }])
 
    # Same encoding steps used in training
    raw_encoded = pd.get_dummies(raw_df, columns=["geography", "gender"], drop_first=True)
    raw_encoded = raw_encoded.reindex(columns=feature_columns, fill_value=0)
 
    X_scaled = scaler.transform(raw_encoded)
    prob = model.predict(X_scaled, verbose=0).ravel()[0]
    prediction = int(prob >= threshold)
 
    st.divider()
    st.subheader("Prediction Result")
 
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **CHURN**")
    else:
        st.success(f"✅ This customer is likely to **STAY**")
 
    st.metric("Churn Probability", f"{prob:.1%}")
    st.progress(float(prob))
    st.caption(f"Decision threshold used: {threshold:.2f} (tuned on validation set, not the default 0.5)")
 
st.divider()
with st.expander("Model info"):
    st.write(f"**Test Accuracy:** {metadata['test_accuracy']:.4f}")
    st.write(f"**Test ROC-AUC:** {metadata['test_auc']:.4f}")
    st.write(f"**Decision threshold:** {threshold:.2f}")