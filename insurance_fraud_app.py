import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_model.pkl")

st.title("Insurance Fraud Detection")

claim_amount = st.number_input("Claim Amount")
policy_premium = st.number_input("Policy Annual Premium")
customer_age = st.number_input("Customer Age")
num_previous_claims = st.number_input("Number of Previous Claims")

if st.button("Predict"):
    claim_to_policy_ratio = claim_amount / policy_premium if policy_premium > 0 else 0
    data = pd.DataFrame([[claim_amount, policy_premium, customer_age, num_previous_claims, claim_to_policy_ratio]],
                        columns=['claim_amount','policy_annual_premium','customer_age','num_previous_claims','claim_to_policy_ratio'])
    prediction = model.predict(data)[0]
    st.write("Prediction:", "Fraudulent" if prediction==1 else "Legit")
