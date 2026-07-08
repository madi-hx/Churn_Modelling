
import streamlit as st
import numpy as np
from tensorflow import keras
import joblib
import pickle

# Load ANN model
with open("churn_modelling.pkl", "rb") as f:
    model = pickle.load(f)
# Load scaler
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Bank Customer Churn Prediction")

st.write("Enter the customer details below:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input("Age", min_value=18, max_value=100, value=40)

tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)

balance = st.number_input("Balance", min_value=0.0, value=60000.0)

num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)

has_card = st.selectbox(
    "Has Credit Card?",
    ["No", "Yes"]
)

is_active = st.selectbox(
    "Is Active Member?",
    ["No", "Yes"]
)

salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Encoding
geo_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_map = {
    "Female": 0,
    "Male": 1
}

card_map = {
    "No": 0,
    "Yes": 1
}

active_map = {
    "No": 0,
    "Yes": 1
}

if st.button("Predict"):

    features = np.array([[

        credit_score,
        geo_map[geography],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_products,
        card_map[has_card],
        active_map[is_active],
        salary

    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    probability = prediction[0][0]

    if probability >= 0.5:
        st.error(f"❌ Customer is likely to EXIT\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability: {(1-probability):.2%}")
