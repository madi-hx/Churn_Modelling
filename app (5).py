import streamlit as st
import numpy as np
from tensorflow import keras
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = keras.models.load_model("churn_ann_model.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter the customer details below to predict whether the customer will stay or exit.")

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.number_input("Credit Score", 300, 900, 600)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input("Age", 18, 100, 40)

tenure = st.number_input("Tenure", 0, 10, 3)

balance = st.number_input("Balance", min_value=0.0, value=60000.0)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["No", "Yes"]
)

is_active = st.selectbox(
    "Is Active Member?",
    ["No", "Yes"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# -----------------------------
# Encoding
# -----------------------------
geo_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_map = {
    "Female": 0,
    "Male": 1
}

yes_no = {
    "No": 0,
    "Yes": 1
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    features = np.array([[
        credit_score,
        geo_map[geography],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_products,
        yes_no[has_card],
        yes_no[is_active],
        salary
    ]])

    features = scaler.transform(features)

    probability = model.predict(features, verbose=0)[0][0]

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error(f"❌ Customer is likely to EXIT")
        st.write(f"**Exit Probability:** {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to STAY")
        st.write(f"**Stay Probability:** {(1 - probability):.2%}")