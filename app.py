import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_FILE = Path("used_car_price_model.pkl")

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Prediction System")
st.write("Enter the vehicle details to estimate its resale price.")

if not MODEL_FILE.exists():
    st.error("Trained model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)

with st.form("car_form"):
    left, right = st.columns(2)

    with left:
        brand = st.selectbox(
            "Car brand",
            ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra"]
        )
        year = st.number_input(
            "Manufacturing year", min_value=1990, max_value=2026, value=2019
        )
        km_driven = st.number_input(
            "Kilometres driven", min_value=0, max_value=500000, value=45000
        )
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])

    with right:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Number of previous owners", ["First", "Second", "Third"])
        engine_cc = st.selectbox("Engine capacity (CC)", [800, 1000, 1200, 1500, 1800, 2000])
        seats = st.selectbox("Number of seats", [4, 5, 7])

    submitted = st.form_submit_button("Predict car price")

if submitted:
    input_data = pd.DataFrame([{
        "brand": brand,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner,
        "engine_cc": engine_cc,
        "seats": seats
    }])

    predicted_price = float(model.predict(input_data)[0])
    st.success(f"Estimated resale price: ₹{predicted_price:,.0f}")
    st.caption(
        "This is an educational estimate from a demo model, not a guaranteed market valuation."
    )
