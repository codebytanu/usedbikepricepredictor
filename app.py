import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the model and encoders
model = joblib.load("bike_price_model.pkl")

brand_encoder = pickle.load(open("brand_encoder.pkl", "rb"))
bike_encoder = pickle.load(open("bike_encoder.pkl", "rb"))
city_encoder = pickle.load(open("city_encoder.pkl", "rb"))
owner_encoder = pickle.load(open("owner_encoder.pkl", "rb"))

st.title("🏍 Used Bike Price Predictor")

# Input fields
brand = st.text_input("Brand (e.g., Yamaha, Honda)")
bike_name = st.text_input("Bike Name (e.g., FZ S V 2.0)")
city = st.text_input("City (e.g., Delhi, Bangalore)")
kms_driven = st.number_input("KMs Driven", min_value=0)
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner"])
age = st.number_input("Age (in years)", min_value=0)
power = st.number_input("Power (in cc)", min_value=0)

# Predict button
if st.button("Predict Price"):
    try:
        # Clean and encode inputs properly
        brand = brand.strip().title()
        bike_name = bike_name.strip().upper()
        city = city.strip().title()
        owner = owner.strip().title()

        brand_encoded = brand_encoder.transform([brand])[0] if brand in brand_encoder.classes_ else 0
        bike_encoded = bike_encoder.transform([bike_name])[0] if bike_name in bike_encoder.classes_ else 0
        city_encoded = city_encoder.transform([city])[0] if city in city_encoder.classes_ else 0
        owner_encoded = owner_encoder.transform([owner])[0] if owner in owner_encoder.classes_ else 0

        input_data = pd.DataFrame({
            'bike_name': [bike_encoded],
            'city': [city_encoded],
            'kms_driven': [kms_driven],
            'owner': [owner_encoded],
            'age': [age],
            'power': [power],
            'brand': [brand_encoded]
        })

        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Price: ₹{prediction[0]:,.0f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
