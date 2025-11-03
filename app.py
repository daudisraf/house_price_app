# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from encoder import SimpleTargetEncoder  # Import from file

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

try:
    loaded = load_model()
    pipeline = loaded['pipeline']
    options = loaded['options']
except Exception as e:
    st.error("Model failed to load. Check logs.")
    st.stop()

st.title("KL House Price Predictor")
st.markdown("Enter details to get an **instant valuation**.")

col1, col2 = st.columns(2)
with col1:
    rooms = st.slider("Rooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 8, 2)
    car_parks = st.slider("Car Parks", 0, 6, 1)
with col2:
    size = st.number_input("Size (sq ft)", 300, 15000, 1000)
    storeys = st.slider("Storeys", 1, 5, 2)

location = st.selectbox("Location", options['locations'])
prop_type = st.selectbox("Property Type", options['types'])
furnishing = st.selectbox("Furnishing", options['furnishings'])
position = st.selectbox("Position", options['positions'])

if st.button("Predict Price", type="primary"):
    input_data = {
        'Rooms': rooms, 'Bathrooms': bathrooms, 'Car Parks': car_parks,
        'Size': size, 'Storeys': storeys,
        'Location': location, 'Type': prop_type,
        'Furnishing': furnishing, 'Position': position
    }
    df = pd.DataFrame([input_data])
    price = np.exp(pipeline.predict(df)[0])
    st.success(f"**Predicted Price: RM {price:,.0f}**")
    st.balloons()