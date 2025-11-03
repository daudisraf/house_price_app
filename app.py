import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the PKL (pipeline and options)
loaded = joblib.load('house_price_model.pkl')
pipeline = loaded['pipeline']
options = loaded['options']

# App title and description
st.title("House Price Predictor")
st.write("Enter the house details below to get a predicted price in RM.")

# User inputs (adjust min/max/value based on your data's describe())
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
car_parks = st.number_input("Car Parks", min_value=0, max_value=10, value=1, step=1)
size = st.number_input("Size (sq ft)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
storeys = st.number_input("Storeys", min_value=1, max_value=10, value=2, step=1)

location = st.selectbox("Location", options['locations'])
prop_type = st.selectbox("Type", options['types'])
furnishing = st.selectbox("Furnishing", options['furnishings'])
position = st.selectbox("Position", options['positions'])

# Predict button
if st.button("Predict Price"):
    # Create input DataFrame (column order must match training X)
    input_data = {
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Car Parks': car_parks,
        'Size': size,
        'Storeys': storeys,
        'Location': location,
        'Type': prop_type,
        'Furnishing': furnishing,
        'Position': position
    }
    input_df = pd.DataFrame([input_data])
    
    # Predict (pipeline handles preprocessing)
    log_pred = pipeline.predict(input_df)[0]
    predicted_price = np.exp(log_pred)  # Back to original scale
    
    st.success(f"Predicted House Price: RM {predicted_price:,.2f}")

# Footer
st.write("Model based on Kuala Lumpur housing data. Predictions are estimates only.")