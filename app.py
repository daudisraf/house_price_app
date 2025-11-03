import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

data = load_model()
rf = data['model']
yeo = data['yeo']
ohe = data['ohe']
target_map = data['target_map']
global_mean = data['global_mean']
options = data['options']

st.title("KL House Price Predictor")
st.markdown("**18,000+ properties trained · Instant valuation**")

col1, col2 = st.columns(2)
with col1:
    rooms = st.slider("Rooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 8, 2)
    car_parks = st.slider("Car Parks", 0, 6, 1)
with col2:
    size = st.number_input("Size (sq ft)", 300, 15000, 1000)
    storeys = st.slider("Storeys", 1, 5, 2)

location = st.selectbox("Location", options['locations'])
prop_type = st.selectbox("Type", options['types'])
furnishing = st.selectbox("Furnishing", options['furnishings'])
position = st.selectbox("Position", options['positions'])

if st.button("Predict Price", type="primary"):
    # Build input
    input_df = pd.DataFrame([{
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Car Parks': car_parks,
        'Size': size,
        'Storeys': storeys,
        'Location': location,
        'Type': prop_type,
        'Furnishing': furnishing,
        'Position': position
    }])

    # Transform
    df = input_df.copy()
    df['Rooms_sqrt'] = np.sqrt(df['Rooms'])
    df['Bathrooms_sqrt'] = np.sqrt(df['Bathrooms'])
    df['Car Parks_log1p'] = np.log1p(df['Car Parks'])
    df['Storeys_sqrt'] = np.sqrt(df['Storeys'])
    df['Size_yj'] = yeo.transform(df[['Size']])
    df['Location_target'] = df['Location'].map(target_map).fillna(global_mean)

    # One-hot
    ohe_input = ohe.transform(df[['Type', 'Furnishing', 'Position']])
    ohe_cols = ohe.get_feature_names_out(['Type', 'Furnishing', 'Position'])
    ohe_df = pd.DataFrame(ohe_input, columns=ohe_cols)

    # Final
    final = pd.concat([
        df[['Rooms_sqrt', 'Bathrooms_sqrt', 'Car Parks_log1p', 'Storeys_sqrt', 'Size_yj', 'Location_target']],
        ohe_df
    ], axis=1)

    pred = np.exp(rf.predict(final)[0])
    st.success(f"**Predicted Price: RM {pred:,.0f}**")
    st.balloons()