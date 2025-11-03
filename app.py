import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# === SIMPLE TARGET ENCODER (MUST MATCH TRAINING) ===
class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y, index=X.index)
        self.global_mean_ = y.mean()
        self.mappings_ = {}
        self.feature_names_in_ = list(X.columns)

        for i, col in enumerate(X.columns):
            df_temp = pd.DataFrame({'cat': X[col], 'target': y})
            stats = df_temp.groupby('cat')['target'].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (stats['count'] + self.smoothing)
            self.mappings_[i] = smooth.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(X.columns):
            mapping = self.mappings_.get(i, {})
            default = self.global_mean_
            out[:, i] = X[col].map(mapping).fillna(default).values
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

# === LOAD MODEL ===
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

# === APP UI ===
st.title("Kuala Lumpur House Price Predictor")
st.markdown("Enter property details to get an **instant price estimate**.")

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
    pred_log = pipeline.predict(df)[0]
    price = np.exp(pred_log)
    st.success(f"**Predicted Price: RM {price:,.0f}**")
    st.balloons()

st.caption("Model trained on 18k+ KL properties. Predictions are estimates.")