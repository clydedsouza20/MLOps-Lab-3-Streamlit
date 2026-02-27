import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_wine

# --- CONFIGURATION ---
st.set_page_config(page_title="Clyde's Wine Quality Analytics", layout="wide")

# Custom CSS to change the feel
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #722f37; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- TITLE & DESCRIPTION ---
st.title("🍷 Premium Wine Quality Predictor")
st.markdown("Developed as part of MLOps Lab Submission - Northeastern University.")

# --- DATA & MODEL ---
@st.cache_resource
def prepare_data():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model, X, data.feature_names

model, X_df, features = prepare_data()

# --- SIDEBAR FOR UNIQUE MODIFICATION ---
st.sidebar.header("Adjust Wine Characteristics")
inputs = {}
for col in features[:6]: # Using a subset for simplicity
    inputs[col] = st.sidebar.slider(col, float(X_df[col].min()), float(X_df[col].max()), float(X_df[col].mean()))

# --- PREDICTION ENGINE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model Interpretation")
    # Add a custom chart that isn't in the original lab
    st.bar_chart(X_df.mean()) 
    st.info("The chart above shows the average feature distribution of the dataset.")

with col2:
    st.subheader("Run Prediction")
    if st.button("Analyze Quality"):
        # Prepare input for prediction
        input_array = np.array([list(inputs.values()) + [0]*7]) # Padding for remaining features
        prediction = model.predict(input_array)
        
        st.metric(label="Predicted Quality Score", value=f"{prediction[0]:.2f}")
        st.progress(prediction[0] / 2.0)