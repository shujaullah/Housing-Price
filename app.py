import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load Trained Model ---
model = joblib.load(os.path.join("model-training", "best_model.pkl"))

# --- Load Raw, Unencoded Training Data ---
# Ensure this is raw data before encoding or transformation
X_train = pd.read_csv(r"pre-process-data\train.csv")

# --- Feature Definitions ---
numerical_features = ['LotArea', 'LotFrontage']
categorical_features = ['MSZoning', 'Street']
top_5_features = ['LotArea', 'MSSubClass', 'LotFrontage', 'MSZoning', 'Street']

# --- Building Class Dropdown Mapping ---
building_class_map = {
    "1-Story Single Family (All Ages)": 20,
    "1-Story Attached (All Ages)": 30,
    "1-Story w/ Finished Attic (All Ages)": 40,
    "1-1/2 Story - Unfinished All Ages": 45,
    "1-1/2 Story - Finished All Ages": 50,
    "2-Story 1946 & Newer": 60,
    "2-Story 1945 & Older": 70,
    "2-1/2 Story All Ages": 75,
    "Split or Multi-Level": 80,
    "Split Foyer": 85,
    "Duplex": 90,
    "1-Story PUD": 120,
    "1-1/2 Story PUD": 150,
    "2-Story PUD": 160,
    "PUD - Multi-Level": 180,
    "Two-family Conversion, Multi-Family": 190
}

# --- Encodings Used in Training ---
mszoning_map = {'C (all)': 0, 'FV': 1, 'RH': 2, 'RL': 3, 'RM': 4}
street_map = {'Grvl': 0, 'Pave': 1}

# --- Descriptive Labels ---
feature_labels = {
    'LotArea': "Lot Area (in Sq. Ft.)",
    'LotFrontage': "Lot Frontage (in Ft.)",
    'MSSubClass': "Building Class (MSSubClass)",
    'MSZoning': "Zoning Classification (MSZoning)",
    'Street': "Road Type (Street)"
}

# --- UI Setup ---
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below:")

user_input = {}

# --- Manual Number Inputs for Numerical Features ---
for feature in numerical_features:
    min_val = float(X_train[feature].min())
    max_val = float(X_train[feature].max())
    user_input[feature] = st.number_input(
        label=feature_labels[feature],
        min_value=min_val,
        max_value=max_val,
        value=min_val,
        step=0.5,
        format="%.2f"
    )

# --- Building Class Dropdown ---
user_input['MSSubClass'] = st.selectbox(
    label=feature_labels['MSSubClass'],
    options=list(building_class_map.keys())
)

# --- Categorical Dropdowns ---
user_input['MSZoning'] = st.selectbox(
    label=feature_labels['MSZoning'],
    options=list(mszoning_map.keys())
)

user_input['Street'] = st.selectbox(
    label=feature_labels['Street'],
    options=list(street_map.keys())
)

# --- Prediction Button ---
if st.button("🔍 Predict Price"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Convert Building Class to numeric
    input_df['MSSubClass'] = input_df['MSSubClass'].map(building_class_map)

    # Log-transform numerical inputs
    for feature in numerical_features:
        input_df[feature] = np.log1p(input_df[feature])

    # Encode categorical inputs
    input_df['MSZoning'] = input_df['MSZoning'].map(mszoning_map)
    input_df['Street'] = input_df['Street'].map(street_map)

    # Ensure correct column order
    input_df = input_df[top_5_features]

    # Make prediction
    log_price_prediction = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price_prediction)

    # Display result
    st.markdown("---")
    st.markdown("### 💰 Predicted House Price:")
    st.markdown(f"## :green[{predicted_price:,.2f} USD]")
