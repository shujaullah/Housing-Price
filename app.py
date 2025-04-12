import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and feature list
model = joblib.load(r"model-training/best_model.pkl")
expected_cols = joblib.load("feature_columns.pkl")

# Get feature importances
feature_importances = model.feature_importances_
importances = pd.Series(feature_importances, index=model.feature_names_in_)
top_5_features = importances.nlargest(5).index.tolist()
# Print feature importances
st.write("Feature Importances:")
for feature, importance in importances.nlargest(5).items():
    st.write(f"{feature}: {importance:.4f}")
# Define feature configuration for the top 5 features
feature_config = {
    feature: {'min': 0, 'max': 100} for feature in top_5_features
}

# Create Streamlit UI
st.title('Prediction App')
st.write('Enter values for the top 5 most important features:')

# Create input fields for features
user_input = {}
for feature in top_5_features:
    user_input[feature] = st.slider(
        feature,
        min_value=float(feature_config[feature]['min']),
        max_value=float(feature_config[feature]['max']),
        value=float((feature_config[feature]['min'] + feature_config[feature]['max']) / 2)
    )

# Add predict button
if st.button('Predict'):
    # Create input array
    input_data = pd.DataFrame([user_input])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.write(f'Prediction: {prediction[0]}')
