import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import time
import pickle
from sklearn.model_selection import GridSearchCV

# Load the trained model and scaler from the pickles folder
xgbmodel = joblib.load('pickles/xgb.pkl')
random_forest_model = joblib.load('pickles/random_forest_model.pkl')
scaler = joblib.load('pickles/scaler_minmax.pkl')

# Function to scale the input features
def scale_features(features):
    return scaler.transform(features)

# Define the Streamlit app
def main():
    st.title("Temperature Prediction App")
    

    # Create a form for user inputs
    with st.form(key='input_form'):
        col1, col2, col3 = st.columns(3)
        # Create input fields for the features
        with col1:
            month = st.selectbox("Month", list(range(1, 13)))  # January=1, December=12
            cloud_cover = st.number_input("Cloud Cover (0-100)", min_value=0.0)
            wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0)
        with col2:
            wind_gust = st.number_input("Wind Gust (m/s)", min_value=0.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0)
            pressure = st.number_input("Pressure (hPa)", min_value=0.0, format="%.4f")


        with col3:
            global_radiation = st.number_input("Global Radiation (W/m²)", min_value=0.0)
            precipitation = st.number_input("Precipitation (mm)", min_value=0.0)
            sunshine = st.number_input("Sunshine (hours)", min_value=0.0)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict Temperature')

    # Create a DataFrame from the input features after form submission
    if submit_button:
        features = pd.DataFrame({
            'MONTH': [month],
            'MUENCHEN_cloud_cover': [cloud_cover],
            'MUENCHEN_wind_speed': [wind_speed],
            'MUENCHEN_wind_gust': [wind_gust],
            'MUENCHEN_humidity': [humidity],
            'MUENCHEN_pressure': [pressure],
            'MUENCHEN_global_radiation': [global_radiation],
            'MUENCHEN_precipitation': [precipitation],
            'MUENCHEN_sunshine': [sunshine]
        })


        # Scale the features
        scaled_features = scale_features(features)

        scaled_features = pd.DataFrame(scaled_features, columns=features.columns)


        # Make prediction
        xgbprediction = xgbmodel.predict(scaled_features)
        rfprediction = random_forest_model.predict(scaled_features)
        st.success(f"The predicted temperature (By XGBOOST MODEL) is: {xgbprediction[0]:.2f} °C")
        st.success(f"The predicted temperature (BY RANDOM FOREST) is: {rfprediction[0]:.2f} °C")

if __name__ == "__main__":
    main()
