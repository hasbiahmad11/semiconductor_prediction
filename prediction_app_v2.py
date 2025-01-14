import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet


# Load Models and Scaler
scaler = joblib.load("scaler.pkl")
linear_regressor = joblib.load("linear_regressor.pkl")
random_forest_regressor = joblib.load("random_forest_model.pkl")
xgboost_regressor = joblib.load("xgb_model.pkl")

mse = MeanSquaredError()

# Load the model
lstm_model = load_model("lstm_model.h5", custom_objects={'mse': mse})
#lstm_model = load_model("lstm_model.h5")

# Load the pre-trained Prophet model
prophet_model = joblib.load('prophet_model.pkl')  # Load the Prophet model from the file

# Load training data for Prophet
df_train = pd.read_csv('data_train.csv')
df_train_prophet = df_train[['datetime', 'LBL']].copy()

# Input Features
features = ['storageCost', 'interestRate', 'compidx0lt2', 'sku1', 'demandseg1', 'timeunit']

# App Title
st.title("Semiconductor Supply Chain: Regression and Forecasting Models")

# Tabs for Manual Input and CSV Upload
tab1, tab2, tab3 = st.tabs(["Manual Input", "Upload CSV", 'FBProphet Forecasting'])

with tab1:
    st.header("Manual Input")

    # Input Fields for Manual Entry
    input_values = {}
    for feature in features:
        input_values[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

    # Convert Input Values to DataFrame
    input_df = pd.DataFrame([input_values])

    # Scale Input Data
    input_scaled = scaler.transform(input_df)

    if st.button("Predict for Manual Input"):
        # Predictions
        lr_pred = linear_regressor.predict(input_scaled)[0]
        rf_pred = random_forest_regressor.predict(input_scaled)[0]
        xgb_pred = xgboost_regressor.predict(input_scaled)[0]

        # LSTM Input Adjustment
        lstm_input = np.array(input_scaled).reshape(1, input_scaled.shape[1], 1)
        lstm_pred = lstm_model.predict(lstm_input)[0][0]

        # Display Results
        st.write("### Label Predictions")
        st.write(f"Linear Regression: {lr_pred:.2f}")
        st.write(f"Random Forest: {rf_pred:.2f}")
        st.write(f"XGBoost: {xgb_pred:.2f}")
        st.write(f"LSTM: {lstm_pred:.2f}")

with tab2:
    st.header("Upload CSV")

    # CSV File Upload
    uploaded_file = st.file_uploader("Upload CSV file containing input features", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        csv_data = pd.read_csv(uploaded_file)

        # Validate Features
        if all(feature in csv_data.columns for feature in features):
            # Scale Input Data
            csv_scaled = scaler.transform(csv_data[features])

            # Predictions for Each Model
            csv_data["Linear_Regression"] = linear_regressor.predict(csv_scaled)
            csv_data["Random_Forest"] = random_forest_regressor.predict(csv_scaled)
            csv_data["XGBoost"] = xgboost_regressor.predict(csv_scaled)

            # LSTM Predictions
            lstm_input_csv = csv_scaled.reshape(csv_scaled.shape[0], csv_scaled.shape[1], 1)
            csv_data["LSTM"] = lstm_model.predict(lstm_input_csv).flatten()

            # Display Results
            st.write("### Predictions")
            st.write(csv_data)

            # Option to Download Results
            csv_download = csv_data.to_csv(index=False)
            st.download_button("Download Predictions", data=csv_download, file_name="predictions.csv", mime="text/csv")
        else:
            st.error(f"CSV file must contain the following columns: {features}")

with tab3:
    st.header("Prophet Forecasting")

    # Input for number of forecast days
    n_days = st.number_input("Number of days for forecast", min_value=1, max_value=365, value=30)

    # Prepare training data for Prophet (rename columns for Prophet)
    df_train_prophet['datetime'] = pd.to_datetime(df_train_prophet['datetime'])
    df_train_prophet.rename(columns={'datetime': 'ds', 'LBL': 'y'}, inplace=True)

    # Display the training data
    st.subheader("Training Data")
    st.write(df_train_prophet.tail(20))  # Show last 20 rows of training data

    # Make future predictions using the pre-trained Prophet model
    future = prophet_model.make_future_dataframe(periods=n_days)
    forecast = prophet_model.predict(future)

    # Extract relevant columns (ds: datetime, yhat: prediction, yhat_lower, yhat_upper: prediction intervals)
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Rename for clarity
    forecast_result.rename(columns={'ds': 'datetime', 'yhat': 'LBL_pred'}, inplace=True)

    # Display the predictions for the next n days
    st.subheader(f"Forecast for the Next {n_days} Days")
    st.write(forecast_result.tail(n_days))

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=forecast_result, x='datetime', y='LBL_pred', label='LBL Prediction', color='blue', ax=ax)
    ax.set_title('Prophet Forecast for LBL')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('LBL')
    st.pyplot(fig)

    # Plot the forecast components (trend, seasonality)
    fig_components = prophet_model.plot_components(forecast)
    st.pyplot(fig_components)