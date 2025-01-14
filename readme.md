# Semiconductor Supply Chain: Regression and Forecasting Models

This is a Streamlit app for predicting semiconductor supply chain labels using multiple regression models (Linear Regression, Random Forest, XGBoost, LSTM) and time series forecasting using Prophet.

## Features
- **Manual Input:** You can input values for the features and get predictions from all the models.
- **CSV Upload:** Upload a CSV file containing input features to get predictions from all models.
- **FBProphet Forecasting:** Use the pre-trained Prophet model to forecast future values for a specified number of days.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/hasbiahmad11/semiconductor_prediction.git
    cd semiconductor_prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure your model files (e.g., `scaler.pkl`, `linear_regressor.pkl`, `random_forest_model.pkl`, `xgb_model.pkl`, `lstm_model.h5`, `prophet_model.pkl`, `data_train.csv`) are in the appropriate folders.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run prediction_app_v2.py
    ```

2. Navigate to the app in your browser (usually `http://localhost:8501`).

3. Use the app to input values manually or upload a CSV file with the required features to get predictions from the regression models and the forecasting results from Prophet.

## Model Overview
- **Linear Regression:** A basic regression model to predict the semiconductor labels.
- **Random Forest Regressor:** A robust ensemble model to handle complex relationships.
- **XGBoost:** An optimized gradient boosting model for better performance.
- **LSTM:** A deep learning model for sequential data.
- **FBProphet:** A forecasting model for time series data, used to predict future values based on historical data.

## File Structure

