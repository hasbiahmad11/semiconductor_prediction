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
### Model Evaluation Metrics
For evaluating the performance of our regression models, we use the following metrics:
1. **Mean Squared Error (MSE)**
   MSE calculates the average squared difference between the predicted and actual values. It gives a sense of how well the model is performing by penalizing larger errors more heavily due to the squaring of the residuals.
2. **Root Mean Squared Error (RMSE)**
   RMSE is the square root of the MSE, bringing the metric back to the original units of the target variable. It provides an interpretable measure of error magnitude.
3. **R-squared (R²)**
   R² quantifies how well the model explains the variance in the target variable. A higher R² indicates a better fit of the model to the data.
### Why These Metrics?
Using **MSE**, **RMSE**, and **R²** together gives you a comprehensive view of the model's accuracy, interpretability, and goodness of fit:
- **MSE** and **RMSE** focus on **prediction error**, indicating how far off the predictions are from the true values.
- **R²** provides insight into how well the model explains the **variance** in the data, reflecting the proportion of the target's variability that is accounted for by the model.
By considering all of these metrics, a well-rounded assessment of regression model's performance can be achieved, understanding both its error and its explanatory power.


## Data Structure
1. Temporal Component:
   - timeunit: Likely indicates time (e.g., daily, weekly, or monthly units). In this project, timeunit refers to the sequence of days within a year. For example, **timeunit = 1** indicates January 1, 2024.
2. Economic Indicators:
   - storageCost, interestRate: Represent costs associated with inventory storage and financing rates, crucial for inventory control.
3. Component Indices:
   - Features like compidx0lt2, compidx0lt2l1, compidx0lt2l2, etc., appear to track component metrics with thresholds (e.g., <2, <6, <10, <20, <30).
   - Suffixes like l1, l2, l4, l8 may represent lagged values (e.g., metrics from previous time steps).
4. SKU (Stock Keeping Unit) Information:
   - sku1, sku1l1, sku1l2, etc., are SKU-specific features, potentially related to inventory levels or production.
5. Demand Segments:
   - demandseg1, demandseg2, etc., with lags (l1, l2, etc.), likely capture segmented demand data, critical for forecasting and anomaly detection.
6. Labels or Targets:
   - LBL: This might be a categorical or numerical label for classification or regression tasks.
   - MTLp2, MTLp3, etc.: Could represent multi-tiered lead times or multi-level performance metrics.
