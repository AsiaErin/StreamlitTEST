import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Function to generate forecast
def generate_forecast(start_date, end_date):
    try:
        # Prepare the data (replace this with your actual data)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # for reproducibility
        sales = np.random.randint(50, 100, size=len(dates))
        
        df = pd.DataFrame({'ds': dates, 'y': sales})
        
        # Add lag feature if there is enough data
        if len(df) > 1:
            df['y_lag1'] = df['y'].shift(1)
            df = df.dropna()
            extra_regressors = ['y_lag1']
        else:
            extra_regressors = []

        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)

        # Only add regressors if they exist
        for reg in extra_regressors:
            model.add_regressor(reg)

        model.fit(df[['ds', 'y'] + extra_regressors])

        # Make predictions for the next 7 days
        future = model.make_future_dataframe(periods=7)
        
        if 'y_lag1' in extra_regressors:
            last_y = df['y'].iloc[-1]
            future['y_lag1'] = [last_y] * len(future)

        forecast = model.predict(future)
        
        # Clip negative predictions
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
        
        # Calculate accuracy (on the training data)
        if len(df) >= 2:
            y_true = df['y']
            y_pred = model.predict(df)['yhat']
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            average_actual = y_true.mean()
            accuracy = 1 - (rmse / average_actual) if average_actual != 0 else 0
        else:
            accuracy = None

        return forecast[['ds', 'yhat']].tail(7), accuracy

    except Exception as e:
        return None, str(e)

# Streamlit app
def main():
    st.title("Sales Forecasting Documentation")
    
    st.markdown("""
    <h2>Sales Forecasting Tool</h2>
    <p>This tool predicts sales for the next 7 days based on historical data. The predictions are generated using a machine learning model (Prophet) that analyzes trends and seasonal variations in your sales data.</p>
    
    <h3>Input Requirements</h3>
    <ul>
        <li><strong>Start Date:</strong> The date when your historical sales data begins.</li>
        <li><strong>End Date:</strong> The date when your historical sales data ends.</li>
    </ul>
    <p><strong>Note:</strong> For better prediction accuracy, a period of at least 30 days of data is recommended. The more data you provide, the more accurate the forecast will be.</p>
    
    <h3>Output Results</h3>
    <ul>
        <li><strong>Forecasted Sales for the Next 7 Days:</strong> The predicted sales figures for the upcoming 7 days.</li>
        <li><strong>Accuracy:</strong> The accuracy of the forecast, calculated based on the historical data provided.</li>
    </ul>
    <p>The forecasted sales are given for each of the next 7 days, and the accuracy is measured based on the historical sales data provided.</p>
    
    <h3>Usage Recommendations</h3>
    <p>The tool is most effective when at least 30 days of data is provided. However, shorter data ranges can still provide useful predictions, though the accuracy may be lower.</p>
    <p><strong>⚠️ Warning:</strong> Predictions made with less than 30 days of data may have reduced accuracy.</p>
    """, unsafe_allow_html=True)

    # Input Section for Start and End Date
    st.subheader("Enter Start and End Dates to Get Forecast")
    start_date = st.date_input("Start Date", min_value=datetime(2020, 1, 1), max_value=datetime.today())
    end_date = st.date_input("End Date", min_value=start_date, max_value=datetime.today())

    # Button to Trigger Forecast
    if st.button("Generate Forecast"):
        if end_date < start_date:
            st.error("End Date must be after Start Date.")
        else:
            # Generate forecast
            forecast_data, accuracy = generate_forecast(str(start_date), str(end_date))
            
            if forecast_data is not None:
                st.write("### Forecast Results")
                st.write(forecast_data)

                st.write(f"### Accuracy: {accuracy * 100 if accuracy is not None else 'N/A'}%")
                st.write("### Note: Forecast generated successfully.")
            else:
                st.error(f"Error: {accuracy}")

if __name__ == "__main__":
    main()
