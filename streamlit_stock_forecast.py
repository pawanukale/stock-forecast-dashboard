#!/usr/bin/env python
# coding: utf-8

# In[1]:


# streamlit_stock_forecast.py

import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

# ML & DL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------------------
# 📤 Upload Dataset
st.title("📈 Stock Forecasting Dashboard with AI Models")
uploaded_file = st.file_uploader("Upload your stock CSV file (Date & Close columns required)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'Close']].dropna()
    data.set_index('Date', inplace=True)

    st.subheader("🔍 Historical Stock Price")
    st.line_chart(data['Close'])

    forecast_days = st.slider("Select forecast horizon (days)", 7, 90, 30)

    # 📏 ARIMA Forecast
    arima_model = ARIMA(data['Close'], order=(5, 1, 0)).fit()
    arima_forecast = arima_model.forecast(steps=forecast_days)

    # 📏 SARIMA Forecast
    sarima_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit()
    sarima_forecast = sarima_model.forecast(steps=forecast_days)

    # 📏 Prophet Forecast
    prophet_df = data.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    prophet = Prophet()
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=forecast_days)
    prophet_pred = prophet.predict(future)[['ds', 'yhat']].set_index('ds').tail(forecast_days)

    # 📏 LSTM Forecast
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X, y, epochs=5, batch_size=32, verbose=0)

    input_seq = scaled_data[-60:]
    lstm_preds = []
    for _ in range(forecast_days):
        pred = model_lstm.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
        lstm_preds.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)

    lstm_forecast = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days)
    lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM': lstm_forecast}).set_index('Date')

    # 📊 Combine Forecasts
    combined_df = pd.DataFrame({
        'ARIMA': arima_forecast.values,
        'SARIMA': sarima_forecast.values,
        'Prophet': prophet_pred['yhat'].values,
        'LSTM': lstm_forecast_df['LSTM'].values
    }, index=future_dates)

    st.subheader("📈 Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))
    for col in combined_df.columns:
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], mode='lines', name=col))
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Forecasts generated successfully!")
    
    # 📥 Download Forecast CSV
    st.subheader("📁 Download Forecast Results")
    csv_download = combined_df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_download,
        file_name="stock_forecast_results.csv",
        mime='text/csv'
    )


else:
    st.info("👆 Upload a CSV file to get started!")


# In[ ]:




