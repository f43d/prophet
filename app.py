import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

# Disable auto-reload
st.set_page_config(page_title="Prophet App", layout="wide")
st.cache_resource.clear()

# Set parameters
ticker = 'AAPL'
num_years = 20
start_date = dt.datetime.now() - dt.timedelta(days=365.25 * num_years)
end_date = dt.datetime.now()

# Fetch stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare data for Prophet
data.reset_index(inplace=True)
data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Create and fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(data)

# Create future dataframe and make predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot predictions using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='markers', name='Actual'))
st.plotly_chart(fig)
