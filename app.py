import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from model import train_lstm, predict_future
from backtester import backtest
from portfolio import portfolio_value
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🚀 AI Hedge Fund Boom/Bust Platform")

TICKERS = {
    "NVIDIA": "NVDA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "GE Aerospace": "GE",
    "GE Vernova": "GEV",
    "Caterpillar": "CAT"
}

stock = st.selectbox("Select Stock", list(TICKERS.keys()))
ticker = TICKERS[stock]

end = datetime.today()
start = end - timedelta(days=365)

df = yf.download(ticker, start=start, end=end)

# -------- AI FORECAST --------
with st.spinner("Training AI Model..."):
    model, scaler = train_lstm(df)
future_price = predict_future(model, scaler, df)
future_price = float(np.array(future_price).flatten()[-1])

current_price = df["Close"].iloc[-1]
probability = (future_price - current_price) / current_price * 100

# -------- BACKTEST --------
strategy_return = backtest(df)

# -------- CRASH PREDICTOR --------
volatility = df["Close"].pct_change().rolling(20).std().iloc[-1]
crash_risk = "HIGH" if volatility > 0.04 else "MODERATE" if volatility > 0.02 else "LOW"

# -------- DASHBOARD --------
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${round(current_price,2)}")
col2.metric("AI Predicted Price (30d)", f"${round(future_price,2)}")
col3.metric("Expected % Move", f"{round(probability,2)}%")

st.subheader("📈 Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
st.plotly_chart(fig)

st.subheader("📊 Strategy Backtest (1 Year)")
st.write(f"Strategy Return: {round(strategy_return*100,2)}%")

st.subheader("⚠️ Crash Risk Level")
st.write(crash_risk)

# -------- BUY/SELL SIGNAL --------
st.subheader("🔥 Signal")
print(type(probability))

if probability > 0.5 and crash_risk == "LOW":
    st.success("🚀 STRONG BUY")
elif probability > 2:
    st.info("📈 BUY")
elif probability < -5:
    st.error("💥 STRONG SELL")
elif probability < -2:
    st.warning("📉 SELL")
else:
    st.warning("⚖️ HOLD")

# -------- PORTFOLIO --------
st.subheader("💼 Portfolio Tracker")

shares = st.number_input("Shares Owned", min_value=0)
portfolio = [{"shares": shares, "price": current_price}]
st.write("Portfolio Value: $", round(portfolio_value(portfolio),2))

st.caption("Educational use only.")
