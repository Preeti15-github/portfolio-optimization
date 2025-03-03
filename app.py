import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import matplotlib.pyplot as plt


# Function to fetch historical stock data from Yahoo Finance
@st.cache_data
def fetch_stock_data(tickers):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        df["Ticker"] = ticker
        data[ticker] = df
    return data


# Sector-wise Stock Mapping (For UI selection)
sector_mapping = {
    "Technology": ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["PFE", "JNJ", "ABBV"],
    "Finance": ["JPM", "BAC", "GS"],
    "Energy": ["XOM", "CVX", "SLB"]
}

# Sidebar UI Components
st.sidebar.header("📊 Portfolio Optimization Settings")
selected_sector = st.sidebar.selectbox("Select Sector", list(sector_mapping.keys()))
selected_stocks = st.sidebar.multiselect("Select Stocks", sector_mapping[selected_sector])

if not selected_stocks:
    st.sidebar.warning("⚠ Please select at least one stock.")
    st.stop()

# Fetch Stock Data
stock_data = fetch_stock_data(selected_stocks)

# Convert Data to DataFrame
stock_prices = pd.DataFrame({ticker: stock_data[ticker]['Close'] for ticker in selected_stocks})
stock_prices.dropna(inplace=True)

# Calculate Daily Returns
returns = stock_prices.pct_change().dropna()

# Stock Weight Customization
st.sidebar.header("⚖ Stock Weights")
weights = []
for stock in selected_stocks:
    weight = st.sidebar.slider(f"{stock} Weight (%)", min_value=0, max_value=100, value=100 // len(selected_stocks))
    weights.append(weight / 100)

weights = np.array(weights) / np.sum(weights)  # Normalize Weights

# Portfolio Performance Calculation
expected_return = np.sum(weights * returns.mean()) * 252
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
sharpe_ratio = expected_return / portfolio_volatility

# Portfolio Metrics Display
st.header("📈 Portfolio Insights")
st.write(f"**📊 Expected Annual Return:** `{expected_return:.2%}`")
st.write(f"**📉 Portfolio Risk (Volatility):** `{portfolio_volatility:.2%}`")
st.write(f"**💰 Sharpe Ratio:** `{sharpe_ratio:.2f}`")

# Historical Performance Analysis
st.subheader("📅 Historical Performance")
fig_hist = px.line(stock_prices, title="📈 Stock Price Trends (Last 5 Years)")
st.plotly_chart(fig_hist, use_container_width=True)

# Efficient Frontier Simulation
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    rand_weights = np.random.random(len(selected_stocks))
    rand_weights /= np.sum(rand_weights)

    exp_return = np.sum(rand_weights * returns.mean()) * 252
    port_volatility = np.sqrt(np.dot(rand_weights.T, np.dot(returns.cov() * 252, rand_weights)))
    sharpe_ratio = exp_return / port_volatility

    results[0, i] = exp_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio

# Efficient Frontier Plot (Plotly)
df_results = pd.DataFrame({"Return": results[0, :], "Volatility": results[1, :], "Sharpe Ratio": results[2, :]})
fig = px.scatter(df_results, x="Volatility", y="Return", color="Sharpe Ratio", color_continuous_scale="viridis",
                 title="📊 Efficient Frontier", labels={"Volatility": "Risk (Volatility)", "Return": "Expected Return"})
st.plotly_chart(fig, use_container_width=True)

# Portfolio Allocation Pie Chart
fig_pie = px.pie(names=selected_stocks, values=weights, title="💼 Portfolio Allocation", hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)
